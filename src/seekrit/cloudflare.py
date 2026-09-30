"""Async resolution inside Cloudflare Python Workers.

    from seekrit.cloudflare import AsyncClient

    secrets = await AsyncClient(token=self.env.SEEKRIT_TOKEN).resolve()

Uses the runtime's native fetch, without sockets or a thread pool. Decryption
is the same golden-vector-tested implementation as :class:`seekrit.Client`;
Pywrangler installs the Pyodide build of ``cryptography``. Runtime imports are
lazy so importing this module on a regular Python host is harmless.
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, Optional

from ._client_config import _ClientConfig
from ._crypto import materialize
from .errors import SeekritError

__all__ = ["AsyncClient"]


class AsyncClient(_ClientConfig):
    """A read-only async client for Cloudflare Python Workers.

    Accepts the same ``token``, ``api_url``, ``overrides``, ``timeout`` (seconds)
    and ``interpolate`` arguments as :class:`seekrit.Client`. Pass the token
    from ``self.env.SEEKRIT_TOKEN`` explicitly. Resolve inside a request or
    event handler; every call fetches fresh ciphertext and decrypts locally.

    There is no plaintext cache or ``os.environ`` injection. Keep the returned
    values in the handler rather than shared state. Redirects are refused so
    the service token cannot be forwarded to a different endpoint.
    """

    async def resolve(self) -> Dict[str, str]:
        """Fetch, decrypt, and merge; fail closed on any fetch/decrypt error."""
        return materialize(
            await self._fetch(), self._key, interpolate=self._interpolate
        )

    async def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Resolve and return a single secret, or ``default`` when absent."""
        return (await self.resolve()).get(name, default)

    async def _fetch(self) -> dict:
        try:
            from js import AbortController
            from workers import fetch
        except ImportError as exc:
            raise SeekritError(
                "seekrit.cloudflare.AsyncClient requires the Cloudflare Python Workers runtime"
            ) from exc

        controller = AbortController.new()

        async def request():
            response = await fetch(
                self._resolve_url(),
                method="GET",
                headers={
                    "authorization": f"Bearer {self._token}",
                    "accept": "application/json",
                },
                # Workers implements "manual" and "follow", but not "error".
                redirect="manual",
                signal=controller.signal,
            )
            if 300 <= response.status < 400:
                # Refuse without following or reading the redirect.
                return response.status, b""
            # Keep the timeout active while consuming the body too.
            return response.status, await response.bytes()

        try:
            status, body = await asyncio.wait_for(request(), timeout=self._timeout)
        except asyncio.TimeoutError as exc:
            raise SeekritError("resolve request timed out") from exc
        except Exception as exc:
            raise SeekritError("resolve request failed") from exc
        finally:
            # Cancelling a Python await alone does not cancel the JS fetch.
            controller.abort()

        if not 200 <= status < 300:
            raise self._api_error(status, body)
        try:
            response = json.loads(body)
        except (ValueError, UnicodeError) as exc:
            raise SeekritError("resolve response is not valid JSON") from exc
        if not isinstance(response, dict) or not isinstance(
            response.get("layers"), list
        ):
            raise SeekritError("resolve response is missing layers")
        return response
