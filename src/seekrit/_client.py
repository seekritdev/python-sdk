"""The resolve client: fetch ``GET /v1/resolve`` and decrypt it locally."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, Mapping, MutableMapping, Optional

from ._crypto import TokenKey, materialize
from .errors import SeekritApiError, SeekritCryptoError, SeekritError

DEFAULT_API_URL = "https://api.seekrit.dev"


class Client:
    """A read-only seekrit client bound to one service token.

    A service token selects exactly one app environment (plus its composed
    group slices); resolving returns the merged, decrypted secrets for it.

    Args:
        token: ``skt_...`` service token. Defaults to ``$SEEKRIT_TOKEN``.
        api_url: API base URL. Defaults to ``$SEEKRIT_API_URL`` or
            ``https://api.seekrit.dev``.
        overrides: optional ``{group_slug: env_slug}`` map to pull a different
            environment slice of a composed group (the ``?with=`` override).
        timeout: per-request timeout in seconds.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        api_url: Optional[str] = None,
        overrides: Optional[Mapping[str, str]] = None,
        timeout: float = 30.0,
    ) -> None:
        token = token or os.environ.get("SEEKRIT_TOKEN")
        if not token:
            raise SeekritError("no service token: pass token= or set SEEKRIT_TOKEN")
        self._token = token
        self._key = TokenKey.parse(token)  # fail fast on a bad token
        self._api_url = (api_url or os.environ.get("SEEKRIT_API_URL") or DEFAULT_API_URL).rstrip("/")
        self._overrides = dict(overrides or {})
        self._timeout = timeout

    def resolve(self) -> Dict[str, str]:
        """Fetch, decrypt, and merge; return ``{NAME: value}``."""
        return materialize(self._fetch(), self._key)

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Return a single secret's value, or ``default`` if it is not present."""
        return self.resolve().get(name, default)

    def into_env(
        self,
        env: Optional[MutableMapping[str, str]] = None,
        *,
        override: bool = False,
    ) -> Dict[str, str]:
        """Load resolved secrets into ``env`` (default ``os.environ``).

        By default an existing variable is left untouched (process env wins);
        pass ``override=True`` to let resolved secrets take precedence.
        Returns the merged secrets that were resolved.
        """
        target = os.environ if env is None else env
        merged = self.resolve()
        for name, value in merged.items():
            if override or name not in target:
                target[name] = value
        return merged

    # -- internal ---------------------------------------------------------

    def _fetch(self) -> dict:
        url = self._api_url + "/v1/resolve"
        query = "&".join(f"with={g}:{e}" for g, e in sorted(self._overrides.items()))
        if query:
            url += "?" + query
        request = urllib.request.Request(
            url,
            method="GET",
            headers={"authorization": f"Bearer {self._token}", "accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise self._api_error(exc.code, exc.read()) from exc
        except urllib.error.URLError as exc:
            raise SeekritError(f"resolve request failed: {exc.reason}") from exc

    @staticmethod
    def _api_error(status: int, body: bytes) -> SeekritApiError:
        code, message = "internal", f"HTTP {status}"
        try:
            error = json.loads(body).get("error", {})
            code = error.get("code", code)
            message = error.get("message", message)
        except (ValueError, AttributeError):
            pass
        return SeekritApiError(status, code, message)


__all__ = ["Client", "DEFAULT_API_URL", "SeekritError", "SeekritApiError", "SeekritCryptoError"]
