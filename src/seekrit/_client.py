"""The resolve client: fetch ``GET /v1/resolve`` and decrypt it locally."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, MutableMapping, Optional, Tuple

from ._client_config import DEFAULT_API_URL, _ClientConfig
from ._crypto import materialize
from .errors import SeekritApiError, SeekritCryptoError, SeekritError


class Client(_ClientConfig):
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
        interpolate: expand ``${OTHER_SECRET}`` references in resolved values
            (default ``True``); ``False`` returns the stored text verbatim.
    """

    def resolve(self) -> Dict[str, str]:
        """Fetch, decrypt, and merge; return ``{NAME: value}``."""
        return materialize(self._fetch(), self._key, interpolate=self._interpolate)

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
        pass ``override=True`` to let resolved secrets take precedence. Note that
        :func:`seekrit.load` — the one-call front door — defaults the other way.
        Returns the merged secrets that were resolved.
        """
        target = os.environ if env is None else env
        merged = self.resolve()
        for name, value in merged.items():
            if override or name not in target:
                target[name] = value
        return merged

    def _resolve_detailed(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """:meth:`resolve` plus the response's ``scope`` slugs, in one request.

        The scope names the org/app/environment the token is bound to — labels,
        not secrets. Used by :func:`seekrit.load` to say what it loaded.
        """
        response = self._fetch()
        secrets = materialize(response, self._key, interpolate=self._interpolate)
        return secrets, dict(response.get("scope") or {})

    # -- internal ---------------------------------------------------------

    def _fetch(self) -> dict:
        request = urllib.request.Request(
            self._resolve_url(),
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


__all__ = ["Client", "DEFAULT_API_URL", "SeekritError", "SeekritApiError", "SeekritCryptoError"]
