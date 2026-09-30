"""Configuration and wire conventions shared by the two resolve clients."""

from __future__ import annotations

import json
import os
from typing import Mapping, Optional
from urllib.parse import urlencode

from ._crypto import TokenKey
from .errors import SeekritApiError, SeekritError

DEFAULT_API_URL = "https://api.seekrit.dev"


class _ClientConfig:
    def __init__(
        self,
        token: Optional[str] = None,
        *,
        api_url: Optional[str] = None,
        overrides: Optional[Mapping[str, str]] = None,
        timeout: float = 30.0,
        interpolate: bool = True,
    ) -> None:
        token = token or os.environ.get("SEEKRIT_TOKEN")
        if not token:
            raise SeekritError("no service token: pass token= or set SEEKRIT_TOKEN")
        self._token = token
        self._key = TokenKey.parse(token)
        self._api_url = (
            api_url or os.environ.get("SEEKRIT_API_URL") or DEFAULT_API_URL
        ).rstrip("/")
        self._overrides = dict(overrides or {})
        self._timeout = timeout
        self._interpolate = interpolate

    def _resolve_url(self) -> str:
        query = urlencode(
            [("with", f"{g}:{e}") for g, e in sorted(self._overrides.items())]
        )
        return self._api_url + "/v1/resolve" + ("?" + query if query else "")

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
