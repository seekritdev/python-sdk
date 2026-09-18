"""LiteLLM secret manager: a gateway reads its provider keys from seekrit.

A LiteLLM proxy is where an organisation's model credentials collect — one
process holding every provider key, plus its own database URL, master key and
salt. LiteLLM's answer is a *secret manager*: config values written
``os.environ/OPENAI_API_KEY`` are looked up through one, and the key never has
to exist in a file or a deployment's environment.

    general_settings:
      key_management_system: custom
      key_management_settings:
        custom_secret_manager: seekrit_secret_manager.SeekritSecretManager
        access_mode: read_only

    model_list:
      - model_name: gpt-5.6-terra
        litellm_params:
          model: openai/gpt-5.6-terra
          api_key: os.environ/OPENAI_API_KEY

**The manager is loaded from a file, not from a package.** LiteLLM splits that
``custom_secret_manager`` value on ``.`` into exactly two parts and imports the
first as ``<directory of config.yaml>/<part>.py``. So ``seekrit.litellm.
SeekritSecretManager`` cannot be written there — it raises ``too many values to
unpack`` — and even a two-part spelling would look for a file, not for this
module. What goes next to ``config.yaml`` is a shim, which is
:data:`SHIM_TEMPLATE` and nothing else::

    # seekrit_secret_manager.py
    from seekrit.litellm import SeekritSecretManager  # noqa: F401

Three things this module does that the two-method interface does not imply.

**It caches.** ``get_secret`` consults the manager *before* the environment and
has no cache of its own, so every ``os.environ/…`` lookup — at startup, per
router rebuild, and (as of LiteLLM 1.83) on the invoke path — is one call in.
Uncached that is an HTTPS round trip plus an unwrap per model call. One resolve
serves every name in the environment, so the cache holds a whole snapshot with
one TTL rather than an entry per name.

**A miss returns ``None``, and so does a failure.** LiteLLM falls back to the
process environment when the manager has nothing, which is what lets seekrit
answer for ``OPENAI_API_KEY`` while ``DATABASE_URL`` keeps coming from the
deployment. It also means an unreachable API *silently* falls back rather than
failing closed — so a failed resolve keeps serving the last good snapshot for
:data:`DEFAULT_STALE_TTL_SECONDS` and logs every time, instead of quietly
handing the gateway whatever the environment happens to hold.

Be ready for what a miss looks like from the other side: LiteLLM turns it into
``ValueError: No secret found in Custom Secret Manager for <name>``, catches it,
and logs it at ERROR *with a traceback* before falling back. Every name the
gateway looks up and seekrit does not hold produces one at startup. They are the
fallback working, not a fault — the way to quiet them is to move the remaining
names into the environment, not to change anything here.

**Reads only.** ``store_virtual_keys`` would have LiteLLM write its generated
virtual keys back through this manager; writing is not something a read-path SDK
can do (it has no encrypt path, and the key hierarchy it would need is the
browser's), so the write methods raise with that explanation rather than
inheriting a bare ``NotImplementedError``. Keep ``access_mode: read_only``.

This is the same boundary as ``seekrit run``: the gateway process holds the
plaintext for as long as it runs. To keep provider keys *out* of it, put
``{{seekrit:OPENAI_API_KEY}}`` in the environment instead and run LiteLLM behind
the egress proxy in forward mode — see the guide.

No extra to install: like :mod:`seekrit.hermes`, this module imports LiteLLM
only inside :func:`secret_manager_class`, so ``pip install seekrit`` into the
environment the proxy already runs in is the whole installation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import urllib.error
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from ._client import Client
from .errors import SeekritApiError, SeekritCryptoError, SeekritError

__all__ = [
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_ERROR_TTL_SECONDS",
    "DEFAULT_STALE_TTL_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_TOKEN_ENV",
    "SHIM_FILENAME",
    "SHIM_TEMPLATE",
    "SecretResolver",
    "secret_manager_class",
]

logger = logging.getLogger(__name__)

#: The environment variable a resolver reads its service token from.
DEFAULT_TOKEN_ENV = "SEEKRIT_TOKEN"

#: How long one resolved snapshot is served before it is fetched again. Five
#: minutes is the usual "rotate and see it within the hour" trade; it is also
#: what bounds how long a revoked secret keeps working in a running gateway.
DEFAULT_CACHE_TTL_SECONDS = 300.0

#: How long a *failed* resolve is remembered. Without this, a proxy that starts
#: while the API is unreachable makes one failing request per model entry per
#: startup — and then one per request afterwards.
DEFAULT_ERROR_TTL_SECONDS = 10.0

#: How long the last good snapshot keeps answering while refreshes fail. Matches
#: the proxy's ``[cache] max_age`` default: long enough to ride out an outage,
#: short enough that a revoked token stops working the same day.
DEFAULT_STALE_TTL_SECONDS = 86400.0

#: Per-resolve timeout. A resolve is one request and one unwrap.
DEFAULT_TIMEOUT_SECONDS = 30.0

#: What the shim file is conventionally called. Only the spelling in
#: ``config.yaml`` has to match it — LiteLLM derives the path from the name.
SHIM_FILENAME = "seekrit_secret_manager.py"

#: The entire shim. LiteLLM imports the *file* beside ``config.yaml``, so this
#: is the one piece that cannot live in the installed package.
SHIM_TEMPLATE = """\
# Loaded by LiteLLM as `seekrit_secret_manager.SeekritSecretManager`.
#
# LiteLLM imports a custom secret manager from a file next to config.yaml, not
# from an installed package, so this shim has to exist. It re-exports the class
# from `seekrit.litellm`; configure it there (or by environment variable), not
# here.
from seekrit.litellm import SeekritSecretManager  # noqa: F401
"""


def _float_env(env: Mapping[str, str], name: str, default: float) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s is not a number (%r) — using %s", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s cannot be negative (%r) — using %s", name, raw, default)
        return default
    return value


class SecretResolver:
    """One seekrit environment, cached, answering by name.

    Framework-free on purpose: everything LiteLLM-shaped lives in
    :func:`secret_manager_class`, and every rule worth testing is here, so the
    tests run in an environment that has never heard of LiteLLM.

    Args:
        token_env: environment variable holding the ``skt_`` service token.
        api_url: API base URL; defaults to ``$SEEKRIT_API_URL``.
        overrides: ``{group_slug: env_slug}`` composed-group overrides.
        timeout: per-resolve timeout in seconds.
        cache_ttl: how long a snapshot is served before refetching. ``0``
            disables caching, which is a supported but expensive choice.
        error_ttl: how long a failure is remembered before retrying.
        stale_ttl: how long the last good snapshot answers while refreshes fail.
        allow: if given, the only names this resolver will answer. Every other
            name is a miss, so LiteLLM falls back to the process environment —
            a courtesy for keeping ``DATABASE_URL`` local, not a boundary: the
            token still resolves the whole environment into this process.
        env: environment mapping to read configuration from.
        client_factory: ``(token, api_url, overrides, timeout) -> client``.
            Exists so the tests can run without an API.
        clock: monotonic clock, for tests.
    """

    __slots__ = (
        "_allow",
        "_api_url",
        "_cache_ttl",
        "_clock",
        "_error_ttl",
        "_factory",
        "_fetched_at",
        "_failed_at",
        "_last_error",
        "_lock",
        "_overrides",
        "_stale_ttl",
        "_timeout",
        "_token_env",
        "_env",
        "_values",
    )

    def __init__(
        self,
        *,
        token_env: str = DEFAULT_TOKEN_ENV,
        api_url: Optional[str] = None,
        overrides: Optional[Mapping[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
        error_ttl: float = DEFAULT_ERROR_TTL_SECONDS,
        stale_ttl: float = DEFAULT_STALE_TTL_SECONDS,
        allow: Optional[Sequence[str]] = None,
        env: Optional[Mapping[str, str]] = None,
        client_factory: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._token_env = token_env
        self._api_url = api_url
        self._overrides = dict(overrides or {})
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._error_ttl = error_ttl
        self._stale_ttl = stale_ttl
        self._allow = frozenset(allow) if allow is not None else None
        self._env = env if env is not None else os.environ
        self._factory = client_factory or _default_client_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._values: Optional[Dict[str, str]] = None
        self._fetched_at = 0.0
        self._failed_at = 0.0
        self._last_error: Optional[str] = None

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None, **overrides: Any
    ) -> "SecretResolver":
        """Build a resolver from the environment, for LiteLLM's zero-arg load.

        The loader instantiates the manager class with no arguments, so this is
        the only configuration surface an operator has that does not involve
        editing the shim. Keyword arguments win over the environment, which is
        what makes subclassing in the shim the *other* way to configure it.
        """
        env = os.environ if env is None else env
        settings: Dict[str, Any] = {
            "token_env": env.get("SEEKRIT_LITELLM_TOKEN_ENV") or DEFAULT_TOKEN_ENV,
            "cache_ttl": _float_env(env, "SEEKRIT_LITELLM_CACHE_TTL", DEFAULT_CACHE_TTL_SECONDS),
            "stale_ttl": _float_env(env, "SEEKRIT_LITELLM_STALE_TTL", DEFAULT_STALE_TTL_SECONDS),
            "env": env,
        }
        allow = (env.get("SEEKRIT_LITELLM_ALLOW") or "").strip()
        if allow:
            settings["allow"] = [name.strip() for name in allow.split(",") if name.strip()]
        settings.update(overrides)
        return cls(**settings)

    # ── reading ──────────────────────────────────────────────────────────────

    def read(self, name: str, timeout: Optional[float] = None) -> Optional[str]:
        """The value for ``name``, or ``None`` if this resolver has none.

        ``None`` covers every kind of miss — not in the environment, not in
        ``allow``, empty, or unreachable — because LiteLLM's contract for a
        miss is to fall back to ``os.environ``, and distinguishing them here
        would only give it a way to fail that it does not implement.

        An **empty** value is a miss rather than an answer: applying ``""`` over
        a credential the environment already holds turns a misconfigured secret
        into a gateway that looks configured and 401s.
        """
        if not name:
            return None
        if self._allow is not None and name not in self._allow:
            return None
        values = self._snapshot(timeout)
        value = values.get(name)
        return value or None

    def refresh(self, timeout: Optional[float] = None) -> None:
        """Drop the cached snapshot so the next read fetches."""
        with self._lock:
            self._fetched_at = 0.0
            self._failed_at = 0.0
        self._snapshot(timeout)

    @property
    def token_env(self) -> str:
        """The environment variable this resolver reads its token from."""
        return self._token_env

    @property
    def token_present(self) -> bool:
        """Whether that variable currently holds something."""
        return bool((self._env.get(self._token_env) or "").strip())

    @property
    def names(self) -> Sequence[str]:
        """Names in the current snapshot, sorted. Never values."""
        with self._lock:
            return tuple(sorted(self._values or {}))

    @property
    def last_error(self) -> Optional[str]:
        """The most recent resolve failure, as a message safe to log."""
        with self._lock:
            return self._last_error

    @property
    def fresh(self) -> bool:
        """Whether a read would be answered without touching the network."""
        with self._lock:
            return self._fresh_locked()

    def _fresh_locked(self) -> bool:
        if self._values is None:
            return False
        if self._cache_ttl <= 0:
            return False
        return (self._clock() - self._fetched_at) < self._cache_ttl

    def _snapshot(self, timeout: Optional[float]) -> Mapping[str, str]:
        # One lock around the fetch, not just around the cache: N concurrent
        # misses at startup should be one resolve, not N. A resolve is short and
        # this is not a hot path once warm.
        with self._lock:
            if self._fresh_locked():
                return self._values or {}
            now = self._clock()
            if self._failed_at and (now - self._failed_at) < self._error_ttl:
                # Still inside the failure window. Serve a stale snapshot if one
                # is young enough, else nothing.
                return self._stale_locked(now)
            try:
                values = self._fetch(timeout)
            except Exception as exc:  # a read must not raise into LiteLLM
                self._failed_at = now
                self._last_error = _describe(exc)
                logger.warning("seekrit resolve failed: %s", self._last_error)
                return self._stale_locked(now)
            self._values = values
            self._fetched_at = now
            self._failed_at = 0.0
            self._last_error = None
            logger.debug("seekrit resolved %d secret(s)", len(values))
            return values

    def _stale_locked(self, now: float) -> Mapping[str, str]:
        if self._values is None:
            return {}
        age = now - self._fetched_at
        if age > self._stale_ttl:
            # Past the staleness bound the snapshot is dropped, so a revoked
            # token stops working even though nothing can replace it. Failing to
            # answer is the fail-closed half of a fallback we do not control.
            logger.warning(
                "seekrit snapshot is %.0fs old and refreshes are failing — dropping it", age
            )
            self._values = None
            return {}
        logger.warning("serving a %.0fs-old seekrit snapshot: %s", age, self._last_error)
        return self._values

    def _fetch(self, timeout: Optional[float]) -> Dict[str, str]:
        token = (self._env.get(self._token_env) or "").strip()
        if not token:
            raise SeekritError(f"{self._token_env} is not set")
        client = self._factory(
            token,
            self._api_url or self._env.get("SEEKRIT_API_URL") or None,
            self._overrides,
            float(timeout) if isinstance(timeout, (int, float)) and timeout else self._timeout,
        )
        return dict(client.resolve())


def _default_client_factory(
    token: str,
    api_url: Optional[str],
    overrides: Mapping[str, str],
    timeout: float,
) -> Client:
    return Client(
        token,
        api_url=api_url,
        overrides=dict(overrides) if overrides else None,
        timeout=timeout,
    )


def _describe(exc: BaseException) -> str:
    """A failure as a message built from status codes and variable names only —
    never from a value, and never from the API's own message body."""
    if isinstance(exc, SeekritApiError):
        if exc.status in (401, 403):
            return f"the service token was rejected ({exc.status})"
        if exc.status == 404:
            return "the service token resolves no environment (404)"
        return f"the API returned {exc.status} {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return "could not reach the seekrit API"
    if isinstance(exc, SeekritCryptoError):
        return "the service token could not decrypt this environment"
    if isinstance(exc, SeekritError):
        return str(exc)
    return f"{type(exc).__name__} while resolving seekrit secrets"


_WRITE_REFUSAL = (
    "seekrit's LiteLLM manager is read-only: this SDK has no encrypt path, and "
    "writing a secret needs the key hierarchy that lives in the dashboard and "
    "the CLI. Set `access_mode: read_only` and `store_virtual_keys: false`, and "
    "keep LiteLLM's virtual keys in its own database."
)


def secret_manager_class() -> type:
    """Build the ``CustomSecretManager`` subclass, importing LiteLLM only now.

    A factory rather than a module-level class so that importing
    :mod:`seekrit.litellm` does not import LiteLLM — which is what lets the
    resolver above be tested, and shipped, without the framework present. The
    module's ``SeekritSecretManager`` attribute calls this on first access, so
    the shim beside ``config.yaml`` stays a single import.
    """
    try:
        from litellm.integrations.custom_secret_manager import (  # type: ignore[import-not-found]
            CustomSecretManager,
        )
    except ImportError as exc:  # pragma: no cover - exercised by the extras install
        raise ImportError(
            "seekrit.litellm needs LiteLLM's custom secret manager API "
            "(litellm.integrations.custom_secret_manager). Install seekrit into the "
            "same environment as the LiteLLM proxy, and use a LiteLLM new enough to "
            "support `key_management_system: custom`."
        ) from exc

    class SeekritSecretManager(CustomSecretManager):  # type: ignore[misc, valid-type]
        """Answer LiteLLM's secret lookups from one seekrit environment."""

        def __init__(self, resolver: Optional[SecretResolver] = None, **kwargs: Any) -> None:
            # LiteLLM's loader instantiates this with no arguments, so every
            # parameter needs a default. A shim that wants different settings
            # subclasses this and passes a resolver.
            super().__init__(secret_manager_name="seekrit", **kwargs)
            self.resolver = resolver or SecretResolver.from_env()

        # ── reads ────────────────────────────────────────────────────────────

        def sync_read_secret(
            self,
            secret_name: str,
            optional_params: Optional[dict] = None,
            timeout: Optional[Any] = None,
        ) -> Optional[str]:
            # `optional_params` carries LiteLLM's own key_management_settings.
            # Nothing in it names a seekrit environment, so it is not read here;
            # configure the resolver in the shim or by environment variable.
            return self.resolver.read(secret_name, _seconds(timeout))

        async def async_read_secret(
            self,
            secret_name: str,
            optional_params: Optional[dict] = None,
            timeout: Optional[Any] = None,
        ) -> Optional[str]:
            seconds = _seconds(timeout)
            if self.resolver.fresh:
                # The common case by a wide margin, and a thread hop for a dict
                # lookup would be the expensive part of it.
                return self.resolver.read(secret_name, seconds)
            # The resolve itself is blocking (this SDK is dependency-free, so
            # urllib), and blocking the proxy's event loop on it would stall
            # every in-flight request rather than just this lookup.
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.resolver.read, secret_name, seconds)

        # ── writes, refused ──────────────────────────────────────────────────

        async def async_write_secret(
            self,
            secret_name: str,
            secret_value: str,
            description: Optional[str] = None,
            optional_params: Optional[dict] = None,
            timeout: Optional[Any] = None,
            tags: Optional[Any] = None,
        ) -> Dict[str, Any]:
            raise NotImplementedError(_WRITE_REFUSAL)

        async def async_delete_secret(
            self,
            secret_name: str,
            recovery_window_in_days: Optional[int] = 7,
            optional_params: Optional[dict] = None,
            timeout: Optional[Any] = None,
        ) -> dict:
            raise NotImplementedError(_WRITE_REFUSAL)

        # ── diagnostics ──────────────────────────────────────────────────────

        def validate_environment(self) -> bool:
            """Whether a token is present. Called at startup, so a missing
            credential is reported there rather than as a 401 per model."""
            if not self.resolver.token_present:
                logger.error(
                    "%s is not set — seekrit will answer no secrets and LiteLLM will "
                    "fall back to the process environment",
                    self.resolver.token_env,
                )
                return False
            return True

        async def async_health_check(self) -> bool:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.resolver.refresh)
            return self.resolver.last_error is None

        def __repr__(self) -> str:  # pragma: no cover - debugging aid
            # Names only: this object holds live credentials, and its repr must
            # not be what puts them in a log line.
            return f"SeekritSecretManager(names={list(self.resolver.names)!r})"

    return SeekritSecretManager


def _seconds(timeout: Any) -> Optional[float]:
    """LiteLLM passes ``float | httpx.Timeout | None``. Only the number can mean
    anything to a urllib fetch; an ``httpx.Timeout`` is ignored rather than
    guessed at from one of its four fields."""
    return float(timeout) if isinstance(timeout, (int, float)) else None


def __getattr__(name: str) -> Any:
    """Build ``SeekritSecretManager`` on first access (PEP 562).

    So the shim beside ``config.yaml`` is one import line, while importing this
    module still costs no LiteLLM import.
    """
    if name == "SeekritSecretManager":
        cls = secret_manager_class()
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
