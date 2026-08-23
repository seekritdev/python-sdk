"""Hermes Agent secret sources: resolve provider credentials from seekrit.

Hermes reads credentials from ``~/.hermes/.env`` and the process environment.
A *secret source* fills that environment from somewhere else at startup, before
Hermes reads any of it — which is the seam this module implements twice, because
Hermes distinguishes two shapes and they behave differently under precedence:

``seekrit`` (**bulk**)
    One seekrit environment, whole. Every secret in it becomes an environment
    variable. Nothing to enumerate and nothing to maintain as the environment
    grows — the usual case::

        secrets:
          sources: [seekrit]
          seekrit:
            enabled: true

``seekrit_refs`` (**mapped**)
    Explicit ``VAR: skt://NAME`` bindings. Use it to rename a secret, to take
    only some of an environment, to read more than one environment, or when the
    binding has to win a contest against a bulk source — Hermes lets a mapped
    claim beat a bulk one::

        secrets:
          sources: [seekrit_refs]
          seekrit_refs:
            enabled: true
            tokens:
              billing: SEEKRIT_TOKEN_BILLING
            env:
              OPENAI_API_KEY: skt://OPENAI_KEY
              STRIPE_SECRET_KEY: skt://billing/STRIPE_SECRET_KEY

Both are registered by ``register(ctx)`` and both are inert until enabled, so
installing this costs nothing until a config section turns one on.

**Tokens are named, never written here.** A ``tokens:`` entry names an
*environment variable* that holds a service token — ``config.yaml`` is not a
place for a credential, and Hermes' own docs say so. Every token variable this
module reads is also returned from ``protected_env_vars``, so no secret source
(including this one) can overwrite the credential the next startup needs.

Importing this module does **not** import Hermes. The ``SecretSource`` subclasses
are built inside :func:`secret_source_classes`, called from :func:`register`, so
the resolution logic below is plain Python that can be tested — and shipped —
without the framework present. That is the same reason the other adapters in
this SDK type their framework structurally: a released SDK must not break on a
host version bump.
"""

from __future__ import annotations

import os
import re
import urllib.error
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ._client import Client
from .errors import SeekritApiError, SeekritCryptoError, SeekritError

__all__ = [
    "BULK_SOURCE_NAME",
    "MAPPED_SOURCE_NAME",
    "Outcome",
    "REFERENCE_SCHEME",
    "SecretReference",
    "parse_reference",
    "register",
    "resolve_bulk",
    "resolve_mapped",
    "secret_source_classes",
]

BULK_SOURCE_NAME = "seekrit"
MAPPED_SOURCE_NAME = "seekrit_refs"

#: The URI scheme the mapped source owns, the way 1Password owns ``op``.
REFERENCE_SCHEME = "skt"

#: The environment variable a source reads its token from unless told otherwise.
DEFAULT_TOKEN_ENV = "SEEKRIT_TOKEN"

#: Hermes' own default is 120s; a resolve is one request and one unwrap.
DEFAULT_TIMEOUT_SECONDS = 30

_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Outcome:
    """What a fetch produced, in terms this module owns.

    Deliberately not Hermes' ``FetchResult``: keeping the result type local is
    what lets every test below run without the framework installed. The adapter
    in :func:`secret_source_classes` copies these three fields across, turning
    :attr:`error_kind` into the real ``ErrorKind`` member of the same name.
    """

    __slots__ = ("secrets", "error", "error_kind")

    def __init__(
        self,
        secrets: Optional[Mapping[str, str]] = None,
        error: str = "",
        error_kind: str = "",
    ) -> None:
        self.secrets: Dict[str, str] = dict(secrets or {})
        self.error = error
        self.error_kind = error_kind

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        # Names only. An Outcome holds live credentials, so its repr must not be
        # the thing that puts them in a log line.
        return (
            f"Outcome(names={sorted(self.secrets)!r}, "
            f"error_kind={self.error_kind!r}, error={self.error!r})"
        )


class SecretReference:
    """A parsed ``skt://[token-alias/]NAME`` reference."""

    __slots__ = ("token_alias", "name")

    def __init__(self, name: str, token_alias: Optional[str] = None) -> None:
        self.name = name
        self.token_alias = token_alias

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SecretReference)
            and other.name == self.name
            and other.token_alias == self.token_alias
        )

    def __repr__(self) -> str:
        alias = f"{self.token_alias}/" if self.token_alias else ""
        return f"SecretReference({REFERENCE_SCHEME}://{alias}{self.name})"


def parse_reference(raw: str) -> Optional[SecretReference]:
    """Parse ``skt://NAME`` or ``skt://alias/NAME``; ``None`` if malformed.

    A service token is bound to exactly one environment, so a reference cannot
    address an arbitrary app and environment the way ``op://vault/item/field``
    can — it names a *token* (by alias) and a secret in that token's
    environment. Inventing an ``skt://app/env/NAME`` spelling this SDK has no
    way to resolve would be a reference that always fails.
    """
    if not isinstance(raw, str):
        return None
    prefix = f"{REFERENCE_SCHEME}://"
    if not raw.startswith(prefix):
        return None
    rest = raw[len(prefix) :].strip()
    if not rest:
        return None
    parts = rest.split("/")
    if len(parts) == 1:
        name = parts[0]
        return SecretReference(name) if _ENV_VAR_RE.match(name) else None
    if len(parts) != 2:
        return None
    alias, name = parts
    if not alias or not _ENV_VAR_RE.match(name):
        return None
    return SecretReference(name, alias)


# ── configuration ────────────────────────────────────────────────────────────


def _str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


def _token_env_names(cfg: Mapping[str, Any]) -> List[str]:
    """Every environment variable this config expects to hold a token."""
    names = [str(cfg.get("token_env") or DEFAULT_TOKEN_ENV)]
    tokens = cfg.get("tokens")
    if isinstance(tokens, Mapping):
        names.extend(str(var) for var in tokens.values())
    return [name for name in dict.fromkeys(names) if _ENV_VAR_RE.match(name)]


def _client_for(
    token_env: str,
    cfg: Mapping[str, Any],
    env: Mapping[str, str],
) -> Client:
    token = (env.get(token_env) or "").strip()
    if not token:
        raise _NotConfigured(f"{token_env} is not set")
    api_url = cfg.get("api_url") or env.get("SEEKRIT_API_URL") or None
    overrides = cfg.get("overrides") if isinstance(cfg.get("overrides"), Mapping) else None
    timeout = float(cfg.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    return Client(
        token,
        api_url=str(api_url) if api_url else None,
        overrides=dict(overrides) if overrides else None,
        timeout=timeout,
    )


class _NotConfigured(Exception):
    """Enabled, but missing something the operator has to supply."""


# ── error classification ─────────────────────────────────────────────────────


def _classify(exc: BaseException) -> Tuple[str, str]:
    """Map an SDK failure onto a Hermes ``ErrorKind`` name and a safe message.

    Every message here is built from status codes, error codes, and variable
    names — never from a value, and never from the API's own message body, which
    a caller has no way to be sure of.
    """
    if isinstance(exc, _NotConfigured):
        return "NOT_CONFIGURED", str(exc)
    if isinstance(exc, SeekritApiError):
        if exc.status in (401, 403):
            return "AUTH_FAILED", f"seekrit rejected the service token ({exc.status})"
        if exc.status == 404:
            return "REF_INVALID", "the service token resolves no environment (404)"
        if exc.status == 429 or exc.status >= 500:
            return "NETWORK", f"seekrit returned {exc.status}"
        return "INTERNAL", f"seekrit returned {exc.status} {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return "NETWORK", "could not reach the seekrit API"
    # A token that parses but cannot decrypt is the wrong credential for this
    # environment, which is an auth problem however the crypto reports it.
    if isinstance(exc, SeekritCryptoError):
        return "AUTH_FAILED", "the service token could not decrypt this environment"
    if isinstance(exc, SeekritError):
        return "NOT_CONFIGURED", str(exc)
    return "INTERNAL", f"{type(exc).__name__} while resolving seekrit secrets"


REMEDIATION = {
    "NOT_CONFIGURED": (
        "Set SEEKRIT_TOKEN in ~/.hermes/.env to a service token "
        "(`seekrit token create --app <app> --env <env>`)."
    ),
    "AUTH_FAILED": (
        "The service token was rejected or cannot decrypt this environment. "
        "Check it is granted a key for the environment it points at "
        "(`seekrit access list`)."
    ),
    "AUTH_EXPIRED": "The service token has expired — mint a new one with `seekrit token create`.",
    "REF_INVALID": (
        f"A reference must be {REFERENCE_SCHEME}://NAME or "
        f"{REFERENCE_SCHEME}://<token-alias>/NAME, and the alias must appear under `tokens:`."
    ),
    "NETWORK": "Could not reach the seekrit API. Check connectivity and SEEKRIT_API_URL.",
    "EMPTY_VALUE": "The secret exists but holds an empty value — set it with `seekrit secrets set`.",
    "TIMEOUT": "Raise `secrets.<source>.timeout_seconds` or check connectivity to the seekrit API.",
}


# ── the two resolvers, framework-free ────────────────────────────────────────


def resolve_bulk(
    cfg: Mapping[str, Any],
    env: Optional[Mapping[str, str]] = None,
    client_factory: Any = None,
) -> Outcome:
    """Resolve a whole seekrit environment into ``{VAR: value}``.

    ``include`` / ``exclude`` narrow what is contributed. They are a courtesy,
    not a boundary: the token still resolves the entire environment, so a name
    that must not reach this process needs a narrower token, not an ``exclude``.

    ``client_factory`` exists so the tests can run without an API.
    """
    env = os.environ if env is None else env
    factory = client_factory or _client_for
    token_env = str(cfg.get("token_env") or DEFAULT_TOKEN_ENV)

    try:
        client = factory(token_env, cfg, env)
        values = client.resolve()
    except Exception as exc:  # the contract: fetch() never raises
        kind, message = _classify(exc)
        return Outcome(error=message, error_kind=kind)

    include = set(_str_list(cfg.get("include")))
    exclude = set(_str_list(cfg.get("exclude")))
    secrets: Dict[str, str] = {}
    skipped_empty: List[str] = []
    for name, value in values.items():
        if include and name not in include:
            continue
        if name in exclude:
            continue
        if not _ENV_VAR_RE.match(name):
            # Not addressable as an environment variable, so not this source's
            # to contribute — and silently mangling the name would be worse.
            continue
        if value == "":
            skipped_empty.append(name)
            continue
        secrets[name] = value

    if not secrets and skipped_empty:
        # Never apply "" over a good credential — Hermes has a kind for exactly
        # this, and applying it would look like a working config with a broken key.
        return Outcome(
            error=f"every resolved secret was empty: {', '.join(sorted(skipped_empty))}",
            error_kind="EMPTY_VALUE",
        )
    return Outcome(secrets)


def resolve_mapped(
    cfg: Mapping[str, Any],
    env: Optional[Mapping[str, str]] = None,
    client_factory: Any = None,
) -> Outcome:
    """Resolve an explicit ``{VAR: skt://...}`` map.

    Each distinct token is resolved once, not once per reference: a map of
    twenty variables against one environment is one request.
    """
    env = os.environ if env is None else env
    factory = client_factory or _client_for

    mapping = cfg.get("env")
    if not isinstance(mapping, Mapping) or not mapping:
        return Outcome(
            error=f"secrets.{MAPPED_SOURCE_NAME}.env is empty — nothing to resolve",
            error_kind="NOT_CONFIGURED",
        )
    aliases = cfg.get("tokens") if isinstance(cfg.get("tokens"), Mapping) else {}

    # Parse the whole map before touching the network, so a typo is reported as
    # a bad reference rather than as whatever the request happens to fail with.
    wanted: Dict[str, List[Tuple[str, str]]] = {}
    for var, raw in mapping.items():
        var = str(var)
        if not _ENV_VAR_RE.match(var):
            return Outcome(error=f"not a usable variable name: {var}", error_kind="REF_INVALID")
        ref = parse_reference(str(raw))
        if ref is None:
            return Outcome(error=f"not a valid reference for {var}", error_kind="REF_INVALID")
        if ref.token_alias is None:
            token_env = str(cfg.get("token_env") or DEFAULT_TOKEN_ENV)
        else:
            token_env = str(aliases.get(ref.token_alias) or "")
            if not token_env:
                return Outcome(
                    error=(
                        f"{var} names token alias '{ref.token_alias}', which is not "
                        f"listed under secrets.{MAPPED_SOURCE_NAME}.tokens"
                    ),
                    error_kind="REF_INVALID",
                )
        wanted.setdefault(token_env, []).append((var, ref.name))

    secrets: Dict[str, str] = {}
    missing: List[str] = []
    empty: List[str] = []
    for token_env, bindings in wanted.items():
        try:
            values = factory(token_env, cfg, env).resolve()
        except Exception as exc:
            kind, message = _classify(exc)
            return Outcome(error=message, error_kind=kind)
        for var, name in bindings:
            value = values.get(name)
            if value is None:
                missing.append(f"{var} -> {name}")
            elif value == "":
                empty.append(var)
            else:
                secrets[var] = value

    if missing:
        # Fail the source rather than contribute a partial map: a mapped binding
        # is an explicit statement that this variable comes from here, and half
        # of one is a config error worth surfacing at startup.
        return Outcome(
            error=f"no such secret for: {', '.join(sorted(missing))}",
            error_kind="REF_INVALID",
        )
    if empty and not secrets:
        return Outcome(
            error=f"every mapped secret was empty: {', '.join(sorted(empty))}",
            error_kind="EMPTY_VALUE",
        )
    return Outcome(secrets)


# ── config schemas ───────────────────────────────────────────────────────────

_COMMON_SCHEMA: Dict[str, Any] = {
    "enabled": {"type": "boolean", "default": False},
    "override_existing": {"type": "boolean", "default": True},
    "timeout_seconds": {"type": "integer", "default": DEFAULT_TIMEOUT_SECONDS},
    "token_env": {"type": "string", "default": DEFAULT_TOKEN_ENV},
    "api_url": {"type": "string"},
}

BULK_SCHEMA: Dict[str, Any] = dict(
    _COMMON_SCHEMA,
    include={"type": "array", "items": {"type": "string"}},
    exclude={"type": "array", "items": {"type": "string"}},
    overrides={"type": "object", "additionalProperties": {"type": "string"}},
)

MAPPED_SCHEMA: Dict[str, Any] = dict(
    _COMMON_SCHEMA,
    env={"type": "object", "additionalProperties": {"type": "string"}},
    tokens={"type": "object", "additionalProperties": {"type": "string"}},
)


# ── the Hermes adapter ───────────────────────────────────────────────────────


def secret_source_classes() -> Tuple[type, type]:
    """Build the two ``SecretSource`` subclasses, importing Hermes only now.

    Returns ``(bulk, mapped)``. Raises :class:`ImportError` when Hermes is not
    installed, which is the correct outcome for a caller that only reaches here
    from :func:`register`.
    """
    try:
        from agent.secret_sources.base import (  # type: ignore[import-not-found]
            ErrorKind,
            FetchResult,
            SecretSource,
        )
    except ImportError as exc:  # pragma: no cover - exercised by the extras install
        raise ImportError(
            "seekrit.hermes needs Hermes Agent's secret-source API "
            "(agent.secret_sources.base); it is importable only inside a Hermes process."
        ) from exc

    def to_fetch_result(outcome: Outcome) -> Any:
        result = FetchResult()
        result.secrets = outcome.secrets
        if outcome.error:
            result.error = outcome.error
            # `getattr` rather than a hard reference: an ErrorKind this Hermes
            # does not have costs an INTERNAL, not an AttributeError at startup.
            result.error_kind = getattr(ErrorKind, outcome.error_kind, ErrorKind.INTERNAL)
        return result

    class _Base(SecretSource):  # type: ignore[misc, valid-type]
        """Shared behaviour: timeouts, token protection, and remediation hints."""

        def override_existing(self, cfg: Mapping[str, Any]) -> bool:
            # `True`, like both bundled sources, because seekrit is the store of
            # record: after a rotation the new value must win over whatever a
            # stale `.env` still holds. Pin individual variables with
            # `preserve_existing` when a local override is the point.
            return bool(cfg.get("override_existing", True))

        def fetch_timeout_seconds(self, cfg: Mapping[str, Any]) -> int:
            return int(cfg.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)

        def protected_env_vars(self, cfg: Mapping[str, Any]) -> Any:
            # Every variable this source reads a token from. Without this, one
            # source can overwrite the credential another needs next startup.
            return frozenset(_token_env_names(cfg))

        def remediation(self, kind: Any, cfg: Mapping[str, Any]) -> str:
            name = getattr(kind, "name", str(kind))
            return REMEDIATION.get(name, "")

    class SeekritSecretSource(_Base):
        """One seekrit environment, whole."""

        name = BULK_SOURCE_NAME
        label = "seekrit"
        shape = "bulk"

        def fetch(self, cfg: Mapping[str, Any], home_path: Any) -> Any:
            return to_fetch_result(resolve_bulk(cfg))

        def config_schema(self) -> Dict[str, Any]:
            return dict(BULK_SCHEMA)

    class SeekritReferenceSecretSource(_Base):
        """Explicit ``VAR: skt://NAME`` bindings."""

        name = MAPPED_SOURCE_NAME
        label = "seekrit (refs)"
        shape = "mapped"
        scheme = REFERENCE_SCHEME

        def fetch(self, cfg: Mapping[str, Any], home_path: Any) -> Any:
            return to_fetch_result(resolve_mapped(cfg))

        def config_schema(self) -> Dict[str, Any]:
            return dict(MAPPED_SCHEMA)

    return SeekritSecretSource, SeekritReferenceSecretSource


def register(ctx: Any) -> None:
    """Hermes plugin entry point: register both secret sources.

    Registering both is free — a source does nothing until a ``secrets.<name>``
    section enables it — and it means switching between whole-environment and
    explicit-binding mode is a config edit rather than a reinstall.
    """
    bulk, mapped = secret_source_classes()
    ctx.register_secret_source(bulk())
    ctx.register_secret_source(mapped())
