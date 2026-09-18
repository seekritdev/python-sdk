"""The HTTP-client-independent half of the placeholder-substituting transports.

Two modules bind this to a client: :mod:`seekrit.transport` to ``httpx`` and
:mod:`seekrit.transport_httpx2` to ``httpx2``. Everything that decides anything —
the allowlist check, the scope, the resolve cache, the refusal — lives here, so
the two bindings cannot drift in behaviour. A binding supplies only the module to
build requests and responses with, and the base classes to inherit from.

``httpx2`` is a separate distribution, not a newer release of ``httpx``, so its
``BaseTransport`` is an unrelated class: a client that accepts one rejects the
other. That is the whole reason there are two bindings rather than one.

Nothing here is public API. Import from :mod:`seekrit.transport` or
:mod:`seekrit.transport_httpx2`.
"""

from __future__ import annotations

import asyncio
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from ._client import Client
from ._policy import AllowRule, evaluate, narrow, rules_from_allow
from ._scope import Scope, current_scope
from ._substitute import Lookup, has_placeholder, substitute
from .errors import SeekritError, SeekritSubstitutionError

#: Headers the HTTP client must recompute when we change the body.
_RECOMPUTED = ("content-length", "transfer-encoding")

InjectHook = Callable[[Dict[str, object]], None]
RefuseHook = Callable[[SeekritSubstitutionError], None]
ScopeSource = Callable[[], Optional[Scope]]


class ResolveSource(Protocol):
    """The one method a resolve source needs. :class:`seekrit.Client` satisfies it."""

    def resolve(self) -> Dict[str, str]:  # pragma: no cover - a structural type
        ...


#: Where resolved values come from. A single source is bound to one set of group
#: overrides, so a per-tenant setup passes a **callable** of the overrides.
ClientSource = Union[ResolveSource, Callable[[Optional[Mapping[str, str]]], ResolveSource]]


def refusal_body(error: SeekritSubstitutionError) -> str:
    """The body text of a refusal. Never contains a value."""
    if error.code == "denied":
        return (
            "placeholder {{seekrit:"
            + error.secret_name
            + "}} is not allowed toward this upstream"
        )
    if error.code == "scope_required":
        return "no scope is in effect and require_scope is set"
    return (
        "placeholder {{seekrit:"
        + error.secret_name
        + "}} references a secret that is not available"
    )


def refusal_response(hx: Any, error: SeekritSubstitutionError) -> Any:
    """The 403 the proxy answers with, so both halves fail the same way.

    A provider SDK wraps anything its HTTP layer *raises* into its own opaque
    connection error **and retries it** — a denied placeholder surfaces as
    "Connection error" after six attempts instead of naming the secret. A 403 is
    terminal in every provider SDK, and mirrors ``Reject::into_response`` in
    ``apps/proxy/src/proxy.rs`` verbatim, so swapping this transport for the
    proxy does not change error handling.
    """
    return hx.Response(
        403,
        text=refusal_body(error),
        headers={
            # Machine-checkable, so a caller can tell our refusal from an
            # upstream 403.
            "x-seekrit-refusal": error.code,
            "x-seekrit-secret": error.secret_name,
        },
    )


def build_rules(
    allow: Optional[Mapping[str, Sequence[str]]],
    rules: Optional[Sequence[AllowRule]],
) -> List[AllowRule]:
    built: List[AllowRule] = list(rules or [])
    if allow:
        built.extend(rules_from_allow(allow))
    if not built:
        raise SeekritError(
            "a seekrit transport needs an allowlist: pass allow={...} or rules=[...]"
        )
    return built


class Resolver:
    """Resolve and decrypt, cached per scope for ``ttl_seconds``.

    Values live in memory only. A concurrent burst may resolve more than once —
    harmless, and cheaper than a lock on every request.
    """

    def __init__(
        self,
        *,
        client: Optional["ResolveSource"],
        token: Optional[str],
        api_url: Optional[str],
        ttl_seconds: float,
    ) -> None:
        self._client = client
        self._token = token
        self._api_url = api_url
        self._ttl = max(0.0, ttl_seconds)
        self._cache: Dict[str, Tuple[float, Dict[str, str]]] = {}

    def _client_for(self, scope: Optional[Scope]) -> "ResolveSource":
        """The resolve source for one scope.

        The awkward case is a caller who passed a single ``client`` *and* a scope
        with group overrides: that client is bound to its own overrides and
        cannot be re-scoped, so silently using it would resolve the wrong tenant.
        If a token is available we build a correctly-scoped client; if not, say
        exactly that rather than surfacing "no service token" from three frames
        down.
        """
        overrides = scope.overrides if scope else None
        if callable(self._client):
            return self._client(overrides)
        if not overrides and self._client is not None:
            return self._client
        try:
            return Client(self._token, api_url=self._api_url, overrides=overrides)
        except SeekritError:
            if self._client is not None:
                raise SeekritError(
                    "a scope with group overrides cannot reuse a single client, which is bound "
                    "to its own overrides: pass client as a callable of the overrides, or pass "
                    "token= so a scoped client can be built"
                ) from None
            raise

    def _cached(self, key: str) -> Optional[Dict[str, str]]:
        hit = self._cache.get(key)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        return None

    def _store(self, key: str, values: Dict[str, str]) -> None:
        if self._ttl > 0:
            self._cache[key] = (time.monotonic() + self._ttl, values)

    def values(self, scope: Optional[Scope]) -> Dict[str, str]:
        key = scope.key() if scope else ""
        cached = self._cached(key)
        if cached is not None:
            return cached
        values = self._client_for(scope).resolve()
        self._store(key, values)
        return values

    async def avalues(self, scope: Optional[Scope]) -> Dict[str, str]:
        key = scope.key() if scope else ""
        cached = self._cached(key)
        if cached is not None:
            return cached
        client = self._client_for(scope)
        # Client.resolve() is blocking (urllib), so keep it off the event loop.
        values = await asyncio.get_running_loop().run_in_executor(None, client.resolve)
        self._store(key, values)
        return values


class Rewriter:
    """Decide, substitute, rebuild the request.

    ``hx`` is the HTTP client module — ``httpx`` or ``httpx2``. Only three things
    here touch it: the "body was never read" exception, the rebuilt request, and
    the refusal response.
    """

    def __init__(
        self,
        *,
        hx: Any,
        rules: List[AllowRule],
        scan_body: bool,
        require_scope: bool,
        scope_source: Optional[ScopeSource],
        on_inject: Optional[InjectHook],
        refusal: str,
        on_refuse: Optional[RefuseHook],
    ) -> None:
        self._hx = hx
        self._rules = rules
        self._scan_body = scan_body
        self._require_scope = require_scope
        self._scope_source = scope_source
        self._on_inject = on_inject
        if refusal not in ("respond", "raise"):
            raise SeekritError('refusal must be "respond" or "raise"')
        self._refusal = refusal
        self._on_refuse = on_refuse

    def scope(self) -> Optional[Scope]:
        return self._scope_source() if self._scope_source else current_scope()

    def _body(self, request: Any) -> Optional[str]:
        if not self._scan_body:
            return None
        try:
            raw = request.content
        except self._hx.RequestNotRead:
            return None  # a streamed body is never buffered here
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None  # binary upload: nothing to scan

    def needs_work(self, request: Any) -> bool:
        """Whether this request carries a placeholder anywhere we scan."""
        if has_placeholder(str(request.url)):
            return True
        for _, value in request.headers.multi_items():
            if has_placeholder(value):
                return True
        body = self._body(request)
        return body is not None and has_placeholder(body)

    def rewrite(
        self,
        request: Any,
        scope: Optional[Scope],
        values: Mapping[str, str],
    ) -> Any:
        # `is not None`, not truthiness: an *empty* allow list is the whole point
        # of an exhaustive `tools={...}` — a tool nobody named may inject nothing.
        # Reading `()` as "no opinion" would hand it the unnarrowed rules instead.
        narrowed = scope is not None and scope.allow is not None
        rules = narrow(self._rules, scope.allow) if narrowed else self._rules
        host = request.url.host
        path = request.url.path
        method = request.method.upper()
        injected: set = set()

        def lookup(name: str) -> Lookup:
            verdict = evaluate(rules, host=host, method=method, path=path, secret=name)
            if verdict.decision != "allow":
                return Lookup.denied(verdict.decision)
            if name not in values:
                return Lookup.unknown()
            injected.add(name)
            return Lookup.found(values[name])

        url_text = str(request.url)
        new_url, _ = substitute(url_text, lookup)

        headers: List[Tuple[str, str]] = []
        for name, value in request.headers.multi_items():
            headers.append((name, substitute(value, lookup)[0]))

        body = self._body(request)
        new_body: Optional[bytes] = None
        if body is not None:
            rewritten, _ = substitute(body, lookup)
            if rewritten != body:
                new_body = rewritten.encode("utf-8")

        if injected and self._on_inject:
            self._on_inject(
                {
                    "host": host,
                    "method": method,
                    "path": path,
                    "names": sorted(injected),
                    "label": scope.label if scope else "",
                }
            )

        if new_body is None and new_url == url_text and headers == list(request.headers.multi_items()):
            return request

        if new_body is not None:
            # Let the client recompute the framing headers for the new length.
            headers = [(k, v) for k, v in headers if k.lower() not in _RECOMPUTED]
            return self._hx.Request(
                method=request.method,
                url=new_url,
                headers=headers,
                content=new_body,
                extensions=request.extensions,
            )
        try:
            content = request.content
        except self._hx.RequestNotRead:
            return self._hx.Request(
                method=request.method,
                url=new_url,
                headers=headers,
                stream=request.stream,
                extensions=request.extensions,
            )
        return self._hx.Request(
            method=request.method,
            url=new_url,
            headers=headers,
            content=content,
            extensions=request.extensions,
        )

    def guard(self, scope: Optional[Scope]) -> None:
        if self._require_scope and scope is None:
            raise SeekritSubstitutionError("scope_required", "")

    def refuse(self, error: SeekritSubstitutionError) -> Optional[Any]:
        """Turn a refusal into a 403, or return ``None`` to let it propagate."""
        if self._on_refuse:
            self._on_refuse(error)
        return refusal_response(self._hx, error) if self._refusal == "respond" else None


class _Core:
    """Constructor shared by both transports, sync and async.

    A binding sets ``_hx`` to its HTTP client module. Keeping the signature in
    one place is what makes the two bindings take identical arguments — a new
    option reaches both, or neither.
    """

    #: The HTTP client module this subclass is bound to.
    _hx: Any = None
    #: Attribute on ``_hx`` holding the default inner transport class.
    _default_transport: str = ""

    def __init__(
        self,
        *,
        allow: Optional[Mapping[str, Sequence[str]]] = None,
        rules: Optional[Sequence[AllowRule]] = None,
        client: Optional[ClientSource] = None,
        token: Optional[str] = None,
        api_url: Optional[str] = None,
        scope: Optional[ScopeSource] = None,
        ttl_seconds: float = 60.0,
        body: bool = True,
        require_scope: bool = False,
        refusal: str = "respond",
        transport: Optional[Any] = None,
        on_inject: Optional[InjectHook] = None,
        on_refuse: Optional[RefuseHook] = None,
    ) -> None:
        self._rewriter = Rewriter(
            hx=self._hx,
            rules=build_rules(allow, rules),
            scan_body=body,
            require_scope=require_scope,
            scope_source=scope,
            on_inject=on_inject,
            refusal=refusal,
            on_refuse=on_refuse,
        )
        self._resolver = Resolver(
            client=client, token=token, api_url=api_url, ttl_seconds=ttl_seconds
        )
        self._inner = transport or getattr(self._hx, self._default_transport)()


class SyncCore(_Core):
    """The synchronous half. Mixed in *before* the client's ``BaseTransport``."""

    _default_transport = "HTTPTransport"

    def handle_request(self, request: Any) -> Any:
        if not self._rewriter.needs_work(request):
            return self._inner.handle_request(request)
        scope = self._rewriter.scope()
        try:
            self._rewriter.guard(scope)
            values = self._resolver.values(scope)
            prepared = self._rewriter.rewrite(request, scope, values)
        except SeekritSubstitutionError as error:
            refused = self._rewriter.refuse(error)
            if refused is None:
                raise
            return refused
        return self._inner.handle_request(prepared)

    def close(self) -> None:
        self._inner.close()


class AsyncCore(_Core):
    """The asynchronous half. Mixed in *before* the client's ``AsyncBaseTransport``."""

    _default_transport = "AsyncHTTPTransport"

    async def handle_async_request(self, request: Any) -> Any:
        if not self._rewriter.needs_work(request):
            return await self._inner.handle_async_request(request)
        scope = self._rewriter.scope()
        try:
            self._rewriter.guard(scope)
            values = await self._resolver.avalues(scope)
            prepared = self._rewriter.rewrite(request, scope, values)
        except SeekritSubstitutionError as error:
            refused = self._rewriter.refuse(error)
            if refused is None:
                raise
            return refused
        return await self._inner.handle_async_request(prepared)

    async def aclose(self) -> None:
        await self._inner.aclose()
