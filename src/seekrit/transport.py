"""``httpx`` transports that hold a placeholder instead of a credential.

    import httpx
    from openai import OpenAI
    from seekrit.transport import SeekritTransport

    client = OpenAI(
        api_key="{{seekrit:OPENAI_API_KEY}}",
        http_client=httpx.Client(
            transport=SeekritTransport(allow={"api.openai.com": ["OPENAI_API_KEY"]}),
        ),
    )

The key never exists in your source, your ``.env``, or ``os.environ``. It is
resolved and decrypted here, substituted into the outbound request, and nowhere
else — so it cannot reach model context, a tool result, or a trace exporter,
which is where credentials actually leak in an agent. That also puts it out of
reach of an environment-scraping bug in a framework you depend on: see
CVE-2025-68664, where ``langchain-core`` deserialization would read any named
environment variable back out.

Because every major Python agent toolkit bottoms out in the ``openai`` client's
``http_client=`` (LangChain's ``ChatOpenAI``, Pydantic AI's ``OpenAIProvider``,
the OpenAI Agents SDK's ``set_default_openai_client``, LlamaIndex's ``OpenAI``)
or in ``litellm.client_session``, one transport covers all of them.

**What this is not.** It runs in your process, so it is not the trust boundary
``apps/proxy`` is: code in this process can read the value or replace this
transport. It is the rung of the ladder above environment variables and below
the proxy. Reach for the proxy when the code holding the placeholder is code you
do not trust.

Only requests that *carry a placeholder* are gated. A request with no
placeholder passes straight through — this is a credential shim, not an egress
firewall, and silently blocking unrelated traffic would be a worse lie than not
blocking it.

Install with ``pip install 'seekrit[httpx]'``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import httpx
except ImportError as exc:  # pragma: no cover - exercised by the extras install
    raise ImportError(
        "seekrit.transport requires httpx. Install it with: pip install 'seekrit[httpx]'"
    ) from exc

from ._client import Client
from ._policy import AllowRule, evaluate, narrow, rules_from_allow
from ._scope import Scope, current_scope, use_scope
from ._substitute import Lookup, has_placeholder, substitute
from .errors import SeekritError, SeekritSubstitutionError

__all__ = [
    "SeekritTransport",
    "AsyncSeekritTransport",
    "AllowRule",
    "Scope",
    "current_scope",
    "use_scope",
]

#: Headers httpx must recompute when we change the body.
_RECOMPUTED = ("content-length", "transfer-encoding")

InjectHook = Callable[[Dict[str, object]], None]
RefuseHook = Callable[[SeekritSubstitutionError], None]
ScopeSource = Callable[[], Optional[Scope]]


def refusal_response(error: SeekritSubstitutionError) -> "httpx.Response":
    """The 403 the proxy answers with, so both halves fail the same way.

    A provider SDK wraps anything its HTTP layer *raises* into its own opaque
    connection error **and retries it** — a denied placeholder surfaces as
    "Connection error" after six attempts instead of naming the secret. A 403 is
    terminal in every provider SDK, and mirrors
    ``Reject::into_response`` in ``apps/proxy/src/proxy.rs`` verbatim, so
    swapping this transport for the proxy does not change error handling.
    """
    if error.code == "denied":
        body = (
            "placeholder {{seekrit:"
            + error.secret_name
            + "}} is not allowed toward this upstream"
        )
    elif error.code == "scope_required":
        body = "no scope is in effect and require_scope is set"
    else:
        body = (
            "placeholder {{seekrit:"
            + error.secret_name
            + "}} references a secret that is not available"
        )
    return httpx.Response(
        403,
        text=body,
        headers={
            # Machine-checkable, so a caller can tell our refusal from an
            # upstream 403.
            "x-seekrit-refusal": error.code,
            "x-seekrit-secret": error.secret_name,
        },
    )


def _build_rules(
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


class _Resolver:
    """Resolve and decrypt, cached per scope for ``ttl_seconds``.

    Values live in memory only. A concurrent burst may resolve more than once —
    harmless, and cheaper than a lock on every request.
    """

    def __init__(
        self,
        *,
        client: Optional[Client],
        token: Optional[str],
        api_url: Optional[str],
        ttl_seconds: float,
    ) -> None:
        self._client = client
        self._token = token
        self._api_url = api_url
        self._ttl = max(0.0, ttl_seconds)
        self._cache: Dict[str, Tuple[float, Dict[str, str]]] = {}

    def _client_for(self, scope: Optional[Scope]) -> Client:
        overrides = scope.overrides if scope else None
        if not overrides and self._client is not None:
            return self._client
        return Client(self._token, api_url=self._api_url, overrides=overrides)

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


class _Rewriter:
    """The transport-independent half: decide, substitute, rebuild the request."""

    def __init__(
        self,
        *,
        rules: List[AllowRule],
        scan_body: bool,
        require_scope: bool,
        scope_source: Optional[ScopeSource],
        on_inject: Optional[InjectHook],
        refusal: str,
        on_refuse: Optional[RefuseHook],
    ) -> None:
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

    def _body(self, request: "httpx.Request") -> Optional[str]:
        if not self._scan_body:
            return None
        try:
            raw = request.content
        except httpx.RequestNotRead:
            return None  # a streamed body is never buffered here
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None  # binary upload: nothing to scan

    def needs_work(self, request: "httpx.Request") -> bool:
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
        request: "httpx.Request",
        scope: Optional[Scope],
        values: Mapping[str, str],
    ) -> "httpx.Request":
        rules = narrow(self._rules, scope.allow) if scope and scope.allow else self._rules
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
            # Let httpx recompute the framing headers for the new length.
            headers = [(k, v) for k, v in headers if k.lower() not in _RECOMPUTED]
            return httpx.Request(
                method=request.method,
                url=new_url,
                headers=headers,
                content=new_body,
                extensions=request.extensions,
            )
        try:
            content = request.content
        except httpx.RequestNotRead:
            return httpx.Request(
                method=request.method,
                url=new_url,
                headers=headers,
                stream=request.stream,
                extensions=request.extensions,
            )
        return httpx.Request(
            method=request.method,
            url=new_url,
            headers=headers,
            content=content,
            extensions=request.extensions,
        )

    def guard(self, scope: Optional[Scope]) -> None:
        if self._require_scope and scope is None:
            raise SeekritSubstitutionError("scope_required", "")

    def refuse(self, error: SeekritSubstitutionError) -> Optional["httpx.Response"]:
        """Turn a refusal into a 403, or return ``None`` to let it propagate."""
        if self._on_refuse:
            self._on_refuse(error)
        return refusal_response(error) if self._refusal == "respond" else None


class SeekritTransport(httpx.BaseTransport):
    """A synchronous ``httpx`` transport that substitutes ``{{seekrit:NAME}}``.

    Args:
        allow: shorthand allowlist, ``{"api.openai.com": ["OPENAI_API_KEY"]}``,
            permitting those names toward that host for any method or path.
        rules: full rules, host by host. This is the wire shape of a signed
            ``ap1.`` bundle's ``rules``, so a verified bundle's list can be
            passed straight in (see :meth:`AllowRule.from_dict`).
        client: a pre-built :class:`seekrit.Client`, used when no per-request
            scope overrides are in play.
        token: ``skt_...`` service token. Defaults to ``$SEEKRIT_TOKEN``.
        api_url: API base URL. Defaults to ``$SEEKRIT_API_URL``.
        scope: called once per request instead of reading the ambient scope.
        ttl_seconds: how long a resolved set may be reused per scope
            (default 60). ``0`` resolves on every request that carries a
            placeholder — correct, and one extra round trip per model call.
        body: also scan the request body (default ``True``). A streamed or
            non-UTF-8 body is never scanned, because buffering it here would
            break streaming uploads.
        require_scope: refuse a placeholder-carrying request when no scope is in
            effect. Set this when a framework adapter narrows per tool call, so
            a lost context fails closed instead of widening the allowlist.
        refusal: how a refusal reaches the caller. ``"respond"`` (default)
            answers with the same **403** the proxy answers with and never sends
            the request; ``"raise"`` raises
            :class:`~seekrit.errors.SeekritSubstitutionError` instead. The
            default exists because a provider SDK wraps anything its HTTP layer
            raises into its own opaque connection error *and retries it* — a
            denied placeholder would surface as "Connection error" after six
            attempts instead of naming the secret. A failure to *resolve* always
            raises either way: that one is genuinely transient.
        transport: the inner transport to send with. Defaults to
            ``httpx.HTTPTransport()``.
        on_inject: called after a successful substitution with a dict of
            ``host``, ``method``, ``path``, ``names`` and ``label``. Names only
            — never values.
        on_refuse: called on every refusal, whichever way it surfaces.
    """

    def __init__(
        self,
        *,
        allow: Optional[Mapping[str, Sequence[str]]] = None,
        rules: Optional[Sequence[AllowRule]] = None,
        client: Optional[Client] = None,
        token: Optional[str] = None,
        api_url: Optional[str] = None,
        scope: Optional[ScopeSource] = None,
        ttl_seconds: float = 60.0,
        body: bool = True,
        require_scope: bool = False,
        refusal: str = "respond",
        transport: Optional[httpx.BaseTransport] = None,
        on_inject: Optional[InjectHook] = None,
        on_refuse: Optional[RefuseHook] = None,
    ) -> None:
        self._rewriter = _Rewriter(
            rules=_build_rules(allow, rules),
            scan_body=body,
            require_scope=require_scope,
            scope_source=scope,
            on_inject=on_inject,
            refusal=refusal,
            on_refuse=on_refuse,
        )
        self._resolver = _Resolver(
            client=client, token=token, api_url=api_url, ttl_seconds=ttl_seconds
        )
        self._inner = transport or httpx.HTTPTransport()

    def handle_request(self, request: "httpx.Request") -> "httpx.Response":
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


class AsyncSeekritTransport(httpx.AsyncBaseTransport):
    """The ``async`` twin of :class:`SeekritTransport`; same arguments.

    ``transport`` defaults to ``httpx.AsyncHTTPTransport()``. The resolve itself
    is synchronous (the SDK speaks ``urllib``), so it runs in the default
    executor rather than blocking the event loop.
    """

    def __init__(
        self,
        *,
        allow: Optional[Mapping[str, Sequence[str]]] = None,
        rules: Optional[Sequence[AllowRule]] = None,
        client: Optional[Client] = None,
        token: Optional[str] = None,
        api_url: Optional[str] = None,
        scope: Optional[ScopeSource] = None,
        ttl_seconds: float = 60.0,
        body: bool = True,
        require_scope: bool = False,
        refusal: str = "respond",
        transport: Optional[httpx.AsyncBaseTransport] = None,
        on_inject: Optional[InjectHook] = None,
        on_refuse: Optional[RefuseHook] = None,
    ) -> None:
        self._rewriter = _Rewriter(
            rules=_build_rules(allow, rules),
            scan_body=body,
            require_scope=require_scope,
            scope_source=scope,
            on_inject=on_inject,
            refusal=refusal,
            on_refuse=on_refuse,
        )
        self._resolver = _Resolver(
            client=client, token=token, api_url=api_url, ttl_seconds=ttl_seconds
        )
        self._inner = transport or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: "httpx.Request") -> "httpx.Response":
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
