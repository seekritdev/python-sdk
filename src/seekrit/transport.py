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

For a client built on ``httpx2`` — a separate distribution, whose transport base
class is unrelated to this one — use :mod:`seekrit.transport_httpx2`. Same
arguments, same behaviour.

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

from typing import Any

try:
    import httpx
except ImportError as exc:  # pragma: no cover - exercised by the extras install
    raise ImportError(
        "seekrit.transport requires httpx. Install it with: pip install 'seekrit[httpx]'"
    ) from exc

from . import _shim
from ._policy import AllowRule
from ._scope import Scope, current_scope, use_scope
from ._shim import ClientSource, InjectHook, RefuseHook, ResolveSource, ScopeSource
from ._shim import Resolver as _Resolver
from ._shim import build_rules as _build_rules
from .errors import SeekritSubstitutionError

__all__ = [
    "SeekritTransport",
    "AsyncSeekritTransport",
    "AllowRule",
    "ClientSource",
    "ResolveSource",
    "Scope",
    "current_scope",
    "use_scope",
]


def refusal_response(error: SeekritSubstitutionError) -> "httpx.Response":
    """The 403 a refusal answers with. See :func:`seekrit._shim.refusal_response`."""
    return _shim.refusal_response(httpx, error)


class _Rewriter(_shim.Rewriter):
    """:class:`seekrit._shim.Rewriter` bound to ``httpx``.

    ``seekrit.crewai`` builds one of these directly: CrewAI's interceptor
    contract hands it an ``httpx.Request`` with no transport in the picture.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(hx=httpx, **kwargs)


class SeekritTransport(_shim.SyncCore, httpx.BaseTransport):
    """A synchronous ``httpx`` transport that substitutes ``{{seekrit:NAME}}``.

    Args:
        allow: shorthand allowlist, ``{"api.openai.com": ["OPENAI_API_KEY"]}``,
            permitting those names toward that host for any method or path.
        rules: full rules, host by host. This is the wire shape of a signed
            ``ap1.`` bundle's ``rules``, so a verified bundle's list can be
            passed straight in (see :meth:`AllowRule.from_dict`).
        client: where resolved values come from. A :class:`seekrit.Client` (or
            anything with ``resolve()``) serves scopes that do not re-scope; pass
            a **callable** of the overrides when ``scope`` returns ``with``
            overrides, since one client is bound to one set of them. Omit it and
            a client is built from ``token`` / ``$SEEKRIT_TOKEN`` per scope.
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

    _hx = httpx


class AsyncSeekritTransport(_shim.AsyncCore, httpx.AsyncBaseTransport):
    """The ``async`` twin of :class:`SeekritTransport`; same arguments.

    ``transport`` defaults to ``httpx.AsyncHTTPTransport()``. The resolve itself
    is synchronous (the SDK speaks ``urllib``), so it runs in the default
    executor rather than blocking the event loop.
    """

    _hx = httpx
