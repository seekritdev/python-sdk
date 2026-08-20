"""LangChain middleware: per-request credentials, and per-tool credential scoping.

    from langchain.agents import create_agent
    from seekrit.langchain import SeekritCredentials

    agent = create_agent(
        model=model,
        tools=[refund, search],
        context_schema=Context,
        middleware=[
            SeekritCredentials(
                scope=lambda ctx: {"tenants": ctx.tenant},
                tools={"refund": ["STRIPE_SECRET_KEY"]},
            ),
        ],
    )

Two things this does that a process wrapper cannot.

**Per-request credentials without a per-tenant model cache.** The obvious
implementation of "different key per tenant" is to build a ``ChatOpenAI`` inside
``wrap_model_call`` for each tenant — but that model constructs its HTTP client
in ``validate_environment``, so per-tenant instances mean a per-tenant object
cache and a per-tenant connection pool. Instead this middleware sets an ambient
:class:`~seekrit._scope.Scope` for the duration of the call and lets
:class:`seekrit.transport.SeekritTransport` resolve against it. One model
instance, credentials that vary per request, nothing to invalidate.

**Per-tool credential scoping.** ``tools={"refund": ["STRIPE_SECRET_KEY"]}``
means the ``refund`` tool may substitute the Stripe key and nothing else, and
every other tool may substitute nothing — so a prompt-injected ``search`` call
cannot reach a payment credential. When ``tools`` is given it is exhaustive: a
tool not named there is allowed no secrets at all.

For that narrowing to be a real boundary the transport must fail closed when the
scope is missing, so pair this with ``require_scope=True``::

    transport = SeekritTransport(allow={...}, require_scope=True)

Requires ``langchain >= 1.0``: ``pip install 'seekrit[langchain]'``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError as exc:  # pragma: no cover - exercised by the extras install
    raise ImportError(
        "seekrit.langchain requires langchain >= 1.0. Install it with: "
        "pip install 'seekrit[langchain]'"
    ) from exc

from ._scope import Scope, use_scope

#: ``(context) -> {group_slug: env_slug} | None`` — read off ``runtime.context``.
ScopeFn = Callable[[Any], Optional[Mapping[str, str]]]


class SeekritCredentials(AgentMiddleware):
    """Scope seekrit resolves and injections to one model call or one tool call.

    Args:
        scope: called with ``runtime.context`` to produce the
            ``{group_slug: env_slug}`` overrides for this request, or ``None``
            for the token's own environment. Omit for a single-tenant agent.
        model: narrow the allowlist during model calls to these secret names.
            Omit to leave the transport's static rules in charge.
        tools: per-tool allowlists, ``{"refund": ["STRIPE_SECRET_KEY"]}``. When
            given it is exhaustive — an unlisted tool may inject nothing.
    """

    def __init__(
        self,
        *,
        scope: Optional[ScopeFn] = None,
        model: Optional[Sequence[str]] = None,
        tools: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        super().__init__()
        self._scope = scope
        self._model_allow = tuple(model) if model is not None else None
        self._tool_allow = {name: tuple(v) for name, v in tools.items()} if tools else None

    # -- scope construction --------------------------------------------------

    def _overrides(self, runtime: Any) -> Optional[Mapping[str, str]]:
        if self._scope is None:
            return None
        return self._scope(getattr(runtime, "context", None))

    def _model_scope(self, request: Any) -> Scope:
        return Scope(
            overrides=self._overrides(getattr(request, "runtime", None)),
            allow=self._model_allow,
            label="model",
        )

    def _tool_scope(self, request: Any) -> Scope:
        call = getattr(request, "tool_call", None) or {}
        name = call.get("name", "") if isinstance(call, Mapping) else ""
        allow: Optional[Sequence[str]] = None
        if self._tool_allow is not None:
            # Exhaustive by design: an unlisted tool gets an empty allowlist.
            allow = self._tool_allow.get(name, ())
        return Scope(
            overrides=self._overrides(getattr(request, "runtime", None)),
            allow=allow,
            label=f"tool:{name}" if name else "tool",
        )

    # -- hooks ---------------------------------------------------------------

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        with use_scope(self._model_scope(request)):
            return handler(request)

    async def awrap_model_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        with use_scope(self._model_scope(request)):
            return await handler(request)

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        with use_scope(self._tool_scope(request)):
            return handler(request)

    async def awrap_tool_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        with use_scope(self._tool_scope(request)):
            return await handler(request)
