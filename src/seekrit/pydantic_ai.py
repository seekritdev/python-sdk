"""Pydantic AI adapter: per-run credentials, and a key scoped to one tool.

    from pydantic_ai import Agent
    from pydantic_ai.toolsets import FunctionToolset
    from seekrit.pydantic_ai import SeekritToolset, use_scope

    toolset = SeekritToolset(
        FunctionToolset([refund, search]),
        scope=lambda deps: {"tenants": deps.tenant},
        tools={"refund": ["STRIPE_SECRET_KEY"]},
    )
    agent = Agent("openai:gpt-5.6-terra", deps_type=Deps, toolsets=[toolset])

    async def handle(tenant: str, prompt: str) -> str:
        # Covers the model call as well as the tools.
        with use_scope(Scope(overrides={"tenants": tenant})):
            result = await agent.run(prompt, deps=Deps(tenant=tenant))
        return result.output

Pydantic AI's seam is :class:`~pydantic_ai.toolsets.WrapperToolset`, whose
``call_tool`` wraps every tool invocation with ``ctx`` — and therefore
``ctx.deps`` — in scope. That is the right place to decide *which* credentials a
tool may inject, which is the question that matters: a provider key buys tokens,
but a tool's Stripe key does something irreversible.

**The model call is not a tool call**, so it is not covered by the toolset.
Pydantic AI builds its provider (and its HTTP client) when the agent is
constructed, so there is no per-request model hook to use. Wrap the run in
:func:`use_scope` instead, as above: the scope is a context variable, so it
covers the model call and every tool call inside that run, and the toolset
narrows further per tool on top of it.

Pair this with ``require_scope=True`` on the transport so a lost context refuses
rather than falling back to the unnarrowed allowlist::

    transport = AsyncSeekritTransport(allow={...}, require_scope=True)

Requires ``pydantic-ai``: ``pip install 'seekrit[pydantic-ai]'``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

try:
    from pydantic_ai import WrapperToolset
except ImportError as exc:  # pragma: no cover - exercised by the extras install
    raise ImportError(
        "seekrit.pydantic_ai requires pydantic-ai. Install it with: "
        "pip install 'seekrit[pydantic-ai]'"
    ) from exc

from ._scope import Scope, current_scope, use_scope

#: ``(deps) -> {group_slug: env_slug} | None`` — read off the run's ``deps``.
ScopeFn = Callable[[Any], Optional[Mapping[str, str]]]

__all__ = ["SeekritToolset", "Scope", "current_scope", "use_scope"]


class SeekritToolset(WrapperToolset):
    """Wrap a toolset so each tool call runs under its own seekrit scope.

    Args:
        wrapped: the toolset to delegate to — a ``FunctionToolset``, an MCP
            toolset, a combined one, anything implementing ``AbstractToolset``.
        scope: called with the run's ``deps`` to produce this run's
            ``{group_slug: env_slug}`` overrides. Omit it and the overrides
            already in effect (from a surrounding :func:`use_scope`) are kept, so
            wrapping the run and wrapping the tools compose rather than fight.
        tools: per-tool allowlists, ``{"refund": ["STRIPE_SECRET_KEY"]}``. When
            given it is exhaustive — a tool not named there may inject nothing at
            all, so a prompt-injected call cannot reach a credential its tool was
            never meant to have.
    """

    def __init__(
        self,
        wrapped: Any,
        *,
        scope: Optional[ScopeFn] = None,
        tools: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        super().__init__(wrapped)
        self._scope = scope
        self._tool_allow: Optional[Dict[str, tuple]] = (
            {name: tuple(v) for name, v in tools.items()} if tools else None
        )

    def _scope_for(self, name: str, ctx: Any) -> Scope:
        inherited = current_scope()
        if self._scope is not None:
            overrides = self._scope(getattr(ctx, "deps", None))
        else:
            # No deriver: keep whatever the caller established around the run.
            overrides = inherited.overrides if inherited else None

        allow: Optional[Sequence[str]] = None
        if self._tool_allow is not None:
            # Exhaustive by design: an unlisted tool gets an empty allowlist.
            allow = self._tool_allow.get(name, ())
        elif inherited is not None:
            allow = inherited.allow

        return Scope(overrides=overrides, allow=allow, label=f"tool:{name}")

    async def call_tool(
        self,
        name: str,
        tool_args: Dict[str, Any],
        ctx: Any,
        tool: Any,
    ) -> Any:
        with use_scope(self._scope_for(name, ctx)):
            return await super().call_tool(name, tool_args, ctx, tool)
