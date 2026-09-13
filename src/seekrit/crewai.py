"""CrewAI adapter: a provider key that exists only inside one HTTP call, and
per-agent / per-tool credential scoping across a crew.

    from crewai import LLM, Agent, Crew, Task
    from seekrit.crewai import SeekritCredentials, SeekritInterceptor

    llm = LLM(
        model="openai/gpt-5.6-terra",
        api_key="{{seekrit:OPENAI_API_KEY}}",
        interceptor=SeekritInterceptor(
            allow={"api.openai.com": ["OPENAI_API_KEY"]},
            require_scope=True,
        ),
    )

    with SeekritCredentials(
        agents={"Research Specialist": ["OPENAI_API_KEY"]},
        tools={"refund": ["OPENAI_API_KEY", "STRIPE_SECRET_KEY"]},
    ):
        crew.kickoff()

Two seams, and they are meant to be used together.

**The interceptor is the credential seam.** ``crewai.llms.hooks.BaseInterceptor``
is CrewAI's own contract for touching the raw ``httpx.Request`` a provider is
about to send; CrewAI installs it as that client's transport. So the key is
resolved and decrypted here, substituted into one outbound request, and exists
nowhere else — not in ``.env``, not in ``os.environ``, not in model context,
not in a trace exporter. It is per-``LLM``-instance, which is what makes it
different from ``litellm.client_session``: two agents in one crew can hold two
different credentials.

**The hooks are the scoping seam.** A crew is several agents sharing one
process, so without narrowing every agent effectively has every credential the
token can resolve. ``agents=`` and ``tools=`` bound that per model call and per
tool call, so a prompt-injected research agent cannot reach a payment key.

**Pair them with ``require_scope=True``.** Narrowing is only a boundary if a
lost scope fails closed; without it, a scope that does not reach the transport
silently falls back to the full allowlist.

Which providers honour ``interceptor``
--------------------------------------

Verified against crewai 1.15: the native **OpenAI** provider and everything
built on it (``openrouter``, ``deepseek``, ``ollama``, ``hosted_vllm``,
``cerebras``, ``dashscope``) and the native **Anthropic** provider. Gemini,
Azure and Bedrock reject an interceptor loudly at construction, which is fine —
you find out immediately. The trap is the **LiteLLM fallback**: it declares an
``interceptor`` field and never reads it, so a model string that does not route
to a native provider silently sends your placeholder to the upstream instead of
a key. :func:`ensure_intercepted` is one line that turns that into an error;
:ref:`the LiteLLM path <litellm>` below is the supported alternative.

.. _litellm:

If you must stay on LiteLLM, its only seam is a module global, which is
process-wide and therefore cannot vary per agent::

    import httpx, litellm
    from seekrit.transport import AsyncSeekritTransport, SeekritTransport

    allow = {"api.openai.com": ["OPENAI_API_KEY"]}
    litellm.client_session = httpx.Client(transport=SeekritTransport(allow=allow))
    litellm.aclient_session = httpx.AsyncClient(transport=AsyncSeekritTransport(allow=allow))

**What this is not.** Everything here runs in the crew's own process, so it is
not the trust boundary ``seekrit proxy`` is: code in this process can read a
resolved value or replace the interceptor. It is the rung above environment
variables and below the proxy. Reach for the proxy when the code holding the
placeholder is code you do not trust — and note that only the proxy can bound
*operations* (methods and paths) for a tool that already has its credential.

Requires ``crewai``: ``pip install 'seekrit[crewai]'``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

try:
    from crewai.hooks import (
        InterceptionPoint,
        register_after_llm_call_hook,
        register_after_tool_call_hook,
        register_before_llm_call_hook,
        register_before_tool_call_hook,
        register_hook,
        unregister_after_llm_call_hook,
        unregister_after_tool_call_hook,
        unregister_before_llm_call_hook,
        unregister_before_tool_call_hook,
        unregister_hook,
    )
    from crewai.llms.hooks import BaseInterceptor
except ImportError as exc:  # pragma: no cover - exercised by the extras install
    raise ImportError(
        "seekrit.crewai requires crewai >= 1.15 (for crewai.llms.hooks and "
        "crewai.hooks). Install it with: pip install 'seekrit[crewai]'"
    ) from exc

from ._policy import AllowRule
from ._scope import Scope, current_scope, set_scope, use_scope
from .errors import SeekritError, SeekritSubstitutionError
from .transport import (
    ClientSource,
    InjectHook,
    RefuseHook,
    ScopeSource,
    _build_rules,
    _Resolver,
    _Rewriter,
)

__all__ = [
    "AllowRule",
    "Scope",
    "SeekritCredentials",
    "SeekritInterceptor",
    "SeekritRefusal",
    "current_scope",
    "ensure_intercepted",
    "use_scope",
]

#: ``(hook context) -> {group_slug: env_slug} | None``. The context is CrewAI's
#: own ``LLMCallHookContext`` or ``ToolCallHookContext``, so ``.agent``,
#: ``.task``, ``.crew`` — and ``.tool_name`` on a tool call — are all in scope.
ScopeFn = Callable[[Any], Optional[Mapping[str, str]]]


class SeekritRefusal(BaseException):
    """A refusal that a provider SDK's retry loop cannot swallow.

    Deriving from :class:`BaseException` rather than :class:`Exception` is
    deliberate, and it is the whole reason this class exists. An interceptor can
    only *raise*: unlike :class:`seekrit.transport.SeekritTransport`, which
    answers the same 403 the proxy answers, it never gets to produce a response,
    because CrewAI's transport calls ``on_outbound`` and then sends whatever it
    returns. And every provider SDK treats an exception from its HTTP layer as a
    transient network fault — ``openai`` catches ``Exception``, wraps it in
    ``APIConnectionError`` and **retries**, so a denied placeholder would
    surface as "Connection error" after six attempts with the real reason buried
    on ``__cause__``. That is the exact failure this SDK already learned once
    with LangChain. A ``BaseException`` walks straight out past ``except
    Exception`` in the provider SDK and in CrewAI's executor, so a
    misconfiguration fails once, immediately, and says what it was.

    Pass ``terminal=False`` to :class:`SeekritInterceptor` for ordinary
    :class:`~seekrit.errors.SeekritSubstitutionError` semantics instead.

    Attributes:
        error: the underlying :class:`~seekrit.errors.SeekritSubstitutionError`.
            Its ``code`` and ``secret_name`` are safe to log — never a value.
    """

    def __init__(self, error: SeekritSubstitutionError) -> None:
        super().__init__(str(error))
        self.error = error

    @property
    def code(self) -> str:
        return self.error.code

    @property
    def secret_name(self) -> str:
        return self.error.secret_name


class SeekritInterceptor(BaseInterceptor):  # type: ignore[misc, type-arg]
    """Substitute ``{{seekrit:NAME}}`` in a CrewAI provider's outbound request.

    Arguments mirror :class:`seekrit.transport.SeekritTransport`, minus the ones
    that make no sense for an interceptor (there is no inner transport to send
    with, and no way to answer a response of our own).

    Args:
        allow: shorthand allowlist, ``{"api.openai.com": ["OPENAI_API_KEY"]}``.
        rules: full rules, host by host — the wire shape of a signed ``ap1.``
            bundle's ``rules``, so a verified bundle can be passed straight in.
        client: where resolved values come from. A :class:`seekrit.Client` (or
            anything with ``resolve()``) serves scopes that do not re-scope; pass
            a **callable** of the overrides when a scope carries group
            overrides, since one client is bound to one set of them.
        token: ``skt_...`` service token. Defaults to ``$SEEKRIT_TOKEN``.
        api_url: API base URL. Defaults to ``$SEEKRIT_API_URL``.
        scope: called once per request instead of reading the ambient scope that
            :class:`SeekritCredentials` installs.
        ttl_seconds: how long a resolved set may be reused per scope
            (default 60). Values live in memory only. Note that a crew kickoff
            is one long process: a TTL of ``0`` re-resolves every call, which is
            the only way a rotation mid-run is picked up.
        body: also scan the request body (default ``True``).
        require_scope: refuse a placeholder-carrying request when no scope is in
            effect. Set it whenever you use :class:`SeekritCredentials`, so a
            scope that failed to reach the transport fails closed instead of
            widening back to the full allowlist.
        terminal: raise :class:`SeekritRefusal` on a denied placeholder
            (default), so the provider SDK's retry loop cannot swallow it. Set
            ``False`` for a plain
            :class:`~seekrit.errors.SeekritSubstitutionError`.
        on_inject: called after a successful substitution with ``host``,
            ``method``, ``path``, ``names`` and ``label``. Names only, never
            values.
        on_refuse: called on every refusal, before it is raised.
    """

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
        terminal: bool = True,
        on_inject: Optional[InjectHook] = None,
        on_refuse: Optional[RefuseHook] = None,
    ) -> None:
        self._rewriter = _Rewriter(
            rules=_build_rules(allow, rules),
            scan_body=body,
            require_scope=require_scope,
            scope_source=scope,
            on_inject=on_inject,
            refusal="raise",
            on_refuse=on_refuse,
        )
        self._resolver = _Resolver(
            client=client, token=token, api_url=api_url, ttl_seconds=ttl_seconds
        )
        self._terminal = terminal

    def _refuse(self, error: SeekritSubstitutionError) -> BaseException:
        self._rewriter.refuse(error)  # fires on_refuse; returns None in "raise" mode
        return SeekritRefusal(error) if self._terminal else error

    def on_outbound(self, message: Any) -> Any:
        """Substitute into the request CrewAI is about to send."""
        if not self._rewriter.needs_work(message):
            return message
        scope = self._rewriter.scope()
        try:
            self._rewriter.guard(scope)
            values = self._resolver.values(scope)
            return self._rewriter.rewrite(message, scope, values)
        except SeekritSubstitutionError as error:
            raise self._refuse(error) from None

    def on_inbound(self, message: Any) -> Any:
        """Pass the response through untouched.

        Nothing to do: the credential went out, and a response is not a place a
        placeholder can appear. Required by the ``BaseInterceptor`` contract.
        """
        return message

    async def aon_outbound(self, message: Any) -> Any:
        """The async twin. The resolve itself runs off the event loop."""
        if not self._rewriter.needs_work(message):
            return message
        scope = self._rewriter.scope()
        try:
            self._rewriter.guard(scope)
            values = await self._resolver.avalues(scope)
            return self._rewriter.rewrite(message, scope, values)
        except SeekritSubstitutionError as error:
            raise self._refuse(error) from None

    async def aon_inbound(self, message: Any) -> Any:
        return message


def _narrow(
    first: Optional[Sequence[str]], second: Optional[Sequence[str]]
) -> Optional[Tuple[str, ...]]:
    """Intersect two allowlists. ``None`` means "no opinion", never "everything"."""
    if first is None:
        return tuple(second) if second is not None else None
    if second is None:
        return tuple(first)
    keep = set(second)
    return tuple(name for name in first if name in keep)


def _role(agent: Any) -> str:
    role = getattr(agent, "role", None)
    return role if isinstance(role, str) else ""


class SeekritCredentials:
    """Scope what each agent, and each tool call, may inject.

    Use it as a context manager around ``kickoff()``, or call
    :meth:`install` / :meth:`uninstall` yourself::

        with SeekritCredentials(tools={"refund": ["STRIPE_SECRET_KEY"]}):
            crew.kickoff()

    Args:
        scope: called with the hook context to produce this call's
            ``{group_slug: env_slug}`` overrides, or ``None`` for the token's own
            environment. The context carries ``.agent``, ``.task`` and ``.crew``
            (and ``.tool_name`` on a tool call), so a per-tenant crew can read
            the tenant off its inputs. Omit for a single-tenant crew.
        model: allowlist for model calls, when every agent gets the same one.
        agents: per-agent-role allowlists,
            ``{"Research Specialist": ["OPENAI_API_KEY"]}``. When given it is
            **exhaustive** — an agent not named here may inject nothing, so
            adding an agent to the crew fails closed rather than quietly
            handing it every credential. Takes precedence over ``model``.
        tools: per-tool allowlists, ``{"refund": ["STRIPE_SECRET_KEY"]}``. Also
            exhaustive when given. Intersected with the agent's own list when
            both are given, so delegation cannot be used to widen: the narrowest
            of the two wins.

    Three things about CrewAI's hooks are worth knowing, because they shape what
    this can promise.

    **The registry is process-global.** CrewAI registers hooks on module-level
    lists, not on a ``Crew``, so an installed ``SeekritCredentials`` applies to
    every crew running in this process. Two crews needing different scoping
    means two processes, or ``scope=`` doing the discriminating.

    **Register before kickoff.** The executors snapshot the LLM hook list when
    they are built, so installing mid-run may not take effect for agents already
    running.

    **A hook that raises is swallowed.** CrewAI catches any exception from a
    hook and continues (fail-open, to survive a buggy user hook), so this class
    never tries to enforce by raising. Enforcement lives in the transport, where
    a narrowed scope and ``require_scope=True`` make a missing or wrong scope
    refuse the request.
    """

    def __init__(
        self,
        *,
        scope: Optional[ScopeFn] = None,
        model: Optional[Sequence[str]] = None,
        agents: Optional[Mapping[str, Sequence[str]]] = None,
        tools: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self._scope = scope
        self._model_allow = tuple(model) if model is not None else None
        self._agent_allow: Optional[Dict[str, Tuple[str, ...]]] = (
            {name: tuple(v) for name, v in agents.items()} if agents is not None else None
        )
        self._tool_allow: Optional[Dict[str, Tuple[str, ...]]] = (
            {name: tuple(v) for name, v in tools.items()} if tools is not None else None
        )
        self._installed = False

    # -- scope construction --------------------------------------------------

    def _overrides(self, context: Any) -> Optional[Mapping[str, str]]:
        if self._scope is None:
            return None
        return self._scope(context)

    def _agent_scope(self, context: Any) -> Optional[Tuple[str, ...]]:
        if self._agent_allow is None:
            return None
        # Exhaustive by design: an unlisted agent gets an empty allowlist.
        return self._agent_allow.get(_role(getattr(context, "agent", None)), ())

    def _model_scope(self, context: Any) -> Scope:
        allow = self._agent_scope(context)
        if allow is None:
            allow = self._model_allow
        role = _role(getattr(context, "agent", None))
        return Scope(
            overrides=self._overrides(context),
            allow=allow,
            label="agent:" + role if role else "model",
        )

    def _tool_scope(self, context: Any) -> Scope:
        name = getattr(context, "tool_name", "") or ""
        tool_allow: Optional[Sequence[str]] = None
        if self._tool_allow is not None:
            tool_allow = self._tool_allow.get(name, ())
        return Scope(
            overrides=self._overrides(context),
            allow=_narrow(tool_allow, self._agent_scope(context)),
            label="tool:" + name if name else "tool",
        )

    # -- hooks ---------------------------------------------------------------
    #
    # Every hook returns None on purpose. A non-None return from an after hook
    # replaces the model response or the tool result, and False from
    # before_tool_call blocks the call — this class decides what may be
    # injected, and must not edit what an agent said or did.

    def _before_llm_call(self, context: Any) -> None:
        set_scope(self._model_scope(context))

    def _after_llm_call(self, context: Any) -> None:
        set_scope(None)

    def _before_tool_call(self, context: Any) -> None:
        set_scope(self._tool_scope(context))

    def _after_tool_call(self, context: Any) -> None:
        set_scope(None)

    def _on_execution_end(self, context: Any) -> None:
        """Clear at the end of a run, successful or failed.

        The before/after pairs above are separate dispatches, so a call that
        raises between them strands its scope. Nothing is *widened* by that (the
        next call sets its own, and a stranded scope is a narrowed one), but a
        crew that ends mid-flight should not leave one installed for whatever
        this thread does next. CrewAI dispatches ``execution_end`` exactly once
        per kickoff, both on success and on failure.
        """
        set_scope(None)

    # -- lifecycle -----------------------------------------------------------

    def install(self) -> "SeekritCredentials":
        """Register the hooks. Idempotent; returns self."""
        if self._installed:
            return self
        register_before_llm_call_hook(self._before_llm_call)
        register_after_llm_call_hook(self._after_llm_call)
        register_before_tool_call_hook(self._before_tool_call)
        register_after_tool_call_hook(self._after_tool_call)
        register_hook(InterceptionPoint.EXECUTION_END, self._on_execution_end)
        self._installed = True
        return self

    def uninstall(self) -> None:
        """Unregister the hooks and clear any scope left in effect."""
        if not self._installed:
            return
        unregister_before_llm_call_hook(self._before_llm_call)
        unregister_after_llm_call_hook(self._after_llm_call)
        unregister_before_tool_call_hook(self._before_tool_call)
        unregister_after_tool_call_hook(self._after_tool_call)
        unregister_hook(InterceptionPoint.EXECUTION_END, self._on_execution_end)
        self._installed = False
        set_scope(None)

    def __enter__(self) -> "SeekritCredentials":
        return self.install()

    def __exit__(self, *exc: Any) -> None:
        self.uninstall()


#: LLM classes whose provider does not read ``interceptor``. The LiteLLM
#: fallback declares the field for interface compatibility and ignores it —
#: the one case that fails silently rather than at construction.
_IGNORES_INTERCEPTOR = ("litellm",)


def ensure_intercepted(llm: Any) -> Any:
    """Fail now if this ``LLM`` would ignore its interceptor. Returns the LLM.

        llm = ensure_intercepted(
            LLM(model="openai/gpt-5.6-terra", api_key="{{seekrit:OPENAI_API_KEY}}",
                interceptor=SeekritInterceptor(allow={...})),
        )

    ``LLM.__new__`` routes to a native provider or falls back to LiteLLM based
    on the model string, and the LiteLLM path accepts ``interceptor`` without
    ever using it. Unchecked, that ships a crew that sends the literal
    ``{{seekrit:OPENAI_API_KEY}}`` to the upstream and fails with a provider
    authentication error naming nothing useful. Gemini, Azure and Bedrock need
    no check here: they reject an interceptor when the ``LLM`` is constructed.

    Raises:
        SeekritError: if no interceptor is set, or the LLM routed to LiteLLM.
    """
    if getattr(llm, "interceptor", None) is None:
        raise SeekritError(
            "this LLM has no interceptor: pass interceptor=SeekritInterceptor(...) "
            "to LLM(...), or the placeholder is sent to the provider as-is"
        )
    llm_type = getattr(llm, "llm_type", None)
    if llm_type in _IGNORES_INTERCEPTOR:
        raise SeekritError(
            "LLM(model=" + repr(getattr(llm, "model", "")) + ") routed to the LiteLLM "
            "fallback, which accepts an interceptor and never calls it. Name a native "
            "provider (an 'openai/...' or 'anthropic/...' model, or provider=...), or "
            "set litellm.client_session to a SeekritTransport instead — see "
            "seekrit.crewai"
        )
    return llm

