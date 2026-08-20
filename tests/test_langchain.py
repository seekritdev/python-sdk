"""The LangChain middleware, run through a real agent.

Not a mock of the middleware protocol: this builds an actual ``create_agent``
graph with a fake chat model that emits two tool calls, and asserts what the
ambient scope looks like *inside* the model call and *inside* each tool. That is
the only way to know the hooks fire, that the async hooks fire on ``ainvoke``,
and that per-tool narrowing reaches the code that would make the HTTP call.

Skipped when the ``langchain`` extra is not installed (``pip install
'seekrit[langchain]'``); runnable with either ``pytest`` or ``python -m unittest``.
"""

import asyncio
import unittest
from dataclasses import dataclass

try:
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from seekrit._scope import current_scope
    from seekrit.langchain import SeekritCredentials

    HAVE_LANGCHAIN = True
except ImportError:  # pragma: no cover - depends on the install
    HAVE_LANGCHAIN = False

OBSERVED = []


def _snapshot(where):
    """Record the scope visible at this point, in the shape a transport reads."""
    scope = current_scope()
    OBSERVED.append(
        (
            where,
            None
            if scope is None
            else (dict(scope.overrides or {}), tuple(scope.allow or ()), scope.label),
        )
    )


if HAVE_LANGCHAIN:

    @dataclass
    class Ctx:
        tenant: str

    @tool
    def refund(charge_id: str) -> str:
        """Refund a charge."""
        _snapshot("tool:refund")
        return "refunded"

    @tool
    def search(query: str) -> str:
        """Search for something."""
        _snapshot("tool:search")
        return "found"

    class RecordingModel(GenericFakeChatModel):
        """A fake model that reports the scope in effect when it is called."""

        def bind_tools(self, tools, **kwargs):
            return self  # the fake emits tool calls directly; binding is a no-op

        def _generate(self, *args, **kwargs):
            _snapshot("model")
            return super()._generate(*args, **kwargs)

    def build_agent():
        messages = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "refund", "args": {"charge_id": "ch_1"}, "id": "c1"},
                        {"name": "search", "args": {"query": "x"}, "id": "c2"},
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        return create_agent(
            model=RecordingModel(messages=messages),
            tools=[refund, search],
            context_schema=Ctx,
            middleware=[
                SeekritCredentials(
                    scope=lambda ctx: {"tenants": ctx.tenant},
                    model=["OPENAI_API_KEY"],
                    tools={"refund": ["STRIPE_SECRET_KEY"]},
                )
            ],
        )


@unittest.skipUnless(HAVE_LANGCHAIN, "langchain extra not installed")
class MiddlewareTests(unittest.TestCase):
    def setUp(self):
        OBSERVED.clear()

    def _run_sync(self, tenant):
        build_agent().invoke(
            {"messages": [{"role": "user", "content": "refund ch_1"}]},
            context=Ctx(tenant=tenant),
        )
        return dict(OBSERVED)

    def _run_async(self, tenant):
        agent = build_agent()
        asyncio.run(
            agent.ainvoke(
                {"messages": [{"role": "user", "content": "refund ch_1"}]},
                context=Ctx(tenant=tenant),
            )
        )
        return dict(OBSERVED)

    def test_the_model_call_sees_the_tenant_and_the_model_allowlist(self):
        seen = self._run_sync("northwind")
        self.assertEqual(seen["model"], ({"tenants": "northwind"}, ("OPENAI_API_KEY",), "model"))

    def test_a_listed_tool_sees_only_its_own_secret(self):
        seen = self._run_sync("northwind")
        self.assertEqual(
            seen["tool:refund"],
            ({"tenants": "northwind"}, ("STRIPE_SECRET_KEY",), "tool:refund"),
        )

    def test_an_unlisted_tool_may_inject_nothing(self):
        seen = self._run_sync("northwind")
        self.assertEqual(seen["tool:search"], ({"tenants": "northwind"}, (), "tool:search"))

    def test_the_async_hooks_scope_the_same_way(self):
        seen = self._run_async("lumen")
        self.assertEqual(seen["model"], ({"tenants": "lumen"}, ("OPENAI_API_KEY",), "model"))
        self.assertEqual(
            seen["tool:refund"], ({"tenants": "lumen"}, ("STRIPE_SECRET_KEY",), "tool:refund")
        )
        self.assertEqual(seen["tool:search"], ({"tenants": "lumen"}, (), "tool:search"))

    def test_the_scope_does_not_leak_past_the_run(self):
        self._run_sync("northwind")
        self.assertIsNone(current_scope())

    def test_without_tools_the_static_rules_stay_in_charge(self):
        middleware = SeekritCredentials(scope=lambda ctx: {"tenants": ctx.tenant})
        agent = create_agent(
            model=RecordingModel(messages=iter([AIMessage(content="done")])),
            tools=[],
            context_schema=Ctx,
            middleware=[middleware],
        )
        agent.invoke(
            {"messages": [{"role": "user", "content": "hi"}]}, context=Ctx(tenant="northwind")
        )
        seen = dict(OBSERVED)
        # allow is None ⇒ no narrowing, so the transport's own rules apply.
        self.assertEqual(seen["model"], ({"tenants": "northwind"}, (), "model"))

    def test_without_a_scope_function_no_overrides_are_produced(self):
        middleware = SeekritCredentials(model=["OPENAI_API_KEY"])
        agent = create_agent(
            model=RecordingModel(messages=iter([AIMessage(content="done")])),
            tools=[],
            middleware=[middleware],
        )
        agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
        seen = dict(OBSERVED)
        self.assertEqual(seen["model"], ({}, ("OPENAI_API_KEY",), "model"))


if __name__ == "__main__":
    unittest.main()
