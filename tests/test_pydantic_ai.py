"""The Pydantic AI toolset, run through a real agent.

Not a mock of the toolset protocol: this builds an actual ``Agent`` with
``TestModel`` (which calls every tool once) and asserts the ambient scope visible
*inside* each tool — then, in the last case, that a tool's own HTTP call really
comes out with the right credential substituted and a refusal for one it may not
have.

Skipped when the ``pydantic-ai`` extra is not installed (``pip install
'seekrit[pydantic-ai]'``); runnable with either ``pytest`` or
``python -m unittest``.
"""

import asyncio
import unittest
from dataclasses import dataclass

try:
    import httpx
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.toolsets import FunctionToolset

    from seekrit._scope import Scope, current_scope, use_scope
    from seekrit.pydantic_ai import SeekritToolset
    from seekrit.transport import AsyncSeekritTransport, SeekritTransport

    HAVE_PYDANTIC_AI = True
except ImportError:  # pragma: no cover - depends on the install
    HAVE_PYDANTIC_AI = False

OBSERVED = []
KEYS = {"OPENAI_API_KEY": "sk-live-abc", "STRIPE_SECRET_KEY": "sk_test_stripe"}


def _snapshot(where):
    """Record the scope visible here, in the shape a transport reads."""
    scope = current_scope()
    OBSERVED.append(
        (
            where,
            None
            if scope is None
            else (dict(scope.overrides or {}), tuple(scope.allow or ()), scope.label),
        )
    )


if HAVE_PYDANTIC_AI:

    @dataclass
    class Deps:
        tenant: str

    async def refund(ctx: RunContext[Deps], charge_id: str) -> str:
        """Refund a charge."""
        _snapshot("refund")
        return "refunded"

    async def search(ctx: RunContext[Deps], query: str) -> str:
        """Search for something."""
        _snapshot("search")
        return "found"

    def build_agent(**kwargs):
        return Agent(
            TestModel(),
            deps_type=Deps,
            toolsets=[SeekritToolset(FunctionToolset([refund, search]), **kwargs)],
        )


@unittest.skipUnless(HAVE_PYDANTIC_AI, "pydantic-ai extra not installed")
class ToolsetScopeTests(unittest.TestCase):
    def setUp(self):
        OBSERVED.clear()

    def test_a_listed_tool_sees_only_its_own_secret(self):
        agent = build_agent(
            scope=lambda deps: {"tenants": deps.tenant},
            tools={"refund": ["STRIPE_SECRET_KEY"]},
        )
        agent.run_sync("go", deps=Deps(tenant="northwind"))
        seen = dict(OBSERVED)
        self.assertEqual(
            seen["refund"], ({"tenants": "northwind"}, ("STRIPE_SECRET_KEY",), "tool:refund")
        )

    def test_an_unlisted_tool_may_inject_nothing(self):
        agent = build_agent(
            scope=lambda deps: {"tenants": deps.tenant},
            tools={"refund": ["STRIPE_SECRET_KEY"]},
        )
        agent.run_sync("go", deps=Deps(tenant="northwind"))
        seen = dict(OBSERVED)
        self.assertEqual(seen["search"], ({"tenants": "northwind"}, (), "tool:search"))

    def test_without_a_scope_deriver_the_surrounding_run_scope_is_kept(self):
        # Wrapping the run and wrapping the tools must compose, not fight: the
        # run establishes the tenant, the toolset narrows what each tool may use.
        agent = build_agent(tools={"refund": ["STRIPE_SECRET_KEY"]})
        with use_scope(Scope(overrides={"tenants": "lumen"}, label="run")):
            agent.run_sync("go", deps=Deps(tenant="ignored"))
        seen = dict(OBSERVED)
        self.assertEqual(
            seen["refund"], ({"tenants": "lumen"}, ("STRIPE_SECRET_KEY",), "tool:refund")
        )
        self.assertEqual(seen["search"], ({"tenants": "lumen"}, (), "tool:search"))

    def test_without_tools_the_surrounding_allowlist_is_kept(self):
        agent = build_agent()
        with use_scope(Scope(overrides={"tenants": "acme"}, allow=("OPENAI_API_KEY",))):
            agent.run_sync("go", deps=Deps(tenant="x"))
        seen = dict(OBSERVED)
        self.assertEqual(
            seen["refund"], ({"tenants": "acme"}, ("OPENAI_API_KEY",), "tool:refund")
        )

    def test_the_async_path_scopes_the_same_way(self):
        agent = build_agent(
            scope=lambda deps: {"tenants": deps.tenant},
            tools={"refund": ["STRIPE_SECRET_KEY"]},
        )
        asyncio.run(agent.run("go", deps=Deps(tenant="async-tenant")))
        seen = dict(OBSERVED)
        self.assertEqual(seen["refund"][0], {"tenants": "async-tenant"})
        self.assertEqual(seen["refund"][1], ("STRIPE_SECRET_KEY",))

    def test_the_scope_does_not_leak_past_the_run(self):
        agent = build_agent(scope=lambda deps: {"tenants": deps.tenant})
        agent.run_sync("go", deps=Deps(tenant="northwind"))
        self.assertIsNone(current_scope())


@unittest.skipUnless(HAVE_PYDANTIC_AI, "pydantic-ai extra not installed")
class ToolCredentialTests(unittest.TestCase):
    """The point of all this: a tool's own HTTP call gets exactly one key."""

    def setUp(self):
        OBSERVED.clear()
        self.wire = []

    def _transport(self, **kwargs):
        def handler(request):
            self.wire.append(
                {"url": str(request.url), "auth": request.headers.get("authorization")}
            )
            return httpx.Response(200, json={"ok": True})

        return SeekritTransport(
            allow={
                "api.stripe.com": ["STRIPE_SECRET_KEY"],
                "api.openai.com": ["OPENAI_API_KEY"],
            },
            client=lambda overrides: _FakeClient(overrides),
            transport=httpx.MockTransport(handler),
            require_scope=True,
            **kwargs,
        )

    def test_a_tool_injects_its_own_secret_and_is_refused_another(self):
        results = {}
        transport = self._transport()

        async def charge(ctx: RunContext[Deps], amount: int) -> str:
            """Charge a card."""
            with httpx.Client(transport=transport) as client:
                allowed = client.post(
                    "https://api.stripe.com/v1/refunds",
                    headers={"authorization": "Bearer {{seekrit:STRIPE_SECRET_KEY}}"},
                )
                # The host allows the model key, but this tool does not.
                refused = client.post(
                    "https://api.openai.com/v1/responses",
                    headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
                )
            results["allowed"] = allowed.status_code
            results["refused"] = refused.status_code
            results["refused_secret"] = refused.headers.get("x-seekrit-secret")
            return "charged"

        agent = Agent(
            TestModel(),
            deps_type=Deps,
            toolsets=[
                SeekritToolset(
                    FunctionToolset([charge]),
                    scope=lambda deps: {"tenants": deps.tenant},
                    tools={"charge": ["STRIPE_SECRET_KEY"]},
                )
            ],
        )
        agent.run_sync("go", deps=Deps(tenant="northwind"))

        self.assertEqual(results["allowed"], 200)
        self.assertEqual(self.wire[0]["auth"], "Bearer sk_test_stripe-northwind")
        self.assertEqual(results["refused"], 403)
        self.assertEqual(results["refused_secret"], "OPENAI_API_KEY")
        self.assertEqual(len(self.wire), 1, "the refused call never reached an upstream")

    def test_outside_a_run_the_same_transport_refuses(self):
        # require_scope=True is what makes the narrowing a boundary rather than a
        # hint: with no scope in effect there is nothing to narrow, so it refuses.
        transport = self._transport()
        with httpx.Client(transport=transport) as client:
            response = client.post(
                "https://api.stripe.com/v1/refunds",
                headers={"authorization": "Bearer {{seekrit:STRIPE_SECRET_KEY}}"},
            )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.headers["x-seekrit-refusal"], "scope_required")


class _FakeClient:
    """Resolves per-tenant values, so the wire shows which tenant was used."""

    def __init__(self, overrides):
        self.overrides = dict(overrides or {})

    def resolve(self):
        suffix = self.overrides.get("tenants", "default")
        return {name: f"{value}-{suffix}" for name, value in KEYS.items()}


@unittest.skipUnless(HAVE_PYDANTIC_AI, "pydantic-ai extra not installed")
class AsyncToolCredentialTests(unittest.TestCase):
    def test_an_async_tool_injects_through_the_async_transport(self):
        wire = []

        def handler(request):
            wire.append(request.headers.get("authorization"))
            return httpx.Response(200, json={"ok": True})

        transport = AsyncSeekritTransport(
            allow={"api.stripe.com": ["STRIPE_SECRET_KEY"]},
            client=lambda overrides: _FakeClient(overrides),
            transport=httpx.MockTransport(handler),
            require_scope=True,
        )

        async def refund_charge(ctx: RunContext[Deps], charge_id: str) -> str:
            """Refund a charge."""
            async with httpx.AsyncClient(transport=transport) as client:
                await client.post(
                    "https://api.stripe.com/v1/refunds",
                    headers={"authorization": "Bearer {{seekrit:STRIPE_SECRET_KEY}}"},
                )
            return "refunded"

        agent = Agent(
            TestModel(),
            deps_type=Deps,
            toolsets=[
                SeekritToolset(
                    FunctionToolset([refund_charge]),
                    scope=lambda deps: {"tenants": deps.tenant},
                    tools={"refund_charge": ["STRIPE_SECRET_KEY"]},
                )
            ],
        )
        asyncio.run(agent.run("go", deps=Deps(tenant="lumen")))
        self.assertEqual(wire, ["Bearer sk_test_stripe-lumen"])


if __name__ == "__main__":
    unittest.main()
