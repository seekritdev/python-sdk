"""The CrewAI adapter, run through real CrewAI machinery.

Not a mock of the interceptor protocol: these build an actual ``crewai.LLM``
against a stub upstream and assert what the *server* received in its
``Authorization`` header, and they drive the scoping hooks through CrewAI's own
dispatcher rather than calling them directly. That is the only way to know the
interceptor is installed where CrewAI says it is, that the hooks fire, and that
a narrowed scope reaches the code that would make the HTTP call.

Skipped when the ``crewai`` extra is not installed (``pip install
'seekrit[crewai]'``); runnable with either ``pytest`` or ``python -m unittest``.
"""

import asyncio
import json
import threading
import uuid
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    import httpx
    from crewai import LLM
    from crewai.hooks import InterceptionPoint, clear_all_global_hooks
    from crewai.hooks.contexts import ExecutionEndContext
    from crewai.hooks.dispatch import clear_all as clear_all_hooks, dispatch
    from crewai.hooks.tool_hooks import (
        ToolCallHookContext,
        run_after_tool_call_hooks,
        run_before_tool_call_hooks,
    )

    from seekrit._scope import current_scope, set_scope
    from seekrit.crewai import (
        SeekritCredentials,
        SeekritInterceptor,
        SeekritRefusal,
        ensure_intercepted,
    )
    from seekrit.errors import SeekritError, SeekritSubstitutionError

    HAVE_CREWAI = True
except ImportError:  # pragma: no cover - depends on the install
    HAVE_CREWAI = False

SECRETS = {"OPENAI_API_KEY": "sk-live-openai", "STRIPE_SECRET_KEY": "sk-live-stripe"}


class FakeClient:
    """A resolve source that never leaves the process."""

    def __init__(self, values=None):
        self.values = dict(values if values is not None else SECRETS)
        self.calls = 0

    def resolve(self):
        self.calls += 1
        return dict(self.values)


class FakeAgent:
    """Structurally an agent: a role for the adapter, an id for CrewAI's events."""

    def __init__(self, role):
        self.role = role
        self.id = uuid.uuid4()


COMPLETION = {
    "id": "chatcmpl-stub",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
        length = int(self.headers.get("content-length", 0))
        self.rfile.read(length)
        self.server.seen.append(dict(self.headers))
        body = json.dumps(COMPLETION).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the test output clean
        pass


class StubUpstream:
    """A one-endpoint OpenAI stand-in that records the headers it was sent."""

    def __enter__(self):
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.server.seen = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def url(self):
        host, port = self.server.server_address[:2]
        return "http://{}:{}/v1".format(host, port)

    @property
    def authorizations(self):
        return [h.get("authorization") for h in self.server.seen]


def placeholder_request(name="OPENAI_API_KEY", host="api.openai.com"):
    return httpx.Request(
        "POST",
        "https://{}/v1/chat/completions".format(host),
        headers={"authorization": "Bearer {{seekrit:" + name + "}}"},
        json={"model": "gpt-4o"},
    )


@unittest.skipUnless(HAVE_CREWAI, "crewai not installed")
class InterceptorTest(unittest.TestCase):
    """The credential seam, exercised directly on the contract CrewAI calls."""

    def tearDown(self):
        set_scope(None)

    def test_substitutes_into_the_outbound_request(self):
        injected = []
        interceptor = SeekritInterceptor(
            allow={"api.openai.com": ["OPENAI_API_KEY"]},
            client=FakeClient(),
            on_inject=lambda event: injected.append(event),
        )
        out = interceptor.on_outbound(placeholder_request())
        self.assertEqual(out.headers["authorization"], "Bearer sk-live-openai")
        self.assertEqual(injected[0]["names"], ["OPENAI_API_KEY"])
        self.assertEqual(injected[0]["host"], "api.openai.com")

    def test_leaves_a_request_without_a_placeholder_alone(self):
        client = FakeClient()
        interceptor = SeekritInterceptor(
            allow={"api.openai.com": ["OPENAI_API_KEY"]}, client=client
        )
        request = httpx.Request("GET", "https://api.openai.com/v1/models")
        self.assertIs(interceptor.on_outbound(request), request)
        self.assertEqual(client.calls, 0)  # no resolve for a request we do not touch

    def test_inbound_is_untouched(self):
        interceptor = SeekritInterceptor(
            allow={"api.openai.com": ["OPENAI_API_KEY"]}, client=FakeClient()
        )
        response = httpx.Response(200)
        self.assertIs(interceptor.on_inbound(response), response)

    def test_a_denied_name_refuses_terminally(self):
        """The refusal must escape ``except Exception``, or it becomes a retry."""
        refused = []
        interceptor = SeekritInterceptor(
            allow={"api.openai.com": ["OPENAI_API_KEY"]},
            client=FakeClient(),
            on_refuse=lambda error: refused.append(error.code),
        )
        with self.assertRaises(SeekritRefusal) as caught:
            interceptor.on_outbound(placeholder_request("STRIPE_SECRET_KEY"))
        self.assertEqual(caught.exception.code, "denied")
        self.assertEqual(caught.exception.secret_name, "STRIPE_SECRET_KEY")
        self.assertEqual(refused, ["denied"])
        # The point of the class: a provider SDK's `except Exception: retry`
        # cannot see it.
        self.assertNotIsInstance(caught.exception, Exception)

    def test_terminal_false_raises_an_ordinary_exception(self):
        interceptor = SeekritInterceptor(
            allow={"api.openai.com": ["OPENAI_API_KEY"]},
            client=FakeClient(),
            terminal=False,
        )
        with self.assertRaises(SeekritSubstitutionError) as caught:
            interceptor.on_outbound(placeholder_request("STRIPE_SECRET_KEY"))
        self.assertEqual(caught.exception.code, "denied")

    def test_require_scope_refuses_when_no_scope_is_in_effect(self):
        interceptor = SeekritInterceptor(
            allow={"api.openai.com": ["OPENAI_API_KEY"]},
            client=FakeClient(),
            require_scope=True,
        )
        with self.assertRaises(SeekritRefusal) as caught:
            interceptor.on_outbound(placeholder_request())
        self.assertEqual(caught.exception.code, "scope_required")

    def test_a_scope_narrows_the_allowlist(self):
        from seekrit._scope import Scope

        interceptor = SeekritInterceptor(
            allow={"api.openai.com": ["OPENAI_API_KEY", "STRIPE_SECRET_KEY"]},
            client=FakeClient(),
            require_scope=True,
        )
        set_scope(Scope(allow=("OPENAI_API_KEY",), label="tool:search"))
        self.assertEqual(
            interceptor.on_outbound(placeholder_request()).headers["authorization"],
            "Bearer sk-live-openai",
        )
        with self.assertRaises(SeekritRefusal):
            interceptor.on_outbound(placeholder_request("STRIPE_SECRET_KEY"))

    def test_async_outbound(self):
        interceptor = SeekritInterceptor(
            allow={"api.openai.com": ["OPENAI_API_KEY"]}, client=FakeClient()
        )
        out = asyncio.run(interceptor.aon_outbound(placeholder_request()))
        self.assertEqual(out.headers["authorization"], "Bearer sk-live-openai")
        self.assertIs(
            asyncio.run(interceptor.aon_inbound(httpx.Response(204))).status_code, 204
        )


@unittest.skipUnless(HAVE_CREWAI, "crewai not installed")
class LiveLLMTest(unittest.TestCase):
    """A real ``crewai.LLM``, a real provider client, a stub upstream.

    This is what proves the seam: nothing here reaches into CrewAI's internals,
    so if CrewAI stopped honouring ``interceptor=`` these would fail.
    """

    def tearDown(self):
        clear_all_global_hooks()
        clear_all_hooks()
        set_scope(None)

    def build(self, upstream, **kwargs):
        interceptor = SeekritInterceptor(
            allow={"127.0.0.1": ["OPENAI_API_KEY"]},
            client=FakeClient(),
            **kwargs,
        )
        return ensure_intercepted(
            LLM(
                model="openai/gpt-4o",
                api_key="{{seekrit:OPENAI_API_KEY}}",
                base_url=upstream.url,
                max_retries=0,
                interceptor=interceptor,
            )
        )

    def test_the_upstream_receives_the_resolved_key(self):
        with StubUpstream() as upstream:
            self.build(upstream).call("hello")
            self.assertEqual(upstream.authorizations, ["Bearer sk-live-openai"])

    def test_the_agents_allowlist_reaches_the_request(self):
        """A model call under ``agents=`` runs with that agent's narrowing."""
        injected = []
        with StubUpstream() as upstream:
            llm = self.build(
                upstream,
                require_scope=True,
                on_inject=lambda event: injected.append(event["label"]),
            )
            with SeekritCredentials(agents={"Researcher": ["OPENAI_API_KEY"]}):
                llm.call("hello", from_agent=FakeAgent("Researcher"))
        self.assertEqual(upstream.authorizations, ["Bearer sk-live-openai"])
        self.assertEqual(injected, ["agent:Researcher"])

    def test_an_agent_missing_from_the_allowlist_is_refused(self):
        """``agents=`` is exhaustive: a new agent fails closed, not open."""
        with StubUpstream() as upstream:
            llm = self.build(upstream, require_scope=True)
            with SeekritCredentials(agents={"Researcher": ["OPENAI_API_KEY"]}):
                with self.assertRaises(SeekritRefusal) as caught:
                    llm.call("hello", from_agent=FakeAgent("Auditor"))
            self.assertEqual(caught.exception.code, "denied")
            self.assertEqual(upstream.authorizations, [])  # never sent

    def test_no_scope_refuses_under_require_scope(self):
        """Without the hooks installed there is no scope, so nothing goes out."""
        with StubUpstream() as upstream:
            llm = self.build(upstream, require_scope=True)
            with self.assertRaises(SeekritRefusal) as caught:
                llm.call("hello")
            self.assertEqual(caught.exception.code, "scope_required")
            self.assertEqual(upstream.authorizations, [])


@unittest.skipUnless(HAVE_CREWAI, "crewai not installed")
class CredentialsHookTest(unittest.TestCase):
    """The scoping seam, driven through CrewAI's own hook dispatcher."""

    def tearDown(self):
        clear_all_global_hooks()
        clear_all_hooks()
        set_scope(None)

    def tool_context(self, name, role="Researcher"):
        return ToolCallHookContext(
            tool_name=name,
            tool_input={},
            tool=None,
            agent=FakeAgent(role),
            task=None,
            crew=None,
        )

    def scope_during_tool_call(self, credentials, name, role="Researcher"):
        """Run the real before/after dispatch and report the scope in between."""
        context = self.tool_context(name, role)
        blocked = run_before_tool_call_hooks(context)
        seen = current_scope()
        run_after_tool_call_hooks(context)
        return blocked, seen, current_scope()

    def test_tools_are_exhaustive(self):
        with SeekritCredentials(tools={"refund": ["STRIPE_SECRET_KEY"]}) as credentials:
            blocked, during, after = self.scope_during_tool_call(credentials, "refund")
            self.assertFalse(blocked)  # scoping never blocks a call
            self.assertEqual(during.allow, ("STRIPE_SECRET_KEY",))
            self.assertEqual(during.label, "tool:refund")
            self.assertIsNone(after)  # the after hook clears

            _, during, _ = self.scope_during_tool_call(credentials, "search")
            self.assertEqual(during.allow, ())  # unlisted tool: nothing at all

    def test_agent_and_tool_lists_intersect(self):
        """The narrower of the two wins, so delegation cannot widen."""
        credentials = SeekritCredentials(
            agents={"Researcher": ["OPENAI_API_KEY"]},
            tools={"refund": ["OPENAI_API_KEY", "STRIPE_SECRET_KEY"]},
        )
        with credentials:
            _, during, _ = self.scope_during_tool_call(credentials, "refund")
            self.assertEqual(during.allow, ("OPENAI_API_KEY",))

    def test_scope_derives_overrides_from_the_context(self):
        credentials = SeekritCredentials(
            scope=lambda ctx: {"tenants": ctx.agent.role.lower()},
            tools={"refund": ["STRIPE_SECRET_KEY"]},
        )
        with credentials:
            _, during, _ = self.scope_during_tool_call(credentials, "refund", "Acme")
            self.assertEqual(dict(during.overrides), {"tenants": "acme"})

    def test_execution_end_clears_a_stranded_scope(self):
        with SeekritCredentials(tools={"refund": ["STRIPE_SECRET_KEY"]}):
            run_before_tool_call_hooks(self.tool_context("refund"))
            self.assertIsNotNone(current_scope())  # after hook never ran
            dispatch(InterceptionPoint.EXECUTION_END, ExecutionEndContext())
            self.assertIsNone(current_scope())

    def test_uninstall_removes_the_hooks(self):
        """Including the LLM pair, which is registered by bound method."""
        from crewai.hooks import get_before_llm_call_hooks

        credentials = SeekritCredentials(tools={"refund": ["STRIPE_SECRET_KEY"]})
        credentials.install()
        credentials.install()  # idempotent
        self.assertEqual(len(get_before_llm_call_hooks()), 1)
        credentials.uninstall()
        self.assertEqual(get_before_llm_call_hooks(), [])
        _, during, _ = self.scope_during_tool_call(credentials, "refund")
        self.assertIsNone(during)


@unittest.skipUnless(HAVE_CREWAI, "crewai not installed")
class EnsureInterceptedTest(unittest.TestCase):
    """The guard for the one CrewAI path that fails silently."""

    def interceptor(self):
        return SeekritInterceptor(
            allow={"api.openai.com": ["OPENAI_API_KEY"]}, client=FakeClient()
        )

    def test_rejects_an_llm_with_no_interceptor(self):
        llm = LLM(model="openai/gpt-4o", api_key="{{seekrit:OPENAI_API_KEY}}")
        with self.assertRaises(SeekritError) as caught:
            ensure_intercepted(llm)
        self.assertIn("no interceptor", str(caught.exception))

    def test_the_litellm_marker_is_still_spelled_that_way(self):
        """Pin the string the guard matches on, against CrewAI itself.

        ``crewai[litellm]`` is an optional extra as of 1.15, so the fallback may
        not be installed at all — but the class is always importable, and its
        ``llm_type`` default is what ``ensure_intercepted`` keys off. If CrewAI
        renames it, this fails here rather than by quietly passing a crew that
        never calls its interceptor.
        """
        from crewai.llm import LLM as LiteLLMClass

        self.assertEqual(LiteLLMClass.model_fields["llm_type"].default, "litellm")

    def test_rejects_the_litellm_fallback(self):
        """It accepts ``interceptor`` and never calls it — the trap this catches."""
        try:
            llm = LLM(
                model="groq/llama-3.3-70b",
                api_key="{{seekrit:OPENAI_API_KEY}}",
                interceptor=self.interceptor(),
            )
        except ImportError:  # crewai[litellm] not installed, so no fallback exists
            self.skipTest("crewai[litellm] is not installed")
        self.assertEqual(getattr(llm, "llm_type", None), "litellm")
        with self.assertRaises(SeekritError) as caught:
            ensure_intercepted(llm)
        self.assertIn("LiteLLM", str(caught.exception))

    def test_accepts_a_native_provider(self):
        llm = LLM(
            model="openai/gpt-4o",
            api_key="{{seekrit:OPENAI_API_KEY}}",
            interceptor=self.interceptor(),
        )
        self.assertIs(ensure_intercepted(llm), llm)


if __name__ == "__main__":
    unittest.main()
