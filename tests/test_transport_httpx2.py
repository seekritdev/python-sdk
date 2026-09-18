"""The httpx2 transports: the same behaviour as the httpx ones, and parity with them.

``httpx2`` is a separate distribution rather than a newer ``httpx``, so its
transport base class is unrelated and a client that takes one rejects the other.
That is the only difference, and these tests are here to keep it the only one:
the behaviour cases run the httpx2 binding end to end through a real
``httpx2.Client``, and :class:`ParityTests` fails if the two modules' public API
or constructor signatures drift apart.

Skipped when the ``httpx2`` extra is not installed (``pip install
'seekrit[httpx2]'``); runnable with either ``pytest`` or ``python -m unittest``.
"""

import asyncio
import inspect
import unittest

try:
    import httpx2

    from seekrit._policy import AllowRule
    from seekrit._scope import Scope, use_scope
    from seekrit.errors import SeekritSubstitutionError
    from seekrit.transport_httpx2 import AsyncSeekritTransport, SeekritTransport

    HAVE_HTTPX2 = True
except ImportError:  # pragma: no cover - depends on the install
    HAVE_HTTPX2 = False

try:
    import seekrit.transport as httpx_transport
    import seekrit.transport_httpx2 as httpx2_transport

    HAVE_BOTH = True
except ImportError:  # pragma: no cover - depends on the install
    HAVE_BOTH = False

KEYS = {"TYPESAFE_API_KEY": "ts-live-abc", "STRIPE_SECRET_KEY": "sk_test_stripe"}


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
        return httpx2.Response(200, json={"ok": True})

    return seen, httpx2.MockTransport(handler)


@unittest.skipUnless(HAVE_HTTPX2, "httpx2 extra not installed")
class SubstitutionTests(unittest.TestCase):
    def _client(self, **kwargs):
        seen, mock = recorder()
        kwargs.setdefault("client", FakeClient())
        return seen, httpx2.Client(transport=SeekritTransport(transport=mock, **kwargs))

    def test_substitutes_a_header(self):
        injected = []
        seen, client = self._client(
            allow={"api.typesafe.ai": ["TYPESAFE_API_KEY"]}, on_inject=injected.append
        )
        response = client.post(
            "https://api.typesafe.ai/v1/systemone",
            headers={"authorization": "Bearer {{seekrit:TYPESAFE_API_KEY}}"},
            json={"state": "x"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen["headers"]["authorization"], "Bearer ts-live-abc")
        self.assertEqual(
            injected,
            [
                {
                    "host": "api.typesafe.ai",
                    "method": "POST",
                    "path": "/v1/systemone",
                    "names": ["TYPESAFE_API_KEY"],
                    "label": "",
                }
            ],
        )

    def test_a_request_without_a_placeholder_does_not_resolve(self):
        resolver = FakeClient()
        seen, client = self._client(
            allow={"api.typesafe.ai": ["TYPESAFE_API_KEY"]}, client=resolver
        )
        client.get("https://api.typesafe.ai/v1/models", headers={"authorization": "Bearer plain"})
        self.assertEqual(seen["headers"]["authorization"], "Bearer plain")
        self.assertEqual(resolver.calls, 0)

    def test_substitutes_the_body_and_recomputes_content_length(self):
        seen, client = self._client(allow={"hooks.slack.com": ["TYPESAFE_API_KEY"]})
        client.post(
            "https://hooks.slack.com/services/x",
            content='{"k":"{{seekrit:TYPESAFE_API_KEY}}"}',
        )
        self.assertEqual(seen["body"], '{"k":"ts-live-abc"}')
        self.assertEqual(seen["headers"]["content-length"], str(len(seen["body"])))

    def test_substitutes_the_query_string(self):
        seen, client = self._client(allow={"api.typesafe.ai": ["TYPESAFE_API_KEY"]})
        client.get("https://api.typesafe.ai/v1/x?key={{seekrit:TYPESAFE_API_KEY}}")
        self.assertEqual(seen["url"], "https://api.typesafe.ai/v1/x?key=ts-live-abc")

    def test_body_scanning_can_be_turned_off(self):
        seen, client = self._client(
            allow={"hooks.slack.com": ["TYPESAFE_API_KEY"]}, body=False
        )
        client.post(
            "https://hooks.slack.com/services/x",
            headers={"x-k": "{{seekrit:TYPESAFE_API_KEY}}"},
            content='{"k":"{{seekrit:TYPESAFE_API_KEY}}"}',
        )
        self.assertEqual(seen["headers"]["x-k"], "ts-live-abc")
        self.assertEqual(seen["body"], '{"k":"{{seekrit:TYPESAFE_API_KEY}}"}')


@unittest.skipUnless(HAVE_HTTPX2, "httpx2 extra not installed")
class RefusalTests(unittest.TestCase):
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
        client = httpx2.Client(
            transport=SeekritTransport(
                allow={"api.typesafe.ai": ["TYPESAFE_API_KEY"]},
                client=FakeClient(),
                transport=mock,
            )
        )
        response = client.post(
            "https://api.typesafe.ai/v1/systemone",
            headers={"authorization": "Bearer {{seekrit:STRIPE_SECRET_KEY}}"},
        )
        self._assert_refused(response, "denied", "STRIPE_SECRET_KEY")
        self.assertEqual(seen, {}, "the request must never reach the upstream")

    def test_the_same_name_toward_an_unlisted_host_is_denied(self):
        _, mock = recorder()
        client = httpx2.Client(
            transport=SeekritTransport(
                allow={"api.typesafe.ai": ["TYPESAFE_API_KEY"]},
                client=FakeClient(),
                transport=mock,
            )
        )
        response = client.post(
            "https://evil.example.com/v1/x",
            headers={"authorization": "Bearer {{seekrit:TYPESAFE_API_KEY}}"},
        )
        self._assert_refused(response, "denied", "TYPESAFE_API_KEY")

    def test_an_allowed_name_that_did_not_resolve_is_refused(self):
        _, mock = recorder()
        client = httpx2.Client(
            transport=SeekritTransport(
                allow={"api.typesafe.ai": ["ABSENT_KEY"]},
                client=FakeClient(),
                transport=mock,
            )
        )
        response = client.get(
            "https://api.typesafe.ai/v1/x", headers={"x-k": "{{seekrit:ABSENT_KEY}}"}
        )
        self._assert_refused(response, "unresolved", "ABSENT_KEY")

    def test_refusal_raise_gives_the_typed_error_instead(self):
        refused = []
        _, mock = recorder()
        client = httpx2.Client(
            transport=SeekritTransport(
                allow={"api.typesafe.ai": ["TYPESAFE_API_KEY"]},
                client=FakeClient(),
                transport=mock,
                refusal="raise",
                on_refuse=refused.append,
            )
        )
        with self.assertRaises(SeekritSubstitutionError) as caught:
            client.post(
                "https://api.typesafe.ai/v1/systemone",
                headers={"authorization": "Bearer {{seekrit:STRIPE_SECRET_KEY}}"},
            )
        self.assertEqual(caught.exception.code, "denied")
        self.assertEqual(caught.exception.secret_name, "STRIPE_SECRET_KEY")
        self.assertNotIn("sk_test_stripe", str(caught.exception))
        self.assertEqual(len(refused), 1, "on_refuse fires in both modes")

    def test_method_and_path_constraints_are_enforced(self):
        rules = [
            AllowRule(
                host="api.typesafe.ai",
                methods=("POST",),
                paths=("/v1/systemone",),
                allow=("TYPESAFE_API_KEY",),
            )
        ]
        _, mock = recorder()
        client = httpx2.Client(
            transport=SeekritTransport(rules=rules, client=FakeClient(), transport=mock)
        )
        allowed = client.post(
            "https://api.typesafe.ai/v1/systemone",
            headers={"authorization": "Bearer {{seekrit:TYPESAFE_API_KEY}}"},
        )
        self.assertEqual(allowed.status_code, 200)
        denied = client.post(
            "https://api.typesafe.ai/v1/models",
            headers={"authorization": "Bearer {{seekrit:TYPESAFE_API_KEY}}"},
        )
        self._assert_refused(denied, "denied", "TYPESAFE_API_KEY")


@unittest.skipUnless(HAVE_HTTPX2, "httpx2 extra not installed")
class ScopeAndCachingTests(unittest.TestCase):
    def _client(self, **kwargs):
        seen, mock = recorder()
        kwargs.setdefault("client", FakeClient())
        return seen, httpx2.Client(transport=SeekritTransport(transport=mock, **kwargs))

    def test_an_ambient_scope_narrows_the_allowlist(self):
        seen, client = self._client(
            allow={"api.typesafe.ai": ["TYPESAFE_API_KEY", "STRIPE_SECRET_KEY"]}
        )
        with use_scope(Scope(allow=("TYPESAFE_API_KEY",), label="answer")):
            ok = client.post(
                "https://api.typesafe.ai/v1/systemone",
                headers={"authorization": "Bearer {{seekrit:TYPESAFE_API_KEY}}"},
            )
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(seen["headers"]["authorization"], "Bearer ts-live-abc")

            narrowed = client.post(
                "https://api.typesafe.ai/v1/systemone",
                headers={"authorization": "Bearer {{seekrit:STRIPE_SECRET_KEY}}"},
            )
            self.assertEqual(narrowed.status_code, 403)
            self.assertEqual(narrowed.headers["x-seekrit-refusal"], "denied")

    def test_require_scope_fails_closed_with_no_scope_in_effect(self):
        _, client = self._client(
            allow={"api.typesafe.ai": ["TYPESAFE_API_KEY"]}, require_scope=True
        )
        response = client.post(
            "https://api.typesafe.ai/v1/systemone",
            headers={"authorization": "Bearer {{seekrit:TYPESAFE_API_KEY}}"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.headers["x-seekrit-refusal"], "scope_required")

    def test_resolves_once_within_the_ttl(self):
        resolver = FakeClient()
        _, client = self._client(
            allow={"api.typesafe.ai": ["TYPESAFE_API_KEY"]}, client=resolver
        )
        for _ in range(3):
            client.post(
                "https://api.typesafe.ai/v1/systemone",
                headers={"authorization": "Bearer {{seekrit:TYPESAFE_API_KEY}}"},
            )
        self.assertEqual(resolver.calls, 1)

    def test_ttl_zero_resolves_every_time(self):
        resolver = FakeClient()
        _, client = self._client(
            allow={"api.typesafe.ai": ["TYPESAFE_API_KEY"]},
            client=resolver,
            ttl_seconds=0,
        )
        for _ in range(3):
            client.post(
                "https://api.typesafe.ai/v1/systemone",
                headers={"authorization": "Bearer {{seekrit:TYPESAFE_API_KEY}}"},
            )
        self.assertEqual(resolver.calls, 3)


@unittest.skipUnless(HAVE_HTTPX2, "httpx2 extra not installed")
class AsyncTransportTests(unittest.TestCase):
    def test_substitutes_on_the_async_path(self):
        seen, mock = recorder()

        async def run():
            transport = AsyncSeekritTransport(
                allow={"api.typesafe.ai": ["TYPESAFE_API_KEY"]},
                client=FakeClient(),
                transport=mock,
            )
            async with httpx2.AsyncClient(transport=transport) as client:
                return await client.post(
                    "https://api.typesafe.ai/v1/systemone",
                    headers={"authorization": "Bearer {{seekrit:TYPESAFE_API_KEY}}"},
                )

        response = asyncio.run(run())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen["headers"]["authorization"], "Bearer ts-live-abc")

    def test_denies_on_the_async_path(self):
        seen, mock = recorder()

        async def run():
            transport = AsyncSeekritTransport(
                allow={"api.typesafe.ai": ["TYPESAFE_API_KEY"]},
                client=FakeClient(),
                transport=mock,
            )
            async with httpx2.AsyncClient(transport=transport) as client:
                return await client.post(
                    "https://api.typesafe.ai/v1/systemone",
                    headers={"authorization": "Bearer {{seekrit:STRIPE_SECRET_KEY}}"},
                )

        response = asyncio.run(run())
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.headers["x-seekrit-refusal"], "denied")
        self.assertEqual(seen, {}, "the request must never reach the upstream")


@unittest.skipUnless(HAVE_BOTH, "both httpx and httpx2 extras are needed")
class ParityTests(unittest.TestCase):
    """The two bindings must stay interchangeable.

    An option added to one module and not the other is the failure this catches:
    a caller who migrates a client from ``httpx`` to ``httpx2`` would find the
    keyword silently missing, and the shim would run wider than they wrote.
    """

    PAIRS = (("SeekritTransport", "sync"), ("AsyncSeekritTransport", "async"))

    def test_the_two_modules_export_the_same_names(self):
        self.assertEqual(httpx_transport.__all__, httpx2_transport.__all__)

    def test_the_constructors_take_identical_arguments(self):
        for name, _ in self.PAIRS:
            with self.subTest(cls=name):
                self.assertEqual(
                    inspect.signature(getattr(httpx_transport, name).__init__),
                    inspect.signature(getattr(httpx2_transport, name).__init__),
                )

    def test_each_class_is_bound_to_its_own_client(self):
        import httpx

        for name, _ in self.PAIRS:
            with self.subTest(cls=name):
                self.assertIs(getattr(httpx_transport, name)._hx, httpx)
                self.assertIs(getattr(httpx2_transport, name)._hx, httpx2)

    def test_each_class_subclasses_its_own_clients_base_transport(self):
        import httpx

        self.assertTrue(issubclass(httpx_transport.SeekritTransport, httpx.BaseTransport))
        self.assertTrue(
            issubclass(httpx_transport.AsyncSeekritTransport, httpx.AsyncBaseTransport)
        )
        self.assertTrue(
            issubclass(httpx2_transport.SeekritTransport, httpx2.BaseTransport)
        )
        self.assertTrue(
            issubclass(
                httpx2_transport.AsyncSeekritTransport, httpx2.AsyncBaseTransport
            )
        )
        # The point of two modules: neither base class accepts the other's.
        self.assertFalse(
            issubclass(httpx2_transport.SeekritTransport, httpx.BaseTransport)
        )

    def test_a_refusal_is_byte_for_byte_the_same_in_both(self):
        error = SeekritSubstitutionError("denied", "TYPESAFE_API_KEY")
        one = httpx_transport.refusal_response(error)
        two = httpx2_transport.refusal_response(error)
        self.assertEqual(one.status_code, two.status_code)
        self.assertEqual(one.text, two.text)
        self.assertEqual(
            one.headers["x-seekrit-refusal"], two.headers["x-seekrit-refusal"]
        )
        self.assertEqual(one.headers["x-seekrit-secret"], two.headers["x-seekrit-secret"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
