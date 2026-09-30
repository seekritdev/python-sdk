"""Native-fetch contract and crypto parity for the Python Workers client.

The runtime-specific boundary is stubbed here; e2e/cloudflare/run.py runs the
same client in a real local Worker through Pywrangler.
"""

import asyncio
import copy
import inspect
import json
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import seekrit
from seekrit.cloudflare import AsyncClient

VECTORS = json.loads(
    (Path(__file__).parent.parent / "testdata/vectors.json").read_text()
)


class Controller:
    def __init__(self):
        self.signal = object()
        self.aborted = False

    def abort(self):
        self.aborted = True


def response(status=200, body=None):
    raw = json.dumps(VECTORS["resolve"] if body is None else body).encode()
    return types.SimpleNamespace(status=status, bytes=AsyncMock(return_value=raw))


class CloudflareTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.controller = Controller()
        self.fetch = AsyncMock(return_value=response())
        self.runtime = patch.dict(
            "sys.modules",
            {
                "workers": types.SimpleNamespace(fetch=self.fetch),
                "js": types.SimpleNamespace(
                    AbortController=types.SimpleNamespace(new=lambda: self.controller)
                ),
            },
        )
        self.runtime.start()
        self.addCleanup(self.runtime.stop)
        self.client = AsyncClient(VECTORS["token"])

    async def test_native_fetch_decrypts_the_complete_golden_response(self):
        self.assertEqual(await self.client.resolve(), VECTORS["expectedManagedValues"])
        self.fetch.assert_awaited_once_with(
            "https://api.seekrit.dev/v1/resolve",
            method="GET",
            headers={
                "authorization": "Bearer " + VECTORS["token"],
                "accept": "application/json",
            },
            redirect="manual",
            signal=self.controller.signal,
        )
        self.assertTrue(self.controller.aborted)

    async def test_custom_url_and_sorted_overrides_are_encoded(self):
        client = AsyncClient(
            VECTORS["token"],
            api_url="https://api.test///",
            overrides={"z": "dev&with=other:prod", "a": "staging"},
        )
        await client.resolve()
        self.assertEqual(
            self.fetch.call_args.args[0],
            "https://api.test/v1/resolve?with=a%3Astaging&with=z%3Adev%26with%3Dother%3Aprod",
        )
        # The synchronous client uses the same wire conventions.
        sync = seekrit.Client(
            VECTORS["token"],
            api_url="https://api.test///",
            overrides={"z": "dev&with=other:prod", "a": "staging"},
        )
        self.assertEqual(sync._resolve_url(), client._resolve_url())

    async def test_get_and_default_fetch_fresh_each_time(self):
        self.assertEqual(await self.client.get("SHARED"), "from-app")
        self.assertEqual(await self.client.get("MISSING", "fallback"), "fallback")
        self.assertEqual(self.fetch.await_count, 2)

    async def test_interpolation_can_be_disabled(self):
        client = AsyncClient(VECTORS["token"], interpolate=False)
        values = await client.resolve()
        self.assertEqual(values["REFERENCING"], "url=${DATABASE_URL};shared=${SHARED}")

    async def test_tampered_later_layer_raises_instead_of_returning_partial_values(
        self,
    ):
        body = copy.deepcopy(VECTORS["resolve"])
        body["layers"][-1]["secrets"][0]["name"] += "_SWAPPED"
        self.fetch.return_value = response(body=body)
        with self.assertRaises(seekrit.SeekritCryptoError):
            await self.client.resolve()

    async def test_api_refusal_then_recovery_does_not_cache_a_failure(self):
        self.fetch.side_effect = [
            response(401, {"error": {"code": "unauthorized", "message": "revoked"}}),
            response(),
        ]
        with self.assertRaises(seekrit.SeekritApiError) as raised:
            await self.client.resolve()
        self.assertEqual(
            (raised.exception.status, raised.exception.code), (401, "unauthorized")
        )
        self.assertEqual(await self.client.resolve(), VECTORS["expectedManagedValues"])

    async def test_non_json_api_error_keeps_http_status(self):
        self.fetch.return_value = types.SimpleNamespace(
            status=503, bytes=AsyncMock(return_value=b"unavailable")
        )
        with self.assertRaises(seekrit.SeekritApiError) as raised:
            await self.client.resolve()
        self.assertEqual(
            (raised.exception.status, raised.exception.code), (503, "internal")
        )

    async def test_redirect_is_refused_without_reading_or_following_it(self):
        redirect = response(302, {})
        self.fetch.return_value = redirect
        with self.assertRaises(seekrit.SeekritApiError) as raised:
            await self.client.resolve()
        self.assertEqual(raised.exception.status, 302)
        self.assertEqual(self.fetch.await_count, 1)
        redirect.bytes.assert_not_awaited()
        self.assertTrue(self.controller.aborted)

    async def test_malformed_success_is_an_sdk_error(self):
        for body in [b"not JSON", b"[]", b"{}", b'{"layers": null}']:
            with self.subTest(body=body):
                self.fetch.return_value = types.SimpleNamespace(
                    status=200, bytes=AsyncMock(return_value=body)
                )
                with self.assertRaises(seekrit.SeekritError):
                    await self.client.resolve()

    async def test_network_failure_aborts_without_echoing_the_network_error(self):
        self.fetch.side_effect = OSError("sensitive network diagnostic")
        with self.assertRaisesRegex(seekrit.SeekritError, "^resolve request failed$"):
            await self.client.resolve()
        self.assertTrue(self.controller.aborted)

    async def test_timeout_covers_response_body_and_aborts_fetch(self):
        async def stalled_body():
            await asyncio.sleep(60)

        self.fetch.return_value = types.SimpleNamespace(status=200, bytes=stalled_body)
        client = AsyncClient(VECTORS["token"], timeout=0.01)
        with self.assertRaisesRegex(seekrit.SeekritError, "timed out"):
            await client.resolve()
        self.assertTrue(self.controller.aborted)

    async def test_caller_cancellation_propagates_and_aborts_fetch(self):
        waiting = asyncio.Event()

        async def stalled_fetch(*args, **kwargs):
            waiting.set()
            await asyncio.sleep(60)

        self.fetch.side_effect = stalled_fetch
        task = asyncio.create_task(self.client.resolve())
        await waiting.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.controller.aborted)

    def test_configuration_matches_the_regular_client(self):
        self.assertEqual(
            inspect.signature(AsyncClient), inspect.signature(seekrit.Client)
        )
        with self.assertRaises(seekrit.SeekritCryptoError):
            AsyncClient("invalid-token")


if __name__ == "__main__":
    unittest.main()
