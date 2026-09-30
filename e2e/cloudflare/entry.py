"""Runs inside workerd, against a host fixture server holding only test vectors."""

from seekrit import SeekritApiError, SeekritCryptoError, SeekritError
from seekrit.cloudflare import AsyncClient
from vectors import VECTORS
from workers import Response, WorkerEntrypoint


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        def client(path="ok", **kwargs):
            return AsyncClient(
                token=VECTORS["token"],
                api_url=f"{self.env.FIXTURE_URL}/{path}",
                **kwargs,
            )

        values = await client(overrides={"shared": "dev&with=other:prod"}).resolve()
        assert values == VECTORS["expectedManagedValues"]
        assert await client().get("SHARED") == "from-app"
        assert await client().get("MISSING", "fallback") == "fallback"
        raw = await client(interpolate=False).resolve()
        assert raw["REFERENCING"] == "url=${DATABASE_URL};shared=${SHARED}"
        passed = ["golden vectors", "encoded overrides", "get", "literal references"]

        for path, error_type in [
            ("unauthorized", SeekritApiError),
            ("tampered", SeekritCryptoError),
            ("malformed", SeekritError),
            ("redirect", SeekritError),
            ("slow-body", SeekritError),
        ]:
            try:
                await client(
                    path, timeout=0.05 if path == "slow-body" else 10
                ).resolve()
            except error_type as exc:
                if path == "unauthorized":
                    assert (exc.status, exc.code) == (401, "unauthorized")
                if path == "slow-body":
                    assert str(exc) == "resolve request timed out"
            else:
                raise AssertionError(f"{path} did not fail closed")
            passed.append(path)

        # A failure must not poison future resolves or return cached plaintext.
        assert await client().resolve() == VECTORS["expectedManagedValues"]
        return Response.from_json({"passed": passed})
