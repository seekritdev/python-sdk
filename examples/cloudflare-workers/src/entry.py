from seekrit import SeekritError
from seekrit.cloudflare import AsyncClient
from workers import Response, WorkerEntrypoint


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        if request.method != "GET":
            return Response("Method not allowed", status=405, headers={"allow": "GET"})
        try:
            secrets = await AsyncClient(
                token=getattr(self.env, "SEEKRIT_TOKEN", None)
            ).resolve()
        except SeekritError:
            # Do not log tokens or resolved values, or serve stale values on failure.
            return Response("Secret resolution unavailable", status=503)

        # Pass secrets["YOUR_API_KEY"] to your application's provider client here.
        # This setup check returns no secret names or values.
        return Response.from_json({"configured": bool(secrets)})
