# Seekrit in a Cloudflare Python Worker

This example uses the SDK in this checkout through `[tool.uv.sources]`.
Install current [uv](https://docs.astral.sh/uv/getting-started/installation/)
(0.12.3 or newer) and Node.js 22 or newer, then, from this directory:

```sh
npm install
uv run pywrangler dev
```

Put `SEEKRIT_TOKEN=<your service token>` in a gitignored `.dev.vars` before
starting the Worker. Create the token for your application's environment in
Seekrit; it needs permission to resolve secrets and the matching key grants.
The response confirms resolution without disclosing a secret name or value.
Replace that response with your application logic using the `secrets` mapping.

For production, deploy the example and store the token using the interactive
secret prompt (with Cloudflare authentication already configured):

```sh
uv run pywrangler deploy
uv run pywrangler secret put SEEKRIT_TOKEN
```

The Worker returns 503 until the token is set. For a standalone project, remove
`[tool.uv.sources]` and use a published Seekrit SDK release that includes
`seekrit.cloudflare`. The runtime adapter is new in this source change; the
previous 0.10.0 release does not include it.

The client uses native asynchronous `workers.fetch`, aborts timed-out requests,
refuses redirects, and decrypts inside the Worker with the Pyodide build of
`cryptography` installed by Pywrangler. It does not populate `os.environ` or
cache plaintext. Each request resolves again, so rotations are read without
redeploying the application. Do not put decrypted values in KV, R2, Durable
Object storage, shared state, logs, or responses.

See the [integration guide](https://seekrit.dev/docs/guides/cloudflare-python-workers)
for FastAPI, environment separation, and the alternative of syncing secrets
into Cloudflare bindings.
