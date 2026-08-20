# seekrit — Python SDK

Read-path SDK for [seekrit](https://seekrit.dev). Authenticate with a service
token, resolve your environment, and get **decrypted** secrets — the API only
ever returns ciphertext; decryption happens in your process.

> This repo is a **read-only mirror** published from seekrit's monorepo so the
> code that holds your token and decrypts plaintext is auditable. Don't commit
> here — it's overwritten on each sync. Issues and PRs welcome.

## Install

```sh
pip install seekrit
```

Requires Python 3.9+. The only dependency is [`cryptography`](https://cryptography.io).

## Usage

```python
import seekrit

client = seekrit.Client()            # token from $SEEKRIT_TOKEN
secrets = client.resolve()           # {"DATABASE_URL": "postgres://…", …}

db_url = client.get("DATABASE_URL")
api_key = client.get("API_KEY", default="")
```

Load everything into the process environment:

```python
import os, seekrit
seekrit.Client().into_env()          # existing os.environ vars win by default
print(os.environ["DATABASE_URL"])
```

### Configuration

| Argument | Env var | Default |
| --- | --- | --- |
| `token` | `SEEKRIT_TOKEN` | — (required) |
| `api_url` | `SEEKRIT_API_URL` | `https://api.seekrit.dev` |
| `overrides` | — | `{}` |
| `timeout` | — | `30.0` (seconds) |

A service token binds to a single app environment (plus its composed group
slices). To pull a different environment slice of a composed group, pass
`overrides` (the `?with=` override):

```python
seekrit.Client(overrides={"shared": "dev"}).resolve()
```

### Errors

- `SeekritApiError` — non-2xx from the API; has `.status` and `.code`
  (`"unauthorized"`, `"forbidden"`, `"not_found"`, …).
- `SeekritCryptoError` — a token or ciphertext could not be parsed/decrypted.
- `SeekritError` — base class (also covers network failures).

The client is **fail-closed**: any resolve or decrypt failure raises rather than
returning partial results.

## Notebooks

`seekrit.load()` is the one-call form: resolve, load `os.environ`, done. Put it
at the top of a notebook or script.

```python
import seekrit

seekrit.load()
```

It's built around the two ways a notebook leaks a credential:

- **No token in a cell.** `load()` takes the token from `$SEEKRIT_TOKEN`, and
  when there isn't one it asks through a password prompt (ipykernel routes
  `getpass` to the notebook frontend) — so the token stays in kernel memory
  instead of being saved into the `.ipynb`. Pass `prompt=False` to never ask, or
  set `SEEKRIT_TOKEN` for headless runs like `papermill`.
- **No values in cell outputs.** `load()` returns the names it loaded and the
  scope they came from — never the values — so displaying it in a cell writes a
  summary into the notebook file and nothing more.

```python
loaded = seekrit.load()
loaded                       # <seekrit: 7 secrets loaded from acme/analytics/staging: API_KEY, …>
len(loaded)                  # 7
"DATABASE_URL" in loaded     # True
os.environ["DATABASE_URL"]   # the value lives here, not on the result
```

Re-running the cell refreshes: `load()` defaults to `override=True`, unlike
`into_env()`, so a rotated secret takes effect on a re-run rather than being
skipped as already-set. Pass `override=False` to keep what the environment
already has (those names are then listed in `loaded.skipped`).

This guards the summary, not your own cells — `print(os.environ["API_KEY"])`
still writes a secret into the notebook. Strip outputs before committing.

## Hold a placeholder instead of a key

`seekrit.transport` substitutes `{{seekrit:NAME}}` placeholders into outbound
requests, so a provider key is never in your source, your `.env`, or
`os.environ`:

```bash
pip install 'seekrit[httpx]'
```

```python
import httpx
from openai import OpenAI
from seekrit.transport import SeekritTransport

client = OpenAI(
    api_key="{{seekrit:OPENAI_API_KEY}}",
    http_client=httpx.Client(
        transport=SeekritTransport(allow={"api.openai.com": ["OPENAI_API_KEY"]}),
    ),
)
```

One transport covers every Python agent toolkit, because they all reach the
network through the same `http_client=`: LangChain's `ChatOpenAI`, Pydantic AI's
`OpenAIProvider`, the OpenAI Agents SDK's `set_default_openai_client`,
LlamaIndex's `OpenAI`. Use `AsyncSeekritTransport` for the async client.

The allowlist is the boundary, and it is default-deny: a name that is not
permitted toward that host, method, and path is refused, and so is a name that
did not resolve. Neither sends the request.

A refusal answers with the same **403** the proxy answers with, carrying
`x-seekrit-refusal` and the secret's name but never its value. That is on
purpose: a provider SDK wraps anything its HTTP layer raises into an opaque
connection error *and retries it*, so raising would turn a denied placeholder
into "Connection error" after six attempts. Pass `refusal="raise"` to get the typed error
instead.

### LangChain middleware

`pip install 'seekrit[langchain]'` adds agent middleware that scopes credentials
to a single tool call:

```python
from langchain.agents import create_agent
from seekrit.langchain import SeekritCredentials

agent = create_agent(
    model=model,
    tools=[refund, search],
    context_schema=Context,
    middleware=[
        SeekritCredentials(
            scope=lambda ctx: {"tenants": ctx.tenant},
            tools={"refund": ["STRIPE_SECRET_KEY"]},
        ),
    ],
)
```

`refund` may substitute the Stripe key; `search` may substitute nothing. `scope`
also picks which tenant's secrets to resolve, per request, without rebuilding
the model. Pair it with `require_scope=True` on the transport so a lost context
fails closed. Details:
<https://seekrit.dev/docs/guides/agent-proxy/in-process>.

Because it runs in your process, this is a weaker boundary than the
[egress proxy](https://seekrit.dev/docs/guides/agent-proxy). What it does buy:
the value exists only inside one HTTP call, so it never reaches model context, a
tool result, or a trace exporter — nor an environment-scraping bug in a
dependency.

## Secret references

A secret's value may reference another with `${OTHER_SECRET}`. References are
stored literally and expanded here, after the layers are merged — so a reference
picks up whichever layer won that name, and rotating the referenced secret
updates every value that uses it. `$${OTHER_SECRET}` is a literal; an unknown
name is left as written; a reference cycle raises. Full rules:
[seekrit.dev/docs/guides/references](https://seekrit.dev/docs/guides/references).

```python
client = seekrit.Client(interpolate=False)   # get the stored text instead
```

## Zero-knowledge

`GET /v1/resolve` returns ciphertext plus a data-encryption key wrapped to your
token's public key. This SDK recovers the token's private key, unwraps the DEK
(ECDH P-256 → HKDF-SHA256 → AES-256-GCM), and decrypts each secret
(AES-256-GCM, AAD-bound to `environmentId/NAME`) — the exact scheme used by the
CLI, `seekrit run`, and every other seekrit client. See
[seekrit.dev/docs](https://seekrit.dev/docs/concepts/encryption).

## License

MIT
