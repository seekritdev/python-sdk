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

## Zero-knowledge

`GET /v1/resolve` returns ciphertext plus a data-encryption key wrapped to your
token's public key. This SDK recovers the token's private key, unwraps the DEK
(ECDH P-256 → HKDF-SHA256 → AES-256-GCM), and decrypts each secret
(AES-256-GCM, AAD-bound to `environmentId/NAME`) — the exact scheme used by the
CLI, `seekrit run`, and every other seekrit client. See
[seekrit.dev/docs](https://seekrit.dev/docs/concepts/encryption).

## License

MIT
