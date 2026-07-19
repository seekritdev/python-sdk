"""seekrit — read-path SDK for the zero-knowledge secrets manager.

    import seekrit

    client = seekrit.Client()          # reads $SEEKRIT_TOKEN
    secrets = client.resolve()         # {"DATABASE_URL": "...", ...}
    db = client.get("DATABASE_URL")

Secrets are decrypted in-process; the API only ever sees ciphertext.
"""

from ._client import Client, DEFAULT_API_URL
from ._crypto import TokenKey, decrypt_secret, materialize, secret_aad
from .errors import SeekritApiError, SeekritCryptoError, SeekritError

__version__ = "0.2.0"  # x-release-please-version

__all__ = [
    "Client",
    "DEFAULT_API_URL",
    "TokenKey",
    "decrypt_secret",
    "materialize",
    "secret_aad",
    "SeekritError",
    "SeekritApiError",
    "SeekritCryptoError",
    "__version__",
]
