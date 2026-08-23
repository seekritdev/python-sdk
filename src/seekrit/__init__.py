"""seekrit — read-path SDK for the zero-knowledge secrets manager.

    import seekrit

    seekrit.load()                     # one call: secrets -> os.environ

    client = seekrit.Client()          # reads $SEEKRIT_TOKEN
    secrets = client.resolve()         # {"DATABASE_URL": "...", ...}
    db = client.get("DATABASE_URL")

Secrets are decrypted in-process; the API only ever sees ciphertext.
"""

from ._client import Client, DEFAULT_API_URL
from ._crypto import TokenKey, decrypt_secret, materialize, secret_aad
from ._interpolate import Interpolation, interpolate_secrets
from ._load import LoadedSecrets, load
from .errors import (
    SeekritApiError,
    SeekritCryptoError,
    SeekritError,
    SeekritReferenceError,
    SeekritSubstitutionError,
)

__version__ = "0.7.0"  # x-release-please-version

__all__ = [
    "Client",
    "DEFAULT_API_URL",
    "load",
    "LoadedSecrets",
    "TokenKey",
    "decrypt_secret",
    "materialize",
    "secret_aad",
    "Interpolation",
    "interpolate_secrets",
    "SeekritError",
    "SeekritApiError",
    "SeekritCryptoError",
    "SeekritReferenceError",
    "SeekritSubstitutionError",
    "__version__",
]
