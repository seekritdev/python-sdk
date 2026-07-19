"""Zero-knowledge read path, byte-compatible with ``@seekrit/crypto``.

Three steps, matching ``crates/seekrit-core`` and ``packages/crypto``:

1. Recover the service token's P-256 private key  (:class:`TokenKey`).
2. Unwrap each environment DEK                     (:meth:`TokenKey.unwrap_dek`).
3. Decrypt each secret                             (:func:`decrypt_secret`).

Blob formats (all segments base64url, no padding):

* token       ``skt_<id>_<pkcs8 private key>``
* wrapped DEK ``wd1.<ephemeral pub (raw SEC1)>.<hkdf salt>.<iv>.<ciphertext||tag>``
* secret      ``sc1.<iv>.<ciphertext||tag>``
"""

from __future__ import annotations

import base64
import re
from typing import Dict, List

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import load_der_private_key

from .errors import SeekritCryptoError

_HKDF_INFO = b"seekrit/wrap-dek/v1"
# Only the first two underscores are separators; the key segment may contain "_".
_TOKEN_RE = re.compile(r"^(skt_[0-9A-Za-z]+)_([A-Za-z0-9_-]+)$")


def _b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _split_blob(blob: str, prefix: str, segments: int) -> List[str]:
    parts = blob.split(".")
    if not parts or parts[0] != prefix:
        raise SeekritCryptoError(f'expected a "{prefix}" blob, got "{parts[0] if parts else ""}"')
    if len(parts) != segments + 1 or any(p == "" for p in parts):
        raise SeekritCryptoError(f'malformed "{prefix}" blob')
    return parts[1:]


def secret_aad(environment_id: str, name: str) -> bytes:
    """AAD binding a secret ciphertext to its environment and name."""
    return f"{environment_id}/{name}".encode("utf-8")


class TokenKey:
    """The P-256 private key recovered from a ``skt_...`` service token."""

    __slots__ = ("token_id", "_private_key")

    def __init__(self, token_id: str, private_key: ec.EllipticCurvePrivateKey) -> None:
        self.token_id = token_id
        self._private_key = private_key

    @classmethod
    def parse(cls, token: str) -> "TokenKey":
        match = _TOKEN_RE.match(token)
        if not match:
            raise SeekritCryptoError("not a valid seekrit service token")
        token_id, key_b64 = match.group(1), match.group(2)
        try:
            key = load_der_private_key(_b64url_decode(key_b64), password=None)
        except Exception as exc:  # noqa: BLE001 - opaque on purpose
            raise SeekritCryptoError("service token private key is corrupted") from exc
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise SeekritCryptoError("service token key is not an EC private key")
        return cls(token_id, key)

    def unwrap_dek(self, wrapped: str) -> bytes:
        """Recover a 32-byte environment DEK from a ``wd1.`` blob."""
        eph_b64, salt_b64, iv_b64, ct_b64 = _split_blob(wrapped, "wd1", 4)
        try:
            peer = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256R1(), _b64url_decode(eph_b64)
            )
            shared = self._private_key.exchange(ec.ECDH(), peer)
            wrapping_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=_b64url_decode(salt_b64),
                info=_HKDF_INFO,
            ).derive(shared)
            dek = AESGCM(wrapping_key).decrypt(_b64url_decode(iv_b64), _b64url_decode(ct_b64), None)
        except (InvalidTag, ValueError) as exc:
            raise SeekritCryptoError(
                "DEK unwrap failed: wrong private key or tampered grant"
            ) from exc
        if len(dek) != 32:
            raise SeekritCryptoError("unwrapped DEK has wrong length")
        return dek


def decrypt_secret(dek: bytes, blob: str, aad: bytes) -> str:
    """Decrypt an ``sc1.`` secret ciphertext to its UTF-8 plaintext."""
    iv_b64, ct_b64 = _split_blob(blob, "sc1", 2)
    try:
        plaintext = AESGCM(dek).decrypt(_b64url_decode(iv_b64), _b64url_decode(ct_b64), aad)
    except (InvalidTag, ValueError) as exc:
        raise SeekritCryptoError(
            "secret decryption failed: wrong key, tampered data, or mismatched context"
        ) from exc
    return plaintext.decode("utf-8")


def materialize(resolve_response: dict, key: TokenKey) -> Dict[str, str]:
    """Decrypt every layer and merge by precedence.

    ``layers`` arrive lowest precedence first (composed groups, then the app
    environment); later layers overwrite earlier ones on a name collision.
    """
    merged: Dict[str, str] = {}
    for layer in resolve_response["layers"]:
        dek = key.unwrap_dek(layer["wrappedDek"])
        env_id = layer["environmentId"]
        for secret in layer["secrets"]:
            merged[secret["name"]] = decrypt_secret(
                dek, secret["ciphertext"], secret_aad(env_id, secret["name"])
            )
    return merged
