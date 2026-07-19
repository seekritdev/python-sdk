"""Cross-implementation parity: decrypt the shared golden vectors and assert we
recover exactly what the canonical @seekrit/crypto (WebCrypto) produced.

Runnable with either ``pytest`` or ``python -m unittest``.
"""

import base64
import json
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

import seekrit
from seekrit import SeekritCryptoError

VECTORS = json.loads((Path(__file__).parent.parent / "testdata" / "vectors.json").read_text())


def _make_valid_but_different_token() -> str:
    """A well-formed token carrying a freshly generated (wrong) P-256 key."""
    pkcs8 = ec.generate_private_key(ec.SECP256R1()).private_bytes(
        Encoding.DER, PrivateFormat.PKCS8, NoEncryption()
    )
    key_b64 = base64.urlsafe_b64encode(pkcs8).rstrip(b"=").decode("ascii")
    return "skt_AAAAAAAAAAAAAAAAAAAAAA_" + key_b64


class VectorTest(unittest.TestCase):
    def test_materialize_matches_expected(self):
        key = seekrit.TokenKey.parse(VECTORS["token"])
        merged = seekrit.materialize(VECTORS["resolve"], key)
        self.assertEqual(merged, VECTORS["expectedManagedValues"])

    def test_app_layer_wins_over_group(self):
        key = seekrit.TokenKey.parse(VECTORS["token"])
        merged = seekrit.materialize(VECTORS["resolve"], key)
        self.assertEqual(merged["SHARED"], "from-app")

    def test_unicode_and_empty_round_trip(self):
        key = seekrit.TokenKey.parse(VECTORS["token"])
        merged = seekrit.materialize(VECTORS["resolve"], key)
        self.assertEqual(merged["UNICODE"], "héllo-🌍-\n-tab\tend")
        self.assertEqual(merged["EMPTY"], "")

    def test_wrong_token_cannot_unwrap(self):
        # A valid, well-formed token holding the wrong key must not silently
        # succeed — the DEK grant was wrapped to a different public key.
        key = seekrit.TokenKey.parse(_make_valid_but_different_token())
        with self.assertRaises(SeekritCryptoError):
            seekrit.materialize(VECTORS["resolve"], key)


if __name__ == "__main__":
    unittest.main()
