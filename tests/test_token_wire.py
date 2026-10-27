import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from seekrit import Client
from seekrit._crypto import TokenKey, service_token_auth
from seekrit.errors import SeekritCryptoError

V = json.loads((Path(__file__).parent.parent / "testdata/vectors.json").read_text())

class TokenWireTests(unittest.TestCase):
    def test_auth_only_on_wire_and_local_decryption(self):
        def fetch(request, **kwargs):
            self.assertEqual(request.get_header("Authorization"), "Bearer " + V["authToken"])
            self.assertNotIn(V["token"].split(".")[2], str(request.headers))
            return io.BytesIO(json.dumps(V["resolve"]).encode())
        with patch("urllib.request.urlopen", side_effect=fetch) as request:
            self.assertEqual(Client(token=V["token"]).resolve(), V["expectedManagedValues"])
            request.assert_called_once()
        with self.assertRaises(SeekritCryptoError):
            TokenKey.parse(V["authToken"])

    def test_legacy_is_offline_only(self):
        legacy = V["tokenId"] + "_" + V["token"].split(".")[2]
        self.assertEqual(TokenKey.parse(legacy).token_id, V["tokenId"])
        with patch("urllib.request.urlopen") as request:
            with self.assertRaisesRegex(SeekritCryptoError, "v2 service token"):
                Client(token=legacy)
            request.assert_not_called()
        for invalid in [V["token"] + "\n", V["token"] + ".extra", "skt_bad"]:
            with self.assertRaises(SeekritCryptoError):
                service_token_auth(invalid)
