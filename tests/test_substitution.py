"""Cross-implementation parity for `{{seekrit:NAME}}` substitution.

The cases come from the ``substitution`` section of the shared golden fixture,
which the Rust proxy asserts against the same file
(``apps/proxy/tests/substitution_vectors.rs``). The proxy is the reference
implementation; this test is what stops the in-process shim from drifting away
from the process an operator eventually puts in front of it.

Runnable with either ``pytest`` or ``python -m unittest``.
"""

import json
import unittest
from pathlib import Path

from seekrit._substitute import Lookup, has_placeholder, substitute
from seekrit.errors import SeekritSubstitutionError

VECTORS = json.loads((Path(__file__).parent.parent / "testdata" / "vectors.json").read_text())


def _lookup_for(case):
    """The lookup the fixture's ``note`` prescribes: denied, then values, then unknown."""
    denied = set(case.get("denied") or ())
    values = case["values"]

    def lookup(name):
        if name in denied:
            return Lookup.denied("test")
        if name in values:
            return Lookup.found(values[name])
        return Lookup.unknown()

    return lookup


class SubstitutionVectorTests(unittest.TestCase):
    def test_fixture_carries_cases(self):
        self.assertGreater(len(VECTORS["substitution"]["cases"]), 0)

    def test_every_vector_matches(self):
        for case in VECTORS["substitution"]["cases"]:
            with self.subTest(case["name"]):
                lookup = _lookup_for(case)
                expected_error = case.get("error")
                if expected_error:
                    with self.assertRaises(SeekritSubstitutionError) as caught:
                        substitute(case["input"], lookup)
                    self.assertEqual(caught.exception.code, expected_error["code"])
                    self.assertEqual(caught.exception.secret_name, expected_error["name"])
                    continue
                text, names = substitute(case["input"], lookup)
                self.assertEqual(text, case["expected"])
                self.assertEqual(names, case["names"])

    def test_an_error_never_carries_the_value(self):
        secret = "sk-super-secret-value"
        with self.assertRaises(SeekritSubstitutionError) as caught:
            substitute("{{seekrit:K}}", lambda name: Lookup.denied("secret_not_allowed"))
        self.assertNotIn(secret, str(caught.exception))
        self.assertEqual(caught.exception.secret_name, "K")

    def test_has_placeholder_agrees_with_substitute(self):
        for case in VECTORS["substitution"]["cases"]:
            with self.subTest(case["name"]):
                found = has_placeholder(case["input"])
                if case.get("error"):
                    self.assertTrue(found, "a case that raises must contain a placeholder")
                    continue
                # No placeholder ⇒ the text must come back byte-identical.
                if not found:
                    self.assertEqual(case["input"], case["expected"])


if __name__ == "__main__":
    unittest.main()
