"""``seekrit.load()``: injection, precedence, token acquisition, and the
guarantee that its result never carries a secret value.

The decrypt path is real — the golden resolve response from ``testdata`` is fed
in where the network would be, so these assert plaintext actually lands in the
target environment.

Runnable with either ``pytest`` or ``python -m unittest``.
"""

import json
import os
import unittest
from pathlib import Path
from unittest import mock

import seekrit
from seekrit import SeekritError
from seekrit._load import LoadedSecrets, _stdin_can_prompt

VECTORS = json.loads((Path(__file__).parent.parent / "testdata" / "vectors.json").read_text())
TOKEN = VECTORS["token"]
EXPECTED = VECTORS["expectedManagedValues"]
SCOPE = VECTORS["resolve"]["scope"]


def _patch_fetch():
    """Serve the golden resolve response instead of calling the API."""
    return mock.patch.object(seekrit.Client, "_fetch", lambda self: VECTORS["resolve"])


class LoadTest(unittest.TestCase):
    def test_loads_decrypted_values_into_the_given_env(self):
        env = {}
        with _patch_fetch():
            result = seekrit.load(TOKEN, env=env, prompt=False)
        self.assertEqual(env, EXPECTED)
        self.assertEqual(set(result.names), set(EXPECTED))
        self.assertEqual(result.skipped, ())

    def test_defaults_to_os_environ(self):
        name = "DATABASE_URL"
        with _patch_fetch(), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(name, None)
            seekrit.load(TOKEN, prompt=False)
            self.assertEqual(os.environ[name], EXPECTED[name])

    def test_secrets_win_by_default(self):
        """Unlike Client.into_env: calling load() declares seekrit is the
        source of truth."""
        env = {"DATABASE_URL": "postgres://local"}
        with _patch_fetch():
            result = seekrit.load(TOKEN, env=env, prompt=False)
        self.assertEqual(env["DATABASE_URL"], EXPECTED["DATABASE_URL"])
        self.assertEqual(result.skipped, ())
        self.assertIn("DATABASE_URL", result.names)

    def test_override_false_keeps_existing_and_reports_it(self):
        env = {"DATABASE_URL": "postgres://local"}
        with _patch_fetch():
            result = seekrit.load(TOKEN, env=env, prompt=False, override=False)
        self.assertEqual(env["DATABASE_URL"], "postgres://local")
        self.assertEqual(result.skipped, ("DATABASE_URL",))
        self.assertNotIn("DATABASE_URL", result.names)

    def test_re_running_the_cell_is_idempotent(self):
        """Re-running the setup cell is the normal notebook motion; it must
        reload rather than report loading nothing."""
        env = {}
        with _patch_fetch():
            first = seekrit.load(TOKEN, env=env, prompt=False)
            second = seekrit.load(TOKEN, env=env, prompt=False)
        self.assertEqual(env, EXPECTED)
        self.assertEqual(first.names, second.names)
        self.assertEqual(second.skipped, ())

    def test_reads_token_from_environment(self):
        env = {}
        with _patch_fetch(), mock.patch.dict(os.environ, {"SEEKRIT_TOKEN": TOKEN}):
            seekrit.load(env=env, prompt=False)
        self.assertEqual(env, EXPECTED)

    def test_interpolate_false_returns_stored_text(self):
        env = {}
        with _patch_fetch():
            seekrit.load(TOKEN, env=env, prompt=False, interpolate=False)
        self.assertIn("${", env["REFERENCING"])

    def test_carries_the_scope_labels(self):
        with _patch_fetch():
            result = seekrit.load(TOKEN, env={}, prompt=False)
        self.assertEqual(result.org, SCOPE["orgSlug"])
        self.assertEqual(result.app, SCOPE["appSlug"])
        self.assertEqual(result.environment, SCOPE["envSlug"])
        self.assertEqual(
            result.scope, f"{SCOPE['orgSlug']}/{SCOPE['appSlug']}/{SCOPE['envSlug']}"
        )


class NoValueLeakTest(unittest.TestCase):
    """The result is what a notebook cell displays and saves — it must not hold
    plaintext, in any representation."""

    def _result(self):
        with _patch_fetch():
            return seekrit.load(TOKEN, env={}, prompt=False)

    def test_repr_and_html_contain_no_secret_value(self):
        result = self._result()
        for rendering in (repr(result), result._repr_html_(), str(result)):
            for name, value in EXPECTED.items():
                if value:  # EMPTY is "" and trivially a substring
                    self.assertNotIn(value, rendering, f"{name} leaked into {rendering!r}")

    def test_repr_names_the_scope_and_the_count(self):
        result = self._result()
        rendering = repr(result)
        self.assertIn(str(len(EXPECTED)), rendering)
        self.assertIn(SCOPE["envSlug"], rendering)

    def test_repr_truncates_a_long_name_list(self):
        """The vectors hold exactly the display limit, so drive the boundary
        directly rather than depending on the fixture's size."""
        names = [f"SECRET_{index:02d}" for index in range(20)]
        result = LoadedSecrets(names, [], SCOPE)
        rendering = repr(result)
        self.assertIn("SECRET_00", rendering)
        self.assertIn("+12 more", rendering)
        self.assertNotIn("SECRET_19", rendering)

        at_limit = LoadedSecrets(names[:8], [], SCOPE)
        self.assertNotIn("more", repr(at_limit))

    def test_html_repr_escapes_names(self):
        """Names come from the API; the HTML rendering must not pass markup
        through into notebook output."""
        result = LoadedSecrets(["<script>alert(1)</script>"], ["A&B"], SCOPE)
        rendering = result._repr_html_()
        self.assertNotIn("<script>", rendering)
        self.assertIn("&lt;script&gt;", rendering)
        self.assertIn("A&amp;B", rendering)

    def test_result_exposes_names_not_values(self):
        result = self._result()
        self.assertEqual(len(result), len(EXPECTED))
        self.assertIn("DATABASE_URL", result)
        self.assertEqual(sorted(result), sorted(EXPECTED))
        self.assertFalse(hasattr(result, "values"))
        self.assertFalse(hasattr(result, "__getitem__"))


class TokenPromptTest(unittest.TestCase):
    def test_prompts_when_no_token_is_available(self):
        env = {}
        with _patch_fetch(), mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("seekrit._load.getpass.getpass", return_value=TOKEN) as prompt:
                seekrit.load(env=env, prompt=True)
        prompt.assert_called_once()
        self.assertEqual(env, EXPECTED)

    def test_does_not_prompt_when_a_token_is_present(self):
        with _patch_fetch(), mock.patch("seekrit._load.getpass.getpass") as prompt:
            seekrit.load(TOKEN, env={}, prompt=True)
        prompt.assert_not_called()

    def test_prompt_false_raises_with_actionable_guidance(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SeekritError) as caught:
                seekrit.load(env={}, prompt=False)
        message = str(caught.exception)
        self.assertIn("SEEKRIT_TOKEN", message)

    def test_auto_does_not_prompt_when_stdin_cannot_carry_it(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("seekrit._load._stdin_can_prompt", return_value=False):
                with mock.patch("seekrit._load.getpass.getpass") as prompt:
                    with self.assertRaises(SeekritError):
                        seekrit.load(env={}, prompt="auto")
        prompt.assert_not_called()

    def test_unavailable_stdin_is_reported_as_a_seekrit_error(self):
        """papermill and friends run kernels with stdin disabled."""

        class StdinNotImplementedError(Exception):
            pass

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch(
                "seekrit._load.getpass.getpass", side_effect=StdinNotImplementedError()
            ):
                with self.assertRaises(SeekritError) as caught:
                    seekrit.load(env={}, prompt=True)
        self.assertIn("stdin is unavailable", str(caught.exception))

    def test_empty_prompt_response_is_not_a_token(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("seekrit._load.getpass.getpass", return_value="  "):
                with self.assertRaises(SeekritError):
                    seekrit.load(env={}, prompt=True)

    def test_keyboard_interrupt_at_the_prompt_is_not_swallowed(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("seekrit._load.getpass.getpass", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    seekrit.load(env={}, prompt=True)


class StdinDetectionTest(unittest.TestCase):
    def test_an_active_ipython_kernel_can_prompt(self):
        """ipykernel routes getpass to the frontend, so a kernel counts even
        though its stdin is not a tty."""
        shell = mock.Mock(kernel=object())
        fake_ipython = mock.Mock(get_ipython=mock.Mock(return_value=shell))
        with mock.patch.dict("sys.modules", {"IPython": fake_ipython}):
            self.assertTrue(_stdin_can_prompt())

    def test_ipython_installed_but_not_running_falls_back_to_the_tty_check(self):
        fake_ipython = mock.Mock(get_ipython=mock.Mock(return_value=None))
        with mock.patch.dict("sys.modules", {"IPython": fake_ipython}):
            with mock.patch("sys.stdin") as stdin:
                stdin.isatty.return_value = False
                self.assertFalse(_stdin_can_prompt())
                stdin.isatty.return_value = True
                self.assertTrue(_stdin_can_prompt())

    def test_no_ipython_and_no_tty(self):
        with mock.patch.dict("sys.modules", {}, clear=False):
            import sys as _sys

            _sys.modules.pop("IPython", None)
            with mock.patch("sys.stdin", None):
                self.assertFalse(_stdin_can_prompt())


if __name__ == "__main__":
    unittest.main()
