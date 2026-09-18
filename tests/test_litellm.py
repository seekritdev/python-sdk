"""LiteLLM custom secret manager.

Two halves, and the split is the point. :class:`SecretResolver` is plain Python
and is tested here without LiteLLM installed — which is also how it ships, since
importing ``seekrit.litellm`` must not import the framework. The
``CustomSecretManager`` subclass is exercised against a stub
``litellm.integrations.custom_secret_manager`` installed for the duration of a
test, and once more against the real base class when one is importable.

What these pin, beyond "it returns the value":

* **Caching.** ``get_secret`` asks the manager before the environment and caches
  nothing, so an uncached manager is one HTTPS round trip per model lookup.
* **A miss is ``None``, never an exception and never ``""``.** LiteLLM's
  fallback to ``os.environ`` is what makes partial adoption work, and an empty
  string applied over a working key is a gateway that 401s while looking healthy.
* **A failure does not become a silent fallback.** The last good snapshot keeps
  answering, bounded, and every serve of it is logged.
* **Writes refuse with a reason.** A read-path SDK has no encrypt path.

Runnable with either ``pytest`` or ``python -m unittest``.
"""

import ast
import asyncio
import logging
import sys
import types
import unittest
import urllib.error
from contextlib import contextmanager

from seekrit.errors import SeekritApiError, SeekritCryptoError, SeekritError
from seekrit.litellm import (
    DEFAULT_TOKEN_ENV,
    SHIM_FILENAME,
    SHIM_TEMPLATE,
    SecretResolver,
    secret_manager_class,
)


def setUpModule():
    # Half these tests deliberately provoke the warnings this module logs, and
    # the ones that *care* about a log line assert it with assertLogs. Keeping
    # the rest off stderr is what makes a CI failure readable.
    logging.getLogger("seekrit.litellm").setLevel(logging.CRITICAL)


class FakeClient:
    def __init__(self, values=None, raises=None):
        self._values = values or {}
        self._raises = raises

    def resolve(self):
        if self._raises is not None:
            raise self._raises
        return dict(self._values)


class Recorder:
    """A client factory that counts fetches and can change its answer."""

    def __init__(self, values=None, raises=None):
        self.values = values if values is not None else {}
        self.raises = raises
        self.calls = []

    def __call__(self, token, api_url, overrides, timeout):
        self.calls.append(
            {"token": token, "api_url": api_url, "overrides": dict(overrides), "timeout": timeout}
        )
        return FakeClient(self.values, self.raises)

    @property
    def count(self):
        return len(self.calls)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def resolver(factory, *, env=None, clock=None, **kwargs):
    return SecretResolver(
        env={DEFAULT_TOKEN_ENV: "skt_test"} if env is None else env,
        client_factory=factory,
        clock=clock or Clock(),
        **kwargs,
    )


# ── reading ──────────────────────────────────────────────────────────────────


class TestRead(unittest.TestCase):
    def test_reads_a_value(self):
        r = resolver(Recorder({"OPENAI_API_KEY": "sk-live"}))
        self.assertEqual(r.read("OPENAI_API_KEY"), "sk-live")

    def test_absent_name_is_a_miss(self):
        # Not an exception: LiteLLM falls back to os.environ on None, which is
        # what lets one gateway take some names from seekrit and some from its
        # own deployment.
        r = resolver(Recorder({"OPENAI_API_KEY": "sk-live"}))
        self.assertIsNone(r.read("DATABASE_URL"))

    def test_empty_value_is_a_miss(self):
        # "" applied over a key the environment already holds is a gateway that
        # looks configured and 401s.
        r = resolver(Recorder({"OPENAI_API_KEY": ""}))
        self.assertIsNone(r.read("OPENAI_API_KEY"))

    def test_empty_name_is_a_miss(self):
        self.assertIsNone(resolver(Recorder({})).read(""))

    def test_allow_narrows_what_is_answered(self):
        r = resolver(
            Recorder({"OPENAI_API_KEY": "sk-live", "DATABASE_URL": "postgres://"}),
            allow=["OPENAI_API_KEY"],
        )
        self.assertEqual(r.read("OPENAI_API_KEY"), "sk-live")
        self.assertIsNone(r.read("DATABASE_URL"))

    def test_allow_does_not_fetch_for_an_excluded_name(self):
        factory = Recorder({"DATABASE_URL": "postgres://"})
        r = resolver(factory, allow=["OPENAI_API_KEY"])
        self.assertIsNone(r.read("DATABASE_URL"))
        self.assertEqual(factory.count, 0)

    def test_names_never_exposes_values(self):
        r = resolver(Recorder({"B": "2", "A": "1"}))
        r.read("A")
        self.assertEqual(list(r.names), ["A", "B"])


# ── caching ──────────────────────────────────────────────────────────────────


class TestCache(unittest.TestCase):
    def test_many_reads_are_one_resolve(self):
        factory = Recorder({"A": "1", "B": "2", "C": "3"})
        r = resolver(factory)
        for name in ("A", "B", "C", "A", "MISSING"):
            r.read(name)
        self.assertEqual(factory.count, 1)

    def test_snapshot_expires(self):
        clock = Clock()
        factory = Recorder({"A": "1"})
        r = resolver(factory, clock=clock, cache_ttl=300.0)
        self.assertEqual(r.read("A"), "1")
        clock.advance(299)
        self.assertEqual(r.read("A"), "1")
        self.assertEqual(factory.count, 1)
        clock.advance(2)
        factory.values = {"A": "2"}
        self.assertEqual(r.read("A"), "2")
        self.assertEqual(factory.count, 2)

    def test_zero_ttl_fetches_every_read(self):
        factory = Recorder({"A": "1"})
        r = resolver(factory, cache_ttl=0)
        r.read("A")
        r.read("A")
        self.assertEqual(factory.count, 2)

    def test_fresh_reports_whether_a_read_touches_the_network(self):
        clock = Clock()
        r = resolver(Recorder({"A": "1"}), clock=clock, cache_ttl=60.0)
        self.assertFalse(r.fresh)
        r.read("A")
        self.assertTrue(r.fresh)
        clock.advance(61)
        self.assertFalse(r.fresh)

    def test_refresh_refetches(self):
        factory = Recorder({"A": "1"})
        r = resolver(factory)
        r.read("A")
        factory.values = {"A": "2"}
        r.refresh()
        self.assertEqual(r.read("A"), "2")
        self.assertEqual(factory.count, 2)

    def test_concurrent_misses_are_one_resolve(self):
        import threading

        factory = Recorder({"A": "1"})
        r = resolver(factory)
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            r.read("A")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(factory.count, 1)


# ── failure ──────────────────────────────────────────────────────────────────


class TestFailure(unittest.TestCase):
    def test_missing_token_is_a_miss_not_a_raise(self):
        r = resolver(Recorder({"A": "1"}), env={})
        self.assertIsNone(r.read("A"))
        self.assertIn(DEFAULT_TOKEN_ENV, r.last_error)

    def test_api_failure_is_a_miss(self):
        r = resolver(Recorder(raises=SeekritApiError(500, "internal", "boom")))
        self.assertIsNone(r.read("A"))
        self.assertIn("500", r.last_error)

    def test_rejected_token_is_described_without_the_body(self):
        r = resolver(Recorder(raises=SeekritApiError(401, "unauthorized", "token skt_abc bad")))
        r.read("A")
        self.assertNotIn("skt_abc", r.last_error)
        self.assertIn("401", r.last_error)

    def test_network_failure_is_described(self):
        r = resolver(Recorder(raises=urllib.error.URLError("no route")))
        r.read("A")
        self.assertIn("could not reach", r.last_error)

    def test_crypto_failure_is_described(self):
        r = resolver(Recorder(raises=SeekritCryptoError("bad tag")))
        r.read("A")
        self.assertIn("decrypt", r.last_error)

    def test_unexpected_failure_is_described_by_type(self):
        r = resolver(Recorder(raises=RuntimeError("surprise")))
        r.read("A")
        self.assertIn("RuntimeError", r.last_error)
        self.assertNotIn("surprise", r.last_error)

    def test_failures_are_not_retried_on_every_read(self):
        # A proxy starting against an unreachable API would otherwise make one
        # failing request per model entry, and then one per request afterwards.
        clock = Clock()
        factory = Recorder(raises=urllib.error.URLError("down"))
        r = resolver(factory, clock=clock, error_ttl=10.0)
        for _ in range(5):
            r.read("A")
        self.assertEqual(factory.count, 1)
        clock.advance(11)
        r.read("A")
        self.assertEqual(factory.count, 2)

    def test_a_stale_snapshot_keeps_answering_and_says_so(self):
        # Serving stale silently is the failure mode worth avoiding: LiteLLM's
        # own fallback would otherwise hide an outage behind whatever the
        # process environment happens to hold.
        clock = Clock()
        factory = Recorder({"A": "1"})
        r = resolver(factory, clock=clock, cache_ttl=60.0, error_ttl=0.0)
        self.assertEqual(r.read("A"), "1")
        factory.raises = urllib.error.URLError("down")
        clock.advance(61)
        with self.assertLogs("seekrit.litellm", level="WARNING") as logs:
            self.assertEqual(r.read("A"), "1")
        self.assertTrue(any("old seekrit snapshot" in line for line in logs.output))
        self.assertIsNotNone(r.last_error)

    def test_a_stale_snapshot_is_dropped_past_the_bound(self):
        # The fail-closed half of a fallback we do not control: past the bound a
        # revoked token stops working even though nothing can replace it.
        clock = Clock()
        factory = Recorder({"A": "1"})
        r = resolver(factory, clock=clock, cache_ttl=60.0, error_ttl=0.0, stale_ttl=3600.0)
        r.read("A")
        factory.raises = urllib.error.URLError("down")
        clock.advance(3601)
        self.assertIsNone(r.read("A"))
        self.assertEqual(list(r.names), [])

    def test_recovery_replaces_the_stale_snapshot(self):
        clock = Clock()
        factory = Recorder({"A": "1"})
        r = resolver(factory, clock=clock, cache_ttl=60.0, error_ttl=0.0)
        r.read("A")
        factory.raises = urllib.error.URLError("down")
        clock.advance(61)
        r.read("A")
        factory.raises = None
        factory.values = {"A": "2"}
        clock.advance(61)
        self.assertEqual(r.read("A"), "2")
        self.assertIsNone(r.last_error)


# ── configuration ────────────────────────────────────────────────────────────


class TestFromEnv(unittest.TestCase):
    def test_defaults(self):
        r = SecretResolver.from_env({})
        self.assertEqual(r.token_env, DEFAULT_TOKEN_ENV)
        self.assertFalse(r.token_present)

    def test_reads_its_settings(self):
        r = SecretResolver.from_env(
            {
                "SEEKRIT_LITELLM_TOKEN_ENV": "GATEWAY_TOKEN",
                "GATEWAY_TOKEN": "skt_x",
                "SEEKRIT_LITELLM_ALLOW": "OPENAI_API_KEY, ANTHROPIC_API_KEY",
            }
        )
        self.assertEqual(r.token_env, "GATEWAY_TOKEN")
        self.assertTrue(r.token_present)

    def test_allow_from_env_narrows(self):
        factory = Recorder({"A": "1", "B": "2"})
        r = SecretResolver.from_env(
            {DEFAULT_TOKEN_ENV: "skt_x", "SEEKRIT_LITELLM_ALLOW": "A"},
            client_factory=factory,
        )
        self.assertEqual(r.read("A"), "1")
        self.assertIsNone(r.read("B"))

    def test_a_bad_ttl_falls_back_to_the_default(self):
        r = SecretResolver.from_env({"SEEKRIT_LITELLM_CACHE_TTL": "soon"})
        self.assertIsInstance(r, SecretResolver)  # constructed, not raised

    def test_keyword_arguments_win_over_the_environment(self):
        r = SecretResolver.from_env(
            {"SEEKRIT_LITELLM_TOKEN_ENV": "FROM_ENV"}, token_env="FROM_KWARG"
        )
        self.assertEqual(r.token_env, "FROM_KWARG")

    def test_timeout_is_passed_through(self):
        factory = Recorder({"A": "1"})
        r = resolver(factory, timeout=5.0)
        r.read("A")
        self.assertEqual(factory.calls[0]["timeout"], 5.0)

    def test_a_per_call_timeout_wins(self):
        factory = Recorder({"A": "1"})
        r = resolver(factory, timeout=5.0)
        r.read("A", 2.0)
        self.assertEqual(factory.calls[0]["timeout"], 2.0)

    def test_api_url_comes_from_the_environment_when_unset(self):
        factory = Recorder({"A": "1"})
        r = resolver(
            factory,
            env={DEFAULT_TOKEN_ENV: "skt_x", "SEEKRIT_API_URL": "https://api.example.test"},
        )
        r.read("A")
        self.assertEqual(factory.calls[0]["api_url"], "https://api.example.test")


# ── the shim ─────────────────────────────────────────────────────────────────


class TestShim(unittest.TestCase):
    """LiteLLM imports the manager from a *file* beside config.yaml.

    ``custom_secret_manager`` is split on ``.`` into exactly two parts, so a
    dotted package path cannot be written there and the shim is not optional.
    These pin the spelling the docs tell people to use.
    """

    def test_the_shim_is_valid_python(self):
        ast.parse(SHIM_TEMPLATE)

    def test_the_shim_imports_the_class_from_this_package(self):
        tree = ast.parse(SHIM_TEMPLATE)
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "seekrit.litellm")
        self.assertEqual([alias.name for alias in imports[0].names], ["SeekritSecretManager"])

    def test_the_documented_path_has_exactly_two_segments(self):
        # LiteLLM does `file_name, class_name = value.split(".")`. Three
        # segments raise "too many values to unpack" at proxy startup.
        module = SHIM_FILENAME[: -len(".py")]
        documented = f"{module}.SeekritSecretManager"
        self.assertEqual(len(documented.split(".")), 2)

    def test_the_shim_filename_is_importable_as_a_module_name(self):
        self.assertTrue(SHIM_FILENAME.endswith(".py"))
        self.assertTrue(SHIM_FILENAME[: -len(".py")].isidentifier())


# ── the LiteLLM adapter ──────────────────────────────────────────────────────


class StubCustomSecretManager:
    """Stands in for ``litellm.integrations.custom_secret_manager``.

    Only the shape the real base class imposes: a name-taking constructor, and
    abstract reads the subclass must supply.
    """

    def __init__(self, secret_manager_name=None, **kwargs):
        self.secret_manager_name = secret_manager_name


@contextmanager
def stub_litellm():
    """Install a stub ``litellm.integrations.custom_secret_manager``."""
    saved = {
        name: sys.modules.get(name)
        for name in (
            "litellm",
            "litellm.integrations",
            "litellm.integrations.custom_secret_manager",
        )
    }
    litellm = types.ModuleType("litellm")
    integrations = types.ModuleType("litellm.integrations")
    module = types.ModuleType("litellm.integrations.custom_secret_manager")
    module.CustomSecretManager = StubCustomSecretManager
    integrations.custom_secret_manager = module
    litellm.integrations = integrations
    sys.modules["litellm"] = litellm
    sys.modules["litellm.integrations"] = integrations
    sys.modules["litellm.integrations.custom_secret_manager"] = module
    try:
        yield module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def manager(factory, **kwargs):
    cls = secret_manager_class()
    return cls(resolver=resolver(factory, **kwargs))


class TestSecretManager(unittest.TestCase):
    def test_it_subclasses_the_framework_base(self):
        with stub_litellm() as module:
            cls = secret_manager_class()
            self.assertTrue(issubclass(cls, module.CustomSecretManager))

    def test_it_constructs_with_no_arguments(self):
        # LiteLLM's loader does `_secret_manager_class()`. Every parameter has
        # to have a default or the proxy fails at startup.
        with stub_litellm():
            cls = secret_manager_class()
            instance = cls()
            self.assertIsInstance(instance.resolver, SecretResolver)
            self.assertEqual(instance.secret_manager_name, "seekrit")

    def test_sync_read(self):
        with stub_litellm():
            m = manager(Recorder({"OPENAI_API_KEY": "sk-live"}))
            self.assertEqual(m.sync_read_secret("OPENAI_API_KEY"), "sk-live")
            self.assertIsNone(m.sync_read_secret("NOPE"))

    def test_sync_read_ignores_optional_params(self):
        # LiteLLM passes its whole key_management_settings here. Nothing in it
        # names a seekrit environment, and reading it would be a second, silent
        # configuration surface.
        with stub_litellm():
            m = manager(Recorder({"A": "1"}))
            self.assertEqual(m.sync_read_secret("A", {"access_mode": "read_only"}), "1")

    def test_async_read_when_warm(self):
        with stub_litellm():
            m = manager(Recorder({"A": "1"}))
            m.sync_read_secret("A")
            self.assertEqual(asyncio.run(m.async_read_secret("A")), "1")

    def test_async_read_when_cold(self):
        with stub_litellm():
            factory = Recorder({"A": "1"})
            m = manager(factory)
            self.assertEqual(asyncio.run(m.async_read_secret("A")), "1")
            self.assertEqual(factory.count, 1)

    def test_an_httpx_timeout_object_is_ignored_not_guessed_at(self):
        with stub_litellm():
            factory = Recorder({"A": "1"})
            m = manager(factory, timeout=7.0)
            m.sync_read_secret("A", None, object())
            self.assertEqual(factory.calls[0]["timeout"], 7.0)

    def test_writes_refuse_with_a_reason(self):
        with stub_litellm():
            m = manager(Recorder({}))
            with self.assertRaises(NotImplementedError) as caught:
                asyncio.run(m.async_write_secret("A", "1"))
            self.assertIn("read-only", str(caught.exception))
            self.assertIn("store_virtual_keys", str(caught.exception))

    def test_deletes_refuse_with_a_reason(self):
        with stub_litellm():
            m = manager(Recorder({}))
            with self.assertRaises(NotImplementedError):
                asyncio.run(m.async_delete_secret("A"))

    def test_validate_environment_reports_a_missing_token(self):
        with stub_litellm():
            self.assertFalse(manager(Recorder({}), env={}).validate_environment())
            self.assertTrue(manager(Recorder({})).validate_environment())

    def test_health_check(self):
        with stub_litellm():
            self.assertTrue(asyncio.run(manager(Recorder({"A": "1"})).async_health_check()))
            unreachable = manager(Recorder(raises=urllib.error.URLError("down")))
            self.assertFalse(asyncio.run(unreachable.async_health_check()))

    def test_repr_holds_no_values(self):
        with stub_litellm():
            m = manager(Recorder({"OPENAI_API_KEY": "sk-live"}))
            m.sync_read_secret("OPENAI_API_KEY")
            self.assertIn("OPENAI_API_KEY", repr(m))
            self.assertNotIn("sk-live", repr(m))


class TestImportIsLazy(unittest.TestCase):
    def test_importing_this_module_does_not_import_litellm(self):
        # The reason the resolver above is framework-free: `import seekrit.litellm`
        # has to be free — and has to *work* — where LiteLLM is not installed.
        # In a subprocess, because this file imports LiteLLM itself when it can.
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import seekrit.litellm, sys; print('litellm' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_the_module_attribute_builds_the_class(self):
        import seekrit.litellm as module

        with stub_litellm():
            # Drop any class a previous test cached, so __getattr__ runs.
            module.__dict__.pop("SeekritSecretManager", None)
            self.assertIs(module.SeekritSecretManager, module.__dict__["SeekritSecretManager"])
        module.__dict__.pop("SeekritSecretManager", None)

    def test_an_unknown_attribute_still_raises(self):
        import seekrit.litellm as module

        with self.assertRaises(AttributeError):
            module.NoSuchThing


try:  # pragma: no cover - depends on the install
    from litellm.integrations.custom_secret_manager import (  # noqa: F401
        CustomSecretManager as _RealBase,
    )

    HAS_LITELLM = True
except Exception:  # noqa: BLE001 - any import failure means "not installed"
    HAS_LITELLM = False


@unittest.skipUnless(HAS_LITELLM, "litellm is not installed")
class TestAgainstRealLiteLLM(unittest.TestCase):
    """The same class against LiteLLM's own base.

    The stub above pins the shape this SDK codes against; this pins that the
    shape is still LiteLLM's. A signature change upstream — or a new abstract
    method — fails here rather than at a customer's proxy startup.
    """

    def test_it_is_a_custom_secret_manager(self):
        cls = secret_manager_class()
        self.assertTrue(issubclass(cls, _RealBase))

    def test_it_instantiates_with_no_arguments(self):
        cls = secret_manager_class()
        instance = cls()  # abstract methods unimplemented would raise here
        self.assertIsInstance(instance.resolver, SecretResolver)

    def test_it_reads(self):
        cls = secret_manager_class()
        instance = cls(resolver=resolver(Recorder({"OPENAI_API_KEY": "sk-live"})))
        self.assertEqual(instance.sync_read_secret("OPENAI_API_KEY"), "sk-live")
        self.assertEqual(asyncio.run(instance.async_read_secret("OPENAI_API_KEY")), "sk-live")

    def test_the_loader_contract_holds(self):
        # What `load_custom_secret_manager` does: split the configured path in
        # two, import that file, getattr the class, check the subclass, call it.
        import importlib.util
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            config = os.path.join(directory, "config.yaml")
            file_name, class_name = f"{SHIM_FILENAME[:-3]}.SeekritSecretManager".split(".")
            with open(os.path.join(directory, file_name + ".py"), "w") as handle:
                handle.write(SHIM_TEMPLATE)
            path = os.path.join(os.path.dirname(config), file_name) + ".py"
            spec = importlib.util.spec_from_file_location(class_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded = getattr(module, class_name)
            self.assertTrue(issubclass(loaded, _RealBase))
            self.assertIsInstance(loaded(), _RealBase)


if __name__ == "__main__":
    unittest.main()
