"""Hermes secret sources.

Hermes' contract for a secret source is unusually strict, and every rule in it
exists because breaking it hurts a *different* plugin or the user's own `.env`:
``fetch()`` must never raise, must never prompt, must classify failures into
machine-readable kinds, must never write ``os.environ`` itself, and must never
apply an empty value over a working credential. Those are what these tests pin.

Nothing here imports Hermes — that is the point of keeping the resolvers
framework-free. The logic runs in CI that has never heard of the host, and the
``SecretSource`` adapter is exercised against a stub ``agent.secret_sources.base``
installed for the duration of a test.

Runnable with either ``pytest`` or ``python -m unittest``.
"""

import sys
import types
import unittest
import urllib.error
from contextlib import contextmanager

from seekrit.errors import SeekritApiError, SeekritCryptoError, SeekritError
from seekrit.hermes import (
    BULK_SOURCE_NAME,
    MAPPED_SOURCE_NAME,
    REFERENCE_SCHEME,
    Outcome,
    SecretReference,
    parse_reference,
    register,
    resolve_bulk,
    resolve_mapped,
    secret_source_classes,
)


class FakeClient:
    def __init__(self, values=None, raises=None):
        self._values = values or {}
        self._raises = raises

    def resolve(self):
        if self._raises is not None:
            raise self._raises
        return dict(self._values)


def factory_for(**by_token_env):
    """A client factory keyed by the env var each token would come from."""
    calls = []

    def factory(token_env, cfg, env):
        calls.append(token_env)
        entry = by_token_env.get(token_env)
        if entry is None:
            raise SeekritError("no service token: %s is not set" % token_env)
        return entry

    factory.calls = calls
    return factory


def no_token(token_env, cfg, env):
    raise SeekritError("no service token: %s is not set" % token_env)


# ── references ───────────────────────────────────────────────────────────────


class TestParseReference(unittest.TestCase):
    def test_bare_name(self):
        self.assertEqual(parse_reference("skt://OPENAI_API_KEY"), SecretReference("OPENAI_API_KEY"))

    def test_token_alias(self):
        self.assertEqual(
            parse_reference("skt://billing/STRIPE_SECRET_KEY"),
            SecretReference("STRIPE_SECRET_KEY", "billing"),
        )

    def test_rejects_malformed(self):
        for raw in [
            "",
            "OPENAI_API_KEY",
            "op://vault/item/field",
            "skt://",
            "skt://a/b/C",
            "skt://not-a-var",
            "skt://billing/",
            "skt:///NAME",
            "SKT://NAME",
            None,
        ]:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_reference(raw))

    def test_scheme_is_the_one_we_publish(self):
        # Hermes rejects a registration whose `scheme` another source already
        # owns, so this string is a compatibility surface, not a detail.
        self.assertEqual(REFERENCE_SCHEME, "skt")


# ── bulk ─────────────────────────────────────────────────────────────────────


class TestResolveBulk(unittest.TestCase):
    def test_resolves_the_whole_environment(self):
        outcome = resolve_bulk(
            {"enabled": True},
            env={"SEEKRIT_TOKEN": "skt_x"},
            client_factory=factory_for(SEEKRIT_TOKEN=FakeClient({"A": "1", "B": "2"})),
        )
        self.assertEqual(outcome.secrets, {"A": "1", "B": "2"})
        self.assertEqual(outcome.error_kind, "")

    def test_include_narrows(self):
        outcome = resolve_bulk(
            {"include": ["A"]},
            env={},
            client_factory=factory_for(SEEKRIT_TOKEN=FakeClient({"A": "1", "B": "2"})),
        )
        self.assertEqual(outcome.secrets, {"A": "1"})

    def test_include_accepts_a_bare_string(self):
        outcome = resolve_bulk(
            {"include": "A"},
            env={},
            client_factory=factory_for(SEEKRIT_TOKEN=FakeClient({"A": "1", "B": "2"})),
        )
        self.assertEqual(outcome.secrets, {"A": "1"})

    def test_exclude_removes(self):
        outcome = resolve_bulk(
            {"exclude": ["B"]},
            env={},
            client_factory=factory_for(SEEKRIT_TOKEN=FakeClient({"A": "1", "B": "2"})),
        )
        self.assertEqual(outcome.secrets, {"A": "1"})

    def test_reads_an_alternate_token_variable(self):
        factory = factory_for(SEEKRIT_TOKEN_CI=FakeClient({"A": "1"}))
        outcome = resolve_bulk({"token_env": "SEEKRIT_TOKEN_CI"}, env={}, client_factory=factory)
        self.assertEqual(outcome.secrets, {"A": "1"})
        self.assertEqual(factory.calls, ["SEEKRIT_TOKEN_CI"])

    def test_skips_names_that_are_not_env_vars(self):
        # A seekrit secret name is normally a variable name, but nothing stops a
        # dashed one existing — and mangling it into a legal one is worse.
        outcome = resolve_bulk(
            {},
            env={},
            client_factory=factory_for(SEEKRIT_TOKEN=FakeClient({"A": "1", "not-a-var": "2"})),
        )
        self.assertEqual(outcome.secrets, {"A": "1"})

    def test_never_applies_an_empty_value(self):
        # Hermes has a dedicated kind for this because applying "" silently
        # replaces a working credential with a broken one.
        outcome = resolve_bulk(
            {},
            env={},
            client_factory=factory_for(SEEKRIT_TOKEN=FakeClient({"A": "", "B": "2"})),
        )
        self.assertEqual(outcome.secrets, {"B": "2"})

    def test_reports_empty_value_when_that_is_all_there_was(self):
        outcome = resolve_bulk(
            {}, env={}, client_factory=factory_for(SEEKRIT_TOKEN=FakeClient({"A": ""}))
        )
        self.assertEqual(outcome.secrets, {})
        self.assertEqual(outcome.error_kind, "EMPTY_VALUE")
        self.assertIn("A", outcome.error)


# ── mapped ───────────────────────────────────────────────────────────────────


class TestResolveMapped(unittest.TestCase):
    def test_binds_and_renames(self):
        outcome = resolve_mapped(
            {"env": {"OPENAI_API_KEY": "skt://OPENAI_KEY"}},
            env={},
            client_factory=factory_for(SEEKRIT_TOKEN=FakeClient({"OPENAI_KEY": "sk-1"})),
        )
        self.assertEqual(outcome.secrets, {"OPENAI_API_KEY": "sk-1"})

    def test_resolves_each_token_once(self):
        factory = factory_for(
            SEEKRIT_TOKEN=FakeClient({"A": "1", "B": "2"}),
            SEEKRIT_TOKEN_BILLING=FakeClient({"C": "3"}),
        )
        outcome = resolve_mapped(
            {
                "tokens": {"billing": "SEEKRIT_TOKEN_BILLING"},
                "env": {"A": "skt://A", "B": "skt://B", "C": "skt://billing/C"},
            },
            env={},
            client_factory=factory,
        )
        self.assertEqual(outcome.secrets, {"A": "1", "B": "2", "C": "3"})
        self.assertEqual(sorted(factory.calls), ["SEEKRIT_TOKEN", "SEEKRIT_TOKEN_BILLING"])

    def test_an_empty_or_missing_map_is_not_configured(self):
        for cfg in [{"env": {}}, {}]:
            with self.subTest(cfg=cfg):
                outcome = resolve_mapped(cfg, env={}, client_factory=factory_for())
                self.assertEqual(outcome.error_kind, "NOT_CONFIGURED")

    def test_bad_reference_is_reported_without_resolving(self):
        # Validated before the network, so a typo reads as a bad reference rather
        # than as whatever the request happened to fail with.
        factory = factory_for(SEEKRIT_TOKEN=FakeClient({"A": "1"}))
        outcome = resolve_mapped({"env": {"A": "op://v/i/f"}}, env={}, client_factory=factory)
        self.assertEqual(outcome.error_kind, "REF_INVALID")
        self.assertEqual(factory.calls, [])

    def test_unknown_token_alias_is_a_bad_reference(self):
        outcome = resolve_mapped(
            {"env": {"A": "skt://nope/A"}}, env={}, client_factory=factory_for()
        )
        self.assertEqual(outcome.error_kind, "REF_INVALID")
        self.assertIn("nope", outcome.error)

    def test_unusable_variable_name_is_a_bad_reference(self):
        outcome = resolve_mapped(
            {"env": {"not-a-var": "skt://A"}}, env={}, client_factory=factory_for()
        )
        self.assertEqual(outcome.error_kind, "REF_INVALID")

    def test_a_missing_secret_fails_the_whole_map(self):
        # A mapped binding says "this variable comes from here". Contributing
        # half a map hides a config error behind a partly-working agent.
        outcome = resolve_mapped(
            {"env": {"A": "skt://A", "B": "skt://NOPE"}},
            env={},
            client_factory=factory_for(SEEKRIT_TOKEN=FakeClient({"A": "1"})),
        )
        self.assertEqual(outcome.secrets, {})
        self.assertEqual(outcome.error_kind, "REF_INVALID")
        self.assertIn("NOPE", outcome.error)

    def test_never_applies_an_empty_value(self):
        outcome = resolve_mapped(
            {"env": {"A": "skt://A", "B": "skt://B"}},
            env={},
            client_factory=factory_for(SEEKRIT_TOKEN=FakeClient({"A": "1", "B": ""})),
        )
        self.assertEqual(outcome.secrets, {"A": "1"})


# ── the never-raises contract ────────────────────────────────────────────────

CLASSIFICATIONS = [
    (SeekritApiError(401, "unauthorized", "nope"), "AUTH_FAILED"),
    (SeekritApiError(403, "forbidden", "nope"), "AUTH_FAILED"),
    (SeekritApiError(404, "not_found", "nope"), "REF_INVALID"),
    (SeekritApiError(429, "rate_limited", "slow down"), "NETWORK"),
    (SeekritApiError(503, "internal", "down"), "NETWORK"),
    (SeekritApiError(400, "bad_request", "hm"), "INTERNAL"),
    (urllib.error.URLError("no route to host"), "NETWORK"),
    (SeekritCryptoError("could not decrypt"), "AUTH_FAILED"),
    (SeekritError("no service token"), "NOT_CONFIGURED"),
    (ValueError("something unexpected"), "INTERNAL"),
]


class TestNeverRaises(unittest.TestCase):
    def test_classifies_every_failure_without_raising(self):
        for resolve in (resolve_bulk, resolve_mapped):
            for exc, kind in CLASSIFICATIONS:
                with self.subTest(resolve=resolve.__name__, exc=type(exc).__name__, kind=kind):
                    outcome = resolve(
                        {"env": {"A": "skt://A"}},
                        env={},
                        client_factory=factory_for(SEEKRIT_TOKEN=FakeClient(raises=exc)),
                    )
                    self.assertIsInstance(outcome, Outcome)
                    self.assertEqual(outcome.error_kind, kind)
                    self.assertEqual(outcome.secrets, {})

    def test_a_missing_token_is_not_configured(self):
        for resolve in (resolve_bulk, resolve_mapped):
            with self.subTest(resolve=resolve.__name__):
                outcome = resolve({"env": {"A": "skt://A"}}, env={}, client_factory=no_token)
                self.assertEqual(outcome.error_kind, "NOT_CONFIGURED")

    def test_an_outcome_repr_holds_no_values(self):
        # An Outcome holds live credentials, and its repr is what ends up in a
        # traceback or a debug log.
        outcome = Outcome({"OPENAI_API_KEY": "sk-live-abc123"})
        self.assertNotIn("sk-live-abc123", repr(outcome))
        self.assertIn("OPENAI_API_KEY", repr(outcome))

    def test_error_messages_are_built_from_codes_not_bodies(self):
        # The API's own message text is not ours to trust or relay.
        outcome = resolve_bulk(
            {},
            env={},
            client_factory=factory_for(
                SEEKRIT_TOKEN=FakeClient(raises=SeekritApiError(401, "unauthorized", "sk-live-xyz"))
            ),
        )
        self.assertNotIn("sk-live-xyz", outcome.error)


# ── the adapter, against a stub Hermes ───────────────────────────────────────


class _ErrorKind:
    NOT_CONFIGURED = "NOT_CONFIGURED"
    BINARY_MISSING = "BINARY_MISSING"
    AUTH_FAILED = "AUTH_FAILED"
    AUTH_EXPIRED = "AUTH_EXPIRED"
    REF_INVALID = "REF_INVALID"
    NETWORK = "NETWORK"
    EMPTY_VALUE = "EMPTY_VALUE"
    TIMEOUT = "TIMEOUT"
    INTERNAL = "INTERNAL"


class _FetchResult:
    def __init__(self):
        self.secrets = {}
        self.error = ""
        self.error_kind = None


class _SecretSource:
    api_version = 1


@contextmanager
def stub_hermes(error_kind=_ErrorKind):
    """Install a stand-in for ``agent.secret_sources.base``."""
    names = ("agent", "agent.secret_sources", "agent.secret_sources.base")
    saved = {name: sys.modules.get(name) for name in names}
    base = types.ModuleType("agent.secret_sources.base")
    base.ErrorKind = error_kind
    base.FetchResult = _FetchResult
    base.SecretSource = _SecretSource
    sys.modules["agent"] = types.ModuleType("agent")
    sys.modules["agent.secret_sources"] = types.ModuleType("agent.secret_sources")
    sys.modules["agent.secret_sources.base"] = base
    try:
        yield base
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class TestSecretSourceClasses(unittest.TestCase):
    def test_importing_this_module_does_not_import_hermes(self):
        # `pip install seekrit` declares a `hermes_agent.plugins` entry point, so
        # importing the module it names must not require the host to be present.
        self.assertNotIn("agent.secret_sources.base", sys.modules)

    def test_raises_a_clear_import_error_without_hermes(self):
        with self.assertRaises(ImportError) as caught:
            secret_source_classes()
        self.assertIn("agent.secret_sources.base", str(caught.exception))

    def test_declares_the_names_shapes_and_scheme_hermes_registers_on(self):
        with stub_hermes():
            bulk, mapped = secret_source_classes()
        self.assertEqual(bulk.name, BULK_SOURCE_NAME)
        self.assertEqual(bulk.shape, "bulk")
        self.assertIsNone(getattr(bulk, "scheme", None))
        self.assertEqual(mapped.name, MAPPED_SOURCE_NAME)
        self.assertEqual(mapped.shape, "mapped")
        self.assertEqual(mapped.scheme, REFERENCE_SCHEME)

    def test_leaves_enablement_to_hermes_default(self):
        # Registering both sources is only free if neither does anything unasked,
        # and Hermes' own default is `cfg.get("enabled")`. Overriding
        # `is_enabled` here would be how one of them becomes on-by-default.
        with stub_hermes():
            bulk, mapped = secret_source_classes()
        for source in (bulk, mapped):
            with self.subTest(source=source.__name__):
                self.assertFalse(hasattr(source, "is_enabled"))

    def test_protects_every_token_variable_it_reads(self):
        with stub_hermes():
            _, mapped = secret_source_classes()
            protected = mapped().protected_env_vars(
                {"token_env": "SEEKRIT_TOKEN_CI", "tokens": {"billing": "SEEKRIT_TOKEN_BILLING"}}
            )
        self.assertEqual(protected, frozenset({"SEEKRIT_TOKEN_CI", "SEEKRIT_TOKEN_BILLING"}))

    def test_default_token_variable_is_protected_without_config(self):
        with stub_hermes():
            bulk, _ = secret_source_classes()
            self.assertEqual(bulk().protected_env_vars({}), frozenset({"SEEKRIT_TOKEN"}))

    def test_overrides_existing_values_by_default(self):
        # seekrit is the store of record: after a rotation the new value has to
        # beat whatever a stale .env still holds.
        with stub_hermes():
            bulk, _ = secret_source_classes()
            self.assertIs(bulk().override_existing({}), True)
            self.assertIs(bulk().override_existing({"override_existing": False}), False)

    def test_timeout_is_positive_and_configurable(self):
        with stub_hermes():
            bulk, _ = secret_source_classes()
            self.assertGreater(bulk().fetch_timeout_seconds({}), 0)
            self.assertEqual(bulk().fetch_timeout_seconds({"timeout_seconds": 5}), 5)

    def test_remediation_is_a_pure_kind_to_string_map(self):
        with stub_hermes():
            bulk, _ = secret_source_classes()
            source = bulk()
            self.assertIn("seekrit token create", source.remediation(_ErrorKind.NOT_CONFIGURED, {}))
            # An unknown kind suppresses the hint rather than raising.
            self.assertEqual(source.remediation("SOMETHING_NEW", {}), "")

    def test_config_schemas_describe_their_own_keys(self):
        with stub_hermes():
            bulk, mapped = secret_source_classes()
            self.assertIn("include", bulk().config_schema())
            self.assertNotIn("env", bulk().config_schema())
            self.assertIn("env", mapped().config_schema())
            self.assertIn("tokens", mapped().config_schema())

    def test_fetch_returns_a_fetch_result_carrying_the_real_error_kind(self):
        with stub_hermes():
            bulk, _ = secret_source_classes()
            result = bulk().fetch({"token_env": "SEEKRIT_TOKEN_ABSENT_IN_TESTS"}, None)
        self.assertIsInstance(result, _FetchResult)
        self.assertEqual(result.error_kind, _ErrorKind.NOT_CONFIGURED)
        self.assertEqual(result.secrets, {})

    def test_an_unknown_error_kind_degrades_to_internal(self):
        # An ErrorKind a given Hermes build lacks must cost an INTERNAL, not an
        # AttributeError during startup.
        class Sparse:
            INTERNAL = "INTERNAL"

        with stub_hermes(error_kind=Sparse):
            bulk, _ = secret_source_classes()
            result = bulk().fetch({"token_env": "SEEKRIT_TOKEN_ABSENT_IN_TESTS"}, None)
        self.assertEqual(result.error_kind, "INTERNAL")


class TestRegister(unittest.TestCase):
    def test_registers_both_sources(self):
        registered = []

        class Ctx:
            def register_secret_source(self, source):
                registered.append(source)

        with stub_hermes():
            register(Ctx())
        self.assertEqual(
            [type(s).__name__ for s in registered],
            ["SeekritSecretSource", "SeekritReferenceSecretSource"],
        )
        self.assertEqual({s.name for s in registered}, {BULK_SOURCE_NAME, MAPPED_SOURCE_NAME})


if __name__ == "__main__":
    unittest.main()
