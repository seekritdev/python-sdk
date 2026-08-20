"""The allowlist — which secret may be injected toward which upstream.

This is the same evaluation the proxy performs (``RuleSet::decide`` in
``crates/seekrit-core``, mirrored in ``packages/core/src/agent-policy.ts``), and
:class:`AllowRule` is the *wire* shape of a rule inside a signed ``ap1.`` policy
bundle. That is deliberate: the ``rules`` array out of a verified bundle can be
handed to :class:`seekrit.transport.SeekritTransport` unchanged.

Default-deny throughout. ``allow=()`` means no secret may be injected toward
that host; empty ``methods``/``paths`` mean "any", because an empty list as
"deny everything" is a shape that only ever arises by mistake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, NamedTuple, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AllowRule:
    """One upstream host and what may be injected toward it."""

    host: str
    """Bare hostname, lowercased: no scheme, no port, no path."""
    methods: Tuple[str, ...] = ()
    """Uppercased HTTP methods. Empty means any."""
    paths: Tuple[str, ...] = ()
    """Path globs (``*`` within a segment, ``**`` across). Empty means any."""
    allow: Tuple[str, ...] = ()
    """Secret names injectable toward this host. Empty means none."""
    label: str = ""
    """Free-text note. Never used in matching."""

    @staticmethod
    def from_dict(raw: Mapping[str, object]) -> "AllowRule":
        """Build a rule from a bundle's JSON object (lists, not tuples)."""
        def strings(key: str) -> Tuple[str, ...]:
            value = raw.get(key) or ()
            if isinstance(value, str):  # a bare string is a common typo
                return (value,)
            return tuple(str(v) for v in value)  # type: ignore[union-attr]

        return AllowRule(
            host=str(raw.get("host", "")),
            methods=strings("methods"),
            paths=strings("paths"),
            allow=strings("allow"),
            label=str(raw.get("label", "") or ""),
        )


class Verdict(NamedTuple):
    """Why a query was allowed or refused, and which rule decided."""

    decision: str
    """``allow`` | ``no_rule`` | ``method_not_allowed`` | ``path_not_allowed`` | ``secret_not_allowed``"""
    rule_index: Optional[int]


def match_path(pattern: str, path: str) -> bool:
    """Match a request path against a glob.

    ``*`` matches within one segment, ``**`` matches any number of segments (so
    ``/v1/**`` covers ``/v1``). Case-sensitive, and the query string never
    participates.
    """
    bare = path.split("?", 1)[0]
    return _match_segments(pattern.split("/"), bare.split("/"))


def _match_segments(pattern: Sequence[str], segments: Sequence[str]) -> bool:
    if not pattern:
        return not segments
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        return any(_match_segments(rest, segments[skip:]) for skip in range(len(segments) + 1))
    if not segments:
        return False
    return _match_segment(head, segments[0]) and _match_segments(rest, segments[1:])


def _match_segment(pattern: str, segment: str) -> bool:
    if "*" not in pattern:
        return pattern == segment
    parts = pattern.split("*")
    rest = segment
    last = len(parts) - 1
    for i, part in enumerate(parts):
        if part == "":
            continue
        if i == 0:
            if not rest.startswith(part):
                return False
            rest = rest[len(part) :]
        elif i == last:
            return len(rest) >= len(part) and rest.endswith(part)
        else:
            at = rest.find(part)
            if at == -1:
                return False
            rest = rest[at + len(part) :]
    return True


def _covers_method(rule: AllowRule, method: str) -> bool:
    if not rule.methods:
        return True
    wanted = method.strip().upper()
    return any(m.strip().upper() == wanted for m in rule.methods)


def _covers_path(rule: AllowRule, path: str) -> bool:
    if not rule.paths:
        return True
    return any(match_path(p, path) for p in rule.paths)


def evaluate(
    rules: Sequence[AllowRule],
    *,
    host: str,
    method: str,
    path: str,
    secret: Optional[str] = None,
) -> Verdict:
    """Decide one query against an ordered rule set — first match wins.

    A refusal says *which* constraint refused. Naming the constraint matters: a
    default-deny rule set fails in exactly the confusing direction.
    """
    wanted_host = host.strip().lower()
    host_matched = False
    path_matched_index: Optional[int] = None

    for i, rule in enumerate(rules):
        if rule.host.strip().lower() != wanted_host:
            continue
        host_matched = True
        paths_ok = _covers_path(rule, path)
        if paths_ok and _covers_method(rule, method):
            if secret is not None and secret not in rule.allow:
                return Verdict("secret_not_allowed", i)
            return Verdict("allow", i)
        if paths_ok and path_matched_index is None:
            path_matched_index = i

    if path_matched_index is not None:
        return Verdict("method_not_allowed", path_matched_index)
    return Verdict("path_not_allowed" if host_matched else "no_rule", None)


def rules_from_allow(allow: Mapping[str, Sequence[str]]) -> List[AllowRule]:
    """Expand the ``{host: [names]}`` shorthand into rules permitting any operation."""
    return [AllowRule(host=host, allow=tuple(names)) for host, names in allow.items()]


def narrow(rules: Sequence[AllowRule], names: Sequence[str]) -> List[AllowRule]:
    """Intersect every rule's ``allow`` with ``names`` — narrowing only.

    A name the static rules never permitted cannot be introduced by a scope.
    """
    wanted = set(names)
    return [
        AllowRule(
            host=rule.host,
            methods=rule.methods,
            paths=rule.paths,
            allow=tuple(n for n in rule.allow if n in wanted),
            label=rule.label,
        )
        for rule in rules
    ]
