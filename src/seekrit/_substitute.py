"""Placeholder substitution — the in-process twin of ``apps/proxy/src/substitute.rs``.

Scans text for ``{{seekrit:NAME}}`` and replaces each occurrence with a
looked-up value. **Fail-closed**: a placeholder naming a secret that is not
permitted toward this upstream, or one that does not resolve, raises rather than
forwarding the placeholder (or letting the wrong host receive a real
credential). Errors carry the *name* only — never the value.

This is a different feature from :func:`seekrit.interpolate_secrets`, which
expands ``${OTHER_SECRET}`` references *between stored secrets*. Two syntaxes,
two engines: ``${...}`` is about how a value is composed, ``{{seekrit:...}}``
is about where a value is injected on the way out.

The rules are pinned by the shared golden fixture (``testdata/vectors.json``,
``substitution``), which the Rust proxy asserts against the same file.
"""

from __future__ import annotations

import re
from typing import Callable, List, NamedTuple, Tuple

from .errors import SeekritSubstitutionError

OPEN = "{{seekrit:"
CLOSE = "}}"
_VALID_NAME = re.compile(r"\A[A-Za-z0-9_]+\Z")


class Lookup(NamedTuple):
    """The result of looking up one placeholder name for one outbound request."""

    kind: str  # "value" | "denied" | "unknown"
    value: str = ""
    reason: str = ""

    @staticmethod
    def found(value: str) -> "Lookup":
        """Permitted and resolved: substitute this decrypted value."""
        return Lookup("value", value)

    @staticmethod
    def denied(reason: str = "") -> "Lookup":
        """Referenced but not permitted toward this upstream (default-deny)."""
        return Lookup("denied", "", reason)

    @staticmethod
    def unknown() -> "Lookup":
        """Permitted, but no such secret resolved (fail-closed)."""
        return Lookup("unknown")


def substitute(text: str, lookup: Callable[[str], Lookup]) -> Tuple[str, List[str]]:
    """Replace every ``{{seekrit:NAME}}`` in ``text`` using ``lookup``.

    A malformed or unterminated marker is left verbatim — it is not a valid
    placeholder, so it is not a credential reference either.

    Returns:
        ``(rewritten_text, injected_names)``. The names are sorted and
        deduplicated, and are safe to log.

    Raises:
        SeekritSubstitutionError: on a denied or unresolved name.
    """
    names = set()
    out: List[str] = []
    i = 0
    length = len(text)

    while i < length:
        at = text.find(OPEN, i)
        if at == -1:
            out.append(text[i:])
            break
        out.append(text[i:at])

        after = at + len(OPEN)
        close = text.find(CLOSE, after)
        if close == -1:
            # No closing marker anywhere: the remainder is literal.
            out.append(text[at:])
            break

        name = text[after:close]
        if not _VALID_NAME.match(name):
            # Not a placeholder. Emit the opener and rescan from just after it,
            # so a nested `{{seekrit:` inside the junk is still found.
            out.append(OPEN)
            i = after
            continue

        found = lookup(name)
        if found.kind == "denied":
            raise SeekritSubstitutionError("denied", name, found.reason)
        if found.kind == "unknown":
            raise SeekritSubstitutionError("unresolved", name)
        out.append(found.value)
        names.add(name)
        i = close + len(CLOSE)

    return "".join(out), sorted(names)


def has_placeholder(text: str) -> bool:
    """Whether ``text`` contains at least one syntactically valid placeholder."""
    i = 0
    while True:
        at = text.find(OPEN, i)
        if at == -1:
            return False
        after = at + len(OPEN)
        close = text.find(CLOSE, after)
        if close == -1:
            return False
        if _VALID_NAME.match(text[after:close]):
            return True
        i = after
