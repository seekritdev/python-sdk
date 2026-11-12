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

import base64
import binascii
import re
from typing import Callable, List, NamedTuple, Optional, Tuple

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


def substitute_header(
    name: str, value: str, lookup: Callable[[str], Lookup]
) -> Tuple[str, List[str]]:
    """:func:`substitute` for one request header's value — plus the one place a
    client encodes a placeholder before sending it.

    git over HTTPS (and anything else speaking HTTP Basic) sends
    ``user:password`` base64-encoded, so
    ``https://x-access-token:{{seekrit:GITHUB_TOKEN}}@github.com`` leaves the
    client as ``Authorization: Basic eC1hY2Nlc3Mt…`` and the marker is invisible
    to a text scan. For exactly that shape — an ``Authorization`` or
    ``Proxy-Authorization`` value of ``Basic`` (any case), one or more spaces,
    then canonical padded base64 that decodes to UTF-8 — the decoded text goes
    through the same ``lookup`` and is re-encoded, keeping the scheme and
    spacing as sent. A denied or unknown name raises exactly as it does anywhere
    else.

    Every other value, a Basic credential failing any of those checks included,
    is substituted as plain text — so ``Basic {{seekrit:GIT_BASIC}}`` still
    works, and no other header or scheme is ever base64-decoded. Pinned, like
    the rest, by the ``header`` cases of the shared vectors
    (``apps/proxy/src/substitute.rs`` is the reference).

    Raises:
        SeekritSubstitutionError: on a denied or unresolved name.
    """
    basic = _basic_credential(name, value)
    if basic is None:
        return substitute(value, lookup)
    prefix, decoded = basic
    text, names = substitute(decoded, lookup)
    if not names:
        # Nothing injected: forward the credential exactly as it arrived.
        return value, names
    return prefix + base64.b64encode(text.encode("utf-8")).decode("ascii"), names


def header_has_placeholder(name: str, value: str) -> bool:
    """Whether a header value carries a placeholder :func:`substitute_header`
    would act on — :func:`has_placeholder`, seeing inside a Basic credential."""
    basic = _basic_credential(name, value)
    return has_placeholder(value if basic is None else basic[1])


_CREDENTIAL_HEADERS = frozenset(("authorization", "proxy-authorization"))
# `Basic` in any ASCII case (no Unicode case folding), then one or more spaces.
_BASIC_PREFIX = re.compile(r"\A[Bb][Aa][Ss][Ii][Cc] +")
# Standard alphabet (RFC 4648 §4), padded. Length and padding bits are checked
# by re-encoding, the same way every other implementation checks them.
_BASE64_SHAPE = re.compile(r"\A[A-Za-z0-9+/]+={0,2}\Z")


def _basic_credential(name: str, value: str) -> Optional[Tuple[str, str]]:
    """``(scheme_and_spaces, decoded_text)`` for a Basic credential in a header
    that carries one; ``None`` for anything to be scanned as text."""
    if name.lower() not in _CREDENTIAL_HEADERS:
        return None
    match = _BASIC_PREFIX.match(value)
    if match is None:
        return None
    token = value[match.end() :]
    if len(token) % 4 or not _BASE64_SHAPE.match(token):
        return None
    try:
        raw = base64.b64decode(token, validate=True)
    except (binascii.Error, ValueError):
        return None
    # Canonical only: a lenient decoder accepts stray padding bits, and every
    # leniency is a different answer to "is this a credential?" than the proxy's.
    if base64.b64encode(raw).decode("ascii") != token:
        return None
    try:
        return match.group(0), raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


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
