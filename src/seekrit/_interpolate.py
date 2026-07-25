"""Secret references: ``${OTHER_SECRET}`` inside a secret value.

Expansion happens here, in-process, after the layers are merged — the API only
ever holds the ciphertext of the literal ``${OTHER_SECRET}`` text. The rules are
fixed by the shared golden fixture (``testdata/vectors.json``, key
``interpolation``), which every seekrit client is tested against:

* ``${NAME}`` becomes ``NAME``'s value from the same merged set, recursively.
* ``NAME`` must look like an environment variable (``[A-Za-z_][A-Za-z0-9_]*``);
  anything else (``${FOO:-bar}``, ``${1}``) is left exactly as written.
* A name that is not in the set is left literal and listed in ``unresolved``, so
  a stored value containing e.g. ``${GITHUB_SHA}`` keeps working.
* ``$${NAME}`` is an escape for the literal text ``${NAME}``.
* A reference cycle raises :class:`SeekritReferenceError`.
"""

from __future__ import annotations

import re
from typing import Dict, Iterator, List, Mapping, NamedTuple, Optional, Tuple

from .errors import SeekritReferenceError

_REFERENCE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Cap on a single expanded value: nested references can multiply length.
MAX_EXPANDED_LENGTH = 1_048_576


class Interpolation(NamedTuple):
    """The outcome of expanding a variable set."""

    values: Dict[str, str]
    #: Names whose value had at least one reference expanded.
    expanded: List[str]
    #: Referenced names that exist nowhere in the set, deduped and sorted.
    unresolved: List[str]


def _scan(text: str) -> Iterator[Tuple[Optional[str], str]]:
    """Yield ``(reference, text)`` pairs: a reference name, or literal text.

    The single tokenizer the rules above are expressed in terms of. When
    ``reference`` is set, ``text`` is the raw ``${NAME}`` source it came from.
    """
    i = 0
    length = len(text)
    while i < length:
        dollar = text.find("$", i)
        if dollar == -1:
            yield None, text[i:]
            return
        if dollar > i:
            yield None, text[i:dollar]

        if text[dollar + 1 : dollar + 3] == "${":
            # `$${` — escape: emit a literal `${` and skip all three chars, so
            # the `{NAME}` that follows can never be read as a reference.
            yield None, "${"
            i = dollar + 3
            continue

        close = text.find("}", dollar + 2) if text[dollar + 1 : dollar + 2] == "{" else -1
        reference = text[dollar + 2 : close] if close != -1 else None
        if reference is not None and _REFERENCE_NAME.match(reference):
            yield reference, text[dollar : close + 1]
            i = close + 1
            continue

        # A plain `$`, an unterminated `${`, or a non-name: literal. Advancing a
        # single char lets a `${` later in the run still be recognized.
        yield None, "$"
        i = dollar + 1


def interpolate_secrets(values: Mapping[str, str]) -> Interpolation:
    """Expand ``${NAME}`` references throughout a merged variable set.

    Pure: the input mapping is never mutated. Raises
    :class:`~seekrit.errors.SeekritReferenceError` on a reference cycle.
    """
    resolved: Dict[str, str] = {}
    unresolved: set = set()
    expanded: List[str] = []
    stack: List[str] = []

    def resolve(name: str) -> str:
        if name in resolved:
            return resolved[name]
        if name in stack:
            chain = stack[stack.index(name) :] + [name]
            raise SeekritReferenceError("CYCLE", "secret reference cycle: " + " -> ".join(chain))

        stack.append(name)
        parts: List[str] = []
        for reference, text in _scan(values[name]):
            if reference is None:
                parts.append(text)
            elif reference in values:
                parts.append(resolve(reference))
            else:
                unresolved.add(reference)
                parts.append(text)
        stack.pop()

        out = "".join(parts)
        if len(out) > MAX_EXPANDED_LENGTH:
            raise SeekritReferenceError(
                "TOO_LARGE", f"{name} expands to more than {MAX_EXPANDED_LENGTH} bytes"
            )
        resolved[name] = out
        return out

    result: Dict[str, str] = {}
    for name, original in values.items():
        value = resolve(name)
        result[name] = value
        if value != original:
            expanded.append(name)

    return Interpolation(values=result, expanded=expanded, unresolved=sorted(unresolved))
