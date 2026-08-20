"""Exception hierarchy for the seekrit SDK."""

from __future__ import annotations


class SeekritError(Exception):
    """Base class for every error raised by this SDK."""


class SeekritCryptoError(SeekritError):
    """A token, wrapped DEK, or secret ciphertext could not be parsed or decrypted.

    Deliberately unspecific: a decrypt failure does not distinguish "wrong key"
    from "tampered data" from "mismatched context", to avoid acting as an oracle.
    """


class SeekritReferenceError(SeekritError):
    """A ``${OTHER_SECRET}`` reference could not be expanded.

    Raised only for cases with no possible answer: a reference cycle, or an
    expansion that grows without bound. An *unknown* reference is not an error —
    it is left as literal text (see :func:`seekrit.interpolate_secrets`).

    Attributes:
        code: ``"CYCLE"`` or ``"TOO_LARGE"``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SeekritApiError(SeekritError):
    """The resolve API returned a non-2xx response.

    Attributes:
        status: HTTP status code.
        code: the ``error.code`` string from the API (e.g. ``"unauthorized"``,
            ``"forbidden"``, ``"not_found"``), or ``"internal"`` if the body did
            not parse.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code


class SeekritSubstitutionError(SeekritError):
    """A ``{{seekrit:NAME}}`` placeholder could not be substituted.

    Both cases are fail-closed on purpose: forwarding the literal placeholder
    would leak an internal name to the upstream, and substituting anyway would
    hand a credential to a host that is not allowed to have it. The request is
    not sent.

    Attributes:
        code: ``"denied"`` (the allowlist refused this name toward this
            upstream), ``"unresolved"`` (allowed, but the token resolved no
            secret by that name), or ``"scope_required"`` (``require_scope`` is
            set and no scope was in effect, so narrowing could not be applied).
        secret_name: the placeholder name that failed. Safe to log — this
            exception never carries a value.
    """

    def __init__(self, code: str, secret_name: str, detail: str = "") -> None:
        if code == "denied":
            suffix = f" ({detail})" if detail else ""
            message = f"secret {secret_name} is not allowed toward this upstream{suffix}"
        elif code == "scope_required":
            message = (
                "require_scope is set and no scope is in effect: refusing to inject with "
                "the unnarrowed allowlist"
            )
        else:
            message = f"secret {secret_name} is allowed but did not resolve"
        super().__init__(message)
        self.code = code
        self.secret_name = secret_name
