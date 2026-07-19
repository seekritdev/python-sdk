"""Exception hierarchy for the seekrit SDK."""

from __future__ import annotations


class SeekritError(Exception):
    """Base class for every error raised by this SDK."""


class SeekritCryptoError(SeekritError):
    """A token, wrapped DEK, or secret ciphertext could not be parsed or decrypted.

    Deliberately unspecific: a decrypt failure does not distinguish "wrong key"
    from "tampered data" from "mismatched context", to avoid acting as an oracle.
    """


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
