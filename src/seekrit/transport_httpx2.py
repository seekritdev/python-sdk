"""``httpx2`` transports that hold a placeholder instead of a credential.

    import httpx2
    from typesafe_sdk import TypeSafeClient
    from seekrit.transport_httpx2 import SeekritTransport

    client = TypeSafeClient(
        api_key="{{seekrit:TYPESAFE_API_KEY}}",
        transport=SeekritTransport(allow={"api.typesafe.ai": ["TYPESAFE_API_KEY"]}),
    )

Same classes, same arguments, and the same behaviour as
:mod:`seekrit.transport` — the shared half lives in :mod:`seekrit._shim`, so
there is one implementation of the allowlist, the scope, the resolve cache and
the refusal. Only the HTTP client differs.

It differs because ``httpx2`` is a separate distribution rather than a newer
release of ``httpx``: ``httpx2.BaseTransport`` and ``httpx.BaseTransport`` are
unrelated classes, so a client that takes one rejects the other. A client built
on ``httpx2`` therefore cannot use :mod:`seekrit.transport` at all, whatever the
version installed. The TypeSafe SDK is the first such client we ship a path for.

Both modules can be imported in the same process. They share no state, and a
project mid-migration can keep an ``httpx`` transport on one client and an
``httpx2`` transport on another.

Do not call ``httpx2.alias_httpx()`` to avoid this: it rebinds ``import httpx``
process-wide, and httpx2's own documentation says libraries must never do it.
Pick the module that matches the client you are configuring.

Install with ``pip install 'seekrit[httpx2]'``.
"""

from __future__ import annotations

from typing import Any

try:
    import httpx2
except ImportError as exc:  # pragma: no cover - exercised by the extras install
    raise ImportError(
        "seekrit.transport_httpx2 requires httpx2. Install it with: "
        "pip install 'seekrit[httpx2]'"
    ) from exc

from . import _shim
from ._policy import AllowRule
from ._scope import Scope, current_scope, use_scope
from ._shim import ClientSource, InjectHook, RefuseHook, ResolveSource, ScopeSource
from ._shim import Resolver as _Resolver
from ._shim import build_rules as _build_rules
from .errors import SeekritSubstitutionError

__all__ = [
    "SeekritTransport",
    "AsyncSeekritTransport",
    "AllowRule",
    "ClientSource",
    "ResolveSource",
    "Scope",
    "current_scope",
    "use_scope",
]


def refusal_response(error: SeekritSubstitutionError) -> "httpx2.Response":
    """The 403 a refusal answers with. See :func:`seekrit._shim.refusal_response`."""
    return _shim.refusal_response(httpx2, error)


class _Rewriter(_shim.Rewriter):
    """:class:`seekrit._shim.Rewriter` bound to ``httpx2``."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(hx=httpx2, **kwargs)


class SeekritTransport(_shim.SyncCore, httpx2.BaseTransport):
    """A synchronous ``httpx2`` transport that substitutes ``{{seekrit:NAME}}``.

    The arguments are :class:`seekrit.transport.SeekritTransport`'s, unchanged,
    and ``tests/test_transport_httpx2.py`` fails if the two signatures drift.
    ``transport`` defaults to ``httpx2.HTTPTransport()``.
    """

    _hx = httpx2


class AsyncSeekritTransport(_shim.AsyncCore, httpx2.AsyncBaseTransport):
    """The ``async`` twin of :class:`SeekritTransport`; same arguments.

    ``transport`` defaults to ``httpx2.AsyncHTTPTransport()``. The resolve itself
    is synchronous (the SDK speaks ``urllib``), so it runs in the default
    executor rather than blocking the event loop.
    """

    _hx = httpx2
