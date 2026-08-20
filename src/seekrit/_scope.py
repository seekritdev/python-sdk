"""The ambient scope — per-request narrowing, carried in a context variable.

A framework adapter (see :mod:`seekrit.langchain`) sets a scope for the duration
of one model call or one tool call; the transport reads it when a request goes
out. That composition is what keeps the *model instance* static while the
*credentials* vary per request — the alternative, rebuilding a chat model per
tenant, means a per-tenant object cache and a per-tenant connection pool.

Being a :class:`~contextvars.ContextVar`, a scope propagates into the same task
and into threads started with :func:`asyncio.to_thread` (which copies the
context), but **not** into a bare :meth:`~asyncio.loop.run_in_executor` call.
When a lost scope would silently widen an allowlist rather than narrow it, set
``require_scope=True`` on the transport so the request is refused instead.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Scope:
    """What this request may resolve, and what it may inject."""

    overrides: Optional[Mapping[str, str]] = None
    """``{group_slug: env_slug}`` — resolve a different slice for this request."""
    allow: Optional[Sequence[str]] = None
    """Narrow the allowlist to these names. Intersected, never widening."""
    label: str = ""
    """Free text for logs — a tool name, a tenant id. Never a value."""

    def key(self) -> str:
        """A stable cache key for the resolve this scope implies."""
        if not self.overrides:
            return ""
        return repr(sorted(self.overrides.items()))


_current: ContextVar[Optional[Scope]] = ContextVar("seekrit_scope", default=None)


def current_scope() -> Optional[Scope]:
    """The scope in effect, or ``None`` outside any :func:`use_scope` block."""
    return _current.get()


@contextmanager
def use_scope(scope: Optional[Scope]) -> Iterator[None]:
    """Install ``scope`` for the duration of the block, then restore the previous."""
    token = _current.set(scope)
    try:
        yield
    finally:
        _current.reset(token)
