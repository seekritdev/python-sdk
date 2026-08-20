"""``seekrit.load()`` — one call to put an environment's secrets in ``os.environ``.

Written for notebooks, where the two usual footguns are cell outputs (a plain
``dict`` of secrets typed as a cell's last expression is *saved into the
.ipynb*) and token handling (a kernel launched from JupyterHub or a desktop app
does not inherit the shell's ``SEEKRIT_TOKEN``, so the tempting move is to paste
a live credential into a cell). So :func:`load` returns a value-free
:class:`LoadedSecrets` — names and scope, never plaintext — and prompts for the
token when one is missing and stdin can carry it.

Nothing here is notebook-specific, though: the same call works in a script.
"""

from __future__ import annotations

import getpass
import html
import os
import sys
from typing import Iterator, Mapping, MutableMapping, Optional, Sequence, Union

from ._client import Client
from .errors import SeekritError

_MAX_NAMES_IN_REPR = 8

PromptMode = Union[bool, str]  # True | False | "auto"


def _as_code(names: Sequence[str]) -> str:
    """Secret names as escaped ``<code>`` spans. Names come from the API, so
    they are escaped rather than trusted to be shell-safe identifiers."""
    return "".join(f"<code>{html.escape(name)}</code> " for name in names)


class LoadedSecrets:
    """What :func:`load` put in the environment — **names only, never values**.

    Deliberately not a mapping of secrets: this object is what a notebook cell
    displays and what gets written into the saved ``.ipynb``, so it cannot hold
    plaintext to leak. Read the values back from the environment you loaded into
    (``os.environ["DATABASE_URL"]``).

    Attributes:
        names: secret names written to the environment, sorted.
        skipped: names already set in the target environment and left alone
            (only ever populated when called with ``override=False``).
        org, app, environment: the scope slugs the token resolved to.
    """

    __slots__ = ("names", "skipped", "org", "app", "environment")

    def __init__(
        self,
        names: Sequence[str],
        skipped: Sequence[str],
        scope: Mapping[str, str],
    ) -> None:
        self.names = tuple(sorted(names))
        self.skipped = tuple(sorted(skipped))
        self.org = scope.get("orgSlug", "")
        self.app = scope.get("appSlug", "")
        self.environment = scope.get("envSlug", "")

    @property
    def scope(self) -> str:
        """``"org/app/environment"`` — the slugs the token is bound to."""
        return "/".join(part for part in (self.org, self.app, self.environment) if part)

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __repr__(self) -> str:
        shown = ", ".join(self.names[:_MAX_NAMES_IN_REPR])
        if len(self.names) > _MAX_NAMES_IN_REPR:
            shown += f", +{len(self.names) - _MAX_NAMES_IN_REPR} more"
        parts = [f"{len(self.names)} secret{'' if len(self.names) == 1 else 's'} loaded"]
        if self.scope:
            parts.append(f"from {self.scope}")
        if self.skipped:
            parts.append(f"({len(self.skipped)} already set, kept)")
        return f"<seekrit: {' '.join(parts)}{': ' + shown if shown else ''}>"

    def _repr_html_(self) -> str:
        """Rich display for Jupyter. Names only — same guarantee as ``__repr__``."""
        names = _as_code(self.names) or "<em>none</em>"
        skipped = (
            f"<div><small>{len(self.skipped)} already set and left alone: "
            f"{_as_code(self.skipped)}</small></div>"
            if self.skipped
            else ""
        )
        scope = f" from <strong>{html.escape(self.scope)}</strong>" if self.scope else ""
        return (
            "<div>"
            f"<div>seekrit loaded <strong>{len(self.names)}</strong> "
            f"secret{'' if len(self.names) == 1 else 's'} into the environment{scope}"
            "</div>"
            f"<div>{names}</div>{skipped}"
            "<div><small>Values are in the environment, not in this output.</small></div>"
            "</div>"
        )


def load(
    token: Optional[str] = None,
    *,
    api_url: Optional[str] = None,
    overrides: Optional[Mapping[str, str]] = None,
    env: Optional[MutableMapping[str, str]] = None,
    override: bool = True,
    prompt: PromptMode = "auto",
    timeout: float = 30.0,
    interpolate: bool = True,
) -> LoadedSecrets:
    """Resolve this token's environment and load it into ``os.environ``.

    The one-call form, for the top of a notebook or script::

        import seekrit
        seekrit.load()

    Args:
        token: ``skt_...`` service token. Defaults to ``$SEEKRIT_TOKEN``, then to
            an interactive prompt (see ``prompt``).
        api_url: API base URL. Defaults to ``$SEEKRIT_API_URL``.
        overrides: ``{group_slug: env_slug}`` to pull a different environment
            slice of a composed group (the ``?with=`` override).
        env: environment to load into. Defaults to ``os.environ``.
        override: resolved secrets win over what is already set (default
            ``True``) — the opposite of :meth:`Client.into_env`, on purpose.
            Calling this is a declaration that seekrit is the source of truth,
            and it makes re-running the cell idempotent rather than a no-op that
            reports loading nothing. Pass ``False`` to keep existing values,
            which are then listed in :attr:`LoadedSecrets.skipped`.
        prompt: ``"auto"`` (default) asks for the token via :mod:`getpass` when
            none was found *and* stdin can carry it — which keeps a live
            credential out of the notebook file. ``False`` never prompts;
            ``True`` requires the prompt and raises if stdin is unavailable.
        timeout: per-request timeout in seconds.
        interpolate: expand ``${OTHER_SECRET}`` references (default ``True``).

    Returns:
        A :class:`LoadedSecrets` describing what was loaded — names and scope,
        no values, so displaying it in a cell leaks nothing.

    Raises:
        SeekritError: no token available, or the request failed.
        SeekritApiError: the API returned a non-2xx response.
        SeekritCryptoError: a token or ciphertext would not decrypt.
    """
    client = Client(
        _require_token(token, prompt),
        api_url=api_url,
        overrides=overrides,
        timeout=timeout,
        interpolate=interpolate,
    )
    secrets, scope = client._resolve_detailed()

    target = os.environ if env is None else env
    loaded, skipped = [], []
    for name, value in secrets.items():
        if override or name not in target:
            target[name] = value
            loaded.append(name)
        else:
            skipped.append(name)
    return LoadedSecrets(loaded, skipped, scope)


# -- token acquisition ----------------------------------------------------


def _require_token(token: Optional[str], prompt: PromptMode) -> str:
    """The token from the argument, the environment, or an interactive prompt."""
    found = token or os.environ.get("SEEKRIT_TOKEN")
    if found:
        return found
    prompted = _prompt_for_token(prompt)
    if prompted:
        return prompted
    raise SeekritError(
        "no service token: set SEEKRIT_TOKEN, or pass seekrit.load(token=...). "
        "In a notebook, prefer the prompt or the kernel environment over pasting "
        "a token into a cell — cell source is saved in the .ipynb."
    )


def _prompt_for_token(prompt: PromptMode) -> Optional[str]:
    """Ask for a token on stdin, or return ``None`` if we shouldn't/can't ask."""
    if prompt is False:
        return None
    if prompt == "auto" and not _stdin_can_prompt():
        return None
    try:
        return getpass.getpass("seekrit service token (skt_...): ").strip() or None
    except Exception as exc:  # noqa: BLE001 - EOF, closed stdin, or no-stdin kernel
        # papermill and other headless runners execute with stdin disabled; say
        # what to do instead of surfacing the frontend's own error.
        raise SeekritError(
            "cannot prompt for a service token here (stdin is unavailable) — "
            "set SEEKRIT_TOKEN in the environment running this kernel"
        ) from exc


def _stdin_can_prompt() -> bool:
    """True if a :mod:`getpass` prompt has somewhere to go.

    A Jupyter kernel's stdin is not a tty, but ipykernel routes ``getpass`` to
    the frontend as a password field — so an active kernel counts.
    """
    if _ipython_kernel() is not None:
        return True
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except Exception:  # noqa: BLE001 - detached or replaced stdin
        return False


def _ipython_kernel() -> Optional[object]:
    """The active IPython kernel, or ``None``. Never imports IPython itself."""
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return None
    try:
        shell = ipython.get_ipython()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - IPython present but not running
        return None
    return getattr(shell, "kernel", None)


__all__ = ["load", "LoadedSecrets"]
