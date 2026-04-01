"""Helpers for §8 swarm legacy import deprecation (API v1).

Emits :class:`DeprecationWarning` when user code imports deprecated ``swarmmpl*``
packages, while skipping noise for the canonical package, pyplot registration,
and same-subtree internal imports.
"""

from __future__ import annotations

import inspect
import warnings

_CANONICAL = "vdapseisutils.plot.swarm"
_REMOVAL = "Removal no earlier than vdapseisutils v0.2.0."

_SKIP_PREFIXES = (
    "vdapseisutils.plot.swarm",
    "vdapseisutils.plot.mpl",
    "vdapseisutils.core.swarmmpl",
    "vdapseisutils.core.swarmmpl2",
    "vdapseisutils.core.swarmmpl3",
)


def warn_swarm_legacy_package(*, legacy_qualname: str) -> None:
    """Warn if the import stack shows external (non-shim) code."""
    found_skip = False
    for frame in inspect.stack()[2:]:
        mod = inspect.getmodule(frame.frame)
        name = getattr(mod, "__name__", "") or ""
        if name in ("importlib._bootstrap", "importlib._bootstrap_external") or name == "":
            continue
        # Frames inside the legacy package that is currently loading (e.g. its __init__.py).
        if name == legacy_qualname or name.startswith(legacy_qualname + "."):
            continue
        if name == "vdapseisutils":
            found_skip = True
            break
        if any(name == p or name.startswith(p + ".") for p in _SKIP_PREFIXES):
            found_skip = True
            break
        break
    if found_skip:
        return
    msg = (
        f"Importing {legacy_qualname} is deprecated; prefer {_CANONICAL} instead. "
        f"{_REMOVAL}"
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
