"""
Register VDAPSeisUtils figure constructors on ``matplotlib.pyplot`` (API v1 §3).

After calling :func:`register_pyplot`, ``plt.helicorder``, ``plt.clipboard``,
``plt.eqmap``, ``plt.volcano``, and ``plt.swarm`` construct the corresponding
project figure types. See ``.local/api-v1-coord/API_V1_CANONICAL.md`` §3.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import matplotlib.pyplot as plt

# Stable callables assigned to pyplot; built once so repeat registration is idempotent.
_wrappers: Tuple[Callable[..., Any], ...] | None = None


def _build_pyplot_constructors() -> Tuple[Callable[..., Any], ...]:
    from vdapseisutils.core.maps.map import Map
    from vdapseisutils.core.maps.volcano_figure import VolcanoFigure
    from vdapseisutils.plot.swarm import Clipboard, ClipboardClass
    from vdapseisutils.plot.swarm import Helicorder

    def helicorder(*args, **kwargs):
        return Helicorder(*args, **kwargs)

    helicorder.__name__ = "helicorder"
    helicorder.__doc__ = Helicorder.__doc__
    helicorder.__qualname__ = "helicorder"

    def clipboard(*args, **kwargs):
        return Clipboard(*args, **kwargs)

    clipboard.__name__ = "clipboard"
    clipboard.__qualname__ = "clipboard"
    clipboard.__doc__ = (
        Clipboard.__doc__
        or ClipboardClass.__doc__
        or "Swarm-style multi-trace figure (Clipboard); see ClipboardClass."
    )

    def eqmap(*args, **kwargs):
        return Map(*args, **kwargs)

    eqmap.__name__ = "eqmap"
    eqmap.__qualname__ = "eqmap"
    eqmap.__doc__ = Map.__doc__

    def volcano(*args, **kwargs):
        return VolcanoFigure(*args, **kwargs)

    volcano.__name__ = "volcano"
    volcano.__qualname__ = "volcano"
    volcano.__doc__ = VolcanoFigure.__doc__

    def swarm(*args, **kwargs):
        return clipboard(*args, **kwargs)

    swarm.__name__ = "swarm"
    swarm.__qualname__ = "swarm"
    swarm.__doc__ = (
        "``plt.swarm`` constructor (API v1 §3).\n\n"
        "Returns a :class:`vdapseisutils.plot.swarm.SwarmFigure` instance "
        "(alias of :class:`vdapseisutils.plot.swarm.ClipboardClass`), same "
        "backend as ``plt.clipboard`` "
        "(:func:`vdapseisutils.plot.swarm.Clipboard`)."
    )

    return (helicorder, clipboard, eqmap, volcano, swarm)


def register_pyplot() -> None:
    """Attach VDAPSeisUtils constructors to ``matplotlib.pyplot``.

    Safe to call multiple times: later calls re-apply the same callables.

    Adds:

    - ``plt.helicorder`` → :class:`vdapseisutils.plot.swarm.Helicorder`
    - ``plt.clipboard`` → same object as :func:`vdapseisutils.plot.swarm.Clipboard`
    - ``plt.eqmap`` → :class:`vdapseisutils.core.maps.map.Map`
    - ``plt.volcano`` → :class:`vdapseisutils.core.maps.volcano_figure.VolcanoFigure`
    - ``plt.swarm`` → same as ``plt.clipboard``; returns :class:`vdapseisutils.plot.swarm.SwarmFigure`

    Canonical spec: ``API_V1_CANONICAL.md`` §3.
    """
    global _wrappers
    if _wrappers is None:
        _wrappers = _build_pyplot_constructors()

    helicorder, clipboard, eqmap, volcano, swarm = _wrappers
    plt.helicorder = helicorder
    plt.clipboard = clipboard
    plt.eqmap = eqmap
    plt.volcano = volcano
    plt.swarm = swarm
