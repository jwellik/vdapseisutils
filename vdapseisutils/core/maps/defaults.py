"""
Maps styling: constants and optional matplotlib rc registration.

Constants live in :mod:`vdapseisutils.core.maps.defaults_constants` and are
re-exported here for backward compatibility. Matplotlib ``rcParams`` are
**not** modified at import time; call :func:`register_maps_mpl_style` explicitly
or rely on :func:`ensure_maps_mpl_style` (invoked from ``Map``, ``CrossSection``,
``VolcanoFigure``, and ``TimeSeries`` constructors).
"""

from __future__ import annotations

from .defaults_constants import *  # noqa: F401,F403

_maps_mpl_style_registered = False


def register_maps_mpl_style(rc_name: str = "swarmmplrc") -> None:
    """
    Apply vdapseis maps matplotlib style (custom rc + svg font + base font size).

    Safe to call multiple times; repeats are cheap but re-apply the same values.
    """
    import matplotlib

    from vdapseisutils.style import load_custom_rc

    load_custom_rc(rc_name)
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.size"] = 8


def ensure_maps_mpl_style() -> None:
    """Register maps matplotlib style once per process (idempotent)."""
    global _maps_mpl_style_registered
    if not _maps_mpl_style_registered:
        register_maps_mpl_style()
        _maps_mpl_style_registered = True


def reset_maps_mpl_style_registration_for_tests() -> None:
    """Clear one-shot guard (tests only)."""
    global _maps_mpl_style_registered
    _maps_mpl_style_registered = False


def _test_defaults():
    """Simple test to verify defaults module loads correctly."""
    try:
        required_defaults = [
            "HEATMAP_DEFAULTS",
            "TICK_DEFAULTS",
            "AXES_DEFAULTS",
            "CROSSSECTION_DEFAULTS",
            "GRID_DEFAULTS",
            "TITLE_DEFAULTS",
            "SUBTITLE_DEFAULTS",
            "default_volcano",
        ]

        for default in required_defaults:
            if default not in globals():
                raise ValueError(f"Missing required default: {default}")

        required_keys = ["name", "lat", "lon", "elev"]
        for key in required_keys:
            if key not in default_volcano:
                raise ValueError(f"Missing key '{key}' in default_volcano")

        print("✓ All defaults loaded successfully")
        print(
            f"✓ Default volcano: {default_volcano['name']} at ({default_volcano['lat']}, {default_volcano['lon']})"
        )
        return True

    except Exception as e:
        print(f"✗ Defaults test failed: {e}")
        return False


if __name__ == "__main__":
    _test_defaults()
