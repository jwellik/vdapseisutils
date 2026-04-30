"""
VelocityModel1D class for plotting layered 1D velocity models.

This module provides a plotting wrapper that follows the same style as other
vdapseisutils map-stack classes (Map/CrossSection/TimeSeries): it wraps a
matplotlib axis, supports standalone or embedded figures, and offers
chainable plotting utilities.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt

try:
    from .defaults import TICK_DEFAULTS, AXES_DEFAULTS, ensure_maps_mpl_style
except ImportError:
    # Running as script - add package root to path and use absolute imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from vdapseisutils.core.maps.defaults import TICK_DEFAULTS, AXES_DEFAULTS, ensure_maps_mpl_style


class VelocityModel1D:
    """Plot one or more layered 1D Vp/Vs velocity models versus depth."""

    name = "velocity-model-1d"

    def __init__(
        self,
        fig=None,
        depth_extent_km: tuple[float, float] | None = None,
        velocity_extent_kms: tuple[float, float] | None = None,
        title: str | None = None,
        invert_depth_axis: bool = True,
        show_grid: bool = False,
        legend: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        ensure_maps_mpl_style()

        if fig is None:
            fig_kwargs = {k: v for k, v in kwargs.items() if k in ["dpi", "figsize"]}
            fig = plt.figure(**fig_kwargs)

        self.figure = fig
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in ["dpi", "figsize"]}
        self.ax = fig.add_subplot(111, **plot_kwargs)

        self.properties = {
            "depth_extent_km": depth_extent_km,
            "velocity_extent_kms": velocity_extent_kms,
            "invert_depth_axis": invert_depth_axis,
            "show_grid": show_grid,
            "legend": legend,
            "title": title,
            "verbose": verbose,
        }

        for spine in self.ax.spines.values():
            spine.set_linewidth(AXES_DEFAULTS["spine_linewidth"])

        self.ax.tick_params(
            axis="both",
            labelcolor=TICK_DEFAULTS["labelcolor"],
            labelsize=TICK_DEFAULTS["labelsize"],
            color=TICK_DEFAULTS["tick_color"],
            length=TICK_DEFAULTS["tick_size"],
            width=TICK_DEFAULTS["tick_width"],
            direction=TICK_DEFAULTS["tick_direction"],
            pad=TICK_DEFAULTS["tick_pad"],
            left=True,
            labelleft=True,
            bottom=True,
            labelbottom=True,
            right=False,
            labelright=False,
            top=False,
            labeltop=False,
        )
        self.ax.set_xlabel("Velocity (km/s)")
        self.ax.xaxis.set_label_position("bottom")
        self.ax.set_ylabel("Depth (km)")
        self.ax.grid(show_grid)
        if title:
            self.ax.set_title(title)
        if invert_depth_axis:
            self.ax.invert_yaxis()
        if depth_extent_km is not None:
            self.set_depth_extent(depth_extent_km)
        if velocity_extent_kms is not None:
            self.set_velocity_extent(velocity_extent_kms)

    def plot_model(
        self,
        model,
        *,
        label: str | None = None,
        vp_key: str = "vp_kms",
        vs_key: str = "vs_kms",
        top_key: str = "z_top_km",
        bottom_key: str = "z_bot_km",
        colors: tuple[str, str] = ("tab:blue", "tab:red"),
        linewidth: float = 2.0,
        alpha: float = 1.0,
        drawstyle: str = "steps-post",
        vp_label: str = "Vp",
        vs_label: str = "Vs",
    ):
        """Plot one model with Vp and Vs layered step curves."""
        arr = self._normalize_model(model, top_key, bottom_key, vp_key, vs_key)
        x_vp, y_vp = self._to_step_xy(arr[:, 2], arr[:, 0], arr[:, 1])
        x_vs, y_vs = self._to_step_xy(arr[:, 3], arr[:, 0], arr[:, 1])

        prefix = f"{label} " if label else ""
        self.ax.plot(
            x_vp,
            y_vp,
            color=colors[0],
            linewidth=linewidth,
            alpha=alpha,
            drawstyle=drawstyle,
            label=f"{prefix}{vp_label}",
        )
        self.ax.plot(
            x_vs,
            y_vs,
            color=colors[1],
            linewidth=linewidth,
            alpha=alpha,
            drawstyle=drawstyle,
            label=f"{prefix}{vs_label}",
        )

        self._autoscale_if_needed(arr)
        self._finalize_axis_state()
        return self

    def compare_models(
        self,
        models: Sequence,
        *,
        labels: Sequence[str] | None = None,
        model_colors: Sequence[str] | None = None,
        phase_styles: Mapping[str, str] | None = None,
        linewidth: float = 1.8,
        alpha: float = 0.95,
        deduplicate_legend: bool = True,
    ):
        """Overlay multiple models for side-by-side comparison."""
        labels = labels or [None] * len(models)
        phase_styles = phase_styles or {"vp": "-", "vs": "-"}
        model_colors = model_colors or [None] * len(models)

        for i, (model, label) in enumerate(zip(models, labels)):
            if model_colors[i] is None:
                colors = ("tab:blue", "tab:red")
            else:
                colors = (model_colors[i], model_colors[i])

            self.plot_model(
                model,
                label=label,
                colors=colors,
                linewidth=linewidth,
                alpha=alpha,
            )
            lines = self.ax.get_lines()[-2:]
            lines[0].set_linestyle(phase_styles.get("vp", "-"))
            lines[1].set_linestyle(phase_styles.get("vs", "-"))

        if deduplicate_legend:
            self._dedupe_legend()
        self._finalize_axis_state()
        return self

    def set_depth_extent(self, depth_extent_km: tuple[float, float]):
        """Set y-axis depth extent."""
        self.ax.set_ylim(depth_extent_km)
        # Preserve intended "depth increases downward" convention after explicit limits.
        if self.properties.get("invert_depth_axis", False):
            self.ax.invert_yaxis()
        return self

    def set_velocity_extent(self, velocity_extent_kms: tuple[float, float]):
        """Set x-axis velocity extent."""
        self.ax.set_xlim(velocity_extent_kms)
        return self

    def clear(self):
        """Clear axis and re-apply labels."""
        self.ax.cla()
        self.ax.set_xlabel("Velocity (km/s)")
        self.ax.xaxis.set_label_position("bottom")
        self.ax.set_ylabel("Depth (km)")
        return self

    def _normalize_model(
        self,
        model,
        top_key: str,
        bottom_key: str,
        vp_key: str,
        vs_key: str,
    ) -> np.ndarray:
        """
        Convert model input to Nx4 ndarray:
        [z_top_km, z_bot_km, vp_kms, vs_kms].
        """
        if hasattr(model, "loc") and hasattr(model, "columns"):
            vals = model[[top_key, bottom_key, vp_key, vs_key]].to_numpy(dtype=float)
            return vals

        rows = []
        for row in model:
            rows.append([row[top_key], row[bottom_key], row[vp_key], row[vs_key]])
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _to_step_xy(v: np.ndarray, z_top: np.ndarray, z_bot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.repeat(v, 2)
        y = np.column_stack([z_top, z_bot]).ravel()
        return x, y

    def _autoscale_if_needed(self, arr: np.ndarray):
        if self.properties["velocity_extent_kms"] is None:
            vmin = np.nanmin(arr[:, [2, 3]])
            vmax = np.nanmax(arr[:, [2, 3]])
            pad = 0.05 * max(1e-6, vmax - vmin)
            self.ax.set_xlim(vmin - pad, vmax + pad)

        if self.properties["depth_extent_km"] is None:
            zmin = np.nanmin(arr[:, 0])
            zmax = np.nanmax(arr[:, 1])
            self.ax.set_ylim(zmin, zmax)
            if self.properties["invert_depth_axis"]:
                self.ax.invert_yaxis()

    def _dedupe_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        self.ax.legend(seen.values(), seen.keys())

    def _finalize_axis_state(self):
        if self.properties["legend"]:
            self._dedupe_legend()
        return self
