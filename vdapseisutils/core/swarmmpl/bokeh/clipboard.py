"""
Bokeh :class:`SwarmClipboardBk` — aligned with :class:`SwarmClipboard`.

**Phase 1:** ``mode="w"``, ``tick_type`` absolute / relative, ``sync_waves``.
**Phase 2:** ``mode="g"`` (spectrogram only), ``mode="wg"`` (waveform + spectrogram),
using :func:`~vdapseisutils.compute.waveforms.compute_spectrogram`, inferno-style palette,
and optional color bar. HTML export via :meth:`SwarmClipboardBk.save`.
**Phase 3:** ``axvline``, ``plot_peak_value``, ``plot_trace`` / ``plot_horizontals``,
``plot_catalog``, ``scroll_traces``, ``set_alim`` / ``set_flim``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from bokeh.io import save
from bokeh.layouts import column
from bokeh.models import BoxZoomTool, ColorBar, LinearColorMapper, Range1d, Span, WheelZoomTool
from bokeh.plotting import figure as bk_figure
from bokeh.resources import Resources
from matplotlib.colors import to_hex
from obspy import UTCDateTime
from obspy import Stream, Trace
from obspy.core.event import Catalog, Event

from vdapseisutils.compute.waveforms import SpectrogramResult, compute_spectrogram, prepare_waveform_series
from vdapseisutils.style import colors as vdap_colors

_DEFAULT_WIDTH_PX = 850
_DEFAULT_PANEL_HEIGHT_PX = 220

_SPEC_DEFAULTS: dict[str, Any] = {
    "wlen": 2.0,
    "overlap": 0.86,
    "dbscale": True,
    "samp_rate": None,
    "log_power": False,
    "cmap": None,
}


def _normalize_tick_type(raw: str) -> str:
    """Map Swarm / legacy names onto ``absolute`` or ``relative``."""
    k = str(raw).strip().lower()
    if k in ("absolute", "datetime", "time"):
        return "absolute"
    if k in ("relative", "relative_seconds", "seconds"):
        return "relative"
    raise ValueError(
        f"Unknown tick_type {raw!r}; use absolute/datetime/time or relative."
    )


def _normalize_mode(raw: str) -> str:
    m = str(raw).strip().lower()
    if m in ("w", "g", "wg"):
        return m
    raise ValueError(f'Unknown mode {raw!r}; use "w", "g", or "wg".')


def _inferno_u_palette(n: int = 256) -> list[str]:
    """Sample :data:`vdapseisutils.style.colors.inferno_u` to Bokeh hex palette."""
    xs = np.linspace(0.0, 1.0, int(n))
    return [to_hex(vdap_colors.inferno_u(float(x))) for x in xs]


def _mpl_color_to_bokeh(color: Any) -> str:
    """Matplotlib colors (short names like ``'k'``, tuples, …) → CSS hex for Bokeh."""
    if color is None:
        return "#000000"
    try:
        return to_hex(color)
    except (ValueError, TypeError):
        try:
            import matplotlib.colors as mcolors

            return to_hex(mcolors.to_rgb(color))
        except Exception:
            return "#000000"


def _palette_from_cmap(cmap: Any) -> list[str]:
    """Matplotlib colormap or None → Bokeh palette list."""
    if cmap is None:
        return _inferno_u_palette()
    if callable(cmap):
        xs = np.linspace(0.0, 1.0, 256)
        return [to_hex(cmap(float(x))) for x in xs]
    key = str(cmap).lower()
    if key in ("inferno", "inferno_u"):
        return _inferno_u_palette()
    try:
        import matplotlib.pyplot as plt

        mpl_cm = plt.get_cmap(cmap)
        return [to_hex(mpl_cm(x)) for x in np.linspace(0.0, 1.0, 256)]
    except Exception:
        return _inferno_u_palette()


def _trace_times_absolute_ms(ser) -> list[float]:
    """Absolute sample times as Bokeh datetime axis coordinates (ms since epoch)."""
    start = ser.starttime.datetime
    out: list[float] = []
    for t_s in ser.times_s:
        dt = start + timedelta(seconds=float(t_s))
        out.append(dt.timestamp() * 1000.0)
    return out


def _trace_metadata(tr: Trace) -> dict[str, Any]:
    return {
        "network": tr.stats.network,
        "station": tr.stats.station,
        "location": tr.stats.location,
        "channel": tr.stats.channel,
        "id": tr.id,
        "nslc": (
            f"{tr.stats.network}.{tr.stats.station}."
            f"{tr.stats.location}.{tr.stats.channel}"
        ),
        "net_sta": f"{tr.stats.network}.{tr.stats.station}",
        "sta": tr.stats.station,
    }


@dataclass
class _PanelRecord:
    """One clipboard station/panel: trace metadata + optional wave/spec Bokeh figures."""

    index: int
    trace: Trace
    metadata: dict[str, Any]
    wave_figure: Any | None = None
    spec_figure: Any | None = None


def _record_matches_metadata(rec: _PanelRecord, **criteria: Any) -> bool:
    md = rec.metadata
    for key, value in criteria.items():
        if key not in md:
            return False
        if isinstance(value, (list, tuple)):
            if md[key] not in value:
                return False
        else:
            if md[key] != value:
                return False
    return True


def _restrict_zoom_to_x_axis(fig: Any) -> None:
    """Wheel zoom / box zoom affect only the x (time) axis; y limits stay fixed."""
    for tool in fig.toolbar.tools:
        if isinstance(tool, (WheelZoomTool, BoxZoomTool)):
            tool.dimensions = "width"


def _span_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Matplotlib-style line kwargs → :class:`~bokeh.models.Span` props."""
    m: dict[str, Any] = {}
    if "color" in kwargs:
        m["line_color"] = _mpl_color_to_bokeh(kwargs["color"])
    if "lw" in kwargs:
        m["line_width"] = kwargs["lw"]
    elif "linewidth" in kwargs:
        m["line_width"] = kwargs["linewidth"]
    if "ls" in kwargs:
        m["line_dash"] = kwargs["ls"]
    elif "linestyle" in kwargs:
        m["line_dash"] = kwargs["linestyle"]
    if "alpha" in kwargs:
        m["line_alpha"] = kwargs["alpha"]
    return m


class SwarmClipboardBk:
    """
    Interactive multi-panel clipboard built with Bokeh.

    Parameters mirror :class:`~vdapseisutils.core.swarmmpl.clipboard.SwarmClipboard`.
    Unknown ``mode`` / ``tick_type`` raises ``ValueError``; spectrogram failure on a trace
    shows a placeholder figure for that panel.

    **Toolbar / zoom:** ``toolbar_location`` places the tool palette (``'above'``, ``'below'``,
    ``'left'``, ``'right'`` — default ``'right'``). With ``zoom_x_only=True`` (default),
    wheel zoom and box zoom only stretch the time axis; amplitude / frequency limits stay
    fixed unless you change them (``set_alim`` / ``set_flim`` / manual y-range).

    **Wave + spectrogram gap:** In ``mode='wg'``, ``wave_spec_spacing`` is the pixel gap
    between the waveform and spectrogram sub-figures in the column (default ``0``).
    """

    def __init__(
        self,
        data: Stream | list[Trace] | Trace | None = None,
        *,
        sync_waves: bool = True,
        figsize: tuple[float, float] = (10.0, 12.0),
        panel_height: float | None = None,
        tick_type: str = "absolute",
        mode: str = "w",
        wave_settings: dict[str, Any] | None = None,
        spec_settings: dict[str, Any] | None = None,
        panel_spacing: float = 0.02,
        title_space: float = 0.10,
        width_px: int | None = None,
        panel_height_px: int | None = None,
        toolbar_location: str = "right",
        zoom_x_only: bool = True,
        wave_spec_spacing: int = 0,
    ) -> None:
        self.sync_waves = sync_waves
        self.tick_type = tick_type
        self._tick = _normalize_tick_type(tick_type)
        self.mode = _normalize_mode(mode)
        self.wave_settings = dict(wave_settings or {})
        self.spec_settings = {**_SPEC_DEFAULTS, **(spec_settings or {})}
        self._panel_spacing = panel_spacing
        self._title_space = title_space

        traces = self._normalize_traces(data)
        self._traces = traces

        w_px = int(width_px or _DEFAULT_WIDTH_PX)
        n = max(len(traces), 1)
        if panel_height_px is not None:
            h_px = int(panel_height_px)
        elif panel_height is not None:
            h_px = max(int(float(panel_height) * 96), 120)
        else:
            h_px = max(_DEFAULT_PANEL_HEIGHT_PX, int(figsize[1] * 96 / n))

        self._width_px = w_px
        self._panel_height_px = h_px
        # Wave / spectrogram vertical split for mode "wg" (MPL height_ratios ~ 1:3).
        self._wave_frac = 0.25
        self._spec_frac = 0.75

        self._toolbar_location = str(toolbar_location).lower()
        self._zoom_x_only = bool(zoom_x_only)
        self._wave_spec_spacing = int(wave_spec_spacing)

        self._panels: list[_PanelRecord] = []
        self.figures: list[Any] = []
        self.layout = column(children=[], sizing_mode="stretch_width")
        self._build_layout()

    def _bokeh_figure(self, **kwargs: Any) -> Any:
        """
        Create a :func:`bokeh.plotting.figure` with clipboard toolbar defaults.

        ``toolbar_location`` is ``'above'``, ``'below'``, ``'left'``, or ``'right'``.
        When ``zoom_x_only`` is True, wheel zoom and box zoom only change the x-range.
        """
        kwargs.setdefault("toolbar_location", self._toolbar_location)
        fig = bk_figure(**kwargs)
        if self._zoom_x_only:
            _restrict_zoom_to_x_axis(fig)
        return fig

    @staticmethod
    def _normalize_traces(data: Stream | list[Trace] | Trace | None) -> list[Trace]:
        if data is None:
            return []
        if isinstance(data, Trace):
            return [data]
        if isinstance(data, Stream):
            return [tr.copy() for tr in data]
        return [tr.copy() if isinstance(tr, Trace) else tr for tr in data]

    def _resolved_spec_kwargs(self) -> dict[str, Any]:
        k = dict(self.spec_settings)
        k.setdefault("wlen", _SPEC_DEFAULTS["wlen"])
        k.setdefault("overlap", _SPEC_DEFAULTS["overlap"])
        k.setdefault("dbscale", _SPEC_DEFAULTS["dbscale"])
        return k

    def _compute_spec_safe(self, tr: Trace) -> SpectrogramResult | None:
        kw = self._resolved_spec_kwargs()
        try:
            return compute_spectrogram(
                tr.copy(),
                samp_rate=kw.get("samp_rate"),
                wlen=float(kw["wlen"]),
                overlap=float(kw["overlap"]),
                dbscale=bool(kw["dbscale"]),
            )
        except (ValueError, ZeroDivisionError):
            return None

    def _empty_figure(self) -> Any:
        if self._tick == "absolute":
            fig = self._bokeh_figure(
                width=self._width_px,
                height=max(self._panel_height_px, 160),
                title="SwarmClipboardBk — no data",
                x_axis_type="datetime",
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
        else:
            fig = self._bokeh_figure(
                width=self._width_px,
                height=max(self._panel_height_px, 160),
                title="SwarmClipboardBk — no data",
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            fig.xaxis.axis_label = "Time (s)"
        fig.line([], [])
        return fig

    def _compute_shared_range(self, traces: list[Trace]) -> Range1d | None:
        if not self.sync_waves or not traces:
            return None
        if self._tick == "absolute":
            starts_ms: list[float] = []
            ends_ms: list[float] = []
            for tr in traces:
                ser = prepare_waveform_series(tr, relative_offset_s=0.0)
                xs = _trace_times_absolute_ms(ser)
                if xs:
                    starts_ms.append(xs[0])
                    ends_ms.append(xs[-1])
            if not starts_ms or not ends_ms:
                return None
            return Range1d(start=min(starts_ms), end=max(ends_ms), bounds=None)
        t0 = min(tr.stats.starttime for tr in traces)
        t1 = max(tr.stats.endtime for tr in traces)
        span_s = float(t1 - t0)
        return Range1d(start=0.0, end=span_s, bounds=None)

    def _wave_xy(
        self,
        tr: Trace,
        *,
        t0_global,
    ) -> tuple[list[float], list[float]]:
        ser = prepare_waveform_series(tr, relative_offset_s=0.0)
        ys = list(ser.amplitudes)
        ts = np.asarray(ser.times_s, dtype=float)

        if self._tick == "absolute":
            xs = _trace_times_absolute_ms(ser)
            return xs, ys

        if self.sync_waves:
            if t0_global is None:
                raise RuntimeError("sync_waves requires a global reference time.")
            offset_s = float(tr.stats.starttime - t0_global)
            xs = (ts + offset_s).tolist()
        else:
            xs = ts.tolist()
        return xs, ys

    def _per_trace_range(
        self,
        tr: Trace,
        xs: list[float],
        shared_range: Range1d | None,
    ) -> Range1d:
        if self.sync_waves:
            if shared_range is not None:
                return shared_range
            return Range1d(start=0.0, end=1.0)
        if self._tick == "absolute":
            return Range1d(start=xs[0], end=xs[-1])
        dur = float(tr.stats.endtime - tr.stats.starttime)
        return Range1d(start=0.0, end=dur)

    def _spectrogram_extent(
        self,
        tr: Trace,
        spec: SpectrogramResult,
        *,
        t0_global,
    ) -> tuple[float, float, float, float, str]:
        """
        Return ``x_left``, ``dw``, ``y_bottom``, ``dh``, ``x_axis_type_label``.

        X uses the same convention as waveform data (datetime ms or seconds).
        """
        times = np.asarray(spec.times_s, dtype=float)
        if times.size == 0:
            return 0.0, 1.0, 0.1, 1.0, "datetime" if self._tick == "absolute" else "linear"
        if times.size == 1:
            dt = float(tr.stats.endtime - tr.stats.starttime)
        else:
            dt = float(np.median(np.diff(times)))

        if self._tick == "absolute":
            t_left = tr.stats.starttime + float(times[0]) - dt / 2.0
            t_right = tr.stats.starttime + float(times[-1]) + dt / 2.0
            left_ms = float(t_left.timestamp) * 1000.0
            dw_ms = float(t_right.timestamp - t_left.timestamp) * 1000.0
            return left_ms, dw_ms, 0.0, 0.0, "datetime"

        offset_s = (
            float(tr.stats.starttime - t0_global)
            if (self.sync_waves and t0_global is not None)
            else 0.0
        )
        left_s = offset_s + float(times[0]) - dt / 2.0
        right_s = offset_s + float(times[-1]) + dt / 2.0
        return left_s, right_s - left_s, 0.0, 0.0, "linear"

    def _spectrogram_freq_extent(self, spec: SpectrogramResult, tr: Trace) -> tuple[float, float]:
        freqs = np.asarray(spec.frequencies_hz, dtype=float)
        if freqs.size == 0:
            nyq = float(spec.sampling_rate_hz) / 2.0
            y0 = 0.01
            return y0, max(nyq - y0, 0.1)
        df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
        y0 = max(float(freqs[0]) - df / 2.0, 0.01)
        nyq = float(spec.sampling_rate_hz) / 2.0
        y1 = min(float(freqs[-1]) + df / 2.0, nyq)
        return y0, y1 - y0

    def _figure_spectrogram(
        self,
        tr: Trace,
        *,
        shared_x_range: Range1d | None,
        title: str | None,
        height_px: int,
        hide_x_labels: bool,
    ) -> Any:
        spec = self._compute_spec_safe(tr)
        kw = self._resolved_spec_kwargs()
        log_power = bool(kw.get("log_power", False))
        palette = _palette_from_cmap(kw.get("cmap"))

        x_axis_type = "datetime" if self._tick == "absolute" else "linear"
        x_label = "" if hide_x_labels else ("Time" if self._tick == "absolute" else "Time (s)")

        if spec is None:
            return self._bokeh_figure(
                width=self._width_px,
                height=height_px,
                title=(title or tr.id) + " (spectrogram unavailable)",
                x_axis_type=x_axis_type,
                x_range=shared_x_range,
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )

        t0_global = min(tt.stats.starttime for tt in self._traces) if self._traces else None
        x_left, dw, _yb, _ydh, _ = self._spectrogram_extent(tr, spec, t0_global=t0_global)

        if shared_x_range is not None:
            xr = shared_x_range
        else:
            xr = Range1d(start=x_left, end=x_left + dw)

        y0, ydh = self._spectrogram_freq_extent(spec, tr)

        z = np.asarray(spec.power, dtype=float)
        if z.size == 0:
            fig = self._bokeh_figure(
                width=self._width_px,
                height=height_px,
                title=title or tr.id,
                x_axis_type=x_axis_type,
                x_range=xr,
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            return fig

        finite = np.isfinite(z)
        if finite.any():
            lo = float(np.nanmin(z))
            hi = float(np.nanmax(z))
            if lo >= hi:
                lo, hi = lo - 1.0, hi + 1.0
        else:
            lo, hi = 0.0, 1.0

        mapper = LinearColorMapper(palette=palette, low=lo, high=hi, nan_color="#00000000")

        fig = self._bokeh_figure(
            width=self._width_px,
            height=height_px,
            title=title or tr.id,
            x_axis_type=x_axis_type,
            x_range=xr,
            y_axis_type="log" if log_power else "linear",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )

        fig.image(
            image=[z],
            x=x_left,
            y=y0,
            dw=dw,
            dh=ydh,
            color_mapper=mapper,
        )

        fig.yaxis.axis_label = "Frequency (Hz)"
        fig.xaxis.axis_label = x_label
        if hide_x_labels:
            fig.xaxis.major_label_text_font_size = "0pt"
            fig.xaxis.axis_label = ""

        color_bar = ColorBar(
            color_mapper=mapper,
            label_standoff=12,
            border_line_color=None,
            location="right",
            title="dB" if kw.get("dbscale") else "Power",
        )
        fig.add_layout(color_bar, "right")

        return fig

    def _figure_waveform(
        self,
        tr: Trace,
        *,
        xs: list[float],
        ys: list[float],
        xr: Range1d,
        height_px: int,
        hide_x_labels: bool,
        title: str | None,
    ) -> Any:
        color = _mpl_color_to_bokeh(self.wave_settings.get("color", "black"))
        x_axis_type = "datetime" if self._tick == "absolute" else "linear"
        x_label = "" if hide_x_labels else ("Time" if self._tick == "absolute" else "Time (s)")

        fig = self._bokeh_figure(
            width=self._width_px,
            height=height_px,
            title=title or tr.id,
            x_axis_type=x_axis_type,
            x_range=xr,
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            active_scroll="wheel_zoom",
        )
        fig.line(xs, ys, color=color, line_width=1)
        fig.yaxis.axis_label = "Amplitude"
        fig.xaxis.axis_label = x_label
        if hide_x_labels:
            fig.xaxis.major_label_text_font_size = "0pt"
            fig.xaxis.axis_label = ""
        return fig

    def _build_layout(self) -> None:
        traces = self._traces
        children: list[Any] = []

        if not traces:
            self._panels = []
            fig = self._empty_figure()
            self.figures = [fig]
            self.layout.children = [fig]
            return

        t0_global = min(tr.stats.starttime for tr in traces) if traces else None
        shared_range = self._compute_shared_range(traces)

        wave_h = max(int(self._panel_height_px * self._wave_frac), 72)
        spec_h = max(int(self._panel_height_px * self._spec_frac), 96)

        self._panels = []
        pidx = 0
        for tr in traces:
            xs, ys = self._wave_xy(tr, t0_global=t0_global)
            if not xs:
                xs, ys = [0.0], [0.0]

            xr_wave = self._per_trace_range(tr, xs, shared_range)

            if self.mode == "w":
                fig = self._figure_waveform(
                    tr,
                    xs=xs,
                    ys=ys,
                    xr=xr_wave,
                    height_px=self._panel_height_px,
                    hide_x_labels=False,
                    title=tr.id,
                )
                self._panels.append(
                    _PanelRecord(
                        index=pidx,
                        trace=tr,
                        metadata=_trace_metadata(tr),
                        wave_figure=fig,
                        spec_figure=None,
                    )
                )
                pidx += 1
                self.figures.append(fig)
                children.append(fig)
                continue

            if self.mode == "g":
                fig_g = self._figure_spectrogram(
                    tr,
                    shared_x_range=shared_range if self.sync_waves else None,
                    title=tr.id,
                    height_px=self._panel_height_px,
                    hide_x_labels=False,
                )
                self._panels.append(
                    _PanelRecord(
                        index=pidx,
                        trace=tr,
                        metadata=_trace_metadata(tr),
                        wave_figure=None,
                        spec_figure=fig_g,
                    )
                )
                pidx += 1
                self.figures.append(fig_g)
                children.append(fig_g)
                continue

            # mode == "wg"
            wave_fig = self._figure_waveform(
                tr,
                xs=xs,
                ys=ys,
                xr=xr_wave,
                height_px=wave_h,
                hide_x_labels=True,
                title=tr.id,
            )
            spec_fig = self._figure_spectrogram(
                tr,
                shared_x_range=wave_fig.x_range,
                title="",
                height_px=spec_h,
                hide_x_labels=False,
            )
            panel = column(
                [wave_fig, spec_fig],
                sizing_mode="stretch_width",
                spacing=self._wave_spec_spacing,
            )
            self._panels.append(
                _PanelRecord(
                    index=pidx,
                    trace=tr,
                    metadata=_trace_metadata(tr),
                    wave_figure=wave_fig,
                    spec_figure=spec_fig,
                )
            )
            pidx += 1
            self.figures.append(panel)
            children.append(panel)

        self.layout.children = children

    def _t0_global(self) -> UTCDateTime | None:
        if not self._traces:
            return None
        return min(tr.stats.starttime for tr in self._traces)

    def _resolve_target_panels(
        self,
        panels: list[int] | int | None,
        stations,
        networks,
        ids,
        metadata,
    ) -> list[_PanelRecord]:
        if panels is not None:
            if isinstance(panels, int):
                return [self._panels[panels]]
            return [self._panels[i] for i in panels]
        out: list[_PanelRecord] = []
        for rec in self._panels:
            if stations is not None:
                sl = stations if isinstance(stations, (list, tuple)) else [stations]
                if rec.metadata.get("sta") not in sl and rec.metadata.get("station") not in sl:
                    continue
            if networks is not None:
                nl = networks if isinstance(networks, (list, tuple)) else [networks]
                if rec.metadata.get("network") not in nl:
                    continue
            if ids is not None:
                il = ids if isinstance(ids, (list, tuple)) else [ids]
                if rec.metadata.get("id") not in il:
                    continue
            if metadata is not None:
                if not _record_matches_metadata(rec, **metadata):
                    continue
            out.append(rec)
        if stations is None and networks is None and ids is None and metadata is None:
            return list(self._panels)
        return out

    def _figures_for_axes(
        self, rec: _PanelRecord, axes: list[int] | int | None
    ) -> list[Any]:
        if axes is None:
            if self.mode == "w":
                return [rec.wave_figure] if rec.wave_figure is not None else []
            if self.mode == "g":
                return [rec.spec_figure] if rec.spec_figure is not None else []
            out: list[Any] = []
            if rec.wave_figure is not None:
                out.append(rec.wave_figure)
            if rec.spec_figure is not None:
                out.append(rec.spec_figure)
            return out
        idxs = [axes] if isinstance(axes, int) else list(axes)
        out2: list[Any] = []
        if self.mode == "w":
            if 0 in idxs and rec.wave_figure is not None:
                out2.append(rec.wave_figure)
            return out2
        if self.mode == "g":
            if 0 in idxs and rec.spec_figure is not None:
                out2.append(rec.spec_figure)
            return out2
        if 0 in idxs and rec.wave_figure is not None:
            out2.append(rec.wave_figure)
        if 1 in idxs and rec.spec_figure is not None:
            out2.append(rec.spec_figure)
        return out2

    def _figures_for_scroll(self, rec: _PanelRecord) -> list[Any]:
        seen: set[int] = set()
        out: list[Any] = []
        for fig in self._figures_for_axes(rec, None):
            key = id(fig.x_range)
            if key not in seen:
                seen.add(key)
                out.append(fig)
        return out

    def _time_input_to_utc(
        self, rec: _PanelRecord, time_input: Any, t_units: str
    ) -> UTCDateTime:
        if isinstance(time_input, UTCDateTime):
            return time_input
        tu = str(t_units).lower()
        if tu == "absolute" or isinstance(time_input, (str, datetime)):
            if isinstance(time_input, str):
                return UTCDateTime(time_input)
            if isinstance(time_input, datetime):
                return UTCDateTime(time_input)
            return UTCDateTime(time_input)
        start = rec.trace.stats.starttime
        v = float(time_input)
        if tu == "seconds":
            return start + v
        if tu == "minutes":
            return start + timedelta(minutes=v)
        if tu == "hours":
            return start + timedelta(hours=v)
        raise ValueError(f"Unknown t_units: {t_units}")

    def _utc_to_x(self, rec: _PanelRecord, utcdt: UTCDateTime) -> float:
        if self._tick == "absolute":
            return float(utcdt.timestamp) * 1000.0
        t0 = self._t0_global()
        if self.sync_waves and t0 is not None:
            return float(utcdt - t0)
        return float(utcdt - rec.trace.stats.starttime)

    def _scroll_delta(self, seconds: float) -> float:
        if self._tick == "absolute":
            return -float(seconds) * 1000.0
        return -float(seconds)

    def set_alim(self, alim: tuple[float, float] | list[float]) -> SwarmClipboardBk:
        """Set y-axis limits on all waveform panels."""
        lo, hi = float(alim[0]), float(alim[1])
        for rec in self._panels:
            if rec.wave_figure is not None:
                rec.wave_figure.y_range.start = lo
                rec.wave_figure.y_range.end = hi
        return self

    def set_flim(self, flim: tuple[float, float] | list[float]) -> SwarmClipboardBk:
        """Set frequency y-axis limits on all spectrogram panels."""
        lo, hi = float(flim[0]), float(flim[1])
        for rec in self._panels:
            if rec.spec_figure is not None:
                rec.spec_figure.y_range.start = lo
                rec.spec_figure.y_range.end = hi
        return self

    def scroll_traces(self, idx: list[int], seconds: list[float]) -> SwarmClipboardBk:
        """
        Shift x-ranges by ``-seconds`` (legacy Swarm-style: positive seconds moves the
        view opposite the waveform motion). Uses seconds on relative axes and
        milliseconds on absolute datetime axes.
        """
        if not isinstance(idx, list) or not isinstance(seconds, list):
            raise ValueError(
                "Trace index and scroll seconds must be provided as lists of the same size."
            )
        if len(idx) != len(seconds):
            raise ValueError(
                "Trace index and scroll seconds must be provided as lists of the same size."
            )
        for i, sec in zip(idx, seconds):
            if i < 0 or i >= len(self._panels):
                continue
            delta = self._scroll_delta(sec)
            rec = self._panels[i]
            for fig in self._figures_for_scroll(rec):
                xr = fig.x_range
                xr.start = xr.start + delta
                xr.end = xr.end + delta
        return self

    def axvline(
        self,
        time_input: Any,
        *,
        panels: list[int] | int | None = None,
        axes: list[int] | int | None = None,
        t_units: str = "absolute",
        stations=None,
        networks=None,
        ids=None,
        metadata=None,
        **kwargs: Any,
    ) -> SwarmClipboardBk:
        """Vertical span lines on selected panels (metadata filters mirror :class:`SwarmClipboard`)."""
        seq = time_input if isinstance(time_input, (list, tuple, np.ndarray)) else [time_input]
        span_kw = _span_kwargs(kwargs)
        targets = self._resolve_target_panels(
            panels, stations, networks, ids, metadata
        )
        for rec in targets:
            for t_in in seq:
                utcdt = self._time_input_to_utc(rec, t_in, t_units)
                x = self._utc_to_x(rec, utcdt)
                for fig in self._figures_for_axes(rec, axes):
                    fig.add_layout(
                        Span(location=x, dimension="height", **span_kw),
                    )
        return self

    def plot_trace(
        self,
        data: Trace | Stream,
        *,
        stations=None,
        networks=None,
        ids=None,
        metadata=None,
        panels=None,
        axes: list[int] | int | None = None,
        zorder: int = -1,
        **kwargs: Any,
    ) -> SwarmClipboardBk:
        """Overlay extra traces on matching station panels (same x-axis convention as the layout)."""
        invalid_kwargs = {
            "mode",
            "sync_waves",
            "figsize",
            "tick_type",
            "panel_spacing",
            "title_space",
        }
        filtered = {k: v for k, v in kwargs.items() if k not in invalid_kwargs}
        if any(k in kwargs for k in invalid_kwargs):
            bad = [k for k in invalid_kwargs if k in kwargs]
            print(f"⚠️  Ignoring invalid kwargs for plot_trace: {bad}")

        if isinstance(data, Trace):
            stream = Stream([data])
        elif isinstance(data, Stream):
            stream = data
        else:
            raise TypeError("data must be an ObsPy Trace or Stream object")

        target_panels = self._resolve_target_panels(
            panels, stations, networks, ids, metadata
        )
        if not target_panels:
            target_panels = list(self._panels)

        color = _mpl_color_to_bokeh(filtered.pop("color", "gray"))
        alpha = float(filtered.pop("alpha", 1.0))
        lw = filtered.pop("linewidth", filtered.pop("lw", 1.0))
        line_dash = filtered.pop("linestyle", filtered.pop("ls", "solid"))

        t0_global = self._t0_global()
        for trace in stream:
            matching: list[_PanelRecord] = []
            for rec in target_panels:
                if rec.metadata.get("station") == trace.stats.station:
                    matching.append(rec)
            if not matching:
                if panels is None and not any(
                    [stations, networks, ids, metadata]
                ):
                    print(
                        f"⚠️  No panel found for station {trace.stats.station} (trace: {trace.id})"
                    )
                continue
            for rec in matching:
                xs, ys = self._wave_xy(trace, t0_global=t0_global)
                if not xs:
                    continue
                for fig in self._figures_for_axes(rec, axes):
                    fig.line(
                        xs,
                        ys,
                        color=color,
                        line_width=float(lw),
                        alpha=alpha,
                        line_dash=line_dash,
                        level="underlay" if zorder < 0 else "glyph",
                    )
        return self

    def plot_horizontals(
        self,
        stream: Stream,
        color: str = "gray",
        alpha: float = 0.7,
        zorder: int = -1,
        **kwargs: Any,
    ) -> SwarmClipboardBk:
        """Plot N/E channels from ``stream`` on panels whose station matches each trace."""
        if not isinstance(stream, Stream):
            raise TypeError("stream must be an ObsPy Stream object")
        horizontal_stream = Stream()
        for tr in stream:
            ch = tr.stats.channel
            if len(ch) >= 3 and ch[-1].upper() in ("N", "E"):
                horizontal_stream.append(tr)
        if len(horizontal_stream) == 0:
            print("⚠️  No horizontal components (N/E) found in stream")
            return self
        invalid_kwargs = {
            "mode",
            "sync_waves",
            "figsize",
            "tick_type",
            "panel_spacing",
            "title_space",
        }
        filtered = {k: v for k, v in kwargs.items() if k not in invalid_kwargs}
        if any(k in kwargs for k in invalid_kwargs):
            bad = [k for k in invalid_kwargs if k in kwargs]
            print(f"⚠️  Ignoring invalid kwargs for plot_horizontals: {bad}")
        self.plot_trace(
            horizontal_stream,
            color=color,
            alpha=alpha,
            zorder=zorder,
            **filtered,
        )
        return self

    def plot_peak_value(
        self,
        peak_stream: Stream | Trace,
        *,
        window_s: float = 1.0,
        use_raw_values: bool = False,
        aggregate: str = "max",
        cmap: Any = "magma",
        cmap_by_index: dict[int, Any] | None = None,
        alpha: float = 0.7,
        alpha_from_data: bool = False,
        interpolation: str = "nearest",
        zorder: float = 0.2,
        add_colorbar: bool = False,
        colorbar_rect: tuple[float, float, float, float] = (0.2, 0.02, 0.6, 0.025),
        colorbar_label: str = "Normalized peak intensity (0-1)",
    ) -> SwarmClipboardBk:
        """
        Overlay a peak-value raster behind waveform axes (modes ``w`` and ``wg`` only).
        """
        if "w" not in self.mode:
            raise ValueError(
                "plot_peak_value() only supports modes that include waveform axes ('w', 'wg')."
            )
        if isinstance(peak_stream, Trace):
            peak_stream = Stream([peak_stream])
        if not isinstance(peak_stream, Stream):
            raise TypeError("peak_stream must be an ObsPy Stream or Trace.")
        if len(peak_stream) < len(self._panels):
            raise ValueError(
                f"peak_stream has {len(peak_stream)} traces but clipboard has {len(self._panels)} panels."
            )
        if alpha_from_data:
            raise NotImplementedError(
                "alpha_from_data for Bokeh clipboard is not implemented yet."
            )

        first_bar_fig: Any | None = None
        for i, rec in enumerate(self._panels):
            if rec.wave_figure is None:
                continue
            wave_fig = rec.wave_figure
            tr_peak = peak_stream[i]

            data = np.asarray(np.ma.filled(tr_peak.data, fill_value=0.0), dtype=float)
            data = np.where(np.isfinite(data), data, 0.0)
            if bool(use_raw_values):
                peak_norm = np.clip(data, 0.0, 1.0)
                n_bins = int(peak_norm.size)
            else:
                sr = float(tr_peak.stats.sampling_rate)
                win = max(1, int(round(float(window_s) * sr)))
                n_bins = int(np.ceil(data.size / win))
                pad = n_bins * win - data.size
                if pad:
                    data = np.pad(data, (0, pad), mode="constant", constant_values=0.0)
                agg = str(aggregate).lower()
                if agg == "mean":
                    peak = np.mean(np.abs(data).reshape(n_bins, win), axis=1)
                else:
                    peak = np.max(np.abs(data).reshape(n_bins, win), axis=1)
                peak_log = np.log10(peak + 1.0)
                peak_norm = peak_log / peak_log.max() if peak_log.max() > 0 else peak_log

            xs, _ys = self._wave_xy(rec.trace, t0_global=self._t0_global())
            if not xs:
                continue
            x_left = float(min(xs))
            dw = float(max(xs) - min(xs))
            if dw <= 0:
                dw = 1.0

            yr = wave_fig.y_range
            y0 = float(yr.start)
            y1 = float(yr.end)
            dh = y1 - y0
            if dh <= 0:
                dh = 1.0

            this_cmap = cmap_by_index.get(i, cmap) if isinstance(cmap_by_index, dict) else cmap
            palette = _palette_from_cmap(this_cmap)
            mapper = LinearColorMapper(
                palette=palette, low=0.0, high=1.0, nan_color="#00000000"
            )
            z = np.asarray(peak_norm, dtype=float).reshape(1, n_bins)
            wave_fig.image(
                image=[z],
                x=x_left,
                y=y0,
                dw=dw,
                dh=dh,
                color_mapper=mapper,
                global_alpha=float(alpha),
                level="underlay",
            )
            _ = zorder

            if add_colorbar and first_bar_fig is None:
                first_bar_fig = wave_fig

        if add_colorbar and first_bar_fig is not None:
            mapper_bar = LinearColorMapper(
                palette=_palette_from_cmap(cmap), low=0.0, high=1.0
            )
            cbar = ColorBar(
                color_mapper=mapper_bar,
                label_standoff=8,
                border_line_color=None,
                location="bottom_center",
                title=colorbar_label,
                orientation="horizontal",
                bar_line_color=None,
                major_label_text_font_size="9pt",
            )
            first_bar_fig.add_layout(cbar, "below")

        _ = colorbar_rect
        _ = interpolation
        return self

    def plot_catalog(
        self,
        catalog: Catalog | Event,
        *,
        panels: list[int] | int | None = None,
        axes: list[int] | int | None = None,
        plot_picks: bool = True,
        plot_origins: bool = True,
        origin_color: str = "black",
        p_color: str = "red",
        s_color: str = "blue",
        verbose: bool = False,
        stations=None,
        networks=None,
        ids=None,
        metadata=None,
        **kwargs: Any,
    ) -> SwarmClipboardBk:
        """Plot origin times and station-matched picks (same targeting rules as :class:`SwarmClipboard`)."""
        if isinstance(catalog, Event):
            catalog = Catalog([catalog])
        span_base = _span_kwargs(kwargs)
        targets = self._resolve_target_panels(
            panels, stations, networks, ids, metadata
        )

        for event in catalog:
            if plot_origins and event.origins:
                ot = event.origins[0].time
                if verbose:
                    print(f" {ot} | Origin time")
                for rec in targets:
                    x = self._utc_to_x(rec, UTCDateTime(ot))
                    sk = {**span_base, "line_color": _mpl_color_to_bokeh(origin_color)}
                    for fig in self._figures_for_axes(rec, axes):
                        fig.add_layout(Span(location=x, dimension="height", **sk))

            if plot_picks and event.picks:
                for pick in event.picks:
                    wid = pick.waveform_id
                    pick_station = wid.station_code if wid is not None else None
                    if verbose:
                        ph = pick.phase_hint or "?"
                        print(f" {pick.time} | {pick_station!r} : {ph}")
                    ph = pick.phase_hint
                    col = p_color
                    if ph and str(ph).upper() == "S":
                        col = s_color
                    elif ph and str(ph).upper() == "P":
                        col = p_color
                    else:
                        col = p_color
                    if pick_station is None:
                        continue
                    for rec in targets:
                        if (
                            rec.metadata.get("station") != pick_station
                            and rec.metadata.get("sta") != pick_station
                        ):
                            continue
                        x = self._utc_to_x(rec, UTCDateTime(pick.time))
                        pk = {**span_base, "line_color": _mpl_color_to_bokeh(col)}
                        for fig in self._figures_for_axes(rec, axes):
                            fig.add_layout(Span(location=x, dimension="height", **pk))
        return self

    def save(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        resources: Resources | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Write standalone HTML (Bokeh CDN resources by default).

        Mirrors the expectation for gallery notebooks alongside matplotlib Tutorial A.
        """
        from bokeh.resources import CDN

        outfile = Path(path)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        save(
            self.layout,
            filename=str(outfile),
            title=title or "Swarm clipboard",
            resources=resources if resources is not None else CDN,
            **kwargs,
        )
