"""
Bokeh :class:`SwarmClipboardBk` — aligned with :class:`SwarmClipboard`.

**Phase 1:** ``mode="w"``, ``tick_type`` absolute / relative, ``sync_waves``.
**Phase 2:** ``mode="g"`` (spectrogram only), ``mode="wg"`` (waveform + spectrogram),
using :func:`~vdapseisutils.compute.waveforms.compute_spectrogram`, inferno-style palette,
and optional color bar. HTML export via :meth:`SwarmClipboardBk.save`.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from bokeh.io import save
from bokeh.layouts import column
from bokeh.models import ColorBar, LinearColorMapper, Range1d
from bokeh.plotting import figure as bk_figure
from bokeh.resources import Resources
from matplotlib.colors import to_hex
from obspy import Stream, Trace

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


class SwarmClipboardBk:
    """
    Interactive multi-panel clipboard built with Bokeh.

    Parameters mirror :class:`~vdapseisutils.core.swarmmpl.clipboard.SwarmClipboard`.
    Unknown ``mode`` / ``tick_type`` raises ``ValueError``; spectrogram failure on a trace
    shows a placeholder figure for that panel.
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

        self.figures: list[Any] = []
        self.layout = column(children=[], sizing_mode="stretch_width")
        self._build_layout()

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
            fig = bk_figure(
                width=self._width_px,
                height=max(self._panel_height_px, 160),
                title="SwarmClipboardBk — no data",
                x_axis_type="datetime",
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
        else:
            fig = bk_figure(
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
            return bk_figure(
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
            fig = bk_figure(
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

        fig = bk_figure(
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
        color = self.wave_settings.get("color", "black")
        x_axis_type = "datetime" if self._tick == "absolute" else "linear"
        x_label = "" if hide_x_labels else ("Time" if self._tick == "absolute" else "Time (s)")

        fig = bk_figure(
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
            fig = self._empty_figure()
            self.figures = [fig]
            self.layout.children = [fig]
            return

        t0_global = min(tr.stats.starttime for tr in traces) if traces else None
        shared_range = self._compute_shared_range(traces)

        wave_h = max(int(self._panel_height_px * self._wave_frac), 72)
        spec_h = max(int(self._panel_height_px * self._spec_frac), 96)

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
            panel = column([wave_fig, spec_fig], sizing_mode="stretch_width")
            self.figures.append(panel)
            children.append(panel)

        self.layout.children = children

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
