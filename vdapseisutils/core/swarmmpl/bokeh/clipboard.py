"""
Bokeh :class:`SwarmClipboardBk` — aligned with :class:`SwarmClipboard`.

**Phase 1:** waveform-only (``mode="w"``), ``tick_type`` **absolute** (also ``datetime`` /
``time``) or **relative** (seconds), ``sync_waves`` linked or independent x-ranges.
Spectrogram modes raise ``NotImplementedError`` until Phase 2. HTML export via
:meth:`SwarmClipboardBk.save` is supported from day one.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from bokeh.io import save
from bokeh.layouts import column
from bokeh.models import Range1d
from bokeh.plotting import figure as bk_figure
from bokeh.resources import Resources
from obspy import Stream, Trace

from vdapseisutils.compute.waveforms import prepare_waveform_series

_DEFAULT_WIDTH_PX = 850
_DEFAULT_PANEL_HEIGHT_PX = 220


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

    Parameters mirror :class:`~vdapseisutils.core.swarmmpl.clipboard.SwarmClipboard`
    where implemented; unsupported combinations raise ``NotImplementedError``.
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
        self.mode = mode
        self.wave_settings = dict(wave_settings or {})
        self.spec_settings = dict(spec_settings or {})
        self._panel_spacing = panel_spacing
        self._title_space = title_space

        if mode != "w":
            raise NotImplementedError(
                f'SwarmClipboardBk currently supports mode="w" only; got mode={mode!r}.'
            )

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

        # Relative axis: seconds (legacy Clipboard-style alignment).
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

        color = self.wave_settings.get("color", "black")
        x_axis_type = "datetime" if self._tick == "absolute" else "linear"
        x_label = "Time" if self._tick == "absolute" else "Time (s)"

        for tr in traces:
            xs, ys = self._wave_xy(tr, t0_global=t0_global)
            if not xs:
                xs, ys = [0.0], [0.0]

            xr = self._per_trace_range(tr, xs, shared_range)

            fig = bk_figure(
                width=self._width_px,
                height=self._panel_height_px,
                title=tr.id,
                x_axis_type=x_axis_type,
                x_range=xr,
                tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                active_scroll="wheel_zoom",
            )
            fig.line(xs, ys, color=color, line_width=1)
            fig.yaxis.axis_label = "Amplitude"
            fig.xaxis.axis_label = x_label

            self.figures.append(fig)
            children.append(fig)

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
