"""
Bokeh :class:`SwarmClipboardBk` — Phase 0 scaffold aligned with :class:`SwarmClipboard`.

Waveform-only (``mode="w"``) and ``tick_type="absolute"`` are implemented enough for
gallery smoke paths; spectrograms and relative ticks raise ``NotImplementedError`` until
later phases. HTML export via :meth:`SwarmClipboardBk.save` is supported from day one.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from bokeh.io import save
from bokeh.layouts import column
from bokeh.models import Range1d
from bokeh.plotting import figure as bk_figure
from bokeh.resources import Resources
from obspy import Stream, Trace

from vdapseisutils.compute.waveforms import prepare_waveform_series

_DEFAULT_WIDTH_PX = 850
_DEFAULT_PANEL_HEIGHT_PX = 220


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
    where Phase 0 applies; unsupported combinations raise ``NotImplementedError``.
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
        self.mode = mode
        self.wave_settings = dict(wave_settings or {})
        self.spec_settings = dict(spec_settings or {})
        self._panel_spacing = panel_spacing
        self._title_space = title_space

        if mode != "w":
            raise NotImplementedError(
                f'SwarmClipboardBk Phase 0 supports mode="w" only; got mode={mode!r}.'
            )
        if tick_type != "absolute":
            raise NotImplementedError(
                f'SwarmClipboardBk Phase 0 supports tick_type="absolute" only; '
                f"got {tick_type!r}."
            )

        traces = self._normalize_traces(data)
        self._traces = traces

        w_px = int(width_px or _DEFAULT_WIDTH_PX)
        n = max(len(traces), 1)
        if panel_height_px is not None:
            h_px = int(panel_height_px)
        elif panel_height is not None:
            # Rough inch→px mapping consistent with Matplotlib ~96 dpi intent.
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

    def _build_layout(self) -> None:
        traces = self._traces
        children: list[Any] = []

        if not traces:
            fig = bk_figure(
                width=self._width_px,
                height=max(self._panel_height_px, 160),
                title="SwarmClipboardBk — no data",
                x_axis_type="datetime",
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            fig.line([], [])
            self.figures = [fig]
            children.append(fig)
            self.layout.children = children
            return

        children: list[Any] = []
        shared_range: Range1d | None = None
        if self.sync_waves:
            starts_ms: list[float] = []
            ends_ms: list[float] = []
            for tr in traces:
                ser = prepare_waveform_series(tr, relative_offset_s=0.0)
                xs = _trace_times_absolute_ms(ser)
                if xs:
                    starts_ms.append(xs[0])
                    ends_ms.append(xs[-1])
            if starts_ms and ends_ms:
                shared_range = Range1d(
                    start=min(starts_ms),
                    end=max(ends_ms),
                    bounds=None,
                )

        color = self.wave_settings.get("color", "black")

        for tr in traces:
            ser = prepare_waveform_series(tr, relative_offset_s=0.0)
            xs = _trace_times_absolute_ms(ser)
            ys = list(ser.amplitudes)

            xr = shared_range
            if xr is None and xs:
                xr = Range1d(start=xs[0], end=xs[-1])

            fig = bk_figure(
                width=self._width_px,
                height=self._panel_height_px,
                title=tr.id,
                x_axis_type="datetime",
                x_range=xr,
                tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                active_scroll="wheel_zoom",
            )
            fig.line(xs, ys, color=color, line_width=1)
            fig.yaxis.axis_label = "Amplitude"

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
