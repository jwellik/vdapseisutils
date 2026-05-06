"""
Bokeh-backed Swarm-style helicorder (chunk-1 core renderer + API subset).
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from bokeh.events import MouseMove, Tap
from bokeh.io import save as bokeh_save
from bokeh.io import show as bokeh_show
from bokeh.layouts import column
from bokeh.models import BoxZoomTool, Div, FixedTicker, WheelZoomTool, ZoomInTool, ZoomOutTool
from bokeh.plotting import figure as bk_figure
from bokeh.resources import CDN, Resources
from obspy import Stream, UTCDateTime
from obspy.core.event import Catalog, Event

from vdapseisutils.core.swarmmpl.bokeh.clipboard import SwarmClipboardBk
from vdapseisutils.core.swarmmpl.colors import earthworm_colors_hex, greyscale_hex, swarm_colors_hex

_DEFAULT_WIDTH_PX = 900
_DEFAULT_HEIGHT_PX = 650
_LINE_WIDTH = 0.8
_LINE_ALPHA = 0.95
_STRIP_HALF_HEIGHT = 0.45


def _normalize_interval_minutes(interval: float | int) -> int:
    v = float(interval)
    if v <= 15:
        return 15
    if v <= 30:
        return 30
    if v <= 60:
        return 60
    return int(math.ceil(v / 60.0) * 60)


def _round_to_minute(utc_datetime: UTCDateTime) -> UTCDateTime:
    if utc_datetime.second >= 30:
        return utc_datetime.replace(second=0, microsecond=0) + 60
    return utc_datetime.replace(second=0, microsecond=0)


def _color_sequence(color: Any) -> list[str]:
    if isinstance(color, str):
        k = color.lower().strip()
        if k == "swarm":
            return list(swarm_colors_hex)
        if k == "greyscale":
            return list(greyscale_hex)
        if k == "earthworm":
            return list(earthworm_colors_hex)
        if k == "obspy":
            # Practical parity target: simple grayscale-ish sequence.
            return ["#111111", "#444444", "#777777", "#aaaaaa"]
        return [color]
    if isinstance(color, (list, tuple)):
        return [str(c) for c in color] or ["#000000"]
    return ["#000000"]


def _restrict_zoom_to_x_axis(fig: Any) -> None:
    for tool in fig.toolbar.tools:
        if isinstance(tool, (WheelZoomTool, BoxZoomTool, ZoomInTool, ZoomOutTool)):
            tool.dimensions = "width"


class SwarmHelicorderBk:
    """
    Chunk-1 Bokeh helicorder implementation focused on rendering parity subset.

    Known chunk-1 parity deltas:
    - Timezone footer labels are represented with bottom y-tick labels and a compact text
      footer, not with full MPL left/right mirrored axis handling.
    - Multi-trace composition is additive across traces without ObsPy dayplot-specific
      decimation.
    """

    name = "helicorder_bokeh"

    def __init__(
        self,
        st: Stream,
        interval: int = 60,
        color: Any = "swarm",
        one_bar_range: Any = None,
        clip_threshold: Any = "auto",
        title: str | None = None,
        utc_offset_left: str = "UTC",
        utc_offset_right: str = "UTC",
        *,
        width_px: int | None = None,
        height_px: int | None = None,
        toolbar_location: str = "right",
        zoom_x_only: bool = True,
        **kwargs: Any,
    ) -> None:
        if not isinstance(st, Stream) or len(st) == 0:
            raise ValueError("SwarmHelicorderBk requires a non-empty ObsPy Stream.")

        self.stream = st.copy()
        self.starttime = kwargs.get("starttime") or min(tr.stats.starttime for tr in self.stream)
        self.endtime = kwargs.get("endtime") or max(tr.stats.endtime for tr in self.stream)
        self.stream.trim(self.starttime, self.endtime, pad=True)

        self.line_len_min = _normalize_interval_minutes(interval)
        self.interval = int(self.line_len_min * 60)
        self.title = title if title is not None else self.stream[0].id
        self.label_spacing = int(kwargs.get("label_spacing", 4))
        self.utc_offset_left = str(utc_offset_left)
        self.utc_offset_right = str(utc_offset_right)
        self.colors = _color_sequence(color)
        self.width_px = int(width_px or _DEFAULT_WIDTH_PX)
        self.height_px = int(height_px or _DEFAULT_HEIGHT_PX)
        self.toolbar_location = toolbar_location
        self.zoom_x_only = bool(zoom_x_only)

        self.one_bar_range = self._calculate_vertical_scaling_range(one_bar_range)
        self.clip_threshold = self._resolve_clip_threshold(clip_threshold)

        self._strip_rows = self._build_strip_rows()
        self.nlines = len(self._strip_rows)

        self.figure = self._build_figure()
        self._build_helicorder_lines()
        self._overlay_renderers: list[Any] = []
        self._footer = Div(text="", width=self.width_px)
        self.set_tticks(update_tzlabels=True)
        self.set_tzticklabel()
        self._layout_placement = "below"
        self._attached_clipboard: SwarmClipboardBk | None = None
        self._clipboard_window_s = 600.0
        self._clipboard_sync_focus = True
        self.focus_time: UTCDateTime | None = None
        self._focus_data_provider = None
        self._focus_interaction_enabled = bool(kwargs.get("enable_focus_interactions", True))
        self.layout = column(self.figure, self._footer, sizing_mode="stretch_width")
        self._bind_focus_events()

    def _resolve_clip_threshold(self, clip_threshold: Any) -> float | None:
        if clip_threshold == "auto":
            return float(3.0 * self.one_bar_range)
        if clip_threshold is None:
            return None
        return float(clip_threshold)

    def _calculate_vertical_scaling_range(self, one_bar_range: Any) -> float:
        if one_bar_range is None:
            return float(self._calculate_per_interval_percentile(99.5))
        if one_bar_range == "obspy":
            return float(self._calculate_per_interval_percentile(99.5))
        if isinstance(one_bar_range, (int, float)):
            if one_bar_range <= 0:
                raise ValueError("Data value must be positive")
            return float(one_bar_range)
        if isinstance(one_bar_range, tuple) and len(one_bar_range) == 2:
            value, unit = one_bar_range
            if unit == "percentile":
                if not (0 <= value <= 100):
                    raise ValueError("Percentile must be between 0 and 100")
                return float(self._calculate_per_interval_percentile(float(value)))
            if unit == "data":
                if value <= 0:
                    raise ValueError("Data value must be positive")
                return float(value)
            raise ValueError("Unit must be 'percentile' or 'data'")
        raise ValueError("one_bar_range must be None, 'obspy', int/float, or tuple (value, unit)")

    def _calculate_per_interval_percentile(self, percentile: float) -> float:
        total_duration = max(float(self.endtime - self.starttime), 0.0)
        n_intervals = int(math.ceil(total_duration / float(self.interval))) if total_duration > 0 else 1
        interval_percentiles: list[float] = []
        for i in range(max(n_intervals, 1)):
            i0 = self.starttime + i * self.interval
            i1 = min(i0 + self.interval, self.endtime)
            vals: list[float] = []
            for tr in self.stream:
                sr = float(tr.stats.sampling_rate)
                sidx = max(0, int((i0 - tr.stats.starttime) * sr))
                eidx = min(len(tr.data), int((i1 - tr.stats.starttime) * sr))
                if sidx < eidx:
                    vals.extend(np.asarray(tr.data[sidx:eidx], dtype=float).tolist())
            if vals:
                interval_percentiles.append(float(np.percentile(np.abs(vals), percentile)))
        if interval_percentiles:
            return max(interval_percentiles)
        all_data = np.concatenate([np.asarray(tr.data, dtype=float) for tr in self.stream])
        return float(np.percentile(np.abs(all_data), percentile)) if all_data.size else 1.0

    def _build_strip_rows(self) -> list[tuple[UTCDateTime, UTCDateTime]]:
        total = max(float(self.endtime - self.starttime), 0.0)
        nlines = int(math.ceil(total / self.interval)) if total > 0 else 1
        rows: list[tuple[UTCDateTime, UTCDateTime]] = []
        for i in range(nlines):
            row_start = self.starttime + i * self.interval
            row_end = row_start + self.interval
            rows.append((row_start, row_end))
        return rows

    def _build_figure(self) -> Any:
        fig = bk_figure(
            width=self.width_px,
            height=self.height_px,
            title=self.title,
            toolbar_location=self.toolbar_location,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        if self.zoom_x_only:
            _restrict_zoom_to_x_axis(fig)
        fig.xaxis.axis_label = "Seconds within strip"
        fig.yaxis.axis_label = ""
        fig.x_range.start = 0.0
        fig.x_range.end = float(self.interval)
        fig.y_range.start = -0.75
        fig.y_range.end = max(float(self.nlines), 1.0)
        return fig

    def _build_helicorder_lines(self) -> None:
        if self.one_bar_range <= 0:
            self.one_bar_range = 1.0
        strip_xs: dict[int, list[float]] = defaultdict(list)
        strip_ys: dict[int, list[float]] = defaultdict(list)
        for tr in self.stream:
            sr = float(tr.stats.sampling_rate)
            data = np.asarray(tr.data, dtype=float)
            npts = len(data)
            if npts == 0:
                continue
            t0 = tr.stats.starttime
            ts = np.arange(npts, dtype=float) / sr
            abs_s = np.asarray([float((t0 + float(dt)) - self.starttime) for dt in ts], dtype=float)
            valid = (abs_s >= 0.0) & (abs_s <= float(self.endtime - self.starttime))
            if not np.any(valid):
                continue
            abs_s = abs_s[valid]
            data = data[valid]
            if self.clip_threshold is not None:
                data = np.clip(data, -self.clip_threshold, self.clip_threshold)

            strip_idx = np.floor(abs_s / float(self.interval)).astype(int)
            in_range = (strip_idx >= 0) & (strip_idx < self.nlines)
            strip_idx = strip_idx[in_range]
            abs_s = abs_s[in_range]
            data = data[in_range]
            if strip_idx.size == 0:
                continue

            x_local = abs_s - (strip_idx * float(self.interval))
            amp_norm = np.clip(data / float(self.one_bar_range), -1.0, 1.0)
            y_center = self.nlines - strip_idx - 0.5
            y_local = y_center + amp_norm * _STRIP_HALF_HEIGHT

            for sidx, xv, yv in zip(strip_idx.tolist(), x_local.tolist(), y_local.tolist()):
                strip_xs[sidx].append(float(xv))
                strip_ys[sidx].append(float(yv))

        for sidx in range(self.nlines):
            xs = strip_xs.get(sidx, [])
            ys = strip_ys.get(sidx, [])
            if not xs:
                continue
            order = np.argsort(np.asarray(xs))
            xs_ord = np.asarray(xs)[order].tolist()
            ys_ord = np.asarray(ys)[order].tolist()
            line_color = self.colors[sidx % len(self.colors)]
            self.figure.line(xs_ord, ys_ord, line_color=line_color, line_width=_LINE_WIDTH, line_alpha=_LINE_ALPHA)

    def _refresh_layout(self) -> None:
        core = [self.figure, self._footer]
        if self._attached_clipboard is None:
            self.layout = column(*core, sizing_mode="stretch_width")
            return
        clip_layout = getattr(self._attached_clipboard, "layout", self._attached_clipboard)
        if self._layout_placement == "above":
            self.layout = column(clip_layout, *core, sizing_mode="stretch_width")
        else:
            self.layout = column(*core, clip_layout, sizing_mode="stretch_width")

    def _bind_focus_events(self) -> None:
        if not self._focus_interaction_enabled:
            return
        try:
            self.figure.on_event(MouseMove, self._on_hover_focus)
            self.figure.on_event(Tap, self._on_tap_focus)
        except Exception:
            # Keep static rendering robust in environments that don't expose event callbacks.
            self._focus_interaction_enabled = False

    def _on_hover_focus(self, event: Any) -> None:
        self._update_focus_from_xy(getattr(event, "x", None), getattr(event, "y", None))

    def _on_tap_focus(self, event: Any) -> None:
        self._update_focus_from_xy(getattr(event, "x", None), getattr(event, "y", None))

    def _time2xy(self, time: Any) -> tuple[float, float] | tuple[None, None]:
        t = UTCDateTime(time)
        if t < self.starttime or t > self.endtime:
            return None, None
        abs_s = float(t - self.starttime)
        strip_idx = int(np.floor(abs_s / float(self.interval)))
        strip_idx = int(np.clip(strip_idx, 0, max(self.nlines - 1, 0)))
        x_local = abs_s - float(strip_idx * self.interval)
        y_center = float(self.nlines - strip_idx - 0.5)
        return float(x_local), y_center

    def _xy_to_time(self, x_value: Any, y_value: Any) -> UTCDateTime | None:
        try:
            x = float(x_value)
            y = float(y_value)
        except (TypeError, ValueError):
            return None
        if self.nlines <= 0:
            return None
        row_idx = int(np.floor(float(self.nlines) - y))
        if row_idx < 0 or row_idx >= self.nlines:
            return None
        x_offset = float(np.clip(x, 0.0, float(self.interval)))
        row_start_time = self.starttime + row_idx * self.interval
        t = row_start_time + x_offset
        if t < self.starttime or t > self.endtime:
            return None
        return t

    def _update_focus_from_xy(self, x_value: Any, y_value: Any) -> UTCDateTime | None:
        t = self._xy_to_time(x_value, y_value)
        if t is None:
            return None
        self.set_focus_time(t)
        return t

    def set_focus_data_provider(self, provider: Any) -> SwarmHelicorderBk:
        """
        Register an optional future provider hook for focus-window data retrieval.

        Static mode remains default; provider integration is intentionally minimal in chunk 2.
        """
        self._focus_data_provider = provider
        return self

    def _maybe_fetch_focus_window_data(self, left: UTCDateTime, right: UTCDateTime) -> Any:
        provider = self._focus_data_provider
        if provider is None:
            return None
        if callable(provider):
            return provider(self, left, right)
        fetch = getattr(provider, "fetch_window", None)
        if callable(fetch):
            return fetch(self, left, right)
        return None

    def _sync_clipboard_focus_window(self) -> None:
        if (
            self._attached_clipboard is None
            or not self._clipboard_sync_focus
            or self.focus_time is None
        ):
            return
        window = float(self._clipboard_window_s)
        if window <= 0:
            return
        half = window / 2.0
        left = self.focus_time - half
        right = self.focus_time + half
        _ = self._maybe_fetch_focus_window_data(left, right)
        self._attached_clipboard.set_xlim(left=left, right=right)

    def set_focus_time(self, focus_time: Any, *, sync_clipboard: bool = True) -> SwarmHelicorderBk:
        self.focus_time = UTCDateTime(focus_time) if focus_time is not None else None
        if sync_clipboard:
            self._sync_clipboard_focus_window()
        return self

    def attach_clipboard(
        self,
        clipboard: SwarmClipboardBk | None = None,
        *,
        focus_time: Any = None,
        mode: str = "wg",
        window_s: float = 600,
        sync_focus: bool = True,
        create_if_missing: bool = True,
        placement: str = "below",
    ) -> SwarmClipboardBk | None:
        if clipboard is None and create_if_missing:
            clipboard = SwarmClipboardBk(data=self.stream.copy(), mode=mode, tick_type="absolute")
        if clipboard is None:
            self._attached_clipboard = None
            self._refresh_layout()
            return None
        self._attached_clipboard = clipboard
        self._clipboard_window_s = float(window_s)
        self._clipboard_sync_focus = bool(sync_focus)
        self._layout_placement = "above" if str(placement).lower() == "above" else "below"
        self._refresh_layout()
        if focus_time is not None:
            self.set_focus_time(focus_time, sync_clipboard=sync_focus)
        elif self.focus_time is not None and sync_focus:
            self._sync_clipboard_focus_window()
        return clipboard

    def plot_tags(
        self,
        times: Any,
        marker: str = "circle",
        color: str = "red",
        markersize: float = 10,
        markeredgecolor: str = "black",
        alpha: float = 0.9,
        **kwargs: Any,
    ) -> list[Any]:
        seq = times if isinstance(times, (list, tuple, np.ndarray)) else [times]
        xs: list[float] = []
        ys: list[float] = []
        for t in seq:
            xv, yv = self._time2xy(t)
            if xv is None or yv is None:
                continue
            xs.append(float(xv))
            ys.append(float(yv))
        if not xs:
            return []
        marker_name = str(marker or "circle")
        if marker_name == "o":
            marker_name = "circle"
        if marker_name == "*":
            marker_name = "asterisk"
        if marker_name == "|":
            marker_name = "dash"
        renderer = self.figure.scatter(
            x=xs,
            y=ys,
            marker=marker_name,
            size=float(markersize),
            fill_color=color,
            line_color=markeredgecolor,
            fill_alpha=float(alpha),
            line_alpha=float(alpha),
            **kwargs,
        )
        self._overlay_renderers.append(renderer)
        return [renderer]

    def _event_origin_time(self, event: Event) -> UTCDateTime | None:
        origin = event.preferred_origin() if hasattr(event, "preferred_origin") else None
        if origin is None and getattr(event, "origins", None):
            origin = event.origins[0]
        return None if origin is None else UTCDateTime(origin.time)

    def _matching_station_ids(self) -> set[str]:
        out: set[str] = set()
        for tr in self.stream:
            out.add(str(getattr(tr.stats, "station", "")))
            out.add(str(getattr(tr.stats, "network", "")))
            out.add(str(tr.id))
            n = str(getattr(tr.stats, "network", ""))
            s = str(getattr(tr.stats, "station", ""))
            l = str(getattr(tr.stats, "location", ""))
            c = str(getattr(tr.stats, "channel", ""))
            out.add(f"{n}.{s}.{l}.{c}")
            out.add(f"{n}.{s}.{l}")
        return out

    def plot_catalog(
        self,
        catalog: Catalog | Event,
        *,
        plot_picks: bool = True,
        plot_origins: bool = True,
        origin_marker: str = "diamond",
        origin_color: str = "black",
        pick_marker: str = "dash",
        p_color: str = "red",
        s_color: str = "blue",
        pick_size: float = 14,
        **kwargs: Any,
    ) -> list[Any]:
        if isinstance(catalog, Event):
            catalog = Catalog([catalog])
        renderers: list[Any] = []
        known_ids = self._matching_station_ids()
        for event in catalog:
            if plot_origins:
                ot = self._event_origin_time(event)
                if ot is not None:
                    renderers.extend(
                        self.plot_tags(
                            [ot],
                            marker=origin_marker,
                            color=origin_color,
                            markersize=float(kwargs.pop("markersize", 12)),
                            markeredgecolor=origin_color,
                            **kwargs,
                        )
                    )
            if plot_picks:
                for pick in getattr(event, "picks", []):
                    wid = getattr(pick, "waveform_id", None)
                    sta = getattr(wid, "station_code", None) if wid is not None else None
                    seed = (
                        wid.get_seed_string() if (wid is not None and hasattr(wid, "get_seed_string")) else None
                    )
                    if sta is None and seed is None:
                        continue
                    if sta not in known_ids and (seed is None or seed not in known_ids):
                        continue
                    phase = str(getattr(pick, "phase_hint", "") or "").upper()
                    pcol = s_color if phase == "S" else p_color
                    renderers.extend(
                        self.plot_tags(
                            [pick.time],
                            marker=pick_marker,
                            color=pcol,
                            markersize=pick_size,
                            markeredgecolor=pcol,
                        )
                    )
        return renderers

    def plot_events(self, events: Catalog | Event | list[Event], **kwargs: Any) -> list[Any]:
        if isinstance(events, Event):
            return self.plot_catalog(Catalog([events]), **kwargs)
        if isinstance(events, Catalog):
            return self.plot_catalog(events, **kwargs)
        return self.plot_catalog(Catalog(list(events)), **kwargs)

    def _format_tticklabels(self, ticktimes: list[UTCDateTime]) -> list[str]:
        dts = [t.datetime for t in ticktimes]
        labels: list[str] = []
        previous_date = None
        for i, dt in enumerate(dts):
            current_date = dt.date()
            is_new_date = (i == 0) or (current_date != previous_date)
            if is_new_date and dt.hour == 0 and dt.minute == 0:
                label = dt.strftime("%Y/%m/%d")
            elif is_new_date:
                label = dt.strftime("%Y/%m/%d\n%H:%M")
            else:
                label = dt.strftime("%H:%M")
            labels.append(label)
            previous_date = current_date
        return labels

    def set_tticks(
        self,
        label_spacing: int | None = None,
        utc_offset: int = 0,
        axes: str = "left",
        update_tzlabels: bool = True,
    ) -> SwarmHelicorderBk:
        if label_spacing is not None:
            self.label_spacing = int(label_spacing)

        yticks = (np.arange(self.nlines, 0, -1, dtype=float) - 0.5).tolist()
        tticks_left = [_round_to_minute(self.endtime - (y + 0.5) * self.interval) for y in yticks]
        tticks_right = [_round_to_minute(self.endtime - (y + 1.5) * self.interval) for y in yticks]

        indices: list[int] = []
        for i, time in enumerate(tticks_left):
            if time.hour % max(self.label_spacing, 1) == 0 and time.minute == 0:
                indices.append(i)
        if not indices:
            indices = [0]

        yt = [yticks[i] for i in indices]
        tleft = [tticks_left[i] for i in indices]
        tright = [tticks_right[i] for i in indices]
        labels_left = self._format_tticklabels(tleft)
        labels_right = self._format_tticklabels(tright)

        self._yticks_core = yt
        self._ylabels_left = labels_left
        self._ylabels_right = labels_right
        self.figure.yaxis.ticker = FixedTicker(ticks=yt)
        self.figure.yaxis.major_label_overrides = {float(v): s for v, s in zip(yt, labels_left)}

        if update_tzlabels:
            self.set_tzticklabel(utc_offset=utc_offset, axes=axes)
        return self

    def set_tzticklabel(
        self,
        custom: str | None = None,
        utc_offset: int = 0,
        axes: str = "left",
    ) -> SwarmHelicorderBk:
        base = custom if custom else f"UTC{int(utc_offset):+03}:00"
        left_text = self.utc_offset_left if self.utc_offset_left != "UTC" else base
        right_text = self.utc_offset_right if self.utc_offset_right != "UTC" else base

        yticks = list(getattr(self, "_yticks_core", []))
        labels_left = list(getattr(self, "_ylabels_left", []))
        labels_right = list(getattr(self, "_ylabels_right", []))
        tz_tick = -0.5
        yticks.append(tz_tick)
        labels_left.append(left_text)
        labels_right.append(right_text)
        if str(axes).lower() == "right":
            labels = labels_right
        else:
            labels = labels_left

        self.figure.yaxis.ticker = FixedTicker(ticks=yticks)
        self.figure.yaxis.major_label_overrides = {float(v): s for v, s in zip(yticks, labels)}
        self._footer.text = f"<div style='font-size:11px;color:#444;'>Left: {left_text} | Right: {right_text}</div>"
        return self

    def show(self, **kwargs: Any) -> None:
        bokeh_show(self.layout, **kwargs)

    def save(
        self,
        path: str | Path,
        title: str | None = None,
        resources: Resources | None = None,
        **kwargs: Any,
    ) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        bokeh_save(
            self.layout,
            filename=str(out),
            title=title or (self.title or "Swarm helicorder"),
            resources=resources if resources is not None else CDN,
            **kwargs,
        )
