"""
Origin selection helpers for ObsPy events.

Public API for consistent use of preferred vs fallback origins across plotting,
event-rate extraction, and catalog tabulation.
"""

from __future__ import annotations

from obspy.core.event import Event, Origin


def get_primary_origin(event: Event) -> Origin | None:
    """
    Return the origin used for maps, event rates, and catalog point prep.

    Uses :meth:`~obspy.core.event.Event.preferred_origin` when ObsPy resolves it;
    otherwise falls back to the first origin in ``event.origins``. Returns
    ``None`` if the event has no origins.

    Parameters
    ----------
    event : obspy.core.event.Event

    Returns
    -------
    obspy.core.event.Origin or None
    """
    if not event.origins:
        return None
    preferred = event.preferred_origin()
    if preferred is not None:
        return preferred
    return event.origins[0]
