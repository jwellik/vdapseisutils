"""
Core VCatalog class with mixin functionality.

This module contains the main VCatalog class that extends ObsPy's Catalog
with volcano seismology specific functionality through mixins.
"""

from obspy import Catalog
from obspy.core.event import Event
from .plotting import VCatalogPlottingMixin
from .analysis import VCatalogAnalysisMixin
from .conversion import VCatalogConversionMixin
from .comparison import VCatalogComparisonMixin
from .utils import VCatalogUtilsMixin
from .pickqc import VCatalogPickQCMixin


class VEvent(Event):
    """
    Extended :class:`~obspy.core.event.Event` with catalog helpers (e.g. ``to_picklog``,
    ``to_txyzm``, ``get_waveforms``).

    When built from an existing ``Event``, instance state is copied from that object's
    ``__dict__`` (shallow on list fields like ``picks`` / ``origins``), then
    :meth:`~obspy.core.event.event.Event.scope_resource_ids` is run so resource IDs
    refer to this instance.
    """

    def __init__(self, event=None, **kwargs):
        if event is not None:
            super().__init__()
            self.__dict__.update(event.__dict__)
            self.scope_resource_ids()
        else:
            super().__init__(**kwargs)
    
    # Conversion methods
    def to_picklog(self, verbose=False):
        """Convert event picks to a DataFrame (picklog format)."""
        temp_catalog = VCatalog([self])
        return temp_catalog.to_picklog(verbose=verbose)
    
    def to_txyzm(self, **kwargs):
        """Convert event to time/lat/lon/depth/mag format."""
        return VCatalogConversionMixin.to_txyzm(self, **kwargs)
    
    def to_basics(self, **kwargs):
        """Alias for to_txyzm()."""
        return self.to_txyzm(**kwargs)
    
    # Utility methods
    def get_eventid(self, fallback_prefix="id", verbose=False):
        """Extract event ID from this Event object."""
        return VCatalogUtilsMixin.get_eventid(self, self, fallback_prefix, verbose)
    
    def extract_event_id(self, fallback_prefix="id", verbose=False):
        """Alias for get_eventid()."""
        return self.get_eventid(fallback_prefix, verbose)
    
    def get_waveforms(self, client, **kwargs):
        """Get waveforms for this single event."""
        # Create temporary catalog with just this event
        temp_catalog = VCatalog([self])
        return temp_catalog.get_waveforms(client, **kwargs)
    
    def short_str(self):
        """
        Returns a short string representation of the event with depth information.
        This overrides the default Event.short_str() method to include depth.
        """
        try:
            if self.origins:
                origin = self.origins[0]
                time_str = str(origin.time)
                lat = origin.latitude
                lon = origin.longitude
                depth = origin.depth
                
                # Format coordinates with + signs
                lat_str = f"+{lat:.3f}" if lat >= 0 else f"{lat:.3f}"
                lon_str = f"+{lon:.3f}" if lon >= 0 else f"{lon:.3f}"
                
                # Format depth (convert from meters to km if needed)
                if depth is not None:
                    depth_km = depth / 1000.0  # Convert meters to km
                    depth_str = f"{depth_km:.2f}"
                else:
                    depth_str = "N/A"
                
                # Get magnitude information
                mag_str = "N/A"
                if self.magnitudes:
                    mag = self.magnitudes[0]
                    if mag.mag is not None:
                        mag_str = f"{mag.mag:>4.1f} {mag.magnitude_type:<3}"  # Should be max 8 characters
                
                # Determine method (manual/automatic) based on evaluation mode
                method = "automatic"
                if hasattr(origin, 'evaluation_mode') and origin.evaluation_mode:
                    if origin.evaluation_mode.lower() == 'manual':
                        method = "manual"
                
                # Format the event line with depth included
                return f"{time_str:<27} | {lat_str:>8}, {lon_str:>8}, {depth_str:>6} | {mag_str:>8} | {method:<9}"
            else:
                return f"{self.resource_id} | No origin information"
        except Exception:
            return f"{self.resource_id} | Error formatting event"

    def sort_picks(self, inplace=True, verbose=False):
        """
        Sort picks by earliest arrival time for this event.
        
        Parameters
        ----------
        inplace : bool, default True
            If True, modify the event in place. If False, return a new VEvent.
        verbose : bool, default False
            If True, print detailed information about the sorting process
            
        Returns
        -------
        VEvent or None
            If inplace=False, returns a new VEvent with sorted picks.
            If inplace=True, returns None and modifies the current event.
        """
        from copy import deepcopy
        
        # Work on copy if not inplace
        if inplace:
            target_event = self
        else:
            target_event = deepcopy(self)
        
        if not target_event.picks:
            if verbose:
                print("Event has no picks to sort")
            if not inplace:
                return target_event
            return
        
        pick_count = len(target_event.picks)
        
        if verbose:
            print(f">>> VEvent.sort_picks()")
            print(f"Sorting {pick_count} picks")
        
        # Filter out picks without time information
        picks_with_time = []
        picks_without_time = []
        
        for pick in target_event.picks:
            if hasattr(pick, 'time') and pick.time is not None:
                picks_with_time.append(pick)
            else:
                picks_without_time.append(pick)
                if verbose:
                    station = "unknown"
                    try:
                        if hasattr(pick, 'waveform_id') and pick.waveform_id:
                            parts = pick.waveform_id.id.split('.')
                            if len(parts) >= 2:
                                station = parts[1]
                    except:
                        pass
                    print(f"  WARNING: Pick at station {station} has no time, placing at end")
        
        # Sort picks with time by arrival time
        if picks_with_time:
            picks_with_time.sort(key=lambda pick: pick.time)
            
            if verbose:
                earliest_time = picks_with_time[0].time
                latest_time = picks_with_time[-1].time
                time_span = latest_time - earliest_time
                print(f"  Time range: {earliest_time} to {latest_time} (span: {time_span:.2f}s)")
        
        # Combine sorted picks with time + picks without time at the end
        sorted_picks = picks_with_time + picks_without_time
        target_event.picks = sorted_picks

        if verbose:
            print(f"  Sorted {len(picks_with_time)} picks by time, {len(picks_without_time)} without time placed at end")
        
        if not inplace:
            return target_event


class VCatalog(VCatalogConversionMixin, VCatalogPlottingMixin, VCatalogAnalysisMixin, 
               VCatalogComparisonMixin, VCatalogUtilsMixin, VCatalogPickQCMixin, Catalog):
    """Extended Catalog class for volcano seismology workflows."""

    @staticmethod
    def _as_vevent(event):
        """Ensure ``event`` is a :class:`VEvent` (single object identity in ``self.events``)."""
        if isinstance(event, VEvent):
            return event
        if not isinstance(event, Event):
            raise TypeError("Expected obspy Event or VEvent, got %s" % type(event).__name__)
        return VEvent(event)

    def __init__(self, events=None, resource_id=None, description=None,
                 comments=None, creation_info=None, catalog=None):
        """
        Initialize VCatalog.

        Parameters:
        -----------
        events : list, optional
            List of Event objects (standard ObsPy way)
        resource_id : ResourceIdentifier, optional
            Resource identifier (standard ObsPy way)
        description : str, optional
            Description (standard ObsPy way)
        comments : list, optional
            Comments (standard ObsPy way)
        creation_info : CreationInfo, optional
            Creation info (standard ObsPy way)
        catalog : obspy.Catalog, optional
            Existing Catalog object to convert to VCatalog
        """
        # Handle case where first argument is a Catalog object
        if hasattr(events, 'events') and hasattr(events, 'resource_id'):
            # events is actually a Catalog object
            catalog_obj = events
            super().__init__(
                events=catalog_obj.events,
                resource_id=catalog_obj.resource_id,
                description=getattr(catalog_obj, 'description', None),
                comments=getattr(catalog_obj, 'comments', None),
                creation_info=getattr(catalog_obj, 'creation_info', None)
            )
        elif catalog is not None:
            # Initialize from catalog parameter
            super().__init__(
                events=catalog.events,
                resource_id=catalog.resource_id,
                description=getattr(catalog, 'description', None),
                comments=getattr(catalog, 'comments', None),
                creation_info=getattr(catalog, 'creation_info', None)
            )
        else:
            # Standard initialization
            super().__init__(
                events=events,
                resource_id=resource_id,
                description=description,
                comments=comments,
                creation_info=creation_info
            )
        self.events = [self._as_vevent(e) for e in self.events]

    def __getitem__(self, index):
        """
        Return the same event objects stored in :attr:`events` (each a :class:`VEvent`).
        Integer index → :class:`VEvent`; slice → :class:`VCatalog` over the same instances.
        """
        if index == "extra":
            return self.__dict__[index]
        if isinstance(index, slice):
            return self.__class__(events=self.events[index])
        return self.events[index]

    def __setitem__(self, index, event):
        if isinstance(index, str):
            super().__setitem__(index, event)
        else:
            self.events[index] = self._as_vevent(event)

    def append(self, event):
        if not isinstance(event, Event):
            msg = "Append only supports a single Event object as an argument."
            raise TypeError(msg)
        self.events.append(self._as_vevent(event))

    def extend(self, event_list):
        if isinstance(event_list, list):
            for _i in event_list:
                if not isinstance(_i, Event):
                    msg = "Extend only accepts a list of Event objects."
                    raise TypeError(msg)
            self.events.extend(self._as_vevent(_i) for _i in event_list)
        elif isinstance(event_list, Catalog):
            self.extend(event_list.events)
        else:
            msg = "Extend only supports a list of Event objects as argument."
            raise TypeError(msg)

    def __str__(self, print_all=False):
        """
        Returns short summary string of the current catalog with depth information.

        It will contain the number of Events in the Catalog and the return
        value of each Event's short_str() method, modified to include depth.

        :type print_all: bool, optional
        :param print_all: If True, all events will be printed, otherwise a
            maximum of ten event will be printed.
            Defaults to False.
        """
        out = str(len(self.events)) + ' Event(s) in Catalog:\n'
        if len(self) <= 10 or print_all is True:
            out += "\n".join([self._get_vevent(i).short_str() for i in range(len(self))])
        else:
            out += "\n".join([self._get_vevent(i).short_str() for i in range(2)])
            out += "\n...\n"
            out += "\n".join([self._get_vevent(i).short_str() for i in range(len(self)-2, len(self))])
            out += "\nTo see all events call " + \
                   "'print(CatalogObject.__str__(print_all=True))'"
        return out

    def _get_vevent(self, index):
        """
        Helper method to get a VEvent object for a given index.
        This ensures we get VEvent objects with the custom short_str() method.
        """
        return self[index]  # This will return a VEvent object due to __getitem__ override

    def write(self, filename, format=None, **kwargs):
        """
        Override the default write method to handle NLLOC_OBS format specially.
        
        For NLLOC_OBS format, this method writes one file per event to the specified directory.
        For all other formats, it delegates to the parent class write method.
        
        Parameters
        ----------
        filename : str
            For NLLOC_OBS format: Directory path where files should be written
            For other formats: File path as usual
        format : str, optional
            Output format. If "NLLOC_OBS", writes one file per event.
        **kwargs
            Additional arguments passed to the parent write method
        """
        print(f"DEBUG: VCatalog.write() called with filename={filename}, format={format}")
        
        if format == "NLLOC_OBS":
            print("DEBUG: Using VCatalog.write_nlloc_obs() method")
            return self.write_nlloc_obs(filename, **kwargs)
        else:
            print("DEBUG: Delegating to parent class write() method")
            # Delegate to parent class for all other formats
            return super().write(filename, format=format, **kwargs)


# Example usage:
#
# # Method 1: Standard ObsPy way
# vcat = VCatalog(events=[event1, event2], description="My catalog")
#
# # Method 2: From existing catalog (simple form)
# regular_cat = client.get_events(...)
# vcat = VCatalog(regular_cat)  # Clean conversion!
#
# # Method 3: From existing catalog (explicit parameter)
# vcat = VCatalog(catalog=regular_cat)
#
# # Instance methods (work on VCatalog instances)
# vcat.plot_eventrate(freq="1D")
# vcat.print_eventrate(freq="1H", top_n=10)
#
# # Class methods (work on any catalog)
# VCatalog.plot_eventrate_from_catalog(regular_cat, freq="1D")
# VCatalog.print_eventrate_from_catalog(regular_cat, freq="1H", top_n=10)
#
# # Catalog comparison methods:
# # Class method - compare two catalogs
# matching_cat, idx1, idx2 = VCatalog.compare_catalogs(cat1, cat2, threshold_seconds=5)
#
# # Instance method - unified compare() that handles both catalogs and times
# matching_cat, idx_self, idx_other = vcat.compare(other_catalog, threshold_seconds=5)  # with catalog
# matching_cat, idx_self, idx_times = vcat.compare(times, threshold_seconds=5)         # with times
#
# # Utility method - compare times between two lists
# matches = VCatalog.compare_times(times1, times2, threshold_seconds=5)

# TODO get_event_waveforms - Return a list of Streams, 1 for each event 