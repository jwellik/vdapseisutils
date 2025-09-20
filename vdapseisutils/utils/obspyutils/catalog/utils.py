"""
Utility functionality for VCatalog.

This module provides utility methods for earthquake catalogs including
sorting and other core functionality.
"""

from copy import deepcopy
from obspy.core.event import Event


# Note: Event methods are now provided via VEvent wrapper class in core.py
# This eliminates the need for monkey-patching

# TODO extraxt_event_id and get_eventid, duplicated code?
# TODO _sort_picks_for_event and sort_picks_for_event, what is the purpose? duplicated code?


class VCatalogUtilsMixin:
    """Mixin providing utility functionality for VCatalog."""

    def extract_origin_times(self):
        """
        Extract origin times from all events in the catalog.
        
        Returns
        -------
        list of obspy.core.utcdatetime.UTCDateTime
            List of origin times from all events in the catalog
        """
        origin_times = []
        for event in self:
            if event.preferred_origin():
                origin_times.append(event.preferred_origin().time)
            elif len(event.origins) > 0:
                origin_times.append(event.origins[0].time)
        return origin_times

    def print_all(self):
        print(self.__str__(print_all=True))

    def print_summary(self):
        """
        Print a summary of the catalog including event count, time range, magnitude range, and bounding box.
        """
        if not self.events:
            print("0 Event(s) in Catalog: Empty catalog")
            return
        
        # Event count
        event_count = len(self.events)
        print(f"{event_count} Event(s) in Catalog:")
        
        # Time range
        times = []
        for event in self.events:
            if event.origins and event.origins[0].time:
                times.append(event.origins[0].time)
        
        if times:
            start_time = min(times)
            end_time = max(times)
            duration = end_time - start_time
            
            print(f"Catalog Start: {start_time}")
            print(f"Catalog End: {end_time}")
            
            # Calculate duration components
            years = duration // (365 * 24 * 3600)
            remaining_seconds = duration % (365 * 24 * 3600)
            days = remaining_seconds // (24 * 3600)
            remaining_seconds %= (24 * 3600)
            hours = remaining_seconds // 3600
            remaining_seconds %= 3600
            minutes = remaining_seconds // 60
            seconds = remaining_seconds % 60
            
            duration_parts = []
            if years > 0:
                duration_parts.append(f"{int(years)} year{'s' if years != 1 else ''}")
            if days > 0:
                duration_parts.append(f"{int(days)} day{'s' if days != 1 else ''}")
            if hours > 0:
                duration_parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
            if minutes > 0:
                duration_parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
            if seconds > 0:
                duration_parts.append(f"{int(seconds)} second{'s' if seconds != 1 else ''}")
            
            if duration_parts:
                print(f"Catalog Duration: {', '.join(duration_parts)}")
            else:
                print("Catalog Duration: 0 seconds")
        
        # Magnitude range
        magnitudes = []
        for event in self.events:
            if event.magnitudes and event.magnitudes[0].mag is not None:
                magnitudes.append(event.magnitudes[0].mag)
        
        if magnitudes:
            min_mag = min(magnitudes)
            max_mag = max(magnitudes)
            print(f"Magnitude range: {min_mag:.1f} - {max_mag:.1f}")
        
        # Bounding box
        latitudes = []
        longitudes = []
        for event in self.events:
            if event.origins and event.origins[0].latitude is not None:
                latitudes.append(event.origins[0].latitude)
            if event.origins and event.origins[0].longitude is not None:
                longitudes.append(event.origins[0].longitude)
        
        if latitudes and longitudes:
            minlat = min(latitudes)
            maxlat = max(latitudes)
            minlon = min(longitudes)
            maxlon = max(longitudes)
            print(f"Bounding Box: [{minlat:.4f}, {maxlat:.4f}, {minlon:.4f}, {maxlon:.4f}]")

    def sort(self, key="time", ascending=True, inplace=True, missing="last"):
        """
        Sort the catalog by a given key.

        Parameters
        ----------
        key : str or callable, default "time"
            Sorting criteria. Options:
            - 'magnitude': Sort by magnitude (uses first magnitude if multiple)
            - 'time': Sort by origin time (uses first origin if multiple)
            - 'depth': Sort by depth (uses first origin if multiple)
            - 'latitude': Sort by latitude (uses first origin if multiple)
            - 'longitude': Sort by longitude (uses first origin if multiple)
            - callable: Custom function that takes an Event and returns a sortable value
        ascending : bool, default True
            Sort in ascending order (default). If False, sort descending.
        inplace : bool, default True
            If True, sort the catalog in place and return self. If False, return a new VCatalog.
        missing : {'last', 'first'}, default 'last'
            Where to place events with missing sort values. 'last' puts them at the end, 'first' at the start.

        Returns
        -------
        VCatalog
            The sorted catalog (self if inplace=True, otherwise a new VCatalog)
        """
        from datetime import datetime
        def get_sort_value(event):
            if callable(key):
                value = key(event)
            elif key == 'magnitude':
                value = event.magnitudes[0].mag if event.magnitudes else None
            elif key == 'time':
                value = event.origins[0].time if event.origins else None
            elif key == 'depth':
                value = event.origins[0].depth if event.origins and event.origins[0].depth is not None else None
            elif key == 'latitude':
                value = event.origins[0].latitude if event.origins and event.origins[0].latitude is not None else None
            elif key == 'longitude':
                value = event.origins[0].longitude if event.origins and event.origins[0].longitude is not None else None
            else:
                raise ValueError(f"Unknown sort key: {key}")
            if value is None:
                # Decide where to put missing values
                if (ascending and missing == 'last') or (not ascending and missing == 'first'):
                    return float('inf')
                else:
                    return float('-inf')
            return value
        sorted_events = sorted(self.events, key=get_sort_value, reverse=not ascending)
        if inplace:
            self.events = sorted_events
            return self
        else:
            new_cat = deepcopy(self)
            new_cat.events = sorted_events
            return new_cat

    def extract_event_id(self, event, fallback_prefix="id", verbose=False):
        return self.get_eventid(event, fallback_prefix, verbose)

    def get_eventid(self, event, fallback_prefix="id", verbose=False):
        """
        Extract event ID from an ObsPy Event object. Includes logic for known syntax for different datasources.
        
        This method handles specific datasource patterns:
        - *.anss.org (UW, AV, etc.): Extracts network code + event number from resource_id
        - gfz-potsdam.de: Extracts event ID after last '/'
        - earthquake.usgs.gov: Extracts eventid parameter from query string
        - Default: Returns string after last '/'
        
        Parameters
        ----------
        event : obspy.core.event.Event
            The event object to extract ID from
        fallback_prefix : str, default "id"
            Prefix to use when falling back to index-based ID
        verbose : bool, default False
            If True, print diagnostic information about the extraction
            
        Returns
        -------
        str
            The extracted event ID
            
        Examples
        --------
        >>> catalog = VCatalog(events)
        >>> event_id = catalog.get_eventid(event)
        >>> # Returns "uw61501708", "av93023959", "noa2025daorl", "av93316013", etc.
        """
        # Check if event has resource_id.id
        if not (hasattr(event, 'resource_id') and event.resource_id is not None 
                and hasattr(event.resource_id, 'id') and event.resource_id.id):
            if verbose:
                print(f"No resource_id.id found for event")
            return None
            
        resource_id = event.resource_id.id
        datasource = "unknown"
        eventid = None
        
        # Handle *.anss.org datasources (UW, AV, etc.)
        if ".anss.org" in resource_id:
            # Example: quakeml:uw.anss.org/Event/UW/61501708#162404905471 -> "uw61501708"
            # Example: quakeml:av.anss.org/Event/AV/93023959#172895577681 -> "av93023959"
            try:
                # Extract network code (uw, av, etc.)
                network_part = resource_id.split(".anss.org")[0]
                network_code = network_part.split(":")[-1].lower()  # Get part after last ':'
                
                # Extract event number (part after last '/' before any '#')
                path_part = resource_id.split("/")[-1]
                event_number = path_part.split("#")[0]  # Remove fragment if present
                
                eventid = f"{network_code}{event_number}"
                datasource = f"{network_code}.anss.org"
            except (IndexError, AttributeError):
                # Fallback to default behavior
                eventid = resource_id.split("/")[-1].split("#")[0]
                datasource = "anss.org (parse error)"
                
        # Handle gfz-potsdam.de datasources
        elif "gfz-potsdam.de" in resource_id:
            # Example: smi:org.gfz-potsdam.de/geofon/noa2025daorl -> "noa2025daorl"
            eventid = resource_id.split("/")[-1]
            datasource = "gfz-potsdam.de"
            
        # Handle earthquake.usgs.gov datasources
        elif "earthquake.usgs.gov" in resource_id:
            # Example: quakeml:earthquake.usgs.gov/fdsnws/event/1/query?eventid=av93316013&format=quakeml -> "av93316013"
            try:
                # Look for eventid parameter in query string
                if "eventid=" in resource_id:
                    # Extract the eventid parameter value
                    eventid_part = resource_id.split("eventid=")[1]
                    eventid = eventid_part.split("&")[0]  # Remove any additional parameters
                else:
                    # Fallback to default behavior
                    eventid = resource_id.split("/")[-1].split("#")[0]
                datasource = "earthquake.usgs.gov"
            except (IndexError, AttributeError):
                # Fallback to default behavior
                eventid = resource_id.split("/")[-1].split("#")[0]
                datasource = "earthquake.usgs.gov (parse error)"
            
        # Default: return string after last '/'
        else:
            eventid = resource_id.split("/")[-1].split("#")[0]  # Remove fragment if present
            datasource = "unknown"
        
        if verbose:
            print(f"Resource ID: {resource_id}")
            print(f"Recognized datasource: {datasource}")
            print(f"Extracted eventid: {eventid}")
            
        return eventid

    def filter(self, magnitude=None, longitude=None, latitude=None, depth=None, time=None,
               standard_error=None, azimuthal_gap=None, used_station_count=None, used_phase_count=None,
               inverse=False):
        # TODO Implement remaining parameters
        # TODO Add lat,lon,radius_km

        filter_params = []

        if magnitude is not None:
            filter_params.append("magnitude >= " + str(magnitude[0]))
            filter_params.append("magnitude <= " + str(magnitude[1]))

        if time is not None:
            from obspy import UTCDateTime
            t1 = UTCDateTime(time[0])
            t2 = UTCDateTime(time[1])
            filter_params.append("time >= " + t1.isoformat())
            filter_params.append("time <= " + t2.isoformat())

        # Call the parent class (ObsPy Catalog) filter method and return a new VCatalog
        filtered_catalog = super().filter(*filter_params, inverse=inverse)
        # Create a new instance of the same class type to avoid circular imports
        return type(self)(filtered_catalog)

    def get_picks(self, client, verbose=False):

        new_catalog = self.__class__()

        failed_requests = []

        for event in self.events:

            try:
                # Get event using eventid
                # eventid = event.resource_id.split("/")[-1]
                eventid = self.get_eventid(event, verbose=verbose)
                events = client.get_events(eventid=eventid)
                new_catalog += events
                if verbose:
                    print(f"  Retrieved event {eventid}")

            except Exception as e:
                print(f"  Failed to retrieve event {event.resource_id}: {e}")
                failed_requests.append(event.resource_id)
                continue

        if failed_requests:
            # print(f"Failed to retrieve {len(failed_requests)} events: {failed_requests}")
            print(f"Failed to retrieve {len(failed_requests)} events.")

        return new_catalog

    def get_waveforms(self, client, inventory=None, outdir=None, pre_t=2, post_t=18,
                              verbose=True, components=["Z", "N", "E"],
                              fallback_clients=None, client_timeout=30):
        """
        Retrieve waveforms for all events in a catalog using ObsPy with support for multiple data sources.

        This function downloads seismic waveforms for each event in the provided catalog
        based on the station picks associated with each event. It supports multiple clients
        for different networks/stations and includes fallback mechanisms.

        Parameters
        ----------
        self : VCatalog or obspy.core.event.Event
            VCatalog instance containing events with picks and origin information.
            If called on a single Event, it will be converted to a VCatalog automatically.
        client : obspy.clients.fdsn.Client, dict, or list
            Primary client(s) for waveform data access. Can be:
            - Single Client object: used for all requests
            - Dict mapping network codes to clients: {"IU": client1, "US": client2}
            - Dict mapping station codes to clients: {"ANMO": client1, "CCM": client2}
            - Dict mapping net.sta to clients: {"IU.ANMO": client1, "US.CCM": client2}
            - List of clients: tries each client in order until successful
        inventory : obspy.core.inventory.Inventory, optional
            Station inventory for metadata (currently unused but reserved for future use)
        outdir : str, optional
            Output directory for saving waveforms. If None, waveforms are not saved to disk.
        pre_t : float, default 2
            Time in seconds before event origin time to start waveform window
        post_t : float, default 18
            Time in seconds after event origin time to end waveform window
        verbose : bool, default True
            Enable detailed logging output
        components : list, default ["Z", "N", "E"]
            List of component codes to retrieve (e.g., ["Z"] for vertical only)
        fallback_clients : list or dict, optional
            Additional clients to try if primary client fails. Can be:
            - List of clients: tries each in order
            - Dict with same mapping options as primary client
        client_timeout : float, default 30
            Timeout in seconds for individual client requests

        Returns
        -------
        list of obspy.core.stream.Stream
            List containing one Stream object per event, with all retrieved waveforms
        dict
            Dictionary with retrieval statistics and client usage information

        Raises
        ------
        ValueError
            If catalog is empty or client configuration is invalid
        TypeError
            If client is not a supported type

        Examples
        --------
        >>> from obspy.clients.fdsn import Client
        >>> from obspy import read_events
        >>>
        >>> # Single client with VCatalog
        >>> cat = VCatalog(read_events("my_catalog.xml"))
        >>> client = Client("IRIS")
        >>> streams, stats = cat.get_waveforms(client)
        >>>
        >>> # Single event usage
        >>> event = cat[0]  # Get first event
        >>> streams, stats = event.get_waveforms(client)  # Works on single Event too
        >>>
        >>> # Multiple clients by network
        >>> clients = {
        ...     "IU": Client("IRIS"),
        ...     "US": Client("USGS"),
        ...     "CI": Client("SCEDC")
        ... }
        >>> streams, stats = cat.get_waveforms(clients)
        >>>
        >>> # List of clients with fallbacks
        >>> primary_clients = [Client("IRIS"), Client("USGS")]
        >>> fallback_clients = [Client("SCEDC"), Client("NCEDC")]
        >>> streams, stats = cat.get_waveforms(primary_clients,
        ...                                  fallback_clients=fallback_clients)
        >>>
        >>> # Mixed approach: network mapping with fallback list
        >>> network_clients = {"IU": Client("IRIS"), "US": Client("USGS")}
        >>> fallbacks = [Client("SCEDC"), Client("GEOFON")]
        >>> streams, stats = cat.get_waveforms(network_clients,
        ...                                  fallback_clients=fallbacks)
        """

        import os
        from obspy import Stream
        from obspy.core.event import Catalog
        from collections import defaultdict

        # Handle case where self is a single Event instead of a VCatalog
        if isinstance(self, Event):
            if verbose:
                print("Input is a single Event, converting to VCatalog")
            from vdapseisutils.utils.obspyutils.catalog.core import VCatalog
            temp_catalog = VCatalog()
            temp_catalog.append(self)
            cat = temp_catalog
        else:
            # self is already a VCatalog instance
            cat = self

        if verbose:
            print(">>> utils.catalog_utils.get_catalog_waveforms()")
            print(f"Processing {len(cat)} events from catalog")
            print(f"Time window: {pre_t}s before to {post_t}s after origin")
            print(f"Components to retrieve: {components}")
            if outdir is not None:
                print(f"Output directory: {outdir}")
            else:
                print("Waveforms will not be saved to disk")

        # Input validation
        if not cat:
            raise ValueError("Catalog is empty")

        # Initialize statistics tracking
        stats = {
            'total_events': len(cat),
            'successful_events': 0,
            'total_waveforms': 0,
            'client_usage': defaultdict(int),
            'network_stats': defaultdict(lambda: {'attempts': 0, 'successes': 0}),
            'failed_requests': 0,
            'events_with_data': 0
        }

        # Parse and validate client configuration
        def _get_client_for_request(net, sta, nsl_key):
            """
            Determine which client to use for a specific network/station request.
            Returns list of clients to try in order.
            """
            clients_to_try = []

            # Handle primary client(s)
            if isinstance(client, dict):
                # Try different mapping strategies in order of specificity
                keys_to_try = [nsl_key, f"{net}.{sta}", sta, net]
                for key in keys_to_try:
                    if key in client:
                        clients_to_try.append(client[key])
                        if verbose and len(clients_to_try) == 1:
                            print(f"    Using mapped client for key: {key}")
                        break
            elif isinstance(client, list):
                clients_to_try.extend(client)
            elif hasattr(client, 'get_waveforms'):
                clients_to_try.append(client)
            else:
                raise TypeError(f"Client must be Client object, dict, or list, got {type(client)}")

            # Add fallback clients
            if fallback_clients:
                if isinstance(fallback_clients, dict):
                    keys_to_try = [nsl_key, f"{net}.{sta}", sta, net]
                    for key in keys_to_try:
                        if key in fallback_clients:
                            clients_to_try.append(fallback_clients[key])
                elif isinstance(fallback_clients, list):
                    clients_to_try.extend(fallback_clients)
                elif hasattr(fallback_clients, 'get_waveforms'):
                    clients_to_try.append(fallback_clients)

            return clients_to_try

        def _validate_client(client_obj):
            """Validate that a client object has the required methods."""
            if not hasattr(client_obj, 'get_waveforms'):
                return False
            return True

        # Display client configuration
        if verbose:
            print(f"\n--- Client Configuration ---")
            if isinstance(client, dict):
                print(f"Primary client mapping: {len(client)} entries")
                for key, cli in list(client.items())[:3]:  # Show first 3
                    client_name = getattr(cli, 'base_url', str(cli))
                    print(f"  {key} -> {client_name}")
                if len(client) > 3:
                    print(f"  ... and {len(client) - 3} more")
            elif isinstance(client, list):
                print(f"Primary client list: {len(client)} clients")
                for i, cli in enumerate(client[:3]):
                    client_name = getattr(cli, 'base_url', str(cli))
                    print(f"  {i + 1}. {client_name}")
            else:
                client_name = getattr(client, 'base_url', str(client))
                print(f"Single primary client: {client_name}")

            if fallback_clients:
                if isinstance(fallback_clients, (list, dict)):
                    count = len(fallback_clients)
                    print(f"Fallback clients available: {count}")
                else:
                    print("Single fallback client available")

        # Ensure output directory exists if saving waveforms
        if outdir is not None and not os.path.exists(outdir):
            os.makedirs(outdir)
            if verbose:
                print(f"Created output directory: {outdir}")

        # Initialize containers
        stream_list = []

        # Process each event in the catalog
        for event_idx, event in enumerate(cat):
            if verbose:
                print(f"\n--- Processing event {event_idx + 1}/{len(cat)} ---")

            # Validate event has required information
            if not event.origins:
                print(f"WARNING: Event {event_idx + 1} has no origins, skipping...")
                stream_list.append(Stream())
                continue

            if not event.picks:
                print(f"WARNING: Event {event_idx + 1} has no picks, skipping...")
                stream_list.append(Stream())
                continue

            # Initialize stream for this event
            event_stream = Stream()

            # Extract event information
            try:
                if hasattr(event.resource_id, 'resource_id'):
                    rid_parts = event.resource_id.resource_id.split("/")
                else:
                    rid_parts = str(event.resource_id).split("/")

                if len(rid_parts) >= 2:
                    filename = rid_parts[-2] + "_" + rid_parts[-1]
                else:
                    filename = f"event_{event_idx:04d}"

            except (AttributeError, IndexError):
                filename = f"event_{event_idx:04d}"
                if verbose:
                    print(f"WARNING: Could not parse resource_id, using {filename}")

            if verbose:
                print(f"Event ID: {filename}")

            # Define time window
            origin_time = event.origins[0].time
            starttime = origin_time - pre_t
            endtime = origin_time + post_t

            if verbose:
                print(f"Origin time: {origin_time}")
                print(f"Data window: {starttime} to {endtime}")
                print(f"Found {len(event.picks)} picks for this event")

            # Extract station information and pick times
            station_pick_times = {}  # {net.sta: {'picks': [times], 'locations': set([locs])}}

            for pick in event.picks:
                try:
                    pick_id = pick.waveform_id.id
                    net, sta, loc, cha = pick_id.split(".")
                    loc = "--" if loc == "" else loc

                    # Get pick time
                    pick_time = pick.time
                    if not pick_time:
                        if verbose:
                            print(f"WARNING: Pick has no time, skipping: {pick_id}")
                        continue

                    # Group by net.sta to track pick times and locations per station
                    net_sta_key = f"{net}.{sta}"
                    if net_sta_key not in station_pick_times:
                        station_pick_times[net_sta_key] = {
                            'picks': [],
                            'locations': set(),
                            'net': net,
                            'sta': sta
                        }

                    station_pick_times[net_sta_key]['picks'].append(pick_time)
                    station_pick_times[net_sta_key]['locations'].add(loc)

                except (AttributeError, ValueError) as e:
                    if verbose:
                        print(f"WARNING: Could not parse pick: {e}")
                    continue

            if verbose:
                print(f"Stations with picks: {len(station_pick_times)}")

            # Calculate time windows for each station based on pick times
            station_time_windows = {}
            for net_sta_key, pick_data in station_pick_times.items():
                pick_times = pick_data['picks']
                if not pick_times:
                    continue

                # Find min and max pick times for this station
                min_pick_time = min(pick_times)
                max_pick_time = max(pick_times)

                # Apply buffering
                station_starttime = min_pick_time - pre_t
                station_endtime = max_pick_time + post_t

                station_time_windows[net_sta_key] = {
                    'starttime': station_starttime,
                    'endtime': station_endtime,
                    'min_pick': min_pick_time,
                    'max_pick': max_pick_time,
                    'pick_count': len(pick_times),
                    'locations': pick_data['locations'],
                    'net': pick_data['net'],
                    'sta': pick_data['sta']
                }

                if verbose:
                    duration = station_endtime - station_starttime
                    pick_span = max_pick_time - min_pick_time if len(pick_times) > 1 else 0
                    print(f"  {net_sta_key}: {len(pick_times)} picks, span={pick_span:.1f}s, window={duration:.1f}s")
                    print(f"    Pick range: {min_pick_time} to {max_pick_time}")
                    print(f"    Data window: {station_starttime} to {station_endtime}")
                    print(f"    Locations: {sorted(pick_data['locations'])}")

            # Retrieve waveforms for each station using station-specific time windows
            waveforms_for_event = 0
            failed_requests_for_event = 0

            for net_sta_key, time_window in station_time_windows.items():
                net = time_window['net']
                sta = time_window['sta']
                station_starttime = time_window['starttime']
                station_endtime = time_window['endtime']
                locations = time_window['locations']

                if verbose:
                    print(f"\n  Processing station: {net_sta_key}")
                    print(f"    Time window: {station_starttime} to {station_endtime}")

                # Update network statistics
                stats['network_stats'][net]['attempts'] += len(components) * len(locations)

                # Request data for each location at this station
                for loc in locations:
                    for comp in components:
                        channel_pattern = f"*{comp}"
                        request_successful = False

                        if verbose:
                            print(f"    Requesting: {net}.{sta}.{loc}.{channel_pattern}")

                        # Get list of clients to try for this request
                        nsl_key = f"{net}.{sta}.{loc}"
                        clients_to_try = _get_client_for_request(net, sta, nsl_key)

                        # Try each client until successful
                        for client_idx, current_client in enumerate(clients_to_try):
                            if not _validate_client(current_client):
                                if verbose:
                                    print(f"      WARNING: Invalid client object, skipping")
                                continue

                            try:
                                # Set timeout if client supports it
                                if hasattr(current_client, 'set_timeout'):
                                    current_client.set_timeout(client_timeout)

                                # Get client identifier for statistics
                                client_id = getattr(current_client, 'base_url', f"client_{id(current_client)}")

                                if verbose and len(clients_to_try) > 1:
                                    print(f"      Trying client {client_idx + 1}/{len(clients_to_try)}: {client_id}")

                                # Request waveform data using station-specific time window
                                tmp_stream = current_client.get_waveforms(
                                    network=net,
                                    station=sta,
                                    location=loc,
                                    channel=channel_pattern,
                                    starttime=station_starttime,
                                    endtime=station_endtime
                                )

                                if tmp_stream:
                                    event_stream += tmp_stream
                                    waveforms_for_event += len(tmp_stream)
                                    stats['client_usage'][client_id] += len(tmp_stream)
                                    stats['network_stats'][net]['successes'] += 1
                                    request_successful = True

                                    if verbose:
                                        duration = station_endtime - station_starttime
                                        print(
                                            f"      SUCCESS: Retrieved {len(tmp_stream)} trace(s) from {client_id} ({duration:.1f}s window)")
                                    break  # Success, don't try other clients
                                else:
                                    if verbose:
                                        print(f"      No data returned from {client_id}")

                            except Exception as e:
                                client_id = getattr(current_client, 'base_url', f"client_{id(current_client)}")
                                if verbose:
                                    print(f"      FAILED with {client_id}: {type(e).__name__}: {str(e)}")

                                # If this was the last client to try, count as failed
                                if client_idx == len(clients_to_try) - 1:
                                    failed_requests_for_event += 1
                                    stats['failed_requests'] += 1

                        if not request_successful and verbose:
                            print(f"      All clients failed for {net}.{sta}.{loc}.{channel_pattern}")

            # Summary for this event
            if verbose:
                print(
                    f"Event summary: {waveforms_for_event} waveforms retrieved, {failed_requests_for_event} requests failed")

            if waveforms_for_event > 0:
                stats['successful_events'] += 1
                stats['events_with_data'] += 1
                stats['total_waveforms'] += waveforms_for_event

            # Optional: Save waveforms to disk
            if outdir is not None and event_stream:
                try:
                    output_file = os.path.join(outdir, f"{filename}.mseed")
                    event_stream.write(output_file, format="MSEED")
                    if verbose:
                        print(f"Saved waveforms to: {output_file}")
                except Exception as e:
                    print(f"ERROR: Could not save waveforms for {filename}: {e}")

            # Add stream to list (even if empty to maintain event indexing)
            stream_list.append(event_stream)

        # Final summary
        if verbose:
            print(f"\n=== PROCESSING COMPLETE ===")
            print(f"Total events processed: {stats['total_events']}")
            print(f"Events with waveforms: {stats['events_with_data']}")
            print(f"Total waveforms retrieved: {stats['total_waveforms']}")
            if stats['events_with_data'] > 0:
                print(
                    f"Average waveforms per successful event: {stats['total_waveforms'] / stats['events_with_data']:.1f}")
            print(f"Failed requests: {stats['failed_requests']}")

            print(f"\n--- Client Usage Statistics ---")
            for client_id, count in stats['client_usage'].items():
                print(f"  {client_id}: {count} waveforms")

            print(f"\n--- Network Statistics ---")
            for net, net_stats in stats['network_stats'].items():
                success_rate = (net_stats['successes'] / net_stats['attempts'] * 100) if net_stats[
                                                                                             'attempts'] > 0 else 0
                print(f"  {net}: {net_stats['successes']}/{net_stats['attempts']} ({success_rate:.1f}% success)")

        return stream_list, stats

    def add_network_code(self, channel_network_map=None, default_network=None, inplace=True):
        """
        Add network codes to WaveformStreamID objects for each pick in all events.
        
        This method updates the network field of WaveformStreamID objects based on either:
        1. A channel-to-network mapping dictionary, or
        2. A default network applied to all picks
        
        Parameters
        ----------
        channel_network_map : dict, optional
            Dictionary mapping channel codes to network codes.
            Example: {"SPBG": "AV", "SPU": "AV", "A64K": "AK"}
        default_network : str, optional
            Network code to apply to all picks if channel_network_map is not provided
            or if a channel is not found in the mapping
        inplace : bool, default True
            If True, modify the catalog in place. If False, return a new catalog.
            
        Returns
        -------
        VCatalog or None
            If inplace=False, returns a new VCatalog with updated network codes.
            If inplace=True, returns None and modifies the current catalog.
            
        Raises
        ------
        ValueError
            If neither channel_network_map nor default_network is provided
            
        Examples
        --------
        >>> # Using channel mapping
        >>> channel_map = {"SPBG": "AV", "SPU": "AV", "A64K": "AK"}
        >>> catalog.add_network(channel_network_map=channel_map)
        
        >>> # Using default network for all picks
        >>> catalog.add_network(default_network="AV")
        
        >>> # Using channel mapping with fallback
        >>> channel_map = {"SPBG": "AV", "SPU": "AV"}
        >>> catalog.add_network(channel_network_map=channel_map, default_network="AK")
        """
        if channel_network_map is None and default_network is None:
            raise ValueError("Either channel_network_map or default_network must be provided")
            
        # Work on copy if not inplace
        if inplace:
            target_catalog = self
        else:
            target_catalog = deepcopy(self)
            
        # Track statistics
        updated_picks = 0
        total_picks = 0
        network_assignments = {}
        
        # Process each event
        for event in target_catalog.events:
            if not event.picks:
                continue
                
            # Process each pick in the event
            for pick in event.picks:
                total_picks += 1
                
                if not hasattr(pick, 'waveform_id') or pick.waveform_id is None:
                    continue
                    
                # Extract channel code from waveform_id
                try:
                    waveform_id_parts = pick.waveform_id.id.split(".")
                    if len(waveform_id_parts) >= 4:
                        current_net, sta, loc, cha = waveform_id_parts[:4]
                    else:
                        # Handle cases where full NSLC is not available
                        cha = waveform_id_parts[-1]  # Use last part as channel
                        current_net = pick.waveform_id.network_code if hasattr(pick.waveform_id, 'network_code') else ""
                        sta = pick.waveform_id.station_code if hasattr(pick.waveform_id, 'station_code') else ""
                        loc = pick.waveform_id.location_code if hasattr(pick.waveform_id, 'location_code') else ""
                        
                except (AttributeError, ValueError, IndexError):
                    # If we can't parse the channel, skip this pick
                    continue
                
                # Determine which network to assign
                new_network = None
                
                if channel_network_map and cha in channel_network_map:
                    new_network = channel_network_map[cha]
                elif default_network:
                    new_network = default_network
                    
                # Update the network if we found one
                if new_network:
                    pick.waveform_id.network_code = new_network
                    updated_picks += 1
                    
                    # Track statistics
                    if new_network not in network_assignments:
                        network_assignments[new_network] = []
                    network_assignments[new_network].append(cha)
        
        # Print summary if there were updates
        if updated_picks > 0:
            print(f"Updated {updated_picks}/{total_picks} picks with network codes")
            for network, channels in network_assignments.items():
                unique_channels = list(set(channels))
                print(f"  {network}: {len(channels)} picks across channels {unique_channels}")
        else:
            print("No picks were updated")
            
        if not inplace:
            return target_catalog

    def assign_phase_hint(self, component_phase_map=None, inplace=True, verbose=False):
        """
        Assign phase hints to picks based on the component (last character) of the channel code.
        
        This method updates the phase_hint attribute of Pick objects based on the component
        of their channel code. This is useful for automatically assigning P-wave picks to
        vertical components (Z) and S-wave picks to horizontal components (N, E).
        
        Parameters
        ----------
        component_phase_map : dict, optional
            Dictionary mapping channel components to phase hints.
            Default: {"Z": "P", "N": "S", "E": "S"}
        inplace : bool, default True
            If True, modify the catalog in place. If False, return a new catalog.
        verbose : bool, default False
            If True, print detailed information about the assignment process
            
        Returns
        -------
        VCatalog or None
            If inplace=False, returns a new VCatalog with updated phase hints.
            If inplace=True, returns None and modifies the current catalog.
            
        Examples
        --------
        >>> # Using default mapping (Z->P, N->S, E->S)
        >>> catalog.assign_phase_hint()
        
        >>> # Custom mapping
        >>> custom_map = {"Z": "P", "N": "S", "E": "S", "1": "P", "2": "S", "3": "S"}
        >>> catalog.assign_phase_hint(component_phase_map=custom_map)
        
        >>> # With verbose output
        >>> catalog.assign_phase_hint(verbose=True)
        """
        from copy import deepcopy
        
        # Set default mapping if none provided
        if component_phase_map is None:
            component_phase_map = {"Z": "P", "N": "S", "E": "S"}
        
        # Work on copy if not inplace
        if inplace:
            target_catalog = self
        else:
            target_catalog = deepcopy(self)
        
        # Track statistics
        updated_picks = 0
        total_picks = 0
        phase_assignments = {}
        
        if verbose:
            print(">>> VCatalogUtilsMixin.assign_phase_hint()")
            print(f"Component to phase mapping: {component_phase_map}")
            print(f"Processing {len(target_catalog.events)} events...")
        
        # Process each event
        for event_idx, event in enumerate(target_catalog.events):
            if not event.picks:
                continue
            
            event_updates = 0
            if verbose:
                print(f"\n--- Event {event_idx + 1}/{len(target_catalog.events)} ---")
                print(f"Processing {len(event.picks)} picks")
            
            # Process each pick in the event
            for pick in event.picks:
                total_picks += 1
                
                if not hasattr(pick, 'waveform_id') or pick.waveform_id is None:
                    continue
                
                try:
                    # Extract channel code from waveform_id
                    waveform_parts = pick.waveform_id.id.split(".")
                    if len(waveform_parts) >= 4:
                        full_channel = waveform_parts[3]  # Channel is 4th part (NSLC format)
                    elif len(waveform_parts) >= 1:
                        full_channel = waveform_parts[-1]  # Use last part as channel
                    else:
                        full_channel = pick.waveform_id.channel_code if hasattr(pick.waveform_id, 'channel_code') else ""
                    
                    # Get component (last character of channel)
                    component = full_channel[-1] if full_channel else ""
                    
                    # Check if we have a mapping for this component
                    if component in component_phase_map:
                        new_phase_hint = component_phase_map[component]
                        old_phase_hint = pick.phase_hint if hasattr(pick, 'phase_hint') else None
                        
                        # Update the phase hint
                        pick.phase_hint = new_phase_hint
                        updated_picks += 1
                        event_updates += 1
                        
                        # Track statistics
                        if new_phase_hint not in phase_assignments:
                            phase_assignments[new_phase_hint] = {'count': 0, 'components': set()}
                        phase_assignments[new_phase_hint]['count'] += 1
                        phase_assignments[new_phase_hint]['components'].add(component)
                        
                        if verbose:
                            station = waveform_parts[1] if len(waveform_parts) >= 2 else "?"
                            print(f"  {station}.{full_channel}: {old_phase_hint} -> {new_phase_hint}")
                    
                except (AttributeError, ValueError, IndexError):
                    # Skip picks that can't be parsed
                    continue
            
            if verbose and event_updates > 0:
                print(f"Updated {event_updates} picks in this event")
        
        # Print summary
        if verbose or updated_picks > 0:
            print(f"\n=== PHASE HINT ASSIGNMENT COMPLETE ===")
            print(f"Updated {updated_picks}/{total_picks} picks with phase hints")
            for phase_hint, stats in phase_assignments.items():
                components_list = sorted(list(stats['components']))
                print(f"  {phase_hint}: {stats['count']} picks from components {components_list}")
        
        if not inplace:
            return target_catalog

    def sort_picks(self, inplace=True, verbose=False):
        """
        Sort picks by earliest arrival time for each event in the catalog.

        For each event in the catalog, if there are picks, sort them by their arrival time
        (pick.time) in ascending order (earliest first).

        Parameters
        ----------
        inplace : bool, default True
            If True, modify the catalog in place. If False, return a new catalog.
        verbose : bool, default False
            If True, print detailed information about the sorting process

        Returns
        -------
        VCatalog or None
            If inplace=False, returns a new VCatalog with sorted picks.
            If inplace=True, returns None and modifies the current catalog.

        Examples
        --------
        >>> # Sort picks in place
        >>> catalog.sort_picks()
        
        >>> # Get new catalog with sorted picks
        >>> sorted_catalog = catalog.sort_picks(inplace=False)
        
        >>> # With verbose output
        >>> catalog.sort_picks(verbose=True)
        
        >>> # Can also be called on a single Event object
        >>> from obspy.core.event.event import Event
        >>> event = Event()  # some event with picks
        >>> sorted_catalog = catalog.sort_picks(event)  # Converts Event to VCatalog first
        """
        from obspy.core.event.event import Event
        
        # Handle case where self is a single Event instead of a VCatalog
        if isinstance(self, Event):
            if verbose:
                print("Input is a single Event, converting to VCatalog")
            from vdapseisutils.utils.obspyutils.catalog.core import VCatalog
            temp_catalog = VCatalog()
            temp_catalog.append(self)
            target_catalog = temp_catalog
            # For single Event input, always return the catalog (don't modify in place)
            inplace = False
        else:
            # Work on copy if not inplace
            if inplace:
                target_catalog = self
            else:
                target_catalog = deepcopy(self)

        # Track statistics
        total_events = len(target_catalog.events)
        events_with_picks = 0
        total_picks_sorted = 0
        
        if verbose:
            print(">>> VCatalogUtilsMixin.sort_picks()")
            print(f"Processing {total_events} events...")

        # Process each event
        events_with_picks_counter = [0]  # Use list for mutable reference
        total_picks_sorted_counter = [0]  # Use list for mutable reference
        
        for event_idx, event in enumerate(target_catalog.events):
            self._sort_picks_for_event(event, event_idx, verbose, events_with_picks_counter, total_picks_sorted_counter)
        
        events_with_picks = events_with_picks_counter[0]
        total_picks_sorted = total_picks_sorted_counter[0]

        # Print summary
        if verbose or events_with_picks > 0:
            print(f"\n=== PICK SORTING COMPLETE ===")
            print(f"Events processed: {total_events}")
            print(f"Events with picks: {events_with_picks}")
            print(f"Total picks sorted: {total_picks_sorted}")

        if not inplace:
            return target_catalog

    def _sort_picks_for_event(self, event, event_idx=0, verbose=False, events_with_picks_ref=None, total_picks_sorted_ref=None):
        """
        Helper method to sort picks for a single event.
        
        Parameters
        ----------
        event : Event
            The event whose picks should be sorted
        event_idx : int, default 0
            Index of the event for verbose output
        verbose : bool, default False
            If True, print detailed information
        events_with_picks_ref : list, optional
            Reference to counter for events with picks (for statistics)
        total_picks_sorted_ref : list, optional
            Reference to counter for total picks sorted (for statistics)
        """
        if not event.picks:
            if verbose:
                print(f"Event {event_idx + 1}: No picks to sort")
            return
            
        if events_with_picks_ref is not None:
            events_with_picks_ref[0] += 1
            
        pick_count = len(event.picks)
        
        if verbose:
            print(f"Event {event_idx + 1}: Sorting {pick_count} picks")
            
        # Filter out picks without time information
        picks_with_time = []
        picks_without_time = []
        
        for pick in event.picks:
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
            if total_picks_sorted_ref is not None:
                total_picks_sorted_ref[0] += len(picks_with_time)
            
            if verbose:
                earliest_time = picks_with_time[0].time
                latest_time = picks_with_time[-1].time
                time_span = latest_time - earliest_time
                print(f"  Time range: {earliest_time} to {latest_time} (span: {time_span:.2f}s)")
        
        # Combine sorted picks with time + picks without time at the end
        event.picks = picks_with_time + picks_without_time
        
        if verbose and pick_count > 1:
            print(f"  Sorted {len(picks_with_time)} picks by time, {len(picks_without_time)} without time placed at end")

    def sort_picks_for_event(self, event_index, verbose=False):
        """
        Sort picks for a specific event by index in the catalog.
        
        This method directly modifies the original event in the catalog, ensuring
        that changes are persistent.
        
        Parameters
        ----------
        event_index : int
            Index of the event in the catalog to sort picks for
        verbose : bool, default False
            If True, print detailed information about the sorting process
            
        Examples
        --------
        >>> # Sort picks for event at index 122
        >>> catalog.sort_picks_for_event(122)
        
        >>> # With verbose output
        >>> catalog.sort_picks_for_event(122, verbose=True)
        """
        if event_index < 0 or event_index >= len(self.events):
            raise IndexError(f"Event index {event_index} is out of range. Catalog has {len(self.events)} events.")
        
        event = self.events[event_index]
        self._sort_picks_for_event(event, event_index, verbose)

