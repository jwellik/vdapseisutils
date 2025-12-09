def get_catalog_waveforms(cat, client, inventory=None, outdir="./", pre_t=2, post_t=18,
                          verbose=True, save_waveforms=False, components=["Z", "N", "E"],
                          fallback_clients=None, client_timeout=30):
    """
    Retrieve waveforms for all events in a catalog using ObsPy with support for multiple data sources.

    This function downloads seismic waveforms for each event in the provided catalog
    based on the station picks associated with each event. It supports multiple clients
    for different networks/stations and includes fallback mechanisms.

    Parameters
    ----------
    cat : obspy.core.event.Catalog
        ObsPy catalog containing events with picks and origin information
    client : obspy.clients.fdsn.Client, dict, or list
        Primary client(s) for waveform data access. Can be:
        - Single Client object: used for all requests
        - Dict mapping network codes to clients: {"IU": client1, "US": client2}
        - Dict mapping station codes to clients: {"ANMO": client1, "CCM": client2}
        - Dict mapping net.sta to clients: {"IU.ANMO": client1, "US.CCM": client2}
        - List of clients: tries each client in order until successful
    inventory : obspy.core.inventory.Inventory, optional
        Station inventory for metadata (currently unused but reserved for future use)
    outdir : str, default "./"
        Output directory for saving waveforms (if save_waveforms=True)
    pre_t : float, default 2
        Time in seconds before event origin time to start waveform window
    post_t : float, default 18
        Time in seconds after event origin time to end waveform window
    verbose : bool, default True
        Enable detailed logging output
    save_waveforms : bool, default False
        Whether to save waveforms to disk (not yet implemented)
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
    >>> # Single client (original behavior)
    >>> cat = read_events("my_catalog.xml")
    >>> client = Client("IRIS")
    >>> streams, stats = get_catalog_waveforms(cat, client)
    >>>
    >>> # Multiple clients by network
    >>> clients = {
    ...     "IU": Client("IRIS"),
    ...     "US": Client("USGS"),
    ...     "CI": Client("SCEDC")
    ... }
    >>> streams, stats = get_catalog_waveforms(cat, clients)
    >>>
    >>> # List of clients with fallbacks
    >>> primary_clients = [Client("IRIS"), Client("USGS")]
    >>> fallback_clients = [Client("SCEDC"), Client("NCEDC")]
    >>> streams, stats = get_catalog_waveforms(cat, primary_clients,
    ...                                       fallback_clients=fallback_clients)
    >>>
    >>> # Mixed approach: network mapping with fallback list
    >>> network_clients = {"IU": Client("IRIS"), "US": Client("USGS")}
    >>> fallbacks = [Client("SCEDC"), Client("GEOFON")]
    >>> streams, stats = get_catalog_waveforms(cat, network_clients,
    ...                                       fallback_clients=fallbacks)
    """

    import os
    from obspy import Stream
    from obspy.core.event import Catalog
    from collections import defaultdict

    if verbose:
        print(">>> utils.catalog_utils.get_catalog_waveforms()")
        print(f"Processing {len(cat)} events from catalog")
        print(f"Time window: {pre_t}s before to {post_t}s after origin")
        print(f"Components to retrieve: {components}")
        print(f"Output directory: {outdir}")

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
    if save_waveforms and not os.path.exists(outdir):
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
        if save_waveforms and event_stream:
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
            print(f"Average waveforms per successful event: {stats['total_waveforms'] / stats['events_with_data']:.1f}")
        print(f"Failed requests: {stats['failed_requests']}")

        print(f"\n--- Client Usage Statistics ---")
        for client_id, count in stats['client_usage'].items():
            print(f"  {client_id}: {count} waveforms")

        print(f"\n--- Network Statistics ---")
        for net, net_stats in stats['network_stats'].items():
            success_rate = (net_stats['successes'] / net_stats['attempts'] * 100) if net_stats['attempts'] > 0 else 0
            print(f"  {net}: {net_stats['successes']}/{net_stats['attempts']} ({success_rate:.1f}% success)")

    return stream_list, stats


# Example usage functions
def example_single_client():
    """Example: Single client usage (original behavior)."""
    from obspy.clients.fdsn import Client
    from obspy import read_events

    cat = read_events("catalog.xml")  # Your catalog file
    client = Client("IRIS")

    streams, stats = get_catalog_waveforms(cat, client, verbose=True)
    return streams, stats


def example_network_mapping():
    """Example: Map networks to specific data centers."""
    from obspy.clients.fdsn import Client
    from obspy import read_events

    cat = read_events("catalog.xml")

    # Map network codes to appropriate data centers
    clients = {
        "IU": Client("IRIS"),  # Global Seismographic Network
        "US": Client("USGS"),  # US National networks
        "CI": Client("SCEDC"),  # Southern California
        "NC": Client("NCEDC"),  # Northern California
        "UW": Client("IRIS"),  # Pacific Northwest
        "AK": Client("IRIS"),  # Alaska
    }

    streams, stats = get_catalog_waveforms(cat, clients, verbose=True)
    return streams, stats


def example_station_mapping():
    """Example: Map specific stations to data centers."""
    from obspy.clients.fdsn import Client
    from obspy import read_events

    cat = read_events("catalog.xml")

    # Map specific stations to data centers (useful for temporary deployments)
    clients = {
        "ANMO": Client("IRIS"),  # Specific station
        "CCM": Client("USGS"),  # Another specific station
        "PAS": Client("SCEDC"),  # Caltech station
        # Can also use net.sta format
        "CI.PAS": Client("SCEDC"),
        "IU.ANMO": Client("IRIS"),
    }

    fallback_clients = [Client("IRIS"), Client("GEOFON")]  # Try these if no mapping found

    streams, stats = get_catalog_waveforms(cat, clients,
                                           fallback_clients=fallback_clients,
                                           verbose=True)
    return streams, stats


def example_client_list_with_fallbacks():
    """Example: List of clients with fallback options."""
    from obspy.clients.fdsn import Client
    from obspy import read_events

    cat = read_events("catalog.xml")

    # Primary clients (tried in order)
    primary_clients = [
        Client("IRIS"),
        Client("USGS"),
    ]

    # Fallback clients (tried if primary fails)
    fallback_clients = [
        Client("SCEDC"),
        Client("NCEDC"),
        Client("GEOFON")
    ]

    streams, stats = get_catalog_waveforms(cat, primary_clients,
                                           fallback_clients=fallback_clients,
                                           verbose=True)
    return streams, stats


if __name__ == "__main__":
    print("Multi-client catalog waveform retrieval examples:")
    print("1. example_single_client()")
    print("2. example_network_mapping()")
    print("3. example_station_mapping()")
    print("4. example_client_list_with_fallbacks()")
    print("\nReplace catalog.xml with your actual catalog file.")
