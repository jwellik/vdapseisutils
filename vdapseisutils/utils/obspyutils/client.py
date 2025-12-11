import os
from pathlib import Path
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.filesystem.sds import Client as SDSClient
from obspy.clients.earthworm import Client as EarthwormClient
from obspy.clients.seedlink import Client as SeedlinkClient


class VClient:
    """
    Extended Client class that auto-detects client type based on input arguments.

    This is a wrapper that creates the appropriate ObsPy client based on the input.
    """

    def __init__(self, *args, client_type=None, **kwargs):
        """
        Initialize VClient with automatic client type detection.

        Parameters:
        -----------
        *args : tuple
            Positional arguments passed to the underlying client
        client_type : str, optional
            Force a specific client type. Options: 'fdsn', 'sds', 'earthworm', 'seedlink'
            If None, will auto-detect based on first argument
        **kwargs : dict
            Keyword arguments passed to the underlying client

        Examples:
        ---------
        VClient("/path/to/SDS")           # Auto-detects SDS filesystem
        VClient("IRIS")                   # Auto-detects FDSN client
        VClient("localhost", port=16022)  # Auto-detects based on port (earthworm)
        VClient("iris.edu", client_type="fdsn")  # Force FDSN type
        """

        self.client_type = client_type
        self._client = None
        self.use_bulk = True  # Default, will be set below
        if client_type:
            # Client type explicitly specified
            self._client = self._create_client_by_type(client_type, *args, **kwargs)
        else:
            # Auto-detect client type
            self._client = self._auto_detect_client(*args, **kwargs)
        # Set use_bulk property based on client type
        ctype = self.get_client_type()
        if ctype == 'EarthwormClient':
            self.use_bulk = False
        else:
            self.use_bulk = True

    def _auto_detect_client(self, *args, **kwargs):
        """Auto-detect the appropriate client type based on arguments."""

        if not args:
            # No positional arguments, default to FDSN with IRIS
            return FDSNClient("IRIS", **kwargs)

        first_arg = args[0]

        if isinstance(first_arg, str):
            # Check if it's a filesystem path
            if self._is_filesystem_path(first_arg):
                print(f"Detected SDS filesystem: {first_arg}")
                return SDSClient(first_arg, **kwargs)

            # Check for earthworm indicators
            if self._is_earthworm_client(*args, **kwargs):
                print(f"Detected Earthworm client: {first_arg}")
                return EarthwormClient(*args, **kwargs)

            # Check for seedlink indicators
            if self._is_seedlink_client(*args, **kwargs):
                print(f"Detected SeedLink client: {first_arg}")
                return SeedlinkClient(*args, **kwargs)

            # Default to FDSN client
            print(f"Detected FDSN client: {first_arg}")
            return FDSNClient(first_arg, **kwargs)

        else:
            # Non-string first argument, try FDSN
            return FDSNClient(*args, **kwargs)

    def _create_client_by_type(self, client_type, *args, **kwargs):
        """Create client of specified type."""

        client_map = {
            'fdsn': FDSNClient,
            'sds': SDSClient,
            'earthworm': EarthwormClient,
            'seedlink': SeedlinkClient
        }

        if client_type.lower() not in client_map:
            raise ValueError(f"Unknown client type: {client_type}. "
                             f"Valid types: {list(client_map.keys())}")

        ClientClass = client_map[client_type.lower()]
        print(f"Creating {client_type.upper()} client")
        return ClientClass(*args, **kwargs)

    def _is_filesystem_path(self, path_str):
        """Check if string represents a valid filesystem path."""
        path = Path(path_str)
        return path.exists() and path.is_dir()

    def _is_earthworm_client(self, *args, **kwargs):
        """Check for earthworm client indicators."""
        # Earthworm clients typically use specific ports
        earthworm_ports = [16022, 16023, 16024]  # Common earthworm ports

        # Check for port in kwargs
        if 'port' in kwargs and kwargs['port'] in earthworm_ports:
            return True

        # Check for earthworm-specific parameters
        earthworm_params = ['timeout', 'heartbeat_host', 'heartbeat_port']
        if any(param in kwargs for param in earthworm_params):
            return True

        return False

    def _is_seedlink_client(self, *args, **kwargs):
        """Check for seedlink client indicators."""
        # SeedLink clients typically use port 18000
        seedlink_ports = [18000, 18001, 18002]

        # Check for port in kwargs
        if 'port' in kwargs and kwargs['port'] in seedlink_ports:
            return True

        # Check for seedlink-specific parameters
        seedlink_params = ['autoconnect', 'recover']
        if any(param in kwargs for param in seedlink_params):
            return True

        return False

    def __getattr__(self, name):
        """Delegate attribute access to the underlying client."""
        if self._client is None:
            raise AttributeError(f"Client not initialized")
        return getattr(self._client, name)

    def get_stations(self, *args, **kwargs):
        """Get stations and return VInventory object."""
        from .inventory import VInventory  # Import here to avoid circular imports
        inventory = self._client.get_stations(*args, **kwargs)
        return VInventory(inventory)

    def get_events(self, *args, **kwargs):
        """Get events and return VCatalog object."""
        from .catalog import VCatalog  # Import here to avoid circular imports
        catalog = self._client.get_events(*args, **kwargs)
        return VCatalog(catalog)

    def get_waveforms(self, *args, **kwargs):
        """
        Get waveforms and return VStream object (when implemented).

        Usage:
        ------
        - get_waveforms(network, station, location, channel, starttime, endtime, ...)
        - get_waveforms(id="NET.STA.LOC.CHA", starttime=..., endtime=...)
        - get_waveforms(["NET.STA.LOC.CHA", ...], starttime=..., endtime=...)
        - get_waveforms("NET.STA.LOC.CHA", starttime=..., endtime=...)
        - starttime and endtime can be strings or UTCDateTime
        - Bulk requests are only used if self.use_bulk is True (automatically False for Earthworm clients)
        """
        from vdapseisutils.core.datasource.waveID import waveID
        from obspy import UTCDateTime
        # Handle id/waveformID as positional or keyword
        id_ = kwargs.pop('id', None) or kwargs.pop('waveformID', None)
        if id_ is None and len(args) > 0 and isinstance(args[0], str) and ('.' in args[0]) and len(args) < 5:
            id_ = args[0]
            args = args[1:]
        # Always cast starttime/endtime to UTCDateTime (keyword)
        if 'starttime' in kwargs:
            kwargs['starttime'] = UTCDateTime(kwargs['starttime'])
        if 'endtime' in kwargs:
            kwargs['endtime'] = UTCDateTime(kwargs['endtime'])
        # If id_ is a list, do bulk request if allowed
        if isinstance(id_, list) and self.use_bulk:
            bulk_args = []
            for idstr in id_:
                net, sta, loc, cha = waveID(idstr).network, waveID(idstr).station, waveID(idstr).location, waveID(idstr).channel
                t1 = kwargs.get('starttime', None)
                t2 = kwargs.get('endtime', None)
                # Always cast to UTCDateTime
                t1 = UTCDateTime(t1) if t1 is not None else None
                t2 = UTCDateTime(t2) if t2 is not None else None
                bulk_args.append((net, sta, loc, cha, t1, t2))
            return self._client.get_waveforms_bulk(bulk_args, **kwargs)
        elif isinstance(id_, list):
            # Fallback: call get_waveforms for each id and merge streams
            streams = []
            for idstr in id_:
                net, sta, loc, cha = waveID(idstr).network, waveID(idstr).station, waveID(idstr).location, waveID(idstr).channel
                t1 = kwargs.get('starttime', None)
                t2 = kwargs.get('endtime', None)
                t1 = UTCDateTime(t1) if t1 is not None else None
                t2 = UTCDateTime(t2) if t2 is not None else None
                streams.append(self._client.get_waveforms(net, sta, loc, cha, t1, t2, **kwargs))
            from obspy import Stream
            merged = Stream()
            for st in streams:
                merged += st
            return merged
        elif isinstance(id_, str):
            net, sta, loc, cha = waveID(id_).network, waveID(id_).station, waveID(id_).location, waveID(id_).channel
            t1 = kwargs.get('starttime', None)
            t2 = kwargs.get('endtime', None)
            t1 = UTCDateTime(t1) if t1 is not None else None
            t2 = UTCDateTime(t2) if t2 is not None else None
            return self._client.get_waveforms(net, sta, loc, cha, t1, t2, **kwargs)
        else:
            # Fallback to original signature
            # Also handle starttime/endtime as strings in positional args
            new_args = list(args)
            # Always cast 5th and 6th positional args to UTCDateTime if present
            if len(new_args) >= 5:
                new_args[4] = UTCDateTime(new_args[4])
            if len(new_args) >= 6:
                new_args[5] = UTCDateTime(new_args[5])
            return self._client.get_waveforms(*new_args, **kwargs)

    def get_waveforms_bulk(self, *args, **kwargs):
        """Get waveforms in bulk and return VStream object (when implemented)."""
        # For now, return regular Stream - will return VStream when implemented
        return self._client.get_waveforms_bulk(*args, **kwargs)
        # TODO: Implement VStream and return VStream(stream) instead

    def get_waveforms_from_inventory(self, inventory, **kwargs):
        """
        Get waveforms using station info from an inventory.
        Uses the same method as VInventory.
        """
        from .inventory import VInventory  # Import here to avoid circular imports
        return VInventory.get_waveforms_from_inventory(self._client, inventory, **kwargs)

    def __repr__(self):
        """String representation of VClient."""
        if self._client:
            client_name = self._client.__class__.__name__
            return f"VClient({client_name}: {repr(self._client)})"
        return "VClient(uninitialized)"

    @property
    def client(self):
        """Access to the underlying client object."""
        return self._client

    def get_client_type(self):
        """Get the type of the underlying client."""
        if self._client:
            return self._client.__class__.__name__
        return None

# Example usage:
#
# # Auto-detection examples:
# client1 = VClient("/path/to/SDS_filesystem")      # -> SDS Client
# client2 = VClient("IRIS")                         # -> FDSN Client
# client3 = VClient("localhost", port=16022)        # -> Earthworm Client
# client4 = VClient("localhost", port=18000)        # -> SeedLink Client
#
# # Forced client type:
# client5 = VClient("some.server.com", client_type="fdsn")
# client6 = VClient("/data", client_type="sds")
#
# # Methods return V-objects:
# vinv = client2.get_stations(network="AV", station="*")     # Returns VInventory
# vcat = client2.get_events(starttime=t0, endtime=t1)       # Returns VCatalog
# st = client2.get_waveforms("AV", "REF", "*", "BHZ", t0, t1)  # Returns Stream (VStream when implemented)
#
# # Use V-object methods:
# vcat.plot_timeseries(freq="1D")
# st = vinv.get_waveforms(client2, starttime=t0, endtime=t1)
# st = client2.get_waveforms_from_inventory(vinv, starttime=t0, endtime=t1)