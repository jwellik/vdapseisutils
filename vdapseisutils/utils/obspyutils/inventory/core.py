"""
Core VInventory class for volcano seismology workflows.

This module contains the main VInventory class that extends ObsPy's Inventory
with volcano seismology specific functionality.
"""

from obspy import Inventory, UTCDateTime, Stream
from obspy.clients.fdsn import Client
import pandas as pd
import numpy as np
import os
from obspy.core.inventory import Network, Station, Channel
from obspy.core.util import AttribDict
from typing import Union, Dict, List, Optional, Any


class VInventory(Inventory):
    """Extended Inventory class for volcano seismology workflows.
    
    This class extends ObsPy's Inventory class with additional functionality
    for reading from and writing to CSV files, as well as other formats.
    
    Examples:
    ---------
    # Read from CSV file
    inv = VInventory.read_csv('stations.csv')
    
    # Read from CSV with custom column mappings
    custom_mappings = {'lat': 'latitude', 'lon': 'longitude', 'depth': 'elevation'}
    inv = VInventory.from_df(pd.read_csv('stations.csv'), column_mappings=custom_mappings)
    
    # Write to CSV file
    inv.to_csv('output.csv')
    
    # Read from SWARM latlon.config file
    inv = VInventory.read_swarmlatlon('latlon.config')
    
    # Write to SWARM latlon.config file
    inv.to_swarmlatlon('output_latlon.config')
    
    # Convert to DataFrame
    df = inv.to_df()
    
    # Overwrite location codes
    count = inv.overwrite_loc("", "00")  # Change empty to "00"
    
    # Convert existing inventory
    vinv = VInventory(existing_inventory)
    
    # Read CSV without headers (will use default column names)
    inv = VInventory.read_csv('stations_no_header.csv')
    
    # Get waveforms with flexible client configuration
    st = inv.get_waveforms(client, t1, t2)
    """

    def __init__(self, networks=None, source=None, inventory=None):
        """
        Initialize VInventory.

        Parameters:
        -----------
        networks : list or obspy.Inventory, optional
            List of Network objects (standard ObsPy way) OR
            An existing Inventory object to convert to VInventory
        source : str, optional
            Source information (standard ObsPy way)
        inventory : obspy.Inventory, optional
            Existing Inventory object to convert to VInventory (deprecated parameter name)
        """
        # Handle case where first argument is an Inventory object
        if hasattr(networks, 'networks') and hasattr(networks, 'source'):
            # networks is actually an Inventory object
            inventory_obj = networks
            super().__init__(networks=inventory_obj.networks, source=inventory_obj.source)
            self.sender = getattr(inventory_obj, 'sender', None)
            self.created = getattr(inventory_obj, 'created', None)
        elif inventory is not None:
            # Initialize from inventory parameter
            super().__init__(networks=inventory.networks, source=inventory.source)
            self.sender = getattr(inventory, 'sender', None)
            self.created = getattr(inventory, 'created', None)
        else:
            # Standard initialization
            super().__init__(networks=networks, source=source)

    ########################################################################################################################
    # I/O
    ########################################################################################################################

    def to_csv(self, filepath, **kwargs):
        """
        Write inventory data to CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to output CSV file
        **kwargs : dict
            Additional arguments passed to pandas.DataFrame.to_csv()
            
        Notes:
        ------
        Output CSV format:
        id,lat,lon,elev
        AV.STA1.00.BHZ,19.5,-155.3,1234
        
        Each channel becomes a row with network.station.location.channel as the id.
        """
        # Collect all channel data
        rows = []
        
        for network in self:
            for station in network:
                for channel in station:
                    # Create row data
                    row = {
                        'id': f"{network.code}.{station.code}.{channel.location_code or ''}.{channel.code}"
                    }
                    
                    # Add coordinates (prefer channel over station)
                    if hasattr(channel, 'latitude') and channel.latitude is not None:
                        row['lat'] = channel.latitude
                    elif hasattr(station, 'latitude') and station.latitude is not None:
                        row['lat'] = station.latitude
                    
                    if hasattr(channel, 'longitude') and channel.longitude is not None:
                        row['lon'] = channel.longitude
                    elif hasattr(station, 'longitude') and station.longitude is not None:
                        row['lon'] = station.longitude
                    
                    if hasattr(channel, 'elevation') and channel.elevation is not None:
                        row['elev'] = channel.elevation
                    elif hasattr(station, 'elevation') and station.elevation is not None:
                        row['elev'] = station.elevation
                    
                    # Add any other station attributes
                    for attr in dir(station):
                        if not attr.startswith('_') and attr not in ['code', 'channels', 'networks']:
                            try:
                                value = getattr(station, attr)
                                if value is not None and not callable(value):
                                    row[attr] = value
                            except:
                                pass
                    
                    rows.append(row)
        
        # Create DataFrame and write to CSV
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, **kwargs)
        
        return df

    def to_df(self):
        """
        Convert inventory to pandas DataFrame using ObsPy's get_contents().
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with inventory contents in a simple format
            
        Notes:
        ------
        Uses ObsPy's built-in get_contents() method for simplicity.
        Each row represents a channel with basic station information.
        """
        try:
            # Use ObsPy's get_contents() method
            contents = self.get_contents()
            
            # Convert to DataFrame
            df = pd.DataFrame(contents)
            
            return df
            
        except Exception as e:
            # Fallback to manual conversion if get_contents() fails
            print(f"Warning: ObsPy get_contents() failed, using manual conversion: {e}")
            
            rows = []
            for network in self:
                for station in network:
                    for channel in station:
                        row = {
                            'network': network.code,
                            'station': station.code,
                            'location': channel.location_code or '',
                            'channel': channel.code,
                            'latitude': getattr(station, 'latitude', None),
                            'longitude': getattr(station, 'longitude', None),
                            'elevation': getattr(station, 'elevation', None)
                        }
                        rows.append(row)
            
            return pd.DataFrame(rows)

    @classmethod
    def read_csv(cls, filepath, **kwargs):
        """
        Read station inventory from a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        **kwargs : dict
            Additional arguments passed to pandas.read_csv()
            
        Returns:
        --------
        VInventory
            Inventory object populated from CSV data
            
        Notes:
        ------
        Expected CSV format:
        id,lat,lon,elev
        AV.SPCB.00.BHZ,51.12345,-178.12345,1234
        
        The method automatically parses the 'id' column to extract network, station, 
        location, and channel codes. Other columns are mapped as follows:
        - lat/latitude -> station latitude (required)
        - lon/longitude -> station longitude (required)
        - elev/elevation/depth -> station elevation (optional, defaults to 0.0)
        - Any other columns are stored as station attributes
        
        Stations without latitude or longitude are skipped with warnings.
        If no headers are present, the method will use the first row as data and
        assign default column names: id, lat, lon, elev
        """
        # Check if CSV has headers by trying to read with and without headers
        try:
            # First try to read with headers
            df = pd.read_csv(filepath, **kwargs)
        except:
            # If that fails, try without headers and assign default column names
            default_columns = ['id', 'lat', 'lon', 'elev']
            df = pd.read_csv(filepath, header=None, names=default_columns, **kwargs)
        
        # Create inventory from dataframe
        return cls.from_df(df)

    @classmethod
    def from_df(cls, df, column_mappings=None):
        """
        Create inventory from a pandas DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing station information
        column_mappings : dict, optional
            Dictionary mapping DataFrame column names to inventory field names.
            Default mappings:
            - lat/latitude -> station latitude
            - lon/longitude -> station longitude
            - elev/elevation/depth -> station elevation
            
        Returns:
        --------
        VInventory
            Inventory object populated from DataFrame data
            
        Notes:
        ------
        The DataFrame must have an 'id' column with format: NET.STA.LOC.CHAN
        Other columns are mapped to station attributes based on column_mappings.
        """
        # Default column mappings defined in one place
        DEFAULT_COLUMN_MAPPINGS = {
            'lat': 'latitude',
            'latitude': 'latitude',
            'lon': 'longitude',
            'longitude': 'longitude',
            'elev': 'elevation',
            'elevation': 'elevation',
            'depth': 'elevation',
            'alt': 'elevation',
            'altitude': 'elevation'
        }
        
        if column_mappings is None:
            column_mappings = DEFAULT_COLUMN_MAPPINGS
        
        # Validate required columns
        if 'id' not in df.columns:
            raise ValueError("DataFrame must contain 'id' column")
        
        # Create inventory
        inventory = cls()
        
        # Group by unique station locations (network.station)
        station_groups = {}
        
        for _, row in df.iterrows():
            # Parse station ID
            try:
                parts = row['id'].split('.')
                if len(parts) != 4:
                    raise ValueError(f"Invalid station ID format: {row['id']}. Expected NET.STA.LOC.CHAN")
                
                network_code, station_code, location_code, channel_code = parts
                
                # Create station key
                station_key = f"{network_code}.{station_code}"
                
                if station_key not in station_groups:
                    station_groups[station_key] = {
                        'network_code': network_code,
                        'station_code': station_code,
                        'location_code': location_code,
                        'channels': [],
                        'attributes': {}
                    }
                
                # Add channel
                station_groups[station_key]['channels'].append({
                    'code': channel_code,
                    'location_code': location_code
                })
                
                # Store other attributes
                for col, value in row.items():
                    if col != 'id' and pd.notna(value):
                        if col in column_mappings:
                            mapped_name = column_mappings[col]
                            station_groups[station_key]['attributes'][mapped_name] = value
                        else:
                            # Store unmapped columns as custom attributes
                            station_groups[station_key]['attributes'][col] = value
                            
            except Exception as e:
                print(f"Warning: Skipping row with invalid data: {row}. Error: {e}")
                continue
        
        # Create networks and stations
        networks_dict = {}
        
        for station_key, station_data in station_groups.items():
            network_code = station_data['network_code']
            station_code = station_data['station_code']
            
            # Create or get network
            if network_code not in networks_dict:
                network = Network(code=network_code)
                networks_dict[network_code] = network
                inventory.networks.append(network)
            
            # Check if required coordinates are available
            attrs = station_data['attributes']
            if 'latitude' not in attrs or 'longitude' not in attrs:
                print(f"Warning: Skipping station {station_code} - missing latitude or longitude")
                continue
            
            # Get coordinates (elevation is optional, defaults to 0)
            lat = float(attrs['latitude'])
            lon = float(attrs['longitude'])
            elev = float(attrs.get('elevation', 0.0))
            
            station = Station(code=station_code, latitude=lat, longitude=lon, elevation=elev)
            
            # Add custom attributes
            for key, value in attrs.items():
                if key not in ['latitude', 'longitude', 'elevation']:
                    setattr(station, key, value)
            
            # Add channels
            for channel_data in station_data['channels']:
                channel = Channel(
                    code=channel_data['code'],
                    location_code=channel_data['location_code'],
                    latitude=station.latitude,
                    longitude=station.longitude,
                    elevation=station.elevation,
                    depth=0.0
                )
                
                station.channels.append(channel)
            
            # Add station to network
            networks_dict[network_code].stations.append(station)
        
        return inventory

    @classmethod
    def read_swarmlatlon(cls, filepath):
        """
        Read station inventory from a SWARM latlon.config file.
        
        Parameters:
        -----------
        filepath : str
            Path to the SWARM latlon.config file
            
        Returns:
        --------
        VInventory
            Inventory object populated from SWARM latlon.config data
            
        Notes:
        ------
        SWARM latlon.config format:
        STATION CHANNEL NETWORK LOCATION = Longitude: X; Latitude: Y; Height: Z
        
        Example:
        SPBG BHE AV -- = Longitude: -152.3722; Latitude: 61.2591; Height: 1087.0
        
        The location code can be omitted or specified explicitly.
        When omitted, it will be set to empty string. When specified as '--', it will
        be preserved as '--'. Height is treated as elevation (optional, defaults to 0.0).
        Latitude and longitude are required; stations without them are skipped.
        """
        inventory = cls()
        networks_dict = {}
        
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        # Parse the line: STATION CHANNEL NETWORK LOCATION = Longitude: X; Latitude: Y; Height: Z
                        if '=' not in line:
                            print(f"Warning: Line {line_num} missing '=' separator: {line}")
                            continue
                        
                        # Split on '=' to separate trace_id from coordinates
                        trace_part, coord_part = line.split('=', 1)
                        trace_parts = trace_part.strip().split()
                        
                        if len(trace_parts) < 3:
                            print(f"Warning: Line {line_num} has insufficient trace parts: {line}")
                            continue
                        
                        # Parse trace_id components
                        station_code = trace_parts[0]
                        channel_code = trace_parts[1]
                        network_code = trace_parts[2]
                        
                        # Handle location code (might be omitted or '--')
                        if len(trace_parts) > 3:
                            location_code = trace_parts[3]
                        else:
                            location_code = ""
                        
                        # Parse coordinates
                        coord_parts = coord_part.strip().split(';')
                        longitude = None
                        latitude = None
                        height = None
                        
                        for coord in coord_parts:
                            coord = coord.strip()
                            if coord.startswith('Longitude:'):
                                longitude = float(coord.split(':', 1)[1].strip())
                            elif coord.startswith('Latitude:'):
                                latitude = float(coord.split(':', 1)[1].strip())
                            elif coord.startswith('Height:'):
                                height = float(coord.split(':', 1)[1].strip())
                        
                        if longitude is None or latitude is None:
                            print(f"Warning: Line {line_num} missing required coordinates (latitude/longitude): {line}")
                            continue
                        
                        # Create or get network
                        if network_code not in networks_dict:
                            network = Network(code=network_code)
                            networks_dict[network_code] = network
                            inventory.networks.append(network)
                        
                        # Check if station already exists in this network
                        existing_station = None
                        for station in networks_dict[network_code].stations:
                            if station.code == station_code:
                                existing_station = station
                                break
                        
                        if existing_station is None:
                            # Create new station (height is optional, defaults to 0)
                            elevation = height if height is not None else 0.0
                            station = Station(code=station_code, latitude=latitude, longitude=longitude, elevation=elevation)
                            networks_dict[network_code].stations.append(station)
                        else:
                            # Use existing station (coordinates should be the same)
                            station = existing_station
                        
                        # Create channel (height is optional, defaults to 0)
                        elevation = height if height is not None else 0.0
                        channel = Channel(
                            code=channel_code,
                            location_code=location_code,
                            latitude=latitude,
                            longitude=longitude,
                            elevation=elevation,
                            depth=0.0
                        )
                        
                        # Add channel to station
                        station.channels.append(channel)
                        
                    except Exception as e:
                        print(f"Warning: Error parsing line {line_num}: {line}. Error: {e}")
                        continue
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"SWARM latlon.config file not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error reading SWARM latlon.config file: {e}")
        
        return inventory

    def to_swarmlatlon(self, filepath):
        """
        Write inventory data to SWARM latlon.config format.
        
        Parameters:
        -----------
        filepath : str
            Path to output SWARM latlon.config file
            
        Notes:
        ------
        Output format:
        STATION CHANNEL NETWORK LOCATION = Longitude: X; Latitude: Y; Height: Z
        
        Each channel becomes a line. Location codes are preserved, empty location codes
        are written as '--'.
        """
        try:
            with open(filepath, 'w') as f:
                for network in self:
                    for station in network:
                        for channel in station:
                            # Get coordinates (prefer channel over station)
                            if hasattr(channel, 'latitude') and channel.latitude is not None:
                                lat = channel.latitude
                            elif hasattr(station, 'latitude') and station.latitude is not None:
                                lat = station.latitude
                            else:
                                lat = 0.0
                            
                            if hasattr(channel, 'longitude') and channel.longitude is not None:
                                lon = channel.longitude
                            elif hasattr(station, 'longitude') and station.longitude is not None:
                                lon = station.longitude
                            else:
                                lon = 0.0
                            
                            if hasattr(channel, 'elevation') and channel.elevation is not None:
                                elev = channel.elevation
                            elif hasattr(station, 'elevation') and station.elevation is not None:
                                elev = station.elevation
                            else:
                                elev = 0.0
                            
                            # Handle location code
                            location_code = channel.location_code if channel.location_code else "--"
                            
                            # Write line in SWARM format
                            line = f"{station.code} {channel.code} {network.code} {location_code} = Longitude: {lon}; Latitude: {lat}; Height: {elev}\n"
                            f.write(line)
                            
        except Exception as e:
            raise Exception(f"Error writing SWARM latlon.config file: {e}")
        
        return True

    def overwrite_loc(self, original_loc, new_loc):
        """
        Overwrite any location code that matches original_loc with new_loc.
        
        Parameters:
        -----------
        original_loc : str
            Original location code to replace
        new_loc : str
            New location code to use as replacement
            
        Notes:
        ------
        This method modifies the inventory in-place.
        All channels with location_code == original_loc will have their
        location_code changed to new_loc.
        """
        count = 0
        
        for network in self:
            for station in network:
                for channel in station:
                    if channel.location_code == original_loc:
                        channel.location_code = new_loc
                        count += 1
        
        return count

    ########################################################################################################################
    # Utils
    ########################################################################################################################

    # TODO BULK_REQUEST Take advantage of bulk_request if the client allows it
    # TODO Compare with VCatalog.get_waveforms(); streamline both with VClient.get_waveforms()

    def get_waveforms(
        self, 
        client: Union[Client, Dict[str, Client], List[Client]], 
        starttime: Union[UTCDateTime, str], 
        endtime: Union[UTCDateTime, str], 
        fallback_clients: Optional[Union[List[Client], Dict[str, Client], Client]] = None, 
        client_timeout: float = 30, 
        verbose: bool = True, 
        **kwargs: Any
    ) -> Stream:
        """
        Get waveforms using station info from this inventory with flexible client support.

        This method supports multiple client configurations like VCatalog.get_waveforms():
        - Single Client object: used for all requests
        - Dict mapping network codes to clients: {"IU": client1, "US": client2}
        - Dict mapping station codes to clients: {"ANMO": client1, "CCM": client2}
        - Dict mapping net.sta to clients: {"IU.ANMO": client1, "US.CCM": client2}
        - List of clients: tries each client in order until successful

        Parameters:
        -----------
        client : obspy.clients.fdsn.Client, dict, or list
            Primary client(s) for waveform data access. Can be:
            - Single Client object: used for all requests
            - Dict mapping network codes to clients: {"IU": client1, "US": client2}
            - Dict mapping station codes to clients: {"ANMO": client1, "CCM": client2}
            - Dict mapping net.sta to clients: {"IU.ANMO": client1, "US.CCM": client2}
            - List of clients: tries each client in order until successful
        starttime : obspy.UTCDateTime or str
            Start time for waveform data request
        endtime : obspy.UTCDateTime or str
            End time for waveform data request
        fallback_clients : list or dict, optional
            Additional clients to try if primary client fails. Can be:
            - List of clients: tries each in order
            - Dict with same mapping options as primary client
        client_timeout : float, default 30
            Timeout in seconds for individual client requests
        verbose : bool, default True
            Enable detailed logging output
        **kwargs : dict
            Additional parameters for get_waveforms
            Note: network, station, location, channel will be ignored if present

        Returns:
        --------
        obspy.Stream
            Stream object containing waveforms

        Raises:
        ------
        ValueError
            If client configuration is invalid
        TypeError
            If client is not a supported type
        """
        # Convert starttime and endtime to UTCDateTime objects
        starttime = UTCDateTime(starttime)
        endtime = UTCDateTime(endtime)
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

        # Extract station info from inventory
        bulk_list = []
        for network in self:
            for station in network:
                if len(station.channels) == 0:
                    print(f"Warning: Station {station.code} has no channels")
                for channel in station:
                    bulk_list.append((
                        network.code,
                        station.code,
                        channel.location_code if channel.location_code else "",
                        channel.code,
                        starttime,
                        endtime
                    ))

        if not bulk_list:
            print("Warning: No stations found in inventory to create bulk request")

        # Remove parameters that are now in bulk_list
        bulk_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ['network', 'station', 'location', 'channel']}

        # Initialize stream
        combined_stream = Stream()
        
        # Process each bulk request
        for net, sta, loc, cha, starttime, endtime in bulk_list:
            nsl_key = f"{net}.{sta}.{loc}.{cha}"
            
            if verbose:
                print(f"\nDownloading: {nsl_key}")
            
            # Get clients to try for this request
            clients_to_try = _get_client_for_request(net, sta, nsl_key)
            
            if not clients_to_try:
                if verbose:
                    print(f"  No clients available for {nsl_key}")
                continue
            
            # Try each client until successful
            success = False
            for i, client_obj in enumerate(clients_to_try):
                if not _validate_client(client_obj):
                    if verbose:
                        print(f"  Client {i+1} invalid, skipping")
                    continue
                
                try:
                    if verbose:
                        client_name = getattr(client_obj, 'base_url', str(client_obj))
                        print(f"  Trying client {i+1}: {client_name}")
                    
                    # Make the request
                    st = client_obj.get_waveforms(
                        network=net,
                        station=sta,
                        location=loc,
                        channel=cha,
                        starttime=starttime,
                        endtime=endtime,
                        **bulk_kwargs
                    )
                    
                    if st:
                        combined_stream += st
                        success = True
                        if verbose:
                            print(f"  Success: Retrieved {len(st)} traces")
                        break
                    else:
                        if verbose:
                            print(f"  Client returned empty stream")
                        
                except Exception as e:
                    if verbose:
                        print(f"  Failed: {e}")
                    continue
            
            if not success and verbose:
                print(f"  All clients failed for {nsl_key}")

        if verbose:
            print(f"\n--- Summary ---")
            print(f"Total traces retrieved: {len(combined_stream)}")
            if combined_stream:
                networks_found = set(tr.stats.network for tr in combined_stream)
                print(f"Networks: {', '.join(sorted(networks_found))}")

        return combined_stream

    @classmethod
    def get_waveforms_from_inventory(
        cls, 
        client: Union[Client, Dict[str, Client], List[Client]], 
        inventory: Inventory, 
        **kwargs: Any
    ) -> Stream:
        """
        Get waveforms using station info from an inventory (classmethod version).

        Parameters:
        -----------
        client : obspy.clients.fdsn.Client, dict, or list
            Client(s) for waveform data access (same flexible options as instance method)
        inventory : obspy.Inventory
            Inventory object containing station metadata
        **kwargs : dict
            Parameters for get_waveforms (starttime, endtime, etc.)
            Note: network, station, location, channel will be ignored if present

        Returns:
        --------
        obspy.Stream
            Stream object containing waveforms
        """
        # Convert to VInventory and use instance method
        vinv = cls(inventory)
        return vinv.get_waveforms(client, **kwargs)

    @classmethod
    def from_inventory(cls, inventory):
        """Create VInventory from a regular Inventory object."""
        vinv = cls()
        vinv.networks = inventory.networks
        vinv.source = inventory.source
        vinv.sender = getattr(inventory, 'sender', None)
        vinv.created = getattr(inventory, 'created', None)
        return vinv

    ########################################################################################################################
    # Plotting
    ########################################################################################################################

    def scatter(self, ax=None, **kwargs):
        """
        Create a scatter plot of longitude (x) vs latitude (y) for inventory stations.
        Each station is plotted as an upside down black triangle.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        **kwargs :
            Additional keyword arguments passed to plt.scatter.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the scatter plot.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        lons = []
        lats = []
        station_names = []
        
        # Iterate through networks, stations, and channels to get unique station locations
        for network in self:
            for station in network:
                # Get station coordinates
                if hasattr(station, 'latitude') and hasattr(station, 'longitude'):
                    lat = station.latitude if station.latitude is not None else np.nan
                    lon = station.longitude if station.longitude is not None else np.nan
                    
                    # Only add if we have valid coordinates and haven't seen this station yet
                    station_id = f"{network.code}.{station.code}"
                    if not np.isnan(lat) and not np.isnan(lon) and station_id not in station_names:
                        lons.append(lon)
                        lats.append(lat)
                        station_names.append(station_id)
        
        lons = np.array(lons)
        lats = np.array(lats)
        
        if ax is None:
            fig, ax = plt.subplots()
        
        # Plot stations as upside down black triangles
        sc = ax.scatter(lons, lats, marker='v', c='black', s=100, alpha=kwargs.pop('alpha', 0.8), **kwargs)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Station Locations')
        
        return ax    


# Example usage:
#
# # Method 1: Standard ObsPy way
# vinv = VInventory(networks=[network1, network2], source="My inventory")
#
# # Method 2: From existing inventory (simple form)
# regular_inv = client.get_stations(...)
# vinv = VInventory(regular_inv)  # Clean conversion!
#
# # Method 3: From existing inventory (explicit parameter)
# vinv = VInventory(inventory=regular_inv)
#
# # Get waveforms with flexible client configuration
# st = vinv.get_waveforms(client, t1, t2)
# st = vinv.get_waveforms(client_dict, t1, t2)  # dict mapping
# st = vinv.get_waveforms([client1, client2], t1, t2)  # list of clients
