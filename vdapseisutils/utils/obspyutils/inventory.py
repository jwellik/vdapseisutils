from obspy import Inventory
from obspy.clients.fdsn import Client


class VInventory(Inventory):
    """Extended Inventory class for volcano seismology workflows."""

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

    @staticmethod
    def swarmlatlon2inv():
        pass

    def read_swarmlatlon(self):
        pass

    @staticmethod
    def inv2swarmlatlon():
        pass

    def write_swarmlatlon(self):
        pass

    @staticmethod
    def binderew2inv():
        pass

    def read_binderew(self):
        pass

    @staticmethod
    def inv2binderew():
        pass

    def write_binderew(self):
        pass

    def get_waveforms(self, client, **kwargs):
        """
        Get waveforms using station info from this inventory.

        Parameters:
        -----------
        client : obspy.clients.fdsn.Client
            FDSN client instance
        **kwargs : dict
            Parameters for get_waveforms (starttime, endtime, etc.)
            Note: network, station, location, channel will be ignored if present

        Returns:
        --------
        obspy.Stream
            Stream object containing waveforms
        """

        # Extract station info from inventory
        bulk_list = []
        for network in self:
            for station in network:
                for channel in station:
                    bulk_list.append((
                        network.code,
                        station.code,
                        channel.location_code if channel.location_code else "",
                        channel.code,
                        kwargs.get('starttime'),
                        kwargs.get('endtime')
                    ))

        # Remove parameters that are now in bulk_list
        bulk_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ['starttime', 'endtime', 'network', 'station', 'location', 'channel']}

        return client.get_waveforms_bulk(bulk_list, **bulk_kwargs)

    @classmethod
    def get_waveforms_from_inventory(cls, client, inventory, **kwargs):
        """
        Get waveforms using station info from an inventory (classmethod version).

        Parameters:
        -----------
        client : obspy.clients.fdsn.Client
            FDSN client instance
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

        # Extract station info from inventory
        bulk_list = []
        for network in inventory:
            for station in network:
                for channel in station:
                    bulk_list.append((
                        network.code,
                        station.code,
                        channel.location_code if channel.location_code else "",
                        channel.code,
                        kwargs.get('starttime'),
                        kwargs.get('endtime')
                    ))

        # Remove parameters that are now in bulk_list
        bulk_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ['starttime', 'endtime', 'network', 'station', 'location', 'channel']}

        return client.get_waveforms_bulk(bulk_list, **bulk_kwargs)

    @classmethod
    def from_inventory(cls, inventory):
        """Create VInventory from a regular Inventory object."""
        vinv = cls()
        vinv.networks = inventory.networks
        vinv.source = inventory.source
        vinv.sender = getattr(inventory, 'sender', None)
        vinv.created = getattr(inventory, 'created', None)
        return vinv

# Example usage:
# client = Client("IRIS")
#
# # Method 1: Standard ObsPy way
# vinv = VInventory(networks=[network1, network2], source="IRIS")
#
# # Method 2: From existing inventory (simple form)
# regular_inv = client.get_stations(network="AV", station="*", location="*", channel="BH?",
#                                  starttime=t0, endtime=t1, latitude=volc["lat"],
#                                  longitude=volc["lon"], maxradius=km2dd(rad_km))
# vinv = VInventory(regular_inv)  # This now works!
# st = vinv.get_waveforms(client, starttime=t0, endtime=t1)
#
# # Method 3: From existing inventory (explicit parameter)
# vinv = VInventory(inventory=regular_inv)
#
# # Method 4: Use classmethod directly on any inventory
# st = VInventory.get_waveforms_from_inventory(client, regular_inv, starttime=t0, endtime=t1)