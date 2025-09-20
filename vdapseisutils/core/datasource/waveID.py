from obspy.core.event.base import WaveformStreamID


def parse_wave_id(id_string, order="nslc", sep=".", blank=""):
    """
    Parses an ID string into network, station, location, and channel components.

    :param id_string: The wave ID string to parse.
    :param order: The order of components in the string (default: "nslc").
    :param sep: The separator character (default: ".").
    :param blank: Character to use for missing S C N L item. (default: "")
    :return: A tuple of four strings: (network, station, location, channel).
    :raises ValueError: If parsing fails or the number of components is incorrect.
    """
    components = id_string.split(sep)
    if len(components) != len(order):
        raise ValueError("Invalid ID string: must contain same number of components specified in order.")

    order_map = {"n": 0, "s": 1, "l": 2, "c": 3}  # Order mapping for NSLC
    parsed = [components[order_map[o]] if o in order_map else blank for o in order]
    return tuple(parsed)


class waveID(WaveformStreamID):

    def __init__(self, *args, order="nslc", sep=".", blank="", **kwargs):
        """
        Initializes the waveID object.

        If four ordinal arguments are provided, they are passed directly to the inherited class's __init__.
        If one ordinal argument is provided, it is parsed using parse_wave_id to extract the four components.

        :param args: Positional arguments for initialization.
        :param order: The order of the components in a single input string (default: "nslc").
        :param sep: The separator character in a single input string (default: ".").
        :param blank: Characters to use for missing S C N L item (default: "")
        :param kwargs: Additional keyword arguments for the inherited class's __init__.
        :raises ValueError: If parsing fails for a single input string.
        """

        # Read data as either "NET", "STA", "LOC", "CHA" or as "NET.STA.LOC.CHA"
        # All other input types should be parsed by specific functions
        if len(args) == 4:
            super().__init__(*args, **kwargs)
        elif len(args) == 1:
            if not isinstance(args[0], str):
                raise TypeError("A single argument must be a string if provided.")

            try:
                network, station, location, channel = parse_wave_id(args[0], order=order, sep=sep, blank=blank)
                super().__init__(network_code=network, station_code=station, location_code=location, channel_code=channel, **kwargs)
            except ValueError as e:
                raise ValueError(f"Failed to parse wave ID: {e}")
        else:
            raise TypeError("Invalid number of arguments. Provide either one string or four strings.")

    def from_dict(self, d):
        """
        Read waveID from a dictionary. Understands keys:
            - "network_code", "network", "net", "n"
            - "station_code", "station", "sta", "s"
            - "location_code", "location", "loc", "l"
            - "channel_code", "channel", "cha", "c"

        :param d: dictionary of network, station, location, and channel key:value pairs
        :return: waveID object
        """
        print("FROM_DICT() NOT YET IMPLEMENTED :-(")

    def set(self, property_name, value):
        """
        Dynamically updates the value of an attribute.

        :param property_name: The name of the property to set.
        :param value: The value to set for the property.
        :raises AttributeError: If the property does not exist.
        """
        if hasattr(self, property_name):
            setattr(self, property_name, value)
        else:
            raise AttributeError(f"'{property_name}' is not a valid property of {self.__class__.__name__}")

    def string(self, order="nslc", sep="."):
        """
        Returns a string representation of the components in the given order and separator.

        :param order: The order of the components in the output string (default: "nslc").
        :param sep: The separator character for the output string (default: ".").
        :return: A string representation of the components.
        """
        order_map = {"n": self.network_code, "s": self.station_code, "l": self.location_code, "c": self.channel_code}
        # components = [order_map[o] if o in order_map else blank for o in order]
        components = [order_map[o] for o in order]
        return sep.join(components)

    def parts(self):
        return self.network, self.station, self.location, self.channel

    def nslc(self, sep="."):
        return self.string(order="nslc", sep=sep)

    def scnl(self, sep="."):
        return self.string(order="scnl", sep=sep)

    def scn(self, sep="."):
        return self.string(order="scn", sep=sep)

    def ns(self, sep="."):
        return self.string(order="ns", sep=sep)

    @property
    def network(self):
        return self.get("network_code")

    @property
    def station(self):
        return self.get("station_code")

    @property
    def location(self):
        return self.get("location_code")

    @property
    def channel(self):
        return self.get("channel_code")

    @staticmethod
    def scnl2nslc(scnl, sep=".", new_sep="."):
        scnl_id = waveID(scnl, order="scnl", sep=sep)
        return scnl_id.nslc(sep=new_sep)


def test():

    # Usage example
    wave = waveID("VG.GTOH.00.BHZ", order="nslc", sep=".")
    wave.set("channel_code", "EHZ")
    print(wave.id)
    print(wave.station)
    print(wave.nslc(" "))
    print(wave.ns("_"))


    wave2 = waveID("AV", "SSBA", "--", "BHZ")
    print(wave2.network_code, wave2.station_code, wave2.location_code, wave2.channel_code)


    # Test with different separator
    wave4 = waveID("CC YOCR 00 BHZ", order="nslc", sep=" ")
    print(wave4.scnl(" "))

    # Additional test cases
    # Test with an invalid string
    try:
        wave3 = waveID("TDH.HHZ.UW", order="scnl", sep=".")
    except ValueError as e:
        print(f"Caught expected error: {e}")


    # Test with missing components in order
    # wave5 = waveID("TDH.HHZ.UW", order="scn", sep=".", blank="--")
    # print(wave5.network_code, wave5.station_code, wave5.channel_code, wave5.location_code)