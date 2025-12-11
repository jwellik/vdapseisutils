from obspy import Stream, Trace, Catalog, Inventory
from vdapseisutils.core.datasource.waveID import waveID


import os
import pandas as pd
import numpy as np


class StreamV(Stream):

    from vdapseisutils.utils.obspyutils import streamutils

    def __init__(self):
        print("VDAPSEISUTILS Stream")

    def print(self):
        print("Booyah!")

    def same_data_type(self):
        return streamutils.same_data_type(self)

    def preprocess(self, *args, **kwargs):
        return streamutils.preprocess(self)

    def remove_winston_gaps(self, *args, **kwargs):
        return streamutils.removeWinstonGaps(self, *args, **kwargs)

    def clip(self, clip_threshold):
        return streamutils.clip(self, clip_threshold)

    def sort_by_nslc(self, nslc_list, verbose=False):
        return streamutils.sortStreamByNSLClist(self, nslc_list, verbose=verbose)

    def add_empty_trace(self, *args, **kwargs):
        return streamutils.createEmptyTrace(*args, **kwargs)

    def idselect(self, ids):
        # TODO Deprecate. Use obspy.stream.select(id=...) instead
        return streamutils.idselect(self, ids)

    def ffrsam(self, window_length=60, step=None, freq=None):
        """Computes frequency filtered RSAM using RMS

        * Does NOT affect the original Stream

        :param window_length (int) RSAM window length in minutes
        :param step (int) RSAM window interval in minutes (default: same as 'window_length')
        :param freqmin Lower bound of filterband (default 1000 seconds)
        :param freqmax Upper bound of filterband (default 1000 Hertz)
        """

        step = step if step else window_length  # default step to same as 'window_length'
        if step > window_length:
            raise ValueError("Error: step must be less than or equal to window_length.")

        # Assert that freq is None or a tuple of numbers, cast numbers as integers

        # Prepare Stream
        st = self.copy()  # hard copy of original Stream for operations
        if freq:
            st.filter("bandpass", freqmin=freq[0], freqmax=freq[1])
        st = st.merge(fill_value=0, method=1)  # merge same NSLCs

        # Filter and compute RMS
        for tr in st:
            # initialize ffrsam vectors
            rms = []
            tvec = []

            for ws in tr.slide(window_length*60, step*60):  # convert minutes to seconds
                # rms.append(np.sqrt(np.nanmean(np.square(ws[0].data))))
                rms.append(np.sqrt(np.mean(np.square(ws.data))))
                tvec.append(ws.stats.starttime)

            # Store data back to Trace
            tr.data = np.array(rms)
            tr.stats.starttime = tvec[0]  # force starttime (tvec[0] is a UTCDateTime); endtime is read only
            tr.stats.npts = len(tvec)
            tr.stats.delta = tvec[1] - tvec[0]  # sets delta and sampling_rate (subtract 2 UTCDateTimes returns seconds)

        return st

    def write_sds(self, directory="./"):
        print("This method will write the Stream into the SDS format to the specified directory")


class TraceV(Trace):

    def __init__(self):
        print("VDAPSDISUTILS Trace")


class CatalogV(Catalog):

    def __init__(self):
        print("VDAPSEISUTILS Catalog")
        from obspy import UTCDateTime
        from obspy.core.event.event import Event
        from obspy.core.event.origin import Origin
        from obspy.core.event.magnitude import Magnitude
        super().__init__()

    def read_csv(self):
        print("Not yet implemented.")

    def read_ewhtml_csv(self, file, verbose=False):
        """Reads ewhtml csv output as ObsPy Catalog

        EWHTMLREPORTCSV header:
        QuakeID,OriginTime(UTC),Latitude,Longitude,DepthBSL(km),Md,Ml,Ml_err,RMS,ErrorH(km),ErrorZ(km),Gap,TotalPhases,UsedPhases,SPhasesUsed,Quality

        See Pandas doc for csv reading tips:
            https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

        """

        # print('Read CSV produced by ewhtmlreport into ObsPy Catalog object')

        df = pd.read_csv(file, parse_dates=["OriginTime(UTC)"], infer_datetime_format=True, header=0,
                         # names=['time', 'lat', 'lon', 'depth', 'mag'],
                         )
        df['time'] = df['OriginTime(UTC)']
        df = df.drop(columns=['OriginTime(UTC)'])
        df['depth'] = df['DepthBSL(km)'] * 1000
        df = df.drop(columns=['DepthBSL(km)'])
        df['mag'] = df['Md']  # Just Duration magnitude for now
        df = df.drop(columns=['Md'])
        df['lat'] = df["Latitude"]
        df.drop(columns=["Latitude"])
        df['lon'] = df["Longitude"]
        df.drop(columns=["Longitude"])
        if verbose:
            print(df)
            print()

        self.description = "Catalog imported from CSV file via Pandas DataFrame"
        for index, row in df.iterrows():
            e = Event()
            e.event_type = "not existing"

            o = Origin()
            o.time = UTCDateTime(row['time'])
            o.latitude = row['lat']
            o.longitude = row['lon']
            o.depth = row['depth']
            o.depth_type = "operator assigned"
            o.evaluation_mode = "manual"
            o.evaluation_status = "preliminary"
            # o.region = FlinnEngdahl().get_region(o.longitude, o.latitude)

            m = Magnitude()
            m.mag = row['mag']
            m.magnitude_type = "Md"

            # also included could be: custom picks, amplitude measurements, station magnitudes,
            # focal mechanisms, moment tensors, ...

            # make associations, put everything together
            self.append(e)
            e.origins = [o]
            e.magnitudes = [m]
            m.origin_id = o.resource_id

        if verbose:
            print(self)

    def read_ewhtml_table(self, file, sep="\t", verbose=False):
            """
            ID	Date Time	Lat.	Lon.	Depth	MD	RMS	ErrH	ErrZ	Gap	Pha.	U.	S	Qual.	Event Page
            10153	2019.02.22 21:55:30	-37.8025	-71.0733	2.7	0	0.01	5.26	0.29	292	4	4	0	D	10153
            10152	2019.02.22 21:27:38	-37.7947	-71.0685	2.65	0	0.01	5.62	0.35	304	4	4	0	D	10152
            """

            df = pd.read_csv(file, parse_dates=["Date Time"], infer_datetime_format=True, header=0,
                             # names=['time', 'lat', 'lon', 'depth', 'mag'],
                             )
            df['time'] = df['Date Time']
            df = df.drop(columns=['Date Time'])
            df['depth'] = df['Depth'] * 1000  # convert km to meters
            df = df.drop(columns=['Depth'])
            df['mag'] = df['MD']  # Just Duration magnitude for now
            df = df.drop(columns=['MD'])
            df['lat'] = df["Lat."]
            df.drop(columns=["Lat."])
            df['lon'] = df["Lon."]
            df.drop(columns=["Lon."])
            if verbose:
                print(df)
                print()

            self.description = "Catalog imported from CSV file via Pandas DataFrame"
            for index, row in df.iterrows():
                e = Event()
                e.event_type = "not existing"

                o = Origin()
                o.time = UTCDateTime(row['time'])
                o.latitude = row['lat']
                o.longitude = row['lon']
                o.depth = row['depth']
                o.depth_type = "operator assigned"
                o.evaluation_mode = "manual"
                o.evaluation_status = "preliminary"
                # o.region = FlinnEngdahl().get_region(o.longitude, o.latitude)

                m = Magnitude()
                m.mag = row['mag']
                m.magnitude_type = "Md"

                # also included could be: custom picks, amplitude measurements, station magnitudes,
                # focal mechanisms, moment tensors, ...

                # make associations, put everything together
                self.append(e)
                e.origins = [o]
                e.magnitudes = [m]
                m.origin_id = o.resource_id

            if verbose:
                print(self)

    def read_ew_arcfiles(self, path):
        from vdapseisutils.utils.obspyutils.catalogutils import hypoinverse

        searchdir = path
        arcfiles = os.listdir(searchdir)
        for arcfile in arcfiles:
            arcfile = os.path.join(searchdir, arcfile)
            print("> {} ...".format(arcfile), end=" ")

            try:
                arc = hypoinverse.hypoCatalog()
                arc.readArcFile(arcfile)
                self.append(arc.writeObspyCatalog()[0])
                print("SUCCESS", end="\n")
            except:
                print("FAILED.", end="\n")

    def read_hyp2000_log(self, logfile):
        """READ_HYP2000_LOG Reads log file from hyp2000_mgr, returns ObsPy Catalog object
        """
        # I HATE THIS FORMAT LOL

        import re
        import datetime as dt
        from obspy import UTCDateTime
        from vdapseisutils.utils.geoutils import dms2dd

        df = pd.DataFrame(columns=["time", "latitude", "longitude", "depth", "mag"])

        # Open the file for reading
        with open(logfile, 'r') as file:
            for line in file:
                if re.match(r'\d{8}_UTC_\d{2}:\d{2}:\d{2}', line):  # Check if the line matches the expected pattern
                    parts = line.strip().split()  # Split the line by spaces

                    if len(parts) == 17:  # A line of location data should have 16 parts

                        # time
                        date = parts[1]  # Extract the date part
                        hhmm = parts[2]  # Extract the time part
                        seconds = parts[3]
                        time = UTCDateTime(dt.datetime.strptime(date+hhmm+seconds, "%Y%m%d%H%M%S.%f"))

                        # latitude
                        d = parts[4]  # degrees and decimal minutes are stored here
                        try:
                            d, m = d.split("S")  # e.g., 37S47.96 -- split degrees, minutes by S or N
                            d = float(d) * -1  # mulitply by -1 if South
                            m = float(m)
                        except:
                            d, m = d.split("N")
                            d = float(d)
                            m = float(m)
                        latitude = dms2dd((d,m,0))  # Convert degrees, minutes to decimal degrees

                        # longitude
                        d = float(parts[5][0:-1])
                        d = d * -1 if parts[5][-1] == "W" else d
                        m = float(parts[6])
                        longitude = dms2dd((d, m, 0))

                        depth = float(parts[7])*1000  # convert km to m for ObsPy Catalog
                        mag = float(parts[8])

                        nphases = int(parts[9])

                        column10 = parts[10]
                        column11 = parts[11]
                        column12 = parts[12]
                        column13 = parts[13]
                        column14 = parts[14]
                        column15 = parts[15]
                        column15 = parts[15]

                        # append data to a DataFrame here
                        df = pd.concat([df, pd.DataFrame({"time": time, "latitude": latitude, "longitude": longitude,
                                                          "depth": depth, "mag": mag}, index=[0])],
                                       ignore_index=True)  # thanks chatGPT

        self.read_basics(df)


class InventoryV(Inventory):

    def __init__(self):
        print("VDAPSEISUTILS Inventory")

    def overwrite_loc_code(self, new_loc_code):
        # Overwrites location code for every channel in an inventory.
        # Change is made in place on inv (copy of inventory)

        inv = self.copy()
        for net in inv.networks:
            for sta in net.stations:
                for cha in sta.channels:
                    cha.location_code = new_loc_code

        return inv

    def from_dict(self, data, verbose=False):
        """Converts dictionary or DataFrame to an Inventory object."""

        from obspy.core.inventory.network import Network
        from obspy.core.inventory.station import Station
        from obspy.core.inventory.channel import Channel

        if isinstance(data, dict):
            df = pd.DataFrame(data)

        # assuming your DataFrame is named df
        df[['network', 'station', 'location', 'channel']] = df['nslc'].str.split('.', expand=True)  # convert 'nslc' to 'network' 'station' 'location' 'channel' columns
        for n, group in df.groupby('network'):
            print(f"Network: {n}") if verbose else None
            net = Network(code=n, stations=[])
            for s, station_group in group.groupby('station'):
                print(f"  Station: {s}") if verbose else None
                elev = station_group.iloc[0]["elevation"] if not np.isnan(station_group.iloc[0]["elevation"]) else 0
                sta = Station(code=s, latitude=station_group.iloc[0]["latitude"],
                              longitude=station_group.iloc[0]["longitude"],
                              elevation=elev)
                for i, channel_group in station_group.iterrows():
                    print("    Channel: {}".format(channel_group["channel"])) if verbose else None
                    elev = channel_group["elevation"] if not np.isnan(channel_group["elevation"]) else 0
                    cha = Channel(code=channel_group["channel"], location_code=channel_group["location"],
                                  latitude=channel_group["latitude"], longitude=channel_group["longitude"],
                                  elevation=elev, depth=channel_group["local_depth"])
                    sta.channels.append(cha)
                net.stations.append(sta)
            self.networks.append(net)

    def from_csv(self, inventory_csv, *kwargs):
        df = pd.DataFrame.from_csv(inventory_csv, **kwargs)
        return self.from_dict(df)

    def read_swarm(self, latlonconfig, local_depth_default=0, verbose=False):
        """READ_SWARM Reads Swarm-formatted LatLon.config file of stations

        input: Swarm/LatLon.config file
        output: Pandas DataFrame
        """

        if verbose:
            print("Reading Swarm LatLon.config: " + os.path.abspath(latlonconfig))

        # Using readlines()
        file1 = open(latlonconfig, 'r')
        Lines = file1.readlines()

        data = []

        count = 0
        # Strips the newline character
        for line in Lines:

            line = line.expandtabs()  # convert hidden tabs to spaces
            line = line.replace("\n", "")  # remove end of line characters
            line = line.strip()  # remove leading and trailing spaces from line

            d = dict({})
            count += 1

            name_deets = line.split("=")
            scnl = name_deets[0]
            deets = name_deets[1].replace("\n", "")
            params = deets.split(";")
            for p in params:
                key_val = p.split(":")
                key = key_val[0].replace(" ", "")
                value = key_val[1].replace(" ", "").replace("\n", "")
                if key == "Latitude":
                    d["latitude"] = float(value)
                elif key == "Longitude":
                    d["longitude"] = float(value)
                elif key == "Height":
                    d["elevation"] = float(value)
                else:
                    "Swarm parameter ({},{}) not processed.".format(key, value)
            # d["nslc"] = convertNSLCstr(scnl, order="scnl", sep=" ", neworder="nslc", newsep=".")
            d["nslc"] = waveID(scnl, order="scnl", sep=" ").nslc()  # Convert "S C N L" to "N.S.L.C"
            d["local_depth"] = local_depth_default  # Add local_depth default

            if verbose:
                print(">>> ", d)
            data.append(d)

        return self.from_dict(d)

    def write_swarm(self, outfile="./LatLon.config", verbose=False):
        """WRITE_SWARM Writes inventory as CSV formatted channel,latitude,longitude,elevation"""

        # from vdapseisutils.waveformutils.nslcutils import convertNSLCstr

        # first brack is SCNL in format 'STA CHA NET LOC'
        swarm_format = '{}= Longitude: {}; Latitude: {}; Height: {}'

        channel_strings = []

        for cha in self.get_contents()['channels']:
            coords = self.get_coordinates(cha)
            # cha = convertNSLCstr(cha, order='nslc', sep='.', neworder='scnl', newsep=' ')
            cha = waveID(cha).scnl(sep=" ")
            cha_str = swarm_format.format(cha, coords['longitude'], coords['latitude'], coords['elevation'])
            channel_strings.append(cha_str)

        if outfile:
            with open(outfile, 'w') as f:
                for line in channel_strings:
                    f.write(line)
                    f.write('\n')

        if verbose:
            print("# LatLon.config for Swarm")
            [print(line) for line in channel_strings]
            print()

    def write_ew_pick_ew(self, outfile="./pick_ew.sta", verbose=False, **kwargs):
        from vdapseisutils.utils.ewutils import pickew_StaFile
        pickew_StaFile(self, outfile=outfile, verbose=verbose, **kwargs)

    def write_ew_pick_FP(self, outfile="./pick_FP.sta", verbose=False, **kwargs):
        from vdapseisutils.utils.ewutils import pickfp_StaFile
        pickfp_StaFile(self, outfile=outfile, verbose=verbose, **kwargs)

    def write_ew_hinv_site_file(self, outfile="sta.hinv", verbose=False, **kwargs):
        from vdapseisutils.utils.ewutils import hinv_site_file
        hinv_site_file(self, outfile=outfile, verbose=verbose, **kwargs)

    def write_carl_sta_file(self, outfile="carl.sta", verbose=False, **kwargs):
        from vdapseisutils.utils.ewutils import carl_StationFile
        carl_StationFile(self, outfile=outfile, verbose=verbose, **kwargs)

    def plot(self, *args, **kwargs):
        print("Not yet implemented.")
