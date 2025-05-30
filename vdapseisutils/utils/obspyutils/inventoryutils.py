# UTILS for Station Metadata based on ObsPy's Iventory class
import os
import numpy as np
import pandas as pd
from obspy.core.inventory import Inventory, Network, Station, Channel


########################################################################################################################
# BASIC
########################################################################################################################

def inventory2df(inventory):
    """INVENTORY2DF Converts Inventory kml to Pandas DataFrame
    (includes index)
    """

    stationdf = pd.DataFrame({"nslc": [], "latitude": [], "longitude": [], "elevation": [], "local_depth":[]})

    for net in inventory.networks:
        for sta in net.stations:
            for cha in sta.channels:
                d = dict()
                d["nslc"] = "{}.{}.{}.{}".format(net.code, sta.code, cha.location_code, cha.code)
                d["latitude"] = [sta.latitude]
                d["longitude"] = [sta.longitude]
                d["elevation"] = [sta.elevation]
                d["local_depth"] = [0.0]

                stationdf = pd.concat([stationdf, pd.DataFrame(d)])

    return stationdf


def df2inventory(df):
    """DF2INVENTORY Converts DataFrame of station data to ObsPy Inventory object

    Requires a Pandas DataFrame with required (and optional) columns 'nslc', 'latitude', 'longitude', 'elevation', ('local_depth')
    Station always has the lat, lon, elev of its first channel. This probably isn't great.
    """

    inv = Inventory(networks=[], source="Swarm-LatLon.config")

    # assuming your DataFrame is named df
    df[['network', 'station', 'location', 'channel']] = df['nslc'].str.split('.', expand=True)  # convert 'nslc' to 'network' 'station' 'location' 'channel' columns
    for n, group in df.groupby('network'):
        print(f"Network: {n}")
        net = Network(code=n, stations=[])
        for s, station_group in group.groupby('station'):
            print(f"  Station: {s}")
            elev = station_group.iloc[0]["elevation"] if not np.isnan(station_group.iloc[0]["elevation"]) else 0
            sta = Station(code=s, latitude=station_group.iloc[0]["latitude"], longitude=station_group.iloc[0]["longitude"],
                          elevation=elev)
            for i, channel_group in station_group.iterrows():
                print("    Channel: {}".format(channel_group["channel"]))
                elev = channel_group["elevation"] if not np.isnan(channel_group["elevation"]) else 0
                cha = Channel(code=channel_group["channel"], location_code=channel_group["location"],
                              latitude=channel_group["latitude"], longitude=channel_group["longitude"],
                              elevation=elev, depth=channel_group["local_depth"])
                sta.channels.append(cha)
            net.stations.append(sta)
        inv.networks.append(net)

    return inv


def write_simple_csv(inventory, filename='~/inventory.csv'):
    """Writes inventory as CSV formatted channel,latitude,longitude,elevation"""

    stationdf = inventory2df(inventory)
    try:
        stationdf = stationdf.drop(labels=["local_depth"])
    except:
        pass

    if filename:
        stationdf.to_csv(filename, index=False)

    return stationdf

def write_nslc_list(inventory, filename='~/list.file'):
    """Writes inventory as CSV formatted channel,latitude,longitude,elevation"""

    stationdf = inventory2df(inventory)
    stationdf = stationdf[['nslc']]  # limit to only this column

    if filename:
        stationdf.to_csv(filename, index=False)

    return stationdf

########################################################################################################################
# NSLC
########################################################################################################################

def str2nslc(station_code, order='nslc', sep='.'):
    """Convert any NSLC/SCNL/SCN str to its components"""

    # station_code = station_code.strip()  # start by stripping leading and trailing spaces

    if order == 'nslc':
        n = 0
        s = 1
        l = 2
        c = 3

    elif order == 'scnl':
        s = 0
        c = 1
        n = 2
        l = 3

    elif order == 'scn':
        station_code + sep
        s = 0
        c = 1
        n = 2
        l = 3

    network = station_code.split(sep)[n]
    station = station_code.split(sep)[s]
    location = station_code.split(sep)[l]
    channel = station_code.split(sep)[c]

    return network, station, location, channel


def convertNSLCstr(station_code, order='nslc', neworder='scnl', sep='.', newsep='.'):
    """Convert any NSLC/SCNL/SCN str to another NSLC/SCNL/SCN str"""

    network, station, location, channel = str2nslc(station_code, order=order, sep=sep)
    return build_str(network, station, location, channel, order=neworder, sep=newsep)


def scnl2nslc(code_list, sep=".", newsep="."):
    """SCNL2NSLC Convenience wrapper for convertNSLCstr"""
    input_type = type(code_list)
    if input_type is not list:
        code_list = [code_list]
    out_list = [convertNSLCstr(code, order="scnl", neworder="nslc", sep=sep, newsep=newsep) for code in code_list]
    if len(out_list) == 1:
        out_list = out_list[0]
    return out_list


def nslc2scnl(code_list, sep=".", newsep="."):
    """NSLC2SCNL Convenience wrapper for convertNSLCstr"""
    input_type = type(code_list)
    if input_type is not list:
        code_list = [code_list]
    out_list = [convertNSLCstr(code, order="nslc", neworder="scnl", sep=sep, newsep=newsep) for code in code_list]
    if len(out_list) == 1:
        out_list = out_list[0]
    return out_list


def setNSLC(st, nslc_string, sep='.'):
    network, station, location, channel = str2nslc(nslc_string)

    for i in range(len(st)):
        st[i].stats.observatory = network
        st[i].stats.station = station
        st[i].stats.location = location
        st[i].stats.channel = channel

    return st


def getNSLCstr(tr, order='nslc', sep='.'):
    network = tr.stats.observatory
    station = tr.stats.station
    location = tr.stats.location
    channel = tr.stats.channel

    return build_str(network, station, location, channel, order=order, sep=sep)


def build_str(network, station, location, channel, order='nslc', sep='.'):
    # print('!!! This can be simplified with Trace.id for NSLC order')

    if order == 'nslc':
        syntax = '{0}{4}{1}{4}{2}{4}{3}'
    elif order == 'scnl':
        syntax = '{1}{4}{3}{4}{0}{4}{2}'
    elif order == 'scn':
        syntax = '{1}{4}{3}{4}{0}'
    else:
        print('Order {} not understood'.format(order))

    return syntax.format(network, station, location, channel, sep)


def str2bulk(nslc_list, t1, t2):
    """Formats a list of NSLC strings and t1, t2 into a proper request for ObsPy's get_waveforms_bulk"""
    from obspy import UTCDateTime

    t1 = UTCDateTime(t1)
    t2 = UTCDateTime(t2)

    bulk = []
    for nslc in nslc_list:
        n, s, l, c = str2nslc(nslc)
        bulk.append((n, s, l, c, t1, t2))

    return bulk

# TODO Allow to filter by existing loc code. E.g.,
#   overwrite_loc_code(inventory, old="", new="00")
def overwrite_loc_code(inventory, new_loc_code):
    # Overwrites location code for every channel in an inventory.
    # Change is made in place on inv (copy of inventory)

    print("This code will be deprecated. Please use vdapseisutils.utils.obspy.Inventory.overwrite_loc_code()")

    inv = inventory.copy()
    for net in inv.networks:
        for sta in net.stations:
            for cha in sta.channels:
                cha.location_code = new_loc_code
    return inv


########################################################################################################################
# Swarm
########################################################################################################################

def write_swarm(inventory, verbose=True, filename=None, outfile=None):
    """WRITE_SWARM Writes inventory as CSV formatted channel,latitude,longitude,elevation"""

    # from vdapseisutils.waveformutils.nslcutils import convertNSLCstr

    if outfile:
        print("'outfile' is being used for 'filename'. 'outfile' will be deprecated. Please use 'filename' instead.")
        filename = outfile

    # first brack is SCNL in format 'STA CHA NET LOC'
    swarm_format = '{}= Longitude: {}; Latitude: {}; Height: {}'

    channel_strings = []

    for cha in inventory.get_contents()['channels']:
        coords = inventory.get_coordinates(cha)
        cha = convertNSLCstr(cha, order='nslc', sep='.', neworder='scnl', newsep=' ')
        cha_str = swarm_format.format(cha, coords['longitude'], coords['latitude'], coords['elevation'])
        channel_strings.append(cha_str)

    if outfile:
        with open(filename, 'w') as f:
            for line in channel_strings:
                f.write(line)
                f.write('\n')

    if verbose:
        print("# LatLon.config for Swarm")
        [print(line) for line in channel_strings]
        print()

def read_swarm(latlonconfig, local_depth_default=0, format="DataFrame", verbose=False):
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
        d["nslc"] = convertNSLCstr(scnl, order="scnl", sep=" ", neworder="nslc", newsep=".")
        d["local_depth"] = local_depth_default  # Add local_depth default

        if verbose:
            print(">>> ", d)
        data.append(d)

    if format.lower() == "DataFrame".lower():
        output = pd.DataFrame.from_records(data)  # Convert to DataFrame
    elif format.lower() == "Inventory".lower():
        output = df2inventory(pd.DataFrame.from_records(data))  # Convert to DataFrame, then to Inventory
    else:
        raise Exception("format {} not understood. Options are 'DataFrame' or 'Inventory'".format(format))

    return output

########################################################################################################################
# Misc.
########################################################################################################################

def read_sage_txt(file, fill_channel="BHZ", fill_location="--", fill_depth=0):
    """READ_SAGE_TEXT Reads text output from http://ds.iris.edu/gmap"""

    from obspy import Inventory
    from obspy.core.inventory import Network, Station, Channel

    # Read the data into a DataFrame
    df = pd.read_csv(file, delimiter='|', comment='#', skip_blank_lines=True)

    # Rename columns if needed
    df.columns = ["Network", "Station", "Latitude", "Longitude", "Elevation", "Sitename", "StartTime", "EndTime"]

    networks = []
    for network, group in df.groupby("Network"):
        net = Network(network)

        stations = []
        for index, row in group.iterrows():
            # net = Network(row["Network"])
            sta = Station(row["Station"], row["Latitude"], row["Longitude"], row["Elevation"])
            cha = Channel(fill_channel, fill_location, row["Latitude"], row["Longitude"], row["Elevation"], fill_depth)
            sta.channels = [cha]
            stations.append(sta)

        net.stations = stations
        networks.append(net)

    inv = Inventory(networks)

    return inv
