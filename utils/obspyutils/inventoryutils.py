# UTILS for Station Metadata based on ObsPy's Iventory class
import pandas as pd


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
                d["nslc"] = [net.code+"."+sta.code+".."+cha.location_code+cha.code]
                d["latitude"] = [sta.latitude]
                d["longitude"] = [sta.longitude]
                d["elevation"] = [sta.elevation]
                d["local_depth"] = [0.0]

                stationdf = pd.concat([stationdf, pd.DataFrame(d)])

    return stationdf


def write_simple_csv(inventory, filename='~/inventory.csv'):
    """Writes inventory as CSV formatted channel,latitude,longitude,elevation"""

    stationdf = inventory2df(inventory)
    stationdf = stationdf.drop(labels=["local_depth"])

    if filename:
        stationdf.to_csv(filename, index=False)

    return stationdf

########################################################################################################################
# NSLC
########################################################################################################################

def str2nslc(station_code, order='nslc', sep='.'):
    """Convert any NSLC/SCNL/SCN str to its components"""

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


########################################################################################################################
# Swarm
########################################################################################################################

def write_swarm(inventory, filename=None, verbose=True):
    """WRITE_SWARM Writes inventory as CSV formatted channel,latitude,longitude,elevation"""

    # from vdapseisutils.waveformutils.nslcutils import convertNSLCstr

    # first brack is SCNL in format 'STA CHA NET LOC'
    swarm_format = '{}= Longitude: {}; Latitude: {}; Height: {}'

    channel_strings = []

    for cha in inventory.get_contents()['channels']:
        coords = inventory.get_coordinates(cha)
        cha = convertNSLCstr(cha, order='nslc', sep='.', neworder='scnl', newsep=' ')
        cha_str = swarm_format.format(cha, coords['longitude'], coords['latitude'], coords['elevation'])
        channel_strings.append(cha_str)

    if filename:
        with open(filename, 'w') as f:
            for line in channel_strings:
                f.write(line)
                f.write('\n')

    if verbose:
        print("# LatLon.config for Swarm")
        [print(line) for line in channel_strings]
        print()

def read_swarm(latlonconfig, local_depth_default=0):
    """READ_SWARM Reads Swarm-formatted LatLon.config file of stations

    input: Swarm/LatLon.config file
    output: ObsPy Inventory class
    """

    print("Reading Swarm LatLon.config")

    # Using readlines()
    file1 = open(latlonconfig, 'r')
    Lines = file1.readlines()

    data = []

    count = 0
    # Strips the newline character
    for line in Lines:

        d = dict({})
        count += 1
        # print("Line{}: {}".format(count, line.strip()))

        name_deets = line.split("=")
        scnl = name_deets[0]
        deets = name_deets[1].replace("\n", "")
        # print(" scnl  : {}".format(scnl))
        # print(" deets : {}".format(deets))
        params = deets.split(";")
        for p in params:
            key_val = p.split(":")
            # print(key_val)
            key = key_val[0].replace(" ", "")
            value = key_val[1].replace(" ", "").replace("\n", "")
            # print(" ({}:{})".format(key, value))
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

        # print("::::")
        # print(d)
        # print("::::")
        data.append(d)
        # print()

    return pd.DataFrame.from_records(data)
