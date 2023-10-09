# UTILS for Station Metadata based on ObsPy's Iventory class
import pandas as pd
from vdapseisutils.utils.geoutils import dd2dm

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
        st[i].stats.network = network
        st[i].stats.station = station
        st[i].stats.location = location
        st[i].stats.channel = channel

    return st


def getNSLCstr(tr, order='nslc', sep='.'):
    network = tr.stats.network
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
        if verbose: print(cha_str)

    if filename:
        with open(filename, 'w') as f:
            for line in channel_strings:
                f.write(line)
                f.write('\n')

    if verbose:
        [print(line) for line in channel_strings]


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

########################################################################################################################
# NonLinLoc
########################################################################################################################

def write_nll_EQSTA(inventory, verbose=True):
    # EQSTA Print lines for EQSTA commands in NonLinLoc's control file
    # By default, it creates a line for P & S for each station with 0.0 as the calculated and reported error.
    #
    # NonLinLoc control file documentation:
    # source description (multiple sources can be specified)
    # (EQSRCE (see GTSRCE)
    #
    # EQSRCE  VentiSynth  LATLON   43.805321 7.562109 9.722238  0.0

    # station description (multiple stations can be specified)
    # (EQSTA  label phase  error_type error)
    #    (char[])   label
    #    (char[])   phase
    #    (char[])   calc_error_type
    #    (float)   calc_error
    #    (char[])   report_error_type
    #    (float)   report__error

    # TODO Allow a list of calc_error, report_error, etc. that matches the inventory

    inventorydf = inventory2df(inventory)

    lines = ""
    template_line = "EQSTA  {sta:<6}  {phase_type:1}      {calc_error_type:>3}  {calc_error:> 3.1f}  {report_error_type:>3}  {report_error:> 3.1f}\n"

    phase_type = "P"
    calc_error_type = "GAU"
    calc_error = 0.0
    report_error_type = "GAU"
    report_error = 0.0

    for idx, row in inventorydf.iterrows():
        net, sta, loc, cha = row["nslc"].split(".")
        for phase in ["P", "S"]:
            lines += template_line.format(sta=sta, phase_type=phase,
                                              calc_error_type=calc_error_type, calc_error=calc_error,
                                              report_error_type=report_error_type, report_error=report_error)

    lines = "\n".join(set(lines.split("\n")))

    if verbose: print(lines)

    return lines


def write_nll_GTSRCE(inventory, loc_type="LATLON", verbose=True):
    # GTSRCE Creates GTSRCE lines for NonLinLoc's control file
    #
    # Here is the documentation from NonLinLoc's sample control file:
    # (GTSRCE  label  x_srce  y_srce   z_srce   elev)
    #
    #    (char[])   label
    #
    #    (char[])   loc type (XYZ, LATLON (+/-dec deg), LATLONDM (deg, dec min))
    #  XYZ---------------      LATLON/LATLONDM--------
    #  x_srce : km pos E   or  lat   : pos N
    #  y_srce : km pos N   or  long  : pos E
    #  z_srce : km pos DN  or  depth : pos DN
    #
    #    elev : km pos UP
    #
    # Examples:
    #
    # GTSRCE  STA   XYZ  	27.25  -67.78  0.0  1.242
    # GTSRCE  CALF  LATLON  	43.753  6.922  0.0  1.242
    # GTSRCE  JOU  LATLONDM  43 38.00 N  05 39.52 E   0.0   0.300
    #

    """

    Programmer's note on units
                ObsPy           NonLinLoc
    depth           ?                  km
    elevation       m                  km
    """

    inventorydf = inventory2df(inventory)

    lines = ""
    template_line = "GTSRCE  {sta:<6}  {loc_type:<8}  {lat:>12}  {lon:>12}  {depth:>2.3f}  {elevation:>2.3f}\n"

    for idx, row in inventorydf.iterrows():
        net, sta, loc, cha = row["nslc"].split(".")

        if loc_type == "LATLON":
            # ObsPy inventory is already in LATLON
            lat = row["latitude"]
            lon = row["longitude"]
        elif loc_type == "LATLONDM":
            lat_dm = dd2dm(row["latitude"], hemisphere="latitude")  # returns (degrees, decimal minutes)
            lon_dm = dd2dm(row["longitude"], hemisphere="longitude")
            lat = "{deg:>2.0f} {decmin:>8.5f} {hemi}".format(deg=lat_dm[0], decmin=lat_dm[1], hemi=lat_dm[2])
            lon = "{deg:>3.0f} {decmin:>8.5f} {hemi}".format(deg=lon_dm[0], decmin=lon_dm[1], hemi=lon_dm[2])
        elif loc_type == "XYZ":
            raise ValueError("XYZ not yet supported :-(")
        else:
            print("loc_type not undertsood!")

        depth = row["local_depth"]/1000  # convert depth from m (ObsPy) to km (NLL) ???
        elevation = row["elevation"]/1000  # convert elevation from m (ObsPy) to km (NLL)
        lines += template_line.format(sta=sta, loc_type=loc_type, lat=lat, lon=lon, depth=depth, elevation=elevation)

    lines = "\n".join(set(lines.split("\n")))

    if verbose: print(lines)

    return lines

########################################################################################################################
# Earthworm
########################################################################################################################
