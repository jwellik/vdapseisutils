import os
import datetime
import numpy as np

from vdapseisutils.utils.obspyutils.inventoryutils import inventory2df
from vdapseisutils.utils.geoutils import dd2dms


########################################################################################################################
### UTILS

# Used by both ewutils and nllutils
def print_and_write(data_lines, header="", source="", name="", verbose=True, outfile=None):
    """Completes header and concatenates it to the data lines"""

    # Populate header and concatenate to data
    header = header.format(source=source, name=name, datetime=datetime.datetime.now())  # This is safe to do if header, name, etc aren't included
    template = "{header}{data_lines}".format(header=header, data_lines=data_lines)

    # print and write
    if verbose:
        print(template)
        print()
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)  # ensure that the file and abspath exist
        with open(outfile, "w") as f:
            f.write(template)


########################################################################################################################
### VELOCITY MODELS
# TODO Replace default values with Jeremy's default stratovolcano model

def lay(depth, Vp_top, verbose=True, outfile=None):
    """Prints Earthworm layer commands"""

    header = ["# Earthworm 'lay' commands for ew_binder.d"]

    # Store the layer commands in a list
    layer_commands = []
    for d, v in zip(depth, Vp_top):
        layer_commands.append("lay {:>6.4f}  {:>6.4f}".format(d, v))

    # # Print the layer commands to the console or a file
    # if verbose:
    #     for command in header + layer_commands:
    #         print(command)
    # if outfile:
    #     with open(outfile, 'w') as f:
    #         for command in header + layer_commands:
    #             f.write(command + '\n')
    print_and_write(layer_commands, verbose=verbose, outfile=outfile)

def velocityd(depth=[0.0, 4.0, 9.0, 16.0, 20.0, 25.0, 41.0], velocity=[5.40, 6.38, 6.59, 6.73, 6.86, 6.95, 7.80],
              verbose=True, outfile=None):

    template = "{layer_lines}"  # There is no header to this file

    """
    lay   0.0  5.40
    lay   4.0  6.38
    lay   9.0  6.59
    lay  16.0  6.73
    lay  20.0  6.86
    lay  25.0  6.95
    lay  41.0  7.80 
    """

    layer_line = "lay  {:>4.1f} {:>4.2f}\n"
    layer_lines = ""
    for d, v in zip(depth, velocity):
        layer_line.format(d, v)
        layer_lines += layer_line.format(d, v)

    # Fill template
    # name = filename if name is None else name
    template = template.format(layer_lines=layer_lines)

    # Print and Save
    print_and_write(template, header="", verbose=verbose, outfile=outfile)
    return template

def velocitycrh(depth=[0.0, 4.0, 9.0, 16.0, 20.0, 25.0, 41.0], velocity=[5.40, 6.38, 6.59, 6.73, 6.86, 6.95, 7.80],
              verbose=True, outfile=None,):

    template = "{layer_lines}"  # There is no header to this file

    """
    R0 Rabaul Generic
     1.7  0.0
     2.2  0.7
     3.5  1.0
     4.0  2.0
     4.5  3.0
     6.2  5.0
     6.4 15.0
    """

    layer_line = " {:>4.2f} {:>4.1f}\n"  # Format for one line of velocity data
    layer_lines = ""  # initialize list of lines as empty string

    for d, v in zip(velocity, depth):
        layer_line.format(d, v)
        layer_lines += layer_line.format(d, v)

    # Fill template
    template = template.format(layer_lines=layer_lines)

    print_and_write(template, header="", verbose=verbose, outfile=outfile)
    return template

########################################################################################################################
### STATION INVENTORIES

# CARLSTATRIG carl_StationFile (trig.sta)
def carl_StationFile(inventory, source="", name="", L=None, verbose=True, outfile=None):
    """

    :param invdf:
    :param filename:
    :param source:
    :param name:
    :param L: Optional overwrite for location code
    :return:
    """

    invdf = inventory2df(inventory)

    template = "{station_lines}"

    # Create station lines
    station_line = "  station   {sta:<7} {cha:<7} {net:<9} {loc:<8}   {ttime:<12}\n"  # One line of station data
    station_lines = ""  # initialize list of stations as empty

    # Populate each line of station data
    for idx, row in invdf.iterrows():
        net, sta, loc, cha = row["nslc"].split(".")
        loc = L if L is not None else loc
        station_lines += station_line.format(sta=sta, cha=cha, net=net, loc=loc, ttime=10)

    print_and_write(station_lines, header="# Earthworm configuration lines for carl modules (trig.sta)\n", verbose=verbose, outfile=outfile)
    return station_lines

# HYPOINVERSE hinv_site_file (sta.hinv)
def hinv_site_file(inventory, L=None, source="", name="", verbose=True, outfile=None):
    # Create station lines
    # Hypoinverse station file docs: http://folkworm.ceri.memphis.edu/ew-doc/USER_GUIDE/hypoinv_sta.html
    #           1         2         3         4         5         6         7         8
    #  12345 78901234567890123456789012345678901234567890123456789012345678901234567890123
    #  AAAAA BB CDDD EFF GGGGGGGHIII JJJJJJJKLLLLMMM  NOPPPPP QQQQQ RRRRRSTTTTTUVWWWWWWXX
    # "ASBU  CC  BHE  43 49.2336N121 22.1041W15380.0  A  0.00  0.00  0.00  0.00 1      --"
    # "AAAAA BB CDDD EFF GGGGGGGHIII JJJJJJJKLLLLMMM  NOPPPPP QQQQQ RRRRRSTTTTTUVWWWWWWXX"
    #
    # F 16-17	I2, 1X	Latitude, degrees.
    # G 19-25	F7.4	Latitude, minutes.
    # H 26	A1	N or blank for north latitude, S for south.
    # I 27-29	I3, 1X	Longitude, degrees.
    # J 31-37	F7.4	Longitude, minutes.
    # K 38	A1	W or blank for west longitude, E for east.
    # L 39-42	4X	Reserved for elevation in m. Not used by HYPOINVERSE.

    invdf = inventory2df(inventory)

    station_line = "{Asta:<5} {Bnet:<2} {Copt:1}{Dcha:<3} {Eweight:1}{Flatdeg:>2} {Glatmin:>7}{Hns:1}{Ilondeg:>3} {Jlonmin:>7}{Kew:1}{Lelev:>4}0.0  A  0.00  0.00  0.00  0.00 1      --\n"
    station_lines = ""
    for idx, row in invdf.iterrows():
        net, sta, loc, cha = row["nslc"].split(".")
        loc = L if L is not None else loc
        # latdeg = np.abs(math.floor(row["latitude"]))
        # latmin = "{:>2.4f}".format((row["latitude"] % 1)*60)  # % 1 gets the decimal of a flat | *60 converts decimal degrees to minutes
        dms = dd2dms(row["latitude"])
        latdeg = int(np.abs(dms[0]))
        latmin = "{:>2.4f}".format(dms[1] + dms[2] / 60)
        ns = "N" if row["latitude"] >= 0 else "S"

        # londeg = np.abs(math.floor(row["longitude"]))
        # lonmin = "{:>2.4f}".format((row["longitude"] % 1) * 60)
        dms = dd2dms(row["longitude"])
        londeg = int(np.abs(dms[0]))
        lonmin = "{:>2.4f}".format(dms[1] + dms[2] / 60)
        ew = "E" if row["longitude"] >= 0 else "W"

        elev = int(row["elevation"])

        station_lines += station_line.format(Asta=sta, Bnet=net, Copt=" ", Dcha=cha, Eweight=" ",
                                             Flatdeg=latdeg, Glatmin=latmin, Hns=ns,
                                             Ilondeg=londeg, Jlonmin=lonmin, Kew=ew,
                                             Lelev=elev)

    print_and_write(station_lines, header="# HypoInverse Station File (sta.hinv)\n", verbose=verbose, outfile=outfile)

    return station_lines

# PICKEW pickew_StaFile pick_ew.sta
pick_ew_header = """#
#   AUTO GENERATED by VDAPSEISUTILS
#   - Name    : {name}
#   - Source  : {source}
#   - Created : {datetime}
#
#                     Station file for PICK_EW
#
#   http://folkworm.ceri.memphis.edu/ew-doc/cmd/pick_ew_cmd.html#station
#
#                 MinBigZC    RawDataFilt    LtaFilt         DeadSta          PreEvent
#  Station/  MinSmallZC   MaxMint        StaFilt       RmavFilt           AltCoda
#  Comp/Net  Itr1   MinPeakSize   CharFuncFilt  EventThresh          CodaTerm         Erefs
#  --------------------------------------------------------------------------------------
"""

def pickew_StaFile(inventory, L=None, source="", name="", verbose=True, outfile=None):
    # http://folkworm.ceri.memphis.edu/ew-doc/cmd/pick_ew_cmd.html#station

    """
    EXAMPLE LINES
    rabaul_params/params_generic/pick_ew_RVO.sta
        1   1   SPT   EHZ RV -- 3  40  3  20  500  0  .985  3.  .4  .015 5.  .9961  1200.  882.47  .8  1.5  50000. 2
        1   2   SDA   BHZ RV -- 3  40  3  20  500  0 0.882  3.  .4  .015 5.  .9961  1200.  429.73  .8  1.5  50000. 2048
        1   5   RAL   BHZ RV -- 3  40  3  20  500  0 0.882  3.  .4  .015 5.  .9961  1200.  429.73  .8  1.5  50000. 2048
        1   8   JBH   EHZ RV -- 3  40  3  20  500  0  .985  3.  .4  .015 5.  .9961  1200.  882.47  .8  1.5  50000. 2048
        1   9   KOR   EHZ RV -- 3  40  3  20  500  0  .985  3.  .4  .015 5.  .9961  1200.  882.47  .8  1.5  50000. 2
        1   10  TUN   EHZ RV -- 3  40  3  20  500  0  .985  3.  .4  .015 5.  .9961  1200.  882.47  .8  1.5  50000. 2048
        1   11  BMA   EHZ RV -- 3  40  3  20  500  0  .985  3.  .4  .015 5.  .9961  1200.  882.47  .8  1.5  50000. 2048
        1   12  VIS   BHZ RV -- 3  40  3  20  500  0 0.882  3.  .4  .015 5.  .9961  1200.  429.73  .8  1.5  50000. 2048
        1   15  RPT   EHZ RV -- 3  40  3  20  500  0  .985  3.  .4  .015 5.  .9961  1200.  882.47  .8  1.5  50000. 2048

    pnsn_params/params_generic/pick/pick_ew.sta
         1   501  GASB  BHZ BK 00 3  40  3  20  500  0 0.854  3.  .4  .015 5.  .9961  1200.  409.59  .8  1.5  50000. 23689428
    """

    # Create station lines (SCNL)
    channel_identification = "    {pick_flag:1}  {pin_numb:< 4}  {sta:<5} {cha:3} {net:2} {loc:2} "
    event_termination = " 3  40  3  20  500  0 "  # defaults
    waveform_filtering = "0.854  3.  .4  .015 5.  .9961  1200.  409.59  .8  1.5  50000. 2048"  # defaults
    pin_numb = 0  # initialize pin_numb as 0, will grow iteratively w each line

    station_line = channel_identification + event_termination + waveform_filtering + "\n"  # line for 1 station
    station_lines = ""  # initialize list of lines as empty

    invdf = inventory2df(inventory)
    for _, row in invdf.iterrows():
        pin_numb += 1
        net, sta, loc, cha = row["nslc"].split(".")
        loc = L if L is not None else loc
        station_lines += station_line.format(pick_flag=1, pin_numb=pin_numb,
                                             sta=sta, net=net, cha=cha, loc=loc)

    print_and_write(station_lines, header=pick_ew_header, source=source, name=name, verbose=verbose, outfile=outfile)
    return station_lines

# PICKFP pick_fp.sta
pick_fp_header = """#
#   AUTO GENERATED by VDAPSEISUTILS
#   - Name    : {name}
#   - Source  : {source}
#   - Created : {datetime}
#
#                     Station file for PICK_FP
#
#   http://folkworm.ceri.memphis.edu/ew-doc/cmd/pick_FP_cmd.html#station
#
# Pick  Pin     Sta/Comp           longTermWindow  tUpEvent
# Flag  Numb    Net/Loc       filterWindow  threshold2
# ----  ----    --------      -----------------------------
"""

def pickfp_StaFile(inventory, L=None, source="", name="", verbose=True, outfile=None):

    invdf = inventory2df(inventory)

    # Create station lines (SCNL)
    channel_identification = "    {pick_flag:1}  {pin_numb:< 4}  {sta:<5} {cha:3} {net:2} {loc:2}   "
    # event_termination = " 3  40  3  20  500  0 "
    # waveform_filtering = "0.854  3.  .4  .015 5.  .9961  1200.  409.59  .8  1.5  50000. 2048"
    tuning_parameters = "-1  -1   8.6  17.2   -1"

    station_line = channel_identification + tuning_parameters + "\n"
    station_lines = ""

    pin_numb = 0
    for idx, row in invdf.iterrows():
        pin_numb += 1
        net, sta, loc, cha = row["nslc"].split(".")
        loc = L if L is not None else loc
        station_lines += station_line.format(pick_flag=1, pin_numb=pin_numb,
                                             sta=sta, net=net, cha=cha, loc=loc)

    print_and_write(station_lines, header=pick_fp_header, source=source, name=name, verbose=verbose, outfile=outfile)
    return station_lines


