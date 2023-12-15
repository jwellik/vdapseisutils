from vdapseisutils.utils.ewutils.ewutils import print_and_write
from vdapseisutils.utils.obspyutils.inventoryutils import inventory2df
from vdapseisutils.utils.geoutils import dd2dm


def EQSTA(inventory, verbose=True, outfile=None):
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

    print_and_write(lines, header="# NonLinLoc EQSTA command lines\n", verbose=verbose, outfile=outfile)
    return lines

def GTSRCE(inventory, loc_type="LATLON", verbose=True, outfile=None):
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

    print_and_write(lines, header="# NonLinLoc GTSRCE commands\n", verbose=verbose, outfile=outfile)
    return lines

def LOCSRCE(inventory, L=None, verbose=True, outfile=None):
    """NLL_STA_LIST Creates lines for NonLinLoc sta_list.in

    :param invdf:
    :return:
    """

    """
    LOCSRCE   AA-IS  LATLON  52.2117 -174.2036         0     0.006
    LOCSRCE   AA1    LATLON        0         0         0         0
    LOCSRCE   AAA    LATLON  43.2717   76.9467         0       0.8
    LOCSRCE   AAB    LATLON        0         0         0         0
    LOCSRCE   AAC    LATLON  50.7833    6.0833         0     0.179
    LOCSRCE   AADN   LATLON    22.11     31.55         0       0.2
    LOCSRCE   AAE    LATLON   9.0292   38.7656         0     2.442
    LOCSRCE   AAGR   LATLON -33.0852  -68.8284         0     1.159
    LOCSRCE   AAHD   LATLON  23.7463   32.7528         0         0
    LOCSRCE   AAI    LATLON   -3.687  128.1945         0      0.08
    LOCSRCE   AAK    LATLON   42.639    74.494         0     1.645
    LOCSRCE   AALM   LATLON    36.83   -2.4017         0      0.01
    LOCSRCE   AAM    LATLON  42.3012  -83.6567         0     0.172
    LOCSRCE   AAMC   LATLON   42.278   -83.736         0      0.25
    LOCSRCE   AAP    LATLON    10.42   121.944         0      0.05
    LOCSRCE   AAPN   LATLON  37.3077    -4.121         0      1.16
    LOCSRCE   AAR    LATLON  46.1333    25.895         0     1.101
    LOCSRCE   AARM   LATLON  39.2762 -121.0255         0      0.93
    LOCSRCE   AAS    LATLON   -62.16  -58.4625         0     0.015
    LOCSRCE   AASM   LATLON    38.43 -121.1085         0     0.065
    """

    invdf = inventory2df(inventory)

    station_line = "LOCSRCE   {sta:<4}   LATLON {lat:>8} {lon:>9}         0     1.000\n"
    station_lines = ""
    for idx, row in invdf.iterrows():
        net, sta, loc, cha = row["nslc"].split(".")
        loc = L if L is not None else loc
        lat = "{:6.4f}".format(row["latitude"])
        lon = "{:7.4f}".format(row["longitude"])
        station_lines += station_line.format(sta=sta, lat=lat, lon=lon)

    print_and_write(station_lines, header="# NonLinLoc LOCSRCE commands\n", verbose=verbose, outfile=outfile)
    return station_lines
