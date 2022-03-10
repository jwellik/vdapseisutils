import os.path


def tankplayer_WaveFileLine(nslc_list, tankdir):

    return None

def tanklines(nslc_list, tankdir, inst_id="INST_UNKOWN", mod_id="MOD_ADSEND_A", reclen=4096):
    # tankplayer.d
    # WaveFile        /opt/earthworm/run/data/tankfiles/FRO_HHZ_VV_CP.tnk
    # WaveFile<234567>/full/path/STA_CHA_NT_LC.tnk
    # WaveFile        /opt/earthworm/run/data/tankfiles/PV6A_BHZ_AV_--.tnk

    # wave_serverV.d
    # Tank MLZ HHZ VV CP  4096  INST_VSEIS    MOD_ADSEND_A       1 40  /opt/earthworm/run/data/tankfiles/MLZ_HHZ_VV_CP.tnk
    #                           123456789012341234567890123456789
    # Tank PV6A BHZ AV --  4096  INST_VDAPSEIS MOD_ADSEND_A       1 40  /opt/earthworm/run/data/tankfiles/PV6A_BHZ_AV_--.tnk

    import os
    tankdir = os.path.abspath(tankdir)


    # Make and store lines
    WAVE_SERVERV = []
    TANKPLAYER = []
    for nslc in nslc_list:
        net = nslc.split(".")[0]
        sta = nslc.split(".")[1]
        loc = nslc.split(".")[2]
        cha = nslc.split(".")[3]

        # Create line for wave_serverV.d
        wave_serverV = "Tank {sta} {cha} {net} {loc}{reclen:>6}  {inst_id:<13} {mod_id:18} 1 40  {tankdir}/{sta}_{cha}_{net}_{loc}.tnk".format(
            sta=sta, cha=cha, net=net, loc=loc, reclen=reclen, inst_id=inst_id, mod_id=mod_id, tankdir=tankdir)
        WAVE_SERVERV.append(wave_serverV)

        # Create line for tank_player.d
        tankplayer = "WaveFile        {tankdir}/{sta}_{cha}_{net}_{loc}.tnk".format(
            sta=sta, cha=cha, net=net, loc=loc, tankdir=tankdir)
        TANKPLAYER.append(tankplayer)

    # Print lines to terminal
    print("::: Add these lines to wave_serverV.d")
    for line in WAVE_SERVERV:
        print(line)
    print()
    print("::: Add these lines to tankplayer.d")
    for line in TANKPLAYER:
        print(line)
    print()


def csv2ewsta():
    print("[IN DEVELOPMENT] Prints CSV lines into formatted lines for pick_ew.sta")

    line_fmt = "{space:4}{onoff:1}{space:3}{seq:<5}{sta:<6}{cha:<4}{net:<3}{loc:<3}3  40  3  20  500  0 0.854  3.  .4  .015 5.  .9961  1200.  409.59  .8  1.5  50000. 23689428".format(space=" ",
                 onoff=1, seq=500, sta="PV6A", cha="BHZ", net="AV", loc="--")

    print()
    print("#  --------------------------------------------------------------------------------------")
    print("#                 MinBigZC    RawDataFilt    LtaFilt         DeadSta          PreEvent")
    print("#  Station/  MinSmallZC   MaxMint        StaFilt       RmavFilt           AltCoda")
    print("#  Comp/Net  Itr1   MinPeakSize   CharFuncFilt  EventThresh          CodaTerm         Erefs")
    print("#  --------------------------------------------------------------------------------------")
    print(line_fmt)
    print("")


def nslc2ewstacsv():
    print("[IN DEVELOPMENT] Creates a pick_ew.sta file (in csv format) populated w desired NSLCs and pre-populated w other default values\nRun csv2ewsta() to create pick_ew.sta")
    print()

def nslc2subnet(nslc_list, subnetid=1, nstatrig=None):
    # Subnet 010  4  BRSP.BHZ.CC.-- HIYU.BHZ.CC.-- LSON.BHZ.CC.-- PALM.BHZ.CC.-- SHRK.BHZ.CC.-- TIMB.BHZ.CC.-- YOCR.BHZ.CC.-- AUG.EHZ.UW.--

    import numpy as np

    # if nstatrig is None, define it as a little more than half the number of stations
    if nstatrig is None:
        nstatrig = int(np.ceil(len(nslc_list)/2))

    subnet_str = "Subnet {subnetid:03d}  {nstatrig}  "
    
    scnl_list = []
    for nslc in nslc_list:
        n, s, l, c = nslc.split(".")
        scnl = " {}.{}.{}.{}".format(s,c,n,l)
        subnet_str += scnl
    
    print(subnet_str.format(subnetid=subnetid, nstatrig=nstatrig))


def subnet2nslc(subnet_str):
    # TODO: This isn't working yet
    
    subnet_list = subnet_str.split(" ")
    nslc_list = []
    for scnl in subnet_list[3:]:
        s, c, n, l = scnl.split(".")
        nslc = "{}.{}.{}.{}".format(n,s,l,c)
        nslc_list.append(nslc)  

    print(nslc_list)
    return nslc_list
