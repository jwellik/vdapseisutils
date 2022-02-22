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
    print("::: Add these lines to wave_serverV.d")
    for line in TANKPLAYER:
        print(line)
    print()



