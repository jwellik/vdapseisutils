"""
TODO Needs to be generalized before it's ready for GitHub. Download inventory from IRIS first.

"""


import obspy
from vdapseisutils.utils.nllutils import EQSTA, GTSRCE, LOCSRCE


def main():

    print("Write NonLinLoc control lines from local file...")

    inventory_file = '/home/jwellik/PROJECTS/PHASE_PICKS/sc4projects/Copahue/data/OAVV_20230925.xml'
    inventory_file = "/home/jwellik/Downloads/HawaiiNetwork_20240917_channel.xml"
    inventory = obspy.read_inventory(inventory_file)
    # inventory = inventory.select(location="LM", network="VV",
    #                              # maxradius=10/110, latitude=volc["lat"], longitude=volc["lon"],
    #                              )

    EQSTA(inventory)
    GTSRCE(inventory)
    LOCSRCE(inventory)

    print("Done.")


def nll_from_iris():

    print("Download inventory from IRIS and print NonLinLoc control lines...")

    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime

    client = Client("IRIS")
    starttime = UTCDateTime("2001-01-01")
    endtime = UTCDateTime("2001-01-02")
    inventory = client.get_stations(network="IU", station="A*",
                                    starttime=starttime,
                                    endtime=endtime)
    print(inventory)
    inventory.write("./NonLinLoc_Control_Tutorial.xml", format="XML")

    inventory_file = "./NonLinLoc_Control_Tutorial.xml"
    inventory = obspy.read_inventory(inventory_file)

    EQSTA(inventory)
    GTSRCE(inventory)
    # LOCSRCE(inventory)

    print()


if __name__ == "__main__":
    main()
    # nll_from_iris()
