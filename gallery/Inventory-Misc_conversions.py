import obspy
from vdapseisutils.utils.obspyutils import inventoryutils


def main():

    print("Misc. Inventory Operations")
    print()

    # Open inventory file
    inventory_file = '/home/jwellik/PROJECTS/sc4projects/Copahue/data/OAVV_okk.xml'
    inventory = obspy.read_inventory(inventory_file)
    inventory = inventory.select(network="VV", location="CP")
    # inventory_file = '../data/MtHood_inventory.xml'
    # inventory = obspy.read_inventory(inventory_file)


    # Write lines for NonLinLoc control file
    inventoryutils.write_nll_EQSTA(inventory)
    print()

    inventoryutils.write_nll_GTSRCE(inventory)
    print()

    # Write lines a Swarm LatLon.config file
    inventoryutils.write_swarm(inventory, verbose=True)
    print()

    print("Done.")
    print()

if __name__ == "__main__":
    main()
