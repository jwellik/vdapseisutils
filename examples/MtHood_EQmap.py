# Shows usage of the MapFigure class.
# An example from Wy'East/Mt. Hood, USA.
# All example data from the internet.
# Starting w Copahue example

import matplotlib.pyplot as plt
import obspy

import vdapseisutils.maputils.data.volcanoes as volcanoes
from vdapseisutils.maputils.MapFigure import MapFigure
from utils import read_reav_csv, read_provig_csv

file1 = "/Users/jwellik/Dropbox/JAY-VDAP/_FY22/SeisComP_workflow/Copahue_cats_from_Victoria/Copahue_events_from_reavs.csv"
file2 = "/Users/jwellik/Dropbox/JAY-VDAP/_FY22/SeisComP_workflow/Copahue_cats_from_Victoria/Copahue_provig_VT_events.csv"


# Version that uses a Class
def main():
    print('#' * 50)

    volc = volcanoes.volcs["Mt. Hood"]
    inventory = obspy.read_inventory('/Users/jwellik/Dropbox/PROJECTS/sc4projects/Copahue/data/OAVV_okk.xml')
    cat, _ = read_provig_csv(file2)


    cat_sub = cat  # cat[0:-1:50]
    # print(cat_sub.__str__(print_all=True))

    # Define map parameters
    map_params = dict(origin=(volc['lat'], volc['lon']),
                      radial_extent=50,
                      depth_extent=[3.5, -40],
                      zoom=9, map_color=False,
                      title="Wy'East / Mt. Hood",
                      subtext="USGS Earthquake Catalog",
                      figsize=(8, 8),
                      )

    # Plot Figure
    mfig = MapFigure(**map_params)
    mfig.scatter_catalog(cat_sub)
    mfig.plot_inventory(inventory)
    mfig.plot_volcano(volc['lat'], volc['lon'])
    mfig.info()

    plt.show()

    print("Done.")
    print("#" * 50)


if __name__ == '__main__':
    main()

