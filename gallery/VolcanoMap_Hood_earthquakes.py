import os
import matplotlib.pyplot as plt
from obspy import read_events, read_inventory
from obspy.core.event import Catalog

from vdapseisutils.core.maps import VolcanoMap
from vdapseisutils.utils.obspyutils.catalogutils import catalog2txyzm, catalog2swarm
from vdapseisutils.utils.obspyutils.catalogutils import read_ew_arcfiles_as_catalog


volc = {'synonyms': "Wy'east",
 'lat': 45.374,
 'lon': -121.695,
 'elev': 3426,
}

radial_extent = 10  # kilometers
maxdepth = 20  # kilometers


def main():

    # Read events
    print("Loading catalogs...")
    cat_ew = read_ew_arcfiles_as_catalog("../data/arc_hood")  # Read Hyp2000 Arc files from Earthworm
    cat_nll = read_events("../data/hood_nll.sum.grid0.loc.hyp")[0:-1]  # Read Hypocenter summary from NonLinLoc

    # Output files
    cat_data = catalog2txyzm(cat_nll, filename="../data/hood_nll.sum.grid0.loc.csv")  # Save basic catalog data as a text file
    print(cat_data)
    catalog2swarm([cat_nll], "VV.NIE.LM.HHZ", ["default"], filename="../data/hood_nll.sum.grid0.loc.swarm.csv")  # Save as a Swarm tagger file

    # Plot catalogs
    print("Plotting catalogs...")
    plot_params = dict(lat=volc["lat"], lon=volc["lon"], radial_extent_km=radial_extent*1.1,
                    trange=[],
                    depth_extent=[maxdepth*-1, 4.0],  # np.ceil(volc["elev"]/1000)
                    xs1=(225, radial_extent, "A"),  # West-East Cross-Section
                    xs2=(315, radial_extent, "B"),  # North-South Cross-Section
                    title="Wy'East/Hood, Oregon, USA",
                    figsize=(6, 6), dpi=300,
                    zoom=10,
                    )
    fig = plt.figure(FigureClass=VolcanoMap, **plot_params)
    fig.plot_catalog(cat_ew, s=1.5, c="blue")  # , label="Hyp2000"
    fig.plot_catalog(cat_nll, s=1.5, c="red")  # , label="NLL"
    fig.add_legend()
    fig.info()
    plt.show()

    print("Done.")


if __name__ == '__main__':
    print('#'*80)
    main()
