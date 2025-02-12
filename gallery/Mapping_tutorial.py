"""
KIHOLO BAY EARTHQUAKE
- https://www.usgs.gov/observatories/hvo/news/volcano-watch-foreshocks-mainshocks-and-aftershocks-oh-my

"""


import obspy
from obspy import UTCDateTime
import matplotlib.pyplot as plt
#from vdapseisutils.sandbox.maps.maps import Map, CrossSection, TimeSeries, VolcanoFigure
from vdapseisutils import Map, CrossSection, TimeSeries, VolcanoFigure

radial_extent = 10  # kilometers
maxdepth = 20  # kilometers

hood = dict()
hood["coords"] = [45.374, -121.695, 3426]
hood["name"] = "Hood/Wy'East"
rainier = dict()
rainier["coords"] = [46.853, -121.76, 4392]
kilauea = dict()
kilauea["coords"] = [19.421, -155.287, 1222]
kilauea["map_extent"] = [-155.500, -154.800, 19.200, 19.500]
mauna_kea = dict()
mauna_kea["coords"] = [19.82, -155.47, 1222]
mauna_kea["azimuth"] = 280
pavlof = dict()
pavlof["coords"] = [55.417, -161.894, 2439]
crater_lake = dict()
crater_lake["coords"] = [42.942, -122.107]
banua_wuhu = dict()
banua_wuhu["coords"] = [3.138, 125.491, -5]
banua_wuhu["map_extent"] = [124.085433, 128.538886, 0.966248, 4.532257]
banua_wuhu["points"] = [
    [(3.424226, 125.534186), "Sangihe Island"],
    [(2.798295, 125.419080), "Sitaro Island"],
    [(1.602387, 125.074145), "Sulawesi"],
    [(2.035740, 127.862160), "North Halmahera"],
    [(1.406734, 127.593366), "West Halmahera"],
    [(0.825030, 127.318614), "Ternate"],
]
gamalama = dict({
    "name": "Gamalama",
    "coords": [0.81, 127.3322],
})
ruang = dict({
    "name": "Ruang",
    "coords": [2.3, 125.37],
})



print("Loading catalogs from file...")
cat_kilauea = obspy.read_events("../data/catalog_kilauea_eruption.xml", format="QUAKEML")
cat_hood_msas = obspy.read_events("../data/catalog_hood_msas.xml", format="QUAKEML")
cat_kiholo_msas = obspy.read_events("../data/catalog_kiholo_bay_msas.xml", format="QUAKEML")
cat_tanaga_unkn = obspy.read_events("../data/catalog_tanaga_unknown.xml", format="QUAKEML")
cat_msh1980 = obspy.read_events("../data/catalog_msh1980_eruption.xml", format="QUAKEML")
print("Done.")


def map_usvolcs():

    print("Map")

    print(">> Raininer")
    fig = Map(origin=(rainier["coords"][0], rainier["coords"][1]), radial_extent_km=15.0)
    fig.add_hillshade(resolution="01s")
    fig.add_scalebar()
    fig.suptitle("Rainier, Washington")
    fig.info()
    fig.savefig("./output/Mapping_tutorial/Map_Rainier.png")
    plt.show()

    print(">> Kīlauea")
    fig2 = plt.figure(figsize=(8, 4), dpi=300)
    fig2 = Map(map_extent=kilauea["map_extent"], fig=fig2)
    fig2.add_hillshade()
    fig2.add_scalebar()
    fig2.plot_catalog(cat_kilauea)
    fig2.suptitle("Kīlauea")
    fig2.info()
    fig2.savefig("./output/Mapping_tutorial/Map_Kilauea.png")
    fig2.show()

    print("Done.")
    print()


def xsection_usvolcs():

    print("CrossSection: US Volcanoes")

    fig = plt.figure(figsize=(8, 10))
    subfigs = fig.subfigures(5, 1)

    print(">> Mauna Kea")
    subfigs[0].suptitle("Mauna Kea (Shield Volcano)")
    subfigs[0] = CrossSection(origin=(mauna_kea["coords"][0:2]), radius_km=25.0, azimuth=290, depth_extent=(-2, 5),
                              resolution=500.0, fig=subfigs[0])
    subfigs[0].info()

    print(">> Rainier")
    subfigs[1].suptitle("Rainier (Composite Stratovolcano)")
    subfigs[1] = CrossSection(origin=(rainier["coords"][0:2]), radius_km=25.0, depth_extent=(-2, 5),
                              resolution=500.0, fig=subfigs[1])
    subfigs[1].info()

    print(">> Pavlof")
    subfigs[2].suptitle("Pavlof (Stratvolcano)")
    subfigs[2] = CrossSection(origin=(pavlof["coords"][0:2]), radius_km=25.0, azimuth=0, depth_extent=(-2, 5),
                              resolution=500.0, fig=subfigs[2])
    subfigs[2].info()

    # Providing no location iformation draws a cross section near Mount St Helens, by default
    print(">> St Helens")
    subfigs[3].suptitle("Mount St. Helens: West to East")
    subfigs[3] = CrossSection(depth_extent=(-2, 5), radius_km=25.0, resolution=200.0, fig=subfigs[3])
    subfigs[3].info()

    print(">> Crater Lake")
    subfigs[4].suptitle("Crater Lake - West to East")
    subfigs[4] = CrossSection(points=[(42.967831, -122.349195), (42.895858, -121.845056)], depth_extent=(-2, 5), fig=subfigs[4])
    subfigs[4].info()

    fig.savefig("./output/Mapping_tutorial/CrossSection_USvolcs.png")
    plt.show()

    print("Done.")
    print()


def timeseries_msas_swarm():

    print("Timeseries")

    print(">> Mainshock-Aftershock Sequences & Swarms")
    # # Mainshocks, Aftershocks, and Swarms
    fig = plt.figure(figsize=(8, 6))
    subfigs = fig.subfigures(3, 1)

    subfigs[0].suptitle("Mainshock-Aftershock Sequence: Kiholo Bay, Hawaii")
    subfigs[0] = TimeSeries(fig=subfigs[0], axis_type="magnitude")
    subfigs[0].plot_catalog(cat_kiholo_msas, s=5, c="k")

    subfigs[1].suptitle("Volcanic Swarm: Kīlauea (pre- & syn-eruptive)")
    subfigs[1] = TimeSeries(fig=subfigs[1], axis_type="magnitude")
    subfigs[1].plot_catalog(cat_kilauea, s=5, c="k")
    subfigs[1].ax.set_ylim([1.5, 4.5])

    subfigs[2].suptitle("What do you think?: Tanaga, Alaska")
    subfigs[2] = TimeSeries(fig=subfigs[2], axis_type="magnitude")
    subfigs[2].plot_catalog(cat_tanaga_unkn, s=5, c="k")
    fig.savefig("./output/Mapping_tutorial/TimeSeries_eq_swarms.png")
    plt.show()


    ## Mount St Helens 1980 Eruption and Earthquake Depths
    print(">> St Helens 1980 Unrest")
    fig = TimeSeries()
    fig.suptitle("Earthquake depth: Mount St Helens 1980")
    fig.axvline(UTCDateTime("1980/03/27"), color="b")  # First phreatic eruption
    fig.axvline(UTCDateTime("1980/05/18"), color="r")  # T08:32:17-07 # M5.1 earthquake and debris avalanche
    fig.plot_catalog(cat_msh1980, s=5, c="k")
    fig.savefig("./output/Mapping_tutorial/TimeSeries_MSH1980.png")
    plt.show()

    print("Done.")
    print()


def map_xsction_kilauea():

    print("Map and CrossSection")

    print(">> Kīlauea")
    fig = Map(map_extent=kilauea["map_extent"])
    fig.add_hillshade()
    fig.plot_catalog(cat_kilauea)
    fig.set_title("Kīlauea")
    fig.info()
    fig.savefig("./output/Mapping_tutorial/MapCrossSection_Kilauea.png")
    plt.show()

    print("Done.")
    print()


def volcano_figure():

    """
    Each plotting class: Map, CrossSection, TimeSeries returns a Matplotlib Figure object. The VolcanoFigure class
    creates a Figure object with subfigures for each plot.
    """

    print("VolcanoFigure")

    print(">> Mt Hood")
    volc = hood
    cat = cat_hood_msas

    fig2 = plt.figure(figsize=(8, 8), dpi=300)  # width, height
    fig2 = VolcanoFigure(fig=fig2, origin=(volc["coords"][0], volc["coords"][1]), depth_extent=(-5, 4), radial_extent_km=5.0, ts_axis_type="magnitude")
    fig2.plot_catalog(cat, s="magnitude")
    fig2.title("Mount Hood / Wy'East, Oregon")
    fig2.catalog_subtitle(cat)
    fig2.magnitude_legend(cat)
    fig2.reftext("Jay Wellik (Volcano Disaster Assistance Program)")
    fig2.savefig("./output/Mapping_tutorial/VolcanoFigure_Hood.png")
    plt.show()

    print(">> Ruang, Indonesia")
    volc = ruang

    fig2 = plt.figure(figsize=(8, 8), dpi=300)  # width, height
    fig2 = VolcanoFigure(fig=fig2, origin=(volc["coords"][0], volc["coords"][1]), depth_extent=(-5, 3), radial_extent_km=20.0,
                         xs1={'azimuth': 270-45}, xs2={'azimuth': 360-45},
                         ts_axis_type="magnitude")
    # fig2.add_hillshade(topo=False, bath=True, data_source="igpp", resolution="01s")
    fig2.title("Ruang")
    fig2.reftext("Jay Wellik (Volcano Disaster Assistance Program)")
    fig2.savefig("./output/Mapping_tutorial/VolcanoFigure_Ruang.png")
    plt.show()

    print("Done.")


def bathymetry_map_xsection():

    print(">>> Plot from components (Banua Wuhu, Indonesia)...")

    volc = banua_wuhu

    # Banua Wuhu
    fig = plt.figure(figsize=(16/2, 12/2), dpi=300)  # width, height
    spec = fig.add_gridspec(8, 8)  # rows, columns (height, width)
    fig_m = fig.add_subfigure(spec[0:8, 0:8])  #
    fig_m = Map(map_extent=[125.239416, 125.717357, 2.998957, 3.344241], fig=fig_m)
    fig_m.add_hillshade(resolution="01s", topo=True, bath=False)
    fig_m.add_scalebar()
    fig_m.plot(volc["coords"][0], volc["coords"][1], "^r")
    plt.savefig("./output/Mapping_tutorial/Bathymetry_BanuaWuhu.png")
    plt.show()

    # Regional Map
    fig = plt.figure(figsize=(16/2, 12/2), dpi=300)  # width, height
    spec = fig.add_gridspec(8, 8)  # rows, columns (height, width)
    fig_m = fig.add_subfigure(spec[0:8, 0:8], zorder=0)  #
    fig_m = Map(map_extent=volc["map_extent"], fig=fig_m)
    fig_m.add_locationmap(volc["coords"][0], volc["coords"][1])
    fig_m.add_hillshade(resolution="15s")
    fig_m.add_scalebar()
    fig_m.plot(volc["coords"][0], volc["coords"][1], "^r")
    [fig_m.plot(p[0][0], p[0][1], "ok") for p in volc["points"]]
    plt.savefig("./output/Mapping_tutorial/Bathymetry_BanuaWuhu_regional.png")
    plt.show()

    # Regional Map
    fig = plt.figure(figsize=(16/2, 8/2))  # width, height
    spec = fig.add_gridspec(6, 2)  # rows, columns (height, width)
    fig_m = fig.add_subfigure(spec[0:6, 0:1], zorder=0)  #
    fig_m = Map(map_extent=volc["map_extent"], fig=fig_m)
    fig_m.add_locationmap(volc["coords"][0], volc["coords"][1])
    fig_m.add_hillshade(resolution="15s")
    fig_m.plot(volc["coords"][0], volc["coords"][1], "^r")
    [fig_m.plot(p[0][0], p[0][1], "ok") for p in volc["points"]]
    # plt.savefig("./output/Mapping_tutorial/Bathymetry_BanuaWuhu_regional.png")

    # Multiple Cross-Sections to various points
    i = 0
    for p in volc["points"]:
        fig_i = fig.add_subfigure(spec[i, 1:2], zorder=0)  #
        xs = CrossSection(points=[volc["coords"], p[0]], depth_extent=(-5, 2), resolution=500.0, fig=fig_i)
        xs.suptitle("Banua Wuhu to {}".format(p[1]))
        i += 1
    plt.savefig("./output/Mapping_tutorial/Map_CrossSection_BanuaWuhu.png".format(i))
    plt.show()

    print("Done.")
    print()


if __name__ == '__main__':
    print('#'*80)
    map_usvolcs()
    xsection_usvolcs()
    timeseries_msas_swarm()
    volcano_figure()
    bathymetry_map_xsection()
    print("#"*80)
