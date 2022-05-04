

def wingplotT(catalog,  # ObsPy catalog object to plot
              lat=0.0, lon=0.0, radial_extent_km=50,  # center lat, lon, radial extent of map area
              xs1=(225, 30, "A"),  # start angle, radius, name; e.g., 225d from North at radius of 30 km (60 km long) (SW-NE)
              xs2=(315, 30, "B"),  # start angle, radius, name; e.g., 315d from North at radius of 30 km (60 km long) (NW-SE)
              map_type='terrain-background', zoom=11,
              depth_extent=(-50, 7),
              cmap="inferno_r",  # Colorbar for earthquake times
              trange=None,  # (start, stop) for time series colorbar
              default_mag=3,  # Magnitude that will plot at default scatter maker size; everything else scaled acc.
              title='VOLCANO MAP', subtitle='',
              figsize=(6, 6), dpi=300,
              show=True,
              verbose=False) -> object:

    """

    :param catalog:
    :param lat:
    :param lon:
    :param radial_extent_km:
    :param xs1:
    :param xs2:
    :param map_type:
    :param zoom:
    :param map_color:
    :param depth_extent:
    :param cmap:
    :param trange:
    :param default_mag:
    :param title:
    :param subtitle:
    :param figsize:
    :param dpi:
    :param show:
    :return:


    # [X] Project eq points to cross-section profile
    # [X] Make cross-sections use distance as horizontal measurement
    # [X] Set map extent
    # [X] Set cross section extents
    # [X] Make label sizes bigger, make lines thicker
    # [X] Clean up Magnitude legend and add text
    # [x] Add cross-section annotations to map and cross-sections
    # [x] Fix EQ Mag scales
    # [X] Add basemap
    # [ ] TODO Filter eqs to map extent
    # [ ] TODO Resize to make map bigger
    # [ ] TODO Add grid to map axis
    # [ ] TODO Add North arrow and scale bar
    # [ ] TODO Add radius?
    # [ ] TODO How many EQs excluded from map and cross section?
    # [ ] TODO Print Catalog stats somewhere
    # [X] Fig size and axis positions flexible (currently requires figsize of 12x12)
    """

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    from obspy import UTCDateTime

    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt

    import vdapseisutils.maputils.utils

    ####################################################################
    ## Define plot properties

    plt.rcParams['svg.fonttype'] = 'none'
    titlefontsize = t1fs = 16
    subtitlefontsize = t2fs = 14
    axlabelfontsize = axlf = 8
    annotationfontsize = afs = 8

    ####################################################################
    ## Prepare EQ data
    ## Get info out of Events object

    from vdapseisutils.eventutils.catalogutils import catalog2txyzm
    import pandas as pd
    # returns time(UTCDateTime), lat, lon, depth(km, positive below sea level), mag
    catdata = catalog2txyzm(catalog, time_format="matplotlib")
    catdata = pd.DataFrame(catdata)
    catdata["depth"] *= -1  # below sea level values are negative for plotting purposes

    if verbose:
        print(catdata)

    ####################################################################
    ## Download basemap tiles
    ## Get info out of Events object

    try:
        import urllib2
    except:
        pass

    tiles = cimgt.Stamen("terrain-background")

    ####################################################################
    ## Define Axis Positions
    # define axes positions (inches) LBWH
    s = 12  # square dimension (h == w)
    cw = 11  # content_width - smaller than figure size for padding
    ch = 9  # content_height
    map_pos = list(np.array([1, 4+((s-9)/2), 5, 5])/s)
    xs1_pos = list(np.array([6.25, 7+((s-9)/2), 4.5, 2])/s)
    xs2_pos = list(np.array([6.25, 4+((s-9)/2), 4.5, 2])/s)
    ts_pos  = list(np.array([1, 0.3+((s-9)/2), 7.5, 2.7])/s)
    cbar_pos = list(np.array([1, 0+((s-9)/2), 7.5, 0.3])/s)
    legend_pos = list(np.array([9, 0+((s-9)/2), 1.25, 3])/s)
    title_pos = list(np.array([6, 10+((s-9)/2), 12, 1])/s)
    subtitle_pos = list(np.array([6, 9.5+((s-9)/2), 12, 0.5])/s)

    # Make axes
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axm = fig.add_axes(map_pos, projection=tiles.crs)
    # axm = fig.add_axes(map_pos)
    ax1 = fig.add_axes(xs1_pos)  # Cross Section 1
    ax2 = fig.add_axes(xs2_pos)  # Cross Section 2
    axT = fig.add_axes(ts_pos)   # Time Series (Depth v Time)
    axL = fig.add_axes(legend_pos)  # Legend (Magnitude scale)
    axC = fig.add_axes(cbar_pos)  # Colorbar
    axL.set_visible(True)  # Should be false eventually
    axC.set_visible(True)

    # Title as custom text
    fig.text(title_pos[0], title_pos[1], title, fontsize=t1fs,
             verticalalignment='center', horizontalalignment='center')
    fig.text(subtitle_pos[0], subtitle_pos[1], subtitle, fontsize=t2fs,
             verticalalignment='center', horizontalalignment='center')

    ####################################################################
    ## Set Axis Tick Positions

    # Map
    # Geographic axis. This might not be necessary depending on how axis is made
    axm.tick_params(axis='both', labelsize=axlf, labelcolor='grey',
                    left=True, labelleft=False,
                    bottom=True, labelbottom=False,
                    right=False, labelright=False,
                    top=False, labeltop=False)

    # XSection1
    # Cartesian Axes
    ax1.tick_params(axis='both', labelsize=axlf, labelcolor='grey',
                    left=False, labelleft=False,
                    bottom=True, labelbottom=True,
                    right=True, labelright=True,
                    top=False, labeltop=False)
    ax1.yaxis.set_label_position("right")
    ax1.set_ylabel("Depth", rotation=270, fontsize=axlf, labelpad=10)
    ax1.set_xlabel("Distance (km)", fontsize=axlf, labelpad=5)

    # XSection2
    # Cartesian Axes
    ax2.tick_params(axis='both', labelsize=axlf, labelcolor='grey',
                    left=False, labelleft=False,
                    bottom=True, labelbottom=True,
                    right=True, labelright=True,
                    top=False, labeltop=False)
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Depth", rotation=270, fontsize=axlf, labelpad=10)
    ax2.set_xlabel("Distance (km)", fontsize=axlf, labelpad=5)

    # TimeSeries Axis
    # Cartesian Axes
    axT.tick_params(axis='both', labelsize=axlf, labelcolor='grey',
                    left=True, labelleft=True,
                    bottom=False, labelbottom=False,
                    right=False, labelright=False,
                    top=False, labeltop=False)
    axT.set_ylabel("Depth", fontsize=axlf, labelpad=0)

    # Magnitude Legend Axis
    # Cartesian Axes
    axL.tick_params(axis='both', labelsize=axlf, labelcolor='grey',
                    left=False, labelleft=False,
                    bottom=False, labelbottom=False,
                    right=True, labelright=True,
                    top=False, labeltop=False)
    axL.spines['top'].set_visible(False)
    axL.spines['bottom'].set_visible(False)
    axL.spines['left'].set_visible(False)
    axL.spines['right'].set_visible(False)

    # Colorbar Axis
    axC.tick_params(labelsize=axlf, labelcolor='grey')

    ####################################################################
    ## AXIS EXTENTS
    # Set XSection, Time Series Depth Extent
    map_extent = vdapseisutils.maputils.utils.utils.radial_extent2map_extent(lat, lon, radial_extent_km)
    axm.set_extent(map_extent, crs=ccrs.Geodetic())
    # axm.set_xlim(map_extent[0], map_extent[1])
    # axm.set_ylim(map_extent[2], map_extent[3])
    ax1.set_ylim([depth_extent[0], depth_extent[1]])
    ax2.set_ylim([depth_extent[0], depth_extent[1]])
    axT.set_ylim([depth_extent[0], depth_extent[1]])

    ####################################################################
    ## BASEMAP

    axm.add_image(tiles, zoom)
    # axm.background_img(name='ne_shaded', resolution='low', extent=map_extent, cache=False)

    ####################################################################
    ## CROSS SECTION PTS and TOPOGRAPHY
    # TODO change sight_point_pyproj to take km, not m
    from vdapseisutils.maputils.utils.utils import sight_point_pyproj
    from vdapseisutils.maputils.utils import elev_profile
    import matplotlib.patheffects as pe

    A1 = sight_point_pyproj((lat, lon), xs1[0], xs1[1]*1000)
    A2 = sight_point_pyproj((lat, lon), xs1[0]+180, xs1[1]*1000)
    B1 = sight_point_pyproj((lat, lon), xs2[0], xs2[1]*1000)
    B2 = sight_point_pyproj((lat, lon), xs2[0]+180, xs2[1]*1000)

    try:
        elev_data_A = elev_profile.download_profile(A1, A2, n=100)  # elevation returned in meters
        elev_data_B = elev_profile.download_profile(B1, B2, n=100)  # elevation returned in meters
        plot_profile = True
    # except HTTPError:
    except:
        print("There was a problem downloading elevation data. Moving on...")
        plot_profiles = False

    if plot_profile:
        # Plot data and format axis for A-A'
        h = elev_data_A['d']
        elev = np.array(elev_data_A['elev']) / 1000  # convert to km
        ax1.set_xlim([elev_data_A["d"][0], elev_data_A["d"][-1]])
        ax1.plot(h, elev, color="k", linewidth="0.75")
        # custom spine bounds for a nice clean look
        ax1.spines['top'].set_visible(False)
        ax1.spines["left"].set_bounds((depth_extent[0], elev[0]))  # depth_extent_v[1] is the top elev
        ax1.spines["right"].set_bounds(depth_extent[0], elev[-1])

        # Plot data and format axis for B-B'
        h = elev_data_B['d']
        elev = np.array(elev_data_B['elev']) / 1000  # convert to km
        ax2.set_xlim([elev_data_B["d"][0], elev_data_B["d"][-1]])
        ax2.plot(h, elev, color="k", linewidth="0.75")
        # custom spine bounds for a nice clean look
        ax2.spines['top'].set_visible(False)
        ax2.spines["left"].set_bounds((depth_extent[0], elev[0]))  # depth_extent_v[1] is the top elev
        ax2.spines["right"].set_bounds(depth_extent[0], elev[-1])

        # Plot cross-sections on map
        # axm.plot(A1[1], A1[0], "ok")
        # axm.plot(A2[1], A2[0], "ok")
        # axm.plot(B1[1], B1[0], "sk")
        # axm.plot(B2[1], B2[0], "sk")
        axm.plot([A1[1], A2[1]], [A1[0], A2[0]], linewidth=0.75, color="k", transform=ccrs.Geodetic())
        axm.plot([B1[1], B2[1]], [B1[0], B2[0]], linewidth=0.75, color="k", transform=ccrs.Geodetic())

        # Plot cross-section labels on map
        axm.text(A1[1], A1[0], "{}".format(xs1[2]), transform=ccrs.Geodetic(),
                fontsize=afs,
                verticalalignment='center', horizontalalignment='center',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        axm.text(A2[1], A2[0], "{}'".format(xs1[2]), transform=ccrs.Geodetic(),
                fontsize=afs,
                verticalalignment='center', horizontalalignment='center',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        axm.text(B1[1], B1[0], "{}".format(xs2[2]), transform=ccrs.Geodetic(),
                fontsize=afs,
                verticalalignment='center', horizontalalignment='center',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        axm.text(B2[1], B2[0], "{}'".format(xs2[2]), transform=ccrs.Geodetic(),
                fontsize=afs,
                verticalalignment='center', horizontalalignment='center',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

        # Plot cross-section labels on cross-section
        ax1.text(0, depth_extent[0], "{}".format(xs1[2]),
                fontsize=afs,
                verticalalignment='bottom', horizontalalignment='left',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax1.text(xs1[1]*2, depth_extent[0], "{}'".format(xs1[2]),
                fontsize=afs,
                verticalalignment='bottom', horizontalalignment='right',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax2.text(0, depth_extent[0], "{}".format(xs2[2]),
                fontsize=afs,
                verticalalignment='bottom', horizontalalignment='left',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax2.text(xs1[1]*2, depth_extent[0], "{}'".format(xs2[2]),
                fontsize=afs,
                verticalalignment='bottom', horizontalalignment='right',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    ## Project EQs to Cross-Sections
    # catdata must be a Pandas DataFrame with "lat" and "lon" columns
    print("Projecting points...")
    import pyproj
    import math

    # fwdA
    # backA
    # dA    : EQ distance along A-A'
    # dB    : EQ distance along B-B'
    # Create projection system
    geodesic = pyproj.Geod(ellps='WGS84')  # Create projection system
    # Angle of cross-section vectors
    fwdAA, backAA, distanceAA = geodesic.inv(A1[1], A1[0], A2[1], A2[0])  # (long, lat, long, lat)
    fwdBB, backBB, distanceBB = geodesic.inv(B1[1], B1[0], B2[1], B2[0])  # (long, lat, long, lat)

    # Store angle, distance from Cross-Section to Pt

    # (a) Try to do linearalized
    # catdata["fdwA"], catdata["backA"], catdata["dA"] = geodesic.inv([A1[1]]*len(catdata), [A1[0]]*len(catdata),
    #                                                                 catdata["lon"], catdata["lat"])  # long, lat, long, lat
    # catdata["fdwB"], catdata["backB"], catdata["dB"] = geodesic.inv([A1[1]]*len(catdata), [A1[0]]*len(catdata),
    #                                                                 catdata["lon"], catdata["lat"])  # long, lat, long, lat
    # catdata["alphaA"] = catdata["fdwA"] - fwdA
    # catdata["alphaB"] = catdata["fdwB"] - fwdB
    # catdata["dAA"] = catdata["dA"] * math.cos((catdata["alphaA"]) * (np.pi / 180))  # distance to pt along xsection line
    # print(catdata)

    # (b) for loops and lists
    FWDAA = []
    BACKAA = []
    DISTANCEAA = []
    ALPHAAA = []
    DAA = []

    FWDBB = []
    BACKBB = []
    DISTANCEBB = []
    ALPHABB = []
    DBB = []

    for idx, row in catdata.iterrows():
        fwdA, backA, distanceA = geodesic.inv(A1[1], A1[0], row["lon"], row["lat"])  # long, lat, long, lat
        fwdB, backB, distanceB = geodesic.inv(B1[1], B1[0], row["lon"], row["lat"])  # long, lat, long, lat
        alphaA = fwdA - fwdAA  # angle between A1-pt and A1-A2
        alphaB = fwdB - fwdBB  # angle between A1-pt and A1-A2
        dA = distanceA * math.cos((alphaA) * (np.pi / 180))  # distance to pt along xsection line
        dB = distanceA * math.cos((alphaB) * (np.pi / 180))  # distance to pt along xsection line
        FWDAA.append(fwdA)
        BACKAA.append(backA)
        DISTANCEAA.append(distanceA)
        ALPHAAA.append(alphaA)
        DAA.append(dA)
        FWDBB.append(fwdB)
        BACKBB.append(backB)
        DISTANCEBB.append(distanceB)
        ALPHABB.append(alphaB)
        DBB.append(dB)
    catdata["dAA"] = np.array(DAA)/1000  # Distance along cross-section A-A'
    catdata["dBB"] = np.array(DBB)/1000  # Distance along cross-section A-A'

    print("Done projecting points.")

    # Print Cross-Section info
    # print("::: XS Profiles")
    # print(": {name}-{name}' : {2.3f} - {2.3f}".format(A1, A2, name=xs1[2]))
    # print(": {name}-{name}' : {2.3f} - {2.3f}".format(A1, A2, name=xs2[2]))
    # print()

    ####################################################################




    ####################################################################
    ## MAGNITUDE LEGEND
    # Scale the magnitudes to marker size and scatter plot size
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    # default rcParams['line.markersize'] = 6
    # default scatter size is rcParams['line.markersize']**2
    # markersize is equivalent to line.markersize
    # size is equivalent to line.markersize**2 (the size of the scatter dot)

    default_marker = 6
    default_mag = default_mag
    catdata["markersize"] = default_marker * (2. ** (catdata["mag"] - default_mag))  # equals 'default_marker' with 'default_mag'
    # with a coefficient of 2 (above) the markersize doubles for every mag above default mag (and halves for every mag below it)
    catdata["size"] = catdata["markersize"] ** 2  # markersize**2 for the map

    ## Add scale box to figure
    # Plot scatter makers to scale axis
    scale_mag = np.array([-1, 0, 1, 2, 3, 4, 5])
    scale_mag_markersize = default_marker * (2. ** (scale_mag - default_mag))
    scale_mag_size = scale_mag_markersize ** 2
    axL.scatter([0, 0, 0, 0, 0, 0, 0], y=scale_mag, s=scale_mag_size, color='none', edgecolor='k')  # size or markersize?

    def legend_stats():
        for M, ms, s in zip(scale_mag, scale_mag_markersize, scale_mag_size):
            print("{:5.2f}, {:10.2f}, {:10.2f}".format(M, ms, s))

    # Determine number of EQs at each mag
    nmags = []
    for mag in scale_mag:
        rslt = catdata[(catdata["mag"] >= mag) & (catdata["mag"] < mag+1)]
        nmags.append(len(rslt))

    print("::: Magnitude Distribution")
    for M, N in zip(scale_mag, nmags):
        print(": M{:2d}  : {: 5d}".format(M, N))
    print()

    # Change settings on scale box axes
    # self.fig.axes[AX_MAG].set_ylim(ylim[0], ylim[1])  # Works best with exponential version
    axL.set_ylim(0, 6.5)  # Just guessing
    # axL.set_xlim(-0.04, 0.15)  # arbitrarily determined
    axL.set_xticks([])  # remove xticks
    axL.set_yticks(scale_mag)  # set yticks at height for each circle
    # axL.set_yticklabels(['M{}'.format(m) for m in scale_mag])  # give them a label in the format M3, for example
    # Is the following line working?
    axL.set_yticklabels(['M{} ({} eqs)'.format(m, n) for m, n in
                         zip(scale_mag, nmags)])  # give them a label in the format M3, for example
    axL.yaxis.tick_right()  # put yticklabels on the right
    # axL.tick_params(axis="y", direction="in", pad=-30, right=False)  # put labels on inside and remove ticks
    axL.tick_params(axis="y", direction="out", pad=0, right=False)  # put labels on inside and remove ticks
    axL.patch.set_alpha(0.0)  # set axis background to transparent

    ####################################################################
    ## COLORBAR
    # Use trange, if specified; otherwise, use catalog min/max

    tmin = UTCDateTime(trange[0]) if trange is not None else catalog[0].origins[-1].time
    tmax = UTCDateTime(trange[-1]) if trange is not None else catalog[0].origins[-1].time
    norm = mpl.colors.Normalize(vmin=tmin.matplotlib_date, vmax=tmax.matplotlib_date)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                           cax=axC, orientation='horizontal', label='Time')
    loc = mdates.AutoDateLocator()  # from matplotlib import dates as mdates
    cb.ax.xaxis.set_major_locator(loc)
    cb.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    ####################################################################
    ## PLOT DATA TO AXES

    alpha = 0.5
    kwargs = {}
    # print("MAP AXES TRANSFORM: {}".format(transform))
    axm.scatter(catdata["lon"], catdata["lat"], s=catdata["size"], c=catdata["time"],
                            norm=norm, cmap=cmap, alpha=alpha,
                            transform=ccrs.Geodetic(),
                            **kwargs)
    ax1.scatter(catdata["dAA"], catdata["depth"], s=catdata["size"], c=catdata["time"],
                            norm=norm, cmap=cmap, alpha=alpha, **kwargs)
    ax2.scatter(catdata["dBB"], catdata["depth"], s=catdata["size"], c=catdata["time"],
                            norm=norm, cmap=cmap, alpha=alpha, **kwargs)
    axT.scatter(catdata["time"], catdata["depth"], s=catdata["size"], c=catdata["time"],
                            norm=norm, cmap=cmap, alpha=alpha, **kwargs)

    # if show:
    #     # plt.show()

    return fig