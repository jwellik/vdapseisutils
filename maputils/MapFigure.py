import vdapseisutils.maputils.utils.utils as vmaputils


# Experimental (Unused) BasicMap class for easier map creation
class MapFigure:

    def __init__(self,
                 origin=(-77.53, 167.17),  # -> tuple    # (lat,lon) Defaults to Mount Erebus
                 radial_extent=50,  # -> float    #km
                 depth_extent=(3.5, -50),
                 zoom=12,
                 map_type='terrain-background',
                 map_color=True,
                 figsize=(12, 12),
                 title='Volcano Map',  # -> str
                 ):

        # Configurable features
        self.origin = origin
        self.radial_extent = radial_extent
        self.depth_extent = depth_extent
        self.zoom = zoom
        self.map_type = map_type
        self.map_color = map_color
        self.figsize = figsize
        self.title = title

        self.fig = _create_wingplot(self.origin[0], self.origin[1], radial_extent_km=self.radial_extent,
                                    zoom=self.zoom, map_type=self.map_type, map_color=self.map_color,
                                    figsize=self.figsize, title=self.title)

    # I/O

    # Print info about the map
    def info(self):
        print('::: {} (MapFigure) :::'.format(self.title))
        print('      origin        : {}'.format(self.origin))
        print('      radial_exetnt : {} km'.format(self.radial_extent))
        print('')

    # [Unimplemented] Save .png and .svg versions of the image. Handles oddities for publication quality images.
    def save(self):
        pass

    # Figure Features

    def set_figsize(self):
        pass

    def set_radial_extent(self):
        pass

    def set_extent(self):
        pass

    def get_extent(self):
        pass

    def set_origin(self):
        pass

    # not-necessary?
    def set_title(self, title):
        self.title = title
        self.fig.set_title(title)

    # Map Features

    def plot(self):
        pass

    # Change this to vmpautils call
    def plot_radius(self, lats, lons, rad_km):
        """
        Adds Tissot's indicatrices to the axes.
        https://scitools.org.uk/cartopy/docs/v0.15/_modules/cartopy/mpl/geoaxes.html#GeoAxes.tissot

        Kwargs:

            * rad_km - The radius in km of the the circles to be drawn.

            * lons - A numpy.ndarray, list or tuple of longitude values that
                     locate the centre of each circle. Specifying more than one
                     dimension allows individual points to be drawn whereas a
                     1D array produces a grid of points.

            * lats - A numpy.ndarray, list or tuple of latitude values that
                     that locate the centre of each circle. See lons.

            * n_samples - Integer number of points sampled around the
                          circumference of each circle.

        **kwargs are passed through to `class:ShapelyFeature`.

        """
        from cartopy import geodesic

        geod = geodesic.Geodesic()
        geoms = []

        if lon is None:
            lons = np.linspace(-180, 180, 6, endpoint=False)
        else:
            lons = np.asarray(lons)
        if lats is None:
            lats = np.linspace(-80, 80, 6)
        else:
            lats = np.asarray(lats)

        if lons.ndim == 1 or lats.ndim == 1:
            lons, lats = np.meshgrid(lons, lats)
        lons, lats = lons.flatten(), lats.flatten()

        if lons.shape != lats.shape:
            raise ValueError('lons and lats must have the same shape.')

        for i in range(len(lons)):
            circle = geod.circle(lons[i], lats[i], rad_km,
                                 n_samples=n_samples)
            geoms.append(sgeom.Polygon(circle))

        feature = cartopy.feature.ShapelyFeature(geoms, ccrs.Geodetic(),
                                                 **kwargs)
        return self.add_feature(feature)

    # Plot volcano
    def plot_volcano(self, *args, **kwargs):
        self.fig.axes[0] = vmaputils.plot_volcano(self.fig.axes[0], *args, **kwargs)

    # Plot Hypocenters
    def plot_hypo(self):
        pass

    # Plot Catalog
    def plot_catalog(self, catalog):

        # Plot to Map (handles hypo and error bars)
        self.fig.axes[0] = vmaputils.plot_catalog(self.fig.axes[0], catalog)
        # Plot to XSection (handles hypo and errors)
        self.fig = vmaputils.plot_catalog2xs(self.fig, catalog)

        # Set axes extents. Do this elsewhere?
        radextent = vmaputils.radial_map_extent(self.origin[0], self.origin[1],
                                                self.radial_extent)  # This needs to come right from the object
        lonextent = radextent[0:2];
        latextent = radextent[2:]  # This needs to come right form the object
        self.fig.axes[1].set_xlim(lonextent);
        self.fig.axes[1].set_ylim([-50, 3.5])
        self.fig.axes[2].set_ylim(latextent);
        self.fig.axes[2].set_xlim([3.5, -50])

    # Plot Stations
    def plot_station(self):
        pass

    # Plot Inventory
    def plot_inventory(self, inventory, include_xs=False):
        self.fig.axes[0] = vmaputils.plot_station_inventory(self.fig.axes[0], inventory)
        if include_xs:
            print('Not yet an option.')


def _create_wingplot(lat, lon, radial_extent_km=50,
                     map_type='terrain-background', map_color=True, zoom=9,
                     title='Volcano Map', figsize=(12, 12)) -> object:
    import matplotlib.pyplot as plt

    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt

    import vdapseisutils.maputils.utils

    plt.rcParams['svg.fonttype'] = 'none'

    try:
        import urllib2
    except:
        pass

    if map_color == True:
        tiles = cimgt.Stamen(map_type)
    else:
        tiles = cimgt.Stamen(map_type, desired_tile_form="L")

    # definitions for the axes (% of figure size)
    bottom, left = 0.05, 0.10
    top, right = 0.1, 0.1
    mwidth, mheight, xsheight = 0.65, 0.65, 0.2
    spacing = 0.005

    map_pos = [left, bottom + xsheight + spacing, mwidth, mheight]
    hxs_pos = [left, bottom, mwidth, xsheight]
    vxs_pos = [left + mwidth + spacing, bottom + xsheight + spacing, xsheight, mheight]

    # start with a square Figure
    fig = plt.figure(figsize=figsize)
    axm = fig.add_axes(map_pos, projection=tiles.crs, title=title)
    # axh = fig.add_axes(hxs_pos, sharex=axm)
    # axv = fig.add_axes(vxs_pos, sharey=axm)
    axh = fig.add_axes(hxs_pos)
    axv = fig.add_axes(vxs_pos)

    # Can this be handled better?
    extent = vdapseisutils.maputils.utils.utils.radial_map_extent(lat, lon, radial_extent_km)
    axm.set_extent(extent)

    if map_color == True:
        axm.add_image(tiles, zoom)
    else:
        axm.add_image(tiles, zoom, cmap='Greys_r')

    # Map gridlines
    glv = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=1, color='gray', alpha=0.5)
    glv.xlabels_top = True
    glv.xlabels_bottom = False
    glv.ylabels_left = True
    glv.ylabels_right = False
    glv.xlines = True
    # gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    glv.xlabel_style = {'size': 8, 'color': 'gray'}
    glv.ylabel_style = {'size': 8, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

    # Horizontal xsection gridlines
    # GeoAxes
    # glv = axh.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5)
    # glv.xlabels_top = False
    # glv.xlabels_bottom = False
    # glv.ylabels_left = True
    # glv.ylabels_right = False
    # glv.xlines = True
    # glv.ylabel_style = {'size': 8, 'color': 'gray'}
    # Cartesian Axes
    axh.tick_params(axis='both', labelsize=8, labelcolor='grey',
                    left=True, labelleft=True,
                    bottom=False, labelbottom=False,
                    right=False, labelright=False,
                    top=False, labeltop=False)

    # Vertical xsection gridlines
    # Cartesian Axes
    axv.tick_params(axis='both', labelsize=8, labelcolor='grey',
                    top=True, labeltop=True,
                    bottom=False, labelbottom=False,
                    left=False, labelleft=False,
                    right=False, labelright=False)

    # Draw plot but do not show
    plt.draw()

    return fig
