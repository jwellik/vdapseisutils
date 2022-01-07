'''TO DO

[ ] Limit earthquake catalog to map extent or xsection extent
[ ] We could enumerate the FIGURE AXES SO THERE IS NO CONFUSION

'''

import numpy as np

import vdapseisutils.maputils.utils.utils as vmaputils
from vdapseisutils.maputils.utils import elev_profile, elev_profile

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

from obspy import UTCDateTime


## ENUMERATIONS
# Axes contained by Figure
AXM = 0
AXH = 1
AXV = 2
AX_MAG = 3
AX_CBAR = 4

# BasicMap class for easier map creation
class MapFigure:

    def __init__(self,
                 origin=(-77.53, 167.17),  # -> tuple    # (lat,lon) Defaults to Mount Erebus
                 radial_extent=50,  # -> float    #km
                 depth_extent=(4.0, -50),  # -> float # km (top_altitude, bottom_altitude)
                 zoom=12,
                 map_type='terrain-background',
                 map_color=True,
                 figsize=(12, 12),
                 title='Volcano Map',  # -> str
                 subtext='',
                 ):

        # Configurable features
        self.origin = origin
        self.radial_extent = radial_extent
        self.map_extent = vmaputils.radial_extent2map_extent(origin[0], origin[1], radial_extent)
        self.depth_extent = depth_extent
        self.depth_extent_h = (depth_extent[1], depth_extent[0])  # inverted depth_extent used for horizontal x-section
        self.zoom = zoom
        self.map_type = map_type
        self.map_color = map_color
        self.figsize = figsize
        self.title = title
        self.subtext = subtext

        self.fig = _create_wingplot(self.origin[0], self.origin[1], radial_extent_km=self.radial_extent,
                                    zoom=self.zoom, map_type=self.map_type, map_color=self.map_color,
                                    depth_extent=depth_extent,
                                    figsize=self.figsize, title=self.title, subtext=self.subtext)
        self.add_default_profile()

    # I/O

    # Print info about the map
    def info(self):
        print('::: {} (MapFigure) :::'.format(self.title))
        print('      origin        : {}'.format(self.origin))
        print('      radial_exetnt : {} km'.format(self.radial_extent))
        print('      depth_exetnt  : {}:{} km'.format(self.depth_extent[0], self.depth_extent[1]))
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

    def set_depth_extent(self, depth_extent):
        self.depth_extent = depth_extent
        self.depth_extent_v = (depth_extent[1], depth_extent[0])

    # not-necessary?
    def set_title(self, title):
        self.title = title
        self.fig.set_title(title)

    # Basic Plotting routines

    def scatter(self, lat, lon, *args, **kwargs):
        '''Generic scatter plotting. Assumes data input are lat, lon, depth (optional), **kwargs

        **kwargs : any keyword arguments understood by matplotlib.pyplot.scatter()
        '''

        self.fig.axes[AXM].scatter(lon, lat, **kwargs)
        if len(args) > 0:
            depth = args[0]
            self.fig.axes[AXH].scatter(lon, depth * -1)
            self.fig.axes[AXV].scatter(depth * -1, lat)

        # Set axes extents. Do this elsewhere?
        radextent = vmaputils.radial_extent2map_extent(self.origin[0], self.origin[1],
                                                       self.radial_extent)  # This needs to come right from the object
        lonextent = radextent[0:2];
        latextent = radextent[2:]  # This needs to come right form the object
        self.fig.axes[AXH].set_xlim(lonextent)
        self.fig.axes[AXH].set_ylim(self.depth_extent_h)
        self.fig.axes[AXV].set_ylim(latextent)
        self.fig.axes[AXV].set_xlim(self.depth_extent)

    # Volcano Map Plotting routines

    # Change this to vmpautils call
    def plot_radius(self, lats, lons, rad_km, n_samples=90):
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
        # import matplotlib.axes
        import numpy as np
        import shapely.geometry as sgeom

        import cartopy.crs as ccrs
        import cartopy.img_transform

        # assert matplotlib.__version__ >= '1.3', ('Cartopy is only supported with '
        #                                          'matplotlib 1.3 or greater.')

        from cartopy import geodesic

        geod = geodesic.Geodesic()
        geoms = []

        if lons is None:
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

        # feature = cartopy.feature.ShapelyFeature(geoms, ccrs.Geodetic(),
        #                                          **kwargs)
        feature = cartopy.feature.ShapelyFeature(geoms, ccrs.Geodetic())
        return self.add_feature(feature)

    # Plot volcano
    def plot_volcano(self, *args, **kwargs):
        self.fig.axes[AXM] = vmaputils.plot_volcano(self.fig.axes[0], *args, **kwargs)

    # Plot hypocenter
    def plot_hypo(self, lat, lon, *args, transform=ccrs.Geodetic(), marker='o', color='black', markersize=8, alpha=0.95,
                  **kwargs):
        """*args is supposed to be an optional length 1 to provide the depth"""

        self.fig.axes[AXM].plot(lon, lat, transform=transform, marker=marker, color=color, markersize=markersize,
                              alpha=alpha, **kwargs)

        if len(args) > 0:
            depth = args[0]
            self.fig.axes[AXH].plot(lon, depth * -1, marker=marker, color=color, markersize=markersize, alpha=alpha,
                                  **kwargs)
            self.fig.axes[AXV].plot(depth * -1, lat, marker=marker, color=color, markersize=markersize, alpha=alpha,
                                  **kwargs)

        # Set axes extents. Do this elsewhere?
        radextent = vmaputils.radial_extent2map_extent(self.origin[0], self.origin[1],
                                                       self.radial_extent)  # This needs to come right from the object
        lonextent = radextent[0:2]
        latextent = radextent[2:]  # This needs to come right form the object
        self.fig.axes[AXH].set_xlim(lonextent)
        self.fig.axes[AXH].set_ylim(self.depth_extent_h)
        self.fig.axes[AXV].set_ylim(latextent)
        self.fig.axes[AXV].set_xlim(self.depth_extent)

        # self.fig.axes[1].plot(lon, depth*-1, marker=marker, color=color, markersize=markersize, alpha=alpha)  # horizontal cross-section
        # self.fig.axes[2].plot(depth*-1, lat, marker=marker, color=color, markersize=markersize, alpha=alpha)  # vertical cross-section

    # Plot Catalog
    def plot_catalog(self, catalog):
        print('!!! This function relies on stub setting of cross-section extents')

        # Plot to Map (handles hypo and error bars)
        self.fig.axes[AXM] = vmaputils.plot_catalog(self.fig.axes[0], catalog)
        # Plot to XSection (handles hypo and errors)
        self.fig = vmaputils.plot_catalog2xs(self.fig, catalog)

        # Set axes extents. Do this elsewhere?
        radextent = vmaputils.radial_extent2map_extent(self.origin[0], self.origin[1],
                                                       self.radial_extent)  # This needs to come right from the object
        lonextent = radextent[0:2]
        latextent = radextent[2:]  # This needs to come right from the object
        self.fig.axes[AXH].set_xlim(lonextent)
        self.fig.axes[AXH].set_ylim(self.depth_extent_h)
        self.fig.axes[AXV].set_ylim(latextent)
        self.fig.axes[AXV].set_xlim(self.depth_extent)

    def scatter_catalog(self, catalog, cmap='viridis_r', transform=ccrs.Geodetic(), alpha=0.5, **kwargs):
        print('!!! scatter_catalog() In development')

        import matplotlib as mpl
        import matplotlib.dates as mdates

        self.fig.axes[AX_MAG].set_visible(True)
        self.fig.axes[AX_CBAR].set_visible(True)  # Turn the axis ON

        # set up scatter colorbar
        norm = mpl.colors.Normalize(vmin=catalog[0].origins[-1].time.matplotlib_date,
                                    vmax=catalog[-1].origins[-1].time.matplotlib_date)
        cb = self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                               cax=self.fig.axes[4], orientation='horizontal', label='Time')
        loc = mdates.AutoDateLocator()  # from matplotlib import dates as mdates
        cb.ax.xaxis.set_major_locator(loc)
        cb.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

        # Get info out of Events object
        from vdapseisutils.eventutils.catalogutils import catalog2basics
        time, lat, lon, depth, mag = catalog2basics(catalog)
        depth = list(np.array(depth)/1000*-1)  # meters to km

        # Set up the magnitude scale parameters
        mso = 0 if min(mag) >= 0 else np.floor(np.min(mag)) * -1  # magnitude offset scale to avoid negatives
        mag += mso  # adjusted Magnitude value for plotting purposes
        scale_mag = np.array([1, 2, 3, 4, 5]) + mso  # Array of values for the scale box

        # Define size for each marker
        scale_type = 'linear'
        # #  -Exponential
        if scale_type == 'exponential':
            scatter_scale = 2  # converts magnitude to scatter plot size (try many numbers across orders of magnitude)
            s = scatter_scale ** mag  # markersize**2 for the map
            scale_s = scatter_scale ** scale_mag  # markersize**2 for scale box; alternatively, ms=scatter_scale (and use ms in the scatter() function)
        #  -Linear
        elif scale_type == 'linear':
            scatter_scale = 1  # converts magnitude to scatter plot size (try many numbers across orders of magnitude)
            s = scatter_scale * mag ** 2  # markersize**2 for the map
            scale_s = scatter_scale * scale_mag ** 2  # markersize**2 for scale box; alternatively, ms=scatter_scale (and use ms in the scatter() function)

        # Plot to axes
        self.fig.axes[AXM].scatter(lon, lat, s=s, c=time,
                                 norm=norm, cmap=cmap, transform=transform, alpha=alpha, **kwargs)
        self.fig.axes[AXH].scatter(lon, depth, s=s, c=time,  # Horizontal XSection
                                   norm=norm, cmap=cmap, alpha=alpha, **kwargs)
        self.fig.axes[AXV].scatter(depth, lat, s=s, c=time,  # Vertical XSection
                                   norm=norm, cmap=cmap, alpha=alpha, **kwargs)


        # MAGNITUDE SCALE (define positiong and limits)
        mag_scale_xpos = np.array([0] * len(scale_mag))  # xpos of mag scale circles is 0, make the array
        if scale_type == 'exponential':
            mag_scale_ypos = scale_mag ** 2  # largest on top, smallest on bottom, 1 order of magnitude apart looks nice
            ylim = (0, (scale_mag[-1] + 2) ** 2)
        elif scale_type == 'linear':
            mag_scale_ypos = scale_mag * 10
            ylim = (scale_mag[1] * 10 - 15, scale_mag[-1] * 10 + 15)

        # Add scale box to figure
        self.fig.axes[AX_MAG].scatter(mag_scale_xpos, y=mag_scale_ypos, s=scale_s, color='none',
                                 edgecolor='k')  # Plot scatter makers to scale axis

        # Change settings on scale box axes
        self.fig.axes[AX_MAG].set_ylim(ylim[0], ylim[1])  # Works best with exponential version
        self.fig.axes[AX_MAG].set_xlim(-0.03, 0.05)  # arbitrarily determined
        self.fig.axes[AX_MAG].set_xticks([])  # remove xticks
        self.fig.axes[AX_MAG].set_yticks(mag_scale_ypos)  # set yticks at height for each circle
        self.fig.axes[AX_MAG].set_yticklabels(
            ['M{}'.format(m - mso) for m in scale_mag])  # give them a label in the format M3, for example
        self.fig.axes[AX_MAG].yaxis.tick_right()  # put yticklabels on the right
        self.fig.axes[AX_MAG].tick_params(axis="y", direction="in", pad=-30, right=False)  # put labels on inside and remove ticks
        [self.fig.axes[AX_MAG].spines[pos].set_visible(False) for pos in ["top", "bottom", "left", "right"]]  # remove axis frames
        self.fig.axes[AX_MAG].patch.set_alpha(0.0)  # set axis background to transparent

        # Set axes extents. Do this elsewhere?
        radextent = vmaputils.radial_extent2map_extent(self.origin[0], self.origin[1],
                                                       self.radial_extent)  # This needs to come right from the object
        lonextent = radextent[0:2]
        latextent = radextent[2:]  # This needs to come right from the object

        # Why is all this stuff happening here?
        self.fig.axes[AXH].set_xlim(lonextent)
        self.fig.axes[AXH].set_ylim(self.depth_extent_h)
        self.fig.axes[AXV].set_ylim(latextent)
        self.fig.axes[AXV].set_xlim(self.depth_extent)
        self.fig.axes[AXH].set_yticks([0,-5,-10,-15])
        self.fig.axes[AXV].set_xticks(self.fig.axes[AXH].get_yticks())  # Depth tick locations same for both x-sections

    # Plot Stations
    def plot_station(self):
        pass

    # Plot Inventory
    def plot_inventory(self, inventory):
        self.fig.axes[AXM] = vmaputils.plot_station_inventory(self.fig.axes[0], inventory)

    # ADD CROSS-SECTIONAL PROFILES

    # Cross-Section Profile (Generic)
    # !!! Draw lines across map
    def add_profile_p1p2(self, P1, P2, n=100, axis='h', depth=50, color='black', linewidth=1):
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # THIS ASSUMES THAT THE BASIC WINGPLOT IS STRUCTURED SUCH THAT
        # FIG.AXES[1] IS THE HORIZONTAL CROSS-SECTION &
        # FIG.AXES[2] IS THE VERTICAL CROSS-SECTION
        #
        # depth is in km

        d, elev = elev_profile.download_profile(P1, P2, n=n)  # elevation returned in meters
        elev /= 1000  # convert to km

        # PLOT ELEVATION PROFILE
        base_reg = depth * -1
        if axis == 'h':
            self.fig.axes[AXH].plot(d, elev, color=color, linewidth=linewidth)
            self.fig.axex[AXH].fill_between(d, elev, base_reg, color=color, alpha=0.1)
        if axis == 'v':
            self.fig.axes[AXV].plot(elev, d, color=color, linewidth=linewidth)
            self.fig.axes[AXV].fill_between(elev, d, base_reg, color=color, alpha=0.1)

    # Adds EW profile (horizontal); NS profile (vertical)
    def add_default_profile(self, n=100, color='black', linewidth=1, drawmapline=False):

        # 1) Determine lat/lon for horiztonal P1, P2
        # 2) Determine lat/lon for vertical P1/P2
        # 3) Download profile for h/v
        # 4) Plot profile for h/v

        # Determine (lat, lon) for XC points
        # A1 lat is midway between minlat/maxlat; B2 lon is midway between minlon/maxlon
        A1 = ((self.map_extent[2] + self.map_extent[3]) / 2, self.map_extent[0])
        A2 = ((self.map_extent[2] + self.map_extent[3]) / 2, self.map_extent[1])
        B1 = (self.map_extent[2], (self.map_extent[0] + self.map_extent[1]) / 2)
        B2 = (self.map_extent[3], (self.map_extent[0] + self.map_extent[1]) / 2)

        # Download & plot elevation data for A-A'
        lat, lon, d, elev = elev_profile.download_profile2(A1, A2, n=n)  # elevation returned in meters
        elev = np.array(elev) / 1000  # convert to km
        self.fig.axes[AXH].plot(lon, elev, color=color, linewidth=linewidth)
        # custom spine bounds for a nice clean look
        self.fig.axes[AXH].spines['top'].set_visible(False)
        self.fig.axes[AXH].spines.left.set_bounds((self.depth_extent[1], elev[0]))  # depth_extent[1] is the top elev
        self.fig.axes[AXH].spines.right.set_bounds((self.depth_extent[1], elev[-1]))


        # Download & plot elevation data for B-B'
        lat, lon, d, elev = elev_profile.download_profile2(B1, B2, n=n)  # elevation returned in meters
        elev = np.array(elev) / 1000  # convert to km
        self.fig.axes[AXV].plot(elev, lat, color=color, linewidth=linewidth)
        # custom spine bounds for a nice clean look
        self.fig.axes[AXV].spines['left'].set_visible(False)
        self.fig.axes[AXV].spines.bottom.set_bounds((self.depth_extent[1], elev[0]))  # depth_extent[1] is the top elev
        self.fig.axes[AXV].spines.top.set_bounds((self.depth_extent[1], elev[-1]))


    def add_profile(self, *args, **kwargs):
        # If *args: add_profile_p1p2
        # If no *args: add_default_profile
        pass


def _create_wingplot(lat, lon, radial_extent_km=50.,
                     map_type='terrain-background', map_color=True, zoom=9,
                     depth_extent=(7.0, -50.),
                     title='Volcano Map', subtext='',
                     figsize=(12, 12)) -> object:
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
    bottom, left = 0.08, 0.10
    top, right = 0.1, 0.1
    mwidth, mheight, xsheight = 0.55, 0.55, 0.2
    cbar_height = 0.02
    spacing = 0.005

    # define axes positions
    map_pos = [left, bottom + cbar_height*2 + xsheight + spacing, mwidth, mheight]
    hxs_pos = [left, bottom + cbar_height*2, mwidth, xsheight]
    vxs_pos = [left + mwidth + spacing, bottom + cbar_height*2 + xsheight + spacing, xsheight, mheight]
    mag_scale_pos = [left + mwidth + spacing, bottom + cbar_height*2, xsheight, xsheight]
    cbar_pos = [left, bottom, mwidth + spacing + xsheight, cbar_height]
    title_pos = [0.5, 0.965]
    subtext_pos = [0.5, 0.93]

    # start with a square Figure
    fig = plt.figure(figsize=figsize)
    axm = fig.add_axes(map_pos, projection=tiles.crs)
    axh = fig.add_axes(hxs_pos)
    axv = fig.add_axes(vxs_pos)
    mag_ax = fig.add_axes(mag_scale_pos)
    cbar_ax = fig.add_axes(cbar_pos)
    mag_ax.set_visible(False)
    cbar_ax.set_visible(False)

    # Title as custom text
    fig.text(title_pos[0], title_pos[1], title, fontsize=14,
             verticalalignment='center', horizontalalignment='center')
    fig.text(subtext_pos[0], subtext_pos[1], subtext, fontsize=10,
             verticalalignment='center', horizontalalignment='center')

    # Can this be handled better?
    # Shouldn't this just be a part of the object
    extent = vdapseisutils.maputils.utils.utils.radial_extent2map_extent(lat, lon, radial_extent_km)
    axm.set_extent(extent)

    if map_color == True:
        axm.add_image(tiles, zoom)
    else:
        axm.add_image(tiles, zoom, cmap='Greys_r')

    # Map gridlines
    glv = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5)
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

    # Set XSection Depth Extent
    # depth extents
    axh.set_ylim([depth_extent[1], depth_extent[0]])
    axv.set_xlim([depth_extent[0], depth_extent[
        1]])  # Flipping the normal order of the axis lim values creates an axis in "reverse" order

    # lat/lon extents
    axh.set_xlim(axm.get_xlim())
    axv.set_ylim(axm.get_ylim())

    # Draw plot but do not show
    plt.draw()

    return fig
