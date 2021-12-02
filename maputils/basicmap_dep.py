def create_wingplot(lat, lon, radial_extent_km=50,
                    map_type='terrain-background', map_color=True, zoom=9,
                    title='Volcano Map', figsize=(12,12)) -> object:

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

    extent = vdapseisutils.maputils.utils.radial_map_extent(lat, lon, radial_extent_km)
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


# Temporary Function to create a basic map axis
def create(lat, lon, radial_extent_km=50,
           map_type='terrain-background', map_color=True, zoom=9,
           figsize=[10, 10]) -> object:
    import cartopy.crs as ccrs

    import cartopy.io.img_tiles as cimgt
    import matplotlib.pyplot as plt
    import vdapseisutils.maputils.utils

    plt.rcParams['svg.fonttype'] = 'none'

    try:
        import urllib2
    except:
        pass

    # plt.style.use('./utils/eqmap.mplstyle')

    if map_color == True:
        tiles = cimgt.Stamen(map_type)
    else:
        tiles = cimgt.Stamen(map_type, desired_tile_form="L")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=tiles.crs)

    extent = vdapseisutils.maputils.utils.radial_map_extent(lat, lon, radial_extent_km)
    ax.set_extent(extent)

    if map_color == True:
        ax.add_image(tiles, zoom)
    else:
        ax.add_image(tiles, zoom, cmap='Greys_r')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5)
    gl.xlabels_top = True
    gl.xlabels_bottom = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    # gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

    plt.draw()
    return ax


def hsection(lonlim, zlim):
    """

    @param lonlim: list-like : longitude axis limits
    @param zlim: list-like : depth axis limits
    @return: Axis object : Matplotlib Axis object
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(1, 1, 1)
    if lonlim[1] > lonlim[0]: ax.invert_xaxis()
    ax.set_xlim(lonlim)
    ax.set_ylim(zlim)

    plt.draw()
    return ax


def vsection(latlim, zlim):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.invert_xaxis()
    ax.set_ylim(latlim)
    ax.set_xlim(zlim)

    return ax



