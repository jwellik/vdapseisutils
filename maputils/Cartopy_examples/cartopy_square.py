import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt

import matplotlib.patches as mpatches



extent_eyja      = [-22, -15, 63, 65]
extent_augustine = [-153.75, -153.25, 59.25, 59.5]
pt_eyja      = [-19.613333, 63.62]       # lat,lon
pt_augustine = [-153.43, 59.363]
r_degrees=10/110


def compute_radius(ortho, lat=pt_eyja[0], lon=pt_eyja[1], radius_degrees=r_degrees):
    phi1 = lat + radius_degrees if lat <= 0 else lat - radius_degrees
    _, y1 = ortho.transform_point(lon, phi1, ccrs.PlateCarree())
    return abs(y1)


def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax

def main(extent=extent_eyja, pt=pt_eyja, radius=20, zoom=8):
    request = cimgt.Stamen('terrain-background')   # very responsive
    crg = request.crs   #crs of the projection
    fig, ax = make_map(projection = crg)

    r_ortho = compute_radius(crg, lat=pt[0], lon=pt[1], radius_degrees=radius/110)

    # specify map extent here
    #lonmin, lonmax = -22, -15
    #latmin, latmax = 63, 65
    lonmin, lonmax = extent[0], extent[1]
    latmin, latmax = extent[2], extent[3]

    LL = crg.transform_point(lonmin, latmin, ccrs.Geodetic())
    UR = crg.transform_point(lonmax, latmax, ccrs.Geodetic())
    EW = UR[0] - LL[0]
    SN = UR[1] - LL[1]
    side = max(EW,SN)
    mid_x, mid_y = LL[0]+EW/2.0, LL[1]+SN/2.0  #center location

    extent = [mid_x-side/2.0, mid_x+side/2.0, mid_y-side/2.0, mid_y+side/2.0]   # map coordinates, meters

    ax.set_extent(extent, crs=crg)
    ax.add_image(request, zoom)
    ax.add_patch(mpatches.Circle(xy=[pt[0], pt[1]], radius=r_ortho, color='red', alpha=0.3, transform=crg, zorder=30))
    #ax.coastlines(resolution='10m', color='black', linewidth=1)


    # add a marker at center of the map
    plt.plot(mid_x, mid_y, marker='o',
             color='red', markersize=10,
             alpha=0.7, transform = crg)

    # add a marker for volcano
    plt.plot(pt[0], pt[1], marker='^',
             color='black', markersize=10,
             alpha=0.7, transform = ccrs.Geodetic())



    plt.draw()

if __name__ == '__main__':
    #main()
    main(extent=extent_augustine, pt=pt_augustine, zoom=12)
    plt.show()