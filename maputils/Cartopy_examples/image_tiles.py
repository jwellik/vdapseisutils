# https://scitools.org.uk/cartopy/docs/latest/gallery/web_services/image_tiles.html

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

extent_eyja      = [-22, -15, 63, 65]
extent_augustine = [-153.75, -153.25, 59.25, 59.5]
pt_eyja      = [-19.613333, 63.62]       # lat,lon
pt_augustine = [-153.43, 59.363]
r_degrees=10/110

def compute_radius(ortho, lat=0, lon=0, radius_degrees=1):
    phi1 = lat + radius_degrees if lat <= 0 else lat - radius_degrees
    _, y1 = ortho.transform_point(lon, phi1, ccrs.PlateCarree())
    return abs(y1)

from cartopy.io.img_tiles import Stamen

extent_original  = [-90, -73, 22, 34]
extent_eyja      = [-22, -15, 63, 65]
extent_augustine = [-153.75, -153.25, 59.25, 59.5]
pt_eyja      = [-19.613333, 63.62]       # lat,lon
pt_augustine = [-153.43, 59.363]

def main(extent=extent_original, lat=0, lon=0, r=10, zoom=6):
    tiler = Stamen('terrain-background', desired_tile_form="L")
    mercator = tiler.crs

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=mercator)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_image(tiler, zoom, cmap='Greys_r')
    #ax.coastlines('10m')
    plt.draw()


if __name__ == '__main__':
    #main()
    main(extent=extent_augustine, lat=pt_augustine[0], lon=pt_augustine[1], r=10, zoom=12)
    #main(extent=extent_eyja, zoom=8)
    plt.show()
