#https://stackoverflow.com/questions/66400300/plot-a-circle-on-a-cartopy-mercator-projection

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import cartopy.io.img_tiles as cimgt
import matplotlib.patches as mpatches

extent_eyja      = [-22, -15, 63, 65]
extent_augustine = [-153.75, -153.25, 59.25, 59.5]
pt_eyja      = [-19.613333, 63.62]
pt_augustine = [-153.43, 59.363]         # lat, lon
r_degrees=10/110

#def main(extent=(-121.8,-122.55,37.25,37.85), pt=[-122.4015173428571, 37.78774634285715], rad_deg=(0.021709041989311614 + 0.005):
def main(extent=(-121.8, -122.55, 37.25, 37.85), pt=[-122.4015173428571, 37.78774634285715], rad_deg=50/110, rad_km=50):

    tiles = cimgt.Stamen('terrain-background')
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(1, 1, 1, projection=tiles.crs)

    ax.set_extent(extent)

    ax.add_image(tiles, 11)

    #ax.add_patch(mpatches.Circle(xy=pt, radius = 0.021709041989311614 + 0.005, alpha=0.3, zorder=30, transform=ccrs.PlateCarree()))
    ax.tissot(rad_km=rad_km, lons=[pt[0]], lats=[pt[1]], alpha=0.3)
    print(rad_deg)

    plt.draw()

if __name__ == '__main__':
    main(rad_deg=5/110)
    main(extent=extent_augustine, pt=pt_augustine, rad_km=20)
    plt.show()
