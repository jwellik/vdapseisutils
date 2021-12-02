import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# example: draw circle with 45 degree radius around the North pole
lat = 51.4198101
lon = -0.950854653584
r = 45

extent_eyja      = [-22, -15, 63, 65]
extent_augustine = [-153.75, -153.25, 59.25, 59.5]
pt_eyja      = [-19.613333, 63.62]       # lat,lon
pt_augustine = [-153.43, 59.363]

# Define the projection used to display the circle:



def compute_radius(ortho, lat=lat, lon=lon, radius_degrees=r):
    phi1 = lat + radius_degrees if lat <= 0 else lat - radius_degrees
    _, y1 = ortho.transform_point(lon, phi1, ccrs.PlateCarree())
    return abs(y1)

def main(lat=lat, lon=lon, r=r):

    # Compute the required radius in projection native coordinates:
    proj = ccrs.Orthographic(central_longitude=lon, central_latitude=lat)
    r_ortho = compute_radius(proj, lat=lat, lon=lon, radius_degrees=r)

    # We can now compute the correct plot extents to have padding in degrees:
    pad_radius = compute_radius(proj, lat=lat, lon=lon, radius_degrees=r + 10/110)

    # define image properties
    width = 800
    height = 800
    dpi = 96
    resolution = '50m'

    # create figure
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    # Deliberately avoiding set_extent because it has some odd behaviour that causes
    # errors for this case. However, since we already know our extents in native
    # coordinates we can just use the lower-level set_xlim/set_ylim safely.
    ax.set_xlim([-pad_radius, pad_radius])
    ax.set_ylim([-pad_radius, pad_radius])
    ax.imshow(np.tile(np.array([[cfeature.COLORS['water'] * 255]], dtype=np.uint8), [2, 2, 1]), origin='upper', transform=ccrs.PlateCarree(), extent=[-180, 180, -180, 180])
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', resolution, edgecolor='black', facecolor=cfeature.COLORS['land']))
    #ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', resolution, edgecolor='black', facecolor='none'))
    #ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', resolution, edgecolor='none', facecolor=cfeature.COLORS['water']), alpha=0.5)
    #ax.add_feature(cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', resolution, edgecolor=cfeature.COLORS['water'], facecolor='none'))
    #ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', resolution, edgecolor='gray', facecolor='none'))
    ax.add_patch(mpatches.Circle(xy=[lon, lat], radius=r_ortho, color='red', alpha=0.3, transform=proj, zorder=30))
    fig.tight_layout()
    plt.savefig('CircleTest.png', dpi=dpi)
    plt.draw()

if __name__ == '__main__':
    #main()
    main(lat=pt_augustine[1], lon=pt_augustine[0], r=30/110)
    plt.show()