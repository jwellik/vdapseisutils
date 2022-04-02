# https://www.earthdatascience.org/tutorials/visualize-digital-elevation-model-contours-matplotlib/
# Get data from NASA SRTM data

TMP_FOLDER = "/Users/jwellik/Downloads"


def main():
    import os
    from osgeo import gdal
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import elevation

    from maputils.utils.utils import radial_extent2bounds_bltr

    # $ eio clip -o Shasta-30m-DEM.tif --bounds -122.6 41.15 -121.9 41.6
    # os.system('eio clip -o Shasta-30m-DEM.tif --bounds -122.6 41.15 -121.9 41.6')  # bottom, left, top, right

    bounds = radial_extent2bounds_bltr(41.3, -122.3, 50)
    b, l, t, r = bounds
    print(l, b, r, t, sep=' ')
    elevation.clip(bounds=(l, b, r, t), output=os.path.join(TMP_FOLDER, "tmp-30m-DEM.tif"))
    # elevation.clip(bounds=(-122.6, 41.15, -121.9, 41.6), output=os.path.join(TMP_FOLDER, "tmp-30m-DEM.tif"))
    elevation.clean()

    # filename = "Shasta-30m-DEM.tif"
    filename = os.path.join(TMP_FOLDER, "tmp-30m-DEM.tif")
    gdal_data = gdal.Open(filename)
    gdal_band = gdal_data.GetRasterBand(1)
    nodataval = gdal_band.GetNoDataValue()

    # convert to a numpy array
    data_array = gdal_data.ReadAsArray().astype(float)
    data_array

    # replace missing values if necessary
    if np.any(data_array == nodataval):
        data_array[data_array == nodataval] = np.nan

    # Plot out data with Matplotlib's 'contour'
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    plt.contour(data_array, cmap="viridis",
                levels=list(range(0, 5000, 100)))
    plt.title("Elevation Contours of Mt. Shasta")
    cbar = plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Plot our data with Matplotlib's 'contourf'
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    plt.contourf(data_array, cmap="viridis",
                 levels=list(range(0, 5000, 500)))
    plt.title("Elevation Contours of Mt. Shasta")
    cbar = plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    main()
