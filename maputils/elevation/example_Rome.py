# https://pypi.org/project/elevation/
# and
# https://www.earthdatascience.org/tutorials/visualize-digital-elevation-model-contours-matplotlib/

def main():
    from osgeo import gdal
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import elevation

    # clip the SRTM1 30m DEM of Rome and save it to Rome-DEM.tif
    # bounds=(bottom, left, top, right)
    # elevation.clip(bounds=(12.35, 41.8, 12.65, 42), output='/Users/jwellik/Downloads/Rome-DEM.tif')
    # clean up stale temporary files and fix the cache in the event of a server error
    elevation.clean()

    filename = '/Users/jwellik/Downloads/Rome-DEM.tif'
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
    plt.title("Elevation Contours of Rome 30m")
    cbar = plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Plot our data with Matplotlib's 'contourf'
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    plt.contourf(data_array, cmap="viridis",
                 levels=list(range(0, 5000, 500)))
    plt.title("Elevation Contours of Rome 30m")
    cbar = plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    main()
