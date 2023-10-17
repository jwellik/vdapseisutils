## VolcanoMap
VolcanoMap creates a map and cross sections. It is designed to make square maps around a volcano for the purpose of plotting earthquake location data. Basemaps are currently not working. I am troubleshooting a recent issue with Cartopy and PyGMT. Look at [VolcanoMap_Hood_earthquakes.py](https://github.com/jwellik/vdapseisutils/blob/main/gallery/VolcanoMap_Hood_earthquakes.py) for full functionality. The method returns a MatPlotLib Figure object so that the user can further customize the plot.
```
$ cd /home/jwellik/PYTHON
$ mv vdapseisutils-main vdapseisutils
$ cd /home/jwellik/PYTHON/vdapseisutils/gallery
$ conda activate vseis399
$ python
>>> import sys
>>> sys.path.append("/home/jwellik/PYTHON")  # Add all codes in the repository to your path
>>> from vdapseisutils.gallery import VolcanoMap_Hood_earthquakes
>>> VolcanoMap_Hood_earthquakes.main()  # Make sure your terminal has graphics forwarding
```
<img src="https://github.com/jwellik/vdapseisutils/blob/main/gallery/output/VolcanoMap_Hood_earthquakes.png" width=825 alt="VolcanoMap" />

## Helicorder
```
>>> from vdapseisutils.gallery import Helicorder_Goma_earthquakes
>>> Helicorder_Goma_earthquakes.main()
```
<img src="https://github.com/jwellik/vdapseisutils/blob/main/gallery/output/Helicorder_Goma_earthquakes.png" width=825 alt="Helicorder" />
