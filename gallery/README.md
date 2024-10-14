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
<<<<<<< HEAD
<img src="https://github.com/jwellik/vdapseisutils/blob/main/gallery/output/VolcanoMap_Hood_earthquakes.png" width=600 alt="VolcanoMap" />

=======
<img src="https://github.com/jwellik/vdapseisutils/blob/main/gallery/output/Mapping_tutorial/VolcanoFigure_Hood.png" width=600 alt="VolcanoFigure (Hood)" /><br>
A VolcanoFigure with mostly default settings.
<br><br>
## Maps, Cross Sections, and Time Series
Maps, cross sections, and time series plots can also be created individually. The methods also return MatPlotLib figures. Here are example outputs:<br><br>
<img src="https://github.com/jwellik/vdapseisutils/blob/main/gallery/output/Mapping_tutorial/Bathymetry_BanuaWuhu_regional.png" width=600 alt="Map (Banua Wuhu)" /><br>
Just a map, with a location map added in the top right corner. (Banua Wuhu, Indonesia)
<br><br>
<img src="https://github.com/jwellik/vdapseisutils/blob/main/gallery/output/Mapping_tutorial/CrossSection_USvolcs.png" width=600 alt="CrossSection" /><br>
Five CrossSection figures from US Volcanoes.
<br><br>
<img src="https://github.com/jwellik/vdapseisutils/blob/main/gallery/output/Mapping_tutorial/TimeSeries_MSH1980.png" width=600 alt="TimeSeries" /><br>
TimeSeries figure of earthquakes prior to and after the 1980 eruption of Mount St Helens.
<br><br>
>>>>>>> 52208d8061bfd26bf129975691005c6876f5b36e
## Helicorder
SwarmMPL (vdapseisutils.core.swarmmpl) provides MatPlotLib based codes for plotting helicorders and "clipboards" (waveforms, spectrograms, or both). The Helicorder plotting routines borrow heavily from the 'dayplot' mode available to ObsPy Stream objects. SwarmMPL's Helicorder is more flexible. Look at [Helicorder_Goma_earthquakes.py](https://github.com/jwellik/vdapseisutils/blob/main/gallery/Helicorder_Goma_earthquakes.py) for examples of the flexible usage.
```
>>> from vdapseisutils.gallery import Helicorder_Goma_earthquakes
>>> Helicorder_Goma_earthquakes.main()
```
<img src="https://github.com/jwellik/vdapseisutils/blob/main/gallery/output/Helicorder_Goma_earthquakes.png" width=600 alt="Helicorder" />
