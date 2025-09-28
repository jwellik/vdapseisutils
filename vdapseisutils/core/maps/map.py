"""
Map class and terrain/hillshade utilities.

This module contains the Map class for creating geographic map visualizations
and all terrain/hillshade related functionality.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2025 September 28
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

from .defaults import (
    HEATMAP_DEFAULTS, TICK_DEFAULTS, AXES_DEFAULTS, GRID_DEFAULTS, default_volcano
)
from .utils import prep_catalog_data_mpl, choose_scale_bar_length
from vdapseisutils.utils.geoutils import backazimuth, radial_extent2map_extent


# deprecated - this won't be used anymore
class ShadedReliefESRI(cimgt.GoogleTiles):
    # shaded relief - produces terrain tile w pink hue and blue water
    def _image_url(self, tile):
        x, y, z = tile
        url = ('https://server.arcgisonline.com/ArcGIS/rest/services/' \
               'World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg').format(
            z=z, y=y, x=x)
        return url


def add_hillshade_pygmt(ax, extent=[-180, 180, -90, 90],
                        data_source="igpp",
                        resolution="auto",
                        topo=True, bath=False,
                        radiance=[315, 45],
                        vertical_exag=1.5,
                        cmap="Greys_r", alpha=0.8,
                        blend_mode="overlay",
                        elevation_weight=0.3,
                        hillshade_weight=0.7,
                        normalize_elevation=True,
                        cache_data=True,
                        **kwargs,
                        ):
    """
    Add enhanced hillshade and elevation data to a Cartopy GeoAxes using PyGMT.

    See PyGMT's load_earth_relief() for more details:
    - https://www.pygmt.org/latest/_modules/pygmt/datasets/earth_relief.html#load_earth_relief

    This function creates a more visually appealing terrain visualization by combining
    hillshade and elevation data with customizable blending options.

    Parameters:
    -----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The target axes to add the hillshade to
    extent : list
        Map extent [minlon, maxlon, minlat, maxlat]
    data_source : str
        PyGMT data source ("igpp", "srtm", "gebco", etc.)
    resolution : str
        Data resolution ("auto", "01d", "30s", "15s", etc.). If "auto", resolution
        is automatically selected based on map extent.
    topo : bool
        Include topographic (land) data
    bath : bool
        Include bathymetric (ocean) data
    radiance : list
        Hillshade lighting parameters [azimuth, elevation]
    vertical_exag : float
        Vertical exaggeration factor
    cmap : str
        Colormap for elevation data (default: "Greys_r" for black/white)
    alpha : float
        Overall transparency
    blend_mode : str
        Blending mode: "overlay", "multiply", "hillshade_only", "elevation_only"
    elevation_weight : float
        Weight for elevation data in blend (0-1)
    hillshade_weight : float
        Weight for hillshade data in blend (0-1)
    normalize_elevation : bool
        Normalize elevation data for better contrast
    cache_data : bool
        Cache downloaded data for reuse
    transform_data : str
        Data transformation method. Options:
        - "auto": Apply automatic rotation and flip (default)
        - "none": No transformation
        - "rotate_only": Only rotate 90° clockwise
        - "flip_only": Only flip left-to-right
        - "custom": Apply both rotation and flip

    Returns:
    --------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The modified axes
    """
    import numpy as np
    import pygmt
    import matplotlib.colors as mcolors
    from pathlib import Path
    import hashlib

    # Automatic resolution selection based on map extent
    def auto_resolution(extent):
        """
        Automatically select resolution based on map extent.

        Returns appropriate resolution string based on the size of the map area.
        """
        minlon, maxlon, minlat, maxlat = extent
        width_deg = abs(maxlon - minlon)
        height_deg = abs(maxlat - minlat)
        max_dimension = max(width_deg, height_deg)

        # Resolution selection based on map size
        if max_dimension > 180:  # Global or very large regions
            return "01d"
        elif max_dimension > 90:  # Continental scale
            return "30m"
        elif max_dimension > 45:  # Regional scale
            return "15m"
        elif max_dimension > 20:  # Large local areas
            return "03s"
        elif max_dimension > 10:  # Medium local areas
            return "01s"
        elif max_dimension > 5:  # Small local areas
            return "30s"
        elif max_dimension > 1:  # Very small local areas
            return "15s"
        else:  # Tiny areas
            return "03s"

    # Set resolution if auto-selection is requested
    if resolution == "auto":
        resolution = auto_resolution(extent)
        print(f"Auto-selected resolution: {resolution}")

    # Create cache directory
    if cache_data:
        cache_dir = Path.home() / ".vdapseisutils" / "hillshade_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create cache key from parameters
        cache_params = f"{extent}_{data_source}_{resolution}_{vertical_exag}_{radiance}"
        cache_key = hashlib.md5(cache_params.encode()).hexdigest()
        cache_file = cache_dir / f"{cache_key}.npz"

        # Try to load from cache
        if cache_file.exists():
            try:
                cached_data = np.load(cache_file)
                elevation_data = cached_data['elevation']
                hillshade_data = cached_data['hillshade']
                lats = cached_data['lats']
                lons = cached_data['lons']
                print(f"Loaded hillshade data from cache: {cache_file}")
            except:
                cache_file.unlink(missing_ok=True)
                cached_data = None
        else:
            cached_data = None
    else:
        cached_data = None

    # Download and process data if not cached
    if cached_data is None:
        try:
            # Download elevation data
            srtm = pygmt.datasets.load_earth_relief(
                region=extent,
                data_source=data_source,
                resolution=resolution,
                **kwargs,
            )

            # Apply vertical exaggeration
            elevation_data = srtm.data * vertical_exag

            # Create hillshade
            srtm_hs = pygmt.grdgradient(
                grid=srtm,
                radiance=radiance,
                normalize="t1"  # Normalize to [-1, 1]
            )
            hillshade_data = srtm_hs.data

            # Get coordinate arrays
            lats = srtm.lat.data
            lons = srtm.lon.data

            # Check if latitudes are in descending order (north to south) and flip data if needed
            # PyGMT sometimes returns data with latitudes in descending order, but matplotlib expects ascending
            if len(lats) > 1 and lats[0] > lats[-1]:
                print("Flipping data vertically to match matplotlib's expected coordinate system")
                elevation_data = np.flipud(elevation_data)
                hillshade_data = np.flipud(hillshade_data)
                lats = np.flipud(lats)

            # Check if longitudes need wrapping for regions crossing the 180/-180 meridian
            if len(lons) > 1 and (lons[0] < -170 and lons[-1] > 170):
                print("Detected region crossing 180/-180 meridian, adjusting longitude ordering")
                # For regions crossing the meridian, we might need to roll the data
                # This is a complex case that might need custom handling

            # Cache the data
            if cache_data:
                np.savez(cache_file,
                         elevation=elevation_data,
                         hillshade=hillshade_data,
                         lats=lats,
                         lons=lons)
                print(f"Cached hillshade data: {cache_file}")

        except Exception as e:
            print(f"Failed to load hillshade data: {e}")
            return ax

    # Apply land/ocean masking
    if not bath:
        hillshade_data[elevation_data <= 0] = 1.0  # White for ocean
        elevation_data[elevation_data <= 0] = 0
    if not topo:
        hillshade_data[elevation_data >= 0] = 1.0  # White for land
        elevation_data[elevation_data >= 0] = 0

    # Create custom cream/white colormap for better appearance
    def create_cream_colormap():
        """Create a custom colormap with cream background and black features."""
        colors = ['#FDFBF7', '#F5F1E8', '#E8E0D0', '#D4C8B8', '#C0B0A0',
                  '#A89888', '#908070', '#786858', '#605040', '#483828', '#302010']
        return mcolors.LinearSegmentedColormap.from_list('cream_terrain', colors)

    # Prepare data for plotting
    if blend_mode == "hillshade_only":
        # Pure hillshade - convert to grayscale
        final_data = (hillshade_data + 1) / 2  # Convert from [-1,1] to [0,1]
        plot_cmap = "Greys_r"

    elif blend_mode == "elevation_only":
        # Pure elevation with colormap
        final_data = elevation_data
        if cmap == "Greys_r":
            plot_cmap = create_cream_colormap()
        else:
            plot_cmap = cmap

    elif blend_mode == "multiply":
        # Multiply hillshade with elevation
        hillshade_norm = (hillshade_data + 1) / 2  # Convert to [0,1]
        if normalize_elevation:
            elev_norm = (elevation_data - elevation_data.min()) / (elevation_data.max() - elevation_data.min())
        else:
            elev_norm = elevation_data / elevation_data.max()
        final_data = hillshade_norm * elev_norm
        if cmap == "Greys_r":
            plot_cmap = create_cream_colormap()
        else:
            plot_cmap = cmap

    else:  # "overlay" mode (default)
        # Blend hillshade and elevation
        hillshade_norm = (hillshade_data + 1) / 2  # Convert to [0,1]

        if normalize_elevation:
            elev_norm = (elevation_data - elevation_data.min()) / (elevation_data.max() - elevation_data.min())
        else:
            elev_norm = elevation_data / elevation_data.max()

        # Create blended image
        final_data = hillshade_weight * hillshade_norm + elevation_weight * elev_norm
        if cmap == "Greys_r":
            plot_cmap = create_cream_colormap()
        else:
            plot_cmap = cmap

    plot_data = np.flipud(final_data)

    projection = ccrs.PlateCarree()
    ax.imshow(
        plot_data,
        extent=extent,
        transform=projection,
        cmap=plot_cmap,
        alpha=alpha,
        interpolation='bilinear'
    )

    return ax


class Map:
    """
    Map class that creates map axes without inheriting from plt.Figure
    This avoids conflicts when used with SubFigures
    """
    
    name = "map"

    def __init__(self, fig=None, origin=(default_volcano["lat"], default_volcano["lon"]), 
                 radial_extent_km=50.0, map_extent=None, **kwargs):
        
        # Create figure if none provided
        if fig is None:
            # Extract figure-specific kwargs
            fig_kwargs = {k: v for k, v in kwargs.items() 
                         if k in ['dpi', 'figsize']}
            fig = plt.figure(**fig_kwargs)
        
        # Store the figure reference
        self.figure = fig
        
        # Remove any matplotlib figure-specific kwargs that would cause issues
        plot_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['dpi', 'figsize']}
        
        # Create the principal axes for the map
        self.ax = fig.add_subplot(111, projection=ccrs.Mercator(), **plot_kwargs)
        
        # Set thicker spines
        for spine in self.ax.spines.values():
            spine.set_linewidth(AXES_DEFAULTS['spine_linewidth'])

        # Define Map Properties
        self.properties = dict()
        self.properties["origin"] = origin
        self.properties["radial_extent_km"] = radial_extent_km
        self.properties["map_extent"] = radial_extent2map_extent(origin[0], origin[1], radial_extent_km)
        if map_extent:  # overwrite origin, radial_extent_km if map_extent is set explicitly
            self.properties["origin"] = None
            self.properties["radial_extent_km"] = None
            self.properties["map_extent"] = map_extent

        # Set extent and add labels
        self.ax.set_extent(self.properties["map_extent"])

        # Draw Grid and labels using centralized styling
        glv = self.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                               linewidth=GRID_DEFAULTS['linewidth'], 
                               color=GRID_DEFAULTS['color'], 
                               alpha=GRID_DEFAULTS['alpha'])
        glv.top_labels = False
        glv.bottom_labels = True
        glv.left_labels = True
        glv.right_labels = False
        glv.xlines = GRID_DEFAULTS['xlines']
        glv.xlabel_style = GRID_DEFAULTS['xlabel_style']
        glv.ylabel_style = GRID_DEFAULTS['ylabel_style']

        # Apply centralized tick styling
        self.ax.tick_params(axis='both', 
                            labelcolor=TICK_DEFAULTS['labelcolor'],
                            labelsize=TICK_DEFAULTS['labelsize'],
                            color=TICK_DEFAULTS['tick_color'],
                            length=TICK_DEFAULTS['tick_size'],
                            width=TICK_DEFAULTS['tick_width'],
                            direction=TICK_DEFAULTS['tick_direction'],
                            pad=TICK_DEFAULTS['tick_pad'])

    def info(self):
        """Display information about the map."""
        print("::: MAP AXES :::")
        print(self.properties)
        print()

    def set_ticks(self, x_spacing=None, y_spacing=None, 
                 show_bottom=True, show_left=True, 
                 show_top=False, show_right=False):
        """
        Customize the gridliner ticks on the map.
        
        Parameters:
        -----------
        x_spacing : float, optional
            Spacing for x-axis (longitude) ticks in degrees
        y_spacing : float, optional  
            Spacing for y-axis (latitude) ticks in degrees
        show_bottom : bool, optional
            Show ticks on bottom axis (default: True)
        show_left : bool, optional
            Show ticks on left axis (default: True)
        show_top : bool, optional
            Show ticks on top axis (default: False)
        show_right : bool, optional
            Show ticks on right axis (default: False)
        """
        # Get the gridliner object
        glv = None
        for child in self.ax.get_children():
            if hasattr(child, 'xlocator'):  # This identifies the gridliner
                glv = child
                break
        
        if glv is not None:
            if x_spacing is not None:
                glv.xlocator = plt.MultipleLocator(x_spacing)
            if y_spacing is not None:
                glv.ylocator = plt.MultipleLocator(y_spacing)
            
            # Set which sides show labels
            glv.xlabels_bottom = show_bottom
            glv.ylabels_left = show_left
            glv.xlabels_top = show_top
            glv.ylabels_right = show_right

    def add_hillshade(self, source="PyGMT", data_source="igpp", resolution="auto", 
                     topo=True, bath=False, radiance=[315, 60], alpha=0.8,
                     blend_mode="overlay", elevation_weight=0.3, hillshade_weight=0.7,
                     cmap="Greys_r", vertical_exag=1.5, normalize_elevation=True,
                     cache_data=True, **kwargs):
        """Add hillshade to the map."""
        if source.lower() == "pygmt":
            try:
                self.ax = add_hillshade_pygmt(
                    self.ax, 
                    extent=self.properties["map_extent"], 
                    data_source=data_source,
                    resolution=resolution, 
                    topo=topo, 
                    bath=bath,
                    radiance=radiance, 
                    alpha=alpha,
                    blend_mode=blend_mode,
                    elevation_weight=elevation_weight,
                    hillshade_weight=hillshade_weight,
                    cmap=cmap,
                    vertical_exag=vertical_exag,
                    normalize_elevation=normalize_elevation,
                    cache_data=cache_data,
                    **kwargs,
                )
            except Exception as e:
                print(f"Failed to add hillshade via PyGMT: {e}")
                print("Continuing without hillshade...")
        else:
            print(f"Elevation & hillshade source '{source}' not available.")

    def add_scalebar(self, scale_length_km="auto", position='lower right', 
                     color='black', fontsize=10, pad=0.5, frameon=False):
        """
        Add a scale bar to the map.
        
        Creates a scale bar showing the distance scale of the map. The scale bar
        length can be automatically calculated based on the map extent or manually
        specified. The text label appears on top of the scale bar.
        """
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        import matplotlib.font_manager as fm

        # Get scalebar length in axes percentage
        map_lonL, map_lonR, map_midlat = self.properties["map_extent"][0:3]
        az, d = backazimuth((map_midlat, map_lonL), (map_midlat, map_lonR))
        map_width_km = d / 1000
        
        if scale_length_km == "auto":
            scale_length_km = choose_scale_bar_length(map_width_km, 0.25)
        
        scale_bar_length_ax = scale_length_km / map_width_km

        # Determine appropriate unit and label
        if scale_length_km < 1:
            # Convert to meters for small scales
            scale_length_m = int(scale_length_km * 1000)
            scale_label = f"{scale_length_m} m"
        else:
            # Use kilometers for larger scales
            if scale_length_km == int(scale_length_km):
                scale_label = f"{int(scale_length_km)} km"
            else:
                scale_label = f"{scale_length_km} km"

        # Add scale bar with text on top
        scalebar = AnchoredSizeBar(
            self.ax.transAxes,
            scale_bar_length_ax,
            scale_label,
            position,
            pad=pad,
            color=color,
            frameon=frameon,
            size_vertical=0.01,
            fontproperties=fm.FontProperties(size=fontsize),
            label_top=True  # This puts the text on top of the scale bar
        )

        self.ax.add_artist(scalebar)

    def plot(self, lat, lon, *args, transform=ccrs.Geodetic(), **kwargs):
        """Plot line data on the map."""
        self.ax.plot(lon, lat, *args, transform=transform, **kwargs)

    def scatter(self, lat, lon, size, color, transform=ccrs.Geodetic(), **kwargs):
        """Plot scatter data on the map."""
        self.ax.scatter(lon, lat, size, color, transform=transform, **kwargs)

    def plot_catalog(self, catalog, s="magnitude", c="time", color=None, cmap="viridis_r", alpha=0.5, transform=ccrs.Geodetic(), **kwargs):
        """
        Plot earthquake catalog on the map.
        
        Creates a scatter plot of earthquake events from an ObsPy Catalog object,
        with customizable size, color, and styling options.
        """
        catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")

        if s == "magnitude":
            s = catdata["size"]
        else:
            s = s

        # Handle both 'c' and 'color' parameters like matplotlib's scatter
        if color is not None:
            # 'color' parameter takes precedence over 'c'
            c = color
        elif c == "time":
            c = catdata["time"]
        else:
            c = c

        return self.ax.scatter(catdata["lon"], catdata["lat"], s=s, c=c, cmap=cmap, alpha=alpha, transform=transform, **kwargs)

    def plot_line(self, p1, p2, color="k", linewidth=1,
                  label=None, va='center', ha='center',
                  transform=ccrs.Geodetic(), **kwargs):
        """Plot a line between two points with optional labels."""
        label0 = label if label else ""
        label1 = label+"'" if label else ""
        self.ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=color, linewidth=linewidth,
                     transform=transform, **kwargs)
        self.ax.text(p1[1], p1[0], label0, ha=ha, va=va, color=color,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2),
                     transform=transform)
        self.ax.text(p2[1], p2[0], label1, ha=ha, va=va, color=color,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2),
                     transform=transform)

    def plot_inventory(self, inventory, marker_size=8, color='black', alpha=0.8, 
                      transform=ccrs.Geodetic(), **kwargs):
        """Plot seismic station inventory on the map."""
        try:
            station_lats = []
            station_lons = []
            
            for network in inventory:
                for station in network:
                    if hasattr(station, 'latitude') and hasattr(station, 'longitude'):
                        station_lats.append(station.latitude)
                        station_lons.append(station.longitude)
            
            if station_lats and station_lons:
                self.ax.scatter(station_lons, station_lats, 
                              s=marker_size, 
                              c=color, 
                              marker='v',
                              alpha=alpha,
                              transform=transform,
                              **kwargs)
            else:
                print("No valid station coordinates found in inventory")
                
        except Exception as e:
            print(f"Error plotting inventory: {e}")
            print("Continuing without inventory plot...")

    def plot_heatmap(self, *args, grid_size=HEATMAP_DEFAULTS['grid_size'], 
                     cmap=HEATMAP_DEFAULTS['cmap'], alpha=HEATMAP_DEFAULTS['alpha'], 
                     vmin=HEATMAP_DEFAULTS['vmin'], vmax=HEATMAP_DEFAULTS['vmax'], **kwargs):
        """Plot a heatmap on the map."""
        try:
            # Handle input data based on first argument type
            if len(args) == 1 and hasattr(args[0], 'events'):  # Single Catalog object
                catalog = args[0]
                catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")
                lat = catdata["lat"].values
                lon = catdata["lon"].values
                depth = catdata["depth"].values
            elif len(args) >= 2:  # lat, lon, [depth]
                lat = np.asarray(args[0])
                lon = np.asarray(args[1])
                depth = np.asarray(args[2]) if len(args) > 2 else None
            else:
                raise ValueError("Usage: plot_heatmap(catalog, ...) or plot_heatmap(lat, lon, [depth], ...)")
            
            if len(lat) == 0 or len(lon) == 0:
                print("Warning: Empty coordinate arrays")
                return None
            
            # Create regular grid
            lon_min, lon_max = np.min(lon), np.max(lon)
            lat_min, lat_max = np.min(lat), np.max(lat)
            
            grid_size_deg = grid_size
            if grid_size_deg < 0.001:
                grid_size_deg = 0.001
            
            data_range_lon = lon_max - lon_min
            data_range_lat = lat_max - lat_min
            min_grid_size = max(0.001, min(data_range_lon, data_range_lat) * 0.1)
            if grid_size_deg < min_grid_size:
                grid_size_deg = min_grid_size
            
            if lon_max <= lon_min or lat_max <= lat_min:
                print("Warning: Invalid coordinate ranges for heatmap")
                return None
            
            lon_pad = (lon_max - lon_min) * 0.1
            lat_pad = (lat_max - lat_min) * 0.1
            
            lon_grid = np.arange(lon_min - lon_pad, lon_max + lon_pad, grid_size_deg)
            lat_grid = np.arange(lat_min - lat_pad, lat_max + lat_pad, grid_size_deg)
            
            if len(lon_grid) < 2 or len(lat_grid) < 2:
                print("Warning: Grid too small for heatmap")
                return None
            
            H, xedges, yedges = np.histogram2d(lon, lat, bins=[lon_grid, lat_grid])
            
            if H.size == 0 or np.all(H == 0):
                print("Warning: No data points in the specified region")
                return None
            
            lon_centers = (xedges[:-1] + xedges[1:]) / 2
            lat_centers = (yedges[:-1] + yedges[1:]) / 2
            lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
            
            im = self.ax.pcolormesh(lon_mesh, lat_mesh, H.T, 
                                   cmap=cmap, alpha=alpha, 
                                   vmin=vmin, vmax=vmax,
                                   transform=ccrs.PlateCarree(),
                                   **kwargs)
            
            return im
            
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            print("Continuing without heatmap...")
            return None

    def add_terrain(self, zoom='auto', cache=False):
        """Add terrain background tiles from default source to the map."""
        self.add_arcgis_terrain(zoom=zoom, cache=cache)

    def add_arcgis_terrain(self, zoom='auto', style='terrain', cache=False):
        """Add world terrain background tiles from ArcGIS to the map."""
        from .map_tiles import add_arcgis_terrain
        
        add_arcgis_terrain(
            self.ax, 
            zoom=zoom, 
            style=style, 
            cache=cache, 
            radial_extent_km=self.properties.get("radial_extent_km")
        )

    def add_google_terrain(self, zoom='auto', cache=False, **kwargs):
        """Add Google terrain tiles to the map."""
        from .map_tiles import add_google_terrain
        
        add_google_terrain(
            self.ax, 
            zoom=zoom, 
            cache=cache, 
            radial_extent_km=self.properties.get("radial_extent_km"),
            **kwargs
        )

    def add_google_street(self, zoom='auto', cache=False, **kwargs):
        """Add Google street tiles to the map."""
        from .map_tiles import add_google_street
        
        add_google_street(
            self.ax, 
            zoom=zoom, 
            cache=cache, 
            radial_extent_km=self.properties.get("radial_extent_km"),
            **kwargs
        )

    def add_google_satellite(self, zoom='auto', cache=False, **kwargs):
        """Add Google satellite tiles to the map."""
        from .map_tiles import add_google_satellite
        
        add_google_satellite(
            self.ax, 
            zoom=zoom, 
            cache=cache, 
            radial_extent_km=self.properties.get("radial_extent_km"),
            **kwargs
        )

    def add_world_location_map(self, size=0.18, position='upper left', **kwargs):
        """Add a world location reference map as an inset to the main map."""
        # Calculate center coordinates from main map extent
        map_extent = self.properties["map_extent"]
        center_lon = (map_extent[0] + map_extent[1]) / 2  # Average of min and max longitude
        center_lat = (map_extent[2] + map_extent[3]) / 2  # Average of min and max latitude
        
        # Get main map center for marking
        if self.properties["origin"] is not None:
            main_map_center = self.properties["origin"]
        else:
            main_map_center = (center_lat, center_lon)
                        
        # Calculate the position for the inset axes relative to the main axes
        fig = self.ax.figure
        
        # Get the main axes position in figure coordinates
        main_ax_pos = self.ax.get_position()
        main_left = main_ax_pos.x0
        main_bottom = main_ax_pos.y0
        main_width = main_ax_pos.width
        main_height = main_ax_pos.height
        
        # Calculate position relative to main axes
        if position == 'upper left':
            left = main_left - (size/2) * main_width # + 0.02 * main_width
            bottom = main_bottom + main_height - (size/2) * main_height - 0.05 * main_height
        elif position == 'upper right':
            left = main_left + main_width - (size/2) - 0.02 * main_width
            bottom = main_bottom + main_height - (size/2) * main_height - 0.05 * main_height
        elif position == 'lower left':
            left = main_left - (size/2) * main_width # + 0.02 * main_width
            bottom = main_bottom - (size/2) * main_height + 0.02 * main_height
        elif position == 'lower right':
            left = main_left + main_width - (size/2) - 0.02 * main_width
            bottom = main_bottom - (size/2) * main_height + 0.02 * main_height
        else:  # lower left
            left = main_left - (size/2) * main_width + 0.02 * main_width
            bottom = main_bottom + main_height - (size/2) * main_height - 0.05 * main_height
        
        # Create GeoAxes directly with orthographic projection
        world_ax = fig.add_axes([left, bottom, size, size], 
                               projection=ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat))
        
        # Set global extent
        world_ax.set_global()
        
        # Add features with specified styling
        # Grey oceans
        world_ax.add_feature(cfeature.OCEAN, color='lightgrey', alpha=0.8)
        
        # White land
        world_ax.add_feature(cfeature.LAND, color='white', alpha=1.0)
        
        # Country borders
        world_ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Coastlines
        world_ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Add grid
        gl = world_ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, 
                               linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlines = True
        gl.ylines = True
        
        # Add square marker for main map center if provided
        if main_map_center is not None:
            main_lat, main_lon = main_map_center
            world_ax.plot(main_lon, main_lat, 's', color='black', markersize=5, 
                         transform=ccrs.Geodetic())
        
        # Remove axis labels and ticks
        world_ax.set_xticks([])
        world_ax.set_yticks([])
        
        # Make the plot circular by setting equal aspect ratio
        world_ax.set_aspect('equal')
        
        return world_ax


def _test_map():
    """Simple test to verify Map class works correctly."""
    try:
        # Test Map creation with default parameters
        map_obj = Map()
        print("✓ Map class created successfully")
        print(f"✓ Map extent: {map_obj.properties['map_extent']}")
        print(f"✓ Map origin: {map_obj.properties['origin']}")
        print(f"✓ Radial extent: {map_obj.properties['radial_extent_km']} km")
        
        # Test info method
        print("✓ Map info:")
        map_obj.info()
        
        # Test hillshade function exists (don't actually run it to avoid dependencies)
        print("✓ Hillshade function available")
        
        return True
        
    except Exception as e:
        print(f"✗ Map test failed: {e}")
        return False


if __name__ == "__main__":
    _test_map()
