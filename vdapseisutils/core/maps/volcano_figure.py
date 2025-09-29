"""
VolcanoFigure class for creating comprehensive volcano monitoring plots.

This module contains the VolcanoFigure class that combines Map, CrossSection,
and TimeSeries visualizations into a single comprehensive figure.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2025 September 28
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from .defaults import default_volcano, TITLE_DEFAULTS, SUBTITLE_DEFAULTS
from .map import Map
from .cross_section import CrossSection
from .time_series import TimeSeries
from .utils import prep_catalog_data_mpl
from vdapseisutils.utils.geoutils import radial_extent2map_extent


class VolcanoFigure(plt.Figure):
    """
    Comprehensive volcano monitoring figure combining multiple visualization types.
    
    This class creates a figure with a map, two cross-sections, a time-series plot,
    and a legend area, all properly positioned and styled for volcano monitoring
    applications.
    """

    name = "volcano-figure"

    def __init__(self, *args, origin=(default_volcano["lat"], default_volcano["lon"]), radial_extent_km=50.0,
                 map_extent=None,  # [minlon, maxlon, minlat, maxlat] ([L, R, B, T])
                 ts_axis_type="depth",
                 depth_extent=(-50, 4),
                 xs1=None,
                 xs2=None,
                 figsize=(10, 10),  # Allow custom figure size
                 hillshade=False,
                 dpi=300,
                 **kwargs):

        # Set default DPI in kwargs if not already specified
        if 'dpi' not in kwargs:
            kwargs['dpi'] = dpi

        # Initialize the Figure object properly
        figscale = 8
        self.figscale = figscale
        
        # Create the main figure first
        super().__init__(figsize=figsize, **kwargs)

        # Define Map Properties
        self.properties = dict()
        self.properties["origin"] = origin
        self.properties["radial_extent_km"] = radial_extent_km
        self.properties["map_extent"] = radial_extent2map_extent(origin[0], origin[1], radial_extent_km)
        self.properties["depth_extent"] = depth_extent
        if map_extent:  # overwrite origin, radial_extent_km if map_extent is set explicitly
            self.properties["origin"] = None
            self.properties["radial_extent_km"] = None
            self.properties["map_extent"] = map_extent
        self.properties["ts_axis_type"] = ts_axis_type

        # Parse and prepare Cross-Section arguments
        xs1 = dict({}) if xs1 is None else xs1
        xs2 = dict({}) if xs2 is None else xs2
        xs1_defaults = dict({
            'origin': self.properties["origin"],
            'azimuth': 270,  # East-West
            'radius_km': self.properties["radial_extent_km"],
            'depth_extent': self.properties["depth_extent"],
            'label': 'A'
        })
        xs2_defaults = dict({
            'origin': self.properties["origin"],
            'azimuth': 0,  # North-South
            'radius_km': self.properties["radial_extent_km"],
            'depth_extent': self.properties["depth_extent"],
            'label': 'B'
        })

        ## Create subfigures
        spec = self.add_gridspec(1, 1)

        # subfigure - Map
        self.fig_m = self.add_subfigure(spec[0:1, 0:1])
        self.map_obj = Map(fig=self.fig_m, origin=origin, radial_extent_km=radial_extent_km)
        if hillshade:
            self.map_obj.add_hillshade()
        self.map_obj.add_scalebar()
        lbwh = np.array([0.9, 4.0, 3.0, 3.0])
        self.map_obj.ax.set_position(lbwh / figscale)

        # subfigure - Top Cross-Section
        self.fig_xs1 = self.add_subfigure(spec[0:1, 0:1])
        self.xs1_obj = CrossSection(fig=self.fig_xs1, **{**xs1_defaults, **xs1})  # overwrite defaults w user input
        lbwh = np.array([4.0, 5.8, 3.0, 1.2])
        self.xs1_obj.ax.set_position(lbwh / figscale)

        # subfigure - Bottom Cross-Section
        self.fig_xs2 = self.add_subfigure(spec[0:1, 0:1])
        self.xs2_obj = CrossSection(fig=self.fig_xs2, ** {**xs2_defaults, **xs2})  # overwrite defaults w user input
        lbwh = np.array([4.0, 4.0, 3.0, 1.2])
        self.xs2_obj.ax.set_position(lbwh / figscale)

        # subfigure - TimeSeries
        self.fig_ts = self.add_subfigure(spec[0:1, 0:1])
        self.ts_obj = TimeSeries(fig=self.fig_ts, depth_extent=self.properties["depth_extent"], axis_type=self.properties["ts_axis_type"])
        lbwh = np.array([0.9, 1.2, 6.1, 2.1])
        self.ts_obj.ax.set_position(lbwh / figscale)

        # subfigure - Legend
        self.fig_leg = self.add_subfigure(spec[0:1, 0:1])
        axL = self.fig_leg.add_subplot(111)
        lbwh = np.array([5.5, 1.2, 1.9, 2.1])
        axL.set_position(lbwh / figscale)
        axL.set_visible(False)

        # Plot Cross-Section lines to Map
        self.map_obj.plot_line(self.xs1_obj.properties["points"][0], self.xs1_obj.properties["points"][1], label=self.xs1_obj.properties["label"])
        self.map_obj.plot_line(self.xs2_obj.properties["points"][0], self.xs2_obj.properties["points"][1], label=self.xs2_obj.properties["label"])

    def info(self):
        """Display information about the volcano figure."""
        print("::: VOLCANO FIGURE :::")
        print(self.properties)
        print()

    def add_hillshade(self, *args, **kwargs):
        """Add hillshade to the map."""
        self.map_obj.add_hillshade(*args, **kwargs)

    def add_ocean(self, *args, **kwargs):
        """Add ocean features to the map."""
        self.map_obj.add_ocean(*args, **kwargs)

    def add_coastline(self, *args, **kwargs):
        """Add coastline features to the map."""
        self.map_obj.add_coastline(*args, **kwargs)

    def add_terrain(self, *args, **kwargs):
        """Add terrain tiles from default source to map."""
        self.add_arcgis_terrain(*args, **kwargs)

    def add_arcgis_terrain(self, *args, **kwargs):
        """Add ArcGIS terrain tiles to the map."""
        self.map_obj.add_arcgis_terrain(*args, **kwargs)

    def add_google_terrain(self, *args, **kwargs):
        """Add Google terrain tiles to the map."""
        self.map_obj.add_google_terrain(*args, **kwargs)

    def add_google_street(self, *args, **kwargs):
        """Add Google street tiles to the map."""
        self.map_obj.add_google_street(*args, **kwargs)

    def add_google_satellite(self, *args, **kwargs):
        """Add Google satellite tiles to the map."""
        self.map_obj.add_google_satellite(*args, **kwargs)

    def plot(self, lat=None, lon=None, z=None, z_dir="depth", z_unit="m", transform=ccrs.Geodetic(), **kwargs):
        """
        Plot data on all relevant subplots.
        
        Parameters:
        -----------
        lat : array-like, optional
            Latitude coordinates
        lon : array-like, optional
            Longitude coordinates
        z : array-like, optional
            Depth/elevation values
        z_dir : str, optional
            Direction of z values: "depth" or "elev"
        z_unit : str, optional
            Units of z values: "m" or "km"
        transform : cartopy.crs.Projection, optional
            Coordinate reference system for map data
        **kwargs
            Additional plotting arguments
        """
        self.map_obj.plot(lat, lon, transform=transform, **kwargs)
        self.xs1_obj.plot(lat, lon, z=z, z_dir=z_dir, z_unit=z_unit, **kwargs)
        self.xs2_obj.plot(lat, lon, z=z, z_dir=z_dir, z_unit=z_unit, **kwargs)

    def scatter(self, lat=[], lon=[], x=[], time=[], y=[], transform=ccrs.Geodetic(), **kwargs):
        """
        Create scatter plots on all relevant subplots.
        
        Parameters:
        -----------
        lat : array-like
            Latitude coordinates
        lon : array-like
            Longitude coordinates
        x : array-like
            X coordinates for cross-sections
        time : array-like
            Time values for time-series
        y : array-like
            Y values for time-series
        transform : cartopy.crs.Projection, optional
            Coordinate reference system for map data
        **kwargs
            Additional scatter arguments
        """
        self.map_obj.scatter(lat, lon, transform=transform, **kwargs)
        self.xs1_obj.scatter(lat, lon, **kwargs)
        self.xs2_obj.scatter(lat, lon, **kwargs)
        self.ts_obj.scatter(time, y, **kwargs)

    def plot_catalog(self, *args, transform=ccrs.Geodetic(), **kwargs):
        """
        Plot earthquake catalog on all subplots (map, cross-sections, and time-series).
        
        Creates scatter plots of earthquake events from an ObsPy Catalog object
        on the map, both cross-sections, and time-series plot with consistent
        styling across all subplots.
        
        Parameters:
        -----------
        catalog : obspy.core.event.Catalog
            ObsPy Catalog object containing earthquake events
        s : str or array-like, optional
            Size parameter for scatter points. If "magnitude" (default), 
            point sizes are scaled by earthquake magnitude.
        c : str or array-like, optional
            Color parameter for scatter points. If "time" (default), 
            points are colored by event time.
        color : str or array-like, optional
            Alternative to 'c' parameter for specifying color. If provided,
            takes precedence over 'c'.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap for mapping numeric values to colors (default: "viridis_r")
        alpha : float, optional
            Transparency of scatter points (default: 0.5)
        transform : cartopy.crs.Projection, optional
            Coordinate reference system for the map data (default: ccrs.Geodetic())
        **kwargs
            Additional keyword arguments passed to matplotlib's scatter function
            
        Returns:
        --------
        tuple
            Tuple containing the scatter plot collections from all subplots
            
        Notes:
        ------
        - All subplots use the same styling parameters for consistency
        - The map shows events in geographic coordinates
        - Cross-sections show events projected onto their respective lines
        - Time-series shows events plotted against time
        - Both 'c' and 'color' parameters are supported for compatibility
          with matplotlib's scatter function
        """
        map_scatter = self.map_obj.plot_catalog(*args, transform=transform, **kwargs)
        xs1_scatter = self.xs1_obj.plot_catalog(*args, **kwargs)
        xs2_scatter = self.xs2_obj.plot_catalog(*args, **kwargs)
        ts_scatter = self.ts_obj.plot_catalog(*args, **kwargs)
        return map_scatter, xs1_scatter, xs2_scatter, ts_scatter

    def plot_inventory(self, inventory, s=8, c='black', alpha=0.8, 
                      transform=ccrs.Geodetic(), cross_section_s=6, 
                      **kwargs):
        """
        Plot ObsPy inventory stations on the map and cross-sections.
        
        Parameters:
        -----------
        inventory : obspy.core.inventory.Inventory
            ObsPy Inventory object containing station information
        s : int, optional
            Size of markers on the map (default: 8)
        c : str, optional
            Color of the markers (default: 'black')
        alpha : float, optional
            Transparency of markers (default: 0.8)
        transform : cartopy.crs.Projection, optional
            Coordinate reference system for map data (default: ccrs.Geodetic())
        cross_section_s : int, optional
            Size of markers on cross-sections (default: 6)
        **kwargs
            Additional plotting arguments
        """
        # Plot inventory on the main map
        self.map_obj.plot_inventory(inventory, s=s, c=c, 
                                 alpha=alpha, transform=transform, **kwargs)
        
        # Plot inventory on both cross-sections
        self.xs1_obj.plot_inventory(inventory, s=cross_section_s, 
                                   c=c, alpha=alpha, **kwargs)
        self.xs2_obj.plot_inventory(inventory, s=cross_section_s, 
                                   c=c, alpha=alpha, **kwargs)

    def plot_volcano(self, lat, lon, elev=0, transform=ccrs.Geodetic(), **kwargs):
        """
        Plot volcano location on all subplots (map and cross-sections).
        
        Parameters:
        -----------
        lat : float
            Latitude of the volcano
        lon : float
            Longitude of the volcano
        elev : float, optional
            Elevation of the volcano in meters (default: 0)
        transform : cartopy.crs.Projection, optional
            Coordinate reference system for map data (default: ccrs.Geodetic())
        **kwargs
            Additional plotting arguments to override defaults
        """
        # Plot on the main map
        self.map_obj.plot_volcano(lat, lon, elev, transform=transform, **kwargs)
        
        # Plot on both cross-sections  
        self.xs1_obj.plot_volcano(lat, lon, elev, **kwargs)
        self.xs2_obj.plot_volcano(lat, lon, elev, **kwargs)

    def plot_peak(self, lat, lon, elev=0, transform=ccrs.Geodetic(), **kwargs):
        """
        Plot peak location on all subplots (map and cross-sections).
        
        Parameters:
        -----------
        lat : float
            Latitude of the peak
        lon : float
            Longitude of the peak
        elev : float, optional
            Elevation of the peak in meters (default: 0)
        transform : cartopy.crs.Projection, optional
            Coordinate reference system for map data (default: ccrs.Geodetic())
        **kwargs
            Additional plotting arguments to override defaults
        """
        # Plot on the main map
        self.map_obj.plot_peak(lat, lon, elev, transform=transform, **kwargs)
        
        # Plot on both cross-sections
        self.xs1_obj.plot_peak(lat, lon, elev, **kwargs)
        self.xs2_obj.plot_peak(lat, lon, elev, **kwargs)

    def plot_heatmap(self, *args, **kwargs):
        """
        Plot a heatmap on the map and cross-sections.
        
        Parameters:
        -----------
        *args
            Arguments for heatmap data (catalog or lat, lon, [depth])
        **kwargs
            Additional heatmap arguments
        """
        # Plot heatmap on the main map using the stored map object
        self.map_obj.plot_heatmap(*args, **kwargs)
        
        # Plot heatmap on both cross-sections
        self.xs1_obj.plot_heatmap(*args, **kwargs)
        self.xs2_obj.plot_heatmap(*args, **kwargs)

    def title(self, t, **kwargs):
        """
        Add a title to the figure.
        
        Parameters:
        -----------
        t : str
            Title text
        **kwargs
            Additional text arguments to override defaults. Available options:
            - fontsize: str or float (default: 'large')
            - fontweight: str (default: 'bold')
            - color: str (default: 'black')
            - ha: str (default: 'center')
            - va: str (default: 'top')
            - pad: float (default: 20)
            - y: float (default: 0.95)
            - x: float (default: 0.5)
        """
        # Merge defaults with user kwargs
        title_params = {**TITLE_DEFAULTS, **kwargs}
        
        # Always use text for VolcanoFigure to avoid suptitle issues
        # Calculate y position if not provided
        if title_params.get('y') is None:
            # Use smart spacing calculation
            if title_params.get('auto_spacing', True):
                # Get the main map axes position for smart spacing
                map_ax = self.map_obj.ax
                ax_pos = map_ax.get_position()
                ax_top = ax_pos.y1
                available_space = 1.0 - ax_top
                title_height = 0.05
                if available_space >= title_height:
                    title_y = ax_top + available_space - title_height/2
                else:
                    title_y = ax_top + 0.01
                title_params['y'] = title_y
            else:
                title_params['y'] = 0.95
        
        # Use text with transFigure for reliable positioning
        self.text(x=title_params['x'], y=title_params['y'], s=t,
                 fontsize=title_params['fontsize'],
                 fontweight=title_params['fontweight'],
                 color=title_params['color'],
                 ha=title_params['ha'],
                 va=title_params['va'],
                 transform=self.transFigure)
        
        # Store title info for subtitle positioning
        self._title_text = t
        self._title_y = title_params['y']

    def subtitle(self, t, **kwargs):
        """
        Add a subtitle to the figure.
        
        Parameters:
        -----------
        t : str
            Subtitle text
        **kwargs
            Additional text arguments to override defaults. Available options:
            - fontsize: str or float (default: 'medium')
            - fontweight: str (default: 'normal')
            - color: str (default: 'black')
            - ha: str (default: 'center')
            - va: str (default: 'top')
            - pad: float (default: 10)
            - y: float (default: 0.90)
            - x: float (default: 0.5)
        """
        # Merge defaults with user kwargs
        subtitle_params = {**SUBTITLE_DEFAULTS, **kwargs}
        
        # Calculate y position if not provided
        if subtitle_params.get('y') is None:
            # Use smart spacing calculation
            if subtitle_params.get('auto_spacing', True):
                # Check if there's already a title to position relative to it
                existing_title_y = getattr(self, '_title_y', None)
                if existing_title_y is not None:
                    # Position subtitle below existing title
                    subtitle_height = 0.03
                    min_padding = 0.02
                    subtitle_y = existing_title_y - subtitle_height/2 - min_padding
                else:
                    # No existing title, position relative to axes
                    map_ax = self.map_obj.ax
                    ax_pos = map_ax.get_position()
                    ax_top = ax_pos.y1
                    available_space = 1.0 - ax_top
                    subtitle_height = 0.03
                    if available_space >= subtitle_height:
                        subtitle_y = ax_top + available_space - subtitle_height/2
                    else:
                        subtitle_y = ax_top - 0.01
                
                subtitle_params['y'] = subtitle_y
            else:
                subtitle_params['y'] = 0.90
        
        # Use text with transFigure for reliable positioning
        self.text(x=subtitle_params['x'], y=subtitle_params['y'], s=t,
                 fontsize=subtitle_params['fontsize'],
                 fontweight=subtitle_params['fontweight'],
                 color=subtitle_params['color'],
                 ha=subtitle_params['ha'],
                 va=subtitle_params['va'],
                 transform=self.transFigure)

    def set_titles(self, title_text=None, subtitle_text=None, **kwargs):
        """
        Add both title and subtitle to the volcano figure in one call.
        
        Parameters:
        -----------
        title_text : str, optional
            Title text to display
        subtitle_text : str, optional
            Subtitle text to display
        **kwargs
            Additional text arguments. Use 'title_' prefix for title-specific
            parameters and 'subtitle_' prefix for subtitle-specific parameters.
            For example: title_fontsize='x-large', subtitle_color='gray'
        """
        # Separate title and subtitle kwargs
        title_kwargs = {k.replace('title_', ''): v for k, v in kwargs.items() 
                       if k.startswith('title_')}
        subtitle_kwargs = {k.replace('subtitle_', ''): v for k, v in kwargs.items() 
                          if k.startswith('subtitle_')}
        
        # Add title if provided
        if title_text:
            self.title(title_text, **title_kwargs)
        
        # Add subtitle if provided
        if subtitle_text:
            self.subtitle(subtitle_text, **subtitle_kwargs)
        
        return self  # Enable method chaining

    def text(self, x, y, s, ha='center', va='center', **kwargs):
        """
        Add text to the figure.
        
        Parameters:
        -----------
        x : float
            X position in figure coordinates
        y : float
            Y position in figure coordinates
        s : str
            Text content
        ha : str, optional
            Horizontal alignment (default: 'center')
        va : str, optional
            Vertical alignment (default: 'center')
        **kwargs
            Additional text arguments
        """
        super().text(x, y, s, ha=ha, va=va, **kwargs)

    def reftext(self, x=0.025, y=0.025, s="", color="grey", ha="left", va="center", **kwargs):
        """
        Add reference text (e.g., citation) to the figure.
        
        Parameters:
        -----------
        x : float, optional
            X position in figure coordinates (default: 0.025)
        y : float, optional
            Y position in figure coordinates (default: 0.025)
        s : str, optional
            Reference text (default: "")
        color : str, optional
            Text color (default: "grey")
        ha : str, optional
            Horizontal alignment (default: "left")
        va : str, optional
            Vertical alignment (default: "center")
        **kwargs
            Additional text arguments
        """
        super().text(x, y, s, color=color, ha=ha, va=va, **kwargs)

    def catalog_subtitle(self, catalog):
        """
        Add an automatic subtitle based on catalog statistics.
        
        Parameters:
        -----------
        catalog : obspy.core.event.Catalog
            ObsPy Catalog object to analyze
        """
        n = len(catalog)
        magnitudes = [event.magnitudes[0].mag for event in catalog if event.magnitudes]
        mmin = min(magnitudes)
        mmax = max(magnitudes)
        self.subtitle("{} Earthquakes | M{:2.1f}:M{:2.1f}".format(n, mmin, mmax))

    def magnitude_legend(self, cat):
        """
        Add a magnitude legend to the figure.
        
        Parameters:
        -----------
        cat : obspy.core.event.Catalog
            ObsPy Catalog object for legend scaling
        """
        catdf = prep_catalog_data_mpl(cat)
        scatter = self.fig_leg.axes[0].scatter(catdf["lat"], catdf["lon"], s=catdf["size"], color="w", edgecolors="k")
        kw = dict(prop="sizes", num=3, fmt="M{x:1.0f}",
                  func=lambda s: np.sqrt(s / .3) / 3)
        legend = self.fig_leg.axes[0].legend(*scatter.legend_elements(**kw),
                                        loc="upper center", bbox_to_anchor=[0.0, 0.0, 1, 1],
                                        title="Magnitude", frameon=False)
        self.fig_leg.add_artist(legend)
        self.fig_leg.axes[0].set_visible(False)


def _test_volcano_figure():
    """Simple test to verify VolcanoFigure class works correctly."""
    try:
        # Test VolcanoFigure creation with default parameters
        vf = VolcanoFigure(figsize=(8, 8))
        
        print("✓ VolcanoFigure class created successfully")
        print(f"✓ Origin: {vf.properties['origin']}")
        print(f"✓ Radial extent: {vf.properties['radial_extent_km']} km")
        print(f"✓ Time-series axis type: {vf.properties['ts_axis_type']}")
        
        # Test that all components are created
        print(f"✓ Map object: {vf.map_obj.name}")
        print(f"✓ Cross-section 1: {vf.xs1_obj.name} (label: {vf.xs1_obj.properties['label']})")
        print(f"✓ Cross-section 2: {vf.xs2_obj.name} (label: {vf.xs2_obj.properties['label']})")
        print(f"✓ Time-series: {vf.ts_obj.name}")
        
        # Test info method
        print("✓ VolcanoFigure info:")
        vf.info()
        
        return True
        
    except Exception as e:
        print(f"✗ VolcanoFigure test failed: {e}")
        return False


if __name__ == "__main__":
    _test_volcano_figure()
