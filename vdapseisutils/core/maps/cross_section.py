"""
CrossSection class for creating vertical cross-section plots.

This module contains the CrossSection class for creating cross-sectional views
of seismic data projected onto vertical planes.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2025 September 28
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from .defaults import (
    HEATMAP_DEFAULTS, TICK_DEFAULTS, AXES_DEFAULTS, CROSSSECTION_DEFAULTS, default_volcano
)
from .utils import prep_catalog_data_mpl
from .legends import MagLegend
from vdapseisutils.utils.geoutils import backazimuth, sight_point_pyproj, project2line
from vdapseisutils.core.maps import elev_profile


class CrossSection:
    """
    CrossSection class that creates cross-section axes without inheriting from plt.Figure
    This avoids conflicts when used with SubFigures
    """

    name = "cross-section"

    def __init__(self, fig=None, points=[(46.198776, -122.261317), (46.197484, -122.122234)],
                 origin=None, radius_km=25.0, azimuth=270, map_extent=None,
                 depth_extent=(-50., 4.), resolution="auto", max_n=100,
                 label="A", width=None, maglegend=MagLegend(), verbose=False, **kwargs):

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
        
        # Create the axes
        self.ax = fig.add_subplot(111, **plot_kwargs)
        
        # Set thicker spines
        for spine in self.ax.spines.values():
            spine.set_linewidth(AXES_DEFAULTS['spine_linewidth'])

        # Better organization
        self.properties = dict()

        # Determine A & A' locations
        if origin:
            self.properties["origin"] = origin
            self.properties["azimuth"] = azimuth
            self.properties["radius"] = radius_km * 1000
            self.properties["points"] = [np.nan, np.nan]
            self.properties["points"][0] = sight_point_pyproj(origin, azimuth, self.properties["radius"])
            self.properties["points"][1] = sight_point_pyproj(origin, np.mod(azimuth + 180, 360), self.properties["radius"])
        else:
            if len(points) != 2:
                raise ValueError("ERROR: Points must be a list of 2 tuples of lat,lon coordinates.")
            self.properties["points"] = points
            self.properties["origin"] = None
            self.properties["azimuth"], self.properties["length"] = backazimuth(points[0], points[1])
            self.properties["radius"] = None

        # Extents by which to filter input data
        self.properties["map_extent"] = map_extent
        self.properties["depth_extent"] = depth_extent
        self.properties["depth_range"] = depth_extent[1] - depth_extent[0]
        self.properties["label"] = label
        self.properties["full_label"] = "{a}-{a}'".format(a=self.properties["label"])
        self.properties["width"] = width
        self.properties["orientation"] = "horizontal"

        self.verbose = verbose

        # Stub (I probably shouldn't redefine these here)
        self.A1 = self.properties["points"][0]
        self.A2 = self.properties["points"][1]

        self.profile = elev_profile.TopographicProfile([self.A1, self.A2], resolution=resolution, max_n=max_n)
        if np.any(self.profile.elevation):
            self._add_profile()

        self._setup_axis_formatting()
        self._add_labels_to_xsection()

    def _add_profile(self):
        """Add topographic profile to the cross-section."""
        hd = self.profile.distance / 1000  # horizontal distance along line (convert meters to km)
        elev = np.array(self.profile.elevation / 1000)  # elevation (convert m to km)
        self.plot(x=hd, z=elev, z_dir="elev", color=CROSSSECTION_DEFAULTS['profile_color'], z_unit="km", linewidth=CROSSSECTION_DEFAULTS['profile_linewidth'])
        self.set_depth_extent()
        self.set_horiz_extent()

        # custom spine bounds for a nice clean look
        self.ax.spines['top'].set_visible(False)
        self.ax.spines["left"].set_bounds((self.properties["depth_extent"][0], elev[0]))
        self.ax.spines["right"].set_bounds(self.properties["depth_extent"][0], elev[-1])

    def _setup_axis_formatting(self):
        """Set up axis formatting and styling."""
        # Apply centralized tick styling
        self.ax.tick_params(axis='both', 
                            labelcolor=TICK_DEFAULTS['labelcolor'],
                            labelsize=TICK_DEFAULTS['labelsize'],
                            color=TICK_DEFAULTS['tick_color'],
                            length=TICK_DEFAULTS['tick_size'],
                            width=TICK_DEFAULTS['tick_width'],
                            direction=TICK_DEFAULTS['tick_direction'],
                            pad=TICK_DEFAULTS['tick_pad'],
                            left=False, labelleft=False,
                            bottom=True, labelbottom=True,
                            right=True, labelright=True,
                            top=False, labeltop=False)
        self.ax.yaxis.set_label_position("right")
        self.ax.set_ylabel("Depth (km)", rotation=CROSSSECTION_DEFAULTS['ylabel_rotation'], labelpad=CROSSSECTION_DEFAULTS['ylabel_pad'], 
                           color=TICK_DEFAULTS['axes_labelcolor'], 
                           fontsize=TICK_DEFAULTS['axes_labelsize'])
        
        # Remove xlabel
        self.ax.set_xlabel("")  # Remove xlabel
    
    def _append_km_to_last_xtick(self):
        """Append ' km' to the last xticklabel while preserving original formatting."""
        # Get current tick labels after drawing
        tick_labels = self.ax.get_xticklabels()
        if tick_labels and len(tick_labels) > 0:
            # Get the original text of the last label
            last_label_text = tick_labels[-1].get_text()
            
            # Create new labels list
            new_labels = [label.get_text() for label in tick_labels]
            new_labels[-1] = f'{last_label_text} km'
                
            # Use set_xticks and set_xticklabels together to avoid the warning
            ticks = self.ax.get_xticks()
            self.ax.set_xticks(ticks)
            self.ax.set_xticklabels(new_labels)

    def _add_labels_to_xsection(self):
        """Add A and A' labels to the cross-section."""
        self.set_horiz_extent()
        self.set_depth_extent()
        x1 = self.ax.get_xlim()[0] + (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.03
        x2 = self.ax.get_xlim()[1] - (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.03
        y = self.ax.get_ylim()[0] + (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.03

        self.ax.text(x1, y, "{}".format(self.properties["label"]),
                     verticalalignment='bottom', horizontalalignment='left',
                     path_effects=[pe.withStroke(linewidth=CROSSSECTION_DEFAULTS['text_stroke_linewidth'], foreground=CROSSSECTION_DEFAULTS['text_stroke_color'])])
        self.ax.text(x2, y, "{}'".format(self.properties["label"]),
                     verticalalignment='bottom', horizontalalignment='right',
                     path_effects=[pe.withStroke(linewidth=CROSSSECTION_DEFAULTS['text_stroke_linewidth'], foreground=CROSSSECTION_DEFAULTS['text_stroke_color'])])
        
        # Append ' km' to the last xticklabel
        self._append_km_to_last_xtick()

    def set_depth_extent(self, depth_extent=None):
        """Set the depth extent (y-axis limits) of the cross-section."""
        if depth_extent is None:
            self.ax.set_ylim(self.properties["depth_extent"])
        else:
            self.ax.set_ylim(depth_extent)

    def set_horiz_extent(self, extent=None):
        """Set the horizontal extent (x-axis limits) of the cross-section."""
        if extent is None:
            self.ax.set_xlim(0, self.properties["radius"]*2/1000)
        else:
            self.ax.set_xlim(extent)

    def plot(self, lat=None, lon=None, z=None, x=None, z_dir="depth", z_unit="m", **kwargs):
        """
        Plot data on the cross-section.
        
        Parameters:
        -----------
        lat : array-like, optional
            Latitude coordinates
        lon : array-like, optional
            Longitude coordinates
        z : array-like, optional
            Depth/elevation values
        x : array-like, optional
            Distance along cross-section line (if provided, lat/lon ignored)
        z_dir : str, optional
            Direction of z values: "depth" (positive down) or "elev" (positive up)
        z_unit : str, optional
            Units of z values: "m" or "km"
        **kwargs
            Additional plotting arguments
        """
        if z is None:
            z = np.zeros_like(x if x is not None else lat)

        depth = np.asarray(z)

        if z_unit.lower() == "km":
            z_unit_conv = 1
        elif z_unit.lower() == "m":
            z_unit_conv = 1 / 1000
        else:
            raise ValueError(f"Invalid z_unit '{z_unit}'. Options: 'km' or 'm'.")

        if z_dir.lower() == "depth":
            z_dir_conv = -1
        elif z_dir.lower() == "elev":
            z_dir_conv = 1
        else:
            raise ValueError(f"Invalid z_dir '{z_dir}'. Options: 'depth' or 'elev'.")

        depth = depth * z_unit_conv * z_dir_conv

        if x is None:
            if lat is None or lon is None:
                raise ValueError("Either (lat, lon) or x must be provided.")
            x = project2line(lat, lon, P1=self.A1, P2=self.A2, unit="km")

        x = np.asarray(x)
        self.ax.plot(x, depth, **kwargs)

    def plot_catalog(self, catalog, s="magnitude", c="time", color=None, cmap="viridis_r", alpha=0.5, **kwargs):
        """
        Plot earthquake catalog on the cross-section.
        
        Creates a scatter plot of earthquake events from an ObsPy Catalog object
        projected onto the cross-section line, with customizable size, color, 
        and styling options.
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

        x = project2line(catdata["lat"], catdata["lon"], P1=self.A1, P2=self.A2, unit="km")
        scatter = self.ax.scatter(x, catdata["depth"], s=s, c=c, cmap=cmap, alpha=alpha, **kwargs)
        self.set_depth_extent()
        self.set_horiz_extent()
        return scatter

    def plot_inventory(self, inventory, marker_size=6, color='black', alpha=0.8, **kwargs):
        """Plot seismic station inventory on the cross-section."""
        try:
            station_lats = []
            station_lons = []
            station_elevs = []
            
            for network in inventory:
                for station in network:
                    if hasattr(station, 'latitude') and hasattr(station, 'longitude'):
                        station_lats.append(station.latitude)
                        station_lons.append(station.longitude)
                        station_elevs.append(station.elevation)
            
            if station_lats and station_lons:
                x_coords = project2line(station_lats, station_lons, P1=self.A1, P2=self.A2, unit="km")
                elevs = np.full_like(x_coords, station_elevs)
                
                self.ax.scatter(x_coords, elevs/-1000,
                              s=marker_size, 
                              c=color, 
                              marker='v',
                              alpha=alpha,
                              **kwargs)
                self.set_depth_extent()
                self.set_horiz_extent()
            else:
                print("No valid station coordinates found in inventory for cross-section")
                
        except Exception as e:
            print(f"Error plotting inventory on cross-section: {e}")
            print("Continuing without cross-section inventory plot...")

    def plot_heatmap(self, *args, grid_size=HEATMAP_DEFAULTS['grid_size'], 
                     cmap=HEATMAP_DEFAULTS['cmap'], alpha=HEATMAP_DEFAULTS['alpha'], 
                     vmin=HEATMAP_DEFAULTS['vmin'], vmax=HEATMAP_DEFAULTS['vmax'], **kwargs):
        """Plot a heatmap on the cross-section."""
        try:
            if len(args) == 1 and hasattr(args[0], 'events'):
                catalog = args[0]
                catdata = prep_catalog_data_mpl(catalog, time_format="matplotlib")
                lat = catdata["lat"].values
                lon = catdata["lon"].values
                depth = catdata["depth"].values
            elif len(args) >= 2:
                lat = np.asarray(args[0])
                lon = np.asarray(args[1])
                depth = np.asarray(args[2]) if len(args) > 2 else None
            else:
                raise ValueError("Usage: plot_heatmap(catalog, ...) or plot_heatmap(lat, lon, [depth], ...)")
            
            if len(lat) == 0 or len(lon) == 0:
                print("Warning: Empty coordinate arrays")
                return None
            
            x = project2line(lat, lon, P1=self.A1, P2=self.A2, unit="km")
            
            if len(x) == 0 or np.any(np.isnan(x)):
                print("Warning: Failed to project coordinates to cross-section line")
                return None
            
            depth_km = np.asarray(depth) / 1000.0
            depth_km = -depth_km
            
            x_min, x_max = np.min(x), np.max(x)
            depth_min, depth_max = np.min(depth_km), np.max(depth_km)
            
            grid_size_km = grid_size * 111.0
            
            if grid_size_km < 0.1:
                grid_size_km = 0.1
            
            data_range_x = x_max - x_min
            data_range_depth = depth_max - depth_min
            min_grid_size = max(0.1, min(data_range_x, data_range_depth) * 0.1)
            if grid_size_km < min_grid_size:
                grid_size_km = min_grid_size
            
            x_pad = (x_max - x_min) * 0.1
            depth_pad = (depth_max - depth_min) * 0.1
            
            if x_max <= x_min or depth_max <= depth_min:
                print("Warning: Invalid coordinate ranges for heatmap")
                return None
            
            x_grid = np.arange(x_min - x_pad, x_max + x_pad, grid_size_km)
            depth_grid = np.arange(depth_min - depth_pad, depth_max + depth_pad, grid_size_km)
            
            if len(x_grid) < 2 or len(depth_grid) < 2:
                print("Warning: Grid too small for heatmap")
                return None
            
            H, xedges, yedges = np.histogram2d(x, depth_km, bins=[x_grid, depth_grid])
            
            if H.size == 0 or np.all(H == 0):
                print("Warning: No data points in the specified region")
                return None
            
            x_centers = (xedges[:-1] + xedges[1:]) / 2
            depth_centers = (yedges[:-1] + yedges[1:]) / 2
            x_mesh, depth_mesh = np.meshgrid(x_centers, depth_centers)
            
            im = self.ax.pcolormesh(x_mesh, depth_mesh, H.T, 
                                   cmap=cmap, alpha=alpha, 
                                   vmin=vmin, vmax=vmax,
                                   **kwargs)
            
            return im
            
        except Exception as e:
            print(f"Error creating cross-section heatmap: {e}")
            print("Continuing without heatmap...")
            return None


def _test_cross_section():
    """Simple test to verify CrossSection class works correctly."""
    try:
        # Test CrossSection creation with default parameters
        points = [(46.2, -122.3), (46.1, -122.1)]  # Example points near Mt. Hood
        xs_obj = CrossSection(points=points, label="Test")
        
        print("✓ CrossSection class created successfully")
        print(f"✓ Cross-section points: {xs_obj.properties['points']}")
        print(f"✓ Cross-section label: {xs_obj.properties['label']}")
        print(f"✓ Depth extent: {xs_obj.properties['depth_extent']}")
        
        # Test with origin and azimuth
        xs_obj2 = CrossSection(origin=(46.15, -122.2), azimuth=45, radius_km=10, label="B")
        print(f"✓ CrossSection with origin created: azimuth {xs_obj2.properties['azimuth']}°")
        
        return True
        
    except Exception as e:
        print(f"✗ CrossSection test failed: {e}")
        return False


if __name__ == "__main__":
    _test_cross_section()
