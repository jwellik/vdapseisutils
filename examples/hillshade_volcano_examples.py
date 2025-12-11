#!/usr/bin/env python3
"""
Volcano Hillshade Examples using vdapseisutils.add_hillshade_pygmt()

This script demonstrates the improved hillshade functionality with five different volcanoes
at various scales and resolutions.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2025
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np

# Import the hillshade function
from vdapseisutils.core.maps.maps import add_hillshade_pygmt
from vdapseisutils.utils.geoutils import radial_extent2map_extent

def create_volcano_hillshade_map(lon, lat, name, radius_km, resolution="auto", 
                                blend_mode="overlay", alpha=0.8, vertical_exag=1.5):
    """
    Create a volcano map with hillshade using the add_hillshade_pygmt function.
    
    Parameters:
    -----------
    lon, lat : float
        Longitude and latitude of the volcano
    name : str
        Volcano name for the title
    radius_km : float
        Radius in kilometers for the map extent
    resolution : str
        PyGMT resolution ("auto", "01s", "30s", "01m", etc.)
    blend_mode : str
        Hillshade blending mode ("overlay", "multiply", "hillshade_only", "elevation_only")
    alpha : float
        Transparency of the hillshade
    vertical_exag : float
        Vertical exaggeration factor
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Set up the map projection
    projection = ccrs.PlateCarree()
    
    # Create figure and axes
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=projection)
    
    # Calculate extent for the specified radius using proper geodesic calculations
    extent = radial_extent2map_extent(lat, lon, radius_km)
    ax.set_extent(extent, crs=projection)
    
    # Add enhanced hillshade
    print(f"\nAdding hillshade for {name} ({radius_km} km radius)...")
    add_hillshade_pygmt(
            ax, 
            extent=extent,
            data_source="igpp",  # IGPP data source
            resolution=resolution,
            topo=True,           # Include topography
            bath=False,          # Exclude bathymetry for land volcanoes
            radiance=[315, 45],  # Lighting from northwest
            vertical_exag=vertical_exag,
            cmap="Greys_r",      # Black and white/cream colormap
            alpha=alpha,
            blend_mode=blend_mode,
            elevation_weight=0.3,
            hillshade_weight=0.7,
            normalize_elevation=True,
            cache_data=True,     # Enable caching for better performance
            transform_data="auto" # Apply automatic rotation and flip
        )
    
    # Add coastline and land features
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.LAND, alpha=0.1)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
    
    # Add country borders for context
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', alpha=0.7)
    
    # Plot volcano location with a prominent marker
    ax.plot(lon, lat, 'r^', markersize=15, 
            transform=projection, label=name,
            zorder=10,  # Ensure volcano marker is on top
            markeredgecolor='black', markeredgewidth=2)
    
    # Add gridlines
    gl = ax.gridlines(crs=projection, draw_labels=True,
                     linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # Font sizes removed - uses rcParams automatically
    
    # Add title with volcano info
    plt.title(f'{name} Volcano - {radius_km} km radius', 
              fontweight='bold', pad=20)  # fontsize removed - uses rcParams
    
    # Add legend
    ax.legend(loc='upper right')  # fontsize removed - uses rcParams
    
    # Make axes square
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig, ax

def main():
    """Create five example volcano maps with different hillshade settings."""
    
    # Define volcano locations and parameters
    volcanoes = [
        {
            'name': 'Mount Spurr',
            'lon': -152.251,
            'lat': 61.299,
            'radius_km': 100.,
            'resolution': '03s',  # High resolution for detailed view
            'blend_mode': 'overlay',
            'vertical_exag': 2.0,
            'alpha': 0.9
        },
        {
            'name': 'Kilauea',
            'lon': -155.2875,
            'lat': 19.4061,
            'radius_km': 50,
            'resolution': '03s',  # Medium resolution for large area
            'blend_mode': 'overlay',
            'vertical_exag': 1.5,
            'alpha': 0.8,
        },
        {
            'name': 'Mount Awu',
            'lon': 125.5,
            'lat': 3.67,
            'radius_km': 50,
            'resolution': '01s',  # High resolution for detailed view
            'blend_mode': 'overlay',
            'vertical_exag': 1.8,
            'alpha': 0.85
        },
        {
            'name': 'Mount St. Helens',
            'lon': -122.182,
            'lat': 46.2,
            'radius_km': 15,
            'resolution': '01s',  # Very high resolution for small area
            'blend_mode': 'overlay',
            'vertical_exag': 1.5,
            'alpha': 0.9
        },
        {
            'name': 'Copahue',
            'lon': -71.17,
            'lat': -37.85,
            'radius_km': 20,
            'resolution': '15s',  # Very high resolution for small area
            'blend_mode': 'overlay',
            'vertical_exag': 2.5,
            'alpha': 0.8
        }
    ]
    
    print("Volcano Hillshade Examples - vdapseisutils")
    print("=" * 50)
    print("This script demonstrates the add_hillshade_pygmt() function")
    print("with five different volcanoes at various scales and resolutions.")
    print()
    
    # Create each volcano map
    for i, volcano in enumerate(volcanoes, 1):
        print(f"Creating map {i}/5: {volcano['name']}")
        
        fig, ax = create_volcano_hillshade_map(
            lon=volcano['lon'],
            lat=volcano['lat'],
            name=volcano['name'],
            radius_km=volcano['radius_km'],
            resolution=volcano['resolution'],
            blend_mode=volcano['blend_mode'],
            alpha=volcano['alpha'],
            vertical_exag=volcano['vertical_exag']
        )
        
        # Show the plot
        plt.show()
        
        # Optional: save the figure
        # fig.savefig(f"{volcano['name'].replace(' ', '_').replace('.', '')}_hillshade.png", 
        #             dpi=300, bbox_inches='tight')
        # print(f"Saved: {volcano['name']}_hillshade.png")
        
        plt.close(fig)  # Close to free memory
    
    print("\nAll volcano maps completed!")
    print("\nHillshade blending modes demonstrated:")
    print("- overlay: Combines hillshade and elevation (default)")
    print("- multiply: Multiplies hillshade with elevation")
    print("- hillshade_only: Pure hillshade visualization")
    print("- elevation_only: Pure elevation with colormap")

def demo_different_blend_modes():
    """Demonstrate different blending modes with Mount St. Helens."""
    
    print("\n" + "="*60)
    print("DEMO: Different Blending Modes Comparison")
    print("="*60)
    
    # Mount St. Helens at 15 km radius to show different modes
    lon, lat = -122.182, 46.2
    radius_km = 15
    
    # Create subplots for different blend modes
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hillshade Blending Modes Comparison - Mount St. Helens', 
                 fontweight='bold')
    
    blend_modes = [
        ("overlay", "Overlay Blend (Default)"),
        ("multiply", "Multiply Blend"),
        ("hillshade_only", "Hillshade Only"),
        ("elevation_only", "Elevation Only")
    ]
    
    for idx, (mode, title) in enumerate(blend_modes):
        ax_idx = (idx // 2, idx % 2)
        ax = axes[ax_idx]
        
        # Set up the map projection
        projection = ccrs.PlateCarree()
        ax.set_projection(projection)
        
        # Calculate extent
        extent = radial_extent2map_extent(lat, lon, radius_km)
        ax.set_extent(extent, crs=projection)
        
        # Add hillshade with specific mode
        print(f"Adding {mode} mode...")
        add_hillshade_pygmt(
            ax, 
            extent=extent,
            resolution='03s',
            blend_mode=mode,
            alpha=0.9,
            vertical_exag=2.5,
            transform_data="rotate_only"  # Use automatic transformation
        )
        
        # Add basic features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
        
        # Plot volcano location
        ax.plot(lon, lat, 'r^', markersize=12, 
                transform=projection, label='Mount St. Helens',
                zorder=10, markeredgecolor='black', markeredgewidth=1.5)
        
        # Add gridlines
        gl = ax.gridlines(crs=projection, draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set title
        ax.set_title(title, fontweight='bold')  # fontsize removed - uses rcParams
        ax.legend(loc='upper right')  # fontsize removed - uses rcParams
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def demo_transformation_options():
    """Demonstrate different data transformation options with Mount St. Helens."""
    
    print("\n" + "="*60)
    print("DEMO: Data Transformation Options Comparison")
    print("="*60)
    
    # Mount St. Helens at 10 km radius to show transformation differences
    lon, lat = -122.182, 46.2
    radius_km = 10
    
    # Create subplots for different transformation options
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Data Transformation Options Comparison - Mount St. Helens', 
                 fontweight='bold')
    
    transform_options = [
        ("none", "No Transformation"),
        ("rotate_only", "Rotate 90° Clockwise Only"),
        ("flip_only", "Flip Left-to-Right Only"),
        ("auto", "Auto (Rotate + Flip)")
    ]
    
    for idx, (transform, title) in enumerate(transform_options):
        ax_idx = (idx // 2, idx % 2)
        ax = axes[ax_idx]
        
        # Set up the map projection
        projection = ccrs.PlateCarree()
        ax.set_projection(projection)
        
        # Calculate extent
        extent = radial_extent2map_extent(lat, lon, radius_km)
        ax.set_extent(extent, crs=projection)
        
        # Add hillshade with specific transformation
        print(f"Adding hillshade with {transform} transformation...")
        add_hillshade_pygmt(
            ax, 
            extent=extent,
            resolution='01s',
            blend_mode="overlay",
            alpha=0.9,
            vertical_exag=2.0,
            transform_data=transform
        )
        
        # Add basic features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
        
        # Plot volcano location
        ax.plot(lon, lat, 'r^', markersize=12, 
                transform=projection, label='Mount St. Helens',
                zorder=10, markeredgecolor='black', markeredgewidth=1.5)
        
        # Add gridlines
        gl = ax.gridlines(crs=projection, draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set title
        ax.set_title(title, fontweight='bold')  # fontsize removed - uses rcParams
        ax.legend(loc='upper right')  # fontsize removed - uses rcParams
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Run the main volcano examples
    main()
    
    # Run the blending modes comparison demo
    demo_different_blend_modes()
    
    # Run the transformation options comparison demo
    demo_transformation_options()
    
    print("\n" + "="*60)
    print("SCRIPT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("The add_hillshade_pygmt() function should now display")
    print("topography in the correct orientation after fixing the")
    print("rotation and flip issues.")
    print("\nTransformation options available:")
    print("- 'auto': Applies both 90° clockwise rotation and left-to-right flip")
    print("- 'none': No transformation (raw PyGMT data)")
    print("- 'rotate_only': Only 90° clockwise rotation")
    print("- 'flip_only': Only left-to-right flip")
    print("- 'custom': Same as 'auto' for backward compatibility") 