#!/usr/bin/env python3
"""
Demonstration of improved hillshade functionality in vdapseisutils.

This script shows different hillshade blending modes and settings
for creating visually appealing terrain visualizations.
"""

import matplotlib.pyplot as plt
from vdapseisutils.core.maps.maps import Map

def demo_hillshade_modes():
    """Demonstrate different hillshade blending modes."""
    
    # Define a test region (Mount Hood area)
    origin = (45.374, -121.695)  # Mount Hood
    radial_extent_km = 30.0
    
    # Create figure with subplots for different modes
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Hillshade Blending Modes Comparison", fontsize=16)
    
    modes = [
        ("hillshade_only", "Hillshade Only"),
        ("elevation_only", "Elevation Only"), 
        ("multiply", "Multiply Blend"),
        ("overlay", "Overlay Blend")
    ]
    
    for idx, (mode, title) in enumerate(modes):
        ax_idx = (idx // 2, idx % 2)
        
        # Create map
        map_obj = Map(origin=origin, radial_extent_km=radial_extent_km)
        
        # Add hillshade with specific mode
        map_obj.add_hillshade(
            blend_mode=mode,
            resolution="auto",  # Auto-select resolution
            alpha=0.9,
            cmap="Greys_r"  # Black and white/cream
        )
        
        # Add coastline for reference
        map_obj.add_coastline()
        
        # Add scale bar
        map_obj.add_scalebar()
        
        # Set title
        axes[ax_idx].set_title(title, fontsize=12)
        
        # Copy the map to the subplot
        axes[ax_idx].imshow(map_obj.ax.get_images()[0].get_array(), 
                           extent=map_obj.properties["map_extent"])
        axes[ax_idx].set_aspect('equal')
        axes[ax_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demo_customization():
    """Demonstrate hillshade customization options."""
    
    origin = (45.374, -121.695)  # Mount Hood
    radial_extent_km = 25.0
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Hillshade Customization Examples", fontsize=16)
    
    # Example 1: High contrast terrain
    map1 = Map(origin=origin, radial_extent_km=radial_extent_km)
    map1.add_hillshade(
        blend_mode="overlay",
        elevation_weight=0.4,
        hillshade_weight=0.6,
        cmap="Greys_r",
        vertical_exag=2.0,
        alpha=0.9
    )
    map1.add_coastline()
    axes[0, 0].set_title("High Contrast Terrain", fontsize=12)
    axes[0, 0].imshow(map1.ax.get_images()[0].get_array(), 
                     extent=map1.properties["map_extent"])
    axes[0, 0].set_aspect('equal')
    
    # Example 2: Subtle hillshade
    map2 = Map(origin=origin, radial_extent_km=radial_extent_km)
    map2.add_hillshade(
        blend_mode="overlay",
        elevation_weight=0.2,
        hillshade_weight=0.8,
        cmap="Greys_r",
        vertical_exag=1.2,
        alpha=0.7
    )
    map2.add_coastline()
    axes[0, 1].set_title("Subtle Hillshade", fontsize=12)
    axes[0, 1].imshow(map2.ax.get_images()[0].get_array(), 
                     extent=map2.properties["map_extent"])
    axes[0, 1].set_aspect('equal')
    
    # Example 3: Different lighting
    map3 = Map(origin=origin, radial_extent_km=radial_extent_km)
    map3.add_hillshade(
        blend_mode="multiply",
        radiance=[45, 30],  # Different lighting angle
        cmap="Greys_r",
        alpha=0.8
    )
    map3.add_coastline()
    axes[1, 0].set_title("Different Lighting", fontsize=12)
    axes[1, 0].imshow(map3.ax.get_images()[0].get_array(), 
                     extent=map3.properties["map_extent"])
    axes[1, 0].set_aspect('equal')
    
    # Example 4: Ocean included
    map4 = Map(origin=origin, radial_extent_km=radial_extent_km)
    map4.add_hillshade(
        blend_mode="overlay",
        bath=True,  # Include bathymetry
        cmap="Greys_r",
        alpha=0.9
        )
    map4.add_coastline()
    axes[1, 1].set_title("With Bathymetry", fontsize=12)
    axes[1, 1].imshow(map4.ax.get_images()[0].get_array(), 
                     extent=map4.properties["map_extent"])
    axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def demo_performance():
    """Demonstrate caching and performance improvements."""
    
    origin = (45.374, -121.695)
    radial_extent_km = 20.0
    
    print("Testing hillshade performance with caching...")
    
    # First call (will download and cache)
    print("First call (downloading data)...")
    map1 = Map(origin=origin, radial_extent_km=radial_extent_km)
    map1.add_hillshade(cache_data=True, resolution="auto")
    
    # Second call (should load from cache)
    print("Second call (loading from cache)...")
    map2 = Map(origin=origin, radial_extent_km=radial_extent_km)
    map2.add_hillshade(cache_data=True, resolution="auto")
    
    print("Performance test complete!")

def demo_auto_resolution():
    """Demonstrate automatic resolution selection."""
    
    print("Testing automatic resolution selection...")
    
    # Test different map sizes
    test_cases = [
        ((45.374, -121.695), 5.0, "Small local area"),
        ((45.374, -121.695), 15.0, "Medium local area"),
        ((45.374, -121.695), 50.0, "Large local area"),
        ((45.374, -121.695), 200.0, "Regional area"),
    ]
    
    for origin, radial_extent_km, description in test_cases:
        print(f"\n{description} (extent: {radial_extent_km} km):")
        map_obj = Map(origin=origin, radial_extent_km=radial_extent_km)
        # Just call add_hillshade to trigger auto-resolution selection
        # We'll catch the print statement that shows the selected resolution
        try:
            map_obj.add_hillshade(resolution="auto", cache_data=False)
        except:
            pass  # We're just testing the resolution selection, not the actual plotting

if __name__ == "__main__":
    print("Hillshade Demo - vdapseisutils")
    print("=" * 40)
    
    # Run demonstrations
    demo_hillshade_modes()
    demo_customization()
    demo_performance()
    demo_auto_resolution() 