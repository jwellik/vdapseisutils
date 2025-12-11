"""
Test module for swarmmpl3 with real seismic data
"""

import matplotlib.pyplot as plt
import numpy as np
from obspy import read, UTCDateTime, Stream, Trace
from obspy.clients.fdsn import Client

from . import swarmw, swarmg, swarmwg, Clipboard, TimeAxes, Panel


def get_real_data():
    """
    Get real seismic data for testing from IRIS.
    """
    print("Fetching real seismic data from IRIS...")
    
    try:
        # Try to get recent data from IRIS
        client = Client("IRIS")
        
        # Get data from a reliable station - try recent dates
        dates_to_try = [
            UTCDateTime("2024-01-15T12:00:00"),
            UTCDateTime("2024-01-10T06:00:00"), 
            UTCDateTime("2024-01-05T18:00:00"),
            UTCDateTime("2023-12-20T12:00:00")
        ]
        
        # Try multiple stations to increase chances of getting data
        stations = [
            ("IU", "ANMO", "00", "BHZ"), 
            ("IU", "CCM", "00", "BHZ"), 
            ("US", "WMOK", "00", "BHZ"),
            ("IU", "COLA", "00", "BHZ"),
            ("US", "LRAL", "00", "BHZ")
        ]
        
        st = Stream()
        
        for starttime in dates_to_try:
            endtime = starttime + 300  # 5 minutes of data
            
            for net, sta, loc, cha in stations:
                try:
                    st_temp = client.get_waveforms(net, sta, loc, cha, starttime, endtime)
                    st += st_temp
                    print(f"  ✓ Downloaded {net}.{sta}.{loc}.{cha} from {starttime.date}")
                    
                    if len(st) >= 3:  # Get up to 3 traces
                        break
                except Exception as e:
                    print(f"  ⚠ Could not get {net}.{sta}.{loc}.{cha} from {starttime.date}: {e}")
                    continue
                    
            if len(st) >= 3:
                break
                
        if len(st) > 0:
            # Resample to reasonable rate for spectrograms
            for tr in st:
                if tr.stats.sampling_rate > 20:
                    tr.resample(20.0)
            print(f"Successfully downloaded {len(st)} real traces")
            return st
        else:
            raise Exception("No real data could be downloaded from any source")
            
    except Exception as e:
        print(f"❌ Real data download failed: {e}")
        print("Please check your internet connection and try again.")
        raise


def test_single_plots():
    """Test individual plotting functions."""
    print("\n=== Testing Single Plots ===")
    
    # Get data
    st = get_real_data()
    trace = st[0]  # Use first trace
    
    print(f"\nTesting with trace: {trace.id}")
    print(f"Duration: {trace.stats.endtime - trace.stats.starttime:.1f} seconds")
    print(f"Sampling rate: {trace.stats.sampling_rate} Hz")
    
    # Test 1: Waveform with different tick types
    print("\n1. Testing waveform plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Waveform Tests - {trace.id}', fontsize=14)
    
    tick_types = ["absolute", "seconds", "minutes", "hours"]
    colors = ["blue", "green", "red", "purple"]
    
    for i, (tick_type, color) in enumerate(zip(tick_types, colors)):
        ax = axes[i//2, i%2]
        try:
            p = swarmw(trace, ax=ax, tick_type=tick_type, color=color)
            
            # Add some vertical lines
            start_time = trace.stats.starttime
            p.axvline(start_time + 60, color="red", linewidth=2, alpha=0.7)
            p.axvline(30, t_units="seconds", color="orange", linewidth=1, linestyle="--", alpha=0.7)
            
            ax.set_title(f'Tick type: {tick_type}', fontsize=12)
            print(f"  ✓ {tick_type} waveform successful")
        except Exception as e:
            print(f"  ✗ {tick_type} waveform failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 2: Spectrogram
    print("\n2. Testing spectrogram...")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        p = swarmg(trace, ax=ax, tick_type="minutes")
        
        # Add vertical lines at event times
        start_time = trace.stats.starttime
        p.axvline(start_time + 60, color="white", linewidth=2, label="P-wave")
        p.axvline(start_time + 120, color="yellow", linewidth=2, label="S-wave")
        p.axvline(2, t_units="minutes", color="cyan", linewidth=1, linestyle=":", label="Regional")
        
        ax.set_title(f'Spectrogram - {trace.id}', fontsize=14)
        ax.legend()
        plt.show()
        print("  ✓ Spectrogram successful")
    except Exception as e:
        print(f"  ✗ Spectrogram failed: {e}")


def test_panel_plots():
    """Test Panel (waveform + spectrogram) functionality."""
    print("\n=== Testing Panel Plots ===")
    
    st = get_real_data()
    trace = st[0]
    
    # Test 1: Basic panel
    print("\n1. Testing basic waveform + spectrogram panel...")
    try:
        p = swarmwg(trace, tick_type="seconds", figsize=(14, 8))
        
        # Add event markers
        start_time = trace.stats.starttime
        p.axvline(start_time + 60, color="red", linewidth=2)  # P-wave on both
        p.axvline(start_time + 120, color="blue", axes=[1], linewidth=2)  # S-wave on spectrogram only
        p.axvline(90, t_units="seconds", color="green", linewidth=1, linestyle="--")  # Relative time
        
        # plt.suptitle(f'Panel Test - {trace.id}', fontsize=14)
        plt.show()
        print("  ✓ Basic panel successful")
    except Exception as e:
        print(f"  ✗ Basic panel failed: {e}")
    
    # Test 2: Custom panel settings
    print("\n2. Testing custom panel settings...")
    try:
        wave_settings = {"color": "darkblue", "linewidth": 0.8}
        spec_settings = {"wlen": 8.0, "overlap": 0.9, "dbscale": True, "log_power": True}
        
        p = swarmwg(trace, tick_type="minutes", height_ratios=[1, 4],
                   wave_settings=wave_settings, spec_settings=spec_settings,
                   figsize=(14, 10))
        
        # Add markers for all events
        start_time = trace.stats.starttime
        p.axvline(start_time + 60, color="white", linewidth=2, alpha=0.8)
        p.axvline(start_time + 120, color="yellow", linewidth=2, alpha=0.8)
        p.axvline(start_time + 200, color="cyan", linewidth=2, alpha=0.8)
        
        # plt.suptitle(f'Custom Panel - {trace.id}', fontsize=14)
        plt.show()
        print("  ✓ Custom panel successful")
    except Exception as e:
        print(f"  ✗ Custom panel failed: {e}")


def test_clipboard_plots():
    """Test Clipboard (multiple traces) functionality."""
    print("\n=== Testing Clipboard Plots ===")
    
    st = get_real_data()
    
    if len(st) < 2:
        print("  ⚠ Need at least 2 traces for clipboard tests")
        print("  ⚠ Skipping clipboard tests - not enough traces available")
        return
    
    print(f"Testing with {len(st)} traces")
    
    # Test 1: Clipboard without sync
    print("\n1. Testing clipboard without time synchronization...")
    try:
        cb = Clipboard(st, sync_waves=False, tick_type="seconds", figsize=(14, 12))
        
        # Add vertical lines to different panels
        start_time = st[0].stats.starttime
        cb.axvline(start_time + 60, color="red", linewidth=2)  # All panels
        cb.axvline(start_time + 120, panels=[0], color="blue", linewidth=2)  # First panel only
        cb.axvline(150, t_units="seconds", panels=[1], axes=[1], color="green")  # Second panel spectrogram
        
        # plt.suptitle('Clipboard Test - No Time Sync', fontsize=14)
        plt.show()
        print("  ✓ Clipboard (no sync) successful")
    except Exception as e:
        print(f"  ✗ Clipboard (no sync) failed: {e}")
    
    # Test 2: Clipboard with sync
    print("\n2. Testing clipboard with time synchronization...")
    try:
        cb = Clipboard(st, sync_waves=True, tick_type="minutes", figsize=(14, 14))
        
        # Add synchronized vertical lines
        start_time = st[0].stats.starttime
        cb.axvline(start_time + 60, color="red", linewidth=2, alpha=0.8)  # P-wave
        cb.axvline(start_time + 120, color="blue", linewidth=2, alpha=0.8)  # S-wave
        cb.axvline(3.5, t_units="minutes", color="purple", linewidth=1, linestyle=":", alpha=0.8)  # Regional
        
        # plt.suptitle('Clipboard Test - With Time Sync', fontsize=14)
        plt.show()
        print("  ✓ Clipboard (with sync) successful")
    except Exception as e:
        print(f"  ✗ Clipboard (with sync) failed: {e}")


def test_advanced_features():
    """Test advanced features and direct class usage."""
    print("\n=== Testing Advanced Features ===")
    
    st = get_real_data()
    trace = st[0]
    
    # Test 1: Direct TimeAxes usage
    print("\n1. Testing direct TimeAxes usage...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Waveform with custom styling
        ta1 = TimeAxes(ax=axes[0], tick_type="seconds")
        ta1.plot_waveform(trace, color="darkred", linewidth=0.8)
        ta1.axvline(trace.stats.starttime + 60, color="blue", linewidth=2)
        ta1.set_ylabel("Amplitude")
        ta1.set_title("Custom Waveform")
        
        # Spectrogram with log power
        ta2 = TimeAxes(ax=axes[1], tick_type="seconds")
        ta2.plot_spectrogram(trace, wlen=4.0, log_power=True, dbscale=True)
        ta2.axvline(trace.stats.starttime + 120, color="white", linewidth=2)
        ta2.set_ylabel("Frequency (Hz)")
        ta2.set_title("Log Power Spectrogram")
        
        # Linear spectrogram
        ta3 = TimeAxes(ax=axes[2], tick_type="seconds")
        ta3.plot_spectrogram(trace, wlen=6.0, log_power=False, dbscale=False)
        ta3.axvline(trace.stats.starttime + 200, color="black", linewidth=2)
        ta3.set_ylabel("Frequency (Hz)")
        ta3.set_title("Linear Spectrogram")
        
        plt.tight_layout()
        plt.show()
        print("  ✓ Direct TimeAxes usage successful")
    except Exception as e:
        print(f"  ✗ Direct TimeAxes usage failed: {e}")
    
    # Test 2: Dynamic tick type changes
    print("\n2. Testing dynamic tick type changes...")
    try:
        p = swarmwg(trace, tick_type="absolute", figsize=(14, 8))
        
        # Add initial marker
        p.axvline(trace.stats.starttime + 90, color="red", linewidth=2)
        
        # Change to relative time
        p.set_tick_type("seconds")
        p.axvline(150, t_units="seconds", color="blue", linewidth=2)
        
        # plt.suptitle(f'Dynamic Tick Change - {trace.id}', fontsize=14)
        plt.show()
        print("  ✓ Dynamic tick change successful")
    except Exception as e:
        print(f"  ✗ Dynamic tick change failed: {e}")


def run_all_tests():
    """Run comprehensive test suite."""
    print("🧪 swarmmpl3 Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        # Get data once and reuse for all tests
        print("Getting real seismic data for all tests...")
        st = get_real_data()
        
        print(f"\n📊 Testing with {len(st)} traces:")
        for tr in st:
            print(f"  - {tr.id}: {tr.stats.starttime} to {tr.stats.endtime}")
            print(f"    Duration: {tr.stats.endtime - tr.stats.starttime:.1f} seconds")
            print(f"    Sampling rate: {tr.stats.sampling_rate} Hz")
        
        # Run individual tests (modify to accept data parameter)
        test_single_plots_with_data(st)
        test_panel_plots_with_data(st)
        
        if len(st) >= 2:
            test_clipboard_plots_with_data(st)
        else:
            print("\n⚠️  Only one trace available - skipping clipboard tests")
            
        test_advanced_features_with_data(st)
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("\nClose plot windows to continue between tests.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


# Modified test functions that accept data as parameter
def test_single_plots_with_data(st):
    """Test individual plotting functions with provided data."""
    print("\n=== Testing Single Plots ===")
    
    trace = st[0]  # Use first trace
    
    print(f"\nTesting with trace: {trace.id}")
    print(f"Duration: {trace.stats.endtime - trace.stats.starttime:.1f} seconds")
    print(f"Sampling rate: {trace.stats.sampling_rate} Hz")
    
    # Test 1: Waveform with different tick types
    print("\n1. Testing waveform plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Waveform Tests - {trace.id}', fontsize=14)
    
    tick_types = ["absolute", "seconds", "minutes", "hours"]
    colors = ["blue", "green", "red", "purple"]
    
    for i, (tick_type, color) in enumerate(zip(tick_types, colors)):
        ax = axes[i//2, i%2]
        try:
            p = swarmw(trace, ax=ax, tick_type=tick_type, color=color)
            
            # Add some vertical lines
            start_time = trace.stats.starttime
            p.axvline(start_time + 60, color="red", linewidth=2, alpha=0.7)
            p.axvline(30, t_units="seconds", color="orange", linewidth=1, linestyle="--", alpha=0.7)
            
            ax.set_title(f'Tick type: {tick_type}', fontsize=12)
            print(f"  ✓ {tick_type} waveform successful")
        except Exception as e:
            print(f"  ✗ {tick_type} waveform failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 2: Spectrogram
    print("\n2. Testing spectrogram...")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        p = swarmg(trace, ax=ax, tick_type="minutes")
        
        # Add vertical lines
        start_time = trace.stats.starttime
        p.axvline(start_time + 60, color="white", linewidth=2, label="60s mark")
        p.axvline(start_time + 120, color="yellow", linewidth=2, label="120s mark")
        p.axvline(2, t_units="minutes", color="cyan", linewidth=1, linestyle=":", label="2min mark")
        
        ax.set_title(f'Spectrogram - {trace.id}', fontsize=14)
        ax.legend()
        plt.show()
        print("  ✓ Spectrogram successful")
    except Exception as e:
        print(f"  ✗ Spectrogram failed: {e}")


def test_panel_plots_with_data(st):
    """Test Panel functionality with provided data."""
    print("\n=== Testing Panel Plots ===")
    
    trace = st[0]
    
    # Test 1: Basic panel
    print("\n1. Testing basic waveform + spectrogram panel...")
    try:
        p = swarmwg(trace, tick_type="seconds", figsize=(14, 8))
        
        # Add event markers
        start_time = trace.stats.starttime
        p.axvline(start_time + 60, color="red", linewidth=2)  # Both axes
        p.axvline(start_time + 120, color="blue", axes=[1], linewidth=2)  # Spectrogram only
        p.axvline(90, t_units="seconds", color="green", linewidth=1, linestyle="--")  # Relative time
        
        # plt.suptitle(f'Panel Test - {trace.id}', fontsize=14)
        plt.show()
        print("  ✓ Basic panel successful")
    except Exception as e:
        print(f"  ✗ Basic panel failed: {e}")
    
    # Test 2: Custom panel settings
    print("\n2. Testing custom panel settings...")
    try:
        wave_settings = {"color": "darkblue", "linewidth": 0.8}
        spec_settings = {"wlen": 8.0, "overlap": 0.9, "dbscale": True, "log_power": True}
        
        p = swarmwg(trace, tick_type="minutes", height_ratios=[1, 4],
                   wave_settings=wave_settings, spec_settings=spec_settings,
                   figsize=(14, 10))
        
        # Add markers
        start_time = trace.stats.starttime
        p.axvline(start_time + 60, color="white", linewidth=2, alpha=0.8)
        p.axvline(start_time + 120, color="yellow", linewidth=2, alpha=0.8)
        p.axvline(start_time + 180, color="cyan", linewidth=2, alpha=0.8)
        
        # plt.suptitle(f'Custom Panel - {trace.id}', fontsize=14)
        plt.show()
        print("  ✓ Custom panel successful")
    except Exception as e:
        print(f"  ✗ Custom panel failed: {e}")


def test_clipboard_plots_with_data(st):
    """Test Clipboard functionality with provided data."""
    print("\n=== Testing Clipboard Plots ===")
    
    print(f"Testing with {len(st)} traces")
    
    # Test 1: Clipboard without sync
    print("\n1. Testing clipboard without time synchronization...")
    try:
        cb = Clipboard(st, sync_waves=False, tick_type="seconds", figsize=(14, 12))
        
        # Add vertical lines to different panels
        start_time = st[0].stats.starttime
        cb.axvline(start_time + 60, color="red", linewidth=2)  # All panels
        cb.axvline(start_time + 120, panels=[0], color="blue", linewidth=2)  # First panel only
        if len(st) > 1:
            cb.axvline(150, t_units="seconds", panels=[1], axes=[1], color="green")  # Second panel spectrogram
        
        # plt.suptitle('Clipboard Test - No Time Sync', fontsize=14)
        plt.show()
        print("  ✓ Clipboard (no sync) successful")
    except Exception as e:
        print(f"  ✗ Clipboard (no sync) failed: {e}")
    
    # Test 2: Clipboard with sync
    print("\n2. Testing clipboard with time synchronization...")
    try:
        cb = Clipboard(st, sync_waves=True, tick_type="minutes", figsize=(14, 14))
        
        # Add synchronized vertical lines
        start_time = st[0].stats.starttime
        cb.axvline(start_time + 60, color="red", linewidth=2, alpha=0.8)
        cb.axvline(start_time + 120, color="blue", linewidth=2, alpha=0.8)
        cb.axvline(3, t_units="minutes", color="purple", linewidth=1, linestyle=":", alpha=0.8)
        
        # plt.suptitle('Clipboard Test - With Time Sync', fontsize=14)
        plt.show()
        print("  ✓ Clipboard (with sync) successful")
    except Exception as e:
        print(f"  ✗ Clipboard (with sync) failed: {e}")


def test_advanced_features_with_data(st):
    """Test advanced features with provided data."""
    print("\n=== Testing Advanced Features ===")
    
    trace = st[0]
    
    # Test 1: Direct TimeAxes usage
    print("\n1. Testing direct TimeAxes usage...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Waveform with custom styling
        ta1 = TimeAxes(ax=axes[0], tick_type="seconds")
        ta1.plot_waveform(trace, color="darkred", linewidth=0.8)
        ta1.axvline(trace.stats.starttime + 60, color="blue", linewidth=2)
        ta1.set_ylabel("Amplitude")
        ta1.set_title("Custom Waveform")
        
        # Spectrogram with log power
        ta2 = TimeAxes(ax=axes[1], tick_type="seconds")
        ta2.plot_spectrogram(trace, wlen=4.0, log_power=True, dbscale=True)
        ta2.axvline(trace.stats.starttime + 120, color="white", linewidth=2)
        ta2.set_ylabel("Frequency (Hz)")
        ta2.set_title("Log Power Spectrogram")
        
        # Linear spectrogram
        ta3 = TimeAxes(ax=axes[2], tick_type="seconds")
        ta3.plot_spectrogram(trace, wlen=6.0, log_power=False, dbscale=False)
        ta3.axvline(trace.stats.starttime + 180, color="black", linewidth=2)
        ta3.set_ylabel("Frequency (Hz)")
        ta3.set_title("Linear Spectrogram")
        
        plt.tight_layout()
        plt.show()
        print("  ✓ Direct TimeAxes usage successful")
    except Exception as e:
        print(f"  ✗ Direct TimeAxes usage failed: {e}")
    
    # Test 2: Dynamic tick type changes
    print("\n2. Testing dynamic tick type changes...")
    try:
        p = swarmwg(trace, tick_type="absolute", figsize=(14, 8))
        
        # Add initial marker
        p.axvline(trace.stats.starttime + 90, color="red", linewidth=2)
        
        # Change to relative time
        p.set_tick_type("seconds")
        p.axvline(150, t_units="seconds", color="blue", linewidth=2)
        
        # plt.suptitle(f'Dynamic Tick Change - {trace.id}', fontsize=14)
        plt.show()
        print("  ✓ Dynamic tick change successful")
    except Exception as e:
        print(f"  ✗ Dynamic tick change failed: {e}")
    
    # Test 3: Targeted axvline examples
    if len(st) >= 2:
        print("\n3. Testing targeted axvline functionality...")
        try:
            # Create clipboard with at least 2 traces
            cb = Clipboard(st[:2], sync_waves=False, tick_type="seconds", figsize=(14, 10))
            
            # Add axvlines only to the first panel (both waveform and spectrogram)
            start_time = st[0].stats.starttime
            cb.axvline(start_time + 60, panels=[0], color="red", linewidth=3, alpha=0.8, 
                      label="First Panel Only")
            
            # Add axvline only to the waveform (axes[0]) of the second panel
            start_time2 = st[1].stats.starttime  
            cb.axvline(start_time2 + 90, panels=[1], axes=[0], color="blue", linewidth=3, alpha=0.8)
            
            # Add another line to spectrogram (axes[1]) of second panel for comparison
            cb.axvline(120, t_units="seconds", panels=[1], axes=[1], color="green", linewidth=2, alpha=0.8)
            
            # plt.suptitle('Targeted axvline Examples:\nRed=First Panel Only, Blue=Second Panel Waveform, Green=Second Panel Spectrogram', 
            #             fontsize=14)
            plt.show()
            print("  ✓ Targeted axvline functionality successful")
            print("    - Red line: Only on first panel (both waveform and spectrogram)")
            print("    - Blue line: Only on second panel waveform")  
            print("    - Green line: Only on second panel spectrogram")
        except Exception as e:
            print(f"  ✗ Targeted axvline functionality failed: {e}")
    else:
        print("\n3. Skipping targeted axvline test - need at least 2 traces")


# Keep the original functions for backward compatibility
def test_single_plots():
    """Test individual plotting functions."""
    st = get_real_data()
    test_single_plots_with_data(st)


def test_panel_plots():
    """Test Panel (waveform + spectrogram) functionality."""
    st = get_real_data()
    test_panel_plots_with_data(st)


def test_clipboard_plots():
    """Test Clipboard (multiple traces) functionality."""
    st = get_real_data()
    if len(st) >= 2:
        test_clipboard_plots_with_data(st)
    else:
        print("  ⚠ Need at least 2 traces for clipboard tests")
        print("  ⚠ Skipping clipboard tests - not enough traces available")


def test_advanced_features():
    """Test advanced features and direct class usage."""
    st = get_real_data()
    test_advanced_features_with_data(st)


def test_clipboard_modes():
    """
    Example demonstrating different Clipboard modes.
    
    Shows how to create clipboards with only waveforms, only spectrograms,
    or both (default).
    """
    print("📊 Clipboard Modes Example")
    print("=" * 35)
    
    try:
        # Get real data
        st = get_real_data()
        
        if len(st) < 2:
            print("⚠️  Need at least 2 traces for this example")
            return
            
        # Use first 2 traces
        test_stream = st[:2]
        print(f"Testing with traces: {[tr.id for tr in test_stream]}")
        
        # Example 1: Waveforms only
        print("\n1. Creating Clipboard with waveforms only (mode='w')")
        cb_w = Clipboard(test_stream, mode="w", figsize=(12, 6), tick_type="seconds")
        cb_w.axvline(60, t_units="seconds", color="red", linewidth=2)
        plt.show()
        print("  ✓ Waveform-only clipboard successful")
        
        # Example 2: Spectrograms only  
        print("\n2. Creating Clipboard with spectrograms only (mode='g')")
        cb_g = Clipboard(test_stream, mode="g", figsize=(12, 8), tick_type="seconds")
        cb_g.axvline(90, t_units="seconds", color="blue", linewidth=2)
        plt.show()
        print("  ✓ Spectrogram-only clipboard successful")
        
        # Example 3: Both (default)
        print("\n3. Creating Clipboard with both waveforms and spectrograms (mode='wg')")
        cb_wg = Clipboard(test_stream, mode="wg", figsize=(12, 10), tick_type="seconds")
        cb_wg.axvline(120, t_units="seconds", color="green", linewidth=2)
        plt.show()
        print("  ✓ Combined waveform+spectrogram clipboard successful")
        
        # Example 4: Using convenience functions (each has implicit mode)
        print("\n4. Using swarmg convenience function (spectrograms only)")
        cb_conv = swarmg(test_stream, figsize=(12, 6), tick_type="minutes")
        cb_conv.axvline(2, t_units="minutes", color="purple", linewidth=2)
        plt.show()
        print("  ✓ swarmg convenience function successful")
        
        print("\n✅ Clipboard modes example completed successfully!")
        print("\nModes demonstrated:")
        print("  mode='w'   # Waveforms only (with trace ID y-labels)")
        print("  mode='g'   # Spectrograms only (with trace ID y-labels)") 
        print("  mode='wg'  # Both waveforms and spectrograms (spectrogram gets y-labels)")
        print("\nNote: Y-axis labels show trace IDs for easy identification in all modes!")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


def test_multicomponent_plotting():
    """
    Example demonstrating multicomponent plotting with plot_trace() and plot_horizontals().
    
    Shows how to plot Z components and overlay N/E components underneath.
    """
    print("🔄 Multicomponent Plotting Example")
    print("=" * 40)
    
    try:
        # Get real data
        st = get_real_data()
        
        if len(st) < 1:
            print("⚠️  Need at least 1 trace for this example")
            return
        
        # Create synthetic multicomponent data from the real data
        # (Since we might not have real 3-component data)
        z_stream = st.copy()
        n_stream = st.copy()
        e_stream = st.copy()
        
        # Modify channel codes to simulate Z, N, E components
        for i, tr in enumerate(z_stream):
            tr.stats.channel = tr.stats.channel[:-1] + "Z"  # End with Z
            tr.id = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}"
            
        for i, tr in enumerate(n_stream):
            tr.stats.channel = tr.stats.channel[:-1] + "N"  # End with N
            tr.id = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}"
            # Scale and shift to make it look different
            tr.data = tr.data * 0.6 + tr.data.std() * 0.3
            
        for i, tr in enumerate(e_stream):
            tr.stats.channel = tr.stats.channel[:-1] + "E"  # End with E
            tr.id = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}"
            # Scale and shift to make it look different
            tr.data = tr.data * 0.4 - tr.data.std() * 0.2
        
        # Combine all components
        full_stream = z_stream + n_stream + e_stream
        
        # Use first 2 stations to keep plots manageable
        z_test = z_stream[:2]
        full_test = Stream()
        for tr in full_stream:
            if any(tr.stats.station == ztr.stats.station for ztr in z_test):
                full_test.append(tr)
        
        print(f"Testing with {len(z_test)} Z components and {len(full_test)} total components")
        for tr in z_test:
            print(f"  Z: {tr.id}")
        
        # Example 1: Manual multicomponent plotting
        print("\n1. Manual multicomponent plotting with plot_trace()")
        cb1 = swarmw(z_test, figsize=(12, 6), tick_type="seconds")
        cb1.plot_trace(n_stream, color="red", alpha=0.6, zorder=-1, linewidth=0.8)
        cb1.plot_trace(e_stream, color="blue", alpha=0.6, zorder=-1, linewidth=0.8)
        cb1.axvline(60, t_units="seconds", color="black", linewidth=2)
        plt.show()
        print("  ✓ Manual multicomponent plotting successful")
        
        # Example 2: Convenience method
        print("\n2. Convenience multicomponent plotting with plot_horizontals()")
        cb2 = swarmw(z_test, figsize=(12, 6), tick_type="seconds")
        cb2.plot_horizontals(full_test, color="gray", alpha=0.7)
        cb2.axvline(90, t_units="seconds", color="red", linewidth=2)
        plt.show()
        print("  ✓ plot_horizontals() convenience method successful")
        
        # Example 3: With spectrograms (mode="wg")
        print("\n3. Multicomponent with waveform + spectrogram")
        cb3 = swarmwg(z_test, figsize=(12, 8), tick_type="minutes")
        cb3.plot_horizontals(full_test, axes=[0], color="darkgray", alpha=0.5)  # Only on waveforms
        cb3.axvline(2, t_units="minutes", color="green", linewidth=2)
        plt.show()
        print("  ✓ Multicomponent with spectrograms successful")
        
        # Example 4: Station-specific targeting
        print("\n4. Station-specific multicomponent plotting")
        cb4 = swarmw(z_test, figsize=(12, 6), tick_type="seconds")
        
        # Plot N/E only on first station
        first_station = z_test[0].stats.station
        cb4.plot_trace(n_stream, stations=[first_station], color="orange", alpha=0.8, zorder=-1)
        cb4.plot_trace(e_stream, stations=[first_station], color="purple", alpha=0.8, zorder=-1)
        cb4.axvline(120, t_units="seconds", color="black", linewidth=2)
        plt.show()
        print(f"  ✓ Station-specific plotting on {first_station} successful")
        
        # Example 5: Panel-level multicomponent
        print("\n5. Panel-level multicomponent plotting")
        panel = swarmwg(z_test[0], figsize=(10, 6), tick_type="seconds")
        panel.plot_horizontals(full_test, axes=[0], color="brown", alpha=0.6)  # Only on waveform
        panel.axvline(30, t_units="seconds", color="cyan", linewidth=2)
        plt.show()
        print("  ✓ Panel-level multicomponent successful")
        
        print("\n✅ Multicomponent plotting example completed successfully!")
        print("\nFeatures demonstrated:")
        print("  - plot_trace() for manual component overlay")
        print("  - plot_horizontals() convenience method") 
        print("  - Station-specific targeting (automatic)")
        print("  - Time axis synchronization (preserve existing xlim)")
        print("  - Axes-specific plotting (waveforms only)")
        print("  - Panel and Clipboard level methods")
        print("  - Custom colors, alpha, and zorder control")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


def test_plot_catalog():
    """
    Example demonstrating plot_catalog functionality.
    
    Creates a synthetic catalog and demonstrates plotting picks and origins
    with different targeting methods.
    """
    print("📊 plot_catalog Example")
    print("=" * 30)
    
    try:
        # Get real data
        st = get_real_data()
        
        if len(st) < 2:
            print("⚠️  Need at least 2 traces for this example")
            return
            
        # Create clipboard with first 2 traces
        print(f"Creating Clipboard with traces: {st[0].id} and {st[1].id}")
        cb = Clipboard(st[:2], sync_waves=False, tick_type="seconds", 
                      figsize=(14, 10), panel_spacing=0.03)
        
        # Create synthetic catalog for demonstration
        from obspy import Catalog, Event, Origin, Pick, WaveformStreamID, UTCDateTime
        
        catalog = Catalog()
        
        # Create an event with origin and picks
        event = Event()
        
        # Add origin
        origin = Origin()
        origin.time = st[0].stats.starttime + 120  # 2 minutes after start
        event.origins = [origin]
        
        # Add P picks for both stations
        for i, trace in enumerate(st[:2]):
            # P pick
            p_pick = Pick()
            p_pick.time = origin.time + 5 + i * 2  # P arrives 5-7 seconds after origin
            p_pick.phase_hint = "P"
            p_pick.waveform_id = WaveformStreamID(
                network_code=trace.stats.network,
                station_code=trace.stats.station,
                location_code=trace.stats.location,
                channel_code=trace.stats.channel
            )
            event.picks.append(p_pick)
            
            # S pick
            s_pick = Pick()
            s_pick.time = origin.time + 12 + i * 3  # S arrives 12-15 seconds after origin
            s_pick.phase_hint = "S"
            s_pick.waveform_id = WaveformStreamID(
                network_code=trace.stats.network,
                station_code=trace.stats.station,
                location_code=trace.stats.location,
                channel_code=trace.stats.channel
            )
            event.picks.append(s_pick)
        
        catalog.append(event)
        
        print(f"\nCreated synthetic catalog with {len(catalog)} events")
        print(f"Event has {len(event.picks)} picks for stations: {[p.waveform_id.station_code for p in event.picks]}")
        
        # Example 1: Plot catalog on all panels
        print("\n1. Plotting catalog on all panels (origins + picks)")
        cb.plot_catalog(catalog, verbose=True, linewidth=2, alpha=0.8)
        
        # Example 2: Plot single event (automatic conversion from Event to Catalog)
        print("\n2. Plotting single event directly (Event -> Catalog conversion)")
        cb.plot_catalog(event, plot_origins=False, p_color="orange", s_color="purple", 
                       linewidth=3, alpha=0.6, linestyle=":")
        
        # Example 3: Plot only picks on specific station
        first_station = st[0].stats.station
        print(f"\n3. Adding picks only to {first_station} station")
        cb.plot_catalog(catalog, stations=[first_station], plot_origins=False, 
                       p_color="cyan", s_color="magenta", linewidth=1, alpha=0.9)
        
        plt.show()
        
        print("\n✅ plot_catalog example completed successfully!")
        print("\nKey syntax demonstrated:")
        print("  cb.plot_catalog(catalog)                              # All panels, all picks")
        print("  cb.plot_catalog(event)                                # Single event (auto-converted)")
        print("  cb.plot_catalog(catalog, stations=['ANMO'])           # Specific station only")
        print("  cb.plot_catalog(catalog, plot_origins=False)          # Picks only, no origins")
        print("  cb.plot_catalog(catalog, axes=[1])                    # Only spectrograms")
        print("  cb.plot_catalog(catalog, p_color='red', s_color='blue')  # Custom colors")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


def test_targeted_axvlines():
    """
    Standalone example demonstrating targeted axvline functionality.
    
    This function creates a Clipboard with multiple traces and shows how to:
    1. Add axvlines using numeric targeting (panels=[0])
    2. Add axvlines using metadata targeting (stations=["ANMO"])
    3. Add axvlines to specific axes within targeted panels
    """
    print("🎯 Targeted axvline Example (Numeric + Metadata)")
    print("=" * 55)
    
    try:
        # Get real data
        st = get_real_data()
        
        if len(st) < 2:
            print("⚠️  Need at least 2 traces for this example")
            return
            
        # Create clipboard with first 2 traces
        print(f"Creating Clipboard with traces: {st[0].id} and {st[1].id}")
        cb = Clipboard(st[:2], sync_waves=False, tick_type="seconds", 
                      figsize=(14, 10), panel_spacing=0.03)
        
        # Print metadata for each panel
        print("\nPanel metadata:")
        for i, panel in enumerate(cb.panels):
            print(f"  Panel {i}: station={panel.metadata.get('station')}, network={panel.metadata.get('network')}, id={panel.metadata.get('id')}")
        
        # Example 1: Numeric targeting - Add axvlines only to the first Panel
        print("\n1. Adding RED line to first panel only (numeric: panels=[0])")
        start_time = st[0].stats.starttime
        cb.axvline(start_time + 60, panels=[0], color="red", linewidth=3, alpha=0.8)
        
        # Example 2: Metadata targeting - Add axvline by station name
        first_station = st[0].stats.station
        print(f"2. Adding ORANGE line by station name (stations=['{first_station}'])")
        cb.axvline(start_time + 30, stations=[first_station], color="orange", linewidth=2, alpha=0.8)
        
        # Example 3: Mixed targeting - Add axvline to specific axes of second panel by index
        print("3. Adding BLUE line to second panel waveform only (panels=[1], axes=[0])")
        start_time2 = st[1].stats.starttime  
        cb.axvline(start_time2 + 90, panels=[1], axes=[0], color="blue", linewidth=3, alpha=0.8)
        
        # Example 4: Metadata + axes targeting - Add to spectrogram of second trace by station
        if len(st) > 1:
            second_station = st[1].stats.station
            print(f"4. Adding GREEN line to {second_station} spectrogram (stations=['{second_station}'], axes=[1])")
            cb.axvline(120, t_units="seconds", stations=[second_station], axes=[1], color="green", linewidth=2, alpha=0.8)
        
        # Add title and show
        # plt.suptitle('Targeted axvline Examples:\n' + 
        #             'Red=First Panel (both axes), Blue=Second Panel Waveform, Green=Second Panel Spectrogram',
        #             fontsize=14)
        plt.show()
        
        print("\n✅ Example completed successfully!")
        print("\nKey syntax demonstrated:")
        print("  # Numeric targeting:")
        print("  cb.axvline(time, panels=[0])              # First panel, both axes")
        print("  cb.axvline(time, panels=[1], axes=[0])    # Second panel, waveform only")
        print("  cb.axvline(time, panels=[1], axes=[1])    # Second panel, spectrogram only")
        print("  # Metadata targeting:")
        print("  cb.axvline(time, stations=['ANMO'])       # All ANMO panels")
        print("  cb.axvline(time, networks=['IU'])         # All IU network panels") 
        print("  cb.axvline(time, ids=['IU.ANMO.00.BHZ'])  # Specific trace ID")
        print("  cb.axvline(time, stations=['CCM'], axes=[1])  # CCM spectrogram only")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
