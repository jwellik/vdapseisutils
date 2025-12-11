"""
Comparison functionality for VCatalog.

This module provides comparison methods for earthquake catalogs including
catalog matching and time-based comparisons.
"""

import warnings
from obspy import UTCDateTime


class VCatalogComparisonMixin:
    """Mixin providing comparison functionality for VCatalog."""
    
    @staticmethod
    def compare_times(times1, times2, threshold_seconds=5):
        """
        Compares two lists of datetime objects and finds matching events within a given threshold.

        Parameters
        ----------
        times1, times2 : list
            datetime or UTCDateTime objects to check
        threshold_seconds : int or float, default 5
            Maximum time difference in seconds to consider as a match

        Returns
        -------
        matching_times
            list of datetimes from times1 that have a match in times2
        index_matching_1
            Indices of matching times in times1
        index_matching_2
            Indices of matching times in times2
        """
        # Ensure datetime objects
        times1 = [UTCDateTime(t).datetime for t in times1]
        times2 = [UTCDateTime(t).datetime for t in times2]
        
        matching_times = []
        index_matching_1 = []
        index_matching_2 = []
        
        for idx1, t1 in enumerate(times1):
            for idx2, t2 in enumerate(times2):
                time_diff = abs((t1 - t2).total_seconds())
                if time_diff <= threshold_seconds:
                    matching_times.append(t1)
                    index_matching_1.append(idx1)
                    index_matching_2.append(idx2)
                    break  # Found a match for this time, move to next
        
        return matching_times, index_matching_1, index_matching_2

    @staticmethod
    def compare_catalogs(cat1, cat2, threshold_seconds=5):
        """
        Compares two VCatalog or obspy.Catalog objects and finds matching events within a given threshold
        (Does the same thing as compare_times, but extracts origin times from events in cat1 and cat2)

        Parameters
        ----------
        cat1, cat2 : VCatalog or obspy.Catalog
            Catalogs to compare. If not VCatalog, will be converted
        threshold_seconds : int or float, default 5
            Maximum time difference in seconds to consider as a match

        Returns
        -------
        matching_cat : VCatalog
            Catalog of events from cat1 that have a match in cat2
        index_matching_1
            Indices of matching events in cat1
        index_matching_2
            Indices of matching events in cat2
        """
        # Convert to VCatalog if needed
        if not hasattr(cat1, 'events'):
            cat1 = VCatalogComparisonMixin._convert_to_catalog(cat1)
        if not hasattr(cat2, 'events'):
            cat2 = VCatalogComparisonMixin._convert_to_catalog(cat2)
        
        # Get origin times
        times1 = VCatalogComparisonMixin._extract_times_from_catalog(cat1)
        times2 = VCatalogComparisonMixin._extract_times_from_catalog(cat2)
        
        # Find matches
        matching_times, index_matching_1, index_matching_2 = VCatalogComparisonMixin.compare_times(
            times1, times2, threshold_seconds=threshold_seconds
        )
        
        # Build matching catalog
        matching_events = [cat1[i] for i in index_matching_1]
        matching_cat = VCatalogComparisonMixin._create_catalog_from_events(matching_events)
        
        return matching_cat, index_matching_1, index_matching_2

    def compare(self, reference, threshold_seconds=5):
        """
        Compares VCatalog against another VCatalog, obspy.Catalog, or list of datetime or UTCDateTime objects and finds matching events within a given threshold

        Parameters
        ----------
        reference : VCatalog, obspy.Catalog, or list
            If VCatalog or Catalog: compare events between catalogs
            If list: compare events to the list of times
        threshold_seconds : int or float, default 5
            Maximum time difference in seconds to consider as a match

        Returns
        -------
        matching_cat : VCatalog
            Catalog of events from cat1 that have a match in reference
        index_matching_1
            Indices of matching events in cat1
        index_matching_2
            Indices of matching times in references (catalog events or times)
        """
        # Check if reference is a catalog-like object
        if hasattr(reference, 'events'):
            # It's a catalog - use catalog comparison
            return self.__class__.compare_catalogs(self, reference, threshold_seconds=threshold_seconds)
        
        # Check if reference is a list/tuple of times
        elif isinstance(reference, (list, tuple)):
            # It's a list of times - use time comparison
            self_times = self.__class__._extract_times_from_catalog(self)
            matching_times, index_matching_1, index_matching_2 = self.__class__.compare_times(
                self_times, reference, threshold_seconds=threshold_seconds
            )
            
            # Build matching catalog
            matching_events = [self[i] for i in index_matching_1]
            matching_cat = self.__class__._create_catalog_from_events(matching_events)
            
            return matching_cat, index_matching_1, index_matching_2
        
        else:
            raise TypeError(f"Cannot compare VCatalog with {type(reference)}. "
                          f"Expected VCatalog, Catalog, or list of times.")

    @staticmethod
    def find_matching_times(times1, times2, threshold_seconds=5):
        """
        For each datetime in times1, find indices of all times2 that are within specified threshold. 
        For each datetime in times2, find all indices of time1 that are within specified threshold.

        Parameters
        ----------
        times1: list
            List of datetime or UTCDateTime objects to check
        times2: list
            List of datetime or UTCDateTime objects to check against
        threshold_seconds : int or float, default 5
            Maximum time difference in seconds to consider as a match

        Returns
        -------
        list of lists
            For each datetime in times1, a list of indices from times2 that are within the threshold. 
            If no matches are found for a datetime, an empty list is returned
        """
        # Ensure datetime objects
        times1 = [UTCDateTime(t).datetime for t in times1]
        times2 = [UTCDateTime(t).datetime for t in times2]
        
        results = []
        for t1 in times1:
            matching_indices = []
            for idx, t2 in enumerate(times2):
                time_diff = abs((t1 - t2).total_seconds())
                if time_diff <= threshold_seconds:
                    matching_indices.append(idx)
            results.append(matching_indices)
        
        return results

    @staticmethod
    def find_matching_events(cat1, cat2, threshold_seconds=5):
        """
        Does the same thing as find_matching_times, but extracts origin time from each Event in Catalog

        Parameters
        ----------
        cat1, cat2 : VCatalog or obspy.Catalog
            Catalogs to compare. If not VCatalog, will be converted
        threshold_seconds : int or float, default 5
            Maximum time difference in seconds to consider as a match

        Returns
        -------
        list of lists
            For each event in cat1, a list of indices from cat2 that are within the threshold. 
            If no matches are found for an event, an empty list is returned
        """
        # Convert to VCatalog if needed
        if not hasattr(cat1, 'events'):
            cat1 = VCatalogComparisonMixin._convert_to_catalog(cat1)
        if not hasattr(cat2, 'events'):
            cat2 = VCatalogComparisonMixin._convert_to_catalog(cat2)
        
        # Get origin times
        times1 = VCatalogComparisonMixin._extract_times_from_catalog(cat1)
        times2 = VCatalogComparisonMixin._extract_times_from_catalog(cat2)
        
        # Find matches
        return VCatalogComparisonMixin.find_matching_times(times1, times2, threshold_seconds=threshold_seconds)

    def find_matching(self, reference, threshold_seconds=5):
        """
        Compares VCatalog against another VCatalog, obspy.Catalog, or list of datetimes or UTCDateTime objects that are within a given threshold

        Parameters
        ----------
        reference : VCatalog, obspy.Catalog, or list
            If VCatalog or Catalog: compare events between catalogs
            If list: compare events to the list of times
        threshold_seconds : int or float, default 5
            Maximum time difference in seconds to consider as a match

        Returns
        -------
        list of lists
            For each event in self, a list of indices from reference that are within the threshold. 
            If no matches are found for a datetime, an empty list is returned
        """
        # Check if reference is a catalog-like object
        if hasattr(reference, 'events'):
            # It's a catalog - use event matching
            return self.__class__.find_matching_events(self, reference, threshold_seconds=threshold_seconds)
        
        # Check if reference is a list/tuple of times
        elif isinstance(reference, (list, tuple)):
            # It's a list of times - use time matching
            self_times = self.__class__._extract_times_from_catalog(self)
            return self.__class__.find_matching_times(self_times, reference, threshold_seconds=threshold_seconds)
        
        else:
            raise TypeError(f"Cannot find matching in VCatalog with {type(reference)}. "
                          f"Expected VCatalog, Catalog, or list of times.")

    # Private helper methods
    @staticmethod
    def _extract_times_from_catalog(catalog):
        """Extract origin times from catalog events."""
        times = []
        for event in catalog:
            if event.preferred_origin():
                times.append(event.preferred_origin().time)
            elif len(event.origins) > 0:
                times.append(event.origins[0].time)
        return times

    @staticmethod
    def _convert_to_catalog(obj):
        """Convert an object to a catalog format."""
        # This is a placeholder - you'll need to implement the actual conversion
        # based on your VCatalog implementation
        if hasattr(obj, 'events'):
            return obj
        else:
            raise TypeError(f"Cannot convert {type(obj)} to catalog format")

    @staticmethod
    def _create_catalog_from_events(events):
        """Create a catalog from a list of events."""
        # This is a placeholder - you'll need to implement the actual creation
        # based on your VCatalog implementation
        from obspy import Catalog
        return Catalog(events=events)

    @staticmethod
    def compare_origins(cat1, cat2, threshold_seconds=5, verbose=False):
        """
        Compare origin information between two catalogs and calculate differences.
        
        This method compares origin data between corresponding events in two catalogs.
        It first finds matching events using origin time comparison, then for each matched 
        event pair, it calculates differences in time, location, depth, magnitude, and 
        spatial relationships.
        
        Parameters
        ----------
        cat1 : obspy.core.event.Catalog or obspy.core.event.Event
            First catalog for comparison. If Event, will be converted to single-event Catalog
        cat2 : obspy.core.event.Catalog or obspy.core.event.Event
            Second catalog for comparison. If Event, will be converted to single-event Catalog
        threshold_seconds : int or float, default 5
            Maximum time difference in seconds for matching events by origin time
        verbose : bool, default False
            If True, print detailed information about the comparison process
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing comparison results with columns:
            - idx: Event pair index from matching results (0-based)
            - idx1: Original index of event in cat1
            - idx2: Original index of event in cat2
            - time1: Origin time from cat1
            - time2: Origin time from cat2
            - delta_time: Time difference (cat1 - cat2) in seconds
            - delta_abs_time: Absolute value of delta_time in seconds
            - lat1: Latitude from cat1
            - lat2: Latitude from cat2
            - delta_lat: Latitude difference (cat1 - cat2) in degrees
            - lon1: Longitude from cat1
            - lon2: Longitude from cat2
            - delta_lon: Longitude difference (cat1 - cat2) in degrees
            - depth1: Depth from cat1 in km
            - depth2: Depth from cat2 in km
            - delta_depth: Depth difference (cat1 - cat2) in km
            - mag1: Magnitude from cat1
            - mag2: Magnitude from cat2
            - delta_mag: Magnitude difference (cat1 - cat2)
            - delta_xy: Horizontal distance between events in km
            - az_xy: Azimuth from cat1 to cat2 in degrees (0°=N, 90°=E)
            - delta_xyz: 3D distance between events in km
            - alpha_xyz: Elevation angle from cat1 to cat2 in degrees (positive=cat2 deeper)
            
        Examples
        --------
        >>> df = VCatalogComparisonMixin.compare_origins(manual_catalog, automatic_catalog)
        >>> print(df[['delta_time', 'delta_xy', 'delta_depth']].head())
        
        >>> # With verbose output
        >>> df = VCatalogComparisonMixin.compare_origins(cat1, cat2, verbose=True)
        """
        import pandas as pd
        import numpy as np
        from obspy.core.event import Catalog, Event
        from obspy.geodetics import gps2dist_azimuth
        
        if verbose:
            print(">>> VCatalogComparisonMixin.compare_origins()")
            print(f"Initial catalog sizes: cat1={len(cat1) if hasattr(cat1, '__len__') else 1}, cat2={len(cat2) if hasattr(cat2, '__len__') else 1}")
            print(f"Event matching threshold: {threshold_seconds} seconds")
        
        # Convert single Events to Catalogs if needed
        if isinstance(cat1, Event):
            if verbose:
                print("Converting cat1 from Event to single-event Catalog")
            temp_cat1 = Catalog()
            temp_cat1.append(cat1)
            cat1 = temp_cat1
            
        if isinstance(cat2, Event):
            if verbose:
                print("Converting cat2 from Event to single-event Catalog")
            temp_cat2 = Catalog()
            temp_cat2.append(cat2)
            cat2 = temp_cat2
        
        # Find matching events using compare_catalogs
        if verbose:
            print(f"\nFinding matching events between catalogs...")
        matching_cat, idx1, idx2 = VCatalogComparisonMixin.compare_catalogs(
            cat1, cat2, threshold_seconds=threshold_seconds
        )
        
        if verbose:
            print(f"Found {len(idx1)} matching event pairs out of {len(cat1)} and {len(cat2)} events")
        
        # Initialize results list
        results = []
        events_with_origins = 0
        
        if verbose:
            print(f"\nProcessing {len(idx1)} matched event pairs for origin comparisons...")
        
        # Process each matched event pair
        for pair_idx, (i1, i2) in enumerate(zip(idx1, idx2)):
            event1 = cat1[i1]
            event2 = cat2[i2]
            
            if verbose:
                print(f"\n--- Event pair {pair_idx + 1}/{len(idx1)} (indices {i1}, {i2}) ---")
            
            # Get preferred or first origin for each event
            origin1 = event1.preferred_origin() if event1.preferred_origin() else (event1.origins[0] if event1.origins else None)
            origin2 = event2.preferred_origin() if event2.preferred_origin() else (event2.origins[0] if event2.origins else None)
            
            # Get preferred or first magnitude for each event
            mag1_obj = event1.preferred_magnitude() if event1.preferred_magnitude() else (event1.magnitudes[0] if event1.magnitudes else None)
            mag2_obj = event2.preferred_magnitude() if event2.preferred_magnitude() else (event2.magnitudes[0] if event2.magnitudes else None)
            
            if verbose:
                print(f"Origin 1: {'Found' if origin1 else 'Missing'}")
                print(f"Origin 2: {'Found' if origin2 else 'Missing'}")
                print(f"Magnitude 1: {'Found' if mag1_obj else 'Missing'}")
                print(f"Magnitude 2: {'Found' if mag2_obj else 'Missing'}")
            
            # Extract values or set to NaN if missing
            time1 = origin1.time if origin1 and origin1.time else np.nan
            time2 = origin2.time if origin2 and origin2.time else np.nan
            lat1 = origin1.latitude if origin1 and origin1.latitude is not None else np.nan
            lat2 = origin2.latitude if origin2 and origin2.latitude is not None else np.nan
            lon1 = origin1.longitude if origin1 and origin1.longitude is not None else np.nan
            lon2 = origin2.longitude if origin2 and origin2.longitude is not None else np.nan
            depth1 = origin1.depth / 1000.0 if origin1 and origin1.depth is not None else np.nan  # Convert m to km
            depth2 = origin2.depth / 1000.0 if origin2 and origin2.depth is not None else np.nan  # Convert m to km
            mag1 = mag1_obj.mag if mag1_obj and mag1_obj.mag is not None else np.nan
            mag2 = mag2_obj.mag if mag2_obj and mag2_obj.mag is not None else np.nan
            
            # Calculate differences
            delta_time = float(time1 - time2) if not (pd.isna(time1) or pd.isna(time2)) else np.nan
            delta_lat = lat1 - lat2 if not (np.isnan(lat1) or np.isnan(lat2)) else np.nan
            delta_lon = lon1 - lon2 if not (np.isnan(lon1) or np.isnan(lon2)) else np.nan
            delta_depth = depth1 - depth2 if not (np.isnan(depth1) or np.isnan(depth2)) else np.nan
            delta_mag = mag1 - mag2 if not (np.isnan(mag1) or np.isnan(mag2)) else np.nan
            
            # Calculate spatial relationships
            if not (np.isnan(lat1) or np.isnan(lon1) or np.isnan(lat2) or np.isnan(lon2)):
                # Great circle distance and azimuth
                distance_m, az_forward, az_back = gps2dist_azimuth(lat1, lon1, lat2, lon2)
                delta_xy = distance_m / 1000.0  # Convert m to km
                az_xy = az_forward  # Azimuth from cat1 to cat2
                
                # 3D distance incorporating depth
                if not (np.isnan(depth1) or np.isnan(depth2)):
                    depth_diff = depth2 - depth1  # Positive if cat2 is deeper
                    delta_xyz = np.sqrt(delta_xy**2 + depth_diff**2)
                    # Elevation angle: positive if cat2 is deeper
                    alpha_xyz = np.degrees(np.arctan2(depth_diff, delta_xy))
                else:
                    delta_xyz = delta_xy  # Use horizontal distance if depth missing
                    alpha_xyz = np.nan
            else:
                delta_xy = np.nan
                az_xy = np.nan
                delta_xyz = np.nan
                alpha_xyz = np.nan
            
            if verbose:
                print(f"  Time difference: {delta_time:.3f}s" if not np.isnan(delta_time) else "  Time difference: NaN")
                print(f"  Horizontal distance: {delta_xy:.3f}km" if not np.isnan(delta_xy) else "  Horizontal distance: NaN")
                print(f"  3D distance: {delta_xyz:.3f}km" if not np.isnan(delta_xyz) else "  3D distance: NaN")
                print(f"  Depth difference: {delta_depth:.3f}km" if not np.isnan(delta_depth) else "  Depth difference: NaN")
            
            # Calculate absolute value of delta_time
            delta_abs_time = abs(delta_time) if not np.isnan(delta_time) else np.nan
            
            # Add result
            results.append({
                'idx': pair_idx,
                'idx1': i1,
                'idx2': i2,
                'time1': time1,
                'time2': time2,
                'delta_time': delta_time,
                'delta_abs_time': delta_abs_time,
                'lat1': lat1,
                'lat2': lat2,
                'delta_lat': delta_lat,
                'lon1': lon1,
                'lon2': lon2,
                'delta_lon': delta_lon,
                'depth1': depth1,
                'depth2': depth2,
                'delta_depth': delta_depth,
                'mag1': mag1,
                'mag2': mag2,
                'delta_mag': delta_mag,
                'delta_xy': delta_xy,
                'az_xy': az_xy,
                'delta_xyz': delta_xyz,
                'alpha_xyz': alpha_xyz
            })
            
            events_with_origins += 1
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Ensure column order
        if not df.empty:
            df = df[['idx', 'idx1', 'idx2', 'time1', 'time2', 'delta_time', 'delta_abs_time',
                     'lat1', 'lat2', 'delta_lat',
                     'lon1', 'lon2', 'delta_lon',
                     'depth1', 'depth2', 'delta_depth',
                     'mag1', 'mag2', 'delta_mag',
                     'delta_xy', 'az_xy', 'delta_xyz', 'alpha_xyz']]
        
        # Final summary
        if verbose:
            print(f"\n=== PROCESSING COMPLETE ===")
            print(f"Event pairs processed: {len(idx1)}")
            print(f"Events with origin data: {events_with_origins}")
            if not df.empty:
                print(f"Origin comparison statistics:")
                for col in ['delta_time', 'delta_xy', 'delta_xyz', 'delta_depth', 'delta_mag']:
                    if col in df.columns:
                        valid_data = df[col].dropna()
                        if len(valid_data) > 0:
                            print(f"  {col}: mean={valid_data.mean():.3f}, std={valid_data.std():.3f}, "
                                  f"min={valid_data.min():.3f}, max={valid_data.max():.3f}")
                print(f"DataFrame shape: {df.shape}")
            else:
                print("No origin comparisons found")
        
        return df
