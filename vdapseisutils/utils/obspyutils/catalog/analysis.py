"""
Analysis functionality for VCatalog.

This module provides analysis methods for earthquake catalogs including
event rate analysis and statistics.
"""

import pandas as pd
from datetime import datetime


class VCatalogAnalysisMixin:
    """Mixin providing analysis functionality for VCatalog."""
    
    def print_eventrate(self, freq="1D", top_n=None, sort_by="count",
                        ascending=None, date_format=None, show_stats=True):
        """
        Print a nicely formatted table of event rates.

        Parameters
        ----------
        freq : str, optional
            Frequency string for binning (default "1D" for daily)
        top_n : int, optional
            Show only top N periods by event count. If None, shows all periods
        sort_by : str, optional
            Column to sort by ("count", "start_time", "end_time")
        ascending : bool, optional
            Sort order. If None, defaults to False for count, True for times
        date_format : str, optional
            Date format string. If None, uses default format
        show_stats : bool, optional
            Whether to show summary statistics at the end
        """
        return self._print_eventrate_impl(self, freq, top_n, sort_by, ascending, date_format, show_stats)

    @staticmethod
    def _print_eventrate_impl(catalog, freq="1D", top_n=None, sort_by="count",
                              ascending=None, date_format=None, show_stats=True):
        """Implementation for print_eventrate (works as static method too)."""
        # Extract origin times from all events in the catalog
        origin_times = []
        for event in catalog:
            if event.preferred_origin():
                origin_times.append(event.preferred_origin().time.datetime)
            elif len(event.origins) > 0:
                origin_times.append(event.origins[0].time.datetime)

        if not origin_times:
            print("No events with origin times found in catalog")
            return

        # Convert to pandas datetime index and resample to get counts per time period
        time_series = pd.Series(1, index=pd.DatetimeIndex(origin_times))
        counts = time_series.resample(freq).count()

        # Create DataFrame with start and end times for each period
        df = pd.DataFrame({
            'start_time': counts.index,
            'end_time': counts.index + pd.Timedelta(freq),
            'count': counts.values
        })

        # Remove periods with zero events
        df = df[df['count'] > 0].copy()

        if df.empty:
            print("No events found in the specified time periods")
            return

        # Set default sort order
        if ascending is None:
            ascending = False if sort_by == "count" else True

        # Sort the DataFrame
        df = df.sort_values(sort_by, ascending=ascending)

        # Limit to top N if specified
        if top_n is not None:
            df = df.head(top_n)

        # Set date format
        if date_format is None:
            date_format = "%Y-%m-%d %H:%M:%S"

        # Print the table
        print(f"\nEvent Rate Analysis (freq={freq})")
        print("=" * 80)
        print(f"{'Start Time':<20} {'End Time':<20} {'Count':<8}")
        print("-" * 80)

        for _, row in df.iterrows():
            start_str = row['start_time'].strftime(date_format)
            end_str = row['end_time'].strftime(date_format)
            print(f"{start_str:<20} {end_str:<20} {row['count']:<8}")

        # Show statistics if requested
        if show_stats:
            print("-" * 80)
            print(f"Total events: {len(origin_times)}")
            print(f"Time span: {min(origin_times)} to {max(origin_times)}")
            print(f"Periods with events: {len(df)}")
            print(f"Average events per period: {df['count'].mean():.2f}")
            print(f"Max events in a period: {df['count'].max()}")
            print(f"Min events in a period: {df['count'].min()}")

    @classmethod
    def print_eventrate_from_catalog(cls, catalog, freq="1D", top_n=None, sort_by="count",
                                     ascending=None, date_format=None, show_stats=True):
        """
        Class method to print event rate from any catalog object.

        Parameters
        ----------
        catalog : obspy.Catalog
            Catalog object to analyze
        freq : str, optional
            Frequency string for binning (default "1D" for daily)
        top_n : int, optional
            Show only top N periods by event count. If None, shows all periods
        sort_by : str, optional
            Column to sort by ("count", "start_time", "end_time")
        ascending : bool, optional
            Sort order. If None, defaults to False for count, True for times
        date_format : str, optional
            Date format string. If None, uses default format
        show_stats : bool, optional
            Whether to show summary statistics at the end
        """
        return cls._print_eventrate_impl(catalog, freq, top_n, sort_by, ascending, date_format, show_stats)
