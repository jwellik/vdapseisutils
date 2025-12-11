"""
Conversion functionality for VCatalog.

This module provides conversion methods for earthquake catalogs including
file format conversions and data export/import functionality.
"""

import pandas as pd
from obspy import UTCDateTime
from obspy.core.event import Event, Origin, Magnitude, Catalog
from vdapseisutils.core.datasource.waveID import waveID


class VCatalogConversionMixin:
    """Mixin providing conversion functionality for VCatalog."""
    
    def to_basics(self, *args, **kwargs):
        return self.to_txyzm(*args, **kwargs)

    def to_txyzm(self, depth_unit="km", z_dir="depth", time_format="UTCDateTime", verbose=False, filename=False, **to_csv_kwargs):
        """Returns time(UTCDateTime), lat, lon, depth(kilometers), and mag from ObsPy Catalog object"""
        # Adjust for depth_unit
        if depth_unit == "m":
            dconvert = 1
        elif depth_unit == "km":
            dconvert = 1000
        else:
            if verbose: print("'depth_unit' not understood. Default value 'km' is used.")
            dconvert = 1000
        # Assume depths are depths (positive down); multiply by -1 if given as elevations (positive up)
        if z_dir == "depth":
            z_dir_convert = 1
        elif z_dir == "elev":
            z_dir_convert = -1
        else:
            if verbose: print("'depth_unit' not understood. Default value 'depth' is used.")
            z_dir_convert = 1
        def convert_time_format(utcdatetime):
            if time_format == "UTCDateTime":
                return utcdatetime
            elif time_format == "matplotlib":
                return utcdatetime.matplotlib_date
            elif time_format == "datetime":
                return utcdatetime.datetime
            else:
                return utcdatetime.strftime(time_format)
        time = []
        lat = []
        lon = []
        depth = []
        mag = []
        for event in self:
            try:
                lat.append(event.origins[-1].latitude)
                lon.append(event.origins[-1].longitude)
                depth.append(event.origins[-1].depth / dconvert * z_dir_convert)
                if event.magnitudes:
                    mag.append(event.magnitudes[-1].mag if event.magnitudes[-1].mag is not None else -1)
                else:
                    mag.append(-1)
                time.append(convert_time_format(event.origins[-1].time))
            except Exception as err:
                print(f"Skipping event due to error: {err}")
        data = dict({"time": time, "lat": lat, "lon": lon, "depth": depth, "mag": mag})
        if filename:
            pd.DataFrame(data).to_csv(filename, index=False, **to_csv_kwargs)
            if verbose: print(f"Catalog printed : {filename}")
        return data

    def to_picklog(self, verbose=False):
        pick_list = []
        for event in self.events:
            for pick in event.picks:
                d = dict(pick)
                wsi = d["waveform_id"]
                d["nslc"] = wsi.get_seed_string()
                pick_list.append(d)
        df = pd.DataFrame.from_dict(pick_list)
        return df

    @classmethod
    def from_txyzm(cls, data):
        cat = cls()
        cat.description = "Catalog imported from CSV file via Pandas DataFrame"
        for index, row in data.iterrows():
            e = Event()
            e.event_type = "not existing"
            o = Origin()
            o.time = UTCDateTime(row['time'])
            o.latitude = row['latitude']
            o.longitude = row['longitude']
            o.depth = row['depth']
            o.depth_type = "operator assigned"
            o.evaluation_mode = "manual"
            o.evaluation_status = "preliminary"
            m = Magnitude()
            m.mag = row['mag']
            m.magnitude_type = "Md"
            cat.append(e)
            e.origins = [o]
            e.magnitudes = [m]
            m.origin_id = o.resource_id
        return cat

    @classmethod
    def from_basics(cls, *args, **kwargs):
        return cls.from_txyzm(*args, **kwargs)

    def to_swarm(self, nslc, tags=["default"], filename="swarm_tagger.csv", mode="w"):
        """
        Write the catalog to a Swarm tag CSV file.

        Parameters
        ----------
        nslc : str
            NSLC string to use for all events (or provide a list of same length as catalog).
        tags : list, default ["default"]
            List of tags to assign to each event (or a single tag to repeat).
        filename : str, default "swarm_tagger.csv"
            Output CSV file name.
        mode : str, default "w"
            File mode: "w" for overwrite, "a" for append.
        """
        import csv
        if len(tags) == 1:
            tags = tags * len(self)
        if isinstance(nslc, str):
            nslc = [nslc] * len(self)
        elif len(nslc) == 1:
            nslc = nslc * len(self)
        with open(filename, mode, newline='') as f:
            writer = csv.writer(f)
            for event, nslc_val, tag in zip(self, nslc, tags):
                # Use event origin time
                if event.origins:
                    t = event.origins[0].time.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
                else:
                    continue  # skip events without origin
                scnl = waveID(nslc_val).string(order="scnl", sep=" ")
                writer.writerow([t, scnl, tag])

    @staticmethod
    def read_swarm_tags(swarm_tag_file, scnl_format="scnl"):
        """
        Read a Swarm tag CSV file and return a pandas DataFrame.

        Parameters
        ----------
        swarm_tag_file : str
            Path to the Swarm tag CSV file.
        scnl_format : str, default "scnl"
            Format of the SCNL string (not currently used).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ["time", "scnl", "tag"].
        """
        df = pd.read_csv(swarm_tag_file, header=None, names=["time", "scnl", "tag"])
        # Handle mixed timestamp formats (with and without microseconds)
        df["time"] = pd.to_datetime(df["time"], format='mixed', dayfirst=False)
        return df

    @classmethod
    def from_swarm(cls, swarm_tag_file, scnl_format="scnl"):
        """
        Create a VCatalog from a Swarm tag CSV file.

        Parameters
        ----------
        swarm_tag_file : str
            Path to the Swarm tag CSV file.
        scnl_format : str, default "scnl"
            Format of the SCNL string (not currently used).

        Returns
        -------
        VCatalog
            Catalog with events for each row in the Swarm tag file. Each event will have time and SCNL as attributes.
        """
        from obspy.core.event import Event, Origin
        df = cls.read_swarm_tags(swarm_tag_file, scnl_format=scnl_format)
        cat = cls()
        cat.description = f"Catalog imported from Swarm tag file: {swarm_tag_file}"
        for _, row in df.iterrows():
            e = Event()
            o = Origin()
            o.time = UTCDateTime(row['time'])
            # Optionally, you could parse SCNL and add as custom attributes or comments
            e.origins = [o]
            # Store SCNL/tag as event comments for traceability
            e.comments = [f"SCNL: {row['scnl']}, tag: {row['tag']}"]
            cat.append(e)
        return cat

    @classmethod
    def from_pyocto(cls, events, assignments):
        """
        Create a VCatalog from PyOcto events (DataFrame) and assignments (DataFrame).

        Parameters
        ----------
        events : DataFrame
            cols: idx,time,x,y,z,picks,latitude,longitude,depth
        assignments : str
            cols: event_idx, pick_idx, residual, station, time, probability, phase

        Returns
        -------
        VCatalog
            VCatalog with Events
        """

        import obspy
        from obspy.core.event import Catalog, Event, Origin, Pick, Arrival, Magnitude
        from obspy.core import UTCDateTime
        import numpy as np

        # Create an empty catalog
        catalog = obspy.Catalog()

        # Convert events and assignments to ObsPy Catalog
        for _, event_row in events.iterrows():
            # Create new Event object
            event = Event()

            # Create Origin object with location and time information
            origin = Origin()
            origin.time = UTCDateTime(event_row['time'])
            origin.latitude = event_row['latitude']
            origin.longitude = event_row['longitude']
            origin.depth = event_row['depth'] * 1000  # Convert km to meters for ObsPy

            # Get all picks associated with this event
            event_picks = assignments[assignments['event_idx'] == event_row['idx']]

            # Create Pick and Arrival objects for each pick
            picks = []
            arrivals = []

            for _, pick_row in event_picks.iterrows():
                # Create Pick object
                pick = Pick()
                pick.time = UTCDateTime(pick_row['time'])

                pick.waveform_id = obspy.core.event.WaveformStreamID(
                    network_code=pick_row['station'].split('.')[0],
                    station_code=pick_row['station'].split('.')[1]
                )
                pick.phase_hint = pick_row['phase']
                pick.evaluation_mode = 'automatic'

                # Create Arrival object linking pick to origin
                arrival = Arrival()
                arrival.pick_id = pick.resource_id
                arrival.phase = pick_row['phase']
                arrival.time_residual = pick_row['residual']
                arrival.distance = np.sqrt(
                    (event_row['x'] - 0) ** 2 + (event_row['y'] - 0) ** 2 + (event_row['z']) ** 2
                ) * 1000  # Convert km to meters

                picks.append(pick)
                arrivals.append(arrival)

            # Add picks to event
            event.picks = picks

            # Add arrivals to origin
            origin.arrivals = arrivals
            origin.quality = obspy.core.event.OriginQuality()
            origin.quality.used_phase_count = len(arrivals)

            # Add origin to event
            event.origins = [origin]
            event.preferred_origin_id = origin.resource_id

            # Add event to catalog
            catalog.append(event)

        return cls(catalog)

    @staticmethod
    def times2swarm(times, scnl, tag, sort=False, filename="swarm_tagger.csv", overwrite=True):
        """
        Convert a list of times to a Swarm tag CSV file.

        Parameters
        ----------
        times : list
            List of datetime objects or UTCDateTime objects.
        scnl : str
            SCNL string to use for all times.
        tag : str
            Tag to assign to all times.
        sort : bool, default False
            Whether to sort times before writing.
        filename : str, default "swarm_tagger.csv"
            Output CSV file name.
        overwrite : bool, default True
            Whether to overwrite existing file.

        Returns
        -------
        str
            Path to the created CSV file.
        """
        import csv
        mode = "w" if overwrite else "a"
        if sort:
            times = sorted(times)
        with open(filename, mode, newline='') as f:
            writer = csv.writer(f)
            for t in times:
                if hasattr(t, 'datetime'):
                    t_str = t.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
                else:
                    t_str = t.strftime("%Y-%m-%d %H:%M:%S.%f")
                scnl_str = waveID(scnl).string(order="scnl", sep=" ")
                writer.writerow([t_str, scnl_str, tag])
        return filename

    def write_nlloc_obs(self, directory, **kwargs):
        """
        Write NLLOC_OBS files for each event in the catalog to the specified directory.
        
        This method writes one NLLOC_OBS file per event, with filenames following the format:
        NLL.yyyymmddHHMMSSffff.eventid.obs
        
        The eventid is extracted from the Event object's resource_id if available,
        otherwise falls back to a prefixed index (e.g., "id000123").
        
        Parameters
        ----------
        directory : str
            Directory path where files should be written
        **kwargs
            Additional arguments passed to the ObsPy write method
            
        Returns
        -------
        list
            List of written file paths
        """
        import os
        from obspy import Catalog
        
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        written_files = []

        for i, event in enumerate(self):
            # Extract event ID, fallback to prefixed index if not found
            event_id = self.extract_event_id(event)
            if event_id is None:
                event_id = f"id{i:06d}"
            
            # Generate filename based on event time
            if event.origins:
                origin_time = event.origins[0].time
                # Format: NLL.yyyymmddHHMMSSffff.obs
                time_str = origin_time.strftime("%Y%m%d%H%M%S%f")[:-3]  # Remove microseconds, keep milliseconds
                filename = f"NLL.{time_str}.obs"
            else:
                # Fallback if no origin time
                filename = f"NLL.unknown.{event_id}.obs"
            
            # Create full file path
            filepath = os.path.join(directory, filename)
            
            try:
                # Create single-event catalog and write it
                single_event_catalog = Catalog(events=[event])
                # Don't pass format in kwargs since we're explicitly setting it
                write_kwargs = {k: v for k, v in kwargs.items() if k != 'format'}
                single_event_catalog.write(filepath, format="NLLOC_OBS", **write_kwargs)
                written_files.append(filepath)
            except Exception as e:
                print(f"Warning: Failed to write event {i} to {filepath}: {e}")
                # Continue with other events
                continue
        
        return written_files
