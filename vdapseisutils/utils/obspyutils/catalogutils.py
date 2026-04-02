import pandas as pd

from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Magnitude
from obspy.geodetics import FlinnEngdahl

# from vdapseisutils.utils.obspyutils import hypoinverse
# from vdapseisutils.utils.obspyutils.inventoryutils import convertNSLCstr
from vdapseisutils.obspy_ext import VStreamID

def example():
    cat = Catalog()
    cat.description = "Just a fictitious toy example catalog built from scratch"

    e = Event()
    e.event_type = "not existing"

    o = Origin()
    o.time = UTCDateTime(2014, 2, 23, 18, 0, 0)
    o.latitude = 47.6
    o.longitude = 12.0
    o.depth = 10000
    o.depth_type = "operator assigned"
    o.evaluation_mode = "manual"
    o.evaluation_status = "preliminary"
    o.region = FlinnEngdahl().get_region(o.longitude, o.latitude)

    m = Magnitude()
    m.mag = 7.2
    m.magnitude_type = "Mw"

    m2 = Magnitude()
    m2.mag = 7.4
    m2.magnitude_type = "Ms"

    # also included could be: custom picks, amplitude measurements, station magnitudes,
    # focal mechanisms, moment tensors, ...

    # make associations, put everything together
    cat.append(e)
    e.origins = [o]
    e.magnitudes = [m, m2]
    m.origin_id = o.resource_id
    m2.origin_id = o.resource_id

    print(cat)
    # cat.write("/tmp/my_custom_events.xml", format="QUAKEML")
    # !cat / tmp / my_custom_events.xml

########################################################################################################################
# Read custom formats
########################################################################################################################

# def read_csv(file, rename_dict=None, read_csv_kwargs=**kwargs)
def read_victoria_csv(file):
    """https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html"""

    print('Read Basic CSV file into ObsPy Catalog object')

    df = pd.read_csv(file, parse_dates=[['date', 'HH:MM']], infer_datetime_format=True, header=0,
                     # names=['time', 'lat', 'lon', 'depth', 'mag'],
                     )
    df['time'] = df['date_HH:MM']
    df = df.drop(columns=['date_HH:MM'])
    df['depth'] = df['depth(km)'] * 1000
    df = df.drop(columns=['depth(km)'])
    df['mag'] = df['ML']
    df = df.drop(columns=['ML'])
    df['lat'] *= -1  # Lat & Lon are not referenced to hemisphere
    df['lon'] *= -1
    print(df)
    print()

    cat = Catalog()
    cat.description = "Catalog imported from CSV file via Pandas DataFrame"
    for index, row in df.iterrows():
        e = Event()
        e.event_type = "not existing"

        o = Origin()
        o.time = UTCDateTime(row['time'])
        o.latitude = row['lat']
        o.longitude = row['lon']
        o.depth = row['depth']
        o.depth_type = "operator assigned"
        o.evaluation_mode = "manual"
        o.evaluation_status = "preliminary"
        # o.region = FlinnEngdahl().get_region(o.longitude, o.latitude)

        m = Magnitude()
        m.mag = row['mag']
        m.magnitude_type = "ML"

        # also included could be: custom picks, amplitude measurements, station magnitudes,
        # focal mechanisms, moment tensors, ...

        # make associations, put everything together
        cat.append(e)
        e.origins = [o]
        e.magnitudes = [m]
        m.origin_id = o.resource_id

    print(cat)
    return cat

########################################################################################################################
# Earthworm catalog files
########################################################################################################################



########################################################################################################################
# Catalog manipulation
########################################################################################################################

def sort_catalog(catalog, key='magnitude', reverse=False):
    """
    Sort an ObsPy Catalog object by various criteria.

    Parameters:
    -----------
    catalog : obspy.Catalog
        The catalog to sort
    key : str or callable
        Sorting criteria. Options:
        - 'magnitude': Sort by magnitude (uses first magnitude if multiple)
        - 'time': Sort by origin time (uses first origin if multiple)
        - 'depth': Sort by depth (uses first origin if multiple)
        - 'latitude': Sort by latitude (uses first origin if multiple)
        - 'longitude': Sort by longitude (uses first origin if multiple)
        - callable: Custom function that takes an Event and returns a sortable value
    reverse : bool
        If True, sort in descending order. Default is False (ascending)

    Returns:
    --------
    obspy.Catalog
        A new sorted catalog
    """

    def get_sort_value(event):
        if callable(key):
            return key(event)

        if key == 'magnitude':
            if event.magnitudes:
                return event.magnitudes[0].mag
            else:
                return float('-inf')  # Events without magnitude go to the end

        elif key == 'time':
            if event.origins:
                return event.origins[0].time
            else:
                return datetime.min  # Events without origin time go to the beginning

        elif key == 'depth':
            if event.origins and event.origins[0].depth is not None:
                return event.origins[0].depth
            else:
                return float('inf')  # Events without depth go to the end

        elif key == 'latitude':
            if event.origins and event.origins[0].latitude is not None:
                return event.origins[0].latitude
            else:
                return float('-inf')

        elif key == 'longitude':
            if event.origins and event.origins[0].longitude is not None:
                return event.origins[0].longitude
            else:
                return float('-inf')

        else:
            raise ValueError(f"Unknown sort key: {key}")

    # Sort events
    sorted_events = sorted(catalog.events, key=get_sort_value, reverse=reverse)

    # Create new catalog with sorted events
    sorted_catalog = Catalog()
    for event in sorted_events:
        sorted_catalog += event

    return sorted_catalog

########################################################################################################################
# Catalog to Text-based files
########################################################################################################################

def catalog2basics(*args, **kwargs):
    return catalog2txyzm(*args, **kwargs)

def catalog2txyzm(
    cat,
    depth_unit="km",
    z_dir="depth",
    time_format="UTCDateTime",
    verbose=False,
    filename=False,
    origin_policy="preferred_or_first",
    **to_csv_kwargs,
):
    """Returns time, lat, lon, depth (km or m), and mag from ObsPy Catalog object.

    - ``time_format``: ``'UTCDateTime'``, ``'matplotlib'``, ``'datetime'``, or a strftime pattern.
    - ``depth_unit``: ``'km'`` or ``'m'``; depths below sea level are positive (depth mode).
    - ``origin_policy``: ``'preferred_or_first'`` (default; aligns maps and VCatalog),
      ``'first'``, or ``'last'`` (legacy last-origin behavior).

    Implementation delegates to :func:`vdapseisutils.compute.catalog.prepare_catalog_points_from_time_format`.

    Print to a txt file by specifying filename=<targetfile.txt>
    Manipulate the text file by specifying any keyword arguments understood by Pandas DataFrame to_csv()
    - default: index=False  # index column is not included in the txt file
    The dictionary is still returned
    e.g., >>> catalog2basics(catalog, filename="./catalog.txt", sep="\t")
    """
    from vdapseisutils.compute.catalog import prepare_catalog_points_from_time_format

    df = prepare_catalog_points_from_time_format(
        cat,
        time_format=time_format,
        origin_policy=origin_policy,
        depth_unit=depth_unit,
        z_dir=z_dir,
        verbose=verbose,
    )
    data = {
        "time": df["time"].tolist(),
        "lat": df["lat"].tolist(),
        "lon": df["lon"].tolist(),
        "depth": df["depth"].tolist(),
        "mag": df["mag"].tolist(),
    }

    if filename:
        pd.DataFrame(data).to_csv(filename, index=False, **to_csv_kwargs)
        if verbose:
            print("Catalog printed : {}".format(filename))

    return data

def catalog2picklog(cat):
    """CATALOG2PICKLOG Returns a DataFrame of Picks"""

    import pandas as pd

    pick_list = []

    for event in cat.events:
        for pick in event.picks:
            d = dict(pick)
            wsi = d["waveform_id"]  # returns ObsPy base WaveformStreamID (https://docs.obspy.org/packages/autogen/obspy.core.event.base.WaveformStreamID.html)
            d["nslc"] = wsi.get_seed_string()  # returns period separated NSLC (e.g., UW.STAR.--.BHZ)
            pick_list.append(d)

    df = pd.DataFrame.from_dict(pick_list)
    return df

def picklog2swarm(picklog, tags=["default"], filename="swarm_tagger.csv", mode="w"):
    """PICKLOG2SWARM Prints a DataFrame of picks to a Swarm tagger csv file"""

    # convert tags to a list the same size as catalog
    if len(tags) == 1:
        tags = tags*len(picklog)   # Repeat tags for each entry

    # df = pd.DataFrame(catalog2txyzm(picklog, time_format="%Y-%m-%d %H:%M:%S.%f"))  # Get time, lat, lon, depth, magnitude
    df = pd.DataFrame()
    df["time"] = [utc.strftime("%Y-%m-%d %H:%M:%S.%f") for utc in picklog["time"]]  # Keep only the "time" column
    df["scnl"] = [convertNSLCstr(nslc, order="nslc", neworder="scnl", sep=".", newsep=" ") for nslc in picklog["nslc"]]  # Add SCNL column
    df["tag"] = tags

    df.to_csv(filename, index=False, header=False, mode=mode)  # Write to CSV with specified mode

def txyzm2catalog(data):
    """
    TXYZM2CATALOG Converts time, latitude, longitude, depth, mag to an ObsPy Catalog object

    Input: dict or DataFrame of earthquake origin fields.
    Requires columns/fields to be time, latitude, longitude, depth, magnitude

    In the future, allow alternative column names to be specified
    colnames={"time": "time", "latitude": "latitude", "longitude": "longitude", "depth": "depth", "magn": "mag"}
    """

    cat = Catalog()
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
        # o.region = FlinnEngdahl().get_region(o.longitude, o.latitude)

        m = Magnitude()
        m.mag = row['mag']
        m.magnitude_type = "Md"

        # also included could be: custom picks, amplitude measurements, station magnitudes,
        # focal mechanisms, moment tensors, ...

        # make associations, put everything together
        cat.append(e)
        e.origins = [o]
        e.magnitudes = [m]
        m.origin_id = o.resource_id

    return cat

def basics2catalog(*args, **kwargs):
    """BASICS2CATALOG Wrapper for TXYZM2CATALOG"""
    return txyzm2catalog(*args, **kwargs)

def catalog2swarm_dep(catalog, nslc, tags=["default"], filename="swarm_tagger.csv"):
    """
    CATALOG2SWARM Create Swarm csv tagger file from ObsPy Catalog

    Usage: catalog2swarm(<ObsPy Catalog>, "UW.JUN.--.EHZ", ["LP"], filename="StHelens_LPs.csv")
    """
    # Developer's Note: A proper line in a tagger.csv file should look like this:
    # 2019-09-09 00:39:51.95,COP HHZ VV CP,VT


    # convert tags to a list the same size as catalog
    if len(tags) == 1:
        tags = tags*len(catalog)

    csv = pd.DataFrame(dict({"time": [], "nslc": [], "tag": []}))
    df = pd.DataFrame(catalog2txyzm(catalog, time_format="%Y-%m-%d %H:%M:%S.%f"))  # returns dictionary with time, lat, lon, depth, magnitude
    df = df.drop(labels=["lat", "lon", "depth", "mag"], axis=1)
    df["nslc"] = convertNSLCstr(nslc, order="nslc", neworder="scnl", sep=".", newsep=" ")   # Make space delimmited in SCNL order
    df["tag"] = tags
    csv = pd.concat([csv, df], ignore_index=False)

    csv = csv.reindex(columns=['time', 'nslc', 'tag'])
    csv.to_csv(filename, index=False, header=False)

def catalog2swarm(catalog, nslc, tags=["default"], filename="swarm_tagger.csv", mode="w"):
    """
    Creates a Swarm csv tagger file from an ObsPy Catalog.

    Args:
        catalog (obspy.Catalog): The ObsPy Catalog object to process.
        nslc (str): The NSLC string to convert to SCNL format for inclusion in the CSV.
        tags (list, optional): A list of tags to assign to each catalog entry. Defaults to ["default"].
        filename (str, optional): The name of the output CSV file. Defaults to "swarm_tagger.csv".
        mode (str, optional): The file opening mode. "w" for overwrite (default), "a" for append.
    """

    # Developer's Note: A proper line in a tagger.csv file should look like this:
    # 2019-09-09 00:39:51.95,COP HHZ VV CP,VT

    # convert tags to a list the same size as catalog
    if len(tags) == 1:
        tags = tags*len(catalog)   # Repeat tags for each entry

    df = pd.DataFrame(catalog2txyzm(catalog, time_format="%Y-%m-%d %H:%M:%S.%f"))  # Get time, lat, lon, depth, magnitude
    df = df[["time"]]  # Keep only the "time" column
    df["nslc"] = VStreamID(nslc).string(order="scnl", sep=" ")  # Create STA CHA NET LOC from NET.STA.LOC.CHA
    df["tag"] = tags

    df.to_csv(filename, index=False, header=False, mode=mode)  # Write to CSV with specified mode

def read_swarm_tags(swarm_tag_file, scnl_format="scnl"):
    df = pd.read_csv(swarm_tag_file, header=None, names=["time", "scnl", "tag"])
    # Handle mixed timestamp formats (with and without microseconds)
    df["time"] = pd.to_datetime(df["time"], format='mixed', dayfirst=False)
    return df

def createSwarmTags(times, nslc, tag, filename="swarm_tagger.csv"):
    print("This function will be deprecated in future versions. Use times2swarm instead.")

    import pandas as pd

    # create a sample nslc
    if len(nslc) == 1:
        nslc = nslc*len(times)

    # create a sample tag
    if len(tag) == 1:
        tag = tag*len(times)

    # create a pandas DataFrame with the times, nslc, and tag
    df = pd.DataFrame({'time': times, 'nslc': nslc, 'tag': tag})

    # convert the 'time' column to the correct format
    df['time'] = pd.to_datetime(df['time'])

    # set the 'time' column as the index
    df.set_index('time', inplace=True)

    # write the DataFrame to a CSV file
    df.to_csv(filename, header=False)

def times2swarm(times, scnl, tag, sort=False, filename="swarm_tagger.csv", overwrite=True):
    import pandas as pd
    import os

    # create a sample nslc
    if len(scnl) == 1:
        scnl = scnl * len(times)

    # create a sample tag
    if len(tag) == 1:
        tag = tag * len(times)

    # create a pandas DataFrame with the times, nslc, and tag
    df = pd.DataFrame({'time': times, 'scnl': scnl, 'tag': tag})

    # convert the 'time' column to the correct format
    df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    # set the 'time' column as the index
    df.set_index('time', inplace=True)

    if sort:
        df.sort_index(inplace=True)

    # check if the file exists and handle overwriting
    if os.path.exists(filename) and not overwrite:
        # Append to the existing file
        df.to_csv(filename, mode='a', header=False)
    else:
        # Write to a new file or overwrite
        df.to_csv(filename, header=False)
    
# Define method for getting waveforms from a catalog
def get_catalog_waveforms_dev(client, catalog, nslc_str, trange=(-2, 28), verbose=False):
    """Get Waveforms based on Event origin times"""
    streams = []
    net, sta, loc, cha = nslc_str.split(".")
    # Loop over events in the catalog
    for event in catalog:
        try:
            resource_id = str(event.resource_id).split("/")[-1]
            t = event.origins[0].time
            st = client.get_waveforms(net, sta, loc, cha, starttime=t+trange[0], endtime=t+trange[1])
            streams.append(st)
            if verbose:
                print("> event {} | {} | Waveforms dowloaded ({} Streams).".format(resource_id, t, len(st)))
        except Exception as e:
            print("> event {} | {} | Waveforms not found. Error: {}".format(resource_id, t, e))
    return streams

def find_matching_times(times1, reference_times, threshold_seconds=5):
    """
    For each datetime in times1, find indices of all reference_times that are within the specified threshold.

    Args:
        times1: List of datetime objects to check
        reference_times: List of reference datetime objects to match against
        threshold_seconds: Maximum time difference in seconds to consider as a match (default: 5)

    Returns:
        List of lists: For each datetime in times1, a list of indices from reference_times that are within
                      the threshold. If no matches are found for a datetime, an empty list is returned.
    """
    from obspy import UTCDateTime
    # Ensure datetime objects
    times1 = [UTCDateTime(t).datetime for t in times1]
    reference_times = [UTCDateTime(t).datetime for t in reference_times]

    results = []

    # For each datetime in times1, find all matching reference_times indices
    for t1 in times1:
        matching_indices = []

        for idx, ref_time in enumerate(reference_times):
            # Calculate the absolute time difference in seconds
            time_diff = abs((t1 - ref_time).total_seconds())

            # If the difference is within the threshold, add index to matches
            if time_diff <= threshold_seconds:
                matching_indices.append(idx)

        # Add the list of matching indices for this datetime to the results
        results.append(matching_indices)

    return results


if __name__ == '__main__':
    example()
    print()


## TO DO:
# [x] catalog2txyzm
# [X] catalog2swarm
# TODO [ ] Add optional magnitude type column
# TODO [ ] Add optional filename output
# TODO [ ] Should contents of data = catalog2basics(...) be arrays instead of lists?