from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Magnitude
from obspy.geodetics import FlinnEngdahl

import pandas as pd

example_file = './data/Copahue_events_from_reavs.csv'


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


# def read_csv(file, rename_dict=None, read_csv_kwargs=**kwargs)
def ewhtmlcsv2catalog(file, verbose=True):
    """EWHTMLCSV2CATALLOG

    EWHTMLREPORTCSV header:
    QuakeID,OriginTime(UTC),Latitude,Longitude,DepthBSL(km),Md,Ml,Ml_err,RMS,ErrorH(km),ErrorZ(km),Gap,TotalPhases,UsedPhases,SPhasesUsed,Quality

    See Pandas doc for csv reading tips:
        https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

    """

    # print('Read CSV produced by ewhtmlreport into ObsPy Catalog object')

    df = pd.read_csv(file, parse_dates=["OriginTime(UTC)"], infer_datetime_format=True, header=0,
                     # names=['time', 'lat', 'lon', 'depth', 'mag'],
                     )
    df['time'] = df['OriginTime(UTC)']
    df = df.drop(columns=['OriginTime(UTC)'])
    df['depth'] = df['DepthBSL(km)'] * 1000
    df = df.drop(columns=['DepthBSL(km)'])
    df['mag'] = df['Md']  # Just Duration magnitude for now
    df = df.drop(columns=['Md'])
    df['lat'] = df["Latitude"]
    df.drop(columns=["Latitude"])
    df['lon'] = df["Longitude"]
    df.drop(columns=["Longitude"])
    if verbose:
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
        m.magnitude_type = "Md"

        # also included could be: custom picks, amplitude measurements, station magnitudes,
        # focal mechanisms, moment tensors, ...

        # make associations, put everything together
        cat.append(e)
        e.origins = [o]
        e.magnitudes = [m]
        m.origin_id = o.resource_id

    if verbose:
        print(cat)
    return cat


def catalog2basics(args, kwargs, **to_csv_kwargs):
    print("CATALOG2BASICS has been replaced by CATALOG2TXYZM, BUT IT MAY STILL WORK.")
    catalog2txyzm(args, kwargs, to_csv_kwargs)


def catalog2txyzm(cat, depth_unit="km", time_format="UTCDateTime", verbose=False, filename=False, **to_csv_kwargs):
    """Returns time(UTCDateTime), lat, lon, depth(kilometers), and mag from ObsPy Catalog object

    - 'time' is returned as a UTCDateTime object
    - 'time_format' can be specified as 'UTCDateTime', 'matplotlib', or 'datetime'

    - 'depth' is returned as 'km' even though ObsPy Catalog default is 'm'
    - 'depth_unit' can be specified as 'km' or 'm'
    - depths below sea level are positive

    Print to a txt file by specificing filename=<targetfile.txt>
    Manipulate the text file by specifying any keyword arguments understood by Pandas DataFrame to_csv()
    - default: index=False  # index column is not included in the txt file
    The dictionary is still returned
    e.g., >>> catalog2basics(catalog, filename="./catalog.txt", sep="\t")
    """

    # Adjust for depth_unit
    if depth_unit == "m":
        dconvert = 1
    elif depth_unit == "km":
        dconvert = 1000
    else:
        if verbose: print("'depth_unit' not understood. Default value 'km' is used.")
        dconvert = 1000

    # Adjust for time_format
    def convert_time_format(utcdatetime):
        if time_format == "UTCDateTime":
            return utcdatetime
        elif time_format == "matplotlib":
            return utcdatetime.matplotlib_date
        elif time_format == "datetime":
            return utcdatetime.datetime
        else:
            print("'time_type' not understood. Default value 'UTCDateTime' is used.")
            return utcdatetime

    # Get info out of Events object
    time = []
    lat = []
    lon = []
    depth = []
    mag = []
    for event in cat:
        lat.append(event.origins[-1].latitude)
        lon.append(event.origins[-1].longitude)
        depth.append(event.origins[-1].depth / dconvert)  # meters (by default)
        mag.append(event.magnitudes[-1].mag if event.magnitudes[-1].mag is not None else -1)  # -1 is the default
        time.append(convert_time_format(event.origins[-1].time))

    data = dict({"time": time, "lat": lat, "lon": lon, "depth": depth, "mag": mag})

    if filename:
        pd.DataFrame(data).to_csv(filename, index=False, **to_csv_kwargs)
        if verbose: print("Catalog printed : {}".format(filename))

    return data


if __name__ == '__main__':
    example()
    print()
    read_victoria_csv(example_file)



"""
TO DO:
catalog2txyzm
[ ] TODO Add optional magnitude type column
[ ] TODO Add optional filename output
[ ] TODO Should contents of data = catalog2basics(...) be arrays instead of lists?
"""