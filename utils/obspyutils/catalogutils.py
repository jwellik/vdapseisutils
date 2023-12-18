import os
import pandas as pd

from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Magnitude
from obspy.geodetics import FlinnEngdahl

from vdapseisutils.utils.obspyutils import hypoinverse


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

# def read_csv(file, rename_dict=None, read_csv_kwargs=**kwargs)
def ewhtmlcsv2catalog(file, verbose=False):
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

def ewhtmltable2catalog(file, sep="\t", verbose=False):
    """
    ID	Date Time	Lat.	Lon.	Depth	MD	RMS	ErrH	ErrZ	Gap	Pha.	U.	S	Qual.	Event Page
    10153	2019.02.22 21:55:30	-37.8025	-71.0733	2.7	0	0.01	5.26	0.29	292	4	4	0	D	10153
    10152	2019.02.22 21:27:38	-37.7947	-71.0685	2.65	0	0.01	5.62	0.35	304	4	4	0	D	10152
    """

    df = pd.read_csv(file, parse_dates=["Date Time"], infer_datetime_format=True, header=0,
                     # names=['time', 'lat', 'lon', 'depth', 'mag'],
                     )
    df['time'] = df['Date Time']
    df = df.drop(columns=['Date Time'])
    df['depth'] = df['Depth'] * 1000  # convert km to meters
    df = df.drop(columns=['Depth'])
    df['mag'] = df['MD']  # Just Duration magnitude for now
    df = df.drop(columns=['MD'])
    df['lat'] = df["Lat."]
    df.drop(columns=["Lat."])
    df['lon'] = df["Lon."]
    df.drop(columns=["Lon."])
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

def read_ew_arcfiles_as_catalog(path):

    cat = Catalog()
    searchdir = path
    arcfiles = os.listdir(searchdir)
    for arcfile in arcfiles:
        arcfile = os.path.join(searchdir, arcfile)
        print("> {} ...".format(arcfile), end=" ")

        try:
            arc = hypoinverse.hypoCatalog()
            arc.readArcFile(arcfile)
            cat.append(arc.writeObspyCatalog()[0])
            print("SUCCESS", end="\n")
        except:
            print("FAILED.", end="\n")

    return cat

def read_hyp2000_log(logfile):
    """READ_HYP2000_LOG Reads log file from hyp2000_mgr, returns ObsPy Catalog object
    """
    # I HATE THIS FORMAT LOL

    import re
    import datetime as dt
    from obspy import UTCDateTime
    from vdapseisutils.utils.geoutils import dms2dd

    df = pd.DataFrame(columns=["time", "latitude", "longitude", "depth", "mag"])

    # Open the file for reading
    with open(logfile, 'r') as file:
        for line in file:
            if re.match(r'\d{8}_UTC_\d{2}:\d{2}:\d{2}', line):  # Check if the line matches the expected pattern
                parts = line.strip().split()  # Split the line by spaces

                if len(parts) == 17:  # A line of location data should have 16 parts

                    # time
                    date = parts[1]  # Extract the date part
                    hhmm = parts[2]  # Extract the time part
                    seconds = parts[3]
                    time = UTCDateTime(dt.datetime.strptime(date+hhmm+seconds, "%Y%m%d%H%M%S.%f"))

                    # latitude
                    d = parts[4]  # degrees and decimal minutes are stored here
                    try:
                        d, m = d.split("S")  # e.g., 37S47.96 -- split degrees, minutes by S or N
                        d = float(d) * -1  # mulitply by -1 if South
                        m = float(m)
                    except:
                        d, m = d.split("N")
                        d = float(d)
                        m = float(m)
                    latitude = dms2dd((d,m,0))  # Convert degrees, minutes to decimal degrees

                    # longitude
                    d = float(parts[5][0:-1])
                    d = d * -1 if parts[5][-1] == "W" else d
                    m = float(parts[6])
                    longitude = dms2dd((d, m, 0))

                    depth = float(parts[7])*1000  # convert km to m for ObsPy Catalog
                    mag = float(parts[8])

                    nphases = int(parts[9])

                    column10 = parts[10]
                    column11 = parts[11]
                    column12 = parts[12]
                    column13 = parts[13]
                    column14 = parts[14]
                    column15 = parts[15]
                    column15 = parts[15]

                    # append data to a DataFrame here
                    df = pd.concat([df, pd.DataFrame({"time": time, "latitude": latitude, "longitude": longitude,
                                                      "depth": depth, "mag": mag}, index=[0])],
                                   ignore_index=True)  # thanks chatGPT

    cat = basics2catalog(df)

    return cat


########################################################################################################################
# Catalog to Text-based files
########################################################################################################################

def catalog2basics(args, **kwargs):
    return catalog2txyzm(args, kwargs)

def catalog2txyzm(cat, depth_unit="km", z_dir="depth", time_format="UTCDateTime", verbose=False, filename=False, **to_csv_kwargs):
    """Returns time(UTCDateTime), lat, lon, depth(kilometers), and mag from ObsPy Catalog object

    - 'time' is returned as a UTCDateTime object
    - 'time_format' can be specified as 'UTCDateTime', 'matplotlib', 'datetime', or a string represented by strftime:
        https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
    - 'depth_unit' can be specified as 'km' or 'm'
    - depths below sea level are positive

    Print to a txt file by specifying filename=<targetfile.txt>
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

    # Assume depths are depths (positive down); multiply by -1 if given as elevations (positive up)
    if z_dir == "depth":
        z_dir_convert = 1
    elif z_dir == "elev":
        z_dir_convert = -1
    else:
        if verbose: print("'depth_unit' not understood. Default value 'depth' is used.")
        z_dir_convert = 1

    # Adjust for time_format
    def convert_time_format(utcdatetime):
        if time_format == "UTCDateTime":
            return utcdatetime
        elif time_format == "matplotlib":
            return utcdatetime.matplotlib_date
        elif time_format == "datetime":
            return utcdatetime.datetime
        else:
            return utcdatetime.strftime(time_format)

    # Get info out of Events object
    time = []
    lat = []
    lon = []
    depth = []
    mag = []
    for event in cat:
        lat.append(event.origins[-1].latitude)
        lon.append(event.origins[-1].longitude)
        depth.append(event.origins[-1].depth / dconvert * z_dir_convert)  # meters (by default)
        if event.magnitudes:
            mag.append(event.magnitudes[-1].mag if event.magnitudes[-1].mag is not None else -1)  # -1 is the default
        else:
            mag.append(-1)
        time.append(convert_time_format(event.origins[-1].time))

    data = dict({"time": time, "lat": lat, "lon": lon, "depth": depth, "mag": mag})

    if filename:
        pd.DataFrame(data).to_csv(filename, index=False, **to_csv_kwargs)
        if verbose: print("Catalog printed : {}".format(filename))

    return data

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

def catalog2swarm(catalog, nslc, tags=["default"], filename="swarm_tagger.csv"):
    """
    CATALOG2SWARM Create Swarm csv tagger file from ObsPy Catalog

    Usage: catalog2swarm(<ObsPy Catalog>, "UW.JUN.--.EHZ", ["LP"], filename="StHelens_LPs.csv")
    """
    # Developer's Note: A proper line in a tagger.csv file should look like this:
    # 2019-09-09 00:39:51.95,COP HHZ VV CP,VT

    from vdapseisutils.utils.obspyutils.inventoryutils import convertNSLCstr

    # convert tags to a list the same size as catalog
    if len(tags)==1:
        tags = tags*len(catalog)

    csv = pd.DataFrame(dict({"time": [], "nslc": [], "tag": []}))
    df = pd.DataFrame(catalog2txyzm(catalog, time_format="%Y-%m-%d %H:%M:%S.%f"))  # returns dictionary with time, lat, lon, depth, magnitude
    df = df.drop(labels=["lat", "lon", "depth", "mag"], axis=1)
    df["nslc"] = convertNSLCstr(nslc, order="nslc", neworder="scnl", sep=".", newsep=" ")   # Make space delimmited in SCNL order
    df["tag"] = tags
    csv = pd.concat([csv, df], ignore_index=False)

    csv = csv.reindex(columns=['time', 'nslc', 'tag'])
    csv.to_csv(filename, index=False, header=False)


# rename times2swarm?
def createSwarmTags(times, nslc, tag, filename="swarm_tagger.csv"):
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

if __name__ == '__main__':
    example()
    print()


## TO DO:
# [x] catalog2txyzm
# [X] catalog2swarm
# TODO [ ] Add optional magnitude type column
# TODO [ ] Add optional filename output
# TODO [ ] Should contents of data = catåalog2basics(...) be arrays instead of lists?