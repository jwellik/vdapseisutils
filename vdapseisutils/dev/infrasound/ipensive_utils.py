import datetime
import os
import importlib.util
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import timezonefinder
import pytz

from obspy.geodetics.base import gps2dist_azimuth


def load_python_config(config_path):
    """Loads a Python file specified by any path as a config file"""

    # Check if the config file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"The configuration file {config_path} does not exist.")

    # Get the module name and path
    module_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = config_module
    spec.loader.exec_module(config_module)

    return config_module


def get_backazimuth(receiver, source):
    """Get direction from receiver to source."""
    tmp = gps2dist_azimuth(receiver[0], receiver[1], source[0], source[1])
    return tmp[1]


def get_volcano_backazimuth(array):
    """Get backazimuth from array to volcano given ipensive configuration dictionary. Updates the dictionary.

    Example dictionary:
    dict({'Name':'Kenai',
				  'SCNL':[
							{'scnl':'KENI.HDF.AV.01'	, 'sta_lat': 60.6413700	, 'sta_lon': -151.070200},
							{'scnl':'KENI.HDF.AV.02'	, 'sta_lat': 60.6404567 , 'sta_lon': -151.070330},
							{'scnl':'KENI.HDF.AV.03'	, 'sta_lat': 60.6406033	, 'sta_lon': -151.072020},
							{'scnl':'KENI.HDF.AV.04'	, 'sta_lat': 60.6412000	, 'sta_lon': -151.073000},
							{'scnl':'KENI.HDF.AV.05'	, 'sta_lat': 60.6415300	, 'sta_lon': -151.072000},
							{'scnl':'KENI.HDF.AV.06'	, 'sta_lat': 60.6409167 , 'sta_lon': -151.071170},
						],
				'digouti': (1/419430.0)/(0.0275),
				'volcano':[
							{'name': 'Wrangell',  'v_lat': 62.00572,  'v_lon': -144.01935},
							{'name': 'Spurr',     'v_lat': 61.29897,  'v_lon': -152.25122},
							{'name': 'Redoubt',   'v_lat': 60.48576,  'v_lon': -152.74282},
							{'name': 'Iliamna',   'v_lat': 60.03220,  'v_lon': -153.09002},
							{'name': 'Augustine', 'v_lat': 59.36107,  'v_lon': -153.42938},
						  ],
				'AZ_MIN': 200,
				'AZ_MAX': 80
			}),
    """
    # Something isn't right here

    lon0 = np.mean([SCNL["sta_lon"] for SCNL in array["SCNL"]])
    lat0 = np.mean([SCNL["sta_lat"] for SCNL in array["SCNL"]])

    for volc in array['volcano']:
        if 'back_azimuth' not in volc:
            # Why does it need to be opposite of original ipensive code
            # tmp=gps2dist_azimuth(lat0,lon0,volc['v_lat'],volc['v_lon'])
            tmp=gps2dist_azimuth(lat0,lon0, volc['v_lat'], volc['v_lon'])
            volc['back_azimuth']=tmp[1]

    return array


def create_source_dict(arrays):

    # Initialize the new dictionary to store unique volcano data
    sources = []

    # Create dictionary of unique sources
    for array in arrays:
        for volcano in array['volcano']:
            volcano_id = volcano['name']
            if volcano_id not in [s["name"] for s in sources]:
                sources.append({
                    'name': volcano['name'],
                    'v_lat': volcano['v_lat'],
                    'v_lon': volcano['v_lon'],
                    'arrays': [],
                })

    # Add all arrays to each source.
    # If the volcano was in the array's list, mark it as 'active'.
    for source in sources:
        for array in arrays:
            active_volcs = [v["name"] for v in arrays[0]["volcano"]]
            lon0 = np.mean([SCNL["sta_lon"] for SCNL in array["SCNL"]])
            lat0 = np.mean([SCNL["sta_lat"] for SCNL in array["SCNL"]])
            tmp = gps2dist_azimuth(lat0, lon0, source["v_lat"], source["v_lon"])
            source["arrays"].append({
                'name': array['Name'],
                'distance': tmp[0],
                'backazimuth': tmp[1],
                'forwardazimuth': tmp[2],
                'active': True if source["name"] in active_volcs else False,
            })

    return sources


def create_array_df(arrays):

    # Initialize DataFrames
    elements = pd.DataFrame()  # contains infrasound elements from all arrays

    # Concatenate DataFrames from arrays
    for a in arrays:
        elem_tmp = pd.DataFrame.from_dict(a["SCNL"])
        elem_tmp["name"] = a["Name"]
        elements = pd.concat([elements, elem_tmp], ignore_index=True)
        sources = pd.concat([sources, pd.DataFrame.from_dict(a["volcano"])])
    sources = sources.drop_duplicates(subset="name")  # Reduce sources DataFrame to unique values according to the "name" column
    # drop back-azimuth from sources bc that's the backazimuth to just one array

    # # Convert lat/lon to Mercator easting/northing
    # elements_xy = latlon2xymeters(elements["sta_lat"], elements["sta_lon"])
    # sources_xy = latlon2xymeters(sources["v_lat"], sources["v_lon"])
    #
    # # Add new columns to DataFrames
    # elements["easting"] = elements_xy[0]
    # elements["northing"] = elements_xy[1]
    # sources["easting"] = sources_xy[0]
    # sources["northing"] = sources_xy[1]
    #
    # # I think the previous two blocks could be made shorter if latlon2xymeters returned two lists instead of a tuple of lists???
    # # Convert lat/lon to Mercator easting/northing and add new columns to DataFrames
    # # elements[["easting", "northing"]] = latlon2xymeters(elements["sta_lat"], elements["sta_lon"])
    # # sources[["easting", "northing"]] = latlon2xymeters(sources["v_lat"], sources["v_lon"])

    # Reset index for both DataFrames
    elements.reset_index(drop=True, inplace=True)

    # Create new DataFrame with one entry for each array
    array_coordinates = elements.groupby("name")[["easting", "northing"]].mean().reset_index()  # Groups elements by arrays names and gets average easting,northing
    array_coordinates.columns = ["name", "easting", "northing"]  # Rename columns for clarity


def get_array_timezone(arrays, t=datetime.datetime.now()):
    """GET TIMEZONE FROM LAT,LON, TIME

    Lat,lon is determined as center point of array

    :return datetime.timedelta object
    """

    print("Trying to get local timezone...")
    try:
        lat = np.mean([i["sta_lat"] for i in arrays["SCNL"]])
        lon = np.mean([i["sta_lon"] for i in arrays["SCNL"]])
        # print(lat)
        # print(lon)
        tf = timezonefinder.TimezoneFinder()
        # print(tf)
        timezone_str = tf.timezone_at(lat=lat, lng=lon)
        # print(timezone_str)
        timezone = pytz.timezone(timezone_str)  # Get the timezone object
        # print(timezone)
        localized_dt = timezone.localize(t)  # Localize the datetime
        # print(localized_dt)
        utc_offset = localized_dt.utcoffset()  # Get the UTC offset
        # print(utc_offset)
    except:
        print("Unable to find Timezone. Returning 0.")
        utc_offset = 0

    return datetime.timedelta(utc_offset)


def read_ipensive_ascii_output(directory, vel_units="km/s"):
    """READS IPENSIVE ASCII FILES AND RETURNS DATA AS A DATAFRAME

    Returns velocity in km/s
     By default, ipensive writes velocity as m/s, but all other utilities assume km/s.
     Therefore, this method returns km/s by default,
     but you can specify the original m/s with vel_units="m/s"
    """

    all_data = []
    nfiles = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                nfiles += 1
                print(">>> Loading {}".format(os.path.join(root, file)))
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path, delimiter='\t')
                data['Time'] = pd.to_datetime(data['Time'])
                all_data.append(data)

    if nfiles > 1:
        combined_data = pd.concat(all_data, ignore_index=True)
    else:
        combined_data = all_data.copy()

    if vel_units not in ["km/s", "m/s"]:
        print("WARNING: vel_units ({}) not understood. Reverting to default 'km/s'".format(vel_units))
        vel_units = "km/s"
    if vel_units == "km/s":
        combined_data["Velocity"] /= 1000
    else:  # "m/s"
        pass

    return combined_data


def wget_ascii_output_v1(webdir, arrays, t1, t2, archivedir="./"):
    """
    https://avosouth.wr.usgs.gov/infrasound/ascii_output/

    Download text files from a web directory. The web directory has the structure:
    <webdir>/<array>/<year>-<month>/<array>_<year>-<month>-<date>.txt

    Given webdir, download all text files that are inbetween t1 and t2 and that are included in the list of arrays named
    'array'. Save all files with the same filestructure in 'archivedir'.
    """
    import requests
    import os
    import pandas as pd

    # Define the web directory URL and the arrays to download
    # webdir = "http://example.com/webdir"
    # arrays = ["array1", "array2", "array3"]

    # Define the time range to download data for
    # t1 = "2022-01-01"
    # t2 = "2022-01-31"

    # Define the directory to save the downloaded data
    # archivedir = "archivedir"

    # Loop over the arrays and download the data for each one
    for array in arrays:
        # Define the URL for the array directory
        array_url = f"{webdir}{array}"

        # Get the list of files in the array directory
        response = requests.get(array_url)
        files = response.text.split("\n")

        # Loop over the files and download the data for each one that is within the time range
        for file in files:
            # Check if the file is within the time range
            filename = file.split("/")[-1]
            if filename.startswith(array) and filename.endswith(".txt") and t1 <= filename[
                                                                                  len(array) + 1:len(filename) - 4] <= t2:
                # Download the file
                file_url = f"{array_url}/{filename}"
                response = requests.get(file_url)

                # Save the file to the archivedir
                os.makedirs(f"{archivedir}/{array}/{filename[:-4]}", exist_ok=True)
                with open(f"{archivedir}/{array}/{filename}", "wb") as f:
                    f.write(response.content)

                # Read the data from the file using the read_ipensive_ascii_output() function
                data = read_ipensive_ascii_output(f"{archivedir}/{array}/{filename}")

                # Save the data to a CSV file
                df = pd.DataFrame(data)
                df.to_csv(f"{archivedir}/{array}/{filename[:-4]}.csv", index=False)

                return df


def wget_ascii_output(webdir, arrays, t1, t2, archivedir="./", auth=None):
    """
    https://avosouth.wr.usgs.gov/infrasound/ascii_output/

    Download text files from a web directory. The web directory has the structure:
    <webdir>/<array>/<year>-<month>/<array>_<year>-<month>-<date>.txt

    Given webdir, download all text files that are inbetween t1 and t2 and that are included in the list of arrays named
    'array'. Save all files with the same filestructure in 'archivedir'.
    """
    import requests
    import os
    import pandas as pd
    import datetime

    # Define the web directory URL and the arrays to download
    # webdir = "http://example.com/webdir"
    # arrays = ["array1", "array2", "array3"]

    # Define the time range to download data for
    # t1 = "2022-01-01"
    # t2 = "2022-01-31"

    # Define the directory to save the downloaded data
    # archivedir = "archivedir"

    t1 = datetime.datetime.strptime(t1, "%Y/%m/%d")
    t2 = datetime.datetime.strptime(t2, "%Y/%m/%d")

    # Loop over the arrays and download the data for each one
    for array in arrays:
        # Define the URL for the array directory
        array_url = f"{webdir}{array}"

        # Loop through the date range and download files
        for day in range((t2 - t1).days + 1):
            date = t1 + datetime.timedelta(days=day)
            year, month = str(date.year), str(date.month).zfill(2)
            file_url = f"{array_url}/{year}-{month}/{array}_{year}-{month}-{date.day}.txt"

            # Download the file
            session = requests.Session()
            session.auth = (auth[0], auth[1])

            # Make a request using the session object
            response = session.get(file_url)

            # Check if the file was downloaded successfully
            if response.status_code == 200:
                # Save the file to the archive directory
                archive_path = f"{archivedir}/{array}/{year}-{month}"
                filename = f"{array}_{year}-{month}-{date.day}.txt"
                filepath = os.path.join(archive_path, filename)
                os.makedirs(archive_path, exist_ok=True)
                with open(filepath, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Error downloading file: {file_url}")

            # # Read the data from the file using the read_ipensive_ascii_output() function
            # data = read_ipensive_ascii_output(filepath)


def filter_data(data, time=None, array=None, azimuth=None, velocity=None, mccm=None, pressure=None, rms=None, day_of_week=None):
    filtered_data = data.copy()

    if time is not None:
        from obspy import UTCDateTime
        time = [UTCDateTime(time[0]).datetime, UTCDateTime(time[1]).datetime]
        filtered_data = filtered_data[(filtered_data['Time'] >= time[0]) & (filtered_data['Time'] <= time[1])]
    if array is not None:
        filtered_data = filtered_data[filtered_data['Array'].isin(array)]

    if azimuth:
        filtered_data = filtered_data[(filtered_data['Azimuth'] >= azimuth[0]) & (filtered_data['Azimuth'] <= azimuth[1])]
    if velocity:
        filtered_data = filtered_data[(filtered_data['Velocity'] >= velocity[0]) & (filtered_data['Velocity'] <= velocity[1])]
    if mccm:
        filtered_data = filtered_data[filtered_data['MCCM'] >= mccm]
    if pressure:
        filtered_data = filtered_data[filtered_data['Pressure'] >= pressure]
    if rms:
        filtered_data = filtered_data[filtered_data['rms'] >= rms]

    # Day 0:6 where 0 is Monday
    if day_of_week is not None:
        filtered_data = filtered_data[filtered_data['Time'].dt.dayofweek.isin(day_of_week)]

    return filtered_data


def plot_polar_time(data, bin_width=60, utc_offset=0, color="skyblue"):
    """Polar plot of detections by hour.

    Data should contain 'Time'

    TODO Actually rotate the bars when changing UTC Offset
    """

    # Plot rose diagram
    hours = data['Time'].dt.hour
    bin_counts = np.bincount(hours, minlength=24)
    theta = np.deg2rad(np.arange(0, 360, 360 / 24))

    ax = plt.subplot(1, 1, 1, polar=True)
    bars = ax.bar(theta, bin_counts, width=np.deg2rad(360/24), color=color, align='edge')
    print(hours)
    print(bin_counts)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    label_offset = utc_offset  # Starting hour for labels
    label_format = lambda x: f"{int((x + label_offset) % 24):02d}00h"  # Format labels as 'hh00h'
    ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax.set_xticklabels([label_format(x) for x in np.arange(0, 24)])

    plt.title('Counts per Time of Day\nLocal Time (UTC+{})\n({} samples)'.format(label_offset, len(data)))

    for bar in bars:
        bar.set_alpha(0.8)

    plt.tight_layout()
    plt.show()


def plot_polar_azimuth(data, bin_width=10, color="skyblue"):
    """Polar plot by azimuth."""

    # Plot rose diagram
    y = data['Azimuth']
    counts, deg = np.histogram(data["Azimuth"], bins=np.arange(0, 365, 10))
    # theta = np.deg2rad(np.arange(0, 360, 10))
    theta = np.deg2rad(deg[:-1])

    ax = plt.subplot(1, 1, 1, polar=True)
    bars = ax.bar(theta, counts, width=np.deg2rad(10), color=color, align='edge')

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    # ax.set_xticklabels(np.arange(0, 24))

    # label_offset = 13
    # label_format = lambda x: f"{int((x + label_offset) % 24):02d}00h"  # Format labels as 'hh00h'
    # ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    # ax.set_xticklabels([label_format(x) for x in np.arange(0, 24)])

    # plt.ylim([0, 25])

    # ax.set_yticklabels([])

    # plt.title('Counts per Time of Day\nLocal Time (UTC+{})\n({} samples)'.format(label_offset, len(data)))

    for bar in bars:
        bar.set_alpha(0.8)

    plt.tight_layout()
    plt.show()

    # Plot polar by azimuth (not as a histogram)
    ax = plt.subplot(1, 1, 1, polar=True)
    theta = np.deg2rad(data["Azimuth"])
    ax.scatter(theta, data["MCCM"])

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)


def axazline(azimuth=None, label=None, ax=None, axhline_kwargs=None, text_kwargs=None):
    """
    Plots horizontal line for backazimuth across axes. Adds a text label. Uses ax.axhline and ax.text

    Note: The xposition of the label will be affected if the xlim of the axes changes.
    """

    # Merge user input into defaults
    default_text_kwargs = {"bbox": {'facecolor': 'white', 'edgecolor': 'white', 'pad': 0},
                       "fontsize": 8, "verticalalignment": 'center', "style": 'italic', "zorder": 10}
    if text_kwargs is not None:
        default_text_kwargs.update(text_kwargs)

    default_axhline_kwargs = {"color": 'black', "linestyle": '--'}
    if axhline_kwargs is not None:
        default_axhline_kwargs.update(axhline_kwargs)

    xpos = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02  # Place label 0.1 axes units from left
    ax.axhline(azimuth, **default_axhline_kwargs)
    ax.text(xpos, azimuth, label, **default_text_kwargs)
