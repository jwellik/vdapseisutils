def download_profile(*args, **kwargs):
    return download_profile_opentopo(*args, **kwargs)


def download_profile_opentopo(p1, p2, n=100,
                              hosturl="https://api.opentopodata.org/v1/",
                              dataset="mapzen",
                              verbose=False):
    """Downloads elevaton data from https://www.opentopodata.org/
    e.g., https://api.opentopodata.org/v1/mapzen?locations=56.35,123.90

    P1 : (lat, lon) pair
    P2 : (lat, lon) pair
    n  : int : number of points for the elevation profile
    """

    import os

    import urllib.request
    import json
    import math
    import numpy as np

    api_url = os.path.join(hosturl, dataset + "?locations={locstr}")

    if verbose:
        print("Downloading data from OpenTopoData...")
        print(" Host       : {}".format(hosturl))
        print(" Dataset    : {}".format(dataset))

    # START-END POINT
    # P1
    # P2

    # NUMBER OF POINTS
    s = n - 1  # I don't know why, but setting this to 100 returns 101
    interval_lat = (p2[0] - p1[0]) / s  # interval for latitude
    interval_lon = (p2[1] - p1[1]) / s  # interval for longitude

    # SET A NEW VARIABLE FOR START POINT
    lat0 = p1[0]
    lon0 = p1[1]

    # LATITUDE AND LONGITUDE LIST
    lat_list = [lat0]
    lon_list = [lon0]

    # GENERATING POINTS
    for i in range(s):
        lat_step = lat0 + interval_lat
        lon_step = lon0 + interval_lon
        lon0 = lon_step
        lat0 = lat_step
        lat_list.append(lat_step)
        lon_list.append(lon_step)

    # HAVERSINE FUNCTION
    def haversine(lat1, lon1, lat2, lon2):
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon1_rad = math.radians(lon1)
        lon2_rad = math.radians(lon2)
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad
        a = math.sqrt(
            (math.sin(delta_lat / 2)) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * (math.sin(delta_lon / 2)) ** 2)
        d = 2 * 6371000 * math.asin(a)
        return d

    # DISTANCE CALCULATION
    d_list = []
    for j in range(len(lat_list)):
        lat_p = lat_list[j]
        lon_p = lon_list[j]
        dp = haversine(lat0, lon0, lat_p, lon_p) / 1000  # km
        d_list.append(dp)
    d_list_rev = d_list[::-1]  # reverse list


    # CONSTRUCT LOCATIONS STRING
    locations_str = ""
    for j in range(len(lat_list)):
        locations_str += "{lat},{lon}|".format(lat=lat_list[j], lon=lon_list[j])

    # SEND REQUEST
    request_url = api_url.format(locstr=locations_str)
    if verbose:
        print(" RequestURL : {}".format(request_url))


    response = urllib.request.Request(request_url)
    fp = urllib.request.urlopen(response)

    # RESPONSE PROCESSING
    res_byte = fp.read()
    res_str = res_byte.decode("utf8")
    js_str = json.loads(res_str)
    # print (js_mystr)
    fp.close()

    # GETTING ELEVATION
    elev_list = []
    for m in range(len(js_str["results"])):
        elev_list.append(js_str['results'][m]['elevation'])

    datadict = dict({'lat': np.array(lat_list), 'lon': np.array(lon_list), 'd': np.array(d_list_rev),
                     'elev': np.array(elev_list),
                     'elev_m': np.array(elev_list), 'elev_km': np.array(elev_list)/1000})

    if verbose:
        print('Done')

    return datadict

def download_profile_open_elevation(p1, p2, n=100, verbose=False):
    """
    ELEVATION PROFILE APP GENERATOR
    ideagora geomatics-2018
    http://geodose.com
    https://www.geodose.com/2018/03/create-elevation-profile-generator-python.html

    P1 : (lat, lon) pair
    P2 : (lat, lon) pair
    n  : int : number of points for the elevation profile
    """
    import urllib.request
    import json
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    # START-END POINT
    # P1
    # P2

    if verbose:
        print('Downloading elevation data from "https://api.open-elevation.com/api/v1/lookup"')

    # NUMBER OF POINTS
    s = n
    s = 100
    interval_lat = (p2[0] - p1[0]) / s  # interval for latitude
    interval_lon = (p2[1] - p1[1]) / s  # interval for longitude

    # SET A NEW VARIABLE FOR START POINT
    lat0 = p1[0]
    lon0 = p1[1]

    # LATITUDE AND LONGITUDE LIST
    lat_list = [lat0]
    lon_list = [lon0]

    # GENERATING POINTS
    for i in range(s):
        lat_step = lat0 + interval_lat
        lon_step = lon0 + interval_lon
        lon0 = lon_step
        lat0 = lat_step
        lat_list.append(lat_step)
        lon_list.append(lon_step)

    # HAVERSINE FUNCTION
    def haversine(lat1, lon1, lat2, lon2):
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon1_rad = math.radians(lon1)
        lon2_rad = math.radians(lon2)
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad
        a = math.sqrt(
            (math.sin(delta_lat / 2)) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * (math.sin(delta_lon / 2)) ** 2)
        d = 2 * 6371000 * math.asin(a)
        return d

    # DISTANCE CALCULATION
    d_list = []
    for j in range(len(lat_list)):
        lat_p = lat_list[j]
        lon_p = lon_list[j]
        dp = haversine(lat0, lon0, lat_p, lon_p) / 1000  # km
        d_list.append(dp)
    d_list_rev = d_list[::-1]  # reverse list

    # CONSTRUCT JSON
    d_ar = [{}] * len(lat_list)
    for i in range(len(lat_list)):
        d_ar[i] = {"latitude": lat_list[i], "longitude": lon_list[i]}
    location = {"locations": d_ar}
    json_data = json.dumps(location, skipkeys=int).encode('utf8')

    # SEND REQUEST
    url = "https://api.open-elevation.com/api/v1/lookup"
    response = urllib.request.Request(url, json_data, headers={'Content-Type': 'application/json'})
    fp = urllib.request.urlopen(response)

    # RESPONSE PROCESSING
    res_byte = fp.read()
    res_str = res_byte.decode("utf8")
    js_str = json.loads(res_str)
    # print (js_mystr)
    fp.close()

    # GETTING ELEVATION
    response_len = len(js_str['results'])
    elev_list = []
    for j in range(response_len):
        elev_list.append(js_str['results'][j]['elevation'])

    datadict = dict({'lat': np.array(lat_list), 'lon': np.array(lon_list), 'd': np.array(d_list_rev),
                     'elev': np.array(elev_list),
                     'elev_m': np.array(elev_list), 'elev_km': np.array(elev_list)/1000})

    if verbose:
        print('Done')

    return datadict


def download_profile_alt():
    # https://gis.stackexchange.com/questions/29632/getting-elevation-at-lat-long-from-raster-using-python
    # https://developers.google.com/maps/documentation/elevation/start#maps_http_elevation_locations-py
    pass


def write(km, elev, file='elevation.csv'):
    import csv

    rows = zip(km, elev)

    with open(file, "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def read(file):
    import csv
    km, elev = [], []
    csv_reader = csv.reader(open(file))
    for line in csv_reader:
        km.append(float(line[0]))
        elev.append(float(line[1]))
    return km, elev


def plot_original(d_list_rev, elev_list):
    import matplotlib.pyplot as plt

    # BASIC STAT INFORMATION
    mean_elev = round((sum(elev_list) / len(elev_list)), 3)
    min_elev = min(elev_list)
    max_elev = max(elev_list)
    distance = d_list_rev[-1]

    # PLOT ELEVATION PROFILE
    base_reg = 0
    plt.figure(figsize=(10, 4))
    plt.plot(d_list_rev, elev_list)
    plt.plot([0, distance], [min_elev, min_elev], '--g', label='min: ' + str(min_elev) + ' m')
    plt.plot([0, distance], [max_elev, max_elev], '--r', label='max: ' + str(max_elev) + ' m')
    plt.plot([0, distance], [mean_elev, mean_elev], '--y', label='ave: ' + str(mean_elev) + ' m')
    plt.fill_between(d_list_rev, elev_list, base_reg, alpha=0.1)
    plt.text(d_list_rev[0], elev_list[0], "P1")
    plt.text(d_list_rev[-1], elev_list[-1], "P2")
    plt.xlabel("Distance(km)")
    plt.ylabel("Elevation(m)")
    plt.grid()
    plt.legend(fontsize='small')
    plt.show()


def plot(h, z, depth_range=[-50., 4.], depth=None, color='black', linewidth=0.75,
         fill_color="black", fill_alpha=0.1):
    import matplotlib.pyplot as plt

    # BASIC STAT INFORMATION
    # mean_elev = round((sum(elev_list) / len(elev_list)), 3)
    # min_elev = min(elev_list)
    # max_elev = max(elev_list)
    # distance = d_list_rev[-1]

    # PLOT ELEVATION PROFILE
    if depth:
        depth_range[0] = depth*-1
        print("DEPTH will be deprecated in future versions. Use DEPTH_RANGE")
    # fig = plt.figure(figsize=(10, 4))
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 4, forward=True)
    ax.plot(h, z, color=color, linewidth=linewidth)
    # plt.plot([0, distance], [min_elev, min_elev], '--g', label='min: ' + str(min_elev) + ' m')
    # plt.plot([0, distance], [max_elev, max_elev], '--r', label='max: ' + str(max_elev) + ' m')
    # plt.plot([0, distance], [mean_elev, mean_elev], '--y', label='ave: ' + str(mean_elev) + ' m')
    if fill_color:
        ax.fill_between(h, z, depth_range[0], color=fill_color, alpha=fill_alpha)  # depth_range[0] is the base
    # plt.text(d_list_rev[0], elev_list[0], "P1")
    # plt.text(d_list_rev[-1], elev_list[-1], "P2")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    # plt.grid()
    # plt.legend(fontsize='small')

    ax.set_ylim(depth_range)
    ax.set_xlim([h[0], h[-1]])
    ax.spines['top'].set_visible(False)
    ax.spines["left"].set_bounds((depth_range[0], z[0]))  # depth_extent_v[1] is the top elev
    ax.spines["right"].set_bounds((depth_range[0], z[-1]))

    #ax.set_yticks([0, -5, -10, -15])

    plt.draw()
    return fig, ax


def test():
    import matplotlib.pyplot as plt
    data = download_profile((-8.169148, 115.283046), (-8.579110, 115.819991))  # Includes bathymetry
    # data = download_profile((-8.2359, 115.5995), (-8.4145, 115.4465))
    fig, ax = plot(data["d"], data["elev"], depth=0)
    print(ax)
    ax.plot(31, 2806, color='k')
    plt.show()


if __name__ == '__main__':
    test()

#
#TO DO:
# TODO Allow profile to be a series of points, instead of just (P1, P2)
#
