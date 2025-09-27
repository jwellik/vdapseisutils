"""
Elevation profile utilities for topographic data.

This module provides functionality to download and process elevation profiles
along lines between two geographic points. It supports multiple data sources:

- PyGMT (default): Uses PyGMT's load_earth_relief() for high-quality elevation data
- OpenTopoData: Uses the OpenTopoData API for elevation data
- Open-Elevation: Uses the Open-Elevation API for elevation data

The TopographicProfile class provides a convenient interface for creating
elevation profiles with automatic resolution selection and error handling.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2024
"""

import numpy as np
from vdapseisutils.utils.geoutils import geoutils

class TopographicProfile:

    # NOTE: Do everything in meters!

    def __init__(self, points, source="opentopo", resolution="auto", max_n=100):

        if len(points) != 2:
            raise ValueError("Exactly two points (lat,lon) must be provided.")
        self.points  = points# list of tuple-like lat,lon coordinates

        # returns azimuth and length (in meters) of line
        self.azimuth, self.length = geoutils.backazimuth(points[0], points[1]) # degrees, kilometers
        # self.length *= 1000  # convert to meters

        self.distance = np.array([])
        self.elevation = np.array([])

        # Handle resolution parameter - support both "auto" and numeric values for backward compatibility
        if resolution == "auto":
            self.resolution, self.n = self._calculate_auto_resolution()
        else:
            # Backward compatibility: numeric resolution value
            resolution = float(resolution)
            max_n = int(max_n)
            max_resolution = self.length/max_n
            if max_resolution > resolution:
                self.resolution = np.max([resolution, max_resolution])
                self.n = max_n
                print("Resolution ({} m) is too fine for the length (~{:2.1f} m) and max_n ({}).\n"
                              "Returned resolution is ~{:2.1f}m (n: {})".format(resolution, self.length, max_n,
                                                                                self.resolution, self.n))
            else:
                self.resolution = resolution
                self.n = int(self.length / self.resolution)

        # Download Topographic Profile from online source
        self.source = source  # where to get data ('pygmt', 'opentopo', 'file')
        try:
            data = download_profile(self.points[0], self.points[1], n=self.n, source=source) # returns dict w lat, lon, d, elev, elev_m, elev_km
            self.elevation = data["elev_m"]  # elevation in meters
            self.distance = data["d"] # horizontal distance along line in meters
        except Exception as e:
            print(f"Error downloading elevation data from {source}: {e}")
            print(f"Profile points: {self.points[0]} to {self.points[1]}")
            print(f"Profile length: {self.length/1000:.2f} km")
            print(f"Requested points: {self.n}")
            print("Elevation and distance arrays are empty.")
            self.elevation = np.array([])
            self.distance = np.array([])

    def _calculate_auto_resolution(self):
        """Calculate optimal resolution and number of points based on profile length.
        
        Returns:
        --------
        tuple : (resolution_meters, n_points)
        """
        # Define resolution tiers based on profile length (in km)
        length_km = self.length / 1000  # Convert to km for easier reasoning
        
        if length_km < 1:  # Very short profiles (< 1 km)
            resolution = 10  # 10 meter resolution
        elif length_km < 5:  # Short profiles (1-5 km)
            resolution = 25  # 25 meter resolution
        elif length_km < 10:  # Medium profiles (5-10 km)
            resolution = 50  # 50 meter resolution
        elif length_km < 25:  # Medium-long profiles (10-25 km)
            resolution = 100  # 100 meter resolution
        elif length_km < 50:  # Long profiles (25-50 km)
            resolution = 200  # 200 meter resolution
        elif length_km < 100:  # Very long profiles (50-100 km)
            resolution = 500  # 500 meter resolution
        else:  # Extremely long profiles (> 100 km)
            resolution = 1000  # 1 km resolution
        
        # Calculate number of points
        n_points = max(10, int(self.length / resolution))  # Minimum 10 points
        
        # Apply reasonable limits to avoid API issues
        n_points = min(n_points, 500)  # Maximum 500 points to avoid API limits
        
        return resolution, n_points

    def plot(self, *args, **kwargs):
        plot_profile(self.distance, self.elevation, *args, **kwargs)

    def write(self, *args, **kwargs):
        write(self.distance, self.elevation, *args, **kwargs)
    
    def info(self):
        """Print information about the topographic profile."""
        print("=== Topographic Profile Information ===")
        print(f"Source: {self.source}")
        print(f"Start point: {self.points[0]}")
        print(f"End point: {self.points[1]}")
        print(f"Azimuth: {self.azimuth:.1f}°")
        print(f"Length: {self.length:.1f} km")
        print(f"Resolution: {self.resolution:.1f} m")
        print(f"Number of points: {self.n}")
        if len(self.elevation) > 0:
            print(f"Elevation range: {self.elevation.min():.1f} to {self.elevation.max():.1f} m")
            print(f"Mean elevation: {self.elevation.mean():.1f} m")
        else:
            print("No elevation data available")
        print("=" * 40)

def download_profile(*args, source="opentopo", **kwargs):
    """Download elevation profile data from various sources.
    
    Parameters:
    -----------
    source : str
        Data source ("pygmt", "opentopo", "open_elevation")
    *args, **kwargs : 
        Arguments passed to the specific download function
    """
    if source.lower() == "pygmt":
        return download_profile_pygmt_dep001(*args, **kwargs)
    elif source.lower() == "opentopo":
        return download_profile_opentopo(*args, **kwargs)
    elif source.lower() == "open_elevation":
        return download_profile_open_elevation(*args, **kwargs)
    elif source.lower() == "3dep" or source.lower() == "usgs":
        return download_profile_3dep(*args, **kwargs)
    else:
        raise ValueError(f"Unknown source '{source}'. Available sources: 'pygmt', 'opentopo', 'open_elevation', '3dep', 'usgs'")

def download_profile_pygmt_dep001(p1, p2, n=100, data_source="igpp", resolution="auto",
                          verbose=False, **kwargs):
    """Download elevation profile data using PyGMT.
    
    Parameters:
    -----------
    p1 : tuple
        Start point (lat, lon)
    p2 : tuple
        End point (lat, lon)
    n : int
        Number of points for the elevation profile
    data_source : str
        PyGMT data source ("igpp", "srtm", "gebco", etc.)
    resolution : str
        Data resolution ("auto", "01d", "30s", "15s", etc.)
    verbose : bool
        Print verbose output
    **kwargs : 
        Additional arguments passed to pygmt.datasets.load_earth_relief()
    
    Returns:
    --------
    dict : Dictionary containing lat, lon, distance, elevation arrays
    """
    try:
        import pygmt
        import math
    except ImportError:
        raise ImportError("PyGMT is required for this function. Install with: pip install pygmt")
    
    if verbose:
        print("Downloading elevation data using PyGMT...")
        print(f" Data source: {data_source}")
        print(f" Resolution: {resolution}")
    
    # Calculate the bounding box for the profile
    lat_min, lat_max = min(p1[0], p2[0]), max(p1[0], p2[0])
    lon_min, lon_max = min(p1[1], p2[1]), max(p1[1], p2[1])
    
    # Add some padding to ensure we get enough data
    lat_pad = (lat_max - lat_min) * 0.1
    lon_pad = (lon_max - lon_min) * 0.1
    
    region = [lon_min - lon_pad, lon_max + lon_pad, 
              lat_min - lat_pad, lat_max + lat_pad]
    
    # Automatic resolution selection based on profile length
    if resolution == "auto":
        # Calculate approximate profile length in degrees
        profile_length_deg = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Resolution selection based on profile size (similar to maps.py)
        if profile_length_deg > 180:  # Global or very large regions
            resolution = "01d"
        elif profile_length_deg > 90:  # Continental scale
            resolution = "30m"
        elif profile_length_deg > 45:  # Regional scale
            resolution = "15m"
        elif profile_length_deg > 20:  # Large local areas
            resolution = "03s"
        elif profile_length_deg > 10:  # Medium local areas
            resolution = "01s"
        elif profile_length_deg > 5:   # Small local areas
            resolution = "30s"
        elif profile_length_deg > 1:   # Very small local areas
            resolution = "15s"
        else:  # Tiny areas
            resolution = "03s"
        
        if verbose:
            print(f" Auto-selected resolution: {resolution}")
    
    # Download elevation data
    try:
        srtm = pygmt.datasets.load_earth_relief(
            region=region, 
            data_source=data_source, 
            resolution=resolution,
            **kwargs,
        )
    except Exception as e:
        print(f"Failed to download elevation data: {e}")
        # Fall back to coarser resolution
        if resolution != "01d":
            print("Trying with coarser resolution (01d)...")
            srtm = pygmt.datasets.load_earth_relief(
                region=region, 
                data_source=data_source, 
                resolution="01d",
                **kwargs,
            )
        else:
            raise
    
    # Generate points along the profile
    s = n - 1
    interval_lat = (p2[0] - p1[0]) / s
    interval_lon = (p2[1] - p1[1]) / s
    
    lat0, lon0 = p1[0], p1[1]
    lat_list = [lat0]
    lon_list = [lon0]
    
    for i in range(s):
        lat_step = lat0 + interval_lat
        lon_step = lon0 + interval_lon
        lon0 = lon_step
        lat0 = lat_step
        lat_list.append(lat_step)
        lon_list.append(lon_step)
    
    # Sample elevation data at profile points
    elev_list = []
    for lat, lon in zip(lat_list, lon_list):
        try:
            # Find the closest grid point
            elev = srtm.sel(lat=lat, lon=lon, method='nearest').data
            elev_list.append(float(elev))
        except:
            # If exact sampling fails, try interpolation
            try:
                elev = srtm.interp(lat=lat, lon=lon).data
                elev_list.append(float(elev))
            except:
                # If all else fails, use nearest neighbor from the grid
                try:
                    lat_idx = np.argmin(np.abs(srtm.lat.data - lat))
                    lon_idx = np.argmin(np.abs(srtm.lon.data - lon))
                    elev_list.append(float(srtm.data[lat_idx, lon_idx]))
                except:
                    # Last resort: use 0 elevation
                    if verbose:
                        print(f"Warning: Could not sample elevation at ({lat:.3f}, {lon:.3f}), using 0")
                    elev_list.append(0.0)
    
    # Calculate distances using haversine formula
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
    
    # Calculate cumulative distances
    d_list = []
    for j in range(len(lat_list)):
        lat_p = lat_list[j]
        lon_p = lon_list[j]
        dp = haversine(p1[0], p1[1], lat_p, lon_p) / 1000  # km
        d_list.append(dp)
    
    datadict = dict({
        'lat': np.array(lat_list), 
        'lon': np.array(lon_list),
        'd': np.array(d_list) * 1000,  # distance in meters
        'elev': np.array(elev_list),
        'elev_m': np.array(elev_list), 
        'elev_km': np.array(elev_list)/1000
    })
    
    if verbose:
        print('Done')
    
    return datadict

def download_profile_pygmt_dev(p1, p2, n=100):

    import pygmt

    # convert p1 to p2 to bounding box with x buffer
    region_map = []

    # auto-compute resolution
    resolution = "10m"

    # Download grid for Earth relief with a resolution of 10 arc-minutes and gridline
    # registration [Default]
    grid_map = pygmt.datasets.load_earth_relief(resolution=resolution, region=region_map)

    # Generate points along a great circle corresponding to the survey line and store them
    # in a pandas.DataFrame
    track_df = pygmt.project(
        center=[p1[1], p1[0]],  # Start point of survey line (longitude, latitude)
        endpoint=[p2[1], p2[0]],  # End point of survey line (longitude, latitude)
        generate=0.1,  # Output data in steps of 0.1 degrees
    )

    # Extract the elevation at the generated points from the downloaded grid and add it as
    # new column "elevation" to the pandas.DataFrame
    track_df = pygmt.grdtrack(grid=grid_map, points=track_df, newcolname="elevation")

    # Extract the elevation at the generated points from the downloaded grid and add it as
    # new column "elevation" to the pandas.DataFrame
    track_df = pygmt.grdtrack(grid=grid_map, points=track_df, newcolname="elevation")

    p = track_df.p
    d = p  # distance in meters, reversed
    elev = track_df.elevation

    datadict = dict({'lat': np.array([]), 'lon': np.array([]),
                     'd': np.array(d) * 1000,  # distance in meters
                     'elev': np.array(elev),
                     'elev_m': np.array(elev), 'elev_km': np.array(elev)/1000})

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

    api_url = f"{hosturl}{dataset}?locations=" + "{locstr}"

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
        dp = haversine(p1[0], p1[1], lat_p, lon_p) / 1000  # km - Fixed: use p1[0], p1[1] instead of lat0, lon0
        d_list.append(dp)


    # BATCH REQUESTS - OpenTopoData has limits on points per request
    max_points_per_request = 100  # Conservative limit
    elev_list = []
    
    # Create SSL context to handle certificate verification
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    if verbose:
        print(f" Making {((len(lat_list) - 1) // max_points_per_request) + 1} batch requests...")
    
    # Process points in batches
    for batch_start in range(0, len(lat_list), max_points_per_request):
        batch_end = min(batch_start + max_points_per_request, len(lat_list))
        batch_lat = lat_list[batch_start:batch_end]
        batch_lon = lon_list[batch_start:batch_end]
        
        # CONSTRUCT LOCATIONS STRING for this batch
        locations_str = ""
        for j in range(len(batch_lat)):
            if j > 0:
                locations_str += "|"
            locations_str += f"{batch_lat[j]},{batch_lon[j]}"

        # SEND REQUEST for this batch
        request_url = api_url.format(locstr=locations_str)
        if verbose:
            print(f" Batch {batch_start//max_points_per_request + 1} URL: {request_url[:100]}...")

        try:
            response = urllib.request.Request(request_url)
            fp = urllib.request.urlopen(response, context=ssl_context)

            # RESPONSE PROCESSING
            res_byte = fp.read()
            res_str = res_byte.decode("utf8")
            js_str = json.loads(res_str)
            fp.close()

            # GETTING ELEVATION for this batch
            for m in range(len(js_str["results"])):
                elev_list.append(js_str['results'][m]['elevation'])
                
        except Exception as e:
            if verbose:
                print(f" Error in batch {batch_start//max_points_per_request + 1}: {e}")
            # Fill with zeros for failed batch
            for _ in range(len(batch_lat)):
                elev_list.append(0.0)
        
        # Small delay between requests to be respectful to the API
        if batch_end < len(lat_list):
            import time
            time.sleep(0.1)

    datadict = dict({'lat': np.array(lat_list), 'lon': np.array(lon_list),
                     'd': np.array(d_list) * 1000,  # distance in meters
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

def download_profile_3dep(p1, p2, n=100, verbose=False):
    """Download elevation profile data from USGS 3DEP dataset.
    
    Parameters:
    -----------
    p1 : tuple
        Start point (lat, lon)
    p2 : tuple
        End point (lat, lon)
    n : int
        Number of points for the elevation profile
    verbose : bool
        Print verbose output
        
    Returns:
    --------
    dict : Dictionary containing lat, lon, distance, elevation arrays
    """
    print("WARNING: THIS METHOD IS UNVERIFIED.")

    try:
        import requests
        import numpy as np
        import math
    except ImportError:
        raise ImportError("Requests library required. Install with: pip install requests")

    if verbose:
        print("Downloading elevation data from USGS 3DEP...")

    # Generate points along profile
    s = n - 1
    interval_lat = (p2[0] - p1[0]) / s
    interval_lon = (p2[1] - p1[1]) / s

    lat0, lon0 = p1[0], p1[1]
    lat_list = [lat0]
    lon_list = [lon0]

    for i in range(s):
        lat_step = lat0 + interval_lat
        lon_step = lon0 + interval_lon
        lon0 = lon_step
        lat0 = lat_step
        lat_list.append(lat_step)
        lon_list.append(lon_step)

    # Query elevation points from 3DEP
    base_url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/identify"
    elev_list = []

    for lat, lon in zip(lat_list, lon_list):
        params = {
            'f': 'json',
            'geometryType': 'esriGeometryPoint',
            'geometry': f'{{"x":{lon},"y":{lat},"spatialReference":{{"wkid":4326}}}}',
            'returnGeometry': False
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            elevation = data['value']
            elev_list.append(elevation)
        except:
            if verbose:
                print(f"Warning: Could not get elevation at ({lat:.3f}, {lon:.3f}), using 0")
            elev_list.append(0.0)

    # Calculate distances
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

    d_list = []
    for j in range(len(lat_list)):
        lat_p = lat_list[j]
        lon_p = lon_list[j]
        dp = haversine(p1[0], p1[1], lat_p, lon_p) / 1000  # km
        d_list.append(dp)

    datadict = dict({
        'lat': np.array(lat_list),
        'lon': np.array(lon_list),
        'd': np.array(d_list) * 1000,  # distance in meters
        'elev': np.array(elev_list),
        'elev_m': np.array(elev_list),
        'elev_km': np.array(elev_list) / 1000
    })

    if verbose:
        print('Done')

    return datadict

def download_profile_alt():
    # https://gis.stackexchange.com/questions/29632/getting-elevation-at-lat-long-from-raster-using-python
    # https://developers.google.com/maps/documentation/elevation/start#maps_http_elevation_locations-py
    pass

def write(d, elev, file='elevation.csv'):
    import csv

    rows = zip(d, elev)

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

def plot_profile(h, z, depth_range=[-50., 4.], depth=None, color='black', linewidth=0.75,
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

def test_download():
    import matplotlib.pyplot as plt
    data = download_profile((-8.169148, 115.283046), (-8.579110, 115.819991))  # Includes bathymetry
    # data = download_profile((-8.2359, 115.5995), (-8.4145, 115.4465))
    fig, ax = plot_profile(data["d"], data["elev"], depth=0)
    print(ax)
    ax.plot(31, 2806, color='k')
    plt.show()

def test_topoprofile_class():

    tp = TopographicProfile((-8.169148, 115.283046), (-8.579110, 115.819991), resolution="auto")
    tp.plot()

def test_source_comparison():
    """Compare different data sources for elevation profiles."""
    import matplotlib.pyplot as plt
    
    p1 = (-8.169148, 115.283046)
    p2 = (-8.579110, 115.819991)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    sources = ["pygmt", "opentopo"]
    titles = ["PyGMT", "OpenTopoData"]
    
    for i, (source, title) in enumerate(zip(sources, titles)):
        try:
            data = download_profile(p1, p2, n=50, source=source, verbose=True)
            ax = axes[i]
            ax.plot(data["d"]/1000, data["elev"], 'b-', linewidth=2)
            ax.set_title(f"{title} Elevation Profile")
            ax.set_xlabel("Distance (km)")
            ax.set_ylabel("Elevation (m)")
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax = axes[i]
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title} - Failed")
    
    # Hide unused subplots
    for i in range(len(sources), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_download()
    test_topoprofile_class()
    test_source_comparison()
