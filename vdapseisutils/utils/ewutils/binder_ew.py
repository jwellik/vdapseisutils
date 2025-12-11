import os
import pandas as pd


# Define a function to parse "hyp" lines
def parse_hypocenter(log_file):
    print("WARNING: This code currently returns a DataFrame. Soon, it will return a Catalog object.")

    hypocenters = []  # List to store parsed hypocenter dictionaries

    with open(log_file, 'r') as file:
        for line in file:
            # Check if the line contains "hyp :"
            if "hyp :" in line:
                try:
                    # Split the line into parts
                    parts = line.strip().split()

                    # Extract data based on the format provided
                    datetime_proc = parts[0]
                    eqid = parts[3]
                    datestamp = parts[4]
                    hrminstamp = parts[5]
                    secondsstamp = float(parts[6])
                    latitude = float(parts[7])
                    longitude = float(parts[8])
                    depth = float(parts[9])
                    nsta = int(parts[10])

                    # Parse hrminstamp into hours and minutes
                    if len(hrminstamp) == 3:
                        hours = int(hrminstamp[0])
                        minutes = int(hrminstamp[1:])
                    elif len(hrminstamp) == 4:
                        hours = int(hrminstamp[:2])
                        minutes = int(hrminstamp[2:])
                    else:
                        raise ValueError(f"Invalid hrminstamp format: {hrminstamp}")

                    # Combine date and time into a pandas datetime object
                    datetime_str = f"{datestamp} {hours:02}:{minutes:02}:{secondsstamp:.2f}"
                    datetime = pd.to_datetime(datetime_str, format="%Y%b%d %H:%M:%S.%f")

                    # Create a dictionary for the parsed data
                    hypocenter = {
                        "datetime_proc": datetime_proc,
                        "eqid": eqid,
                        "datetime": datetime,
                        "latitude": latitude,
                        "longitude": longitude,
                        "depth": depth,
                        "nsta": nsta,
                        "picks": []
                    }

                    # Find the first "pck :" line and parse subsequent lines
                    while True:
                        next_line = file.readline().strip()
                        if not next_line:
                            break
                        if next_line.startswith("pck :"):
                            while next_line.startswith("pck :"):
                                pck_parts = next_line.split()
                                pick = {
                                    "sta": pck_parts[2],
                                    "cha": pck_parts[3],
                                    "net": pck_parts[4],
                                    "loc": pck_parts[5],
                                    "phaseHint": pck_parts[6],
                                    "quality": pck_parts[7][0],
                                    "first_motion": pck_parts[7][1],
                                    "min": float(pck_parts[8]),
                                    "sec": float(pck_parts[9]),
                                    "arg1": float(pck_parts[10]),
                                    "arg2": pck_parts[11]
                                }
                                hypocenter["picks"].append(pick)
                                next_line = file.readline().strip()
                            break

                    # Add the dictionary to the list
                    hypocenters.append(hypocenter)
                except (IndexError, ValueError) as e:
                    print(f"Error parsing line: {line.strip()}\n{e}")

    # Convert the list of dictionaries to a DataFrame for easier manipulation
    hypocenter_df = pd.DataFrame(hypocenters)

    # Sort by datetime_proc oldest to newest
    hypocenter_df = hypocenter_df.sort_values(by="datetime_proc")

    # Remove duplicates, keeping the newest one
    hypocenter_df = hypocenter_df.drop_duplicates(subset="eqid", keep="last")

    return hypocenter_df


def write(template="./params_templates/binder_ew.d", site_file="utils/sta.hinv", velocityd="utils/velocity_model.d",
              lat=0.0, lon=0.0, radius=100, gridz=[0, 100],
              dspace=3.0, rstack=100, tstack=0.6, stack=100, thresh=16, focus=100,
              grid_wt=[4, 4, 4, 4],
              no_P_on_Horiz=True, no_S_on_Z=True,
              name=None,
              verbose=True, outfile=None):

    module_path = os.path.dirname(__file__)
    template_file = os.path.join(module_path, template)
    del template
    with open(template_file, "r") as f:
        template = f.read()

    # Define gassociation grid, set stacking parameters
    # determine grdlat minlat maxlat
    # determine grdlon minlon maxlon
    # determine grdz   minz   maxz
    from vdapseisutils.utils.geoutils import radial_extent2map_extent
    map_extent = radial_extent2map_extent(lat, lon, radius)
    minlon, maxlon, minlat, maxlat = map_extent
    minz, maxz = gridz

    # Set rstack
    # Set tstack
    # Set stack
    # Set thresh
    # Set focus

    # Writ grid_wt lines (loop through grid_wt)
    #  grid_wt 0  4
    #  grid_wt 1  4
    #  grid_wt 2  4
    #  grid_wt 3  4
    grid_wt_lines = ""
    for i, wt in enumerate(grid_wt):
        grid_wt_lines += f" grid_wt {i}  {wt}\n"

    # no_P_on_Horiz - Uncomment or comment
    # no_S_on_Z     - Uncomment or comment
    if no_P_on_Horiz:
        no_P_on_Horiz = "no_P_on_Horiz"
    else:
        no_P_on_Horiz = "# no_P_on_Horiz"
    if no_S_on_Z:
        no_S_on_Z = "no_S_on_Z"
    else:
        no_S_on_Z = "# no_S_on_Z"

    template = template.format(name=name, datetime=datetime.datetime.now(), site_file=site_file, velocityd=velocityd,
                    dspace=dspace, minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon, minz=minz, maxz=maxz,
                    rstack=rstack, tstack=tstack, stack=stack, thresh=thresh, focus=focus,
                    grid_wt_lines=grid_wt_lines,
                    no_P_on_Horiz=no_P_on_Horiz, no_S_on_Z=no_S_on_Z)

    # print and write
    if verbose:
        print(template)
        print()
    if outfile:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)  # ensure that the file and abspath exist
        with open(outfile, "w") as f:
            f.write(template)

    return template


def test_parse_hypocenter():

    # Example usage
    log_file = "binder_ew.log"  # Replace with the path to your log file

    # Example log content
    example_log = """20241212_UTC_02:42:34   hyp : 21291 2024Aug25  505 35.89   3.5918  125.4589   5.81   4
      par : dmin = 7.5, ravg = 9.7, gap = 313.0
      pck : KLGN  BHZ VG 00 P  0D  7.53  37.54 -0.31 <1.00>  
      pck : BEHA  EHZ VG 00 P  0D 11.79  38.73  0.16 <1.00>  
      pck : KNDH  EHZ VG 00 P  0U 10.86  38.41  0.00 <1.00>  
      pck : AWU1  EHZ VG 00 P  0D  8.80  37.62 -0.44 <1.00>  
      rms :   0.23, dmin =   7.53
    """


    # # Example usage
    # log_file = "./binder_ew.log"  # Replace with the path to your log file

    # Write the example log to a temporary file for testing
    # with open(log_file, 'w') as f:
    #     f.write(example_log)

    log_file = "/home/jwellik/Downloads/binder_ew_2024-08-24_25-hyps.log"
    parsed_hypocenters = parse_hypocenter(log_file)

    # Convert the list of dictionaries to a DataFrame for easier manipulation (optional)
    hypocenter_df = pd.DataFrame(parsed_hypocenters)
    print(hypocenter_df)
