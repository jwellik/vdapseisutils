# UTILS for ObsPy Inventory class

def write_simple_csv(inventory, filename='~/inventory.csv', verbose=False):
    """Writes inventory as CSV formatted channel,latitude,longitude,elevation"""
    import pandas as pd

    stationdf = pd.DataFrame({"nslc": [], "latitude": [], "longitude": [], "elevation": [], "local_depth":[]})

    for cha in inventory.get_contents()['channels']:
        print(cha)
        coords = inventory.get_coordinates(cha)
        coords["nslc"] = [cha]
        coords["latitude"] = [coords["latitude"]]
        coords["longitude"] = [coords["longitude"]]
        coords["elevation"] = [coords["elevation"]]
        coords["local_depth"] = [coords["local_depth"]]
        # stationdf = stationdf.append(coords, ignore_index=True)
        stationdf = pd.concat([stationdf, pd.DataFrame(coords)])
    # stationdf = stationdf[['channel', 'latitude', 'longitude', 'elevation']]
    if verbose:
        print(stationdf)
    stationdf.to_csv(filename, index=False)
    return stationdf


def write_swarm(inventory, filename='~/inventorylatlon.config', verbose=False):
    """Writes inventory as CSV formatted channel,latitude,longitude,elevation"""

    from vdapseisutils.waveformutils.nslcutils import convertNSLCstr

    # first brack is SCNL in format 'STA CHA NET LOC'
    swarm_format = '{}= Longitude: {}; Latitude: {}; Height: {}'

    channel_strings = []

    for cha in inventory.get_contents()['channels']:
        coords = inventory.get_coordinates(cha)
        cha = convertNSLCstr(cha, order='nslc', sep='.', neworder='scnl', newsep=' ')
        cha_str = swarm_format.format(cha, coords['longitude'], coords['latitude'], coords['elevation'])
        channel_strings.append(cha_str)
        if verbose: print(cha_str)

    with open(filename, 'w') as f:
        for line in channel_strings:
            f.write(line)
            f.write('\n')

