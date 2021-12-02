def str2nslc( station_code, order='nslc', sep='.', newsep='.' ):
    "Convert any NSLC/SCNL/SCN str to its components"
        
    if order=='nslc':
        n = 0; s=1; l=2; c=3

    elif order=='scnl':
        s = 0; c=1; n=2; l=3

    elif order=='scn':
        sta+sep
        s = 0; c=1; n=2; l=3

    network  = station_code.split(sep)[n]
    station  = station_code.split(sep)[s]
    location = station_code.split(sep)[l]
    channel  = station_code.split(sep)[c]
        
    return network, station, location, channel


def convertNSLCstr( station_code, order='nslc', neworder='scnl', sep='.', newsep='.'):
    "Convert any NSLC/SCNL/SCN str to another NSLC/SCNL/SCN str"
    
    network, station, location, channel = str2nslc(station_code, order=order, sep=sep, newsep=newsep)
    return build_str(network, station, location, channel, order=neworder, sep=newsep)
    
#     # build string
#     if neworder=='nslc':
#         syntax = '{0}{4}{1}{4}{2}{4}{3}'
#     elif neworder=='scnl':
#         syntax = '{1}{4}{3}{4}{0}{4}{2}'
#     elif neworder=='scn':
#         syntax = '{1}{4}{3}{4}{0}'
#     else:
#         print('Order {} not understood'.format(neworder))
    
#     return syntax.format(network, station, location, channel, newsep)    


def setNSLC( st, nslc_string, sep='.' ):
    
    network, station, location, channel = str2nslc(nslc_string)
    
    for i in range(len(st)):
        st[i].stats.network  = network
        st[i].stats.station  = station
        st[i].stats.location = location
        st[i].stats.channel  = channel
    
    return st


def getNSLCstr( tr, order='nslc', sep='.'):
    
    network  = tr.stats.network
    station  = tr.stats.station
    location = tr.stats.location
    channel  = tr.stats.channel

    return build_str(network, station, location, channel, order=order, sep=sep)

    
#     # build string
#     if order=='nslc':
#         syntax = '{0}{4}{1}{4}{2}{4}{3}'
#     elif order=='scnl':
#         syntax = '{1}{4}{3}{4}{0}{4}{2}'
#     elif order=='scn':
#         syntax = '{1}{4}{3}{4}{0}'
#     else:
#         print('Order {} not understood'.format(order))
    
#     return syntax.format(network, station, location, channel, sep)


def build_str( network, station, location, channel, order='nslc', sep='.' ):
    
    #print('!!! This can be simplified with Trace.id for NSLC order')
    
    if order=='nslc':
        syntax = '{0}{4}{1}{4}{2}{4}{3}'
    elif order=='scnl':
        syntax = '{1}{4}{3}{4}{0}{4}{2}'
    elif order=='scn':
        syntax = '{1}{4}{3}{4}{0}'
    else:
        print('Order {} not understood'.format(order))
    
    return syntax.format(network, station, location, channel, sep)