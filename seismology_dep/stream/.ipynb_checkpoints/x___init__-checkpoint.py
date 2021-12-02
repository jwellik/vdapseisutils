#import datasource
#import nslcobject

from obspy import UTCDateTime
import obspy
from obspy.clients.fdsn import Client
from obspy.clients.earthworm import Client as EWClient
from obspy.clients.seedlink import Client as SeedLinkClient
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy.signal.trigger import coincidence_trigger
import numpy as np
from scipy import stats
from scipy.fftpack import fft
import glob, os, itertools

import warnings
warnings.filterwarnings("ignore")

# def getWaveforms
# def getCleanWaveforms
# def downloadWaveforms(datatype='mseed')

"""
def getCleanStream(...)

    1. Create Client                     seismology_dep.stream.datasource.createClient()
    2. Loop through NSLCs                seismology_dep.stream.nslcobject.str2nslc()
        3. Create time chunks            timeutils.createTimeChunks()
           Loop through time chunks      

            4. Retrieve data             obspy.get_waveform()
            5. Clean data                seismology_dep.stream.cleanStream()
               Filter
            6. Merge data                obspy.stream.merge()

            4. or create empty stream    seismology_dep.stream.createEmptyStream()
            
        7. Merge data                    obspy.stream.merge()
    

"""




def getStream2( ds, nslc_list, tstart, tend, nsec=3600, station_code='nslc', verbose=False ):

    """
    Download data from files in a folder, from IRIS, or a Earthworm waveserver
    A note on SAC/miniSEED files: as this makes no assumptions about the naming scheme of
    your data files, please ensure that your headers contain the correct SCNL information!
    tstart: UTCDateTime of beginning of period of interest
    tend: UTCDateTime of end of period of interest
    nsec: <unimplemented>
    nprc: <unimplemented>
    
    Returns ObsPy stream objects, one for cutting and the other for triggering
    """
    
    from seismology.stream.nslcobject import str2nslc
    
    # Convert times to UTCDateTime
    tstart = UTCDateTime(tstart)
    tend = UTCDateTime(tend)

    st = Stream()
    
    # Create client
    server = ds.split(':')[0]
    port   = int(ds.split(':')[1])
    client = EWClient(server, port)
    if verbose: print('>>> Downloading data from {}:{}'.format(server,port)) # reprint this after it's been parsed for verification to user

    # Loop through NSLC codes
    for nslc in nslc_list:
        
#         net = nslc.split('.')[0]
#         sta = nslc.split('.')[1]
#         loc = nslc.split('.')[2]
#         cha = nslc.split('.')[3]

        net, sta, loc, cha = str2nslc(nslc, order=station_code)
        
        try:
            if verbose: print('    - {} ({} - {})'.format(nslc, tsart, tend)) # reprint this after it's been parsed for verification to user
            stmp = client.get_waveforms(net, sta, loc, cha, tstart, tend)
            
            # remove Winston Gaps
            for m in range(len(stmp)):
                stmp[m].data = np.where(stmp[m].data == -2**31, 0, stmp[m].data) # replace -2**31 (Winston NaN token) w 0  
            #stmp = [np.where(stmp2.data == -2**31, 0, stmp2.data) for stmp2 in stmp] # replace -2**31 (Winston NaN token) w 0 <--- Could this be a 1 liner?
                        
        except:
            
            stmp = Stream()
            
    return stmp


def getStream1( ds, nslc_list, tstart, tend, station_code='nslc', nsec=3600 ):
    """
    GET STREAM Loads stream from server or file source.

        datasource : 
        nslc_list :
        tstart :
        tend :
        station_code
        nsec : maximum amount of waveform to get at a time before merging


    A lot of code borrowed from Alicia Hotovec-Ellis and Aaron Wech
    """
    import createClient
    #import _get_waveform_from_server
    #import _create_empty_trace
    from seismology.stream.nslcobject import str2nslc


    client = createClient(ds)
    
    st = Stream()

    for sta in nslc:
        
        #if station_code=='nslc':
        #    n = 0; s=1; l=2; c=3
        #    
        #elif station_code=='scnl':
        #    s = 0; c=1; n=2; l=3
        #    
        #elif station_code=='scn':
        #    sta+'.'
        #    s = 0; c=1; n=2; l=3
        #    
        #network  = sta.split('.')[n]
        #station  = sta.split('.')[s]
        #location = sta.split('.')[l]
        #channel  = sta.split('.')[c]
        
        network, station, location, channel = str2nslc(sta, order=station_code)


        try: # first attempt
            
            stmp = _get_waveform_from_server(client, network, station, location, channel, tstart, tend)
            stmp = stmp.merge(method=1, fill_value=0)
            
        except (obspy.clients.fdsn.header.FDSNException):
            try: # try again
                
                stmp = _get_waveform_from_server(client, network, station, location, channel, tstart, tend)
                stmp = stmp.merge(method=1, fill_value=0)
                
            except (obspy.clients.fdsn.header.FDSNException):
                print('No data found for {}.{}.{}.{}'.format(network,station,location,channel))
                trtmp = _create_empty_trace(network, station, location, channel, samprate)
                stmp = Stream().extend([trtmp.copy()])

        # Last check for length; catches problem with empty waveserver
        if len(stmp) != 1:
            print('No data found for {}.{}.{}.{}'.format(network,station,location,channel))
            trtmp = _create_empty_trace(network, station, location, channel, samprate)
            stmp = Stream().extend([trtmp.copy()])

        st.extend(stmp.copy())

        st = st.trim(starttime=tstart, endtime=tend, pad=True, fill_value=0) # This should be unnecessary, correct?
        stC = st.copy()

        return st, stC

def removeWinstonGaps( st, winston_gap_value=-2**31, fill_value=0 ):   
    for m in range(len(st)):
        st[m].data = np.where(st[m].data == winston_gap_value, fill_value, st[m].data) # replace -2**31 (Winston NaN token) w 0  
    #stmp = [np.where(stmp2.data == -2**31, 0, stmp2.data) for stmp2 in stmp] # replace -2**31 (Winston NaN token) w 0 <--- Could this be a 1 liner?
    return st




def getStreamWech(scnl,T1,T2,fill_value=0):
	# scnl = list of station names (eg. ['PS4A.EHZ.AV.--','PVV.EHZ.AV.--','PS1A.EHZ.AV.--'])
	# T1 and T2 are start/end obspy UTCDateTimes
	# fill_value can be 0 (default), 'latest', or 'interpolate'
	#
	# returns stream of traces with gaps accounted for
	#
    
#     """
#     get_waveform
#     if fill_value==0 or fill_value==None
#         detrend
#         taper
#     fix dtype for data
#     merge
#     -create empty trace w zeros if no data
    
    
    
#     """
    
    
	print('{} - {}'.format(T1.strftime('%Y.%m.%d %H:%M:%S'),T2.strftime('%Y.%m.%d %H:%M:%S')))
	print('Grabbing data...')

	st=Stream()

	t_test1=UTCDateTime.now()
	for sta in scnl:
		if sta.split('.')[2]=='MI':
			client = Client(os.environ['CNMI_WINSTON'], int(os.environ['CNMI_PORT']))
		else:
			client = Client(os.environ['WINSTON_HOST'], int(os.environ['WINSTON_PORT']))
		try:
			tr=client.get_waveforms(sta.split('.')[2], sta.split('.')[0],sta.split('.')[3],sta.split('.')[1], T1, T2, cleanup=True)
			if len(tr)>1:
				print('{:.0f} traces for {}'.format(len(tr),sta))
				if fill_value==0 or fill_value==None:
					tr.detrend('demean')
					tr.taper(max_percentage=0.01)
				for sub_trace in tr:
					# deal with error when sub-traces have different dtypes
					if sub_trace.data.dtype.name != 'int32':
						sub_trace.data=sub_trace.data.astype('int32')
					if sub_trace.data.dtype!=dtype('int32'):
						sub_trace.data=sub_trace.data.astype('int32')
					# deal with rare error when sub-traces have different sample rates
					if sub_trace.stats.sampling_rate!=round(sub_trace.stats.sampling_rate):
						sub_trace.stats.sampling_rate=round(sub_trace.stats.sampling_rate)
				print('Merging gappy data...')
				tr.merge(fill_value=fill_value)
		except:
			tr=Stream()
		# if no data, create a blank trace for that channel
		if not tr:
			tr=Trace()
			tr.stats['station']=sta.split('.')[0]
			tr.stats['channel']=sta.split('.')[1]
			tr.stats['network']=sta.split('.')[2]
			tr.stats['location']=sta.split('.')[3]
			tr.stats['sampling_rate']=100
			tr.stats['starttime']=T1
			tr.data=zeros(int((T2-T1)*tr.stats['sampling_rate']),dtype='int32')
		st+=tr
	print('{} seconds'.format(UTCDateTime.now()-t_test1))
	
	print('Detrending data...')
	st.detrend('demean')
	st.trim(T1,T2,pad=0)
	return st


def getStreamAHE(tstart, tend, opt):

    """
    Download data from files in a folder, from IRIS, or a Earthworm waveserver
    A note on SAC/miniSEED files: as this makes no assumptions about the naming scheme of
    your data files, please ensure that your headers contain the correct SCNL information!
    tstart: UTCDateTime of beginning of period of interest
    tend: UTCDateTime of end of period of interest
    opt: Options object describing station/run parameters
    Returns ObsPy stream objects, one for cutting and the other for triggering
    """
    
    """
    Parse NSLC
    Create client or get filelist
    
    get_waveform
    remove Winston gap
    filter
    taper
    fix sample rate
    -create empty trace if no data (doesn't add data)
    
    """

    nets = opt.network.split(',')
    stas = opt.station.split(',')
    locs = opt.location.split(',')
    chas = opt.channel.split(',')

    st = Stream()

    if opt.server == 'file':

        # Generate list of files
        if opt.server == 'file':
            flist = list(itertools.chain.from_iterable(glob.iglob(os.path.join(
                root,opt.filepattern)) for root, dirs, files in os.walk(opt.searchdir)))

        # Determine which subset of files to load based on start and end times and
        # station name; we'll fully deal with stations below
        flist_sub = []
        for f in flist:
            # Load header only
            stmp = obspy.read(f, headonly=True)
            # Check if station is contained in the stas list
            if stmp[0].stats.station in stas:
                # Check if contains either start or end time
                ststart = stmp[0].stats.starttime
                stend = stmp[-1].stats.endtime
                if (ststart<=tstart and tstart<=stend) or (ststart<=tend and
                    tend<=stend) or (tstart<=stend and ststart<=tend):
                    flist_sub.append(f)

        # Fully load data from file
        stmp = Stream()
        for f in flist_sub:
            tmp = obspy.read(f, starttime=tstart, endtime=tend+opt.maxdt)
            if len(tmp) > 0:
                stmp = stmp.extend(tmp)

        # Filter and merge
        stmp = stmp.filter('bandpass', freqmin=opt.fmin, freqmax=opt.fmax, corners=2,
            zerophase=True)
        stmp = stmp.taper(0.05,type='hann',max_length=opt.mintrig)
        for m in range(len(stmp)):
            if stmp[m].stats.sampling_rate != opt.samprate:
                stmp[m] = stmp[m].resample(opt.samprate)
        stmp = stmp.merge(method=1, fill_value=0)

        # Only grab stations/channels that we want and in order
        netlist = []
        stalist = []
        chalist = []
        loclist = []
        for s in stmp:
            stalist.append(s.stats.station)
            chalist.append(s.stats.channel)
            netlist.append(s.stats.network)
            loclist.append(s.stats.location)

        # Find match of SCNL in header or fill empty
        for n in range(len(stas)):
            for m in range(len(stalist)):
                if (stas[n] in stalist[m] and chas[n] in chalist[m] and nets[n] in
                    netlist[m] and locs[n] in loclist[m]):
                    st = st.append(stmp[m])
            if len(st) == n:
                print("Couldn't find "+stas[n]+'.'+chas[n]+'.'+nets[n]+'.'+locs[n])
                trtmp = Trace()
                trtmp.stats.sampling_rate = opt.samprate
                trtmp.stats.station = stas[n]
                st = st.append(trtmp.copy())

    else:

        if '://' not in opt.server:
            # Backward compatibility with previous setting files
            if '.' not in opt.server:
                client = Client(opt.server)
            else:
                client = EWClient(opt.server, opt.port)
        # New server syntax (more options and server and port on same variable)
        elif 'fdsnws://' in opt.server:
            server = opt.server.split('fdsnws://',1)[1]
            client = Client(server)
        elif 'waveserver://' in opt.server:
            server_str = opt.server.split('waveserver://',1)[1]
            try:
                server = server_str.split(':',1)[0]
                port = server_str.split(':',1)[1]
            except:
                server = server_str
                port = '16017'
            client = EWClient(server, int(port))
        elif 'seedlink://' in opt.server:
            server_str = opt.server.split('seedlink://',1)[1]
            try:
                server = server_str.split(':',1)[0]
                port = server_str.split(':',1)[1]
            except:
                server = server_str
                port = '18000'
            client = SeedLinkClient(server, port=int(port), timeout=1)

        for n in range(len(stas)):
            try:
                stmp = client.get_waveforms(nets[n], stas[n], locs[n], chas[n],
                        tstart, tend+opt.maxdt)
                for m in range(len(stmp)):
                    stmp[m].data = np.where(stmp[m].data == -2**31, 0, stmp[m].data) # replace -2**31 (Winston NaN token) w 0
                stmp = stmp.filter('bandpass', freqmin=opt.fmin, freqmax=opt.fmax,
                    corners=2, zerophase=True)
                stmp = stmp.taper(0.05,type='hann',max_length=opt.mintrig)
                for m in range(len(stmp)):
                    if stmp[m].stats.sampling_rate != opt.samprate:
                        stmp[m] = stmp[m].resample(opt.samprate)
                stmp = stmp.merge(method=1, fill_value=0)
            except (obspy.clients.fdsn.header.FDSNException):
                try: # try again
                    stmp = client.get_waveforms(nets[n], stas[n], locs[n], chas[n],
                            tstart, tend+opt.maxdt)
                    for m in range(len(stmp)):
                        stmp[m].data = np.where(stmp[m].data == -2**31, 0, stmp[m].data) # replace -2**31 (Winston NaN token) w 0
                    stmp = stmp.filter('bandpass', freqmin=opt.fmin, freqmax=opt.fmax,
                        corners=2, zerophase=True)
                    stmp = stmp.taper(0.05,type='hann',max_length=opt.mintrig)
                    for m in range(len(stmp)):
                        if stmp[m].stats.sampling_rate != opt.samprate:
                            stmp[m] = stmp[m].resample(opt.samprate)
                    stmp = stmp.merge(method=1, fill_value=0)
                except (obspy.clients.fdsn.header.FDSNException):
                    print('No data found for {0}.{1}'.format(stas[n],nets[n]))
                    trtmp = Trace()
                    trtmp.stats.sampling_rate = opt.samprate
                    trtmp.stats.station = stas[n]
                    stmp = Stream().extend([trtmp.copy()])

            # Last check for length; catches problem with empty waveserver
            if len(stmp) != 1:
                print('No data found for {0}.{1}'.format(stas[n],nets[n]))
                trtmp = Trace()
                trtmp.stats.sampling_rate = opt.samprate
                trtmp.stats.station = stas[n]
                stmp = Stream().extend([trtmp.copy()])

            st.extend(stmp.copy())

    # Edit 'start' time if using offset option
    if opt.maxdt:
        dts = np.fromstring(opt.offset, sep=',')
        for n, tr in enumerate(st):
            tr.stats.starttime = tr.stats.starttime-dts[n]

    st = st.trim(starttime=tstart, endtime=tend, pad=True, fill_value=0)
    stC = st.copy()

    return st, stC


def createEmptyTrace( nslc, starttime, endtime, sampling_rate=100, fill_value=0):
    from seismology.stream.nslcobject import str2nslc
    from numpy import zeros
    
    net, sta, loc, cha = str2nslc(nslc)
    starttime = UTCDateTime(starttime)
    endtime   = UTCDateTime(endtime)
    
    tr=Trace()
    tr.stats['station']=sta
    tr.stats['channel']=cha
    tr.stats['network']=net
    tr.stats['location']=loc
    tr.stats['sampling_rate']=sampling_rate
    tr.stats['starttime']=starttime
    tr.data=zeros(int((endtime-starttime)*tr.stats['sampling_rate']),dtype='int32')+fill_value
    
    return tr


def trace_string( tr ):
    "Returns a string equivalent to print(tr)"
    return '{} | {} to {} | {} Hz, {} samples'.format(
        tr.id,
        tr.stats.starttime,
        tr.stats.endtime,
        tr.stats.sampling_rate,
        tr.stats.npts
    )