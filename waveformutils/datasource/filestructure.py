# vdapseisutils.waveformutils.datasource.filestructure_v1.py


def get_filelist_all(searchdir, filepattern='*'):
     import glob, os, itertools

     # create an if file thing here?
     if os.path.isdir(searchdir):
          return list(itertools.chain.from_iterable(glob.iglob(os.path.join(
               root, filepattern)) for root, dirs, files in os.walk(searchdir)))  
     else:
          print('{} is not a directory.'.format(searchdir))


def parse_filelist_for_nslc_time(flist, nslc_list, tstart, tend):
     '''NOTE: This does not support wildcards in nslc_list'''

     import obspy
     from obspy import UTCDateTime
     from vdapseisutils.waveformutils.nslcutils import getNSLCstr
     import glob, os, itertools


     print('::: vdapseisutils.waveformutils.datasource.filestructure.parse_filelist_for_nslc_time()')
     print('- Parsing filelist headers. This may take some time if filelist is long...')

     tstart = UTCDateTime(tstart)
     tend   = UTCDateTime(tend)

     # Create file sublist (only files w relevant nslc & time)
     # Determine which subset of files to load based on start and end times and
     # station name; we'll fully deal with stations below
    
     flist_sub = []
     for f in flist:
          # Load header only
          stmp = obspy.read(f, headonly=True)
          # Check if station is contained in the stas list
          if getNSLCstr(stmp[0]) in nslc_list:  # if stmp[0].stats.station in stas:
               # Check if contains either start or end time
               ststart = stmp[0].stats.starttime
               stend = stmp[-1].stats.endtime
               if (ststart <= tstart <= stend) or (ststart <= tend <= stend) or (tstart <= stend and ststart <= tend):
                    flist_sub.append(f)
    
     return flist_sub
     
     
def parse_filelist_for_nslc_time_regexp(flist, nslc_exp, tstart, tend):
     '''Ultimately make this cover all use cases'''     
     
     import re

     import obspy
     from obspy import UTCDateTime
     from vdapseisutils.waveformutils.nslcutils import getNSLCstr, str2nslc



     print('::: vdapseisutils.waveformutils.datasource.filestructure.parse_filelist_for_nslc_time_regexp()')
     print('- Parsing filelist headers. This may take some time if filelist is long...')

     tstart = UTCDateTime(tstart)
     tend   = UTCDateTime(tend)

     # Create file sublist (only files w relevant nslc & time)
     # Determine which subset of files to load based on start and end times and
     # station name; we'll fully deal with stations below
    
     flist_sub = []
     for f in flist:
          # Load header only
          stmp = obspy.read(f, headonly=True)
          # Check if station is contained in NSLC regular expression
          if bool(re.match( re.compile(nslc_exp), getNSLCstr(stmp[0]))):
               # Check if contains either start or end time
               ststart = stmp[0].stats.starttime
               stend = stmp[-1].stats.endtime
               if (ststart <= tstart <= stend) or (ststart <= tend <= stend) or (tstart <= stend and ststart <= tend):
                    flist_sub.append(f)
    
     return flist_sub
