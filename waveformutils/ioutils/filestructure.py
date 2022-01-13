# Various filestructure/filename formats

# soon to be deprecated
sds_standard     = 'BASEDIR/YEAR/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.JDAY.EXTENSION'
sds_standard_ext = 'BASEDIR/YEAR/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEARMONTHDATE-HOURMINUTESECOND.EXTENSION'
sds_single       = 'BASEDIR/NET.STA.LOC.CHAN.TYPE.YEAR.JDAY.EXTENSION'
sds_single_ext   = 'BASEDIR/NET.STA.LOC.CHAN.TYPE.YEARMONTHDATE-HOURMINUTESECOND.EXTENSION'
swarm            = 'BASEDIR/YEARMONTHDATEHOURMINUTESECOND/STA_CHAN_NET_LOC.EXTENSION'
swarm_ext        = 'BASEDIR/YEAR/NET/STA/CHAN.TYPE/STA_CHAN_NET_LOC-YEARMONTHDATE-HOURMINUTESECOND.EXTENSION'
swarm_single_ext = 'BASEDIR/STA_CHAN_NET_LOC-YEARMONTHDATE-HOURMINUTESECOND.EXTENSION'


# Filestructure and Filename Standards
sds_filestructure   = 'YEAR/NET/STA/CHAN.TYPE/'
sds_filename        = 'NET.STA.LOC.CHAN.TYPE.YEAR.JDAY.EXTENSION'
sds_standard        = sds_filestructure + sds_filename

swarm_filestructure = 'YEARMONTHDATEHOURMINUTESECOND/'
swarm_filename      = 'STA_CHAN_NET_LOC.EXTENSION'
swarm_standard      = swarm_filestructure + swarm_filename


sds_starttime        = 'NET.STA.LOC.CHAN.TYPE.YEARMONTHDATE-HOURMINUTESECOND.EXTENSION'
swarm_starttime      = 'STA_CHAN_NET_LOC-YEARMONTHDATE-HOURMINUTESECOND.EXTENSION'
nslc_starttime       = 'NET.STA.LOC.CHAN-YEARMONTHDATE-HOURMINUTESECOND.EXTENSION'


def write2sds( st, basedir='./',
                filestructure=sds_standard,      # filestructure syntax
                #filename=sds_filename,          # filename syntax
                fileformat='mseed',              # seismic data format
                reclen=4096,                     # miniseed byte record-length
             ):
    """
    WRITE2SDS Writes Traces to file in accordance w given filestructure

    Default filestructure creates a SDS archive. All necessary folders will be created.
    
    filestructure : str : Syntax for filestructure. Options are:
        Default: 'BASEDIR/YEAR/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.JDAY.EXTENSION'
        Options:
        BASEDIR        #%BASEDIR
        NET            #%NN
        STA            #%SSSS
        LOC            #%LL
        CHAN           #%CC
        TYPE           #%T
        YEAR           #%YYYY
        JDAY           #%JDAY
        DATE           #%DD
        HOUR           #%HH
        MINUTE         #%MM
        SECOND         #%SS
        #MISCROSEC     #%FFFF
        EXTENSION      #%EXT
    """
    
    import os
    from pathlib import Path    

    filestructure = 'BASEDIR/' + filestructure # Add BASEDIR to path
    output_files  = [] # list of final output files
    
    for tr in st:
        
        #################################################################
        # PARSE INFO FROM TRACE
        
        # Parse NSLC information from Stream (returned as string)
        network  = tr.stats.network
        station  = tr.stats.station
        location = tr.stats.location
        channel  = tr.stats.channel
        
        # Assert datatype 'D'
        datatype = 'D'

        # Parse time information from Stream as zero-padded string
        year   = '{:04d}'.format(tr.stats.starttime.year)       # Create zero-padded strings
        jday   = '{:03d}'.format(tr.stats.starttime.julday)
        month  = '{:02d}'.format(tr.stats.starttime.month)
        date   = '{:02d}'.format(tr.stats.starttime.day)
        hour   = '{:02d}'.format(tr.stats.starttime.hour)
        minute = '{:02d}'.format(tr.stats.starttime.minute)
        second = '{:02d}'.format(tr.stats.starttime.second)

        
        #################################################################
        # Create filename
        # Replace filestructure syntax with variables
        
        fullpath      = filestructure                           # Initialize syntax for this file
        fullpath = fullpath.replace('BASEDIR', basedir)

        fullpath = fullpath.replace('NET', network)
        fullpath = fullpath.replace('STA', station)
        fullpath = fullpath.replace('LOC', location)
        fullpath = fullpath.replace('CHAN', channel)
        fullpath = fullpath.replace('TYPE', datatype)


        fullpath = fullpath.replace('YEAR', year)
        fullpath = fullpath.replace('MONTH', month)
        fullpath = fullpath.replace('DATE', date)                # Day of month
        fullpath = fullpath.replace('JDAY', jday)                # Julian Day of Year

        fullpath = fullpath.replace('HOUR', hour)
        fullpath = fullpath.replace('MINUTE', minute)
        fullpath = fullpath.replace('SECOND', second)

        fullpath = fullpath.replace('EXTENSION', fileformat)

        # Assert proper filestructure path and that create directories
        fullpath = os.path.normcase(os.path.normpath(fullpath)) # Normalize case of filepath; assert proper syntax
        fullpath = os.path.abspath(fullpath)                    # Create absolte path if relative path given
        directories, filename = os.path.split(fullpath)         # Split into directories, filename
        Path(directories).mkdir(parents=True, exist_ok=True)    # Create all directories necessary

        #################################################################
        # Write file!
        tr.write(fullpath, reclen=reclen)
        output_files.append(fullpath)

    return output_files


def make_directories( fullpath ):
    
    import os
    from pathlib import Path
    
    fullpath = os.path.normcase(os.path.normpath(fullpath)) # Normalize case of filepath; assert proper syntax
    fullpath = os.path.abspath(fullpath)                    # Create absolte path if relative path given
    directories, filename = os.path.split(fullpath)              # Split into path filename
    Path(directories).mkdir(parents=True, exist_ok=True)              # Create all directories necessary
