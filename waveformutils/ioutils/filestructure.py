# Various filestructure/filename formats

# Filestructure and Filename Standards

# SeisComP Data Structure (standard format)
sds_filestructure = "{YEAR}/{NET}/{STA}/{CHAN}.{TYPE}/"
sds_filename = "{NET}.{STA}.{LOC}.{CHAN}.{TYPE}.{YEAR:04}.{JDAY:03}.{FORMAT}"
sds_standard = sds_filestructure + sds_filename

# Swarm (standard format)
swarm_filestructure = "{YEAR:04}{MONTH:02}{DATE:02}{HOUR:02}{MINUTE:02}{SECOND:02}/"
swarm_filename = "{STA}_{CHAN}_{NET}_{LOC}.{FORMAT}"
swarm_standard = swarm_filestructure + swarm_filename

# Custom
sds_starttime = "{NET}.{STA}.{LOC}.{CHAN}.{TYPE}.{YEAR:04}{MONTH:02}{DATE:02}-{HOUR:02}{MINUTE:02}{SECOND:02}.{FORMAT}"
swarm_starttime = "{STA}_{CHAN}_{NET}_{LOC}-{YEAR:04}{MONTH:02}{DATE:02}-{HOUR:02}{MINUTE:02}{SECOND:02}.{FORMAT}"
nslc_starttime = "{NET}.{STA}.{LOC}.{CHAN}-{YEAR:04}{MONTH:02}{DATE:02}-{HOUR:02}{MINUTE:02}{SECOND:02}.{FORMAT}"


def write2sds(st, basedir='./',
              filestructure=sds_standard,  # filestructure syntax
              # filename=sds_filename,          # filename syntax
              fileformat='mseed',  # seismic data format
              reclen=4096,  # miniseed byte record-length
              ):
    """
    WRITE2SDS Writes Traces to file in accordance w given filestructure

    Default filestructure creates a SDS archive. All necessary folders will be created.
    
    filestructure : str : Syntax for filestructure. Options are:
        Default: 'BASEDIR/YEAR/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.JDAY.EXTENSION'
        Options:
        BASEDIR
        NET
        STA
        LOC
        CHAN
        ID    : equivalent to "NET.STA.LOC.CHA"
        TYPE
        YEAR
        JDAY
        DATE
        HOUR
        MINUTE
        SECOND
        #MISCROSEC
        EXTENSION

    :return
        output_files : list of full filepaths for output miniseed files
    """

    import os
    from pathlib import Path

    filestructure = os.path.join('{BASEDIR}', filestructure)  # Add BASEDIR to path
    output_files = []  # list of final output files

    for tr in st:
        #################################################################
        # PARSE INFO FROM TRACE

        # Parse NSLC information from Stream (returned as string)
        network = tr.stats.network
        station = tr.stats.station
        location = tr.stats.location
        channel = tr.stats.channel

        id = tr.idS

        # Assert datatype 'D'
        datatype = 'D'

        # Parse time information from Stream as zero-padded string
        # year = '{:04d}'.format(tr.stats.starttime.year)  # Create zero-padded strings
        # jday = '{:03d}'.format(tr.stats.starttime.julday)
        # month = '{:02d}'.format(tr.stats.starttime.month)
        # date = '{:02d}'.format(tr.stats.starttime.day)
        # hour = '{:02d}'.format(tr.stats.starttime.hour)
        # minute = '{:02d}'.format(tr.stats.starttime.minute)
        # second = '{:02d}'.format(tr.stats.starttime.second)

        year = tr.stats.starttime.year  # Create zero-padded strings
        jday = tr.stats.starttime.julday
        month = tr.stats.starttime.month
        date = tr.stats.starttime.day
        hour = tr.stats.starttime.hour
        minute = tr.stats.starttime.minute
        second = tr.stats.starttime.second

        ################################################################
        # Create filename
        # Replace filestructure syntax with variables

        fullpath = filestructure  # Initialize syntax for this file

        fullpath = fullpath.format(
            BASEDIR=basedir,
            NET=network,
            STA=station,
            LOC=location,
            CHAN=channel,
            ID=id,
            TYPE=datatype,

            YEAR=year,
            MONTH=month,
            DATE=date,
            JDAY=jday,

            HOUR=hour,
            MINUTE=minute,
            SECOND=second,

            FORMAT=fileformat,
        )

        # Assert proper filestructure path and that create directories
        fullpath = os.path.normcase(os.path.normpath(fullpath))  # Normalize case of filepath; assert proper syntax
        fullpath = os.path.abspath(fullpath)  # Create absolute path if relative path given
        directories, filename = os.path.split(fullpath)  # Split into directories, filename
        Path(directories).mkdir(parents=True, exist_ok=True)  # Create all directories necessary

        #################################################################
        # Write file!
        tr.write(fullpath, reclen=reclen, format=fileformat)
        output_files.append(fullpath)

    return output_files


# Not using this locally yet
def make_directories(fullpath):
    import os
    from pathlib import Path

    fullpath = os.path.normcase(os.path.normpath(fullpath))  # Normalize case of filepath; assert proper syntax
    fullpath = os.path.abspath(fullpath)  # Create absolute path if relative path given
    directories, filename = os.path.split(fullpath)  # Split into path filename
    Path(directories).mkdir(parents=True, exist_ok=True)  # Create all directories necessary
