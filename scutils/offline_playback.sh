# THIS IS NOT MY CODE!!!
# IT WAS SHARED WITH BY ISTI IN A SEISCOMP4 COURSE

if [ "$#" -lt 2 ]; then
	echo "Usage: $0 [mseed-volume] [output-xml]"; exit 0
fi
set -x; seiscomp stop; seiscomp start scmaster
DBFLAG="-d mysql://sysop:sysop@localhost/seiscomp3"
STORAGE=$DBFLAG
CONFIGFLAGS="--verbosity=4"
FLAGS="$CONFIGFLAGS $STORAGE"
echo "Starting scautoloc...";
seiscomp exec scautoloc $FLAGS --playback --start-stop-msg=1 --auto-shutdown=1 --shutdown-master-module=scautopick &
echo "Starting magtool";
seiscomp exec scmag $FLAGS --start-stop-msg=1 --auto-shutdown=1 --shutdown-master-module=scautoloc &
echo "Starting eventtool";
seiscomp exec scevent $FLAGS --start-stop-msg=1 --auto-shutdown=1 --shutdown-master-module=scmag &
echo "Starting sceplog..."
seiscomp exec sceplog $CONFIGFLAGS --auto-shutdown=1 --shutdown-master-module=scevent > $2 &
pid=$!
echo "Starting autopick...."; seiscomp exec scautopick --playback -I $1 $FLAGS --start-stop-msg=1
echo "Finished waveform processing - waiting for event processing to finish";
wait $pid
echo "Finished event processing"
