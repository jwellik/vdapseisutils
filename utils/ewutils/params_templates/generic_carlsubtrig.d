#
#   {filename} # (carlsubtrig.d)
#   AUTO GENERATED by VDAP Python scripts
#   - Source  : {source}
#   - Created : {datetime}
#
#   {name}
#
# CarlSubTrig's Parameter File
#
#
#  Basic Earthworm Setup
#
MyModuleId      MOD_CARLSUBTRIG # Module id for this instance of CarlSubTrig -
                                #
Debug           2               # Write out debug messages? (0 = No,
                                #   1 = Minimal, 3 = Chatterbox )
RingNameIn      PICK_RING       # Name of ring from which station triggers
                                #   will be read - REQUIRED.
RingNameOut     HYPO_RING       # Name of ring to which triggers will be
                                #   written - REQUIRED.
HeartBeatInterval       30      # Heartbeat Interval (seconds).

#
# CarlSubTrig Specific Setup
#
StationFile     "{StationFile}"  # Name of file containing station information -
                                #   REQUIRED.
Latency         180              # Number of seconds that the Network clock
                                #   is behind wall clock REQUIRED.
NetTriggerDur   25              # Number of seconds for the base network
                                #   trigger duration REQUIRED.
SubnetContrib   18              # Addition contribution to network trigger
                                #   duration for each subnet that triggered
                                #   REQUIRED.
PreEventTime    20              # Number of seconds added to beginning of
                                #   network trigger REQUIRED.
MaxDuration     500            # Maximum duration allowed for network trigger
DefStationDur   120              # Default number of seconds for station
                                #   trigger duration if the trigger-off
                                #   message is not received. REQUIRED.
ListSubnets     2               # Flag to list untriggered stations
                                #   =0 or command not present: list all
                                #        triggered stations
                                #   =1 list all stations in triggered subnets
                                #   =2 list all stations in triggered subnets
                                #        plus any other triggered stations.
                                #   =3 list all stations in subnets that had
                                #        any stations triggered
AllSubnets      10              # If this many subnets trigger, put wildcard
                                #   SCN in event message
#CompAsWild                      # Flag (no value) to list component names in
#                                #   trigger messages as `*' (wildcard).

MaxTrigMsgLen   30000           # maximum length (bytes) of a triglist message;
                                #   up to MAX_BYTES_PER_EW (earthworm.h).

# Load the next valid trigger sequence number
@trig_id.d            # this name is hard-coded; do not change

# List the message logos to grab from transport ring
#              Installation       Module          Message Types (hard-wired)
GetEventsFrom  INST_WILDCARD    MOD_WILDCARD    # TYPE_CARLSTATRIG # REQUIRED.

# Non-seismic or other channels that should be included in all event messages
# List one SCN per line, as many as you need
#Channel  *.TIM.UW

# Optional commands for writing UW2-format pseudo pickfiles
# Uncomment this on the old system (baker and hood)
#WriteFile                       # if this command is present, a UW-format
                                # pseudo pickfile will be written
                                # Comment it out if you don't want it.
# Remaining commands used only if "WriteFile" is present.
#UWDir   /earthworm/REVIEW
#UWSuffix        a       # suffix for UW pickfile from earthworm

# Command to run on UW pick file.
# Command will be run in background, so its output should be collected by
# the command itself.
#UWCommand       /earthworm/bin/ct_proc

# Optional command to have network IDs put into the pick file
# Currently UW pickfiles have only station and component codes.
#UWWriteNet

# Subnet definitions for the CarlSubTrig Earthworm module
# Each Subnet must be on a single line
# Subnet  Minimum to      List of Station.Component.Network
# Number  Trigger         Codes (space delimited)
# ------- ---  -------------------------------------------
{subnet_lines}