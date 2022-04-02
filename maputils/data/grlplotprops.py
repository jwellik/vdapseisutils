import datetime
import pandas as pd


# Figure settings
fig_width_in = 6.4
fig_height_in = 1.5
lr_margin = [2/16, 2/16]
bt_margin = [3/16, 2.25/16]
#bounds = [lr_margin[0], bt_margin[0], 1-sum(lr_margin), 1-sum(bt_margin)]
dpi = 1000

fontsize = 8
fontname = 'Arial'
annotfile = './phase_annotations.csv'

########### DATA FILES #################################

configfile = "./agung_5sta_08ccc.cfg"
outputpath = "./<groupName>/"
outfilename = 'GRL_f1A_OverviewMPL'
rsamfile = "./miscdata/AgungRSAM60_TMKS.txt"
cvghmfile = "./miscdata/magma_event_counts.csv"
bmkgcatalog = "./miscdata/BMKGcatalog4jay.txt"
rsam_sample_dur = '24H'


########### TIME #######################################
xlim_full  = [pd.Timestamp(2017,8,31), pd.Timestamp(2018,1,20)] # full timeseries
xlim_study = [pd.Timestamp('2017/10/18'), pd.Timestamp('2018/01/16')] # study period for 5 stations
#xlim_study = [datetime.datetime(2017,11,16), datetime.datetime(2017,11,19)] # stub - look at bars
tlim_cvghm = xlim_full # time limit for EQs on map

utc_offset_hrs = +8 # Bali/Singapore
utc_offset_days = utc_offset_hrs/24 #hours
utc_offset_days_dt = datetime.timedelta(utc_offset_hrs/24)
timezone = 'Asia/Singapore'



########### PHASES #####################################
# Grabbed this from ./phase_annotations.csv # needs timezone info
phase01 = [xlim_study[0], pd.Timestamp('2017/11/08')]
phase02 = [pd.Timestamp('2017/11/08'), pd.Timestamp('2017/11/21')]
phase03 = [pd.Timestamp('2017/11/21'), xlim_study[1]]
phases = [phase01, phase02, phase03]


########### DATA GAPS #######################################
# Not sure how I derived these exactly
data_gap_start = [pd.Timestamp('2017/11/13 03:00').tz_localize('UTC'), pd.Timestamp('2017/12/20 04:00').tz_localize('UTC')]
data_gap_stop =  [pd.Timestamp('2017/11/14 20:00').tz_localize('UTC'), pd.Timestamp('2017/12/21 22:00').tz_localize('UTC')]

def add_data_gaps(ax, data_gap_start, data_gap_stop, annotate=False):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    plt.rcParams['svg.fonttype'] = 'none'
    for start, stop in zip(data_gap_start, data_gap_stop):
        #ax.axvspan(start, stop, facecolor='k', alpha=0.1, edgecolor='none', zorder=0.1) # add darker background
        #ax.axvspan(start, stop, hatch='/////', edgecolor='white', facecolor='none', zorder=0.2) # add cross hatching
        ax.axvspan(start, stop, facecolor='k', alpha=0.1, edgecolor='none', zorder=0.1) # add darker background
        ax.axvspan(start, stop, hatch='/////', edgecolor='white', facecolor='none', alpha=0.99, zorder=0.2, joinstyle='miter') # add cross hatching
    if annotate: # if desired, add annotation to last instance of data gap
        txt = ax.annotate(
            'No Data', (data_gap_stop[-1], 500), xytext=(-6,-5), textcoords='offset points',
            fontsize=8, rotation=90,
                        )
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')]) # add outline to annotation

def add_data_gaps_v1(ax, data_gap_start, data_gap_stop, annotate=False):
    ax.axvspan(data_gap_start, data_gap_stop, facecolor='k', alpha=0.1, edgecolor='none', zorder=0.3)
    ax.axvspan(data_gap_start, data_gap_stop, hatch='/////', edgecolor='w', facecolor='none', zorder=0.2)
    if annotate:
        txt = ax.annotate('No Data', (data_gap_stop, 500), xytext=(-6,-5), textcoords='offset points',
                fontsize=8, rotation=90, backgroundcolor='w',
                    )


########### MAP DATA #######################################
volc = [
    dict({'name':'Agung', 'lat':-8.343, 'lon':115.508, 'elev':2997}),
#    dict({'name':'Batur', 'lat':-8.242, 'lon':115.375, 'elev':1717}),    
]


stations = [
    dict({'name':'TMKS', 'lon':115.46675, 'lat':-8.36383, 'used':True}),
    dict({'name':'PSAG', 'lon':115.49872, 'lat':-8.37769, 'used':True}),
    dict({'name':'ABNG', 'lon':115.43476667, 'lat':-8.29436667, 'used':True}),
    dict({'name':'YHKR', 'lon':115.50838252, 'lat':-8.38157119, 'used':True}),
    dict({'name':'CEGI', 'lon':115.4716111, 'lat':-8.30494, 'used':True}),

    dict({'name':'BTR', 'lon':115.37636, 'lat':-8.24523, 'used':False}),
    dict({'name':'REND', 'lon':115.43167611, 'lat':-8.42471940, 'used':False}),
    dict({'name':'DUKU', 'lon':115.5341944, 'lat':-8.29586, 'used':False}),
    dict({'name':'BATU', 'lon':115.49954, 'lat':-8.20885, 'used':False}),
    dict({'name':'DNU', 'lon':115.38533, 'lat':-8.26944, 'used':False}),
    dict({'name':'DNU', 'lon':115.38853, 'lat':-8.23, 'used':False}),
]


# Map Plot Options
station_marker_size = 6
used_sta_color = 'black'
not_used_sta_color = 'white'
# map options
used_sta = dict(color='black', fill_color='white')
not_used_sta = dict(color='white', fill_color='grey')

# Time series options
cvghm_counts = dict({'color': 'black', 'fill_alpha':0.3})
orphants = dict({'line_color':'black', 'line_width':3})
triggerts = dict({'line_color':'black', 'line_width':3})
repeaterts = dict({'line_color':'red', 'line_width':3})
rsam = dict({'line_color':'blue', 'line_width':2})

#BMKG
eqlocs = [
    {'lat': -8.3, 'lon': 115.56, 'M': 4.9, 'Label': 'M4.9'},
         ]



# EQ Options
# Buurman and West, 2010 :  VT > -0.4 <= Hybrid > -1.3 <= LF
fi_VT_LF_thresh = -0.3 # empirical