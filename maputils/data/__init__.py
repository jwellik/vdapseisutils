print('maputils.data.__init__ will soon be deprecated, and each component will become its own file')

########### VOLCANO #######################################
volc2 = [
    dict({'name':'Agung', 'lat':-8.343, 'lon':115.508, 'elev':2997}),
    dict({'name':'Batur', 'lat':-8.242, 'lon':115.375, 'elev':1717}),
]


########### SEISMOMETERS #######################################
stations = [
    dict({'name':'TMKS', 'lon':115.46675, 'lat':-8.36383}),
    dict({'name':'PSAG', 'lon':115.49872, 'lat':-8.37769}),
    dict({'name':'ABNG', 'lon':115.43476667, 'lat':-8.29436667}),
    dict({'name':'YHKR', 'lon':115.50838252, 'lat':-8.38157119}),
    dict({'name':'CEGI', 'lon':115.4716111, 'lat':-8.30494}),

    dict({'name':'BTR', 'lon':115.37636, 'lat':-8.24523}),
    dict({'name':'REND', 'lon':115.43167611, 'lat':-8.42471940}),
    dict({'name':'DUKU', 'lon':115.5341944, 'lat':-8.29586}),
    dict({'name':'BATU', 'lon':115.49954, 'lat':-8.20885}),
    dict({'name':'DNU', 'lon':115.38533, 'lat':-8.26944}),
    dict({'name':'DNU', 'lon':115.38853, 'lat':-8.23}),
]

volcs = dict({
    'Agung': dict({
        'lat': -8.343, 'lon':115.508, 'elev':2997,
        'stations' : [
            dict({'name':'TMKS', 'lon':115.46675, 'lat':-8.36383}),
            dict({'name':'PSAG', 'lon':115.49872, 'lat':-8.37769}),
            dict({'name':'ABNG', 'lon':115.43476667, 'lat':-8.29436667}),
            dict({'name':'YHKR', 'lon':115.50838252, 'lat':-8.38157119}),
            dict({'name':'CEGI', 'lon':115.4716111, 'lat':-8.30494}),

            dict({'name':'BTR', 'lon':115.37636, 'lat':-8.24523}),
            dict({'name':'REND', 'lon':115.43167611, 'lat':-8.42471940}),
            dict({'name':'DUKU', 'lon':115.5341944, 'lat':-8.29586}),
            dict({'name':'BATU', 'lon':115.49954, 'lat':-8.20885}),
            dict({'name':'DNU', 'lon':115.38533, 'lat':-8.26944}),
            dict({'name':'DNU', 'lon':115.38853, 'lat':-8.23}),
        ]
    }),

    'Batur': dict({
        'lat': -8.242, 'lon':115.375, 'elev':1717,
        'stations' : [
        ]
    }),

    'Hood': dict({
        'synonyms':"Wy'east",
        'lat': 45.374, 'lon': -121.695, 'elev': 3426,
        'stations': []
    }),


    'Augustine': dict({
        'lat': 59.363, 'lon': -153.43, 'elev': 1252,
        'stations': []
    }),

    'Erebus': dict({
        'lat': -77.53, 'lon': 167.17, 'elev': 3794,
    'stations': []
    }),

})