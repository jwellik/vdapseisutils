################################################################################

spectrogram_settings =
{
'min_frequency': 0.0, 'max_frequency': 25.0,  # 'ylim': [0.0, 25.0]
'power_range_db':[0.0, 120.0],
'window_size_s': 2.0, 'overlap': 0.86,
'log_power': True,
'cmap': 'inferno',
}

wave_settings =
{
#'min_amplitude': -1000.0, 'max_amplitude': 1000.0,  # 'ylim': [-1000.0, 1000.0]
'color':'k',
filter: {'bandpass', 'min_frequency': 1.0, 'max_frequency': 10.0, 'npoles': 4},  # include this?
}

spectra_settings =
{
'min_frequency': 0.0, 'max_frequency': 25.0,  # 'xlim': [0.0, 25.0]
'log_power': True, 'log_frequency': True,
'y_axis_range': [1.0, 5.0]
}

################################################################################
# Contents of vdapseisutils.core.swarmmpl.py

class Helicorder:
    ...

class Clipboard:
    ...


################################################################################
# USAGE EXAMPLES

################################################################################
fig = swarmwg(st, gax={"ylim":[0.1, 20]}, specgram={"per_lap":0.75}, sample_rate=20.0)
fig.plot()


################################################################################
fig = swarmwg(st)
fig.spectrogram_settings(per_lap=0.5, sample_rate=20)
fig.set_gax(ylim=[0.1, 10])
fig.plot()
fig.save()

