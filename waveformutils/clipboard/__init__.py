def plot():
    return []


class Clipboard:

    def __init__(self, st):

        # if st is trace: st=Stream(st) # ensure that st is Stream, not single trace
        # st = st.merge()

        self.a = ''
        self.st = []  # ObsPy stream

        self.figsize = (12, 0.5 + 2.5*(len(st)))
        self.axesextent = dict({
            'left': 0.05,
            'right': 0.95,
            'top': 0.05,
            'bottom': 0.05,
            'space': 0.03,
        })
        self.wgratio = [1, 3]

        self.gspecs = dict({
            'ylim': [0.1, 10],
            'cmap': 'plasma',
            'log_power': False,
            'window_size': 2,
            'nfftpts': None,
            'overlap': 0.86,
        })

        self.wspecs = dict({
            'color': 'k',
            'ylim': None,
        })

        self.taxisspecs = dict({
            'n_seis_ticks': 6,

        })

        self.nslc_label = ''

        self.fig = plot()

    def autotaxis(self):
        pass