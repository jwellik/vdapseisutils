import datetime as dt
from obspy import Stream, UTCDateTime
import matplotlib.pyplot as plt

swarm_colors_hex = [
    "#0000ff",
    "#0000cd",
    "#00009b",
    "#000069",
]

swarm_colors_rgba = [
    (0, 0, 255, 255),
    (0, 0, 205, 255),
    (0, 0, 155, 255),
    (0, 0, 105, 255),
]


class Heli:

    def __init__(self, st, start="2004/10/15", stop="2004/10/16",
                 one_bar_range="auto", clip_threshold="auto",
                 xminutes=30, colors=swarm_colors_hex,
                 ):

        self.st = st

        self.start = UTCDateTime(start)
        self.stop = UTCDateTime(stop)
        self.one_bar_range = one_bar_range
        self.clip_threshold = clip_threshold
        self.xminutes = xminutes
        self.colors = colors

        self.yticklabels = {
            "timezone_left": "UTC",
            "timezone_right": "Local",
            "date_format": "%mmm %dd",
            "hour_format": "%HH:MM",
            "interval": dt.timedelta(minutes=30),
        }

        self.xticklabels = {
            "timezone_left": "UTC",
            "timezone_right": "Local",
            "date_format": "%mmm %dd",
            "hour_format": "%HH:MM",
            "interval": dt.timedelta(minutes=5),
        }

        self.fontsize = 12
        self.xlabel = "+ Minutes"

        self.figsize = (8, 6)
        self.dpi = 300

    def plot(self):
        fig, ax = plt.subpots()
        fig.f
