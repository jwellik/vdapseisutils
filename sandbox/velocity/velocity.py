import os
import vdapseisutils
import matplotlib.pyplot as plt
import pandas as pd


def default_velocity(model="vdap_stratovolcano"):
    if model == "vdap_stratovolcano":
        file = os.path.join(os.path.dirname(vdapseisutils.__file__), "data/Stratovolcano_1DmeanLayer_JDP.csv")
    else:
        file = os.path.join(os.path.dirname(vdapseisutils.__file__), "data/Stratovolcano_1DmeanLayer_JDP.csv")
    return read_csv(file)


def read_csv(csvfile, **kwargs):
    """READ_CSV Uses Pandas.read_csv to parse the velocity model"""
    return pd.read_csv(csvfile, **kwargs)


def read_ew_velocity(filename):
    """Reads Earthworm velocity model .d file"""
    # Earthworm syntax
    # -      depth  velocity
    # lay    0.0    5.40
    return pd.read_csv(filename, header=None, delim_whitespace=True, usecols=[1,2], names=["depth", "Vp_top"], comment="#")


def read_nll_velocity(filename, header=False):
    """
    #LAYER   depth Vp_top vp_grad Vs_top Vs_grad p_top p_grad
    LAYER   0.00  4.2669 0.00 2.4664 0.00 2.7 0.0
    LAYER   1.54  4.6400 0.00 2.6821 0.00 2.7 0.0
    LAYER   2.54  4.9574 0.00 2.8656 0.00 2.7 0.0
    LAYER   3.54  5.2000 0.00 3.0059 0.00 2.7 0.0
    LAYER   4.54  5.3846 0.00 3.1125 0.00 2.7 0.0
    LAYER   5.54  5.5344 0.00 3.1991 0.00 2.7 0.0
    """
    return pd.read_csv(filename, header=header, delim_whitespace=True, usecols=[1,2,4], names=["depth", "Vp", "Vs"], comment="#")


def plot_velocity(depth, Vp_top, Vs=None, color="b", label="", ax=None, title="Velocity Model"):
    """Plot simple 1D Velocity Model, positive down"""

    if ax is None:
        fig = plt.figure(figsize=(3,6), dpi=300)
        ax = fig.add_axes([0.25, 0.1, 0.6, 0.8])
    if ax.get_ylim()[1] > ax.get_ylim()[0]:  # force y_axis to be inverted (positive down)
        ax.invert_yaxis()
    ax.set_xlabel("Velocty (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(title)

    ax.plot(Vp_top, depth, "-", color="grey", alpha=0.25)
    ax.plot(Vp_top, depth, "-", color=color, drawstyle="steps", label=label+" Vp")  # drawstyle="steps-pre" (default) is what we want
    if Vs is not None:
        ax.plot(Vs, depth, "-", color="grey", alpha=0.25)
        ax.plot(Vs, depth, "--", color=color, drawstyle="steps", label=label+" Vs")
    ax.legend()

    return ax
