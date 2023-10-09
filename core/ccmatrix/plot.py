import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import matplotlib.pyplot as plt
import numpy as np

def cmapmpl(ccthresh):

    n1 = int(np.ceil(256*(ccthresh)))  # Number of colors to represent below threshold
    # n2 = int(np.floor(256*(1-ccthresh)))  # Number of colors to represent above threshold
    # Good options: Reds, Oranges, inferno_r, magma_r
    colors = plt.cm.inferno_r(np.linspace(0.0, 1.0, 256))  # take n2 samples from ccthresh to 1
    # combine them and build a new colormap
    # colors = np.vstack((colors1, colors2))  # should be length 256
    colors[0:n1, 3] = np.zeros((1, n1))+0.25
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    return cmap


def ccplotmpl(ccm, vmin=0.0, ccthresh=0.7, nbins=20, facecolor="Purple", filename=None):

    # gridspec inside gridspec
    fig = plt.figure(constrained_layout=True, figsize=(6.5, 4))
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    subfigs[0] = ccmatrixmpl(ccm)
    subfigs[1] = cchistmpl(ccm)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    return fig


def ccmatrixmpl(ccm, vmin=0.0, vmax=1.0, ccthresh=0.7, cmap=None, title="Cross Correlation Matrix", filename=None):

    fig = plt.figure()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    plt.imshow(ccm, norm=norm, cmap=cmap)

    plt.title(title)
    plt.ylabel("Event #")
    plt.xlabel("Event #")

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    return fig


def ccmatrixmpl2(ccm, vmin=0.0, vmax=1.0, ccthresh=0.7, cmap=None, title="Cross Correlation Matrix", filename=None):

    fig = plt.figure()

    nc1 = int(128*ccthresh)
    nc2 = 128-nc1
    colors1 = plt.cm.Greys(np.linspace(ccthresh,0., nc1))
    colors2 = plt.cm.inferno(np.linspace(ccthresh,1.,nc2))
    colors = np.vstack((colors1, colors2))
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    plt.imshow(ccm, norm=norm, cmap=cmap)

    plt.title(title)
    plt.ylabel("Event #")
    plt.xlabel("Event #")

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    return fig


def cchistmpl(ccm, nbins=20, facecolor="Purple", filename=None):

    # Plot cross-correlation histogram
    histfig = plt.figure()
    A = ccm[np.tril_indices(ccm.shape[0], k=-1)]  # returns just the lower part of the matrix
    # n, bins, patches = plt.hist(C.flatten(), 20, density=True, facecolor='orange', alpha=0.5)
    n, bins, patches = plt.hist(A, nbins, facecolor=facecolor, alpha=0.5)
    plt.xlim(0.0, 1.0)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    return histfig


def ccmatrixbokeh(self):
    # Bokeh plot of CCMatrix
    pass

def cchistbokeh():
    pass
