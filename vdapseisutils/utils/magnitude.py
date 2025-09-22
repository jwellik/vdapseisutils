import numpy as np
import matplotlib.pyplot as plt

class MagnitudeUtils:
    """
    Utilities for handling earthquake magnitude conversions and plotting.

    Can be used as a static utility or instantiated with a specific scale/base.

    Example usage:
    --------------
    mscale = MagnitudeUtils(base=20, scale=0.75)
    mscale.plot_legend()
    sizes = mscale.to_size([-1, 0, 1, 2, 3])
    # Or use static methods directly:
    sizes = MagnitudeUtils.magnitude2size([1,2,3])
    moment = MagnitudeUtils.magnitude2moment(3.0)
    mag = MagnitudeUtils.moment2magnitude(moment)
    """
    def __init__(self, base=20, scale=0.75, use_moment=False):
        """
        Parameters
        ----------
        base : float, default 20
            Base marker size for plotting.
        scale : float, default 0.75
            Scaling factor for magnitude.
        use_moment : bool, default False
            If True, interpret input to to_size as seismic moment (Nm), else as magnitude.
        """
        self.base = base
        self.scale = scale
        self.use_moment = use_moment

    def to_size(self, mag_or_moment):
        """
        Convert magnitude or moment to marker size using the instance's settings.
        If use_moment is True, input is interpreted as moment (Nm), else as magnitude.
        """
        if self.use_moment:
            mag = self.moment2magnitude(mag_or_moment)
        else:
            mag = mag_or_moment
        return self.magnitude2size(mag, base=self.base, scale=self.scale)

    def plot_legend(self, ax=None, magnitudes=[-1,0,1,2,3,4,5], **kwargs):
        """
        Create a matplotlib legend for marker sizes corresponding to magnitudes using instance settings.
        """
        if ax is None:
            fig, ax = plt.subplots()
        sizes = self.to_size(magnitudes)
        for mag, size in zip(magnitudes, sizes):
            ax.scatter([], [], s=size, label=f"M {mag}", **kwargs)
        ax.legend(scatterpoints=1, frameon=True, labelspacing=1, title="Magnitude")
        ax.set_axis_off()
        return ax

    @staticmethod
    def magnitude2size(mag, base=20, scale=0.75):
        """
        Convert magnitude to marker size for scatter plots (area proportional to energy).
        """
        mag = np.asarray(mag)
        return base * 10 ** (scale * mag)

    @staticmethod
    def magnitude2moment(mag):
        """
        Convert magnitude to seismic moment (Nm).
        Formula: M0 = 10**(1.5*mag + 9.1)
        """
        mag = np.asarray(mag)
        return 10 ** (1.5 * mag + 9.1)

    @staticmethod
    def moment2magnitude(moment):
        """
        Convert seismic moment (Nm) to magnitude.
        Formula: mag = (2/3) * (log10(M0) - 9.1)
        """
        moment = np.asarray(moment)
        return (2.0 / 3.0) * (np.log10(moment) - 9.1) 