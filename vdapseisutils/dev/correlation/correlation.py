import numpy as np
from eqcorrscan.utils.clustering import cross_chan_correlation as eqccc

# TODO Add functionality for multi-station shift lengths; use this information

# TODO Change name to cross_channel_matrix() ?
def cross_channel_correlation(streams, shift_len=3.0, cores=4, **kwargs):
    """Computes cross correlation matrix and shift-length matrix
    Uses eqcorrscan.utils.clustering.cross_chan_correlation to execute correlations

    Input
    streams : List of ObsPy Stream objects. Each Stream should be an event. Multiple Traces can hold multiple stations.
    shift_len : Maximum number of seconds that waveforms can shift

    Return
    ccmatrix : Numpy symterical matrix of cross correlation values
    shiftmat : Numpy symterical matrix of shift-length values (location of maximum, in seconds)
    """

    n = len(streams)
    ccmatrix = np.full([n, n], np.nan)
    shiftmat = np.full([n, n], np.nan)
    np.fill_diagonal(ccmatrix, 1)  # Does not need to be assigned back to CCM; kinda weird
    np.fill_diagonal(shiftmat, 0)
    for i in range(n - 1):
        try:
            ccdata, shiftdata = eqccc(streams[i], streams[i + 1:], shift_len=shift_len, cores=cores, **kwargs)
            ccmatrix[i + 1:, i] = ccdata  # fill the ith column with i+1 new correlation values
            ccmatrix[i, i + 1:] = ccdata  # fill the ith row with i+1 new correlation values
            shiftmat[i + 1:, i] = shiftdata[:, 0]  # shiftdata is n by nsta
            shiftmat[i, i + 1:] = shiftdata[:, 0]  # shiftdata is n by nsta
        except Exception as e:
            print(f"Error in cross correlation at Event #{i}: {str(e)}")
            # Optionally, for more detailed error information:
            import traceback
            print(traceback.format_exc())
    ccmatrix = np.abs(ccmatrix)

    return ccmatrix, shiftmat


def savemat(mat, filename="matrix.npy"):
    """Saves a symetrical matrix as a .npy file. Only stores the upper triangle."""
    import numpy as np
    np.save(filename, np.triu(mat))


def loadmat(filename="matrix.npy"):
    """Loads a symetrical matrix as a .npy file. Assumes upper triangle is stored."""
    import numpy as np

    # Load the upper triangle
    upper_triangle = np.load(filename)

    # Create a full symmetrical matrix
    reconstructed_matrix = upper_triangle + upper_triangle.T #- np.diag(upper_triangle.diagonal())
    return reconstructed_matrix
