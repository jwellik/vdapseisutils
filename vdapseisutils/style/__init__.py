import os
import warnings
import matplotlib as mpl


def load_custom_rc(customrc):
    # Get the directory of the style module
    style_path = os.path.dirname(__file__)

    # Construct the full path to the custom matplotlibrc file
    matplotlibrc_path = os.path.join(style_path, customrc)

    try:
        mpl.rc_file(matplotlibrc_path)  # Load the custom matplotlibrc file
    except Exception as e:
        # Catch any exception and raise a warning
        warnings.warn(f"WARNING: Could not load rcparams from {matplotlibrc_path}: {e}")
