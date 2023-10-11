import matplotlib.pyplot as plt
from vdapseisutils.sandbox import velocity as vmodels
from vdapseisutils.utils import ewutils as ew


def main():

    print("Velocity Model Examples")

    # Read a 1D Velocity Model from csv format
    df = vmodels.read_csv("../data/Stratovolcano_1DmeanLayer_JDP.csv")

    # Standard velocity model plot
    ax = vmodels.plot_velocity(df["LayerDepth"], df["Vp"], Vs=df["Vp"]*1.73, title="VDAP Stratovolcano Model", label="VDAP")
    ax.set_xlim([3, 15])
    ax.set_ylim([30, -1])  # maxdepth to mindepth (depth positive down)
    ax.set_title("Velocity Models")
    plt.show()

    # Write to Earthworm
    ew.lay(df["LayerDepth"], df["Vp"])  # Prints a series of "lay" commands for Earthworm
    ew.velocityd(depth=df["LayerDepth"], velocity=df["Vp"])  # Prints lines suitable for a velocity.d file
    ew.velocitycrh(depth=df["LayerDepth"], velocity=df["Vp"])  # Prints lines suitable for a .crh velocity file

    print("Done.")


if __name__ == "__main__":
    main()
