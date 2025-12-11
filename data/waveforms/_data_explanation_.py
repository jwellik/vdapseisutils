from obspy import read


def main():


    data = [
        {"filename": "./Agung_test_data_01.mseed",
        "text":
            """
            Mount Agung, Bali, Indonesia (CVGHM)
            """
        },

        {"filename": "./Augustine_test_data_FI.mseed",
         "text":
             """
             Augustine, Alaska, USA (USGS AVO)
             ---------------------------------
             3 Traces. Each trace is a different event (of slightly different lengths) from AU13.
             
             Attempting to plot on same clipboard. Un-synched t axis (relative time axis labels?)
             """
         },

        {"filename": "./gareloi_test_data_20220710-010000.mseed",
         "text":
             """
             Gareloi, Alaska, USA (USGS AVO)
             -------------------------------
             5 Traces (GAEA, GALA, GANE, GANO, GASW) of ~1 hour of data. No gaps.
             
             Sliced to about 10 minutes.
             
             Adds axvlines with a list of timestamps.
             
             Shows multiple color options.
             """
         },

        {"filename": "./IRIS_test_data_01.mseed",
         "text":
             """
             ---
             """
         },

        {"filename": "./IRIS_test_data_02.mseed",
         "text":
             """
             Mount St. Helens
             ----------------
             71 Traces from 2004-09-1 thru 2004-09-19
             
             Sliced down to 10 minutes. 3 Traces SEP, YEL, HSR
             """
         },

        {"filename": "./MSH_IRIS_test_data_01.mseed",
         "text":
             """
             Mount St. Helens
             ----------------
             About 1 day of data from station SEP and HSR. Saved as 18 Traces bc of gaps.
             Should be plotted as a single Clipboard axis.
             
             Challenge: How to plot multiple traces as 1 axes instead of multiple axes.
             (How to plot gappy data w multiple Traces as a single axes).
             """
         },

    ]

    for d in data:
        st = read(d["filename"])
        print(d["text"])
        print(st)
        # fig = plt.figure(FigureClass=ClipboardClass, st=st, sharex=False, g={"cmap":viridis_u})
        # plt.show()
        print()

    print("Done.")


if __name__ == "__main__":
    main()
