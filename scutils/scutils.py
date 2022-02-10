import os


def scmssort(catexp,
             t=None,
             outfile="./sorted.mseed",
             list=None,  # -l
             sort_by_end_time=True,  # -E
             uniqueness=True,  # -u
             verbose=True,  # -v
             ):
    """Runs scmssort by building the proper string

    # Using as many default values as possible, the following two commands are equivalent:
    # >>> scmssort("/home/DATA/*.mseed", t=(t1, t2))
    # $ cat /home/DATA/*.mseed | scmssort -v -E -t 2021/10/01~2021/10/02 -u > ./sorted.mseed

    NOTE: -l --LIST option is not yet implemented

    """

    v = "-v" if verbose else ""
    E = "-E" if sort_by_end_time else ""
    u = "-u" if uniqueness else ""
    t = "" if t is None else "-t {}~{}".format(t[0].isoformat(), t[1].isoformat())

    cmdstr = "cat {catexp} | scmssort {v} {E} {t} {u} > {outfile}".format(
        catexp=catexp, v=v, E=E, t=t, u=u, outfile=outfile
    )

    if verbose:
        print(cmdstr)
    os.system(cmdstr)

    return []
