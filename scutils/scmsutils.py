import os

def scmspreproc(topdir, t1, t2, outfile='sortedtmp.mseed', set_record_length=False, verbose=False):
    # (seismology39) sysop@raung:~/DATA/MtHood$ cat *.mseed | scmssort -v -E -t '2021-10-18~2021-10-19' -u > sorted.mseed 

    # collect all miniseed files within archive between t1 & t2
    # (Use vdapseisutils.waveformutils.DataSource to do this)


    # Sort the miniseed files with SeisComP routine
    # scmssort -v -E -t '2021-10-18~2021-10-19' -u > sorted.mseed
    scmssort_str = "???? | scmssort -v -E -t '{}~{}' -u > {}".format(t1str, t2str, outfile)
    if verbose: print(scmssort_str)
    os.system(scmssort_str)

    # If desired, Ensure 512 record length, etc.
    if set_record_length:
        if verbose: print('Set record length to 512 in ObsPy')
        pass

