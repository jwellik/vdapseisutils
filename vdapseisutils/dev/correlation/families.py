import pandas as pd
import numpy as np
from obspy import Stream


def add_member_times(famdf, detection_df):
    print("Do yo thang...")

# [x] ALlow user to pass family_idx, which can also be filtered and sorted; otherwise use ordinal values
# TODO Sort by ID?
def filter_and_sort(data, min_members=2, order="chron", family_id=None):
    """
    Filter and sort a list of lists containing datetime objects.

    Parameters:
    - data: List of lists where each inner list contains datetime objects
    - min_members: Minimum number of datetime objects required in each inner list
    - order: Sorting criterion ('chron', 'last', 'smallest', 'largest', 'longest', 'shortest', 'recent')

    Returns:
    - sorted_data: Filtered and sorted data
    - indices: List of original indices corresponding to each item in sorted_data
    """

    # Initilaize family ID
    data = data.copy()
    family_id = np.arange(len(data)) if family_id is None else np.array(family_id)

    # Filter data based on min_members criteria
    filtered_data = []
    filtered_indices = []

    for i, family in enumerate(data):
        if len(family) >= min_members:
            filtered_data.append(family)
            filtered_indices.append(i)

    # Pair the filtered data with their indices for sorting
    paired_data = list(zip(filtered_data, filtered_indices))

    # Sort the paired data
    if order in ['chron', 'oldest', 'first']:
        sorted_pairs = sorted(paired_data, key=lambda x: min(x[0]))
    elif order == 'last':
        sorted_pairs = sorted(paired_data, key=lambda x: min(x[0]), reverse=True)
    elif order == 'smallest':
        sorted_pairs = sorted(paired_data, key=lambda x: len(x[0]))
    elif order == 'largest':
        sorted_pairs = sorted(paired_data, key=lambda x: len(x[0]), reverse=True)
    elif order == 'longest':
        sorted_pairs = sorted(paired_data, key=lambda x: max(x[0]) - min(x[0]), reverse=True)
    elif order == 'shortest':
        sorted_pairs = sorted(paired_data, key=lambda x: max(x[0]) - min(x[0]))
    elif order == 'recent':
        sorted_pairs = sorted(paired_data, key=lambda x: max(x[0]), reverse=True)
    else:
        print("Order='{}' not recognized. Families not sorted.".format(order))
        sorted_pairs = paired_data  # No sorting if criterion is not recognized

    # Unpack the sorted pairs
    if sorted_pairs:
        sorted_data, indices = zip(*sorted_pairs)  # indices returned as a tuple
        return list(sorted_data), list(family_id[list(indices)])
    else:
        # Return empty lists if no data passed the filter
        return [], []


# TODO Return output as dict() instead of DataFrame? (more general)
def build_green_neuberg_2006(ccm, mincc=0.7, minmembers=5):
    """BUILD_GREEN_NEUBERG_2006 Method for creating earthquake families from a correlation matrix
    Based on the publication: https://www.sciencedirect.com/science/article/pii/S0377027305003835

    Section 2.3 (Classification Procedure) describes the following steps:

    To sort the matrix into 'm' families

    1) The event with the most correlation values above the threshold value, 'phi', was isolated and chosen as the
       master (sic.) event for the first waveform family.
    2) All the events well correlated with it (r >= 'phi') were removed from the matrix as one group.
    Repeat as long as the groups have equal to or more than 'minmembers' members.

    This left a remainder matrix to which the same process was applied until the entire matrix is classified into
    distinct groups.

    This implementation also allows families to be removed if they do not have a minimum number of events.
    In the future, it could also allow for only a certain number of families to be created (ie, the N largest families).

    """

    famdf = pd.DataFrame(columns=["Main", "Members", "Stack", "MainWave"])
    CCM = ccm.copy()

    # Get Family Members
    nmembers = minmembers
    REPEATERS = []
    while nmembers >= minmembers:
        counts = np.count_nonzero(CCM >= mincc, axis=0)  # Number of events above threshold for each column
        main_idx = np.argmax(counts)  # column index of event with most matches
        members = np.argwhere(CCM[:, main_idx] >= mincc).flatten()  # Row index of events above threshold for column of main event
        nmembers = len(members)  # Number of members matching the main event (including main event?)
        if len(members) >= minmembers:
            # print("Main event: {:04d} {}".format(main_idx, members))
            REPEATERS += list(members)
            CCM[:, members] = np.nan
            CCM[members, :] = np.nan

            tmp = pd.DataFrame.from_dict(dict({"Main": [main_idx], "Members": [members.tolist()],
                              "Stack": [Stream()], "MainWave": [Stream()]}))
            famdf = pd.concat([famdf, tmp], ignore_index=True)

    return famdf, REPEATERS


def stackFams(fams, trigs):
    # fams is a DataFrame with column Main, Members, Stack, MainWave
    # trigs is a DataFrame with column Members, Stream

    i=0
    for idx, row in fams.iterrows():
        st0 = Stream()
        for member in row.Members:
            st0 += trigs.iloc[member]["Stream"].split().merge(method=1, fill_value="interpolate")
        fams.iloc[idx]["Stack"] = st0.stack(group_by="id")  # Let's try this outside of a list
        fams.iloc[idx]["MainWave"] = trigs.Stream[row.Main].split().merge(method=1, fill_value="interpolate")
        i+=1

    return fams


def stats_msg(famdf, trigdf, minmembers):
    # famdf is a DataFrame with columns Main, Members, Stack, MainWave
    # trigdf is a DataFrame with columns ...

    nrepeaters = len(famdf.Members.sum())
    rpts = nrepeaters/len(trigdf)*100

    msg = ""
    msg += "Results ():" + "\n"
    msg += " Triggers  : {: 5d}".format(len(trigdf)) + "\n"
    msg += " Families  : {: 5d} ({}+ members)".format(len(famdf), minmembers) + "\n"
    msg += " Repeaters : {: 5d} ({:3.1f}%)".format(nrepeaters, rpts) + "\n"
    msg += " Rep/Fam   : {:5.1f}".format(nrepeaters/ len(famdf)) + "\n"
    msg += ""

    return msg
