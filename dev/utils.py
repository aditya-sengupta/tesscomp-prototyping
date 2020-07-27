import os
import requests
import pandas as pd
from tess_target_stars import get_stars_from_sector

NUM_TESS_SECTORS = 27

def check_num_tess_sectors():
    i = 1
    has_data = True
    while has_data:
        url = 'https://tess.mit.edu/wp-content/uploads/all_targets_S{}_v1.csv'.format(str(i).zfill(3))
        r = requests.get(url)
        has_data = r.ok
        if has_data:
            i += 1
    if i - 1 != NUM_TESS_SECTORS:
        print("NUM_TESS_SECTORS is listed as {0}, but data was found for {1} sectors: update the variable NUM_TESS_SECTORS for the full data.".format(NUM_TESS_SECTORS, i))

def get_tess_stellar(sectors=None):
    '''
    Wrapper around tess_target_stars.py to merge all sectors.

    Arguments
    ---------
    sectors : list
    A list of sector IDs to query.

    Returns
    -------
    stlr : pd.DataFrame
    The stellar dataframe. Note that this does not drop duplicates
    '''
    if sectors is None:
        sectors = list(range(1, NUM_TESS_SECTORS + 1))
    frames = []
    for s in sectors:
        datapath = "../data/tesstargets/TESS_targets_S{}.csv".format(str(s).zfill(3))
        if os.path.exists(datapath):
            frames.append(pd.read_csv(datapath, comment='#'))
        else:
            frames.append(get_stars_from_sector(s))
    return pd.concat(frames)

if __name__ == "__main__":
    pass
    # print(get_num_tess_stars())