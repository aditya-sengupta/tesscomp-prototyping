import os
import requests
import warnings
import pandas as pd
from functools import reduce
from io import BytesIO
import sys
sys.path.append("..")

NUM_TESS_SECTORS = 27
TESS_DATAPATH = os.path.abspath(os.path.dirname(os.getcwd())) + "/data/tesstargets/" # or change
assert TESS_DATAPATH[-1] == os.path.sep, "must end datapath with {}".format(os.path.sep)

def get_tess_stars_from_sector(sector_num, datapath=TESS_DATAPATH, subpath=None, verbose=True):
    '''
    Queries https://tess.mit.edu/observations/target-lists/ for the input catalog from TESS sector 'sector_num',
    and for each target in that list, gets its data from astroquery and joins the two catalogs.

    NOTE: to avoid automatic conversion of int columns with NaNs to float, NaNs are default-replaced with -1.

    Arguments
    ---------
    sector_num : int
    The TESS sector number for which information is being requested.

    datapath : str
    The top-level path to which data should be stored.

    subpath : str
    The subdirectory (datapath/subpath) to which data should be stored; will create it if it doesn't already exist.

    verbose : bool
    Whether to print statements on the script's progress.

    Returns
    -------
    stars : pd.DataFrame
    The joined TIC and target-list data.
    '''
    from astroquery.mast import Catalogs

    # sets up file paths and names
    sector = str(sector_num).zfill(3)
    if datapath is None:
        datapath = os.getcwd()
    if subpath is None:
        subpath = "TESS_targets_S{}.csv".format(sector)
    fullpath = os.path.join(datapath, subpath)

    # queries the target list
    url = 'https://tess.mit.edu/wp-content/uploads/all_targets_S{}_v1.csv'.format(sector)
    if verbose:
        print("Getting sector {0} observed targets from {1}.".format(sector_num, url))
    req = requests.get(url)
    if not req.ok:
        raise requests.exceptions.HTTPError("Data from sector {} is not available.".format(sector_num))
    observations = pd.read_csv(BytesIO(req.content), comment='#')[['TICID', 'Camera', 'CCD']] # MAST has Tmag, RA, Dec at higher precision
    observed_ticids = observations['TICID'].values

    # queries MAST for stellar data
    if verbose:
        print("Querying MAST for sector {0} observed targets.".format(sector_num))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tic_data = Catalogs.query_criteria(catalog='Tic', ID=observed_ticids).to_pandas()
    # tic_data = tic_data.fillna(-1).astype({'ID': int, 'HIP' : int, 'KIC' : int, 'numcont' : int})
    merged_data = tic_data.merge(observations, left_on='ID', right_on='TICID')
    merged_data.to_csv(fullpath)
    if verbose:
        print("Saved TIC data from TESS sector {0} to path {1}.".format(sector_num, fullpath))
    return merged_data

def get_stellar_data(sectors=True):
    '''
    Utility function to call get_tess_stars_from_sector on a specific directory.

    Arguments
    ---------
    sectors : bool, int, or list of ints
    True for 'all available', an int for just one sector, or a list of ints for a subset of sectors.
    '''
    if sectors is True:
        i = 1
        while True:
            try:
                get_tess_stars_from_sector(i, datapath=TESS_DATAPATH)
                print()
                i += 1
            except requests.exceptions.HTTPError:
                break
    elif isinstance(sectors, int):
        get_tess_stars_from_sector(sectors, datapath=TESS_DATAPATH)
    elif isinstance(sectors, list):
        for s in sectors:
            get_tess_stars_from_sector(s, datapath=TESS_DATAPATH)
            print()
    else:
        print("Datatype of 'sectors' not understood: set to either True, an integer, or a list of integers.")

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
    The stellar dataframe. Note that this does not drop duplicates: each unique observation of a star counts as a separate row.
    To get only the unique stars, use df.drop_duplicates after calling this function.
    '''
    if sectors is None:
        sectors = list(range(1, NUM_TESS_SECTORS + 1))
    frames = []
    for s in sectors:
        datapath = os.path.join(TESS_DATAPATH, "TESS_targets_S{}.csv".format(str(s).zfill(3)))
        if os.path.exists(datapath):
            frames.append(pd.read_csv(datapath, comment='#'))
        else:
            frames.append(get_tess_stars_from_sector(s))
    return pd.concat(frames)

def get_tois(subpath="toi_catalog.csv"):
    '''
    Request a pandas dataframe of all the TESS objects of interest.
    '''
    fullpath = os.path.join(TESS_DATAPATH, subpath)
    if os.path.exists(fullpath):
        return pd.read_csv(fullpath, comment='#')
    else:
        url = "https://tev.mit.edu/data/collection/193/csv/6/"
        req = requests.get(url)
        tois = pd.read_csv(BytesIO(req.content), comment='#')
        tois.to_csv(fullpath)
        return tois
    

if __name__ == "__main__":
    pass
    # print(get_num_tess_stars())