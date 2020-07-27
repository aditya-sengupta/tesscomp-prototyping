# Script to get all the target stars observed by TESS (in specific sectors or overall) and to save their TIC entries.
# Aditya Sengupta, 2020-07-27

import os
import requests
import warnings
from io import BytesIO
import pandas as pd
from urllib.parse import urljoin
from astroquery.mast import Catalogs

def get_stars_from_sector(sector_num, datapath=None, subpath=None, verbose=True):
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
    # sets up file paths and names
    sector = str(sector_num).zfill(3)
    if datapath is None:
        datapath = os.getcwd()
    if subpath is None:
        subpath = "TESS_targets_S{}.csv".format(sector)
    fullpath = urljoin(datapath, subpath)

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
    tic_data = tic_data.fillna(-1).astype({'ID': int, 'HIP' : int, 'KIC' : int, 'numcont' : int})
    merged_data = tic_data.merge(observations, left_on='ID', right_on='TICID')
    merged_data.to_csv(fullpath)
    if verbose:
        print("Saved TIC data from TESS sector {0} to path {1}.".format(sector_num, fullpath))
    return merged_data

if __name__ == "__main__":
    datapath = "~/projects/pals/tesscomp-prototyping/data/tesstargets/" # CHANGE THIS to your desired directory
    sectors = True # True for 'all available', an int for just one sector, or a list of ints for a subset of sectors.
    if sectors is True:
        i = 1
        while True:
            try:
                get_stars_from_sector(i, datapath=datapath)
                print()
                i += 1
            except requests.exceptions.HTTPError:
                break
    elif isinstance(sectors, int):
        get_stars_from_sector(sectors, datapath=datapath)
    elif isinstance(sectors, list):
        for s in sectors:
            get_stars_from_sector(s, datapath=datapath)
            print()
    else:
        print("Datatype of 'sectors' not understood: set to either True, an integer, or a list of integers.")
