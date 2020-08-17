import os
import requests
import warnings
import numpy as np
import pandas as pd
from functools import reduce
from io import BytesIO
import sys
sys.path.append("..")

NUM_TESS_SECTORS = 27
TESS_DATAPATH = os.path.abspath(os.path.dirname(os.getcwd())) + "/data/tesstargets/" # or change
assert TESS_DATAPATH[-1] == os.path.sep, "must end datapath with {}".format(os.path.sep)

def get_tess_stars_from_sector(sector_num, datapath=TESS_DATAPATH, subpath=None, force_redownload=False, verbose=True):
    '''
    Queries https://tess.mit.edu/observations/target-lists/ for the input catalog from TESS sector 'sector_num',
    and for each target in that list, gets its data from astroquery and joins the two catalogs.

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

    if (not os.path.exists(fullpath)) or force_redownload:
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
        tic_data = tic_data.astype({"ID" : int})
        merged_data = tic_data.merge(observations, left_on='ID', right_on='TICID')
        noises_path = os.path.join(datapath, "TESS_noise_S{}.csv".format(sector))
        if os.path.exists(noises_path):
            merged_data = merged_data.merge(pd.read_csv(noises_path, index_col=0, comment='#'), on="ID")
        else:
            print("Noise values not found on path: change file location or download using get_tess_photometric_noise.py.")
        merged_data.to_csv(fullpath)
        if verbose:
            print("Saved TIC data from TESS sector {0} to path {1}.".format(sector_num, fullpath))
        return merged_data
    else:
        return pd.read_csv(fullpath, index_col=0)

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

def get_tess_stellar(sectors=None, unique=True, force_resave=False, force_redownload=False):
    '''
    Wrapper around tess_target_stars.py to merge all sectors.

    Arguments
    ---------
    sectors : list
    A list of sector IDs to query.

    unique : bool
    If true, this function cuts down the stellar dataframe to only unique entries, and adds a few columns.

    force_resave : bool
    If true, forces a reread of the constituent files from the URL (rerun of get_tses_stars_from_sector)

    Returns
    -------
    stlr : pd.DataFrame
    The stellar dataframe. If `unique`, the returned value is instead:

    stlr : pd.DataFrame
    The stellar dataframe, with duplicates dropped and the following columns added:
        sectors, str      : the sectors in which the target was observed.
        dataspan, scalar  : 27.4 days times the number of sectors in which the target was observed.
        dutycycle, scalar : the fraction 13.0/13.6 (for the 0.6 day downlink)
        noise, scalar     : the 1-hour photometric noise (replacement for CDPP but not averaged over timescales)
    '''
    if sectors is None:
        sectors = list(range(1, NUM_TESS_SECTORS + 1))
    frames = []
    sector_obs = {}
    sector_cnt = {}
    noises = {}
    for s in sectors:
        datapath = os.path.join(TESS_DATAPATH, "TESS_targets_S{}.csv".format(str(s).zfill(3)))
        if os.path.exists(datapath) and (not force_resave):
            df = pd.read_csv(datapath, comment='#', index_col=0)
        else:
            df = get_tess_stars_from_sector(s, force_redownload=force_redownload)
        if unique:
            for ticid, noise in zip(df["ID"].values, df["noise"].values):
                if ticid not in sector_obs:
                    sector_obs[ticid] = str(s)
                    sector_cnt[ticid] = 1
                    noises[ticid] = str(noise)
                else:
                    sector_obs[ticid] += ',' + str(s)
                    sector_cnt[ticid] += 1
                    noises[ticid] += ',' + str(noise)
        frames.append(df)
    stlr = pd.concat(frames)
    if unique:
        stlr.drop_duplicates(subset="ID", inplace=True)
        stlr["sectors"] = [sector_obs.get(ticid) for ticid in stlr["ID"].values]
        stlr["noise"] = [noises.get(ticid) for ticid in stlr["ID"].values]
        stlr["dataspan"] = 27.4 * np.array([sector_cnt.get(ticid) for ticid in stlr["ID"].values])
        stlr["dutycycle"] = 13.0/13.7 * np.ones_like(stlr["dataspan"])
    return stlr

def get_tois(subpath="toi_catalog.csv", force_redownload=False):
    '''
    Request a pandas dataframe of all the TESS objects of interest.
    '''
    fullpath = os.path.join(TESS_DATAPATH, subpath)
    if (not force_redownload) and os.path.exists(fullpath):
        return pd.read_csv(fullpath, comment='#', index_col=0)
    else:
        url = "https://tev.mit.edu/data/collection/193/csv/6/"
        print("Retrieving TOI table from {}.".format(url))
        req = requests.get(url)
        tois = pd.read_csv(BytesIO(req.content), comment='#', index_col=0)
        tois = tois.rename(columns={
            "Source Pipeline" : "pipeline",
            "Full TOI ID" : "toi_id",
            "TOI Disposition" : "toi_pdisposition",
            "TIC Right Ascension" : "tic_ra",
            "TIC Declination" : "tic_dec",
            "TMag Value" : "tmag", 
            "TMag Uncertainty" : "tmag_err", 
            "Orbital Epoch Value" : "epoch",
            "Orbital Epoch Error" : "epoch_err",
            "Orbital Period Value" : "toi_period",
            "Orbital Period Error" : "toi_period_err",
            "Transit Duration Value" : "toi_transit_dur",
            "Transit Duration Error" : "toi_transit_dur_err",
            "Transit Depth Value" : "toi_transit_depth",
            "Transit Depth Error" : "toi_transit_depth_err",
            "Sectors" : "sectors",
            "Public Comment" : "comment",
            "Surface Gravity Value" : "surface_grav",
            "Surface Gravity Uncertainty" : "surface_grav_err",
            "Signal ID" : "signal_id",
            "Star Radius Value" : "srad",
            "Star Radius Error" : "srad_err",
            "Planet Radius Value" : "toi_prad",
            "Planet Radius Error" : "toi_prad_err",
            "Planet Equilibrium Temperature (K) Value" : "ptemp",
            "Effective Temperature Value" : "steff",
            "Effective Temperature Uncertainty" : "steff_err",
            "Effective Stellar Flux Value" : "sflux",
            "Signal-to-noise" : "snr",
            "Centroid Offset" : "centroid_offset",
            "TFOP Master" : "tfop_master", 
            "TFOP SG1a" : "tfop_sg1a", 
            "TFOP SG1b" : "tfop_sg1b", 
            "TFOP SG2" : "tfop_sg2", 
            "TFOP SG3" : "tfop_sg3",
            "TFOP SG4" : "tfop_sg4", 
            "TFOP SG5" : "tfop_sg5", 
            "Alerted" : "alerted", 
            "Updated" : "updated"
        })
        tois.to_csv(fullpath)
        return tois
    
if __name__ == "__main__":
    stlr_all = get_tess_stellar(force_resave=True)
    stlr_all.to_csv("../data/tesstargets/tess_stellar_all.csv")

    # print(get_num_tess_stars())