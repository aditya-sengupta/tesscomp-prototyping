import os
import utils
import lightkurve as lk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

DATAPATH = os.path.abspath(os.path.dirname(os.getcwd())) + "/data/toi_lightcurves/"

get_subpath = lambda ticid, category: "{0}_{1}.npy".format(ticid, category)
assert callable(get_subpath), "must have a file name convention"

def download_lightcurve(lc_info):
    ticid, sector = lc_info
    try:
        search_result = lk.search_tesscut('TIC{}'.format(str(ticid)), sector=sector)
        tpf = search_result.download(cutout_size=20)
        target_mask = tpf.create_threshold_mask(threshold=15, reference_pixel='center')
        n_target_pixels = target_mask.sum()
        target_lc = tpf.to_lightcurve(aperture_mask=target_mask)
        background_mask = ~tpf.create_threshold_mask(threshold=0.001, reference_pixel=None)
        n_background_pixels = background_mask.sum()
        background_lc_per_pixel = tpf.to_lightcurve(aperture_mask=background_mask) / n_background_pixels
        background_estimate_lc = background_lc_per_pixel * n_target_pixels
        corrected_lc = target_lc - background_estimate_lc.flux
        np.save(os.path.join(DATAPATH, get_subpath(ticid, "time")), corrected_lc.time)
        np.save(os.path.join(DATAPATH, get_subpath(ticid, "flux")), corrected_lc.flux)
    except (BaseException, lk.search.SearchError):
        print("Skipping TIC ID {0} in sector {1}".format(ticid, sector))

if __name__ == "__main__":
    tois = utils.get_tois()
    ticids = tois["TIC"].values
    sectors = [int(t.split(" ")[0]) for t in tois["sectors"]]
    with Pool(processes=4) as p:
        with tqdm(total=len(ticids)) as pbar:
            for i, _ in enumerate(p.imap_unordered(download_lightcurve, zip(ticids, sectors))):
                pbar.update()

