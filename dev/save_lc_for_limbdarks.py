# Save local lightcurves using the lightkurve preprocessing method, for an MCMC-based limb darkening fit.

import numpy as np
from matplotlib import pyplot as plt
import lightkurve as lk
import pandas as pd
from utils import get_tois
import os
from multiprocessing import Pool
from tqdm import tqdm

DATAPATH = os.path.abspath(os.path.dirname(os.getcwd())) + "/data/toi_local_lightcurves/"

def save_local_lc(toi_data, global_bins=2001, local_bins=201):
    ticid, period, t0, dur = toi_data
    local_lc_filename = os.path.join(DATAPATH, "{0}_local.csv".format(ticid))
    if os.path.exists(local_lc_filename):
        return
    tlc = lk.search_lightcurvefile("TIC {}".format(ticid), mission='TESS').download_all()
    if tlc is None:
        print("No light curve found for TIC {}".format(ticid))
        return
    try:
        lc_raw = tlc.PDCSAP_FLUX.stitch()
        lc_clean = lc_raw.remove_outliers(sigma=20, sigma_upper=4)
        temp_fold = lc_clean.fold(period, t0=t0)
        fractional_duration = (dur / 24.0) / period
        phase_mask = np.abs(temp_fold.phase) < (fractional_duration * 1.5)
        transit_mask = np.in1d(lc_clean.time, temp_fold.time_original[phase_mask])
        lc_flat, trend_lc = lc_clean.flatten(return_trend=True, mask=transit_mask)
        lc_fold = lc_flat.fold(period, t0=t0)
        lc_global = lc_fold.bin(bins=global_bins, method='median').normalize() - 1
        lc_global = (lc_global / np.abs(lc_global.flux.min()) ) * 2.0 + 1
        phase_mask = (lc_fold.phase > -4*fractional_duration) & (lc_fold.phase < 4.0*fractional_duration)
        lc_zoom = lc_fold[phase_mask]
        lc_local = lc_zoom.bin(bins=local_bins, method='median').normalize() -1
        lc_local = (lc_local / np.abs(lc_local.flux.min()) ) * 2.0 + 1
        lc_to_save = pd.DataFrame({"phase" : lc_local.phase, "flux" : lc_local.flux})
        lc_to_save.to_csv()
    except ValueError as e:
        print(e)
        return
    
if __name__ == "__main__":
    tois = get_tois()
    ticids, periods, t0s, durs = tois.TIC.values, tois.toi_period.values, tois.epoch.values, tois.toi_transit_dur.values
    # sectors = [int(t.split(" ")[0]) for t in tois["sectors"]]
    with Pool(processes=4) as p:
        with tqdm(total=len(ticids)) as pbar:
            for i, _ in enumerate(p.imap_unordered(save_local_lc, zip(ticids, periods, t0s, durs))):
                pbar.update()