from astroquery.mast import Catalogs
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import constants
import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import requests
import caffeine

toi = pd.read_csv("csv-file-toi-catalog.csv", comment='#')

def get_TPF(ID, sector):
    """
    Queries TIC for a target star and returns its TPF.
    Args:
        ID (int): TIC ID of the target.
        sectors (numpy array): Sectors in which the target has been observed.
        search_radius (int): Number of pixels from the target star to search.
    Returns:
        TPF (numpy array): Time-series FFI of the target.
        time (numpy array): Times corresponding to each image.
    """

    # find coordinates of the target
    df = Catalogs.query_object("TIC"+str(ID), radius=0.0001, catalog="TIC").to_pandas()
    target = df[["ID", "Tmag", "ra", "dec", "mass", "rad", "Teff", "logg", "lum", "plx"]]
    ra, dec = target["ra"].values, target["dec"].values
    # get the TPF with TESScut
    cutout_coord = SkyCoord(ra[0], dec[0], unit="deg")
    cutout_hdu = Tesscut.get_cutouts(cutout_coord, size=20, sector=sector)[0]
    TPF = cutout_hdu[1].data["Flux"]
    time = cutout_hdu[1].data["Time"]
    return TPF, time

def get_lightcurve(TPF, aperture, subtract_background = True):
    """
    Converts a target's FFIs into a light curve given an aperture and performs a simple background subtraction.
    Args:
        TPF (numpy array): Target Pixel File (time-series FFI) of the target.
        aperture (numpy array): Aperture mask used to extract the light curve.
        subract_background (bool): Whether or not to do background subtraction.
    Returns:
        flux (numpy array): Extracted light curve flux.
    """

    if subtract_background == True:
        # calculate the background for each image by taking the median pixel flux outside of the aperture
        background = np.median(TPF[:,~aperture], axis=1)
        # calculate the flux by summing the values in the aperture pixels and subtracting the background
        n_aperture_pixels = aperture[aperture==True].flatten().shape[0]
        flux = np.sum(TPF[:,aperture], axis=1) - background * n_aperture_pixels
    else:
        # calculate the flux by summing the values in the aperture pixels
        flux = np.sum(TPF[:,aperture], axis=1)
    return flux

mask = np.zeros((20,20), dtype=bool)
mask[9:12,9:12] = True

def get_sector_ints(idx):
    return [int(x) for x in toi['Sectors'].values[idx].split()]

num_rows = sum([len(get_sector_ints(i)) for i in range(len(toi))])
cdpp_vals = np.array([3, 4, 5, 6, 7, 9, 10, 12, 15, 18, 21, 24, 25, 30], dtype=int)
num_cols = len(cdpp_vals)

def main():
    cdpp_matrix = np.empty((num_rows, num_cols))
    j = 0
    for i, ID in enumerate(tqdm.tqdm(toi['TIC'].values)):
        for s in get_sector_ints(i):
            try:
                TPF, time = get_TPF(ID=ID, sector=s)
                flux = get_lightcurve(TPF=TPF, aperture=mask)
                curve = lk.lightcurve.TessLightCurve(time=time, flux=flux)
                est = np.vectorize(lambda x: curve.estimate_cdpp(int(x)))
                cdpp_matrix[j] = est(cdpp_vals)
            except KeyboardInterrupt:
                np.save('tess_cdpp.npy', cdpp_matrix)
                return
            except Exception:
                print("Problem at index {0} and sector {1}, continuing".format(i, s))
            j += 1

    np.save('tess_cdpp.npy', cdpp_matrix)

main()
