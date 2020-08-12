# Given a TESS target and sector, finds the point on the detector where it's seen.

import numpy as np
import pandas as pd
from itertools import product

pointings = pd.read_csv("../data/Year1-3_pointing_long.csv", header=1)
print(pointings.keys())
half_detector_size = 12.
pointing_columns = list(map(lambda a: a[1] + a[0], product(["1", "2", "3", "4"], ["RA", "Dec"])))
roll_columns = ["Roll"+str(i) for i in range(1, 5)]

sector_centers = {
    sector : pointings[pointings["Sector"] == sector][pointing_columns]
    for sector in pointings["Sector"].values
}

sector_rolls = {
    sector : pointings[pointings["Sector"] == sector][roll_columns]
    for sector in pointings["Sector"].values
}

def observable(star, sector):
    '''
    Checks whether a star was observable in a sector.

    Arguments
    ---------
    star : pd.DataFrame row
    A single entry from the stellar dataframe as returned by utils.get_stellar_data

    sector : int
    The sector number.

    Returns
    -------
    camera : int
    The camera (1-4) number on which the star should've been observed; 0 if it wasn't observable.
    Can do bool() on the output to just get a true/false of observability.
    '''
    centers = sector_centers[sector]
    rolls = sector_rolls[sector]
    for camera in range(1, 5):
        center_ra = float(centers[pointing_columns[2*(camera - 1)]])
        center_dec = float(centers[pointing_columns[2*camera-1]])
        center_roll = float(rolls[roll_columns[camera-1]]) * np.pi / 180
        rot_matrix = np.array([[np.cos(center_roll), np.sin(center_roll)], [-np.sin(center_roll), np.cos(center_roll)]])
        extents = rot_matrix @ np.array([half_detector_size, half_detector_size])
        if abs(center_ra - star.ra) * np.cos(center_dec * np.pi / 180) < extents[0] and abs(center_dec - star.dec) < extents[1]:
            return camera
            # camera gaps
            # web tool: TESS web viewer
            # camera boresights
            # try to replicate TESS's observation pattern in the Southern Hemisphere
    return 0

def position(star, sector, camera):
    pass

if __name__ == "__main__":
    for i, star in pd.read_csv("../data/tesstargets/tess_stellar_all.csv").iterrows():
        for sector in star["sectors"].split(','):
            obs = observable(star, int(sector))
            try:
                assert obs == star['Camera']
            except AssertionError:
                print("for star idx {0}, obs says {1} but data says {2}".format(i, obs, star['Camera']))
