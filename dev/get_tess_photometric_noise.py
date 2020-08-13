# Query the Web TESS Viewer for the photometric noise of every star in a sector.

import utils
import urllib.request as request
import urllib.error as error
import pandas as pd
from tqdm import tqdm
import sys
import os
import numpy as np

def get_tess_photometric_noise(sector_num, datapath=utils.TESS_DATAPATH, subpath=None):
    '''
    For all the stars in a sector, plug in its Vmag/Jmag/Kmag to https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py
    and get the photometric noise in ppm.
    '''
    sector = str(sector_num).zfill(3)
    if datapath is None:
        datapath = os.getcwd()
    if subpath is None:
        subpath = "TESS_noise_S{}".format(sector)
    fullpath = os.path.join(datapath, subpath)
    
    get_url = lambda ticid, v, j, k: "https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py?V={0}&J={1}&Ks={2}".format(str(v), str(j), str(k))
    stlr = utils.get_tess_stars_from_sector(sector_num)

    ids_to_skip = set()
    if os.path.exists(fullpath):
        saved_noise_dataframe = pd.read_csv(fullpath)
        if sum(saved_noise_dataframe.noise != 0) == len(stlr):
            return saved_noise_dataframe
        else:
            ids_to_skip = set(saved_noise_dataframe[saved_noise_dataframe.noise != 0].ID.values)
    
    noises = np.zeros(len(stlr))
    try:
        for i, params in tqdm(stlr[["ID", "Vmag", "Jmag", "Kmag"]].iterrows(), total=len(stlr)):
            if params[0] in ids_to_skip:
                noises[i] = float(saved_noise_dataframe[saved_noise_dataframe.ID == params[0]].noise)
            elif any(np.isnan(params)):
                continue
            else:
                query_url = get_url(*params)
                response = request.urlopen(query_url)
                text = str(response.read())
                noises[i] = float(text[text.find("sigma = ") + 7:text.find("ppm")])
    except (error.URLError, ConnectionError, KeyboardInterrupt):
        pass
    noise_dataframe = pd.DataFrame({"ID" : stlr.ID, "noise" : noises})
    noise_dataframe.to_csv(fullpath)
    return noise_dataframe
    
if __name__ == "__main__":
    s = int(sys.argv[1])
    if s < 1 or s > utils.NUM_TESS_SECTORS:
        print("Enter a valid sector number.")
    else:
        _ = get_tess_photometric_noise(s)