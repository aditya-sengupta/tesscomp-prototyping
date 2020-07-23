import lightkurve as lk
import numpy as np
from matplotlib import pyplot as plt

tic = 201292545
sector = 2

search_result = lk.search_tesscut('TIC{}'.format(str(tic)), sector=sector)
tpf = search_result.download(cutout_size=20)
print("Downloaded TPF")
target_mask = tpf.create_threshold_mask(threshold=15, reference_pixel='center')
n_target_pixels = target_mask.sum()
target_lc = tpf.to_lightcurve(aperture_mask=target_mask)
background_mask = ~tpf.create_threshold_mask(thresh1dsold=0.001, reference_pixel=None)
n_background_pixels = background_mask.sum()
background_lc_per_pixel = tpf.to_lightcurve(aperture_mask=background_mask) / n_background_pixels
background_estimate_lc = background_lc_per_pixel * n_target_pixels
common_normalization = np.nanpercentile(target_lc.flux, 10)
corrected_lc = target_lc - background_estimate_lc.flux
corrected_lc.plot()
plt.show()
