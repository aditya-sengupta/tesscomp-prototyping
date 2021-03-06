#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import h5py
from tqdm import tqdm

if __name__ == '__main__':

	with h5py.File(r'sector005_camera1_ccd3.hdf5', 'r') as hdf:


		refindx = hdf['wcs'].attrs['ref_frame']

		for w in tqdm(hdf['wcs']):

			#print(w)
			refindx = int(w)
			#print(refindx)

			hdr_string = hdf['wcs']['%04d' % refindx][0]
			if not isinstance(hdr_string, str): hdr_string = hdr_string.decode("utf-8") # For Python 3

			hdr = fits.Header().fromstring(hdr_string)
			#print(hdr)
			wcs = WCS(header=hdr, relax=True) # World Coordinate system solution.

			#print(wcs)

			#fp1 = wcs.calc_footprint(axes=(2048, 2048))
			fp = wcs.calc_footprint(axes=(2, 2))
			test_coords = np.atleast_2d(fp[0, :])

			#print('---------------------------')
			#print(fp1[0, :])
			#print(fp[0, :])

			try:
				a = wcs.all_world2pix(test_coords, 0, ra_dec_order=True, maxiter=50)
			except ValueError:
				print("NO NO NO")
				print(w)
				print(hdf['imagespaths'][refindx])
