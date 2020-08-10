import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append("..")

invdets = open("../data/tess_invdets.txt").read().split('\n')
period_bins = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
radius_bins = 0.01 / 1.090700992458569 * np.array([1.090700992458569, 5.453504962292846, 10.907009924585692, 27.267524811464227, 54.535049622928454, 81.80257443439268, 109.07009924585691, 136.33762405732114, 163.60514886878536, 190.8726736802496, 218.14019849171382, 272.6752481146423, 327.2102977375707, 436.28039698342764])

occurrence = []
occ_err_pl = []
occ_err_mn = []

for entry in invdets:
    if '+' in entry:
        occurrence.append(float(entry[:entry.index('+')]))
        occ_err_pl.append(float(entry[entry.index('+') + 1:entry.index('-')]))
        occ_err_mn.append(float(entry[entry.index('-') + 1:-1]))

occurrence = np.array(occurrence).reshape((len(period_bins)-1, len(radius_bins)-1))
plt.imshow(occurrence, origin='lower')
plt.colorbar()
_ = plt.xticks(list(range(len(radius_bins)-1)), radius_bins[:-1])
plt.xlabel(r"Radius ($R_E$)")
_ = plt.yticks(list(range(len(period_bins)-1)), period_bins[:-1])
plt.ylabel("Period (days)")
plt.show()