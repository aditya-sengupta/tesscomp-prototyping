import numpy as np
import pandas as pd

'''fails = open("tess_cdpp_fails", 'r')
to_match = fails.read().split('\n')

print([int(x[17:-12].split()[0]) for x in to_match if x[:7] == "Problem"])
print([int(x[17:-12].split()[-1]) for x in to_match if x[:7] == "Problem"])'''

tois = pd.read_csv("csv-file-toi-catalog.csv", comment='#')

problem_idxs = [92, 129, 197, 205, 209, 215, 217, 218, 223, 451, 612, 644, 718, 968, 980, 981, 1029, 1464, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1606, 1606, 1606, 1606, 1606, 1606, 1606, 1606, 1606, 1606, 1607, 1607, 1607, 1607, 1607, 1607, 1607, 1607, 1607, 1607, 1640, 1681, 2065]
problem_sectors = [16, 17, 18, 18, 18, 18, 18, 18, 18, 12, 2, 2, 6, 7, 19, 7, 8, 16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 14, 20]
cdpp_matrix = np.load('tess_cdpp.npy')s
cdpp_inds = np.load('tess_cdpp_idx.npy')
problem_cdpp_inds = np.hstack([np.where(np.logical_and(cdpp_inds.T[0] == problem_idxs[i], cdpp_inds.T[1] == problem_sectors[i]))[0] for i in range(len(problem_idxs))])
cdpp_matrix = np.delete(cdpp_matrix, problem_cdpp_inds, axis=0)
cdpp_inds = np.delete(cdpp_inds, problem_cdpp_inds, axis=0)

# let's pick the lower SNR observation for now, but think about merging the observations later on.

dutycycle = 13.0 / 13.6 # Sullivan et al., section 6.5
dataspan = 27 # days
