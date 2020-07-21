# Make a TESS stellar catalog that matches the Kepler one.
# want to add the following columns: CDPP, CDPP linear fits, dataspan, duty cycle, limb darkening coefficients
# also, want to retain the IDs, stellar masses and errors, stellar radii and errors, density and errors (easy)

import numpy as np
import pandas as pd
import scipy.stats as stats
import warnings

ln10 = np.log(10)
G_conv = 27420 # the Sun's surface g
rho_conv = 1.408 # the sun's mean density

start_catalog = pd.read_csv('csv-file-toi-catalog.csv', comment='#')

# get intrinsic stellar parameters

tic = start_catalog["TIC"].values
warnings.warn("Currently double-counting stellar targets with multiplicity > 1: please fix")

logg = start_catalog["Surface Gravity Value"].values
d_logg = start_catalog["Surface Gravity Uncertainty"].values
R_star = start_catalog["Star Radius Value"].values
dR_star = start_catalog["Star Radius Error"].values
teff_star = start_catalog["Effective Temperature Value"].values
dteff_star = start_catalog["Effective Temperature Uncertainty"].values
M_star = 10 ** logg * R_star ** 2 / G_conv
dM_star = np.sqrt((M_star * ln10 * d_logg) ** 2 + (2 * M_star * dR_star / R_star) ** 2)
dens = rho_conv * M_star / R_star ** 3
d_dens = dens * (dM_star / M_star + 3 * dR_star / R_star)
snr = start_catalog["Signal-to-noise"].values

tess_stellar_dict = {
    "tic" : tic,
    "mass" : M_star,
    "mass_err1" : dM_star,
    "mass_err2" : -dM_star,
    "logg" : logg,
    "logg_err1" : d_logg,
    "logg_err2" : -d_logg,
    "radius" : R_star,
    "radius_err1" : dR_star,
    "radius_err2" : -dR_star,
    "dens" : dens,
    "dens_err1" : d_dens,
    "dens_err2" : -d_dens,
    "teff" : teff_star,
    "teff_err1" : dteff_star,
    "teff_err2" : -dteff_star,
    "snr" : snr
}

# get CDPPs

problem_idxs = [92, 129, 197, 205, 209, 215, 217, 218, 223, 451, 612, 644, 718, 968, 980, 981, 1029, 1464, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600, 1606, 1606, 1606, 1606, 1606, 1606, 1606, 1606, 1606, 1606, 1607, 1607, 1607, 1607, 1607, 1607, 1607, 1607, 1607, 1607, 1640, 1681, 2065]
problem_sectors = [16, 17, 18, 18, 18, 18, 18, 18, 18, 12, 2, 2, 6, 7, 19, 7, 8, 16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 14, 20]

cdpp_vals = np.load('tess_cdpp.npy')
cdpp_inds = np.load('tess_cdpp_idx.npy')
num_tois = max(cdpp_inds[:,0]) + 1
cdpp_matrix = np.hstack((cdpp_inds, cdpp_vals))
rows_to_remove = [np.where(np.logical_and(cdpp_inds[:,0] == idx, cdpp_inds[:,1] == sector))[0][0] for idx, sector in zip(problem_idxs, problem_sectors)]
cdpp_matrix = np.delete(cdpp_matrix, rows_to_remove, axis=0)
cdpp_inds = np.delete(cdpp_inds, rows_to_remove, axis=0)
first_lookups = np.searchsorted(cdpp_inds[:,0], np.arange(num_tois))
new_cdpp_matrix = np.empty((num_tois, cdpp_vals.shape[1]))
no_cdpp_inds = []

for i, f in enumerate(first_lookups):
    if i < num_tois - 1:
        cdpp_slice = cdpp_matrix[f:first_lookups[i+1]]
    else:
        cdpp_slice = cdpp_matrix[f:]
    if len(cdpp_slice) == 0:
        assert i in problem_idxs
        no_cdpp_inds.append(i)
        new_cdpp_matrix[i] = np.zeros(14,)
    else:
        new_cdpp_matrix[i] = cdpp_slice[np.argmin(cdpp_slice[:,2])][2:]

# TOIs at indices 644, 968, 980, 981, 1029, 1640, 1681 do not have any valid CDPP values
# for whatever reason

for k in tess_stellar_dict.keys():
    tess_stellar_dict[k] = np.delete(tess_stellar_dict[k], no_cdpp_inds, axis=0)

num_tois -= len(no_cdpp_inds)

log_long_durations = np.log10(np.array([7.5, 9.0, 10.5, 12.0, 12.5, 15.0]))
log_short_durations = np.log10(np.array([2.0, 2.5, 3.0, 3.5, 4.5]))
cdppslplong = np.array([stats.linregress(log_long_durations, np.log10(cdpp_vals[-6:])).slope for i, cdpp_vals in enumerate(new_cdpp_matrix) if i not in no_cdpp_inds])
cdppslpshrt = np.array([stats.linregress(log_short_durations, np.log10(cdpp_vals[2:7])).slope for i, cdpp_vals in enumerate(new_cdpp_matrix) if i not in no_cdpp_inds])

# data span and duty cycle

def get_sector_ints(idx):
    return [int(x) for x in start_catalog['Sectors'].values[idx].split()]

dutycycle = 13.0 / 13.6 * np.ones(num_tois,) # Sullivan et al., section 6.5
dataspan = 27 * np.array([len(get_sector_ints(i)) for i in range(num_tois)]) # days

# limb darkening

limbdark_coeff1 = np.zeros(num_tois,)
limbdark_coeff2 = np.zeros(num_tois,)
limbdark_coeff3 = np.zeros(num_tois,)
limbdark_coeff4 = np.zeros(num_tois,)

# put stuff in the stellar frame

rrmskeys = ['rrmscdpp01p5', 'rrmscdpp02p0', 'rrmscdpp02p5', 'rrmscdpp03p0', 'rrmscdpp03p5', 'rrmscdpp04p5', 'rrmscdpp05p0', 'rrmscdpp06p0', 'rrmscdpp07p5', 'rrmscdpp09p0', 'rrmscdpp10p5', 'rrmscdpp12p0', 'rrmscdpp12p5', 'rrmscdpp15p0']
mesthreskeys = ['mesthres01p5', 'mesthres02p0', 'mesthres02p5', 'mesthres03p0', 'mesthres03p5', 'mesthres04p5', 'mesthres05p0' 'mesthres06p0', 'mesthres07p5', 'mesthres09p0', 'mesthres10p5', 'mesthres12p0', 'mesthres12p5', 'mesthres15p0']

rrmsdict = {rrmskeys[i] : np.delete(new_cdpp_matrix[:,i], no_cdpp_inds, axis=0) for i in range(len(rrmskeys))}
mesthresdict = {k : 7.1 * np.ones(num_tois,) for k in mesthreskeys}

tess_stellar_dict.update(rrmsdict)
tess_stellar_dict.update(mesthresdict)
tess_stellar_dict.update({
    "cdppslplong" : cdppslplong,
    "cdppslpshrt" : cdppslpshrt,
    "dataspan" : dataspan,
    "dutycycle" : dutycycle,
    "limbdark_coeff1" : limbdark_coeff1,
    "limbdark_coeff2" : limbdark_coeff2,
    "limbdark_coeff3" : limbdark_coeff3,
    "limbdark_coeff4" : limbdark_coeff4,
})


tess_stellar = pd.DataFrame(tess_stellar_dict)
tess_stellar.to_csv('tess_stellar.csv')

# and now, all the same stuff for planetary
# I think all of this could be in Pandas operations, but eh

toi_ids = start_catalog["Full TOI ID"].values
toi_disp = start_catalog["TOI Disposition"].values
toi_period = start_catalog["Orbital Period Value"].values
# toi_duration = 
# toi_depth = 
toi_prad = start_catalog["Planet Radius Value"].values
toi_prad_err1 = start_catalog["Planet Radius Error"].values
toi_prad_err2 = -start_catalog["Planet Radius Error"].values
toi_ror = toi_prad * 0.009158 / R_star

toi_dict = {
    "tessid" : toi_ids,
    "tic" : tic,
    "toi_disp" : toi_disp,
    "toi_period" : toi_period,
    "toi_prad" : toi_prad,
    "toi_prad_err1" : toi_prad_err1,
    "toi_prad_err2" : toi_prad_err2,
    "toi_ror" : toi_ror
}

tois = pd.DataFrame(toi_dict)
tois.to_csv("tois.csv")