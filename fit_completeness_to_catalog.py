import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize, stats, special
import pandas as pd
from tqdm.notebook import tqdm
import emcee
import corner

# constants and frozen parameters, like the bins

Go4pi = 2945.4625385377644/(4 * np.pi * np.pi)
MsoMe_cbrt = 69.31
rng_p = np.array([0.5, 200])
rng_r = np.array([0.5, 4])
num_bins_p = 13
num_bins_r = 17
N1 = 1
sigma1 = 0
N2 = 6.1
sigma2 = 2
fs = 0.55
mixture_params = {"N1": N1, "sigma1": sigma1, "N2": N2, "sigma2": sigma2, "fs": fs}
bins_p = np.exp(np.linspace(*np.log(rng_p), num_bins_p))
bins_r = np.exp(np.linspace(*np.log(rng_r), num_bins_r))
eps = 1e-3

def make_synth_solar_systems(mixture_params=mixture_params, num_stars=10000, mstar=0.4):
    '''
    Makes a synthetic solar system from frozen parameters.
    Follows an empirical distribution, and assumes occurrence is either Pop 1 or Pop 2.
    '''
    rstar = 0.9535 * mstar + 0.0053535 # quick linear fit on the Kepler stellar catalog, filtered on mass bw 0.3 and 0.5 Msuns
    num_pop_1 = int(num_stars * mixture_params['fs'])
    num_pop_2 = num_stars - num_pop_1
    nums_planets = np.empty((0,))
    eccs = np.empty((0,))
    for N, sigma, num_pop in [(mixture_params['N1'], sigma1, num_pop_1), 
                              (mixture_params['N2'], sigma2, num_pop_2)]:
        frac = N % 1
        num_floor = int(num_pop * (1 - frac))
        nums_planets = np.hstack((nums_planets, np.floor(N) * np.ones(num_floor,), np.ceil(N) * np.ones(num_pop - num_floor))).astype(dtype=np.int8)
        # eccs = np.hstack((eccs, stats.rayleigh(scale=sigma, size=num_pop))) # He, Ford, Ragozzine
    ecc_means = 0.584 * np.repeat(nums_planets, nums_planets) ** (-1.2) # Limbach and Turner
    eccs = np.random.rayleigh(np.sqrt(2 / np.pi) * ecc_means)
    num_planets = sum(nums_planets)
    periods = np.exp(np.random.uniform(*np.log(rng_p), size=(num_planets,))) # in days
    prads = np.exp(np.random.uniform(*np.log(rng_r), size=(num_planets,))) # in R_Earths
    pmass = np.maximum(0.8, 2.7 * prads ** 1.3 + np.random.normal(0, 1.9, size=(num_planets,))) # ignoring Zeng/Jacobsen
    system_inds = np.cumsum(nums_planets)
    solar_sys_ids = []
    for i, ind in enumerate(nums_planets):
        solar_sys_ids += [i] * ind
    stabilities = 2 * MsoMe_cbrt * ((periods[1:] ** (2/3) - periods[:-1] ** (2/3)) / (periods[1:] ** (2/3) + periods[:-1] ** (2/3))) * (3 * mstar / (pmass[:-1] + pmass[1:]))
    unstable_inds = np.where(np.logical_and((np.abs(stabilities) <= 2 * np.sqrt(3)), ([x in system_inds for x in range(len(stabilities))])))[0]
    # may need to sort in period order?
    while len(unstable_inds) > 0:
        replace_periods = np.exp(np.random.uniform(*np.log(rng_p), size=(len(unstable_inds),)))
        replace_prads = np.exp(np.random.uniform(*np.log(rng_r), size=(len(unstable_inds),)))
        replace_pmass = 2.7 * replace_prads ** 1.3 + np.random.normal(0, 1.9, size=(len(unstable_inds),)) # ignoring Zeng/Jacobsen
        periods[unstable_inds] = replace_periods
        prads[unstable_inds] = replace_prads
        pmass[unstable_inds] = replace_pmass
        stabilities = 2 * MsoMe_cbrt * ((periods[1:] ** (2/3) - periods[:-1] ** (2/3)) / (periods[1:] ** (2/3) + periods[:-1] ** (2/3))) * (3 * mstar / (pmass[:-1] + pmass[1:]))
        unstable_inds = np.where(np.logical_and((np.abs(stabilities) <= 2 * np.sqrt(3)), ([x in system_inds for x in range(len(stabilities))])))[0]
    a = (Go4pi * periods * periods * mstar) ** (1./3)
    transit_bool = a / rstar * (1 - eccs ** 2) <= 1
    ttv = np.random.binomial(n = 1, p = [{1: 0.035, 2: 0.07, 3: 0.08}.get(x) if x < 4 else 0.104 for x in np.repeat(nums_planets, nums_planets)], size=(num_planets,))
    # densities = (stabilities / 22) ** 6 #size mismatch
    return pd.DataFrame({
                        "ids" : solar_sys_ids,
                        "periods" : periods, 
                         "prads" : prads, 
                         "pmass" : pmass, 
                         "eccs" : eccs,
                         "a" : a, 
                         "transit_bool" : transit_bool,
                         "ttv" : ttv, 
                        })

def get_catalog_and_numstars(name, cut_to_Ms=True):
    if name == "sullivan":
        catalog = np.loadtxt('sullivan_catalog.dat')
        cols = ["alpha", "delta", "prads", "periods", "SoSsun", "K", "rs", "teff", "V", "Ic", "J", "Ks", "DM", "Dil.", "log10_sigmav", "SNR", "mult"]
        catalog = pd.DataFrame({x: catalog[:,i] for i, x in enumerate(cols)})
        if cut_to_Ms:
            catalog = catalog[catalog.teff <= 3700]
        num_stars = len(catalog)
    elif name == "barclay":
        catalog = pd.read_csv('barclay_data/detected_planets.csv', skiprows=42)
        catalog = catalog.rename(columns={"Planet-period" : "periods", "Planet-radius": "prads"})
        if cut_to_Ms:
            catalog = catalog[catalog["Star-teff"] <= 3700]
        num_stars = len(set(catalog["TICID"]))
    print("Selected {} stars".format(num_stars))
    return catalog, num_stars

def make_hists(synth, catalog, plot=True):
    hist_synth = np.histogram2d(synth.periods, synth.prads, bins=[bins_p, bins_r])
    hist_catalog = np.histogram2d(catalog["periods"], catalog["prads"], bins=[bins_p, bins_r])
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        ax1.imshow(hist_catalog[0])
        ax1.set_title("Source catalog search distribution")
        ax2.imshow(hist_synth[0])
        ax2.set_title("Distribution of injected planets")
        plt.show()
    return hist_synth[0], hist_catalog[0]

comp_poly = lambda x, a1, a2, a3, a4: a4 + a1 * x + (a1 * a2) * x ** 2 + (a1/3) * (a2**2 + a3**2) * x ** 3

def make_mcmc_setup(N, D, nwalkers=24):
    log_fact_D = special.gammaln(D + 1)

    def ll(a):
        a_period, a_radius = a[:4], a[4:]
        comp_p, comp_r = comp_poly(bins_p[:-1], *a_period), comp_poly(bins_r[:-1], *a_radius)
        has_negative = np.any(comp_p < 0) or np.any(comp_r < 0)
        if has_negative:
            return -np.inf
        if False:
            if np.any(comp_p > 1) or np.any(comp_r > 1):
                return -np.inf
        comp = np.outer(comp_p, comp_r)
        if np.any(comp < 0):
            return -np.inf
        mu = N * comp
        if np.any(mu < 0):
            return -np.inf
        mu += eps
        # likelihood_mat = mu ** D * np.exp(-mu) / fact_D
        # return np.nansum(np.log(likelihood_mat))
        ll_mat = D * np.log(mu) - mu - log_fact_D
        return np.sum(ll_mat)

    ndim = 8
    optimize_result = optimize.minimize(lambda x: -ll(x), [1.1, -0.55, 0.11, -0.01, 0.14, 0.45, 0.4, 0.2], method='Nelder-Mead', options={"maxiter" :10000})
    leastsq_sol = optimize_result.x
    print("Found least-squares solution: {}".format(leastsq_sol))

    def prior(a):
        return np.all(np.isfinite(a)) and np.all(np.abs(leastsq_sol - a) < np.maximum(0.01, np.abs(4 * leastsq_sol)))

    def ll_with_prior(a):
        if not prior(a):
            return -np.inf
        return ll(a)
    
    flag = False
    p0 = leastsq_sol + np.random.normal(0, 1e-3, size=(nwalkers, ndim))
    while not flag:
        lls = np.array([ll_with_prior(p) for p in p0])
        if -np.inf not in lls:
            flag = True
        else:
            inds = np.where(lls == -np.inf)[0]
            p0[inds] = leastsq_sol + np.random.normal(0, 1e-3, size=(len(inds), ndim))

    print("Set initial condition")
    return ll_with_prior, p0

def plot_marginalized_comps(params):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(comp_poly(bins_p, *params[:4]))
    ax1.set_xlabel("Period bucket")
    ax1.set_ylabel("Completeness fraction")
    ax2.plot(comp_poly(bins_r, *params[4:]))
    ax2.set_xlabel("Radius bucket")
    ax2.set_ylabel("Completeness fraction")
    plt.show()

def plot_overall_comps(repcomp, name):
    comp04m = np.load('ballard_data/Completeness_0.4Msun.npy')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,12))
    ballard_comp = ax1.imshow(comp04m[-1], origin='lower')
    fig.colorbar(ballard_comp, ax=ax1, fraction=0.058)
    ax1.set_xticks(np.arange(13)[::2])
    ax1.set_xticklabels(np.round(bins_p[::2], 2))
    ax1.set_xlabel("Period (days)")
    ax1.set_yticks(np.arange(17)[::2])
    ax1.set_yticklabels(np.round(bins_r[::2], 2))
    ax1.set_ylabel(r"Radius ($R_E$)")
    ax1.set_title("Ballard (2018) completeness")
    self_comp = ax2.imshow(repcomp, origin='lower')
    fig.colorbar(self_comp, ax=ax2, fraction=0.058)
    ax2.set_xticks(np.arange(13)[::2])
    ax2.set_xticklabels(np.round(bins_p[::2], 2))
    ax2.set_xlabel("Period (days)")
    ax2.set_yticks(np.arange(17)[::2])
    ax2.set_yticklabels(np.round(bins_r[::2], 2))
    ax2.set_ylabel(r"Radius ($R_E$)")
    ax2.set_title("Replicated completeness, with {} catalog".format(name))
    plt.show()
