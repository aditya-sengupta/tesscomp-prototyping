import sys
sys.path.append("..")
import batman
import lightkurve as lk
import numpy as np
import copy
import os
from matplotlib import pyplot as plt
import tqdm
from dev.kf import jacobian, KFilter

re = 0.009158
eps = 1e-6

# EKF functions for the light curve

def get_a(period, mstar, Go4pi=2945.4625385377644/(4*np.pi*np.pi)):
    """
    https://dfm.io/posts/exopop/
    Compute the semi-major axis of an orbit in Solar radii.
    
    :param period: the period in days
    :param mstar:  the stellar mass in Solar masses
    
    """
    return (Go4pi*period*period*mstar) ** (1./3)

def set_params():
    # to update more generally
    dt = 30 / (60 * 60 * 24) # 30 minute cadence in days
    mstar = 0.9707
    rstar = 0.964
    params = batman.TransitParams()
    params.t0 = 0.25
    params.per = 2.4706
    params.rp = 13.04 * re / rstar
    params.a = get_a(params.per, mstar)
    params.inc = 90.
    params.ecc = 0.
    params.w = 90.                      #longitude of periastron (in degrees)
    params.u = [0.93, -0.23, 0, 0]      #limb darkening coefficients [u1-u4]'''
    params.limb_dark = "nonlinear"       #limb darkening model
    t = np.arange(0, params.per + dt, dt)
    m = batman.TransitModel(params, t)
    model_flux = m.light_curve(params)
    err_flux = m.calc_err() / 1e6
    return mstar, rstar, m, params, t, model_flux, err_flux

def batman_to_partial_state(params):
    state = np.array([
        params.per,
        params.rp,
        params.inc,
        params.ecc,
        params.w
    ])
    state = np.concatenate((state, params.u))
    return state

def get_measured_flux(star_id = "KIC11446443"):
    if os.path.exists("../data/flux_{}.npy".format(star_id)):
        lc_time, lc_flux = np.load("../data/time_{}.npy".format(star_id)), np.load("../data/flux_{}.npy".format(star_id))
    else:
        pixelfile = lk.search_targetpixelfile(star_id, quarter=1).download(quality_bitmask='hardest')
        lc = pixelfile.to_lightcurve(aperture_mask='all')
        lc_time, lc_flux = lc.time, lc.flux
        lc_flux /= max(lc_flux)
        np.save("time_{}.npy".format(star_id), lc_time)
        np.save("flux_{}.npy".format(star_id), lc_flux)
    return lc_time, lc_flux

    # TIC Contamination Ratio	0.05501

class LightcurveKFilter(KFilter):
    def __init__(self, star_id = "KIC11446443"):
        self.t_idx = 0
        self.mstar, self.rstar, self.transitmodel, self.params, self.t, self.model_flux, self.err_flux = set_params()
        # add in a discretization error term

        Q = np.diag([
            self.err_flux, # var(flux) = variance of batman model flux + inherent variance 
            eps, # var(period) between timesteps, should be small
            eps, # var(prad)
            eps, # var(inc)
            eps, # var(ecc)
            eps, # var(omega)
            *[eps] * len(self.params.u) # var(limbdarks)
        ])
        R = np.array([[eps]]) # to be replaced by PRF stuff
        super().__init__(self.f, Q, self.h, R, state=self.default_state())

    def compute_lightcurve(self):
        self.params.per, self.params.rp, self.params.inc, self.params.ecc, self.params.w = self.state[1:6]
        self.params.a = get_a(self.params.per, self.mstar)
        self.params.u = self.state[6:]
        self.model_flux = self.transitmodel.light_curve(self.params)
     
    def f(self, state=None, mutate=False):
        '''
        Takes in a state: [flux, period, prad, incl, ecc, omega, *limbdark1-4].
        Returns the state at the next timestep.
        '''
        if state is None:
            state = self.state
        if len(state.shape) > 1:
            # passing in an array from the Jacobian
            return np.vstack([self.f(row) for row in state])
        if not np.allclose(batman_to_partial_state(self.params), state[1:]):
            # the Kalman update did something that'll need to be changed in the light curve
            self.compute_lightcurve()
        if mutate:
            self.t_idx += 1
        state[0] = self.model_flux[self.t_idx]
        return state
    
    def h(self, state=None):
        if state is None:
            state = self.state
        return np.array([state[0]]) # and PRFs and stuff later
    
    def default_state(self):
        return np.array([
            self.model_flux[0],
            self.params.per,
            self.params.rp, # normalized prad
            90.,
            0.,
            0.,
            *[1.,0.,0.,0.]
        ], dtype=np.float64)
    
    def reset(self):
        super().reset()
        self.t_idx = 0
        self.state = self.default_state()

if __name__ == "__main__":
    lcfilter = LightcurveKFilter()
    num_steps = 2881
    kf_test_results = np.empty(num_steps)
    for i in tqdm.trange(num_steps - 1):
        lcfilter.predict(mutate=False)
        # print("Innovation: {}".format(lcfilter.state - lcfilter.model_flux[i+1]))
        lcfilter.update(lcfilter.model_flux[i+1])
        kf_test_results[i] = lcfilter.measure()
        
    plt.plot(kf_test_results, label='kf')
    plt.plot(lcfilter.model_flux[:num_steps], label='source data')
    plt.legend()
    plt.show()

    lc_time, lc_flux = get_stellar_candidate()

    lcfilter.reset()
    num_steps = 1200
    kf_test_results = np.empty(num_steps)
    for i in tqdm.trange(num_steps):
        lcfilter.predict()
        lcfilter.update(lc_flux[i])
        measurement = lcfilter.measure()
        kf_test_results[i] = measurement
        
    plt.plot(lc_flux[:num_steps], label="ref")
    plt.plot(kf_test_results, label='kf')
    plt.legend()
    plt.show()
