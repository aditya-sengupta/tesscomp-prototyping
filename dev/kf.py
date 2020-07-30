import numpy as np
import copy
try:
    from tqdm import tqdm
except ImportError:
    pass

eps = 1e-6

def jacobian(f, p, **kwargs):
    '''
    Finds the Jacobian matrix of the function f at the point p.

    Arguments
    ---------
    f : callable
    A function F^n -> F^m (F = R or C);
    takes in an n-length numpy array and returns an m-length numpy array.

    p : np.ndarray, shape (n,)
    The n-length numpy array around which we want to approximate the Jacobian.

    Returns
    -------
    jac : np.ndarray, shape (m, n)
    The Jacobian matrix (the matrix of first-order partial derivatives).
    '''
    n = len(p)
    center = f(p, **kwargs)
    rights = f(p + eps * np.eye(n), **kwargs)
    lefts = f(p - eps * np.eye(n), **kwargs)
    return .5 * ((rights.T - center) / eps + (center - lefts.T) / eps).T

class KFilter:
    '''
    Interface to a Kalman Filter or Extended Kalman Filter.

    Attributes
    ----------
    f : callable or np.ndarray
    The state-transition function or matrix.

    Q : np.ndarray, (s, s).
    The covariance of the state-transition model.

    h : callable or np.ndarray
    The measurement function or matrix.

    R : np.ndarray, (m, m)
    The covariance of the measurement model.

    s : int
    The dimension of the state.

    m : int
    The dimension of the measurement.

    state : np.ndarray, (s,)
    The state vector.

    P : np.ndarray, (s, s)
    The true state covariance.

    prev_P : np.ndarray, (s, s)
    The true state covariance one timestep ago, to check steady-state.

    steady_state : bool
    Whether the filter is in steady state (and can skip covariance updates).

    predict_chain : np.ndarray, (s, num_timesteps)
    The chain of state predictions made at each timestep.

    filter_chain : np.ndarray, (s, num_timesteps)
    The chain of MMSE state estimates at each timestep.

    Methods
    -------
    check_consistency()
    Checks whether dimensions match up.

    predict()
    Carries out a Kalman prediction.

    update(measurement:np.ndarray, (m,))
    Carries out a Kalman update.

    run_kf(signal:np.ndarray, (m, num_timesteps), progress:bool)
    Runs the filter in a predict-update cycle with the measurements in 'signal'.
    Does or doesn't use tqdm for progress based on 'progress'.

    reset()
    Resets the chains and the discovered covariance.
    '''
    def __init__(self, f, Q, h, R, state=None):
        self.f = f
        self.Q = Q
        self.h = h
        self.R = R
        self.s = Q.shape[0]
        self.m = R.shape[0]
        if state is None:
            self.state = np.zeros(self.s)
        else:
            self.state = state
        self.prev_P = np.zeros((self.s, self.s))
        self.P = np.zeros((self.s, self.s))
        self.steady_state = False
        self.predict_chain = np.empty((self.s, 0))
        self.filter_chain = np.empty((self.s, 0))
        self.check_consistency()

    def check_consistency(self):
        assert (isinstance(self.f, np.ndarray) and self.f.shape == (self.s, self.s)) or callable(self.f), "state transition is invalid"
        assert (isinstance(self.h, np.ndarray) and self.h.shape == (self.m, self.s)) or callable(self.h), "state measurement is invalid"
        assert isinstance(self.Q, np.ndarray) and self.Q.shape == (self.s, self.s), "model covariance is invalid"
        assert isinstance(self.R, np.ndarray) and self.R.shape == (self.m, self.m), "measurement covariance is invalid"

    def measure(self):
        if callable(self.h):
            return self.h(self.state)
        else:
            return self.h @ self.state

    def predict(self, **kwargs):
        self.prev_P = copy.deepcopy(self.P)
        if isinstance(self.f, np.ndarray):
            self.state = self.f @ self.state
            self.P = self.f @ self.P @ self.f + self.Q
        else:
            A = jacobian(self.f, copy.deepcopy(self.state), **kwargs)
            self.state = self.f(self.state)
            self.P = A @ self.P @ A + self.Q

    def update(self, measurement):
        if isinstance(self.h, np.ndarray):
            C = self.h
        else:
            C = jacobian(self.h, copy.deepcopy(self.state))
        innovation = measurement - self.measure()
        if not self.steady_state:
            self.K = self.P @ C.T @ np.linalg.inv(C @ self.P @ C.T + self.R)
            self.P -= self.K @ (C @ self.P)
            if np.allclose(self.P, self.prev_P):
                self.steady_state = True
        innovation = measurement - self.measure()
        self.state = self.state + self.K @ innovation

    def run_kf(self, signal, progress=True, **kwargs):
        predict_chain = np.zeros(len(signal), self.s)
        filter_chain = np.zeros(len(signal), self.s)
        if progress and 'tqdm' in globals():
            track_progress = tqdm
        else:
            track_progress = lambda x: x
        try:
            for i, m in enumerate(track_progress(signal)):
                self.predict(**kwargs)
                predict_chain[i] = self.state
                self.update(m, **kwargs)
                filter_chain[i] = self.state
        except KeyboardInterrupt:
            print("Filter terminated early, saving results.")
        for objchain, chain in zip([self.predict_chain, self.filter_chain], [predict_chain, filter_chain]):
            objchain = np.hstack((objchain, chain))

    def reset(self):
        self.predict_chain = np.empty((self.s, 0))
        self.filter_chain = np.empty((self.s, 0))
        self.steady_state = False
        self.prev_P = np.zeros_like(self.P)
        self.P = np.zeros_like(self.P)
        self.state = np.zeros_like(self.state)
