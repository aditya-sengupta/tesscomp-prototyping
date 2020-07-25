import numpy as np
import copy
eps = 1e-6

def jacobian(f, p):
    '''
    Finds the Jacobian matrix of the function f at the point p.

    Arguments
    ---------
    f : callable
    A function F^n -> F^m (F = R usually but not assumed);
    takes in an n-length numpy array and returns an m-length numpy array.

    p : np.ndarray, shape (n,)
    The n-length numpy array around which we want to approximate the Jacobian.

    Returns
    -------
    jac : np.ndarray, shape (m, n)
    The Jacobian matrix (the matrix of first-order partial derivatives).
    '''
    n = len(p)
    center = f(p)
    rights = f(p + eps * np.eye(n))
    lefts = f(p - eps * np.eye(n))
    return .5 * ((rights.T - center) / eps + (center - lefts.T) / eps).T

class EKFilter():
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

    def predict(self):
        self.prev_P = copy.deepcopy(self.P)
        if isinstance(self.f, np.ndarray):
            self.state = self.f @ self.state
            self.P = self.f @ self.P @ self.f + self.Q
        else:
            A = jacobian(self.f, copy.deepcopy(self.state))
            self.state = self.f(self.state)
            self.P = A @ self.P @ A + self.Q

    def update(self, measurement):
        if isinstance(self.h, np.ndarray):
            C = jacobian(self.h, copy.deepcopy(self.state))
            innovation = measurement - self.h(self.state)
        else:
            C = self.h
            innovation = measurement - self.h @ self.state
        if not self.steady_state:
            self.K = self.P @ C.T @ np.linalg.inv(C @ self.P @ C.T + self.R)
            self.P -= self.K @ (C @ self.P)
            if np.allclose(self.P, self.prev_P):
                self.steady_state = True
        innovation = measurement - C @ self.state
        self.state = self.state + self.K @ innovation
