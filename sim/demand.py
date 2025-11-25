import numpy as np


def _circ_gauss(hour, mu, sigma):
    """Circular Gaussian over 24h"""
    h = np.mod(hour, 24.0)
    d = np.minimum(np.abs(h - mu), 24.0 - np.abs(h - mu))
    return np.exp(-(d**2) / (2 * sigma**2))


def diurnal(base_lambda, hour, amp=1.4, mu_am=8, mu_pm=18, sigma=1.2):
    bump = amp * (_circ_gauss(hour, mu_am, sigma) + _circ_gauss(hour, mu_pm, sigma))
    return base_lambda * (1.0 + bump)


def effective_lambda(base_lambda_vec, hour, weather_fac=1.0, event_fac_vec=None):
    lam = diurnal(base_lambda_vec, hour)
    lam = lam * weather_fac
    if event_fac_vec is not None:
        lam = lam * event_fac_vec
    return lam
