import numpy as np


def _circ_gauss(hour: float, mu: float, sigma: float) -> float:
    """Circular Gaussian over 24h."""
    h = np.mod(hour, 24.0)
    d = np.minimum(np.abs(h - mu), 24.0 - np.abs(h - mu))
    return np.exp(-(d**2) / (2 * sigma**2))


def diurnal(
    base_lambda_vec: np.ndarray,
    hour: float,
    amp: float = 1.4,
    mu_am: float = 8,
    mu_pm: float = 18,
    sigma: float = 1.2,
) -> np.ndarray:
    """Compute diurnal demand multiplier at given hour."""
    bump = amp * (_circ_gauss(hour, mu_am, sigma) + _circ_gauss(hour, mu_pm, sigma))
    return base_lambda_vec * (1.0 + bump)


def effective_lambda(
    base_lambda_vec: np.ndarray,
    hour: float,
    weather_fac: float = 1.0,
    event_fac_vec: np.ndarray = None,
) -> np.ndarray:
    """Compute effective demand rate vector at given hour."""
    lam = diurnal(base_lambda_vec, hour)
    lam = lam * weather_fac
    if event_fac_vec is not None:
        lam = lam * event_fac_vec
    return lam
