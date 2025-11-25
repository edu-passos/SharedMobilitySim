import numpy as np


def gini_availability(avail_by_station: np.ndarray) -> float:
    x = np.sort(avail_by_station.astype(float))
    n = len(x)
    if n == 0:
        return 0.0
    cum = np.cumsum(x)  # eu vou sempre achar isto engracado 💦
    # Gini on [0,1] availability per station (approx)
    return 1 - (2 * np.sum(cum) / (n * np.sum(x)) - (n + 1) / n) if np.sum(x) > 0 else 0.0
