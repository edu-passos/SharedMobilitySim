import numpy as np


def events(horizon_steps, N, rng=np.random.default_rng(4)):
    fac = np.ones((horizon_steps, N), dtype=float)
    for _ in range(2):
        i = int(rng.integers(0, N))
        length = int(rng.integers(36, 60))  # ticks if dt=5min (~3–5h)
        start = int(rng.integers(0, max(1, horizon_steps - length)))
        fac[start : start + length, i] = 1.8  # +80% demand boost
    return fac
