import numpy as np


def events(horizon_steps: int, N: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a matrix of demand events that modify the base demand."""
    if rng is None:
        rng = np.random.default_rng(4)

    fac = np.ones((horizon_steps, N), dtype=float)  # demand factor
    for _ in range(2):
        i = int(rng.integers(0, N))  # station index
        # TODO: Maybe support variable time steps
        length = int(rng.integers(36, 60))  # ticks if dt=5min (~3-5h)
        start = int(rng.integers(0, max(1, horizon_steps - length)))
        fac[start : start + length, i] = 1.8  # +80% demand boost
    return fac
