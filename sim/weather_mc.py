from dataclasses import dataclass, field

import numpy as np


@dataclass
class WeatherMC:
    """A discrete-time Markov chain modeling weather states affecting demand."""

    states: tuple[str, ...]  # e.g. ("clear","cloudy","rain","storm")
    P: np.ndarray  # (S,S) row-stochastic transition matrix
    factors: dict[str, float]  # multiplicative demand factor per state
    state_idx: int  # current index in states
    update_every_ticks: int = 12  # e.g. if dt=5 min, 12 ticks = 1 hour
    _tick_counter: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    def __post_init__(self) -> None:
        """Validate inputs."""
        S = len(self.states)
        if self.P.shape != (S, S):
            raise ValueError(f"P must be shape ({S},{S})")

        row_sum = self.P.sum(axis=1)
        if not np.allclose(row_sum, 1.0, atol=1e-8):
            raise ValueError("Each row of P must sum to 1")
        if set(self.states) - set(self.factors):
            raise ValueError("factors missing for some states")

    @property
    def state(self) -> str:
        """Return current weather state."""
        return self.states[self.state_idx]

    @property
    def factor(self) -> float:
        """Return current demand factor."""
        return float(self.factors[self.state])

    def step(self) -> str:
        """Advance internal counter; change weather only when counter hits cadence."""
        self._tick_counter += 1
        if self._tick_counter % self.update_every_ticks == 0:
            self.state_idx = int(self.rng.choice(len(self.states), p=self.P[self.state_idx]))
        return self.state


def make_default_weather_mc(dt_min: int, seed: int = 42) -> WeatherMC:
    """Return default 4-state chain calibrated for urban micromobility.

    Update cadence = 60 minutes by default.
    """
    states = ("clear", "cloudy", "rain", "storm")
    # Row-stochastic transition matrix (clear/cloudy are sticky; storms are brief)
    P = np.array(
        [
            [0.86, 0.11, 0.03, 0.00],  # clear -> ...
            [0.10, 0.80, 0.09, 0.01],  # cloudy -> ...
            [0.05, 0.25, 0.65, 0.05],  # rain   -> ...
            [0.02, 0.18, 0.45, 0.35],  # storm  -> ...
        ],
        dtype=float,
    )
    factors = {"clear": 1.00, "cloudy": 0.90, "rain": 0.60, "storm": 0.45}
    ticks_per_hour = max(round(60 / dt_min), 1)
    return WeatherMC(
        states=states,
        P=P,
        factors=factors,
        state_idx=0,  # start at 'clear'
        update_every_ticks=ticks_per_hour,  # change weather hourly
        rng=np.random.default_rng(seed),
    )
