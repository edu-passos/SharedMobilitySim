"""PortoMicromobilityEnv: simple API for agents.

You ONLY need these two methods:

    env = PortoMicromobilityEnv("configs/network_prtp_10.yaml")
    obs = env.reset()
    obs, reward, done, info = env.step(action)

- obs: dict with:
    - 'fill_ratio': np.ndarray shape (N,)
    - 'soc':        np.ndarray shape (N,)
    - 'waiting':    np.ndarray shape (N,)
    - 'time_of_day': np.ndarray shape (2,) [sin(hour), cos(hour)]

- action: np.ndarray with 4 values in [0,1]:
    [a0, a1, a2, a3] → controls relocation & charging thresholds internally.

- reward: float, higher is better (already includes availability, relocation cost, energy cost)

If you're doing RL, you:
    - write a loop over episodes
    - choose actions based on obs
    - update your model using (obs, action, reward, next_obs, done)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from control.policies import REGISTRY as POLICY_REGISTRY
from sim.core import Sim, SimConfig
from sim.demand import effective_lambda
from sim.events import events
from sim.weather_mc import make_default_weather_mc as weather_mc


@dataclass
class ScoreWeights:
    """Weights for per-tick cost components in reward calculation."""

    alpha_unavailability: float = 100.0  # weight on (1 - availability)
    beta_reloc_km: float = 0.5  # weight on total relocation km (per tick)
    gamma_energy_cost: float = 10.0  # weight on charging cost € (per tick)


class PortoMicromobilityEnv:
    """Lightweight RL-style wrapper around Sim + planners.

    - Action: 4-dim vector in [0,1]^4 that controls relocation + charging knobs.
    - Observation: dict with per-station state + time-of-day features.
    - Reward: negative instantaneous cost (lower J -> higher reward).
    """

    def __init__(
        self,
        cfg_path: str | Path,
        *,
        score_weights: ScoreWeights | None = None,
        episode_hours: int | None = None,
        seed: int = 42,
    ) -> None:
        self.cfg_path = Path(cfg_path)
        self.seed = int(seed)

        with self.cfg_path.open(encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

        # Time horizon (in ticks)
        self.dt_min = int(self._cfg["time"]["dt_minutes"])
        horizon_h = int(self._cfg["time"]["horizon_hours"])
        self.episode_hours = episode_hours if episode_hours is not None else horizon_h
        self.max_steps = int(self.episode_hours * 60 / self.dt_min)

        # Score weights (per-tick cost)
        self.score_weights = score_weights or ScoreWeights()

        # Planners from YAML (default to greedy)
        ops_cfg = self._cfg.get("ops", {})
        planners_cfg = ops_cfg.get("planners", {})

        reloc_cfg = planners_cfg.get("relocation", {"name": "greedy", "params": {}})
        charge_cfg = planners_cfg.get("charging", {"name": "greedy", "params": {}})

        self.reloc_name = reloc_cfg.get("name", "greedy")
        self.base_reloc_params = reloc_cfg.get("params", {}) or {}

        self.charge_name = charge_cfg.get("name", "greedy")
        self.base_charge_params = charge_cfg.get("params", {}) or {}

        self.reloc_planner = POLICY_REGISTRY.get_relocation(self.reloc_name)
        self.charge_planner = POLICY_REGISTRY.get_charging(self.charge_name)

        # These will be initialized in reset()
        self.rng: np.random.Generator | None = None
        self.sim: Sim | None = None
        self.base_lambda: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.W = None
        self.events_matrix: np.ndarray | None = None
        self.step_idx: int = 0

    # -------------------- public API -------------------- #

    def reset(self, *, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset the simulation to t=0 and return initial observation."""
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed)

        # Build SimConfig
        cfg = self._cfg
        N = int(cfg["network"]["n_stations"])
        C = np.full(N, int(cfg["network"]["capacity_default"]))
        tmin = np.array(cfg["network"]["travel_time_min"], dtype=float).reshape(N, N)
        km = np.array(cfg["network"]["distance_km"], dtype=float).reshape(N, N)

        energy = cfg.get("energy", {})
        chargers = np.array(energy.get("chargers_per_station", [2] * N), dtype=int)
        charge_rate = np.array(energy.get("charge_rate_soc_per_hour", [0.25] * N), dtype=float)
        battery_kwh = float(energy.get("battery_kwh_per_vehicle", 0.5))
        energy_cost = float(energy.get("energy_cost_per_kwh_eur", 0.20))

        simcfg = SimConfig(
            dt_min=self.dt_min,
            horizon_h=self.episode_hours,
            capacity=C,
            travel_min=tmin,
            charge_rate=charge_rate,
            cost_km=km,
            chargers=chargers,
            battery_kwh=battery_kwh,
            energy_cost_per_kwh=energy_cost,
        )
        self.sim = Sim(simcfg, self.rng)

        # Demand & OD
        self.base_lambda = np.full(N, float(cfg["demand"]["base_lambda_per_dt"]))
        self.P = np.full((N, N), 1.0 / N, dtype=float)  # uniform OD for now

        # Weather & events
        self.W = weather_mc(dt_min=self.dt_min, seed=int(cfg.get("seed", 42)))
        self.events_matrix = events(self.max_steps, N, rng=self.sim.rng)

        self.step_idx = 0
        return self._build_obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        """Apply agent action, advance sim by one tick.

        Returns:
            Tuple of (obs, reward, done, info)
            - obs: observation dict
            - reward: float
            - done: bool, whether episode is over
            - info: dict with extra diagnostics
        """
        assert self.sim is not None, "Call reset() before step()."
        assert self.base_lambda is not None
        assert self.P is not None

        # 1) Map action ∈ R^4 (or [0,1]^4) to planner params
        reloc_params, charge_params = self._action_to_params(action)

        # 2) Time-of-day and exogenous factors
        hour = (self.step_idx * self.dt_min / 60.0) % 24
        _w_state = self.W.step()
        w_fac = self.W.factor
        ev_vec = self.events_matrix[self.step_idx]

        lam_t = effective_lambda(self.base_lambda, hour, weather_fac=w_fac, event_fac_vec=ev_vec)

        # 3) Compute plans using current state + params
        reloc = self.reloc_planner(
            self.sim.x,
            self.sim.cfg.capacity,
            self.sim.cfg.travel_min,
            params=reloc_params,
        )
        charge_plan = self.charge_planner(
            self.sim.x,
            self.sim.s,
            self.sim.cfg.chargers,
            lam_t,
            params=charge_params,
        )

        # 4) Advance simulation by one tick
        self.sim.step(
            lam_t,
            self.P,
            weather_fac=1.0,
            event_fac=None,  # already in lam_t
            reloc_plan=reloc,
            charging_plan=charge_plan,
        )

        log = self.sim.logs[-1]
        reward = self._compute_reward(log)

        self.step_idx += 1
        done = self.step_idx >= self.max_steps
        obs = self._build_obs()

        info = {
            "hour": hour,
            "weather_state": _w_state,
            "weather_factor": w_fac,
            "lam_t": lam_t,
            "reloc_plan": reloc,
            "charge_plan": charge_plan,
            "kpi": log,
        }
        return obs, reward, done, info

    # -------------------- internals -------------------- #

    def _action_to_params(self, a: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        """Map a 4-dim continuous action a ∈ R^4 to planner parameters.

        We clip to [0,1] and then scale into meaningful ranges.
        """
        a = np.asarray(a, dtype=float)
        if a.shape == ():
            raise ValueError("Action must be array-like, got scalar.")
        if a.size < 4:
            raise ValueError(f"Action must have at least 4 elements, got {a.size}.")
        a = np.clip(a, 0.0, 1.0)

        low = 0.1 + 0.2 * a[0]  # [0.1, 0.3]
        high = 0.6 + 0.3 * a[1]  # [0.6, 0.9]
        target = 0.4 + 0.4 * a[2]  # [0.4, 0.8]
        q_thresh = 0.2 + 0.6 * a[3]  # [0.2, 0.8]

        reloc_params = {
            **self.base_reloc_params,
            "low": low,
            "high": high,
            "target": target,
        }
        # keep hysteresis/max_moves from base params if set, else defaults
        if "hysteresis" not in reloc_params:
            reloc_params["hysteresis"] = 0.03
        if "max_moves" not in reloc_params:
            reloc_params["max_moves"] = 50

        charge_params = {
            **self.base_charge_params,
            "threshold_quantile": q_thresh,
        }

        return reloc_params, charge_params

    def _compute_reward(self, log: dict[str, Any]) -> float:
        """Per-tick reward: negative instantaneous cost.

        J_t = α (1 - availability) + β * reloc_km + γ * charge_cost_eur
        reward_t = -J_t
        """  # noqa: RUF002
        alpha = self.score_weights.alpha_unavailability
        beta = self.score_weights.beta_reloc_km
        gamma = self.score_weights.gamma_energy_cost

        availability = float(log.get("availability", 0.0))
        reloc_km = float(log.get("reloc_km", 0.0))
        charge_cost = float(log.get("charge_cost_eur", 0.0))

        J_t = alpha * (1.0 - availability) + beta * reloc_km + gamma * charge_cost
        return -J_t

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Build observation dict from current sim state.

        Simple version: per-station x, SoC, waiting queue + time-of-day encoding.
        """
        assert self.sim is not None
        N = self.sim.x.shape[0]

        # time-of-day features
        hour = (self.step_idx * self.dt_min / 60.0) % 24
        angle = 2.0 * np.pi * hour / 24.0
        tod = np.array([np.sin(angle), np.cos(angle)], dtype=float)

        # normalize x by capacity
        fill = self.sim.x.astype(float) / np.maximum(self.sim.cfg.capacity, 1)
        waiting = getattr(self.sim, "waiting", np.zeros_like(self.sim.x))

        return {
            "fill_ratio": fill.copy(),  # shape (N,)
            "soc": self.sim.s.copy(),  # shape (N,)
            "waiting": waiting.astype(float).copy(),  # shape (N,)
            "time_of_day": tod,  # shape (2,)
        }
