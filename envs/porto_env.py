"""PortoMicromobilityEnv: simple API for agents.

You need these two methods:

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
class ScoreConfig:
    # weights (dimensionless priorities)
    w_availability: float
    w_reloc: float
    w_charge: float
    w_queue: float

    # scales (baseline magnitudes, per tick)
    A0_unavailability: float
    R0_reloc_km: float
    C0_charge_cost_eur: float
    Q0_queue_total: float

    eps: float


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
        episode_hours: int | None = None,
        seed: int = 42,
        reloc_name_override: str | None = None,
        charge_name_override: str | None = None,
    ) -> None:
        self.cfg_path = Path(cfg_path)
        self.seed = int(seed)

        with self.cfg_path.open(encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)
        score_cfg = self._cfg.get("score", {})
        # Time horizon (in ticks)
        self.dt_min = int(self._cfg["time"]["dt_minutes"])
        horizon_h = int(self._cfg["time"]["horizon_hours"])
        self.episode_hours = episode_hours if episode_hours is not None else horizon_h
        self.max_steps = int(self.episode_hours * 60 / self.dt_min)

        # Add weather and events state
        self._last_weather_factor = 1.0
        self._last_event_mean = 1.0
        self._last_event_max = 1.0

        # Score weights (per-tick cost)
        w_cfg = score_cfg.get("weights", {})
        s_cfg = score_cfg.get("scales", {})

        self.score_cfg = ScoreConfig(
            w_availability=float(w_cfg.get("w_availability", 5.0)),
            w_reloc=float(w_cfg.get("w_reloc", 1.0)),
            w_charge=float(w_cfg.get("w_charge", 1.0)),
            w_queue=float(w_cfg.get("w_queue", 4.0)),
            A0_unavailability=float(s_cfg.get("A0_unavailability", 0.25)),
            R0_reloc_km=float(s_cfg.get("R0_reloc_km", 1.0)),
            C0_charge_cost_eur=float(s_cfg.get("C0_charge_cost_eur", 0.01)),
            Q0_queue_total=float(s_cfg.get("Q0_queue_total", 10.0)),
            eps=float(score_cfg.get("eps", 1e-6)),
        )

        # Planners from YAML (default to greedy), with optional runtime overrides
        ops_cfg = self._cfg.get("ops", {})
        planners_cfg = ops_cfg.get("planners", {})

        reloc_cfg = planners_cfg.get("relocation", {"name": "greedy", "params": {}})
        charge_cfg = planners_cfg.get("charging", {"name": "greedy", "params": {}})

        yaml_reloc_name = str(reloc_cfg.get("name", "greedy"))
        yaml_charge_name = str(charge_cfg.get("name", "greedy"))

        self.reloc_name = str(reloc_name_override) if reloc_name_override is not None else yaml_reloc_name
        self.charge_name = str(charge_name_override) if charge_name_override is not None else yaml_charge_name

        # Base params still come from YAML; action will modulate these
        self.base_reloc_params = reloc_cfg.get("params", {}) or {}
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
        network_cfg = cfg["network"]
        N = int(network_cfg["n_stations"])
        C = np.full(N, int(network_cfg["capacity_default"]))
        tmin = np.array(network_cfg["travel_time_min"], dtype=float).reshape(N, N)
        km = np.array(network_cfg["distance_km"], dtype=float).reshape(N, N)

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
            cost_km=km,
            chargers=chargers,
            charge_rate=charge_rate,
            battery_kwh=battery_kwh,
            energy_cost_per_kwh=energy_cost,
        )
        self.sim = Sim(simcfg, self.rng)

        # Demand & OD
        self.base_lambda = np.full(N, float(cfg["demand"]["base_lambda_per_dt"]))
        self.P = np.full((N, N), 1.0 / N, dtype=float)  # uniform OD for now

        # Weather and events seeds
        w_seed = seed + 12345
        e_seed = seed + 67890

        # Weather & events
        self.W = weather_mc(dt_min=self.dt_min, seed=w_seed)
        self.events_matrix = events(self.max_steps, N, rng=np.random.default_rng(e_seed))

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

        self._last_weather_factor = float(w_fac)
        self._last_event_mean = float(np.mean(ev_vec))
        self._last_event_max = float(np.max(ev_vec))

        # 3) Compute plans using current state + params
        reloc_plan = self.reloc_planner(
            self.sim.x,
            self.sim.cfg.capacity,
            self.sim.cfg.cost_km,
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
            reloc_plan=reloc_plan,
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
            "reloc_plan": reloc_plan,
            "charge_plan": charge_plan,
            "kpi": log,
        }
        return obs, reward, done, info

    # -------------------- internals -------------------- #

    def _action_to_params(self, a: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        """Map a 4-dimensional continuous action (a ∈ R^4) to planner parameters.

        We clip to [0,1] and then scale them into meaningful ranges.
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
        charge_budget = 0.05 + 0.35 * a[3]  # [0.05, 0.40]

        # enforce consistent ordering
        high = max(low + 0.10, high)
        target = float(np.clip(target, low + 0.05, high - 0.05))

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

        # If present, allow eval/sweeps to override charging budget directly
        override = self.base_charge_params.get("charge_budget_frac_override", None)
        if override is not None:
            charge_budget = float(override)

        # do NOT pass override key into planner kwargs
        base_charge = dict(self.base_charge_params)
        base_charge.pop("charge_budget_frac_override", None)

        charge_params = {
            **base_charge,
            "charge_budget_frac": float(charge_budget),
        }

        return reloc_params, charge_params

    def _compute_reward(self, log: dict[str, Any]) -> float:
        """Per-tick reward: negative instantaneous cost.

        J_t = wA * (unavailability / A0)
            + wR * (reloc_km / R0)
            + wC * (charge_cost_eur / C0)
            + wQ * (queue_total / Q0)
        reward_t = -J_t
        """
        cfg = self.score_cfg

        availability = float(log.get("availability", 0.0))
        reloc_km = float(log.get("reloc_km", 0.0))
        charge_cost = float(log.get("charge_cost_eur", 0.0))
        queue_total = float(log.get("queue_total", 0.0))

        unavailability = 1.0 - availability

        A0 = max(cfg.A0_unavailability, cfg.eps)
        R0 = max(cfg.R0_reloc_km, cfg.eps)
        C0 = max(cfg.C0_charge_cost_eur, cfg.eps)
        Q0 = max(cfg.Q0_queue_total, cfg.eps)

        J_t = (
            cfg.w_availability * (unavailability / A0)
            + cfg.w_reloc * (reloc_km / R0)
            + cfg.w_charge * (charge_cost / C0)
            + cfg.w_queue * (queue_total / Q0)
        )
        return -J_t

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Build observation dict from current sim state.

        Simple version: per-station x, SoC, waiting queue + time-of-day encoding.
        """
        assert self.sim is not None
        # last tick KPI scalars (0 if no logs yet)
        last = self.sim.logs[-1] if getattr(self.sim, "logs", None) else {}
        last_reloc_km = float(last.get("reloc_km", 0.0))
        last_reloc_units = float(last.get("reloc_units", 0.0))  # if you log it; else stays 0
        last_charge_cost = float(last.get("charge_cost_eur", 0.0))
        last_queue_total = float(last.get("queue_total", 0.0))
        last_queue_rate = float(last.get("queue_rate", 0.0))

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
            "time_of_day": tod,  # shape (2,),
            "weather_factor": np.array([self._last_weather_factor], dtype=float),  # shape (1,)
            "event_stats": np.array([self._last_event_mean, self._last_event_max], dtype=float),  # shape (2,)
            "last_kpi": np.array(
                [
                    last_reloc_km,
                    last_reloc_units,
                    last_charge_cost,
                    last_queue_total,
                    last_queue_rate,
                ],
                dtype=float,
            ),  # shape (5,),
        }
