"""
bandit_param_arms.py

Episode-level Multi-Armed Bandit (UCB1) over a discrete grid of *planner parameter arms*.

Each arm is a tuple:
  - km_budget:          relocation budget parameter passed to the relocation planner
  - charge_budget_frac: charging budget fraction passed to the charging planner

Per episode:
  1) The bandit selects an arm (km_budget, charge_budget_frac).
  2) We create a fresh PortoMicromobilityEnv with an episode-specific RNG seed.
  3) We override env.base_reloc_params[RELOC_BUDGET_KEY] and
     env.base_charge_params[CHARGE_BUDGET_KEY] with the chosen arm values.
  4) We roll out the episode using a constant action vector (default_action) each tick.
     The action is kept fixed so that the bandit is effectively searching over planner
     budgets, not learning a reactive control policy.

Reward and metrics:
  - The environment reward is assumed to be reward_t = -J_t, where J_t is the normalized
    per-tick objective:
        J_t = wA * (unavailability / A0)
            + wR * (reloc_km / R0)
            + wC * (charge_cost_eur / C0)
            + wQ * (queue_total / Q0)
  - We recompute J_t from sim.logs to produce a consistent episode-level KPI summary.
  - The bandit optimizes for *total episode reward* (sum over ticks), i.e., it maximizes
    -sum(J_t). This is equivalent to minimizing cumulative cost under the above assumption.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.porto_env import PortoMicromobilityEnv


# Bandit: UCB1
class UCB1Bandit:
    """UCB1 over discrete arms with scalar rewards (maximize)."""

    def __init__(self, n_arms: int, *, c: float = 2.0, seed: int = 0) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        self.c = float(c)
        self.rng = np.random.default_rng(seed)

        self.n_arms = int(n_arms)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.means = np.zeros(self.n_arms, dtype=float)
        self.total_pulls = 0

    def select_arm(self) -> int:
        """Pick an arm index using UCB1, exploring untried arms first."""
        untried = np.where(self.counts == 0)[0]
        if untried.size > 0:
            return int(self.rng.choice(untried))

        # Standard: use t = total_pulls + 1 at selection time
        t = self.total_pulls + 1
        bonus = self.c * np.sqrt(np.log(t) / self.counts)
        ucb = self.means + bonus
        return self._argmax_random_tie(ucb)

    def update(self, arm_idx: int, reward: float) -> None:
        self.total_pulls += 1
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        old = self.means[arm_idx]
        self.means[arm_idx] = old + (float(reward) - old) / n

    def best_arm(self) -> int:
        tried = self.counts > 0
        if not np.any(tried):
            return 0
        masked = np.where(tried, self.means, -np.inf)
        return self._argmax_random_tie(masked)

    def _argmax_random_tie(self, x: np.ndarray) -> int:
        m = np.max(x)
        idx = np.flatnonzero(np.isclose(x, m))
        return int(self.rng.choice(idx))


# Arms: grid construction
def make_param_arms(
    km_budgets: List[float],
    charge_fracs: List[float],
) -> List[Dict[str, float]]:
    arms: List[Dict[str, float]] = []
    for km in km_budgets:
        for c in charge_fracs:
            arms.append({"km_budget": float(km), "charge_budget_frac": float(c)})
    return arms


# KPI utilities
def _arr(logs: List[Dict[str, Any]], key: str, default: float = 0.0) -> np.ndarray:
    return np.array([r.get(key, default) for r in logs], dtype=float)

def apply_scenario(
    env: PortoMicromobilityEnv,
    *,
    scenario: str,
    seed: int,
    hotspot_j: int = 0,
    hotspot_p: float = 0.6,
    hetero_strength: float = 0.6,
    event_scale: float = 1.5,
) -> None:
    scenario = str(scenario).lower().strip()
    if scenario in ("", "baseline", "base"):
        return

    assert env.base_lambda is not None, "env.reset() must be called before apply_scenario()"
    assert env.P is not None
    assert env.events_matrix is not None

    N = int(env.base_lambda.shape[0])

    if scenario == "hotspot_od":
        j0 = int(np.clip(hotspot_j, 0, N - 1))
        p_hot = float(np.clip(hotspot_p, 0.0, 0.99))
        P = np.full((N, N), (1.0 - p_hot) / max(N - 1, 1), dtype=float)
        P[:, j0] = p_hot
        if N > 1:
            for i in range(N):
                rem = 1.0 - P[i, j0]
                P[i, :] = rem / (N - 1)
                P[i, j0] = 1.0 - rem
        env.P = P
        return

    if scenario == "hetero_lambda":
        rng = np.random.default_rng(int(seed) + 999)
        f = rng.normal(loc=1.0, scale=float(hetero_strength), size=N)
        f = np.clip(f, 0.3, 2.5)
        f = f / max(float(np.mean(f)), 1e-12)
        env.base_lambda = env.base_lambda * f
        return

    if scenario == "event_heavy":
        E = env.events_matrix.astype(float)
        E_scaled = 1.0 + float(event_scale) * (E - 1.0)
        E_scaled = np.clip(E_scaled, 0.0, None)
        env.events_matrix = E_scaled
        return

    raise ValueError(f"Unknown scenario '{scenario}'. Use: baseline, hotspot_od, hetero_lambda, event_heavy.")

def compute_episode_kpis(env: PortoMicromobilityEnv, total_reward: float) -> Dict[str, Any]:
    sim = env.sim
    if sim is None or not sim.logs:
        return {}

    logs = sim.logs
    T = len(logs)
    dt_min = sim.cfg.dt_min
    scfg = env.score_cfg  # ScoreConfig dataclass

    availability = _arr(logs, "availability", 0.0)
    unavailability = 1.0 - availability

    reloc_km = _arr(logs, "reloc_km", 0.0)
    charge_cost = _arr(logs, "charge_cost_eur", 0.0)
    queue_total = _arr(logs, "queue_total", 0.0)

    # Volumes / service
    demand_total = int(_arr(logs, "demand_total", 0.0).sum())
    served_new_total = int(_arr(logs, "served_new_total", 0.0).sum())
    served_total = int(_arr(logs, "served_total", 0.0).sum())
    unmet_total = int(_arr(logs, "unmet", 0.0).sum())

    availability_demand_weighted = 1.0 if demand_total == 0 else (served_new_total / demand_total)
    unmet_rate = 0.0 if demand_total == 0 else (unmet_total / demand_total)
    availability_tick_avg = float(np.mean(availability))

    # Queue / wait proxy
    total_queue_time_min = float(queue_total.sum() * dt_min)
    avg_wait_min_proxy = float(total_queue_time_min / max(served_total, 1))
    queue_total_p95 = float(np.percentile(queue_total, 95)) if T else 0.0

    # Queue stability: Δqueue_total
    dq = np.diff(queue_total, prepend=queue_total[0])
    dq_mean = float(np.mean(dq)) if T else 0.0
    dq_p95 = float(np.percentile(dq, 95)) if T else 0.0

    # Normalized objective recompute (must match env reward)
    A0 = max(float(scfg.A0_unavailability), float(scfg.eps))
    R0 = max(float(scfg.R0_reloc_km), float(scfg.eps))
    C0 = max(float(scfg.C0_charge_cost_eur), float(scfg.eps))
    Q0 = max(float(scfg.Q0_queue_total), float(scfg.eps))

    J_avail_t = float(scfg.w_availability) * (unavailability / A0)
    J_reloc_t = float(scfg.w_reloc) * (reloc_km / R0)
    J_charge_t = float(scfg.w_charge) * (charge_cost / C0)
    J_queue_t = float(scfg.w_queue) * (queue_total / Q0)

    J_t = J_avail_t + J_reloc_t + J_charge_t + J_queue_t
    J_run = float(np.mean(J_t)) if T else 0.0

    # Consistency: reward ≈ -sum(J_t)
    reward_plus_sumJ = float(total_reward + float(np.sum(J_t)))

    return {
        "ticks": int(T),
        "total_reward": float(total_reward),
        "J_run": float(J_run),
        "reward_plus_sumJ": float(reward_plus_sumJ),
        # Decomposition (per-tick means)
        "J_avail": float(np.mean(J_avail_t)) if T else 0.0,
        "J_reloc": float(np.mean(J_reloc_t)) if T else 0.0,
        "J_charge": float(np.mean(J_charge_t)) if T else 0.0,
        "J_queue": float(np.mean(J_queue_t)) if T else 0.0,
        # Service
        "availability_tick_avg": float(availability_tick_avg),
        "availability_demand_weighted": float(availability_demand_weighted),
        "unmet_rate": float(unmet_rate),
        "unmet_total": int(unmet_total),
        "avg_wait_min_proxy": float(avg_wait_min_proxy),
        # Ops
        "relocation_km_total": float(reloc_km.sum()),
        "charging_cost_eur_total": float(charge_cost.sum()),
        # Queue
        "queue_total_avg": float(queue_total.mean()) if T else 0.0,
        "queue_total_p95": float(queue_total_p95),
        # Stability
        "dq_mean": float(dq_mean),
        "dq_p95": float(dq_p95),
        # Snapshot
        "score_cfg": asdict(env.score_cfg),
    }


# Episode runner with param overrides
RELOC_BUDGET_KEY = "km_budget" 
CHARGE_BUDGET_KEY = "charge_budget_frac"


def run_one_episode_with_arm(
    *,
    cfg_path: str,
    hours: int,
    seed: int,
    reloc_planner: Optional[str],
    charge_planner: Optional[str],
    arm: Dict[str, float],
    default_action: np.ndarray,
    scenario: str,
    scenario_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    env = PortoMicromobilityEnv(
        cfg_path=cfg_path,
        episode_hours=hours,
        seed=seed,
        reloc_name_override=reloc_planner,
        charge_name_override=charge_planner,
    )
    obs = env.reset()
    apply_scenario(env, scenario=scenario, seed=seed, **(scenario_params or {}))

    # Apply arm as param override (per-episode)
    # We write into base_*_params so _action_to_params merges these and passes down.
    km_budget = float(arm["km_budget"])
    c_frac = float(arm["charge_budget_frac"])

    # Reloc override
    env.base_reloc_params = dict(env.base_reloc_params)  # ensure local copy
    env.base_reloc_params[RELOC_BUDGET_KEY] = km_budget

    # Charge override
    env.base_charge_params = dict(env.base_charge_params)
    env.base_charge_params[CHARGE_BUDGET_KEY] = c_frac

    done = False
    total_reward = 0.0

    a = np.asarray(default_action, dtype=float).reshape(4,)
    a = np.clip(a, 0.0, 1.0)

    while not done:
        obs, reward, done, _info = env.step(a)
        total_reward += float(reward)

    kpis = compute_episode_kpis(env, total_reward=total_reward)
    meta = {
        "seed": int(seed),
        "hours": int(hours),
        "scenario": scenario,
        "scenario_params": scenario_params,
        "arm": {"km_budget": km_budget, "charge_budget_frac": c_frac},
        "default_action": a.tolist(),
        "reloc_planner": reloc_planner,
        "charge_planner": charge_planner,
    }
    return kpis, meta


# Main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/network_porto10.yaml")
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed0", type=int, default=42)

    # Planners to use (these should exist in your registry)
    p.add_argument("--reloc", default="budgeted", help="Relocation planner name (e.g., budgeted, greedy, noop).")
    p.add_argument("--charge", default="greedy", help="Charging planner name (e.g., greedy, slack, noop).")

    # Default action (kept constant; arms do the budget control)
    p.add_argument(
        "--default_action",
        type=float,
        nargs=4,
        default=[0.5, 0.5, 0.5, 0.5],
        help="Constant action applied every step. Arms override budget params.",
    )
    # scenarios
    p.add_argument("--scenario", default="baseline", help="baseline | hotspot_od | hetero_lambda | event_heavy")
    p.add_argument("--scenario_params_json", default="", help="JSON dict of scenario params (optional).")

    # Arm grid
    p.add_argument("--km_budgets", type=float, nargs="+", default=[0, 5, 10, 20, 40])
    p.add_argument("--charge_fracs", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])

    # Bandit
    p.add_argument("--ucb_c", type=float, default=2.0)

    # Output
    p.add_argument("--out", default="", help="If set, write JSON here; else print to stdout.")
    args = p.parse_args()

    cfg_path = str(args.config)
    hours = int(args.hours)

    scenario = str(args.scenario).strip()
    scenario_params: Dict[str, Any] = {}
    if args.scenario_params_json.strip():
        scenario_params = json.loads(args.scenario_params_json)
        if not isinstance(scenario_params, dict):
            raise ValueError("--scenario_params_json must be a JSON dict.")
    
    arms = make_param_arms(km_budgets=list(args.km_budgets), charge_fracs=list(args.charge_fracs))
    bandit = UCB1Bandit(n_arms=len(arms), c=float(args.ucb_c), seed=int(args.seed0))

    default_action = np.asarray(args.default_action, dtype=float).reshape(4,)
    reloc_name = str(args.reloc) if args.reloc else None
    charge_name = str(args.charge) if args.charge else None

    episodes_out: List[Dict[str, Any]] = []

    for ep in range(int(args.episodes)):
        arm_idx = bandit.select_arm()
        arm = arms[arm_idx]

        # New randomness each episode
        seed = int(args.seed0) + ep

        kpis, meta = run_one_episode_with_arm(
            cfg_path=cfg_path,
            hours=hours,
            seed=seed,
            reloc_planner=reloc_name,
            charge_planner=charge_name,
            arm=arm,
            default_action=default_action,
            scenario=scenario,
            scenario_params=scenario_params,  
        )

        reward = -float(kpis.get("J_run", 0.0))
        bandit.update(arm_idx, reward)

        row = {
            "episode": int(ep),
            "chosen_arm_idx": int(arm_idx),
            "chosen_arm": dict(arm),
            "bandit_mean_after": float(bandit.means[arm_idx]),
            "bandit_pulls_after": int(bandit.counts[arm_idx]),
            "bandit_best_arm_idx": int(bandit.best_arm()),
            "bandit_best_mean": float(bandit.means[bandit.best_arm()]),
            **meta,
            **kpis,
        }
        episodes_out.append(row)

    best_idx = bandit.best_arm()
    summary = {
        "config": cfg_path,
        "hours": hours,
        "episodes": int(args.episodes),
        "seed0": int(args.seed0),
        "reloc_planner": reloc_name,
        "charge_planner": charge_name,
        "default_action": default_action.tolist(),
        "ucb_c": float(args.ucb_c),
        "n_arms": int(len(arms)),
        "arms": arms,
        "best_arm_idx": int(best_idx),
        "best_arm": dict(arms[best_idx]),
        "best_arm_mean_neg_J_run": float(bandit.means[best_idx]),
        "best_arm_pulls": int(bandit.counts[best_idx]),
    }

    out_obj = {"summary": summary, "episodes": episodes_out}

    if args.out.strip():
        import os

        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2)
        print(f"Saved: {args.out}")
        print(json.dumps(
    {"best_arm": summary["best_arm"], "best_mean_neg_J_run": summary["best_arm_mean_neg_J_run"]}, indent=2))
    else:
        print(json.dumps(out_obj, indent=2))


if __name__ == "__main__":
    main()