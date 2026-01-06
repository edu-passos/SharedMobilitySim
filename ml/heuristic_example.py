"""
eval_heuristic_agent.py

Evaluate a real-time (tick-by-tick) adaptive heuristic agent on PortoMicromobilityEnv,
using the SAME normalized objective decomposition as eval_policies.py:

    J_t = wA * (unavailability / A0)
        + wR * (reloc_km / R0)
        + wC * (charge_cost_eur / C0)
        + wQ * (queue_total / Q0)

This script:
- Runs multiple seeds
- Optionally applies scenarios (baseline / hotspot_od / hetero_lambda / event_heavy)
- Outputs a per-seed episode table + aggregate mean±std
- Saves JSON to --out (same schema style as eval_policies.py: episodes + summary)

NOTE:
- We do NOT override score weights here; we trust env.score_cfg (loaded from YAML config).
- The agent outputs action ∈ [0,1]^4 each tick; env maps it to planner params internally.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.porto_env import PortoMicromobilityEnv


# -----------------------------
# Scenario application (copy of your eval_policies.py)
# -----------------------------
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


# -----------------------------
# Heuristic agent (your logic; unchanged except minor safety)
# -----------------------------
class HeuristicAgent:
    """Tick-level adaptive heuristic policy producing action ∈ [0,1]^4."""

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        fill = np.asarray(obs["fill_ratio"], dtype=float)  # (N,)
        soc = np.asarray(obs["soc"], dtype=float)          # (N,)
        waiting = np.asarray(obs["waiting"], dtype=float)  # (N,)
        tod = np.asarray(obs["time_of_day"], dtype=float)  # (2,)

        avg_fill = float(np.mean(fill))
        empty_frac = float(np.mean(fill < 0.1))
        full_frac = float(np.mean(fill > 0.9))

        avg_soc = float(np.mean(soc))
        avg_wait = float(np.mean(waiting))
        high_wait_frac = float(np.mean(waiting > 5))

        # Decode hour from [sin, cos]
        sin_h, cos_h = float(tod[0]), float(tod[1])
        angle = float(np.arctan2(sin_h, cos_h))
        if angle < 0:
            angle += 2 * np.pi
        hour = 24.0 * angle / (2 * np.pi)

        is_morning_peak = 7 <= hour <= 10
        is_evening_peak = 17 <= hour <= 20
        is_night = (0 <= hour <= 5) or (hour >= 23)

        # a0: low threshold control (higher => more needy stations => more relocation)
        base_a0 = 0.4 if (is_morning_peak or is_evening_peak) else 0.2
        pressure = 0.5 * high_wait_frac + 0.5 * empty_frac
        a0 = float(np.clip(base_a0 + pressure, 0.0, 1.0))

        # a1: high threshold control (lower => more donors => more relocation)
        if is_morning_peak or is_evening_peak:
            base_a1 = 0.2
        elif is_night:
            base_a1 = 0.7
        else:
            base_a1 = 0.4
        congestion = full_frac
        a1 = float(np.clip(base_a1 - 0.5 * congestion, 0.0, 1.0))

        # a2: target fill control (raise target under high demand)
        wait_scale = float(np.clip(avg_wait / 20.0, 0.0, 1.0))
        a2 = float(np.clip(0.3 + 0.7 * wait_scale, 0.0, 1.0))

        # a3: charging control (higher when SoC low or demand pressure high; force higher at night)
        demand_pressure = (avg_wait / 10.0) + high_wait_frac
        soc_lack = 1.0 - avg_soc
        drive = float(np.clip(0.5 * soc_lack + 0.5 * demand_pressure, 0.0, 1.0))
        if is_night:
            drive = max(drive, 0.7)
        a3 = float(np.clip(drive, 0.0, 1.0))

        return np.array([a0, a1, a2, a3], dtype=float)


# -----------------------------
# KPIs: normalized objective recompute (aligned with eval_policies.py)
# -----------------------------
def compute_episode_kpis(env: PortoMicromobilityEnv, total_reward: float) -> Dict[str, Any]:
    sim = env.sim
    if sim is None or not sim.logs:
        return {}

    logs = sim.logs
    T = len(logs)
    dt_min = sim.cfg.dt_min
    scfg = env.score_cfg

    def arr(key: str, default: float = 0.0) -> np.ndarray:
        return np.array([r.get(key, default) for r in logs], dtype=float)

    # service
    demand_total = int(arr("demand_total").sum())
    served_total = int(arr("served_total").sum())
    served_new_total = int(arr("served_new_total").sum())
    unmet_total = int(arr("unmet").sum())

    availability_demand_weighted = 1.0 if demand_total == 0 else (served_new_total / demand_total)
    unmet_rate = 0.0 if demand_total == 0 else (unmet_total / demand_total)
    availability_tick_avg = float(arr("availability").mean())

    # queue
    queue_total = arr("queue_total")
    queue_rate = arr("queue_rate")

    total_queue_time_min = float(queue_total.sum() * dt_min)
    avg_wait_min_proxy = float(total_queue_time_min / max(served_total, 1))

    queue_total_p95 = float(np.percentile(queue_total, 95)) if T else 0.0
    queue_rate_p95 = float(np.percentile(queue_rate, 95)) if T else 0.0

    if T > 1:
        dq = queue_total[1:] - queue_total[:-1]
        queue_delta_mean = float(np.mean(dq))
        queue_delta_p95 = float(np.percentile(dq, 95))
    else:
        queue_delta_mean = 0.0
        queue_delta_p95 = 0.0

    # ops
    reloc_km_total = float(arr("reloc_km").sum())
    charge_cost_eur_total = float(arr("charge_cost_eur").sum())
    charge_energy_kwh_total = float(arr("charge_energy_kwh").sum())
    charge_util_avg = float(arr("charge_utilization").mean())

    # normalized objective decomposition
    unavailability = 1.0 - arr("availability")

    A0 = max(float(scfg.A0_unavailability), float(scfg.eps))
    R0 = max(float(scfg.R0_reloc_km), float(scfg.eps))
    C0 = max(float(scfg.C0_charge_cost_eur), float(scfg.eps))
    Q0 = max(float(scfg.Q0_queue_total), float(scfg.eps))

    J_avail = float(scfg.w_availability) * (unavailability / A0)
    J_reloc = float(scfg.w_reloc) * (arr("reloc_km") / R0)
    J_charge = float(scfg.w_charge) * (arr("charge_cost_eur") / C0)
    J_queue = float(scfg.w_queue) * (queue_total / Q0)

    J_t = J_avail + J_reloc + J_charge + J_queue
    J_run = float(np.mean(J_t))

    reward_plus_sumJ = float(total_reward + float(np.sum(J_t)))

    return {
        "ticks": int(T),
        "total_reward": float(total_reward),
        "J_run": float(J_run),
        "reward_plus_sumJ": float(reward_plus_sumJ),
        "J_avail_run": float(np.mean(J_avail)),
        "J_reloc_run": float(np.mean(J_reloc)),
        "J_charge_run": float(np.mean(J_charge)),
        "J_queue_run": float(np.mean(J_queue)),
        # service
        "availability_tick_avg": float(availability_tick_avg),
        "availability_demand_weighted": float(availability_demand_weighted),
        "unmet_total": int(unmet_total),
        "unmet_rate": float(unmet_rate),
        "avg_wait_min_proxy": float(avg_wait_min_proxy),
        # queue
        "queue_total_avg": float(queue_total.mean()) if T else 0.0,
        "queue_total_p95": float(queue_total_p95),
        "queue_rate_avg": float(queue_rate.mean()) if T else 0.0,
        "queue_rate_p95": float(queue_rate_p95),
        "queue_delta_mean": float(queue_delta_mean),
        "queue_delta_p95": float(queue_delta_p95),
        # ops
        "relocation_km_total": float(reloc_km_total),
        "charging_cost_eur_total": float(charge_cost_eur_total),
        "charging_energy_kwh_total": float(charge_energy_kwh_total),
        "charge_utilization_avg": float(charge_util_avg),
        # snapshot (debug)
        "score_cfg": asdict(env.score_cfg),
    }


def run_one_episode(
    *,
    cfg_path: str,
    hours: int,
    seed: int,
    scenario: str,
    scenario_params: Dict[str, Any],
    agent: HeuristicAgent,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    env = PortoMicromobilityEnv(cfg_path=cfg_path, episode_hours=hours, seed=seed)
    obs = env.reset()

    apply_scenario(env, scenario=scenario, seed=seed, **(scenario_params or {}))

    done = False
    total_reward = 0.0
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)

    kpis = compute_episode_kpis(env, total_reward=total_reward)
    meta = {"seed": int(seed), "hours": int(hours), "scenario": scenario, "scenario_params": scenario_params}
    return kpis, meta


def _aggregate(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)) if len(vals) else 0.0,
            "max": float(np.max(vals)) if len(vals) else 0.0,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/network_porto10.yaml")
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--out", default="out/heuristic_eval.json")

    # scenarios
    p.add_argument("--scenario", default="baseline", help="baseline | hotspot_od | hetero_lambda | event_heavy")
    p.add_argument("--scenario_params_json", default="", help="JSON dict of scenario params (optional).")

    args = p.parse_args()

    scenario = str(args.scenario).strip()
    scenario_params: Dict[str, Any] = {}
    if args.scenario_params_json.strip():
        scenario_params = json.loads(args.scenario_params_json)
        if not isinstance(scenario_params, dict):
            raise ValueError("--scenario_params_json must be a JSON dict.")

    agent = HeuristicAgent()

    rows: List[Dict[str, Any]] = []
    for k in range(int(args.seeds)):
        seed = int(args.seed0) + k
        kpis, meta = run_one_episode(
            cfg_path=str(args.config),
            hours=int(args.hours),
            seed=seed,
            scenario=scenario,
            scenario_params=scenario_params,
            agent=agent,
        )
        row = {**meta, **kpis}
        rows.append(row)

    report_keys = [
        "J_run",
        "total_reward",
        "reward_plus_sumJ",
        "J_avail_run",
        "J_reloc_run",
        "J_charge_run",
        "J_queue_run",
        "availability_demand_weighted",
        "availability_tick_avg",
        "unmet_rate",
        "avg_wait_min_proxy",
        "queue_total_p95",
        "queue_rate_p95",
        "queue_delta_mean",
        "queue_delta_p95",
        "relocation_km_total",
        "charging_cost_eur_total",
        "charge_utilization_avg",
    ]
    summary = _aggregate(rows, report_keys)

    out = {
        "config": str(args.config),
        "hours": int(args.hours),
        "seed0": int(args.seed0),
        "seeds": int(args.seeds),
        "scenario": scenario,
        "scenario_params": scenario_params,
        "agent": "heuristic_tick_level",
        "report_keys": report_keys,
        "summary": summary,
        "episodes": rows,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # compact console report (matching your eval_policies style)
    s = summary
    print("\n=== Heuristic policy summary (mean ± std over seeds) ===")
    print(
        f"- heuristic: "
        f"J_run={s['J_run']['mean']:.3f}±{s['J_run']['std']:.3f}, "
        f"avail_w={s['availability_demand_weighted']['mean']:.3f}±{s['availability_demand_weighted']['std']:.3f}, "
        f"unmet_rate={s['unmet_rate']['mean']:.3f}±{s['unmet_rate']['std']:.3f}, "
        f"reloc_km={s['relocation_km_total']['mean']:.1f}±{s['relocation_km_total']['std']:.1f}, "
        f"charge€={s['charging_cost_eur_total']['mean']:.2f}±{s['charging_cost_eur_total']['std']:.2f}"
    )
    print("\n--- Objective decomposition (per-tick mean contributions) ---")
    print(
        f"- heuristic: "
        f"J_avail={s['J_avail_run']['mean']:.3f}, "
        f"J_reloc={s['J_reloc_run']['mean']:.3f}, "
        f"J_charge={s['J_charge_run']['mean']:.3f}, "
        f"J_queue={s['J_queue_run']['mean']:.3f}"
    )
    print("\n--- Queue stability (Δqueue) ---")
    print(
        f"- heuristic: "
        f"dq_mean={s['queue_delta_mean']['mean']:.3f}, "
        f"dq_p95={s['queue_delta_p95']['mean']:.3f}"
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()