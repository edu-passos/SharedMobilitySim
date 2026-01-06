"""
eval_policies.py

Evaluation runner + controlled sweeps + scenarios.

Policies supported:
- Planner overrides via PortoMicromobilityEnv(reloc_name_override=..., charge_name_override=...)
- Constant-action: fixed action vector applied every step (env maps it to planner params)
- Params override: merged into env.base_reloc_params/env.base_charge_params after reset()
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.porto_env import PortoMicromobilityEnv


# Scenario application
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
    """Mutate env demand/OD/event state after env.reset() to create scenarios.

    This is intentionally lightweight: it does NOT require changes to Sim.
    """
    scenario = str(scenario).lower().strip()
    if scenario in ("", "baseline", "base"):
        return

    assert env.base_lambda is not None, "env.reset() must be called before apply_scenario()"
    assert env.P is not None
    assert env.events_matrix is not None

    N = int(env.base_lambda.shape[0])

    if scenario == "hotspot_od":
        # Bias every origin's destination distribution towards hotspot_j with prob hotspot_p
        j0 = int(np.clip(hotspot_j, 0, N - 1))
        p_hot = float(np.clip(hotspot_p, 0.0, 0.99))
        P = np.full((N, N), (1.0 - p_hot) / max(N - 1, 1), dtype=float)
        P[:, j0] = p_hot
        if N > 1:
            # ensure row sums exactly 1
            for i in range(N):
                # distribute the remaining mass excluding hotspot index
                rem = 1.0 - P[i, j0]
                P[i, :] = rem / (N - 1)
                P[i, j0] = 1.0 - rem
        env.P = P
        return

    if scenario == "hetero_lambda":
        # Make base_lambda vary across stations but keep mean roughly similar.
        # Deterministic per seed.
        rng = np.random.default_rng(int(seed) + 999)
        # multiplicative factors centered near 1.0
        # strength controls spread; clip to keep reasonable
        f = rng.normal(loc=1.0, scale=float(hetero_strength), size=N)
        f = np.clip(f, 0.3, 2.5)
        # re-normalize to keep mean demand unchanged
        f = f / max(float(np.mean(f)), 1e-12)
        env.base_lambda = env.base_lambda * f
        return

    if scenario == "event_heavy":
        # Scale event multipliers; keep >= 0
        E = env.events_matrix.astype(float)
        # if your events are already multiplicative factors around 1.0,
        # scaling "distance from 1.0" is more sensible than scaling absolute.
        # E_scaled = 1 + event_scale*(E - 1)
        E_scaled = 1.0 + float(event_scale) * (E - 1.0)
        E_scaled = np.clip(E_scaled, 0.0, None)
        env.events_matrix = E_scaled
        return

    raise ValueError(f"Unknown scenario '{scenario}'. Use: baseline, hotspot_od, hetero_lambda, event_heavy.")


# KPIs (aligned with normalized objective in env.score_cfg)
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

    # ---------------- volumes ----------------
    demand_total = int(arr("demand_total").sum())
    served_total = int(arr("served_total").sum())
    served_new_total = int(arr("served_new_total").sum())
    unmet_total = int(arr("unmet").sum())
    backlog_served_total = int(max(0, served_total - served_new_total))

    availability_demand_weighted = 1.0 if demand_total == 0 else (served_new_total / demand_total)
    unmet_rate = 0.0 if demand_total == 0 else (unmet_total / demand_total)
    availability_tick_avg = float(arr("availability").mean())

    # ---------------- queue / wait proxy ----------------
    queue_total = arr("queue_total")
    queue_rate = arr("queue_rate")

    total_queue_time_min = float(queue_total.sum() * dt_min)
    avg_wait_min_proxy = float(total_queue_time_min / max(served_total, 1))

    queue_total_p95 = float(np.percentile(queue_total, 95)) if T else 0.0
    queue_rate_p95 = float(np.percentile(queue_rate, 95)) if T else 0.0

    # Queue stability: delta per tick
    if T > 1:
        dq = queue_total[1:] - queue_total[:-1]
        queue_delta_mean = float(np.mean(dq))
        queue_delta_p95 = float(np.percentile(dq, 95))
    else:
        queue_delta_mean = 0.0
        queue_delta_p95 = 0.0

    # ---------------- ops ----------------
    reloc_km_total = float(arr("reloc_km").sum())
    reloc_units_total = int(arr("reloc_units").sum())
    reloc_edges_total = int(arr("reloc_edges").sum())

    charge_energy_kwh_total = float(arr("charge_energy_kwh").sum())
    charge_cost_eur_total = float(arr("charge_cost_eur").sum())
    charge_util_avg = float(arr("charge_utilization").mean())

    # ---------------- SoC / feasibility proxies ----------------
    soc_mean_vehicles_avg = float(arr("soc_mean_vehicles").mean())
    rentable_ratio_pre_avg = float(
        (arr("x_rentable_total_pre") / np.maximum(arr("x_total_pre"), 1.0)).mean()
    )
    soc_bind_frac_avg = float(arr("soc_bind_frac").mean())

    # Additional mechanism KPIs
    plugged_avg = float(arr("plugged").mean())
    plugged_reserve_avg = float(arr("plugged_reserve").mean())
    rentable_frac_avg = float(arr("rentable_frac").mean())
    soc_station_p10_avg = float(arr("soc_station_p10").mean())

    # ---------------- balance ----------------
    empty_ratio_avg = float(arr("empty_ratio").mean())
    full_ratio_avg = float(arr("full_ratio").mean())
    stock_std_avg = float(arr("stock_std").mean())

    # ---------------- normalized objective recompute (with decomposition) ----------------
    unavailability = 1.0 - arr("availability")
    A0 = max(scfg.A0_unavailability, scfg.eps)
    R0 = max(scfg.R0_reloc_km, scfg.eps)
    C0 = max(scfg.C0_charge_cost_eur, scfg.eps)
    Q0 = max(scfg.Q0_queue_total, scfg.eps)

    J_avail = scfg.w_availability * (unavailability / A0)
    J_reloc = scfg.w_reloc * (arr("reloc_km") / R0)
    J_charge = scfg.w_charge * (arr("charge_cost_eur") / C0)
    J_queue = scfg.w_queue * (arr("queue_total") / Q0)

    J_t = J_avail + J_reloc + J_charge + J_queue
    J_run = float(np.mean(J_t))

    J_avail_run = float(np.mean(J_avail))
    J_reloc_run = float(np.mean(J_reloc))
    J_charge_run = float(np.mean(J_charge))
    J_queue_run = float(np.mean(J_queue))

    reward_plus_sumJ = float(total_reward + float(np.sum(J_t)))

    return {
        "ticks": int(T),
        "total_reward": float(total_reward),
        "J_run": float(J_run),
        "reward_plus_sumJ": float(reward_plus_sumJ),
        # objective decomposition
        "J_avail_run": float(J_avail_run),
        "J_reloc_run": float(J_reloc_run),
        "J_charge_run": float(J_charge_run),
        "J_queue_run": float(J_queue_run),
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
        "reloc_units_total": int(reloc_units_total),
        "reloc_edges_total": int(reloc_edges_total),
        "charging_energy_kwh_total": float(charge_energy_kwh_total),
        "charging_cost_eur_total": float(charge_cost_eur_total),
        "charge_utilization_avg": float(charge_util_avg),
        # SoC / feasibility
        "soc_mean_vehicles_avg": float(soc_mean_vehicles_avg),
        "rentable_ratio_pre_avg": float(rentable_ratio_pre_avg),
        "soc_bind_frac_avg": float(soc_bind_frac_avg),
        # charging/rentability mechanisms
        "plugged_avg": float(plugged_avg),
        "plugged_reserve_avg": float(plugged_reserve_avg),
        "rentable_frac_avg": float(rentable_frac_avg),
        "soc_station_p10_avg": float(soc_station_p10_avg),
        # balance
        "empty_ratio_avg": float(empty_ratio_avg),
        "full_ratio_avg": float(full_ratio_avg),
        "stock_std_avg": float(stock_std_avg),
        # volumes
        "demand_total": int(demand_total),
        "served_total": int(served_total),
        "served_new_total": int(served_new_total),
        "backlog_served_total": int(backlog_served_total),
        # score config snapshot
        "score_cfg": asdict(env.score_cfg),
    }


def _run_episode(
    *,
    cfg_path: str,
    hours: int,
    seed: int,
    action: Optional[np.ndarray],
    reloc_name: Optional[str],
    charge_name: Optional[str],
    params_override: Optional[Dict[str, Dict[str, Any]]] = None,
    scenario: str = "baseline",
    scenario_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    env = PortoMicromobilityEnv(
        cfg_path=cfg_path,
        episode_hours=hours,
        seed=seed,
        reloc_name_override=reloc_name,
        charge_name_override=charge_name,
    )
    _ = env.reset()

    # Apply scenario modifications after reset (so it is deterministic per seed)
    sp = scenario_params or {}
    apply_scenario(env, scenario=scenario, seed=seed, **sp)

    # Merge per-policy parameter overrides (minimal implementation)
    if params_override:
        r_over = params_override.get("reloc", {}) or {}
        c_over = params_override.get("charge", {}) or {}
        # These are used by env._action_to_params via **self.base_*_params
        env.base_reloc_params.update(r_over)
        env.base_charge_params.update(c_over)

    done = False
    total_reward = 0.0

    a = None
    if action is not None:
        a = np.asarray(action, dtype=float).reshape(4,)
        a = np.clip(a, 0.0, 1.0)

    while not done:
        # If action=None, we still need to pass something valid
        act = a if a is not None else np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
        _, reward, done, _info = env.step(act)
        total_reward += float(reward)

    kpis = compute_episode_kpis(env, total_reward=total_reward)
    meta = {
        "seed": int(seed),
        "hours": int(hours),
        "action": a.tolist() if a is not None else None,
        "reloc_planner": reloc_name,
        "charge_planner": charge_name,
        "scenario": scenario,
        "params_override": params_override,
    }
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
    p.add_argument("--out", default="out/eval_policies.json")

    # Scenario control
    p.add_argument("--scenario", default="baseline", help="baseline | hotspot_od | hetero_lambda | event_heavy")
    p.add_argument("--scenario_params_json", default="", help="JSON dict of scenario params (optional).")

    # Constant-action policies
    p.add_argument(
        "--actions_json",
        default="",
        help="JSON list of actions, e.g. '[[0,0,0,0],[0.5,0.5,0.5,0.5],[1,1,1,1]]'.",
    )
    p.add_argument(
        "--no_actions",
        action="store_true",
        help="If set, evaluate planner pairs with action=None (env uses default [0.5,0.5,0.5,0.5]).",
    )

    # Planner pairs
    p.add_argument(
        "--planner_pairs_json",
        default="",
        help=(
            "JSON list of [reloc, charge] pairs, "
            "e.g. '[[\"noop\",\"noop\"],[\"budgeted\",\"greedy\"]]'. "
            "If empty, defaults to a baseline set."
        ),
    )

    # Sweep mode (controlled tradeoff grid)
    p.add_argument(
        "--sweep",
        action="store_true",
        help="If set, ignore planner_pairs/actions and run a grid sweep over km budgets and charge fracs.",
    )
    p.add_argument(
        "--sweep_reloc_planner",
        default="budgeted",
        help="Relocation planner name used for sweep (default: budgeted).",
    )
    p.add_argument(
        "--sweep_charge_planner",
        default="greedy",
        help="Charging planner name used for sweep (default: greedy).",
    )
    p.add_argument(
        "--reloc_km_budgets_json",
        default="[0,5,10,20,40]",
        help="JSON list of km budgets for relocation sweep, e.g. '[0,5,10,20,40]'.",
    )
    p.add_argument(
        "--charge_budget_fracs_json",
        default="[0.0,0.25,0.5,0.75,1.0]",
        help="JSON list of charging budget fracs, e.g. '[0.0,0.25,0.5,0.75,1.0]'.",
    )

    args = p.parse_args()

    cfg_path = str(args.config)
    hours = int(args.hours)

    scenario = str(args.scenario).strip()
    scenario_params: Dict[str, Any] = {}
    if args.scenario_params_json.strip():
        scenario_params = json.loads(args.scenario_params_json)
        if not isinstance(scenario_params, dict):
            raise ValueError("--scenario_params_json must be a JSON dict.")

    # ---- actions ----
    if args.no_actions:
        actions: List[Optional[List[float]]] = [None]
    else:
        if args.actions_json.strip():
            actions = json.loads(args.actions_json)
            if not isinstance(actions, list):
                raise ValueError("--actions_json must be a JSON list of 4D action vectors.")
        else:
            actions = [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0, 1.0],
            ]

    policies: List[Dict[str, Any]] = []

    if args.sweep:
        # Controlled grid sweep over budgets.
        reloc_planner = str(args.sweep_reloc_planner)
        charge_planner = str(args.sweep_charge_planner)

        km_budgets = json.loads(args.reloc_km_budgets_json)
        charge_fracs = json.loads(args.charge_budget_fracs_json)
        if not isinstance(km_budgets, list) or not all(isinstance(x, (int, float)) for x in km_budgets):
            raise ValueError("--reloc_km_budgets_json must be a JSON list of numbers.")
        if not isinstance(charge_fracs, list) or not all(isinstance(x, (int, float)) for x in charge_fracs):
            raise ValueError("--charge_budget_fracs_json must be a JSON list of numbers.")

        for km in km_budgets:
            for cf in charge_fracs:
                # Important: for charge sweep to work, env must respect "charge_budget_frac_override"
                # (see patch below).
                params_override = {
                    "reloc": {"km_budget": float(km)},
                    "charge": {"charge_budget_frac_override": float(cf)},
                }
                policies.append(
                    {
                        "name": f"sweep__km{float(km):g}__c{float(cf):g}",
                        "reloc": reloc_planner,
                        "charge": charge_planner,
                        "action": None,  # sweep overrides planner params; action becomes irrelevant
                        "params_override": params_override,
                    }
                )
    else:
        # ---- planner pairs ----
        if args.planner_pairs_json.strip():
            pairs = json.loads(args.planner_pairs_json)
            if not isinstance(pairs, list) or not all(isinstance(x, list) and len(x) == 2 for x in pairs):
                raise ValueError("--planner_pairs_json must be a JSON list of [reloc, charge] pairs.")
        else:
            pairs = [
                ["noop", "noop"],
                ["noop", "greedy"],
                ["greedy", "noop"],
                ["greedy", "greedy"],
            ]

        for reloc_name, charge_name in pairs:
            for i, a in enumerate(actions):
                if a is None:
                    pol_name = f"{reloc_name}_{charge_name}__action_default"
                else:
                    pol_name = f"{reloc_name}_{charge_name}__action_{i}"
                policies.append(
                    {
                        "name": pol_name,
                        "reloc": str(reloc_name),
                        "charge": str(charge_name),
                        "action": a,
                        "params_override": None,
                    }
                )

    # ---- run ----
    all_rows: List[Dict[str, Any]] = []
    per_policy: Dict[str, List[Dict[str, Any]]] = {pol["name"]: [] for pol in policies}

    for pol in policies:
        name = pol["name"]
        reloc_name = pol["reloc"]
        charge_name = pol["charge"]
        params_override = pol.get("params_override", None)

        action_np: Optional[np.ndarray] = None
        if pol["action"] is not None:
            a = np.asarray(pol["action"], dtype=float)
            if a.shape != (4,):
                raise ValueError(f"Policy {name}: action must be shape (4,), got {a.shape}")
            action_np = a

        for k in range(int(args.seeds)):
            seed = int(args.seed0) + k
            kpis, meta = _run_episode(
                cfg_path=cfg_path,
                hours=hours,
                seed=seed,
                action=action_np,
                reloc_name=reloc_name,
                charge_name=charge_name,
                params_override=params_override,
                scenario=scenario,
                scenario_params=scenario_params,
            )
            row = {"policy": name, **meta, **kpis}
            all_rows.append(row)
            per_policy[name].append(row)

    # Aggregate policy summaries
    report_keys = [
        # overall
        "J_run",
        "total_reward",
        "reward_plus_sumJ",
        # decomposition
        "J_avail_run",
        "J_reloc_run",
        "J_charge_run",
        "J_queue_run",
        # service
        "availability_demand_weighted",
        "availability_tick_avg",
        "unmet_rate",
        "avg_wait_min_proxy",
        # queue
        "queue_total_p95",
        "queue_rate_p95",
        "queue_delta_mean",
        "queue_delta_p95",
        # ops
        "relocation_km_total",
        "charging_cost_eur_total",
        "charge_utilization_avg",
        # mechanisms
        "rentable_ratio_pre_avg",
        "soc_bind_frac_avg",
        "plugged_avg",
        "plugged_reserve_avg",
        "rentable_frac_avg",
        "soc_station_p10_avg",
        # SoC & balance
        "soc_mean_vehicles_avg",
        "empty_ratio_avg",
        "full_ratio_avg",
        "stock_std_avg",
    ]

    summaries = {name: _aggregate(rows, report_keys) for name, rows in per_policy.items()}

    out = {
        "config": cfg_path,
        "hours": hours,
        "seed0": int(args.seed0),
        "seeds": int(args.seeds),
        "scenario": scenario,
        "scenario_params": scenario_params,
        "policies": policies,
        "report_keys": report_keys,
        "summaries": summaries,
        "episodes": all_rows,
    }

    # Write output
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Console summary (compact)
    print("\n=== Policy summary (mean ± std over seeds) ===")
    for name in summaries:
        s = summaries[name]
        print(
            f"- {name}: "
            f"J_run={s['J_run']['mean']:.3f}±{s['J_run']['std']:.3f}, "
            f"avail_w={s['availability_demand_weighted']['mean']:.3f}±{s['availability_demand_weighted']['std']:.3f}, "
            f"unmet_rate={s['unmet_rate']['mean']:.3f}±{s['unmet_rate']['std']:.3f}, "
            f"reloc_km={s['relocation_km_total']['mean']:.1f}±{s['relocation_km_total']['std']:.1f}, "
            f"charge€={s['charging_cost_eur_total']['mean']:.2f}±{s['charging_cost_eur_total']['std']:.2f}"
        )

    print("\n--- Objective decomposition (per-tick mean contributions) ---")
    for name in summaries:
        s = summaries[name]
        print(
            f"- {name}: "
            f"J_avail={s['J_avail_run']['mean']:.3f}, "
            f"J_reloc={s['J_reloc_run']['mean']:.3f}, "
            f"J_charge={s['J_charge_run']['mean']:.3f}, "
            f"J_queue={s['J_queue_run']['mean']:.3f}"
        )

    print("\n--- Queue stability (Δqueue) ---")
    for name in summaries:
        s = summaries[name]
        print(
            f"- {name}: "
            f"dq_mean={s['queue_delta_mean']['mean']:.3f}, "
            f"dq_p95={s['queue_delta_p95']['mean']:.3f}"
        )

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
