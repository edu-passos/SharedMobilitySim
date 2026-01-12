"""heuristic.py.

Evaluate a real-time (tick-by-tick) adaptive heuristic agent on PortoMicromobilityEnv:

    J_t = wA * (unavailability / A0)
        + wR * (reloc_km / R0)
        + wC * (charge_cost_eur / C0)
        + wQ * (queue_total / Q0)

This script:
- Runs multiple seeds
- Optionally applies scenarios (baseline / hotspot_od / hetero_lambda / event_heavy)
- Outputs a per-seed episode table + aggregate mean±std
- Saves JSON to --out (same schema style as eval_policies.py: episodes + summary)
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from envs.porto_env import PortoMicromobilityEnv
from sim.kpis import compute_episode_kpis

# Force planners here (keeps this heuristic evaluation independent of budgeted/slack planners)
RELOC_PLANNER_OVERRIDE = "greedy"
CHARGE_PLANNER_OVERRIDE = "greedy"


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


@dataclass
class HeuristicAgent:
    """Tick-level heuristic with feedback control.

    - uses queue_rate for demand pressure
    - throttles relocation using EWMA of last reloc_km
    - charges based on SoC tail (p10) rather than mean
    """

    # EWMA smoothing (higher -> faster response)
    alpha: float = 0.15

    # Relocation targets (interpreted in km/tick, relative)
    reloc_km_target_base: float = 4.0
    reloc_km_target_event: float = 20.0

    # EWMA state
    ewma_reloc_km: float = 0.0
    ewma_queue_rate: float = 0.0
    initialized: bool = False

    def reset(self) -> None:
        self.ewma_reloc_km = 0.0
        self.ewma_queue_rate = 0.0
        self.initialized = False

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        fill = np.asarray(obs["fill_ratio"], dtype=float)  # (N,)
        soc = np.asarray(obs["soc"], dtype=float)  # (N,)
        waiting = np.asarray(obs["waiting"], dtype=float)  # (N,)
        tod = np.asarray(obs["time_of_day"], dtype=float)  # (2,)

        # optional extra signals (safe indexing)
        event_stats = np.asarray(obs.get("event_stats", np.array([1.0, 1.0])), dtype=float).reshape(-1)
        event_max = float(event_stats[1]) if event_stats.size >= 2 else 1.0

        last_kpi = np.asarray(obs.get("last_kpi", np.zeros(5)), dtype=float).reshape(-1)
        last_reloc_km = float(last_kpi[0]) if last_kpi.size >= 1 else 0.0
        last_queue_rate = float(last_kpi[4]) if last_kpi.size >= 5 else 0.0

        # Update EWMAs
        if not self.initialized:
            self.ewma_reloc_km = last_reloc_km
            self.ewma_queue_rate = last_queue_rate
            self.initialized = True
        else:
            a = float(self.alpha)
            self.ewma_reloc_km = (1 - a) * self.ewma_reloc_km + a * last_reloc_km
            self.ewma_queue_rate = (1 - a) * self.ewma_queue_rate + a * last_queue_rate

        # Decode hour from [sin, cos]
        sin_h, cos_h = float(tod[0]), float(tod[1])
        angle = float(np.arctan2(sin_h, cos_h))
        if angle < 0:
            angle += 2 * np.pi
        hour = 24.0 * angle / (2 * np.pi)

        is_morning_peak = 7 <= hour <= 10
        is_evening_peak = 17 <= hour <= 20
        is_night = (0 <= hour <= 5) or (hour >= 23)

        # Core state summaries
        empty_frac = float(np.mean(fill < 0.1))
        full_frac = float(np.mean(fill > 0.9))

        # waiting distribution (more robust than avg)
        high_wait_frac = float(np.mean(waiting > 5))
        very_high_wait_frac = float(np.mean(waiting > 10))

        soc_p10 = float(np.percentile(soc, 10))

        # Pressure based on queue_rate EWMA (faster) + local waiting tail
        # Scale queue_rate into [0,1] using a soft normalization; tune denominator if needed.
        qrate_scale = float(np.clip(self.ewma_queue_rate / 3.0, 0.0, 1.0))
        pressure = float(np.clip(0.45 * qrate_scale + 0.35 * high_wait_frac + 0.20 * empty_frac, 0.0, 1.0))

        # Relocation throttle
        # If event_max is high, allow more relocation.
        # event_max ~ 1.0 normally, >1 during events.
        event_strength = float(np.clip((event_max - 1.0) / 1.0, 0.0, 1.0))  # map ~[1..2] -> [0..1]
        reloc_target = (1 - event_strength) * self.reloc_km_target_base + event_strength * self.reloc_km_target_event

        # throttle in [0,1]: 0 = no throttle, 1 = strong throttle
        # If ewma_reloc_km exceeds target, throttle rises.
        throttle = float(np.clip((self.ewma_reloc_km - reloc_target) / max(reloc_target, 1e-6), 0.0, 1.0))

        # -----------------------------
        # Map to action components [a0,a1,a2,a3] in [0,1]
        # Recall env mapping:
        # low = 0.1 + 0.2*a0 (higher -> more needy -> more reloc)
        # high = 0.6 + 0.3*a1 (higher -> fewer donors -> less reloc)
        # target = 0.4 + 0.4*a2
        # charge_budget = 0.05 + 0.35*a3
        # -----------------------------

        # a0 (low threshold) - raise with pressure, but *decrease* under throttle
        base_a0 = 0.35 if (is_morning_peak or is_evening_peak) else 0.18
        a0 = base_a0 + 0.75 * pressure - 0.70 * throttle
        # extra safety: if lots of full stations, don't mark too many as needy (prevents churn)
        a0 -= 0.20 * full_frac
        a0 = float(np.clip(a0, 0.0, 1.0))

        # a1 (high threshold) - higher => stricter donors => less reloc; increase under throttle
        base_a1 = 0.20 if (is_morning_peak or is_evening_peak) else (0.70 if is_night else 0.45)
        a1 = base_a1 + 0.70 * throttle - 0.30 * pressure
        # if system is very full, relax donors slightly to alleviate overflow
        a1 -= 0.25 * full_frac
        a1 = float(np.clip(a1, 0.0, 1.0))

        # a2 (target fill) - increase when pressure is high; decrease when throttled (reduces reloc need)
        a2 = 0.25 + 0.85 * pressure - 0.25 * throttle
        if not (is_morning_peak or is_evening_peak):
            a2 -= 0.10
        a2 = float(np.clip(a2, 0.0, 1.0))

        # a3 (charging) - depend on SoC tail + pressure; boost at night
        soc_lack_tail = float(np.clip(0.35 - soc_p10, 0.0, 0.35) / 0.35)  # soc_p10 below 0.35 -> higher
        drive = 0.55 * soc_lack_tail + 0.45 * float(np.clip(0.5 * pressure + 0.5 * very_high_wait_frac, 0.0, 1.0))
        if is_night:
            drive = max(drive, 0.75)
        a3 = float(np.clip(drive, 0.0, 1.0))

        return np.array([a0, a1, a2, a3], dtype=float)


def run_one_episode(
    *,
    cfg_path: str,
    hours: int,
    seed: int,
    scenario: str,
    scenario_params: dict[str, Any],
    agent: HeuristicAgent,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env = PortoMicromobilityEnv(
        cfg_path=cfg_path,
        episode_hours=hours,
        seed=seed,
        reloc_name_override=RELOC_PLANNER_OVERRIDE,
        charge_name_override=CHARGE_PLANNER_OVERRIDE,
    )
    obs = env.reset()
    agent.reset()

    apply_scenario(env, scenario=scenario, seed=seed, **(scenario_params or {}))

    done = False
    total_reward = 0.0
    while not done:
        action = agent.act(obs)
        obs, reward, done, _info = env.step(action)
        total_reward += float(reward)

    kpis = compute_episode_kpis(env, total_reward=total_reward)
    meta = {"seed": int(seed), "hours": int(hours), "scenario": scenario, "scenario_params": scenario_params}
    return kpis, meta


def _aggregate(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
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
    scenario_params: dict[str, Any] = {}
    if args.scenario_params_json.strip():
        scenario_params = json.loads(args.scenario_params_json)
        if not isinstance(scenario_params, dict):
            raise ValueError("--scenario_params_json must be a JSON dict.")

    agent = HeuristicAgent()

    rows: list[dict[str, Any]] = []
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
        "dq_mean",
        "dq_p95",
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
        "planners_forced": {"reloc": RELOC_PLANNER_OVERRIDE, "charge": CHARGE_PLANNER_OVERRIDE},
        "report_keys": report_keys,
        "summary": summary,
        "episodes": rows,
    }

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # compact console report
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
    print(f"- heuristic: dq_mean={s['dq_mean']['mean']:.3f}, dq_p95={s['dq_p95']['mean']:.3f}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
