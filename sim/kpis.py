from dataclasses import asdict
from typing import Any

import numpy as np


def compute_episode_kpis(env, *, total_reward: float) -> dict[str, Any]:
    """Compute a comprehensive episode KPI panel from env.sim.logs."""
    sim = getattr(env, "sim", None)
    if sim is None or not getattr(sim, "logs", None):
        return {}

    logs: list[dict[str, Any]] = sim.logs
    T = len(logs)
    dt_min = int(sim.cfg.dt_min)
    scfg = env.score_cfg

    def arr(key: str, default: float = 0.0) -> np.ndarray:
        return np.array([r.get(key, default) for r in logs], dtype=float)

    def pct(x: np.ndarray, q: float) -> float:
        return float(np.percentile(x, q)) if x.size else 0.0

    def safe_mean(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size else 0.0

    def safe_max(x: np.ndarray) -> float:
        return float(np.max(x)) if x.size else 0.0

    # Core series
    availability = arr("availability", 0.0)
    unavailability = 1.0 - availability

    queue_total = arr("queue_total", 0.0)

    queue_rate = arr("queue_rate", 0.0)

    demand_total = arr("demand_total", 0.0)
    served_total = arr("served_total", 0.0)
    served_new_total = arr("served_new_total", 0.0)
    unmet = arr("unmet", 0.0)

    reloc_km = arr("reloc_km", 0.0)
    reloc_units = arr("reloc_units", 0.0)
    reloc_edges = arr("reloc_edges", 0.0)

    charge_energy = arr("charge_energy_kwh", 0.0)
    charge_cost = arr("charge_cost_eur", 0.0)
    plugged = arr("plugged", 0.0)
    plugged_reserve = arr("plugged_reserve", 0.0)
    charge_util = arr("charge_utilization", 0.0)

    empty_ratio = arr("empty_ratio", 0.0)
    full_ratio = arr("full_ratio", 0.0)
    stock_std = arr("stock_std", 0.0)
    fill_p10 = arr("fill_p10", 0.0)
    fill_p90 = arr("fill_p90", 0.0)

    soc_mean = arr("soc_mean", 0.0)
    soc_mean_vehicles = arr("soc_mean_vehicles", 0.0)
    soc_station_min = arr("soc_station_min", 0.0)
    soc_station_p10 = arr("soc_station_p10", 0.0)

    rentable_frac = arr("rentable_frac", 0.0)
    soc_bind_frac = arr("soc_bind_frac", 0.0)

    overflow_rerouted = arr("overflow_rerouted", 0.0)
    overflow_dropped = arr("overflow_dropped", 0.0)
    overflow_extra_min = arr("overflow_extra_min", 0.0)

    # Aggregates: volumes
    D = float(demand_total.sum())
    Snew = float(served_new_total.sum())
    Sunits = float(served_total.sum())
    U = float(unmet.sum())

    availability_demand_weighted = 1.0 if D <= 0 else (Snew / D)
    unmet_rate = 0.0 if D <= 0 else (U / D)

    # Wait proxy: integral of queue_total * dt / served
    total_queue_time_min = float(queue_total.sum() * dt_min)
    avg_wait_min_proxy = float(total_queue_time_min / max(Sunits, 1.0))

    # Queue stability: Δqueue_total
    if T > 1:
        dq = queue_total[1:] - queue_total[:-1]
        dq_mean = float(np.mean(dq))
        dq_p95 = float(np.percentile(dq, 95))
        dq_p99 = float(np.percentile(dq, 99))
        dq_max = float(np.max(dq))
    else:
        dq_mean = dq_p95 = dq_p99 = dq_max = 0.0

    # Normalized objective decomposition (must match env reward)
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

    # sanity: reward is -sum(J_t) (or -mean depending on wrapper)
    # env returns per-tick reward; total_reward is sum(reward_t).
    # Therefore: total_reward + sum(J_t) should be ~0
    reward_plus_sumJ = float(total_reward + float(np.sum(J_t)))

    # Operational efficiency ratios
    reloc_km_total = float(reloc_km.sum())
    charge_cost_total = float(charge_cost.sum())
    charge_energy_total = float(charge_energy.sum())

    km_per_served = float(reloc_km_total / max(Sunits, 1.0))
    eur_per_served = float(charge_cost_total / max(Sunits, 1.0))
    kwh_per_served = float(charge_energy_total / max(Sunits, 1.0))

    # SLA-style thresholds (choose once; keep fixed for report)
    Q10, Q20 = 10.0, 20.0
    frac_ticks_queue_gt_10 = float(np.mean(queue_total > Q10)) if T else 0.0
    frac_ticks_queue_gt_20 = float(np.mean(queue_total > Q20)) if T else 0.0

    SOC20 = 0.20
    frac_ticks_soc_p10_lt_0_2 = float(np.mean(soc_station_p10 < SOC20)) if T else 0.0

    # Pack output
    return {
        "ticks": int(T),
        "dt_min": int(dt_min),
        # reward + objective
        "total_reward": float(total_reward),
        "J_run": float(J_run),
        "reward_plus_sumJ": float(reward_plus_sumJ),
        "J_avail_run": float(np.mean(J_avail_t)) if T else 0.0,
        "J_reloc_run": float(np.mean(J_reloc_t)) if T else 0.0,
        "J_charge_run": float(np.mean(J_charge_t)) if T else 0.0,
        "J_queue_run": float(np.mean(J_queue_t)) if T else 0.0,
        # service
        "demand_total": int(D),
        "served_total": int(Sunits),
        "served_new_total": int(Snew),
        "unmet_total": int(U),
        "availability_tick_avg": safe_mean(availability),
        "availability_demand_weighted": float(availability_demand_weighted),
        "unmet_rate": float(unmet_rate),
        "avg_wait_min_proxy": float(avg_wait_min_proxy),
        # queue levels + tails
        "queue_total_avg": safe_mean(queue_total),
        "queue_total_p95": pct(queue_total, 95),
        "queue_total_p99": pct(queue_total, 99),
        "queue_total_max": safe_max(queue_total),
        "queue_rate_avg": safe_mean(queue_rate),
        "queue_rate_p95": pct(queue_rate, 95),
        "queue_rate_p99": pct(queue_rate, 99),
        "queue_rate_max": safe_max(queue_rate),
        # SLA time-above-threshold
        "frac_ticks_queue_gt_10": float(frac_ticks_queue_gt_10),
        "frac_ticks_queue_gt_20": float(frac_ticks_queue_gt_20),
        # queue stability
        "dq_mean": float(dq_mean),
        "dq_p95": float(dq_p95),
        "dq_p99": float(dq_p99),
        "dq_max": float(dq_max),
        # operations totals
        "relocation_km_total": float(reloc_km_total),
        "reloc_units_total": int(reloc_units.sum()),
        "reloc_edges_total": int(reloc_edges.sum()),
        "charging_energy_kwh_total": float(charge_energy_total),
        "charging_cost_eur_total": float(charge_cost_total),
        "plugged_total": int(plugged.sum()),
        "plugged_reserve_total": int(plugged_reserve.sum()),
        "charge_utilization_avg": safe_mean(charge_util),
        # operations burstiness (per-tick tails)
        "reloc_km_p95": pct(reloc_km, 95),
        "reloc_km_p99": pct(reloc_km, 99),
        "reloc_units_p95": pct(reloc_units, 95),
        "reloc_units_p99": pct(reloc_units, 99),
        # efficiency ratios
        "km_per_served": float(km_per_served),
        "eur_per_served": float(eur_per_served),
        "kwh_per_served": float(kwh_per_served),
        # system state distribution proxies
        "empty_ratio_avg": safe_mean(empty_ratio),
        "empty_ratio_p95": pct(empty_ratio, 95),
        "empty_ratio_max": safe_max(empty_ratio),
        "full_ratio_avg": safe_mean(full_ratio),
        "full_ratio_p95": pct(full_ratio, 95),
        "full_ratio_max": safe_max(full_ratio),
        "stock_std_avg": safe_mean(stock_std),
        "stock_std_p95": pct(stock_std, 95),
        "fill_p10_avg": safe_mean(fill_p10),
        "fill_p90_avg": safe_mean(fill_p90),
        "fill_spread_avg": safe_mean(fill_p90 - fill_p10),
        # energy / SoC health
        "soc_mean_avg": safe_mean(soc_mean),
        "soc_mean_vehicles_avg": safe_mean(soc_mean_vehicles),
        "soc_station_min_avg": safe_mean(soc_station_min),
        "soc_station_min_p05": pct(soc_station_min, 5),
        "soc_station_p10_avg": safe_mean(soc_station_p10),
        "frac_ticks_soc_p10_lt_0_2": float(frac_ticks_soc_p10_lt_0_2),
        # rentability primitives (great for discussion)
        "rentable_frac_avg": safe_mean(rentable_frac),
        "soc_bind_frac_avg": safe_mean(soc_bind_frac),
        # overflow
        "overflow_rerouted_total": int(overflow_rerouted.sum()),
        "overflow_dropped_total": int(overflow_dropped.sum()),
        "overflow_extra_min_total": float(overflow_extra_min.sum()),
        # debug snapshot (optional but handy)
        "score_cfg": asdict(env.score_cfg),
    }
