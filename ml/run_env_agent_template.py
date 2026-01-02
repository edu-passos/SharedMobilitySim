import argparse
import json

import numpy as np

from envs.porto_env import PortoMicromobilityEnv, ScoreWeights


class HeuristicAgent:
    """Heuristic agent for PortoMicromobilityEnv.

    It looks at:
        - fill_ratio (how full stations are)
        - soc (battery)
        - waiting (customers in queue)
        - time_of_day (sin/cos)
    and outputs 4 numbers in [0,1] that control relocation & charging thresholds.
    """

    def __init__(self, action_dim: int = 4) -> None:
        self.action_dim = action_dim

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:

        return np.array([0.5, 0.5, 0.5, 1.0], dtype=float)


def compute_episode_kpis(env: PortoMicromobilityEnv, total_reward: float) -> dict[str, float]:
    """Aggregate KPIs from env.sim.logs for one episode.

    IMPORTANT:
    - Compute J_sum exactly from logs using the same weights as the env reward.
    - Check reward consistency: total_reward should be approximately -J_sum if reward=-J_t.
    """
    sim = env.sim
    if sim is None or not sim.logs:
        return {}

    logs = sim.logs
    T = len(logs)
    dt_min = sim.cfg.dt_min
    w = env.score_weights

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

    # tick-avg of per-tick availability (not the same as demand-weighted)
    availability_tick_avg = float(arr("availability").mean())

    # ---------------- queues / wait proxies ----------------
    queue_total = arr("queue_total")
    queue_rate = arr("queue_rate")

    # Little’s-law proxy: avg minutes in queue per served request
    total_queue_time_min = float(queue_total.sum() * dt_min)
    avg_wait_min_proxy = float(total_queue_time_min / max(served_total, 1))

    # a "tail" proxy: p95 queue_total converted to minutes at dt
    queue_total_p95 = float(np.percentile(queue_total, 95)) if T else 0.0
    queue_rate_p95 = float(np.percentile(queue_rate, 95)) if T else 0.0

    # ---------------- relocation ----------------
    reloc_km_total = float(arr("reloc_km").sum())
    reloc_units_total = int(arr("reloc_units").sum())
    reloc_edges_total = int(arr("reloc_edges").sum())

    reloc_km_per_unit = float(reloc_km_total / max(reloc_units_total, 1))
    reloc_units_per_edge = float(reloc_units_total / max(reloc_edges_total, 1))

    # ---------------- charging (reserve vs executed) ----------------
    plugged_reserve = arr("plugged_reserve")
    plugged_exec = arr("plugged")

    plugged_reserve_total = int(plugged_reserve.sum())
    plugged_exec_total = int(plugged_exec.sum())

    plugged_reserve_avg = float(plugged_reserve.mean())
    plugged_exec_avg = float(plugged_exec.mean())

    # vehicles reserved from rentals but not actually charged (diagnostic)
    plugged_slack_total = int(max(0, plugged_reserve_total - plugged_exec_total))
    plugged_slack_avg = float(max(0.0, plugged_reserve_avg - plugged_exec_avg))

    charge_energy_kwh_total = float(arr("charge_energy_kwh").sum())
    charge_cost_eur_total = float(arr("charge_cost_eur").sum())
    charge_util_avg = float(arr("charge_utilization").mean())

    # normalized efficiency
    kwh_per_served = float(charge_energy_kwh_total / max(served_total, 1))
    eur_per_served = float(charge_cost_eur_total / max(served_total, 1))

    # ---------------- SoC / feasibility ----------------
    soc_mean_avg = float(arr("soc_mean").mean())
    soc_mean_vehicles_avg = float(arr("soc_mean_vehicles").mean())

    # tail SoC stats as logged (already post-step snapshot)
    soc_station_min = arr("soc_station_min")
    soc_station_p10 = arr("soc_station_p10")

    # across-ticks summary of those tails
    soc_station_min_p10_over_time = float(np.percentile(soc_station_min, 10)) if T else 0.0
    soc_station_p10_avg = float(soc_station_p10.mean())

    # rentability constraints
    x_total_pre = arr("x_total_pre")
    x_rentable_total_pre = arr("x_rentable_total_pre")

    x_total_pre_avg = float(x_total_pre.mean())
    x_rentable_total_pre_avg = float(x_rentable_total_pre.mean())
    rentable_ratio_pre_avg = float((x_rentable_total_pre / np.maximum(x_total_pre, 1.0)).mean())

    rentable_frac_avg = float(arr("rentable_frac").mean())
    soc_bind_frac_avg = float(arr("soc_bind_frac").mean())

    # ---------------- balance / distribution ----------------
    empty_ratio_avg = float(arr("empty_ratio").mean())
    full_ratio_avg = float(arr("full_ratio").mean())
    stock_std_avg = float(arr("stock_std").mean())

    fill_p10 = arr("fill_p10")
    fill_p90 = arr("fill_p90")
    fill_p10_avg = float(fill_p10.mean())
    fill_p90_avg = float(fill_p90.mean())
    fill_spread_avg = float((fill_p90 - fill_p10).mean())

    # ---------------- overflow ----------------
    overflow_rerouted_total = int(arr("overflow_rerouted").sum())
    overflow_extra_min_total = float(arr("overflow_extra_min").sum())
    # NOTE: overflow_dropped is NOT in your core log schema.
    overflow_dropped_total = 0

    # ---------------- exact J recompute ----------------
    # uses exactly the same components as env._compute_reward
    J_sum = 0.0
    for r in logs:
        availability_t = float(r.get("availability", 0.0))
        reloc_km_t = float(r.get("reloc_km", 0.0))
        charge_cost_t = float(r.get("charge_cost_eur", 0.0))
        queue_rate_t = float(r.get("queue_rate", 0.0))

        J_t = (
            w.alpha_unavailability * (1.0 - availability_t)
            + w.beta_reloc_km * reloc_km_t
            + w.gamma_energy_cost * charge_cost_t
            + w.delta_queue * queue_rate_t
        )
        J_sum += J_t

    J_run = J_sum / max(T, 1)
    reward_plus_J_sum = float(total_reward + J_sum)

    return {
        # volumes
        "demand_total": demand_total,
        "served_total": served_total,
        "served_new_total": served_new_total,
        "backlog_served_total": backlog_served_total,
        "unmet_total": unmet_total,

        # service quality
        "availability_tick_avg": round(availability_tick_avg, 3),
        "availability_demand_weighted": round(float(availability_demand_weighted), 3),
        "unmet_rate": round(float(unmet_rate), 3),
        "avg_wait_min_proxy": round(float(avg_wait_min_proxy), 2),

        # queue stats
        "queue_total_max": int(queue_total.max()) if T else 0,
        "queue_total_avg": round(float(queue_total.mean()), 2),
        "queue_total_p95": round(queue_total_p95, 2),
        "queue_rate_avg": round(float(queue_rate.mean()), 3),
        "queue_rate_p95": round(queue_rate_p95, 3),

        # relocation effort
        "relocation_km_total": round(reloc_km_total, 2),
        "reloc_units_total": reloc_units_total,
        "reloc_edges_total": reloc_edges_total,
        "reloc_km_per_unit": round(reloc_km_per_unit, 3),
        "reloc_units_per_edge": round(reloc_units_per_edge, 3),

        # charging effort / efficiency
        "charging_energy_kwh_total": round(charge_energy_kwh_total, 2),
        "charging_cost_eur_total": round(charge_cost_eur_total, 2),
        "charge_utilization_avg": round(charge_util_avg, 3),

        "plugged_reserve_total": plugged_reserve_total,
        "plugged_reserve_avg": round(plugged_reserve_avg, 2),
        "plugged_exec_total": plugged_exec_total,
        "plugged_exec_avg": round(plugged_exec_avg, 2),
        "plugged_slack_total": plugged_slack_total,
        "plugged_slack_avg": round(plugged_slack_avg, 2),

        "kwh_per_served": round(kwh_per_served, 4),
        "eur_per_served": round(eur_per_served, 4),

        # SoC / feasibility
        "soc_mean_avg": round(soc_mean_avg, 3),
        "soc_mean_vehicles_avg": round(soc_mean_vehicles_avg, 3),
        "soc_station_min_p10_over_time": round(soc_station_min_p10_over_time, 3),
        "soc_station_p10_avg": round(soc_station_p10_avg, 3),

        "x_total_pre_avg": round(x_total_pre_avg, 2),
        "x_rentable_total_pre_avg": round(x_rentable_total_pre_avg, 2),
        "rentable_ratio_pre_avg": round(rentable_ratio_pre_avg, 3),
        "rentable_frac_avg": round(rentable_frac_avg, 3),
        "soc_bind_frac_avg": round(soc_bind_frac_avg, 3),

        # balance
        "empty_ratio_avg": round(empty_ratio_avg, 3),
        "full_ratio_avg": round(full_ratio_avg, 3),
        "stock_std_avg": round(stock_std_avg, 3),
        "fill_p10_avg": round(fill_p10_avg, 3),
        "fill_p90_avg": round(fill_p90_avg, 3),
        "fill_spread_avg": round(fill_spread_avg, 3),

        # overflow
        "overflow_rerouted_total": overflow_rerouted_total,
        "overflow_dropped_total": overflow_dropped_total,  # always 0 unless you add it to core logs
        "overflow_extra_min_total": round(overflow_extra_min_total, 1),

        # score consistency
        "J_run": round(float(J_run), 3),
        "J_sum": round(float(J_sum), 3),
        "reward_plus_J_sum": round(float(reward_plus_J_sum), 3),
        "ticks": T,
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/network_porto10.yaml")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Make weights explicit (including delta_queue)
    score_weights = ScoreWeights(
        alpha_unavailability=100.0,
        beta_reloc_km=0.5,
        gamma_energy_cost=10.0,
        delta_queue=10.0,
    )

    all_results = []

    for ep in range(args.episodes):
        env = PortoMicromobilityEnv(
            cfg_path=args.config,
            score_weights=score_weights,
            episode_hours=args.hours,
            seed=args.seed + ep,
        )
        agent = HeuristicAgent()

        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)

        kpis = compute_episode_kpis(env, total_reward=total_reward)
        kpis["episode"] = ep
        kpis["total_reward"] = round(total_reward, 3)
        all_results.append(kpis)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
