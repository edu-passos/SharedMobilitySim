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
    w = env.score_weights  # includes delta_queue

    # Simple aggregates
    availability_avg = float(np.mean([r.get("availability", 0.0) for r in logs]))
    queue_total_max = int(max(r.get("queue_total", 0) for r in logs))
    queue_total_avg = float(np.mean([r.get("queue_total", 0) for r in logs]))
    queue_rate_avg = float(np.mean([r.get("queue_rate", 0.0) for r in logs]))

    unmet_total = int(sum(r.get("unmet", 0) for r in logs))

    reloc_km_total = float(sum(r.get("reloc_km", 0.0) for r in logs))
    reloc_ops_total = int(sum(r.get("reloc_ops", 0) for r in logs))

    charge_energy_kwh_total = float(sum(r.get("charge_energy_kwh", 0.0) for r in logs))
    charge_cost_eur_total = float(sum(r.get("charge_cost_eur", 0.0) for r in logs))
    charge_util_avg = float(np.mean([r.get("charge_utilization", 0.0) for r in logs]))

    overflow_rerouted_total = int(sum(r.get("overflow_rerouted", 0) for r in logs))
    overflow_dropped_total = int(sum(r.get("overflow_dropped", 0) for r in logs))
    overflow_extra_min_total = float(sum(r.get("overflow_extra_min", 0.0) for r in logs))

    soc_mean_avg = float(np.mean([r.get("soc_mean", 0.0) for r in logs]))
    full_ratio_avg = float(np.mean([r.get("full_ratio", 0.0) for r in logs]))
    empty_ratio_avg = float(np.mean([r.get("empty_ratio", 0.0) for r in logs]))
    stock_std_avg = float(np.mean([r.get("stock_std", 0.0) for r in logs]))

    # Demand-weighted immediate availability (more robust than tick mean)
    demand_sum = float(sum(r.get("demand_total", 0) for r in logs))
    served_new_sum = float(sum(r.get("served_new_total", 0) for r in logs))
    availability_demand_weighted = 1.0 if demand_sum <= 0 else (served_new_sum / demand_sum)

    # Exact J_sum from logs (must match env._compute_reward)
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

    J_run = J_sum / T

    # Reward consistency check (if reward = -J_t per tick)
    # total_reward should be approximately -J_sum
    reward_minus_negJ = float(total_reward + J_sum)

    return {
        "unmet_total": unmet_total,
        "availability_avg": round(availability_avg, 3),
        "availability_demand_weighted": round(float(availability_demand_weighted), 3),
        "queue_total_max": queue_total_max,
        "queue_total_avg": round(queue_total_avg, 2),
        "queue_rate_avg": round(queue_rate_avg, 3),
        "relocation_km_total": round(reloc_km_total, 2),
        "reloc_ops_total": reloc_ops_total,
        "charging_energy_kwh_total": round(charge_energy_kwh_total, 2),
        "charging_cost_eur_total": round(charge_cost_eur_total, 2),
        "charge_utilization_avg": round(charge_util_avg, 3),
        "overflow_rerouted_total": overflow_rerouted_total,
        "overflow_dropped_total": overflow_dropped_total,
        "overflow_extra_min_total": round(overflow_extra_min_total, 1),
        "soc_mean_avg": round(soc_mean_avg, 3),
        "full_ratio_avg": round(full_ratio_avg, 3),
        "empty_ratio_avg": round(empty_ratio_avg, 3),
        "stock_std_avg": round(stock_std_avg, 3),
        "J_run": round(J_run, 3),
        "J_sum": round(J_sum, 3),
        "reward_plus_J_sum": round(reward_minus_negJ, 3),  # should be ~0
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
