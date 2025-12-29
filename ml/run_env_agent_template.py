import argparse
import json

import numpy as np

from envs.porto_env import PortoMicromobilityEnv, ScoreWeights


class DummyAgent:
    """Template agent.

    Replace the logic in `act()` with your model:
        - can be random
        - can be a neural network
        - can be a GNN that reads obs and outputs 4 numbers in [0,1].
    """

    def __init__(self, action_dim: int = 4) -> None:
        self.action_dim = action_dim

    def act(self, obs) -> np.ndarray:
        # TODO: replace with your policy
        # For now: random action in [0,1]^4
        return np.random.Generator(self.action_dim)


def compute_episode_kpis(env: PortoMicromobilityEnv) -> dict[str, float]:
    """Aggregate KPIs from env.sim.logs for one episode."""
    logs = env.sim.logs
    if not logs:
        return {}

    availability_avg = float(np.mean([r.get("availability", 0.0) for r in logs]))
    unmet_total = int(sum(r.get("unmet", 0) for r in logs))

    # if you log per-tick queue_total in Sim.logs
    queue_max = int(max(r.get("queue_total", 0) for r in logs))
    queue_avg = float(np.mean([r.get("queue_total", 0) for r in logs]))

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

    # Episode-level cost: average per tick (Option B)
    w = env.score_weights
    T = len(logs)

    J_sum = (
        w.alpha_unavailability * T * (1.0 - availability_avg)
        + w.beta_reloc_km * reloc_km_total
        + w.gamma_energy_cost * charge_cost_eur_total
    )
    J_run = J_sum / T  # average cost per tick

    return {
        "unmet_total": unmet_total,
        "availability_avg": round(availability_avg, 3),
        "queue_total_max": queue_max,
        "queue_total_avg": round(queue_avg, 2),
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
        "ticks": T,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    # adjust default path depending on where you run from
    parser.add_argument("--config", default="configs/network_porto10.yaml")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    score_weights = ScoreWeights(
        alpha_unavailability=100.0,
        beta_reloc_km=0.5,
        gamma_energy_cost=10.0,
    )

    results = []

    for ep in range(args.episodes):
        env = PortoMicromobilityEnv(
            cfg_path=args.config,
            score_weights=score_weights,
            episode_hours=args.hours,
            seed=args.seed + ep,
        )
        agent = DummyAgent(action_dim=4)

        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        kpis = compute_episode_kpis(env)
        kpis["episode"] = ep
        kpis["total_reward"] = round(total_reward, 3)
        results.append(kpis)

    # JSON is easy to diff / feed to notebooks
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
