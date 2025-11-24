import argparse
import json

import numpy as np
import yaml

from envs.porto_env import PortoMicromobilityEnv, ScoreWeights


class StaticPolicyAgent:
    """
    Agent that always uses the same 4-dim action vector a in [0,1]^4.
    This is our "static policy"; black-box search will choose a.
    """

    def __init__(self, action: np.ndarray):
        assert action.shape == (4,)
        self.action = action.astype(float)

    def act(self, obs):
        # obs is ignored; this is a pure static policy
        return self.action


def compute_episode_kpis(env: PortoMicromobilityEnv):
    """Aggregate KPIs from env.sim.logs for one episode."""
    logs = env.sim.logs
    if not logs:
        return {}

    availability_avg = float(np.mean([r.get("availability", 0.0) for r in logs]))
    unmet_total = int(sum(r.get("unmet", 0) for r in logs))
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

    # Episode-level cost: average per tick (same as your J_run definition)
    w = env.score_weights
    T = len(logs)

    J_sum = (
        w.alpha_unavailability * T * (1.0 - availability_avg)
        + w.beta_reloc_km * reloc_km_total
        + w.gamma_energy_cost * charge_cost_eur_total
    )
    J_run = J_sum / T

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


def load_score_weights(cfg_path: str) -> ScoreWeights:
    """Read weights from YAML `score:` section, with defaults if missing."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    score_cfg = cfg.get("score", {})
    alpha = float(score_cfg.get("alpha_unavailability", 100.0))
    beta = float(score_cfg.get("beta_reloc_km", 0.5))
    gamma = float(score_cfg.get("gamma_energy_cost", 10.0))
    return ScoreWeights(
        alpha_unavailability=alpha,
        beta_reloc_km=beta,
        gamma_energy_cost=gamma,
    )


def evaluate_static_action(
    action: np.ndarray,
    cfg_path: str,
    score_weights: ScoreWeights,
    episode_hours: int,
    episodes: int,
    base_seed: int,
):
    """
    Run `episodes` episodes with the same static action, return
    average KPIs and average J_run.
    """
    all_kpis = []
    total_rewards = []

    for ep in range(episodes):
        env = PortoMicromobilityEnv(
            cfg_path=cfg_path,
            score_weights=score_weights,
            episode_hours=episode_hours,
            seed=base_seed + ep,
        )
        agent = StaticPolicyAgent(action=action)

        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            a = agent.act(obs)
            obs, reward, done, info = env.step(a)
            total_reward += reward

        kpis = compute_episode_kpis(env)
        kpis["episode"] = ep
        kpis["total_reward"] = round(total_reward, 3)
        all_kpis.append(kpis)
        total_rewards.append(total_reward)

    # aggregate across episodes
    J_runs = [k["J_run"] for k in all_kpis]
    avg_J = float(np.mean(J_runs))
    std_J = float(np.std(J_runs))

    return {
        "action": action.tolist(),
        "episodes": episodes,
        "avg_J_run": round(avg_J, 3),
        "std_J_run": round(std_J, 3),
        "avg_total_reward": round(float(np.mean(total_rewards)), 3),
        "per_episode": all_kpis,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/network_porto10.yaml")
    parser.add_argument("--hours", type=int, default=24, help="episode length in hours")
    parser.add_argument("--episodes_per_candidate", type=int, default=3)
    parser.add_argument("--n_candidates", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    score_weights = load_score_weights(args.config)

    candidates_results = []
    best = None

    for k in range(args.n_candidates):
        # sample a random action in [0,1]^4
        action = rng.random(4)

        res = evaluate_static_action(
            action=action,
            cfg_path=args.config,
            score_weights=score_weights,
            episode_hours=args.hours,
            episodes=args.episodes_per_candidate,
            base_seed=args.seed + 1000 * k,
        )
        candidates_results.append(res)

        if best is None or res["avg_J_run"] < best["avg_J_run"]:
            best = {
                "candidate_index": k,
                **res,
            }

        print(
            f"[{k+1}/{args.n_candidates}] "
            f"action={np.round(action, 3)} "
            f"avg_J_run={res['avg_J_run']:.3f}",
            flush=True,
        )



    print("\n==== BEST STATIC POLICY ====", flush=True)
    print(json.dumps(best, indent=2))



if __name__ == "__main__":
    main()
