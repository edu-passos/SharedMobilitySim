import argparse
import json
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC

from envs.porto_env import PortoMicromobilityEnv, ScoreWeights


class PortoGymWrapper(gym.Env):
    """Gym wrapper for PortoMicromobilityEnv."""

    def __init__(self, cfg_path: str, score_weights: ScoreWeights, episode_hours: int, seed: int) -> None:
        super().__init__()
        self.env = PortoMicromobilityEnv(
            cfg_path=cfg_path,
            score_weights=score_weights,
            episode_hours=episode_hours,
            seed=seed,
        )
        # Observation: concatenate all obs arrays into a flat vector
        obs_space = self.env.reset()
        obs_dim = sum(v.size for v in obs_space.values())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self._obs_keys = list(obs_space.keys())

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        obs = self.env.reset(seed=seed)
        return self._flatten_obs(obs), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        return self._flatten_obs(obs), reward, done, False, info

    def _flatten_obs(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([obs_dict[k].flatten() for k in self._obs_keys]).astype(np.float32)


def compute_episode_kpis(env: PortoMicromobilityEnv) -> dict[str, float]:
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

    # Episode-level J score as *average per tick* (so comparable across episode lengths)
    w = env.score_weights
    T = len(logs)

    # Sum of per-tick costs:
    #   J_sum = Σ_t [ α (1 - availability_t) + β reloc_km_t + γ charge_cost_t ]
    # Approximated using:
    #   availability_avg, reloc_km_total, charge_cost_total
    J_sum = (
        w.alpha_unavailability * T * (1.0 - availability_avg)
        + w.beta_reloc_km * reloc_km_total
        + w.gamma_energy_cost * charge_cost_eur_total
    )

    # Average cost per tick
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
        "ticks": len(logs),
    }


def _convert_ndarray_to_list(obj: object) -> object:
    """Recursively convert numpy arrays and scalars in a dict/list to Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _convert_ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_ndarray_to_list(i) for i in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_ndarray_to_list(i) for i in obj)
    return obj


# --- Main training loop ---
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-model", type=bool, default=True, help="Whether to save the trained model")
    parser.add_argument("--tensorboard-log", type=bool, default=True, help="Whether to log to TensorBoard")
    parser.add_argument("--train-logs", type=bool, default=True, help="Whether to save training logs to a JSON file")
    parser.add_argument("--config", default="configs/network_porto10.yaml")
    parser.add_argument("--hours", type=int, default=24, help="Episode length in hours")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--training-episodes", type=int, default=365 * 4, help="Number of training episodes (approximate)")
    parser.add_argument("--testing-episodes", type=int, default=10, help="Number of testing episodes")
    args = parser.parse_args()

    # Determine parent directory and models folder (when run as a script: `python3 -m ml.sac`)
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    models_dir = parent_dir / "models"
    logs_dir = parent_dir / "logs"
    tensorboard_dir = logs_dir / "sac_tensorboard"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)

    save_filename = f"sac_porto_{time.strftime('%Y%m%d_%H%M%S')}"

    score_weights = ScoreWeights(
        alpha_unavailability=100.0,
        beta_reloc_km=0.5,
        gamma_energy_cost=10.0,
    )

    env = PortoGymWrapper(
        cfg_path=args.config,
        score_weights=score_weights,
        episode_hours=args.hours,
        seed=args.seed,
    )

    train_steps = math.ceil(args.training_episodes * (args.hours * 60 / env.env.dt_min))

    model_path = models_dir / f"{save_filename}_steps{train_steps}"
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tensorboard_dir if args.tensorboard_log else None,
    )
    model.learn(total_timesteps=train_steps)

    if args.save_model:
        model.save(model_path)
    # model = SAC.load("./sac_porto_20251230_124539_steps200000", env=env)

    # Evaluate the trained agent
    results = []
    episode_res = []
    infos = {}
    total_rewards = []

    for ep in range(args.testing_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward

        res = {}
        res["episode"] = ep
        res["total_reward"] = round(total_reward, 3)
        res["kpis"] = compute_episode_kpis(env.env)
        episode_res.append(res)
        total_rewards.append(total_reward)
        infos.update({f"episode{ep}": info})

    # Aggregate across episodes
    J_runs = [ep["kpis"]["J_run"] for ep in episode_res]
    avg_J = float(np.mean(J_runs))
    std_J = float(np.std(J_runs))

    results = {
        "episodes": args.testing_episodes,
        "avg_J_run": round(avg_J, 3),
        "std_J_run": round(std_J, 3),
        "avg_total_reward": round(float(np.mean(total_rewards)), 3),
        "per_episode": episode_res,
    }

    print("=== Evaluation Results ===")
    print("Results:", json.dumps(results, indent=2))
    print("--------------------------")
    print("Additional info:", json.dumps(_convert_ndarray_to_list(infos), indent=2))
    print("==========================")

    if args.train_logs:
        out_data = {"results": results, "additional_info": _convert_ndarray_to_list(infos)}
        out_path = model_path.with_suffix(".json")
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
