import argparse
import json

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


# --- Main training loop ---
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/network_porto10.yaml")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_steps", type=int, default=100_000)
    args = parser.parse_args()

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

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log="./sac_tensorboard/",
    )
    model.learn(total_timesteps=args.train_steps)

    # Evaluate the trained agent
    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
        # Optionally, extract KPIs from info['kpi'] if needed
        results.append({"episode": ep, "total_reward": round(total_reward, 3)})

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
