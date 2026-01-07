"""sac.py.

Train and evaluate a Soft Actor-Critic (SAC) agent on PortoMicromobilityEnv.

This script:
- Trains SAC using stable-baselines3
- Evaluates over multiple seeds
- Optionally applies scenarios (baseline / hotspot_od / hetero_lambda / event_heavy)
- Saves JSON to --out (same schema as other agents) for analyze_results.py compatibility
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC

from envs.porto_env import PortoMicromobilityEnv
from sim.kpis import compute_episode_kpis


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
        env.events_matrix = env.events_matrix * float(event_scale)
        return

    raise ValueError(f"Unknown scenario '{scenario}'. Use: baseline, hotspot_od, hetero_lambda, event_heavy.")


class PortoGymWrapper(gym.Env):
    """Gym wrapper for PortoMicromobilityEnv with optional frame-skipping."""

    def __init__(
        self,
        cfg_path: str,
        episode_hours: int,
        seed: int,
        scenario: str = "baseline",
        scenario_params: dict[str, Any] | None = None,
        action_repeat: int = 1,  # NEW: how many env steps per agent action
    ) -> None:
        super().__init__()
        self.cfg_path = cfg_path
        self.episode_hours = episode_hours
        self.base_seed = seed
        self.scenario = scenario
        self.scenario_params = scenario_params or {}
        self.action_repeat = max(1, int(action_repeat))

        self.env = PortoMicromobilityEnv(
            cfg_path=cfg_path,
            episode_hours=episode_hours,
            seed=seed,
        )

        # Initialize to get observation shape
        obs_space = self.env.reset()
        obs_dim = sum(v.size for v in obs_space.values())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self._obs_keys = list(obs_space.keys())

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        actual_seed = seed if seed is not None else self.base_seed
        obs = self.env.reset(seed=actual_seed)
        apply_scenario(self.env, scenario=self.scenario, seed=actual_seed, **self.scenario_params)
        return self._flatten_obs(obs), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action for action_repeat steps, accumulating reward."""
        total_reward = 0.0
        done = False
        info = {}
        obs = None

        for _ in range(self.action_repeat):
            obs_dict, reward, done, info = self.env.step(action)
            total_reward += float(reward)
            if done:
                break

        obs = self._flatten_obs(obs_dict) if obs_dict is not None else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, total_reward, done, False, info

    def _flatten_obs(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([obs_dict[k].flatten() for k in self._obs_keys]).astype(np.float32)


def run_one_episode(
    *,
    cfg_path: str,
    hours: int,
    seed: int,
    scenario: str,
    scenario_params: dict[str, Any],
    model: SAC,
    obs_keys: list[str],
    action_repeat: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a single evaluation episode with a trained SAC model."""
    env = PortoMicromobilityEnv(cfg_path=cfg_path, episode_hours=hours, seed=seed)
    obs_dict = env.reset()
    apply_scenario(env, scenario=scenario, seed=seed, **scenario_params)

    def flatten_obs(obs_d: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([obs_d[k].flatten() for k in obs_keys]).astype(np.float32)

    obs = flatten_obs(obs_dict)
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        for _ in range(action_repeat):
            obs_dict, reward, done, _info = env.step(action)
            total_reward += float(reward)

            if done:
                break

        obs = flatten_obs(obs_dict)

    kpis = compute_episode_kpis(env, total_reward=total_reward)
    meta = {
        "seed": int(seed),
        "hours": int(hours),
        "scenario": scenario,
        "scenario_params": scenario_params,
        "action_repeat": action_repeat,
    }
    return kpis, meta


def _aggregate(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[k] = {
            "mean": float(np.mean(vals)) if vals.size else float("nan"),
            "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "min": float(np.min(vals)) if vals.size else float("nan"),
            "max": float(np.max(vals)) if vals.size else float("nan"),
        }
    return out


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


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/network_porto10.yaml")
    p.add_argument("--hours", type=int, default=24, help="Episode length in hours")
    p.add_argument("--seed0", type=int, default=42, help="Base random seed")
    p.add_argument("--training_episodes", type=int, default=365, help="Number of training episodes (approximate)")
    p.add_argument("--testing_seeds", type=int, default=30, help="Number of evaluation seeds")

    p.add_argument("--action_repeat", type=int, default=6, help="Repeat each action for N env steps (6 = 30min if dt=5min)")

    # scenarios
    p.add_argument("--scenario", default="baseline", help="baseline | hotspot_od | hetero_lambda | event_heavy")
    p.add_argument("--scenario_params_json", default="", help="JSON dict of scenario params (optional).")

    # model I/O
    p.add_argument("--save_model", action="store_true", default=True, help="Whether to save the trained model")
    p.add_argument("--load_model", default="", help="Path to load a pre-trained model (skip training if set)")
    p.add_argument("--tensorboard_log", action="store_true", default=False, help="Whether to log to TensorBoard")

    # output
    p.add_argument("--out", default="out/sac_eval.json", help="Output JSON path")

    args = p.parse_args()

    scenario = str(args.scenario).strip()
    scenario_params: dict[str, Any] = {}
    if args.scenario_params_json.strip():
        scenario_params = json.loads(args.scenario_params_json)
        if not isinstance(scenario_params, dict):
            raise ValueError("--scenario_params_json must be a JSON dict.")

    # Paths
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    models_dir = parent_dir / "models"
    models_dir.mkdir(exist_ok=True)
    if args.tensorboard_log:
        logs_dir = parent_dir / "logs"
        tensorboard_dir = logs_dir / "sac_tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)

    save_filename = f"sac_porto_{time.strftime('%Y%m%d_%H%M%S')}"

    # Training environment
    train_env = PortoGymWrapper(
        cfg_path=args.config,
        episode_hours=args.hours,
        seed=args.seed0,
        scenario=scenario,
        scenario_params=scenario_params,
        action_repeat=args.action_repeat,
    )

    steps_per_episode = math.ceil(args.hours * 60 / train_env.env.dt_min / args.action_repeat)
    train_steps = args.training_episodes * steps_per_episode
    model_path = models_dir / f"{save_filename}_steps{train_steps}_repeat{args.action_repeat}"

    if args.load_model.strip():
        print(f"Loading pre-trained model from: {args.load_model}")
        model = SAC.load(Path(args.load_model), env=train_env)
    else:
        print(
            f"Training SAC for {train_steps} agent steps (~{args.training_episodes} episodes, action_repeat={args.action_repeat})..."
        )
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            seed=args.seed0,
            tensorboard_log=str(tensorboard_dir) if args.tensorboard_log else None,
        )
        model.learn(total_timesteps=train_steps)

        if args.save_model:
            model.save(str(model_path))
            print(f"Model saved to: {model_path}")

    print(f"Evaluating over {args.testing_seeds} seeds...")
    rows: list[dict[str, Any]] = []
    obs_keys = train_env._obs_keys

    for k in range(int(args.testing_seeds)):
        seed = int(args.seed0) + k
        kpis, meta = run_one_episode(
            cfg_path=str(args.config),
            hours=int(args.hours),
            seed=seed,
            scenario=scenario,
            scenario_params=scenario_params,
            model=model,
            obs_keys=obs_keys,
            action_repeat=args.action_repeat,
        )
        row = {**meta, **kpis}
        rows.append(row)
        print(f"  seed={seed}: J_run={kpis.get('J_run', float('nan')):.3f}")

    # Report keys (matching other agents)
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
        "seeds": int(args.testing_seeds),
        "training_episodes": int(args.training_episodes),
        "scenario": scenario,
        "scenario_params": scenario_params,
        "agent": "sac",
        "model_path": str(model_path) if args.save_model and not args.load_model else args.load_model,
        "report_keys": report_keys,
        "summary": summary,
        "episodes": rows,
    }

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_convert_ndarray_to_list(out), f, indent=2)

    # Console report
    s = summary
    print("\n=== SAC policy summary (mean ± std over seeds) ===")
    print(f"  J_run:         {s['J_run']['mean']:.3f} ± {s['J_run']['std']:.3f}")
    print(f"  total_reward:  {s['total_reward']['mean']:.1f} ± {s['total_reward']['std']:.1f}")
    print(f"  avail (dw):    {s['availability_demand_weighted']['mean']:.3f} ± {s['availability_demand_weighted']['std']:.3f}")
    print(f"  unmet_rate:    {s['unmet_rate']['mean']:.3f} ± {s['unmet_rate']['std']:.3f}")
    print(f"  queue_p95:     {s['queue_total_p95']['mean']:.1f} ± {s['queue_total_p95']['std']:.1f}")
    print(f"  reloc_km:      {s['relocation_km_total']['mean']:.1f} ± {s['relocation_km_total']['std']:.1f}")
    print(f"  charge_eur:    {s['charging_cost_eur_total']['mean']:.2f} ± {s['charging_cost_eur_total']['std']:.2f}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
