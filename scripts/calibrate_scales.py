"""calibrate_scales.py.

Estimate normalization scales (A0, R0, C0, Q0) for the *normalized* client-satisfaction objective:

    J_t = wA * (unavailability / A0)
        + wR * (reloc_km / R0)
        + wC * (charge_cost_eur / C0)
        + wQ * (queue_total / Q0)

We calibrate A0,R0,C0,Q0 as *per-tick baseline magnitudes* under a fixed policy (constant action),
averaged over many random seeds.
"""

import argparse
import json
from dataclasses import asdict
from typing import Any

import numpy as np

from envs.porto_env import PortoMicromobilityEnv


def _collect_tick_arrays(env: PortoMicromobilityEnv) -> dict[str, np.ndarray]:
    sim = env.sim
    if sim is None or not sim.logs:
        raise RuntimeError("No sim logs found. Did the episode run any steps?")

    logs = sim.logs

    def arr(key: str, default: float = 0.0) -> np.ndarray:
        return np.array([r.get(key, default) for r in logs], dtype=float)

    availability = arr("availability")
    unavailability = 1.0 - availability

    return {
        "unavailability": unavailability,
        "reloc_km": arr("reloc_km"),
        "charge_cost_eur": arr("charge_cost_eur"),
        "queue_total": arr("queue_total"),
        "ticks": np.array([len(logs)], dtype=float),
    }


def run_one_episode(
    *,
    cfg_path: str,
    hours: int,
    seed: int,
    action: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
    env = PortoMicromobilityEnv(cfg_path=cfg_path, episode_hours=hours, seed=seed)
    obs = env.reset()

    done = False
    total_reward = 0.0
    a = np.asarray(action, dtype=float)
    if a.shape != (4,):
        raise ValueError(f"baseline action must be shape (4,), got {a.shape}")
    a = np.clip(a, 0.0, 1.0)

    while not done:
        obs, reward, done, info = env.step(a)
        total_reward += float(reward)

    tick = _collect_tick_arrays(env)
    # Per-episode per-tick means (this is what "scales" represent)
    ep_means = {
        "A0_unavailability": float(np.mean(tick["unavailability"])),
        "R0_reloc_km": float(np.mean(tick["reloc_km"])),
        "C0_charge_cost_eur": float(np.mean(tick["charge_cost_eur"])),
        "Q0_queue_total": float(np.mean(tick["queue_total"])),
    }

    meta = {
        "seed": seed,
        "ticks": int(tick["ticks"][0]),
        "total_reward": float(total_reward),
        "score_cfg": asdict(env.score_cfg),
    }
    return ep_means, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/network_porto10.yaml")
    p.add_argument("--hours", type=int, default=24, help="Episode length (hours) for calibration.")
    p.add_argument("--seeds", type=int, default=30, help="Number of independent seeds to average over.")
    p.add_argument("--seed0", type=int, default=42, help="First seed (we will use seed0..seed0+seeds-1).")
    p.add_argument(
        "--action",
        type=float,
        nargs=4,
        default=[0.5, 0.5, 0.5, 0.5],
        help="Baseline action in [0,1]^4 used for all ticks (e.g. 0 0 0 0).",
    )
    p.add_argument("--eps", type=float, default=1e-6, help="Lower bound clamp for scales.")
    p.add_argument("--print_episode_table", action="store_true", help="Print per-seed episode means.")
    args = p.parse_args()

    cfg_path = str(args.config)
    action = np.array(args.action, dtype=float)
    eps = float(args.eps)

    ep_rows: list[dict[str, float]] = []
    metas: list[dict[str, Any]] = []

    for k in range(int(args.seeds)):
        seed = int(args.seed0) + k
        ep_means, meta = run_one_episode(cfg_path=cfg_path, hours=int(args.hours), seed=seed, action=action)
        ep_rows.append(ep_means)
        metas.append(meta)

    # Aggregate
    keys = ["A0_unavailability", "R0_reloc_km", "C0_charge_cost_eur", "Q0_queue_total"]
    mat = np.array([[row[k] for k in keys] for row in ep_rows], dtype=float)

    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)

    # Clamp (important for stability)
    mean_clamped = np.maximum(mean, eps)

    out_scales = dict(zip(keys, map(float, mean_clamped)))

    # Optional per-episode print
    if args.print_episode_table:
        for i, row in enumerate(ep_rows):
            print(f"[seed={int(args.seed0) + i}] " + " ".join(f"{k}={row[k]:.6f}" for k in keys))

    # YAML snippet
    print("\n# ----------------- CALIBRATED SCALES (per tick) -----------------")
    print("score:")
    print("  objective: client_satisfaction_normalized")
    print("  scales:")
    print(f"    A0_unavailability: {out_scales['A0_unavailability']:.6f}")
    print(f"    R0_reloc_km: {out_scales['R0_reloc_km']:.6f}")
    print(f"    C0_charge_cost_eur: {out_scales['C0_charge_cost_eur']:.6f}")
    print(f"    Q0_queue_total: {out_scales['Q0_queue_total']:.6f}")
    print(f"  eps: {eps:.1e}")
    print("# ---------------------------------------------------------------\n")

    # JSON summary (mean/std + inputs)
    summary = {
        "config": cfg_path,
        "hours": int(args.hours),
        "action": action.tolist(),
        "seed0": int(args.seed0),
        "seeds": int(args.seeds),
        "eps": eps,
        "scales_mean": dict(zip(keys, map(float, mean))),
        "scales_std": dict(zip(keys, map(float, std))),
        "scales_mean_clamped": out_scales,
        # sanity: ensure all episodes used the same scoring weights
        "score_cfg_used": metas[0]["score_cfg"] if metas else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
