import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from envs.porto_env import PortoMicromobilityEnv
from sim.kpis import compute_episode_kpis


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


# -----------------------------
# Arms: grid
# -----------------------------
def make_param_arms(km_budgets: list[float], charge_fracs: list[float]) -> list[dict[str, float]]:
    arms: list[dict[str, float]] = []
    for km in km_budgets:
        for c in charge_fracs:
            arms.append({"km_budget": float(km), "charge_budget_frac": float(c)})
    return arms


# Context extraction (from obs)
def context_from_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Return a compact feature vector x (d,).

    Uses only observation dict fields (no sim internals).
    """
    fill = np.asarray(obs["fill_ratio"], dtype=float)
    soc = np.asarray(obs["soc"], dtype=float)
    waiting = np.asarray(obs["waiting"], dtype=float)
    tod = np.asarray(obs["time_of_day"], dtype=float)

    empty_frac = float(np.mean(fill < 0.1))
    full_frac = float(np.mean(fill > 0.9))
    fill_mean = float(np.mean(fill))
    fill_std = float(np.std(fill))

    soc_mean = float(np.mean(soc))
    soc_p10 = float(np.percentile(soc, 10)) if soc.size else 0.0

    wait_mean = float(np.mean(waiting))
    wait_p95 = float(np.percentile(waiting, 95)) if waiting.size else 0.0
    high_wait_frac = float(np.mean(waiting > 5.0))
    very_high_wait_frac = float(np.mean(waiting > 10.0))

    sin_h = float(tod[0]) if tod.size >= 1 else 0.0
    cos_h = float(tod[1]) if tod.size >= 2 else 1.0

    # optional fields (your env includes these)
    weather_fac = float(np.asarray(obs.get("weather_factor", np.array([1.0])))[0])
    event_stats = np.asarray(obs.get("event_stats", np.array([1.0, 1.0])), dtype=float).reshape(-1)
    event_mean = float(event_stats[0]) if event_stats.size >= 1 else 1.0
    event_max = float(event_stats[1]) if event_stats.size >= 2 else 1.0

    # Add bias term at the end
    return np.array(
        [
            empty_frac,
            full_frac,
            fill_mean,
            fill_std,
            soc_mean,
            soc_p10,
            wait_mean,
            wait_p95,
            high_wait_frac,
            very_high_wait_frac,
            weather_fac,
            event_mean,
            event_max,
            sin_h,
            cos_h,
            1.0,  # bias
        ],
        dtype=float,
    )


# -----------------------------
# Context scaling (frozen z-score)
# -----------------------------
class ZScoreScaler:
    """Frozen z-score scaler using Welford running moments during calibration."""

    def __init__(self, d: int, *, eps: float = 1e-8, scale_mask: np.ndarray | None = None) -> None:
        self.d = int(d)
        self.eps = float(eps)
        self.n = 0
        self.mean = np.zeros(self.d, dtype=float)
        self.M2 = np.zeros(self.d, dtype=float)

        if scale_mask is None:
            scale_mask = np.ones(self.d, dtype=bool)
        self.scale_mask = np.asarray(scale_mask, dtype=bool).reshape(self.d)

        self.frozen = False
        self.std = np.ones(self.d, dtype=float)

    def update(self, x: np.ndarray) -> None:
        if self.frozen:
            return
        x = np.asarray(x, dtype=float).reshape(self.d)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def freeze(self) -> None:
        if self.n < 2:
            self.std = np.ones(self.d, dtype=float)
        else:
            var = self.M2 / (self.n - 1)
            std = np.sqrt(np.maximum(var, 0.0))
            self.std = np.maximum(std, self.eps)
        self.frozen = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(self.d)
        z = x.copy()
        if self.frozen:
            z[self.scale_mask] = (z[self.scale_mask] - self.mean[self.scale_mask]) / self.std[self.scale_mask]
        else:
            z[self.scale_mask] = (z[self.scale_mask] - self.mean[self.scale_mask]) / np.maximum(
                self.std[self.scale_mask], self.eps
            )
        return z


def calibrate_context_scaler(
    *,
    cfg_path: str,
    hours: int,
    seed0: int,
    scenario: str,
    scenario_params: dict[str, Any],
    reloc_planner: str | None,
    charge_planner: str | None,
    default_action: np.ndarray,
    block_minutes: int,
    calib_episodes: int,
    d: int,
) -> ZScoreScaler:
    """Collect context vectors at block boundaries and fit a frozen z-score scaler."""
    # Do not scale the bias term (last feature).
    scale_mask = np.ones(d, dtype=bool)
    scale_mask[-1] = False

    scaler = ZScoreScaler(d=d, scale_mask=scale_mask)

    for ep in range(int(calib_episodes)):
        seed = int(seed0) + ep

        env = PortoMicromobilityEnv(
            cfg_path=cfg_path,
            episode_hours=hours,
            seed=seed,
            reloc_name_override=reloc_planner,
            charge_name_override=charge_planner,
        )
        obs = env.reset()
        apply_scenario(env, scenario=scenario, seed=seed, **(scenario_params or {}))

        dt_min = int(env.dt_min)
        block_ticks = max(1, int(block_minutes // dt_min))
        max_steps = int(env.max_steps)

        a = np.asarray(default_action, dtype=float).reshape(4)
        a = np.clip(a, 0.0, 1.0)

        done = False
        step0 = 0
        while not done:
            # Context at the start of the block
            x_raw = context_from_obs(obs)
            scaler.update(x_raw)

            # Roll the block
            for _ in range(block_ticks):
                obs, _r, done, _info = env.step(a)
                step0 += 1
                if done or step0 >= max_steps:
                    done = True
                    break

    scaler.freeze()
    return scaler


# LinUCB (disjoint model per arm)
class LinUCB:
    """Disjoint LinUCB.

    For each arm a:
      A_a = I * reg + sum x x^T
      b_a = sum r x
    theta_a = A_a^{-1} b_a
    p(a|x) = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)
    """

    def __init__(self, n_arms: int, d: int, *, alpha: float = 1.0, reg: float = 1.0, seed: int = 0) -> None:
        self.n_arms = int(n_arms)
        self.d = int(d)
        self.alpha = float(alpha)
        self.reg = float(reg)
        self.rng = np.random.default_rng(seed)

        self.A = np.stack([np.eye(self.d) * self.reg for _ in range(self.n_arms)], axis=0)  # (K,d,d)
        # Maintain inverse directly to avoid repeated inversions in select_arm
        self.A_inv = np.stack([np.eye(self.d) / self.reg for _ in range(self.n_arms)], axis=0)  # (K,d,d)

        self.b = np.zeros((self.n_arms, self.d), dtype=float)  # (K,d)
        self.counts = np.zeros(self.n_arms, dtype=int)

    def select_arm(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(self.d)

        untried = np.where(self.counts == 0)[0]
        if untried.size > 0:
            return int(self.rng.choice(untried))

        scores = np.empty(self.n_arms, dtype=float)
        for a in range(self.n_arms):
            A_inv = self.A_inv[a]
            theta = A_inv @ self.b[a]
            mean = float(theta @ x)

            quad = float(x @ A_inv @ x)
            quad = max(quad, 0.0)  # numerical safety
            bonus = float(self.alpha * np.sqrt(quad))

            scores[a] = mean + bonus

        m = float(np.max(scores))
        idx = np.flatnonzero(np.isclose(scores, m))
        return int(self.rng.choice(idx))

    def update(self, arm_idx: int, x: np.ndarray, reward: float) -> None:
        a = int(arm_idx)
        x = np.asarray(x, dtype=float).reshape(self.d)
        r = float(reward)

        # Update A and b
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x

        # Sherman–Morrison update for inverse:
        # (A + x x^T)^(-1) = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        A_inv = self.A_inv[a]
        v = A_inv @ x
        denom = 1.0 + float(x @ v)
        denom = max(denom, 1e-12)  # safety
        self.A_inv[a] = A_inv - np.outer(v, v) / denom

        self.counts[a] += 1


# Episode runner (receding-horizon contextual bandit)
RELOC_BUDGET_KEY = "km_budget"
CHARGE_BUDGET_KEY = "charge_budget_frac_override"


def run_episode(
    *,
    cfg_path: str,
    hours: int,
    seed: int,
    scenario: str,
    scenario_params: dict[str, Any],
    reloc_planner: str | None,
    charge_planner: str | None,
    arms: list[dict[str, float]],
    bandit: LinUCB,
    default_action: np.ndarray,
    block_minutes: int,
    warmup_blocks: int,
    scaler: ZScoreScaler | None,
) -> dict[str, Any]:
    env = PortoMicromobilityEnv(
        cfg_path=cfg_path,
        episode_hours=hours,
        seed=seed,
        reloc_name_override=reloc_planner,
        charge_name_override=charge_planner,
    )
    obs = env.reset()
    apply_scenario(env, scenario=scenario, seed=seed, **(scenario_params or {}))

    dt_min = int(env.dt_min)
    block_ticks = max(1, int(block_minutes // dt_min))
    max_steps = int(env.max_steps)

    a = np.asarray(default_action, dtype=float).reshape(4)
    a = np.clip(a, 0.0, 1.0)

    blocks: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False

    warmup_arm_idx = int(len(arms) // 2)

    step0 = 0
    while not done:
        b_idx = int(step0 // block_ticks)

        # Context at the start of the block
        x_raw = context_from_obs(obs)
        x = scaler.transform(x_raw) if scaler is not None else x_raw

        if b_idx < int(warmup_blocks):
            arm_idx = warmup_arm_idx
            policy = "warmup"
        else:
            arm_idx = bandit.select_arm(x)
            policy = "linucb"

        arm = arms[arm_idx]
        km_budget = float(arm["km_budget"])
        c_frac = float(arm["charge_budget_frac"])

        # Apply arm for this block (planner parameter overrides)
        env.base_reloc_params = dict(env.base_reloc_params)

        if str(env.reloc_name).lower() in ("budgeted", "relocation_budgeted", "plan_relocation_budgeted"):
            env.base_reloc_params[RELOC_BUDGET_KEY] = km_budget
        else:
            # Ensure we don't leak budget params into non-budget planners
            env.base_reloc_params.pop(RELOC_BUDGET_KEY, None)

        env.base_charge_params = dict(env.base_charge_params)
        env.base_charge_params[CHARGE_BUDGET_KEY] = c_frac

        # Roll the block
        block_reward = 0.0
        block_steps = 0

        for _ in range(block_ticks):
            obs, r, done, _info = env.step(a)
            rr = float(r)
            block_reward += rr
            total_reward += rr
            block_steps += 1

            step0 += 1
            if done or step0 >= max_steps:
                done = True
                break

        # Learning reward:
        # reward_t = -J_t  => mean_reward_block = -(mean_J_block)
        # Higher is better, and it is horizon-invariant if you use the mean.
        mean_block_reward = float(block_reward / max(block_steps, 1))

        if b_idx >= int(warmup_blocks):
            bandit.update(arm_idx, x, mean_block_reward)

        # Optional: cumulative KPIs "so far" (NOT block KPIs)
        # This keeps you strictly on compute_episode_kpis only.
        kpis_so_far = compute_episode_kpis(env, total_reward=total_reward) if (env.sim and env.sim.logs) else {}

        blocks.append(
            {
                "block_idx": int(b_idx),
                "policy": policy,
                "arm_idx": int(arm_idx),
                "arm": {"km_budget": km_budget, "charge_budget_frac": c_frac},
                "context_raw": x_raw.tolist(),
                "context_scaled": x.tolist(),
                "block_reward_sum_env": float(block_reward),
                "block_steps": int(block_steps),
                "learn_reward_mean_env": float(mean_block_reward),
                "episode_kpis_so_far": kpis_so_far,  # cumulative-to-date
                "bandit_pulls_arm": int(bandit.counts[arm_idx]),
            }
        )

    # Episode KPIs (true episode summary)
    episode_kpis = compute_episode_kpis(env, total_reward=total_reward) if (env.sim and env.sim.logs) else {}

    return {
        "seed": int(seed),
        "hours": int(hours),
        "scenario": str(scenario),
        "scenario_params": dict(scenario_params or {}),
        "default_action": a.tolist(),
        "block_minutes": int(block_minutes),
        "block_ticks": int(block_ticks),
        "warmup_blocks": int(warmup_blocks),
        "total_reward_env": float(total_reward),
        "score_cfg": asdict(env.score_cfg),
        "episode_kpis": episode_kpis,
        "blocks": blocks,
    }


# Main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/network_porto10.yaml")
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed0", type=int, default=42)

    # planners
    p.add_argument("--reloc", default="greedy")
    p.add_argument("--charge", default="slack")

    # scenario
    p.add_argument("--scenario", default="baseline", help="baseline | hotspot_od | hetero_lambda | event_heavy")
    p.add_argument("--scenario_params_json", default="", help="JSON dict of scenario params (optional).")

    # arms
    p.add_argument("--km_budgets", type=float, nargs="+", default=[0, 5, 10, 20, 40])
    p.add_argument("--charge_fracs", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])

    # receding horizon
    p.add_argument("--block_minutes", type=int, default=60, help="How often to reselect an arm.")
    p.add_argument("--warmup_blocks", type=int, default=1, help="Blocks to run with neutral arm before learning.")

    # contextual bandit params
    p.add_argument("--linucb_alpha", type=float, default=1.0, help="Exploration strength.")
    p.add_argument("--linucb_reg", type=float, default=1.0, help="Ridge regularization.")

    # context scaling
    p.add_argument("--ctx_scale", choices=["none", "zscore"], default="zscore")
    p.add_argument("--ctx_calib_episodes", type=int, default=5, help="Episodes used to fit frozen context scaler.")

    # constant action (keep neutral to avoid confounding)
    p.add_argument("--default_action", type=float, nargs=4, default=[0.0, 0.0, 0.0, 0.0])

    # output
    p.add_argument("--out", default="out/bandit_contextual_rh.json")

    args = p.parse_args()

    scenario = str(args.scenario).strip()
    scenario_params: dict[str, Any] = {}
    if args.scenario_params_json.strip():
        scenario_params = json.loads(args.scenario_params_json)
        if not isinstance(scenario_params, dict):
            raise ValueError("--scenario_params_json must be a JSON dict.")

    arms = make_param_arms(km_budgets=list(args.km_budgets), charge_fracs=list(args.charge_fracs))

    # Context dim is fixed by context_from_obs: 16
    d = 16

    scaler: ZScoreScaler | None = None
    if args.ctx_scale == "zscore":
        scaler = calibrate_context_scaler(
            cfg_path=str(args.config),
            hours=int(args.hours),
            seed0=int(args.seed0),
            scenario=scenario,
            scenario_params=scenario_params,
            reloc_planner=str(args.reloc) if args.reloc else None,
            charge_planner=str(args.charge) if args.charge else None,
            default_action=np.asarray(args.default_action, dtype=float),
            block_minutes=int(args.block_minutes),
            calib_episodes=int(args.ctx_calib_episodes),
            d=d,
        )

    bandit = LinUCB(
        n_arms=len(arms),
        d=d,
        alpha=float(args.linucb_alpha),
        reg=float(args.linucb_reg),
        seed=int(args.seed0),
    )

    episodes_out: list[dict[str, Any]] = []
    for ep in range(int(args.episodes)):
        seed = int(args.seed0) + ep
        ep_out = run_episode(
            cfg_path=str(args.config),
            hours=int(args.hours),
            seed=seed,
            scenario=scenario,
            scenario_params=scenario_params,
            reloc_planner=str(args.reloc) if args.reloc else None,
            charge_planner=str(args.charge) if args.charge else None,
            arms=arms,
            bandit=bandit,
            default_action=np.asarray(args.default_action, dtype=float),
            block_minutes=int(args.block_minutes),
            warmup_blocks=int(args.warmup_blocks),
            scaler=scaler,
        )
        episodes_out.append(ep_out)

    # Summary: average episode J_run over episodes (from compute_episode_kpis output)
    j_runs = [e.get("episode_kpis", {}).get("J_run", np.nan) for e in episodes_out]
    j_runs = np.array([x for x in j_runs if np.isfinite(x)], dtype=float)

    summary = {
        "config": str(args.config),
        "hours": int(args.hours),
        "episodes": int(args.episodes),
        "seed0": int(args.seed0),
        "scenario": scenario,
        "scenario_params": scenario_params,
        "reloc_planner": str(args.reloc),
        "charge_planner": str(args.charge),
        "block_minutes": int(args.block_minutes),
        "warmup_blocks": int(args.warmup_blocks),
        "linucb_alpha": float(args.linucb_alpha),
        "linucb_reg": float(args.linucb_reg),
        "ctx_scale": str(args.ctx_scale),
        "ctx_calib_episodes": int(args.ctx_calib_episodes),
        "n_arms": len(arms),
        "arms": arms,
        "bandit_arm_pulls": bandit.counts.tolist(),
        "J_run_mean": float(np.mean(j_runs)) if j_runs.size else float("nan"),
        "J_run_std": float(np.std(j_runs, ddof=1)) if j_runs.size > 1 else 0.0,
    }

    out_obj = {"summary": summary, "episodes": episodes_out}

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"Saved: {args.out}")
    print(json.dumps({k: summary[k] for k in ["scenario", "J_run_mean", "J_run_std", "bandit_arm_pulls"]}, indent=2))


if __name__ == "__main__":
    main()
