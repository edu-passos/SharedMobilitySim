import argparse
import json

import numpy as np

from envs.porto_env import PortoMicromobilityEnv

# Multi-Armed Bandit (UCB1)


class UCB1Bandit:
    """UCB1 bandit over a discrete set of continuous action vectors (arms).

    Each arm corresponds to a fixed action a ∈ [0,1]^4 that will be applied for a full episode.
    Reward signal: episode total_reward (sum of per-tick rewards).
    """

    def __init__(self, arms: np.ndarray, c: float = 2.0, seed: int = 0) -> None:
        assert arms.ndim == 2 and arms.shape[1] == 4
        self.arms = arms.astype(float)
        self.c = float(c)
        self.rng = np.random.default_rng(seed)

        self.n_arms = self.arms.shape[0]
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.means = np.zeros(self.n_arms, dtype=float)

        # optional diagnostics
        self.total_pulls = 0

    def select_arm(self) -> int:
        """Pick an arm index using UCB1 (with mandatory exploration of untried arms)."""
        # Explore any untried arms first (in a randomized order to avoid bias)
        untried = np.where(self.counts == 0)[0]
        if untried.size > 0:
            return int(self.rng.choice(untried))

        t = max(self.total_pulls, 1)
        bonus = self.c * np.sqrt(np.log(t) / self.counts)
        ucb = self.means + bonus
        return self._argmax_random_tie(ucb)

    def update(self, arm_idx: int, reward: float) -> None:
        """Incremental mean update."""
        self.total_pulls += 1
        self.counts[arm_idx] += 1

        n = self.counts[arm_idx]
        old_mean = self.means[arm_idx]
        new_mean = old_mean + (reward - old_mean) / n
        self.means[arm_idx] = new_mean

    def best_arm(self) -> int:
        """Return the current best arm according to empirical mean."""
        tried = self.counts > 0
        if not np.any(tried):
            return 0
        masked = np.where(tried, self.means, -np.inf)
        return self._argmax_random_tie(masked)

    def _argmax_random_tie(self, x: np.ndarray) -> int:
        m = np.max(x)
        idx = np.flatnonzero(np.isclose(x, m))
        return int(self.rng.choice(idx))


class MultiArmedBanditAgent:
    """Episode-level MAB agent: chooses one arm per episode and plays it every step."""

    def __init__(self, bandit: UCB1Bandit) -> None:
        self.bandit = bandit
        self.last_arm_idx: int | None = None
        self._action: np.ndarray | None = None

    def begin_episode(self) -> np.ndarray:
        arm_idx = self.bandit.select_arm()
        self.last_arm_idx = arm_idx
        self._action = self.bandit.arms[arm_idx].copy()
        return self._action

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        # Stateless within episode (bandit is not contextual here)
        assert self._action is not None
        return self._action

    def end_episode(self, total_reward: float) -> None:
        assert self.last_arm_idx is not None
        self.bandit.update(self.last_arm_idx, float(total_reward))
        self.last_arm_idx = None
        self._action = None


# Arm construction


def _repair_action(a: np.ndarray) -> np.ndarray:
    """Only clip to [0,1]. Env already enforces low/high/target consistency."""
    return np.clip(a.astype(float), 0.0, 1.0)


def make_arm_set(
    k_random: int,
    seed: int,
    include_baselines: bool = True,
) -> np.ndarray:
    """Create a set of arms (action vectors in [0,1]^4)."""
    rng = np.random.default_rng(seed)
    arms: list[np.ndarray] = []

    if include_baselines:
        # Common baselines / corners
        arms.extend(
            [
                np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
                np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
                np.array([0.2, 0.4, 0.6, 0.8], dtype=float),
                np.array([0.1, 0.1, 0.1, 1.0], dtype=float),
            ]
        )

    # Random arms
    for _ in range(k_random):
        a = rng.random(4)
        arms.append(a)

    # Repair and deduplicate (coarse)
    repaired = np.vstack([_repair_action(a) for a in arms])

    # Dedup by rounding (keeps arms diverse but avoids identical ones)
    key = np.round(repaired, 3)
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    repaired = repaired[np.sort(uniq_idx)]

    return repaired


# KPI aggregation


def compute_episode_kpis(env: PortoMicromobilityEnv, total_reward: float) -> dict[str, float]:
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

    availability_tick_avg = float(arr("availability").mean())

    # ---------------- queues / wait proxies ----------------
    queue_total = arr("queue_total")
    queue_rate = arr("queue_rate")

    total_queue_time_min = float(queue_total.sum() * dt_min)
    avg_wait_min_proxy = float(total_queue_time_min / max(served_total, 1))

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

    plugged_slack_total = int(max(0, plugged_reserve_total - plugged_exec_total))
    plugged_slack_avg = float(max(0.0, plugged_reserve_avg - plugged_exec_avg))

    charge_energy_kwh_total = float(arr("charge_energy_kwh").sum())
    charge_cost_eur_total = float(arr("charge_cost_eur").sum())
    charge_util_avg = float(arr("charge_utilization").mean())

    kwh_per_served = float(charge_energy_kwh_total / max(served_total, 1))
    eur_per_served = float(charge_cost_eur_total / max(served_total, 1))

    # ---------------- SoC / feasibility ----------------
    soc_mean_avg = float(arr("soc_mean").mean())
    soc_mean_vehicles_avg = float(arr("soc_mean_vehicles").mean())

    soc_station_min = arr("soc_station_min")
    soc_station_p10 = arr("soc_station_p10")

    soc_station_min_p10_over_time = float(np.percentile(soc_station_min, 10)) if T else 0.0
    soc_station_p10_avg = float(soc_station_p10.mean())

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
    overflow_dropped_total = 0  # add to logs if/when you log it

    # ---------------- exact J recompute ----------------
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

    J_unavail = 0.0
    J_queue = 0.0
    J_reloc = 0.0
    J_charge = 0.0

    d0 = arr("demand_total")
    a = arr("availability")
    demand0_ticks = int((d0 == 0).sum())
    availability_on_d0_mean = float(a[d0 == 0].mean()) if demand0_ticks > 0 else None

    for r in logs:
        a = float(r["availability"])
        rk = float(r["reloc_km"])
        cc = float(r["charge_cost_eur"])
        qr = float(r["queue_rate"])

        J_unavail += w.alpha_unavailability * (1.0 - a)
        J_reloc += w.beta_reloc_km * rk
        J_charge += w.gamma_energy_cost * cc
        J_queue += w.delta_queue * qr

    J_sum = J_unavail + J_reloc + J_charge + J_queue

    return {
        # volumes
        "demand_total": demand_total,
        "served_total": served_total,
        "served_new_total": served_new_total,
        "backlog_served_total": backlog_served_total,
        "unmet_total": unmet_total,
        # service
        "availability_tick_avg": round(availability_tick_avg, 3),
        "availability_demand_weighted": round(float(availability_demand_weighted), 3),
        "unmet_rate": round(float(unmet_rate), 3),
        "avg_wait_min_proxy": round(float(avg_wait_min_proxy), 2),
        # queue
        "queue_total_max": int(queue_total.max()) if T else 0,
        "queue_total_avg": round(float(queue_total.mean()), 2),
        "queue_total_p95": round(queue_total_p95, 2),
        "queue_rate_avg": round(float(queue_rate.mean()), 3),
        "queue_rate_p95": round(queue_rate_p95, 3),
        # relocation
        "relocation_km_total": round(reloc_km_total, 2),
        "reloc_units_total": reloc_units_total,
        "reloc_edges_total": reloc_edges_total,
        "reloc_km_per_unit": round(reloc_km_per_unit, 3),
        "reloc_units_per_edge": round(reloc_units_per_edge, 3),
        # charging
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
        # SoC
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
        "overflow_dropped_total": overflow_dropped_total,
        "overflow_extra_min_total": round(overflow_extra_min_total, 1),
        # score consistency
        "J_run": round(float(J_run), 3),
        "J_sum": round(float(J_sum), 3),
        "J_unavailability": round(float(J_unavail), 3),
        "J_reloc_km": round(float(J_reloc), 3),
        "J_energy_cost": round(float(J_charge), 3),
        "J_queue": round(float(J_queue), 3),
        "reward_plus_J_sum": round(float(reward_plus_J_sum), 3),
        "ticks": T,
        "availability_on_d0_mean": availability_on_d0_mean,
        "demand0_ticks": demand0_ticks,
    }


# -----------------------------
# Main: train bandit over episodes
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/network_porto10.yaml")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # bandit / arms
    parser.add_argument("--arms_random", type=int, default=40, help="Number of random arms to add.")
    parser.add_argument("--arms_seed", type=int, default=0, help="Seed for arm generation.")
    parser.add_argument("--ucb_c", type=float, default=2.0, help="UCB exploration coefficient.")
    args = parser.parse_args()

    # Build discrete arm set (each arm is a 4D action vector in [0,1])
    arms = make_arm_set(k_random=args.arms_random, seed=args.arms_seed, include_baselines=True)

    bandit = UCB1Bandit(arms=arms, c=args.ucb_c, seed=args.seed)
    agent = MultiArmedBanditAgent(bandit)

    all_results = []

    for ep in range(args.episodes):
        env = PortoMicromobilityEnv(
            cfg_path=args.config,
            episode_hours=args.hours,
            seed=args.seed + ep,  # exogenous randomness changes per episode
        )

        obs = env.reset()
        done = False
        total_reward = 0.0

        chosen_action = agent.begin_episode()
        chosen_arm_idx = int(agent.last_arm_idx)

        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)

        agent.end_episode(total_reward)

        kpis = compute_episode_kpis(env, total_reward=total_reward)
        kpis["episode"] = ep
        kpis["total_reward"] = round(float(total_reward), 3)
        kpis["action"] = [round(float(x), 4) for x in chosen_action.tolist()]
        # Bandit diagnostics
        kpis["bandit_best_mean_reward"] = round(float(bandit.means[bandit.best_arm()]), 3)
        kpis["chosen_arm"] = chosen_arm_idx
        kpis["chosen_arm_mean_after_update"] = round(float(bandit.means[chosen_arm_idx]), 3)
        kpis["chosen_arm_pulls_after_update"] = int(bandit.counts[chosen_arm_idx])
        all_results.append(kpis)

    summary = {
        "n_arms": int(arms.shape[0]),
        "ucb_c": float(args.ucb_c),
        "best_arm": int(bandit.best_arm()),
        "best_arm_action": [round(float(x), 4) for x in bandit.arms[bandit.best_arm()].tolist()],
        "best_arm_mean_reward": round(float(bandit.means[bandit.best_arm()]), 3),
        "best_arm_pulls": int(bandit.counts[bandit.best_arm()]),
    }

    print(json.dumps({"summary": summary, "episodes": all_results}, indent=2))


if __name__ == "__main__":
    main()
