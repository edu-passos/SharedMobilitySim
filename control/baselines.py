import numpy as np


def plan_relocation_greedy(
    x: np.ndarray,
    C: np.ndarray,
    travel_min: np.ndarray,
    *,
    low: float = 0.2,
    high: float = 0.8,
    target: float = 0.6,
    hysteresis: float = 0.03,
    max_moves: int = 50,
    distance_penalty: float = 0.0,
) -> list[tuple[int, int, int]]:
    """Move vehicles from surplus (donor) to deficit (receiver) stations.

    Args:
        x: current vehicle counts
        C: station capacities
        travel_min: min travel times between stations
        low: lower station fill ratio threshold (trigger threshold) - stations below this need vehicles
        high: upper station fill ratio threshold (trigger threshold) - stations above this can donate vehicles
        target: preferred fill ratio after moves
        hysteresis: avoids ping-pong (donor must be > high+hyst, receiver < low-hyst)
        max_moves: limit number of moves per planning call
        distance_penalty: prioritize nearer donors (add penalty * distance to the gap)

    Returns:
        List of (from_station, to_station, num_vehicles) moves
    """
    x = np.asarray(x, dtype=float).copy()
    C = np.asarray(C, dtype=float)
    target = np.clip(target, low + 0.05, high - 0.05)


    # indices that truly need/donate after hysteresis
    need = np.where(x < (low - hysteresis) * C)[0]
    have = np.where(x > (high + hysteresis) * C)[0]

    plan = []
    if need.size == 0 or have.size == 0:
        return plan

    for i in need:
        # donors ordered by "effective cost": distance + small penalty
        donors = sorted(have, key=lambda j: (travel_min[j, i] * (1.0 + distance_penalty)))
        for j in donors:
            # donor surplus above target; receiver gap up to target
            surplus = x[j] - target * C[j]
            gap = target * C[i] - x[i]
            k = int(max(0, np.floor(min(surplus, gap))))
            if k > 0:
                plan.append((j, i, k))
                x[j] -= k
                x[i] += k
                if len(plan) >= max_moves:
                    return plan

            # stop early if station i is no longer in need
            if x[i] >= target * C[i]:
                break
    return plan


# simple strategic charging (prioritize high demand & low SoC)
def plan_charging_greedy(
    x: np.ndarray,
    s: np.ndarray,
    chargers: np.ndarray,
    lam_t: np.ndarray,
    *,
    charge_budget_frac: float = 1.0,
    min_score: float | None = None,
    keep_min_rentable: int = 1,
    keep_frac_rentable: float = 0.5
) -> np.ndarray:
    """Plan charging vehicles at stations based on expected demand and SoC.

    Prioritizes stations with high expected demand and low SoC:
        score[i] = lam_t[i] * (1 - s[i])
    Only stations with score above a threshold are plugged.
    Local constraints: plan[i] <= min(chargers[i], x[i]).

    Args:
        x: current vehicle counts
        s: current average SoC per station
        chargers: number of available chargers per station
        lam_t: expected demand rate per station this tick
        threshold_quantile: quantile for score threshold (e.g., 0.5 means top-50% stations get charged)
        min_score: alternative absolute minimum score threshold (overrides quantile if set)

    Returns:
        Per-station number of vehicles to plug this tick.

    """
    x = np.asarray(x)
    s = np.asarray(s)
    chargers = np.asarray(chargers)
    lam_t = np.asarray(lam_t)

    score = lam_t * (1.0 - s)  # higher = more urgent
    if min_score is not None:
        score = np.where(score >= min_score, score, 0.0) 

    cap = np.minimum(chargers, x).astype(int)
    total_cap = int(cap.sum())
    if total_cap == 0:
        return np.zeros_like(x, dtype=int)
    
    # Budget = fraction of available charging capacity
    budget = int(round(np.clip(charge_budget_frac, 0.0, 1.0) * total_cap))
    if budget <= 0:
        return np.zeros_like(x, dtype=int)

    # Allocate plugs to stations by descending score
    order = np.argsort(-score)
    plan = np.zeros_like(x, dtype=int)
    remaining = budget
    keep = np.maximum(keep_min_rentable, np.ceil(keep_frac_rentable * x)).astype(int)
    cap = np.minimum(chargers, np.maximum(0, x - keep)).astype(int)
    for i in order:
        if remaining <= 0:
            break
        if score[i] <= 0:
            break
        take = min(int(cap[i]), remaining)
        if take > 0:
            plan[i] = take
            remaining -= take
    return plan


def plan_nightly_uniform(x: np.ndarray, C: np.ndarray, target: float = 0.6) -> list[tuple[int, int, int]]:
    # move from >target*C to <target*C once per night (call from main when hour==2)
    return []
