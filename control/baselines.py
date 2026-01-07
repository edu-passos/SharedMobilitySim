import numpy as np


def plan_relocation_greedy(
    x: np.ndarray,
    C: np.ndarray,
    cost_km: np.ndarray,
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
        cost_km: matrix of travel costs (km) between stations
        low: lower station fill ratio threshold (trigger threshold) - stations below this need vehicles
        high: upper station fill ratio threshold (trigger threshold) - stations above this can donate vehicles
        target: preferred fill ratio after moves
        hysteresis: avoids ping-pong (donor must be > high+hyst, receiver < low-hyst)
        max_moves: limit number of moves per planning call
        distance_penalty: extra weight on distance/cost (0.0 = pure cost_km ordering)

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
        donors = sorted(have, key=lambda j: (cost_km[j, i] * (1.0 + distance_penalty)))
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
    keep_frac_rentable: float = 0.5,
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
        charge_budget_frac: fraction of total available charging capacity to use
        min_score: alternative absolute minimum score threshold (overrides quantile if set)
        keep_min_rentable: minimum vehicles to keep rentable at each station
        keep_frac_rentable: fraction of vehicles to keep rentable at each station

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


def plan_relocation_budgeted(
    x: np.ndarray,
    C: np.ndarray,
    cost_km: np.ndarray,
    *,
    low: float = 0.2,
    high: float = 0.8,
    target: float = 0.6,
    hysteresis: float = 0.03,
    max_moves: int = 50,
    km_budget: float = 25.0,
) -> list[tuple[int, int, int]]:
    """Greedy relocation with a hard per-call km budget.

    This provides a controlled 'reloc-lite' baseline: it only moves vehicles as long
    as the implied relocation cost (move * cost_km[i,j]) stays within km_budget.

    Args:
        x, C: counts and capacities
        cost_km: relocation cost/distance matrix used to pick donors
        km_budget: max total relocation 'km' per planning call (per tick)

    Returns:
        List of (from_station, to_station, num_vehicles) moves.
    """
    x = np.asarray(x, dtype=float).copy()
    C = np.asarray(C, dtype=float)
    target = float(np.clip(target, low + 0.05, high - 0.05))

    need = np.where(x < (low - hysteresis) * C)[0]
    have = np.where(x > (high + hysteresis) * C)[0]

    plan: list[tuple[int, int, int]] = []
    if need.size == 0 or have.size == 0:
        return plan

    km_used = 0.0

    for i in need:
        donors = sorted(have, key=lambda j: float(cost_km[j, i]))
        for j in donors:
            if len(plan) >= max_moves:
                return plan

            # target-balanced move amount (same as your greedy)
            surplus = x[j] - target * C[j]
            gap = target * C[i] - x[i]
            k = int(max(0, np.floor(min(surplus, gap))))
            if k <= 0:
                continue

            # budget check: may need to reduce k
            c = float(cost_km[j, i])
            if c <= 0.0:
                # allow free move (rare if diagonal), but still bounded by k
                k_budgeted = k
            else:
                remaining = km_budget - km_used
                if remaining <= 0.0:
                    return plan
                k_budgeted = int(min(k, np.floor(remaining / c)))

            if k_budgeted <= 0:
                continue

            plan.append((int(j), int(i), int(k_budgeted)))
            x[j] -= k_budgeted
            x[i] += k_budgeted
            km_used += float(k_budgeted) * c

            if x[i] >= target * C[i]:
                break

    return plan


def plan_charging_slack(
    x: np.ndarray,
    s: np.ndarray,
    chargers: np.ndarray,
    lam_t: np.ndarray,
    *,
    charge_budget_frac: float = 1.0,
    keep_min_rentable: int = 1,
    keep_frac_rentable: float = 0.5,
    # slack gating (proxy): only charge stations whose demand is below a threshold
    lam_charge_quantile: float = 0.5,  # charge only bottom 50% demand by default
    min_score: float | None = None,
) -> np.ndarray:
    """Charge only where it's 'slack' (low predicted demand), prioritizing low SoC.

    This reduces the chance charging harms availability when reserve_plugged=True.

    score[i] = (1 - s[i]) * slack_mask[i]
    slack_mask[i] = 1 if lam_t[i] <= quantile(lam_t, lam_charge_quantile) else 0

    Returns:
        Per-station plugs (int array).
    """
    x = np.asarray(x)
    s = np.asarray(s)
    chargers = np.asarray(chargers)
    lam_t = np.asarray(lam_t)

    cap0 = np.minimum(chargers, x).astype(int)
    total_cap0 = int(cap0.sum())
    if total_cap0 == 0:
        return np.zeros_like(x, dtype=int)

    budget = int(round(np.clip(charge_budget_frac, 0.0, 1.0) * total_cap0))
    if budget <= 0:
        return np.zeros_like(x, dtype=int)

    # slack gating via demand quantile
    q = float(np.clip(lam_charge_quantile, 0.0, 1.0))
    lam_thr = float(np.quantile(lam_t, q))
    slack_mask = (lam_t <= lam_thr).astype(float)

    # prioritize low SoC but only in slack stations
    score = (1.0 - s) * slack_mask
    if min_score is not None:
        score = np.where(score >= min_score, score, 0.0)

    keep = np.maximum(keep_min_rentable, np.ceil(keep_frac_rentable * x)).astype(int)
    cap = np.minimum(chargers, np.maximum(0, x - keep)).astype(int)

    order = np.argsort(-score)
    plan = np.zeros_like(x, dtype=int)
    remaining = budget

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
