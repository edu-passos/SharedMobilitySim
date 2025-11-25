import numpy as np


def plan_greedy(
    x,
    C,
    travel_min,
    *,
    low=0.2,
    high=0.8,
    target=0.6,
    hysteresis=0.03,
    max_moves=50,
    distance_penalty=0.0,
):
    """
    Move vehicles from surplus to deficit stations.
    - low/high: trigger thresholds
    - target: preferred fill ratio after moves
    - hysteresis: avoids ping-pong (donor must be > high+hyst, receiver < low-hyst)
    - distance_penalty: prioritize nearer donors (add penalty * distance to the gap)
    """
    x = np.asarray(x, dtype=float).copy()
    C = np.asarray(C, dtype=float)
    N = x.size

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
            surplus = x[j] - max(high * C[j], target * C[j])
            gap = min(low * C[i], target * C[i]) - x[i]
            k = int(max(0, np.floor(min(surplus, gap))))
            if k > 0:
                plan.append((j, i, k))
                x[j] -= k
                x[i] += k
                if len(plan) >= max_moves:
                    return plan

            # stop early if station i is no longer in need
            if x[i] >= low * C[i]:
                break
    return plan


# simple strategic charging (prioritize high demand & low SoC)
def plan_charging_greedy(x, s, chargers, lam_t, *, threshold_quantile=0.5, min_score=None):
    """
    Return per-station number of vehicles to plug this tick.
    Prioritizes stations with high expected demand and low SoC:
        score[i] = lam_t[i] * (1 - s[i])
    Only stations with score above a threshold are plugged.
    Local constraints: plan[i] <= min(chargers[i], x[i]).
    """
    x = np.asarray(x)
    s = np.asarray(s)
    chargers = np.asarray(chargers)
    lam_t = np.asarray(lam_t)

    score = lam_t * (1.0 - s)  # higher = more urgent

    # choose a threshold—either absolute (min_score) or quantile
    if min_score is not None:
        thr = float(min_score)
    else:
        thr = float(np.quantile(score, threshold_quantile))  # e.g., top half

    plan = np.zeros_like(x, dtype=int)
    hot = np.where(score >= thr)[0]
    if hot.size:
        plan[hot] = np.minimum(chargers[hot], x[hot]).astype(int)
    # cold stations stay unplugged intentionally (to save energy)
    return plan


def plan_nightly_uniform(x, C, target=0.6):
    # move from >target*C to <target*C once per night (call from main when hour==2)
    return []
