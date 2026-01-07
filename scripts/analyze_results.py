import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

REPORT_KEYS_DEFAULT = [
    # objective
    "J_run",
    "J_avail_run",
    "J_reloc_run",
    "J_charge_run",
    "J_queue_run",
    # service
    "availability_demand_weighted",
    "availability_tick_avg",
    "unmet_rate",
    "unmet_total",
    "avg_wait_min_proxy",
    # queue
    "queue_total_avg",
    "queue_total_p95",
    "queue_rate_avg",
    "queue_rate_p95",
    "queue_delta_mean",
    "queue_delta_p95",
    # ops
    "relocation_km_total",
    "charging_cost_eur_total",
    "charging_energy_kwh_total",
    "charge_utilization_avg",
]


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(a))
    s = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return m, s


def _infer_method(obj: dict[str, Any], path: str) -> str:
    # Prefer explicit
    if isinstance(obj.get("summary"), dict):
        agent = obj["summary"].get("agent")
        if isinstance(agent, str) and agent.strip():
            return agent.strip()

    # Fall back to filename heuristics
    base = Path(path).name.lower()
    if "heuristic" in base:
        return "heuristic"
    if "contextual" in base or "linucb" in base or "rh" in base:
        return "bandit_contextual_rh"
    if "bandit" in base:
        return "bandit_episode"
    if "sweep" in base or "grid" in base:
        return "sweep"
    return "unknown"


def _infer_scenario(obj: dict[str, Any], path: str) -> str:
    if isinstance(obj.get("summary"), dict):
        s = obj["summary"].get("scenario")
        if isinstance(s, str) and s.strip():
            return s.strip()

    # heuristics/bandit scripts sometimes keep scenario at top-level
    s = obj.get("scenario")
    if isinstance(s, str) and s.strip():
        return s.strip()

    # filename fallback
    base = Path(path).name.lower()
    for cand in ["baseline", "hotspot_od", "hetero_lambda", "event_heavy"]:
        if cand in base:
            return cand
    return "unknown"


def _extract_episode_rows(obj: dict[str, Any]) -> list[dict[str, Any]]:
    """Returns a list of episode KPI dicts (flat).
    normalize different JSON shapes into a common list where KPI keys exist at top-level.
    """
    if not isinstance(obj, dict):
        return []

    episodes = obj.get("episodes")
    if not isinstance(episodes, list):
        return []

    # Case A: heuristic_eval / eval_policies style: each episode row already has KPI keys
    if episodes and isinstance(episodes[0], dict) and "J_run" in episodes[0]:
        return episodes  # already flat

    # Case B: contextual_rh: each episode has "episode_kpis"
    if episodes and isinstance(episodes[0], dict) and "episode_kpis" in episodes[0]:
        out: list[dict[str, Any]] = []
        for e in episodes:
            if not isinstance(e, dict):
                continue
            ek = e.get("episode_kpis", {})
            if isinstance(ek, dict) and ek:
                row = dict(ek)
                # Keep a little metadata if present
                if "seed" in e:
                    row["seed"] = e["seed"]
                out.append(row)
        return out

    # Case C: bandit_param_arms style: episodes contain kpis but may not include J_run
    # (still return; caller will handle missing keys)
    if episodes and isinstance(episodes[0], dict):
        return episodes

    return []


def _aggregate_kpis(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = [_safe_float(r.get(k)) for r in rows]
        mean, std = _mean_std(vals)
        out[k] = {"mean": mean, "std": std}
    return out


def _format_mean_std(m: float, s: float, *, nd: int = 3) -> str:
    if not np.isfinite(m):
        return "NA"
    if not np.isfinite(s):
        s = 0.0
    return f"{m:.{nd}f}±{s:.{nd}f}"


def _to_csv(table: list[dict[str, Any]], out_path: Path) -> None:
    if not table:
        return
    cols = list(table[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        f.writelines(",".join(str(row.get(c, "")) for c in cols) + "\n" for row in table)


def _to_md(table: list[dict[str, Any]], out_path: Path) -> None:
    if not table:
        return
    cols = list(table[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        f.writelines("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |\n" for row in table)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", default=[], help="JSON files or globs (e.g., out/*.json).")
    p.add_argument("--out_csv", default="out/summary_table.csv")
    p.add_argument("--out_md", default="out/summary_table.md")
    p.add_argument("--report_keys", nargs="*", default=REPORT_KEYS_DEFAULT)
    p.add_argument("--rank_by", default="J_run", help="KPI key to rank by (lower is better for J_run).")
    args = p.parse_args()

    # Resolve inputs
    files: list[str] = []
    if args.inputs:
        for x in args.inputs:
            files.extend(glob.glob(x))
    else:
        files = glob.glob("out/*.json")

    files = sorted(dict.fromkeys(files))  # dedupe, stable
    if not files:
        print("No input JSON files found. Use --inputs out/*.json")
        return

    rows_out: list[dict[str, Any]] = []
    per_entry: list[tuple[str, str, str, dict[str, dict[str, float]]]] = []

    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue

        method = _infer_method(obj, path)
        scenario = _infer_scenario(obj, path)
        ep_rows = _extract_episode_rows(obj)

        if not ep_rows:
            print(f"[skip] {path}: no episode rows recognized")
            continue

        agg = _aggregate_kpis(ep_rows, list(args.report_keys))
        per_entry.append((scenario, method, path, agg))

        # Build a compact one-row table with formatted mean±std
        row = {
            "scenario": scenario,
            "method": method,
            "file": Path(path).name,
            "n_episodes": len(ep_rows),
        }
        for k in args.report_keys:
            ms = agg.get(k, {"mean": float("nan"), "std": float("nan")})
            row[k] = _format_mean_std(ms["mean"], ms["std"], nd=3)
        rows_out.append(row)

    # Sort by scenario then rank_by (numeric mean)
    rank_key = str(args.rank_by)

    def sort_key(r: dict[str, Any]) -> tuple[str, float]:
        scen = str(r.get("scenario", ""))
        # parse "m±s"
        v = r.get(rank_key, "NA")
        try:
            m_str = str(v).split("±")[0]
            m = float(m_str)
        except Exception:
            m = float("inf")
        return (scen, m)

    rows_out.sort(key=sort_key)

    _to_csv(rows_out, Path(args.out_csv))
    _to_md(rows_out, Path(args.out_md))

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_md}")

    # Console: top-3 per scenario by rank_by mean (lower is better for J_run)
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for r in rows_out:
        by_scenario.setdefault(str(r["scenario"]), []).append(r)

    print(f"\n=== Top methods per scenario (ranked by mean of {rank_key}) ===")
    for scen, lst in by_scenario.items():
        # re-rank within scenario
        lst2 = sorted(lst, key=lambda rr: sort_key(rr)[1])
        print(f"\n[{scen}]")
        for rr in lst2[:3]:
            print(f"- {rr['method']}: {rank_key}={rr.get(rank_key)} (n={rr['n_episodes']}, file={rr['file']})")


if __name__ == "__main__":
    main()
