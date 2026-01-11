#!/usr/bin/env python3
"""
analyze_results.py

Aggregate evaluation JSONs into comparable CSV tables across mixed schemas.

Supported schemas:

1) Standard single-policy eval (heuristic/SAC/etc.)
   - obj["summary"][k] = {"mean","std",...}
   - OR obj["summary"][f"{k}_mean"] / obj["summary"][f"{k}_std"]
   - OR obj[f"{k}_mean"] / obj[f"{k}_std"]

2) LinUCB RH contextual bandit
   - obj["episodes"][i]["episode_kpis"][k] exists
   - aggregate across episodes (optionally last N)

3) Multi-policy sweep outputs (baselines, bandit_ucb1_fixed)
   - obj["summaries"] is a dict: policy_name -> { k -> {"mean","std",...} }
   - produce one result row per policy_name

Key improvements:
- Scenario inferred primarily from filename prefix (handles *fromBaseline* files).
- Method includes directory to avoid collisions, and for multi-policy files includes policy name.
- --debug prints why a file is skipped or missing metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCENARIOS = ["baseline", "event_heavy", "hetero_lambda", "hotspot_od"]

CORE_KEYS = [
    "J_run",
    "unmet_rate",
    "availability_demand_weighted",
    "availability_tick_avg",
    "relocation_km_total",
    "charging_cost_eur_total",
    "charge_utilization_avg",
    "J_avail_run",
    "J_reloc_run",
    "J_charge_run",
    "J_queue_run",
]


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _scenario_from_filename(path: Path) -> str | None:
    """
    Strong rule: your files start with scenario token.
    Avoid matching 'baseline' inside 'fromBaseline'.
    """
    name = path.name.lower()

    for s in SCENARIOS:
        if name.startswith(s + "__") or name.startswith(s + "_") or name.startswith(s + "."):
            return s

    for s in SCENARIOS:
        if re.search(rf"(^|[^a-z0-9]){re.escape(s)}([^a-z0-9]|$)", name):
            return s

    return None


def _infer_scenario(obj: dict[str, Any], path: Path, *, debug: bool = False) -> str:
    scen_file = _scenario_from_filename(path)
    if scen_file:
        return scen_file

    if isinstance(obj.get("scenario"), str):
        return str(obj["scenario"])
    if isinstance(obj.get("summary"), dict) and isinstance(obj["summary"].get("scenario"), str):
        return str(obj["summary"]["scenario"])

    if debug:
        print(f"[debug] could not infer scenario: {path}")
    return "unknown"


def _is_training_only_bandit(obj: dict[str, Any]) -> bool:
    # Skip known “tuning logs”
    summ = obj.get("summary")
    if isinstance(summ, dict) and "best_arm_mean_neg_J_run" in summ:
        return True

    eps = obj.get("episodes")
    if isinstance(eps, list) and eps:
        row0 = eps[0]
        if isinstance(row0, dict) and ("chosen_arm_idx" in row0) and ("episode" in row0):
            return True

    return False


def _detect_linucb(obj: dict[str, Any]) -> bool:
    summ = obj.get("summary")
    if isinstance(summ, dict) and ("linucb_alpha" in summ or "bandit_arm_pulls" in summ):
        return True
    eps = obj.get("episodes")
    if isinstance(eps, list) and eps and isinstance(eps[0], dict) and "episode_kpis" in eps[0]:
        return True
    return False


def _infer_method_single(obj: dict[str, Any], path: Path) -> str:
    """
    For single-policy files.
    Prefer explicit agent/method fields; otherwise include directory to avoid collisions.
    """
    for key in ["agent", "method", "policy", "name"]:
        if isinstance(obj.get(key), str) and obj.get(key).strip():
            val = str(obj[key]).strip()
            if val == "heuristic_tick_level":
                return "heuristic:tick_level"
            return val

    # LinUCB readable name
    summ = obj.get("summary")
    if isinstance(summ, dict) and ("linucb_alpha" in summ or "bandit_arm_pulls" in summ):
        blk = summ.get("block_minutes", "na")
        a = summ.get("linucb_alpha", "na")
        reg = summ.get("linucb_reg", "na")
        return f"{path.parent.name}:linucb_rh__blk{blk}__a{a}__reg{reg}"

    return f"{path.parent.name}:{path.stem}"


def _extract_standard_summary_anywhere(obj: dict[str, Any]) -> dict[str, float]:
    """
    Extract from:
    - obj["summary"][k] = {mean,std}
    - obj["summary"][f"{k}_mean"]
    - obj[f"{k}_mean"]
    """
    out: dict[str, float] = {}

    def grab(container: dict[str, Any]) -> None:
        for k in CORE_KEYS:
            # nested
            if isinstance(container.get(k), dict):
                out[f"{k}_mean"] = _safe_float(container[k].get("mean"))
                out[f"{k}_std"] = _safe_float(container[k].get("std", 0.0))
                continue
            # flat
            mk = f"{k}_mean"
            if mk in container:
                out[mk] = _safe_float(container.get(mk))
                out[f"{k}_std"] = _safe_float(container.get(f"{k}_std", 0.0))

    summ = obj.get("summary")
    if isinstance(summ, dict):
        grab(summ)
    if isinstance(obj, dict):
        grab(obj)

    return out


def _extract_linucb_from_episode_kpis(
    obj: dict[str, Any],
    *,
    last_n_episodes: int | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    eps = obj.get("episodes", [])
    if not isinstance(eps, list):
        return {}, []

    rows: list[dict[str, Any]] = []
    for i, e in enumerate(eps):
        if not isinstance(e, dict):
            continue
        kpis = e.get("episode_kpis")
        if not isinstance(kpis, dict):
            continue

        row: dict[str, Any] = {"episode": i}
        if "seed" in e:
            row["seed"] = e["seed"]
        for k in CORE_KEYS:
            if k in kpis:
                row[k] = kpis[k]
        rows.append(row)

    if last_n_episodes is not None and last_n_episodes > 0 and len(rows) > last_n_episodes:
        rows = rows[-last_n_episodes:]

    out: dict[str, float] = {}
    for k in CORE_KEYS:
        vals = [_safe_float(r.get(k)) for r in rows]
        m, s = _mean_std(vals)
        out[f"{k}_mean"] = m
        out[f"{k}_std"] = s

    return out, rows


def _extract_multi_policy_summaries(
    obj: dict[str, Any],
) -> list[tuple[str, dict[str, float]]]:
    """
    For files with obj["summaries"][policy_name][k] = {mean,std,...}
    Returns: [(policy_name, summary_metrics_dict), ...]
    """
    summaries = obj.get("summaries")
    if not isinstance(summaries, dict):
        return []

    out_rows: list[tuple[str, dict[str, float]]] = []
    for policy_name, pol_summ in summaries.items():
        if not isinstance(policy_name, str) or not isinstance(pol_summ, dict):
            continue

        metrics: dict[str, float] = {}
        for k in CORE_KEYS:
            if isinstance(pol_summ.get(k), dict):
                metrics[f"{k}_mean"] = _safe_float(pol_summ[k].get("mean"))
                metrics[f"{k}_std"] = _safe_float(pol_summ[k].get("std", 0.0))

        if metrics:
            out_rows.append((policy_name, metrics))

    return out_rows


@dataclass
class ParsedRow:
    path: Path
    scenario: str
    method: str
    summary_metrics: dict[str, float]
    episodes_rows: list[dict[str, Any]]


def parse_one(path: Path, *, last_n_linucb: int | None, debug: bool = False) -> list[ParsedRow]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as ex:
        if debug:
            print(f"[debug] failed to read json: {path} ({ex})")
        return []

    if not isinstance(obj, dict):
        if debug:
            print(f"[debug] not a dict json: {path}")
        return []

    if _is_training_only_bandit(obj):
        if debug:
            print(f"[debug] skipped training-only bandit log: {path}")
        return []

    scenario = _infer_scenario(obj, path, debug=debug)

    # Case 3: multi-policy summaries (baselines, bandit_ucb1_fixed)
    multi = _extract_multi_policy_summaries(obj)
    if multi:
        rows: list[ParsedRow] = []
        for policy_name, metrics in multi:
            # method must include directory + policy name to be unique and readable
            method = f"{path.parent.name}:{policy_name}"
            rows.append(ParsedRow(path=path, scenario=scenario, method=method, summary_metrics=metrics, episodes_rows=[]))
        return rows

    # Case 2: LinUCB RH
    if _detect_linucb(obj):
        method = _infer_method_single(obj, path)
        metrics, eps_rows = _extract_linucb_from_episode_kpis(obj, last_n_episodes=last_n_linucb)
        if not metrics and debug:
            print(f"[debug] linucb-like but no episode_kpis extracted: {path}")
        return [ParsedRow(path=path, scenario=scenario, method=method, summary_metrics=metrics, episodes_rows=eps_rows)]

    # Case 1: standard single-policy
    method = _infer_method_single(obj, path)
    metrics = _extract_standard_summary_anywhere(obj)
    if not metrics and debug:
        print(f"[debug] no summary metrics found (schema mismatch): {path}")
    return [ParsedRow(path=path, scenario=scenario, method=method, summary_metrics=metrics, episodes_rows=[])]


def iter_json_files(roots: list[Path], include: str | None, exclude: str | None) -> Iterable[Path]:
    inc_re = re.compile(include) if include else None
    exc_re = re.compile(exclude) if exclude else None

    for root in roots:
        if root.is_file() and root.suffix.lower() == ".json":
            paths = [root]
        else:
            paths = sorted(root.rglob("*.json"))

        for p in paths:
            s = p.as_posix()
            if inc_re and not inc_re.search(s):
                continue
            if exc_re and exc_re.search(s):
                continue
            yield p


def _dedup(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    strategy = str(strategy).lower()
    if strategy == "none":
        return df

    df = df.copy()

    if strategy == "best":
        df["__J"] = pd.to_numeric(df.get("J_run_mean", np.nan), errors="coerce").fillna(np.inf)
        df = df.sort_values(["scenario", "method", "__J", "path"], ascending=[True, True, True, True])
        df = df.drop_duplicates(subset=["scenario", "method"], keep="first").drop(columns=["__J"])
        return df

    if strategy == "latest":
        df["__mtime"] = df["path"].apply(lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0.0)
        df = df.sort_values(["scenario", "method", "__mtime"], ascending=[True, True, False])
        df = df.drop_duplicates(subset=["scenario", "method"], keep="first").drop(columns=["__mtime"])
        return df

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", required=True, help="One or more JSON files or directories to scan.")
    ap.add_argument("--include", default=None, help="Regex include filter on full path (optional).")
    ap.add_argument("--exclude", default=None, help="Regex exclude filter on full path (optional).")

    ap.add_argument("--out_csv", default="out/eval/all_results_summary.csv")
    ap.add_argument("--out_episodes_csv", default="", help="If set, write episode-level CSV here.")

    ap.add_argument("--linucb_last_n", type=int, default=0, help="If >0 use last N episodes for LinUCB aggregation.")
    ap.add_argument("--dedup", default="best", choices=["none", "best", "latest"])
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    roots = [Path(r) for r in args.root]
    last_n_linucb = int(args.linucb_last_n) if int(args.linucb_last_n) > 0 else None

    parsed_rows: list[ParsedRow] = []
    for p in iter_json_files(roots, args.include, args.exclude):
        parsed_rows.extend(parse_one(p, last_n_linucb=last_n_linucb, debug=bool(args.debug)))

    # Keep only rows with metrics
    parsed_rows = [r for r in parsed_rows if r.summary_metrics]
    if not parsed_rows:
        raise SystemExit("No comparable eval JSONs found. Re-run with --debug to see why files are skipped.")

    rows = []
    for r in parsed_rows:
        row = {"scenario": r.scenario, "method": r.method, "path": r.path.as_posix()}
        row.update(r.summary_metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure columns exist for plotting scripts
    keep_cols = ["scenario", "method"]

    SLIDE_KEYS = [
        "J_run",
        "unmet_rate",
        "relocation_km_total",
        "charging_cost_eur_total",
        "availability_demand_weighted",
    ]
    DECOMP_KEYS = ["J_avail_run", "J_reloc_run", "J_charge_run", "J_queue_run"]

    for k in (SLIDE_KEYS + DECOMP_KEYS):
        keep_cols += [f"{k}_mean", f"{k}_std"]
    keep_cols += ["path"]

    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[keep_cols]

    # Dedup
    df = _dedup(df, args.dedup)

    # Sort
    df["J_run_mean"] = pd.to_numeric(df["J_run_mean"], errors="coerce")
    df = df.sort_values(["scenario", "J_run_mean", "method"], ascending=[True, True, True])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved summary CSV: {out_csv}")

    # Print quick table per scenario
    for scen in df["scenario"].dropna().unique():
        sub = df[df["scenario"] == scen].copy()
        print(f"\nScenario: {scen}")
        show = sub[["method", "J_run_mean", "unmet_rate_mean", "relocation_km_total_mean", "charging_cost_eur_total_mean"]]
        with pd.option_context("display.max_rows", 500, "display.width", 220):
            print(show.to_string(index=False))


if __name__ == "__main__":
    main()
