"""analyze_paper_results.py

Scan out/paper/**.json and aggregate results across:
- scripts.eval_policy.py (summaries per policy)
- ml.heuristic.py (summary)
- ml.sac.py (summary)
- ml.bandit_contextual_rh.py (summary + episodes)
- ml.bandit_param_arms.py (summary + episodes)

Outputs (default to out/analysis):
- master_runs.csv
- robustness.csv (delta vs baseline per method/network/phase)
- plots/*.png:
    - jrun_table_<phase>_<network>.png
    - stacked_objective_<phase>_<network>.png
    - bandit_arm_pulls_<phase>_<network>_<scenario>_<method>.png (if data exists)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# pandas is optional; script works without it (CSV writing falls back)
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
DEFAULT_METRICS = [
    "J_run",
    "J_avail_run",
    "J_reloc_run",
    "J_charge_run",
    "J_queue_run",
    "availability_demand_weighted",
    "availability_tick_avg",
    "unmet_rate",
    "queue_total_p95",
    "dq_p95",
    "relocation_km_total",
    "charging_cost_eur_total",
]


# -----------------------------
# Small utilities
# -----------------------------
def _isfinite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _mean_std(values: list[float]) -> tuple[float, float]:
    v = np.array([float(x) for x in values if _isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), 0.0
    return float(np.mean(v)), float(np.std(v, ddof=1))


def _safe_get(d: dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def _try_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# -----------------------------
# Path parsing
# -----------------------------
@dataclass
class PathMeta:
    phase: str
    network: str
    method_dir: str  # folder name under network
    filename: str


def parse_path_meta(root: Path, file_path: Path) -> PathMeta:
    rel = file_path.relative_to(root)
    parts = rel.parts
    # Expected: <phase>/<network>/<method_dir>/<file>.json
    # If not, degrade gracefully.
    phase = parts[0] if len(parts) >= 1 else "unknown_phase"
    network = parts[1] if len(parts) >= 2 else "unknown_network"
    method_dir = parts[2] if len(parts) >= 3 else "unknown_method"
    return PathMeta(phase=phase, network=network, method_dir=method_dir, filename=file_path.name)


# -----------------------------
# Schema detection + parsing
# -----------------------------
def detect_schema(obj: dict[str, Any]) -> str:
    if "summaries" in obj and "policies" in obj:
        return "eval_policy"
    if obj.get("agent") == "heuristic_tick_level" or ("agent" in obj and "heuristic" in str(obj.get("agent", ""))):
        return "heuristic"
    if obj.get("agent") == "sac":
        return "sac"
    if "summary" in obj and "episodes" in obj:
        # Both bandits and other scripts can have summary+episodes; disambiguate by fields
        summ = obj.get("summary", {})
        if isinstance(summ, dict) and ("bandit_arm_pulls" in summ or "linucb_alpha" in summ or "ucb_c" in summ):
            # likely a bandit
            if "linucb_alpha" in summ or "warmup_blocks" in summ:
                return "bandit_linucb_rh"
            if "ucb_c" in summ or "best_arm_idx" in summ:
                return "bandit_ucb1_episode"
        # fallback
        return "summary_episodes_generic"
    # last resort
    return "unknown"


def extract_common_fields(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": _safe_get(obj, "config", ""),
        "hours": _safe_get(obj, "hours", float("nan")),
        "seed0": _safe_get(obj, "seed0", float("nan")),
        "seeds": _safe_get(obj, "seeds", float("nan")),
        "scenario": _safe_get(obj, "scenario", ""),
        "scenario_params": _safe_get(obj, "scenario_params", {}),
    }


def parse_eval_policy(obj: dict[str, Any], pm: PathMeta, metrics: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common = extract_common_fields(obj)

    summaries = obj.get("summaries", {})
    if not isinstance(summaries, dict):
        return rows

    for policy_name, statmap in summaries.items():
        if not isinstance(statmap, dict):
            continue

        row = {
            "phase": pm.phase,
            "network": pm.network,
            "method_family": "baselines",
            "method": str(policy_name),
            "method_dir": pm.method_dir,
            "file": pm.filename,
            **common,
        }

        for m in metrics:
            ms = statmap.get(m, None)
            if isinstance(ms, dict):
                row[f"{m}_mean"] = _try_float(ms.get("mean"))
                row[f"{m}_std"] = _try_float(ms.get("std"))
            else:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")

        rows.append(row)

    return rows


def parse_agent_summary(
    obj: dict[str, Any], pm: PathMeta, metrics: list[str], family: str, method_name: str
) -> list[dict[str, Any]]:
    common = extract_common_fields(obj)
    summ = obj.get("summary", {})
    row = {
        "phase": pm.phase,
        "network": pm.network,
        "method_family": family,
        "method": method_name,
        "method_dir": pm.method_dir,
        "file": pm.filename,
        **common,
    }

    for m in metrics:
        ms = summ.get(m, None) if isinstance(summ, dict) else None
        if isinstance(ms, dict):
            row[f"{m}_mean"] = _try_float(ms.get("mean"))
            row[f"{m}_std"] = _try_float(ms.get("std"))
        else:
            # sometimes bandit summary uses J_run_mean naming; handle elsewhere
            row[f"{m}_mean"] = float("nan")
            row[f"{m}_std"] = float("nan")

    return [row]


def parse_bandit_from_episodes(
    obj: dict[str, Any],
    pm: PathMeta,
    metrics: list[str],
    family: str,
    method_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Returns:
    - rows (one row, aggregated over episodes)
    - extra_bandit_info (for arm plots)
    """
    common = {}
    episodes = obj.get("episodes", [])
    scenario_fallback = ""
    scenario_params_fallback = {}

    if isinstance(episodes, list) and len(episodes) and isinstance(episodes[0], dict):
        scenario_fallback = episodes[0].get("scenario", "") or ""
        scenario_params_fallback = episodes[0].get("scenario_params", {}) or {}
    # bandit scripts often store config/hours/scenario in summary
    summ = obj.get("summary", {})
    if isinstance(summ, dict):
        common = {
            "config": summ.get("config", ""),
            "hours": summ.get("hours", float("nan")),
            "seed0": summ.get("seed0", float("nan")),
            "seeds": summ.get("seeds", summ.get("episodes", summ.get("testing_seeds", float("nan")))),
            "scenario": (summ.get("scenario") if isinstance(summ, dict) else "") or obj.get("scenario", "") or scenario_fallback,
            "scenario_params": (summ.get("scenario_params") if isinstance(summ, dict) else {})
            or obj.get("scenario_params", {})
            or scenario_params_fallback,
        }
    else:
        common = extract_common_fields(obj)

    vals: dict[str, list[float]] = {m: [] for m in metrics}

    # Episodes schema differs:
    # - linucb_rh: episode_kpis nested in each episode
    # - ucb1: KPIs at top-level rows (J_run, etc.)
    for ep in episodes if isinstance(episodes, list) else []:
        if not isinstance(ep, dict):
            continue
        ep_kpis = ep.get("episode_kpis", None)
        if isinstance(ep_kpis, dict):
            src = ep_kpis
        else:
            src = ep  # may contain KPI keys directly
        for m in metrics:
            v = src.get(m, None)
            if _isfinite(v):
                vals[m].append(float(v))

    row = {
        "phase": pm.phase,
        "network": pm.network,
        "method_family": family,
        "method": method_name,
        "method_dir": pm.method_dir,
        "file": pm.filename,
        **common,
    }

    for m in metrics:
        mu, sd = _mean_std(vals[m])
        # if missing, try alternate summary naming for J_run
        if (not _isfinite(mu)) and isinstance(summ, dict) and m == "J_run":
            # bandit_contextual_rh summary uses J_run_mean/J_run_std
            mu2 = summ.get("J_run_mean", None)
            sd2 = summ.get("J_run_std", None)
            mu = float(mu2) if _isfinite(mu2) else mu
            sd = float(sd2) if _isfinite(sd2) else sd
        row[f"{m}_mean"] = mu
        row[f"{m}_std"] = sd

    # Extra info for arm plots
    extra: dict[str, Any] = {"arm_counts": None, "arms": None}
    if isinstance(summ, dict):
        if "bandit_arm_pulls" in summ:
            extra["arm_counts"] = summ.get("bandit_arm_pulls")
            extra["arms"] = summ.get("arms")
        else:
            # ucb1: reconstruct arm counts from episodes if possible
            chosen = []
            for ep in episodes if isinstance(episodes, list) else []:
                if isinstance(ep, dict) and "chosen_arm_idx" in ep:
                    chosen.append(int(ep["chosen_arm_idx"]))
            if chosen:
                K = max(chosen) + 1
                counts = [0] * K
                for i in chosen:
                    if 0 <= i < K:
                        counts[i] += 1
                extra["arm_counts"] = counts
                extra["arms"] = summ.get("arms")

    return [row], extra


def parse_file(file_path: Path, root: Path, metrics: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pm = parse_path_meta(root, file_path)
    try:
        obj = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return [], {}

    schema = detect_schema(obj)

    # Identify method name more consistently
    method_name = pm.method_dir
    if schema in ("heuristic", "sac"):
        method_name = str(obj.get("agent", pm.method_dir))
    elif schema == "eval_policy":
        # method_name varies per-policy; handled there
        pass
    else:
        # bandits: method_dir likely "bandit_linucb" / "bandit_ucb1" etc
        method_name = pm.method_dir

    if schema == "eval_policy":
        return parse_eval_policy(obj, pm, metrics), {}
    if schema == "heuristic":
        return parse_agent_summary(obj, pm, metrics, family="heuristic", method_name="heuristic_tick_level"), {}
    if schema == "sac":
        return parse_agent_summary(obj, pm, metrics, family="sac", method_name="sac"), {}
    if schema == "bandit_linucb_rh":
        return parse_bandit_from_episodes(obj, pm, metrics, family="bandit", method_name="bandit_linucb_rh")
    if schema == "bandit_ucb1_episode":
        return parse_bandit_from_episodes(obj, pm, metrics, family="bandit", method_name="bandit_ucb1_episode")
    if schema == "summary_episodes_generic":
        # treat as generic bandit-like
        return parse_bandit_from_episodes(obj, pm, metrics, family="unknown", method_name=method_name)

    # unknown schema: skip
    return [], {}


# -----------------------------
# Plotting
# -----------------------------
def save_jrun_table(df, out_png: Path, title: str) -> None:
    # Aggregate duplicates: mean over any repeated (method, scenario)
    sub = df.copy()
    sub["J_run_mean"] = sub["J_run_mean"].astype(float)
    sub = sub[np.isfinite(sub["J_run_mean"])]

    if sub.empty:
        return

    agg = sub.groupby(["method", "scenario"], as_index=False)["J_run_mean"].mean()

    methods = sorted(agg["method"].unique())
    scenarios = sorted(agg["scenario"].unique())

    mat = []
    for m in methods:
        row = []
        for s in scenarios:
            v = agg[(agg["method"] == m) & (agg["scenario"] == s)]["J_run_mean"]
            row.append(float(v.iloc[0]) if len(v) else float("nan"))
        mat.append(row)

    fig = plt.figure(figsize=(max(8, 1.2 * len(scenarios)), max(3.5, 0.35 * len(methods) + 2.0)))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title)

    cell_text = [[("" if not np.isfinite(x) else f"{x:.3f}") for x in row] for row in mat]
    table = ax.table(
        cellText=cell_text,
        rowLabels=methods,
        colLabels=scenarios,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_stacked_objective(df, out_png: Path, title: str) -> None:
    comps = ["J_avail_run", "J_reloc_run", "J_charge_run", "J_queue_run"]

    sub = df.copy()

    # Use the *_mean columns as the plotting values
    for c in comps + ["J_run"]:
        sub[c] = sub.get(f"{c}_mean", np.nan)

    # Keep only rows with finite J_run
    sub["J_run"] = sub["J_run"].astype(float)
    sub = sub[np.isfinite(sub["J_run"])]

    if sub.empty:
        return

    # Aggregate duplicates (method, scenario) by mean of each component
    agg_cols = comps + ["J_run"]
    sub = sub.groupby(["scenario", "method"], as_index=False)[agg_cols].mean()

    scenarios = sorted(sub["scenario"].unique())
    methods = sorted(sub["method"].unique())

    n = len(scenarios)
    fig_h = max(4.0, 2.6 * n)
    fig = plt.figure(figsize=(max(10.0, 0.6 * len(methods) + 6.0), fig_h))

    for i, scen in enumerate(scenarios, start=1):
        ax = fig.add_subplot(n, 1, i)
        ss = sub[sub["scenario"] == scen].set_index("method")

        xs = np.arange(len(methods))
        bottom = np.zeros(len(methods), dtype=float)

        for comp in comps:
            vals = []
            for m in methods:
                if m in ss.index and np.isfinite(ss.loc[m, comp]):
                    vals.append(float(ss.loc[m, comp]))
                else:
                    vals.append(0.0)
            vals = np.array(vals, dtype=float)

            ax.bar(xs, vals, bottom=bottom, label=comp.replace("J_", ""))
            bottom += vals

        ax.set_xticks(xs)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_ylabel("mean per-tick")
        ax.set_title(f"{scen}")
        ax.grid(True, axis="y", alpha=0.3)

        if i == 1:
            ax.legend(ncols=4, fontsize=9, loc="upper right")

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_bandit_arm_pulls(arm_counts: list[int], arms: Any, out_png: Path, title: str) -> None:
    if not arm_counts:
        return
    counts = np.array(arm_counts, dtype=float)
    if counts.size == 0 or float(np.sum(counts)) <= 0:
        return

    labels = []
    # arms may be list of dicts with km_budget/charge_budget_frac
    if isinstance(arms, list) and all(isinstance(a, dict) for a in arms) and len(arms) == len(counts):
        for a in arms:
            km = a.get("km_budget", "?")
            cf = a.get("charge_budget_frac", "?")
            labels.append(f"km{km}-c{cf}")
    else:
        labels = [str(i) for i in range(len(counts))]

    fig = plt.figure(figsize=(max(10.0, 0.35 * len(counts) + 6.0), 4.5))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(counts)), counts)
    ax.set_title(title)
    ax.set_ylabel("pulls")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# Robustness table
# -----------------------------
def compute_robustness(df):
    base = df[df["scenario"] == "baseline"].copy()
    base = base[["phase", "network", "hours", "method_family", "method", "J_run_mean"]].rename(
        columns={"J_run_mean": "J_run_baseline"}
    )
    merged = df.merge(base, on=["phase", "network", "hours", "method_family", "method"], how="left")
    merged["dJ_vs_baseline"] = merged["J_run_mean"].astype(float) - merged["J_run_baseline"].astype(float)
    return merged


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="out/paper", help="Root folder containing paper runs.")
    p.add_argument("--out", default="out/analysis", help="Output folder for CSVs and plots.")
    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="Comma-separated KPI keys to aggregate.")
    p.add_argument("--phases", default="", help="Optional comma-separated list of phases to include.")
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    phases_filter = [x.strip() for x in str(args.phases).split(",") if x.strip()]

    _mkdir(out)
    plots_dir = out / "plots"
    _mkdir(plots_dir)

    json_files = sorted(root.rglob("*.json"))
    if phases_filter:
        json_files = [f for f in json_files if parse_path_meta(root, f).phase in phases_filter]

    all_rows: list[dict[str, Any]] = []
    bandit_extras: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for fp in json_files:
        rows, extra = parse_file(fp, root, metrics)
        all_rows.extend(rows)
        if extra:
            # tie extra to row identity (phase/network/scenario/method)
            for r in rows[:1]:
                bandit_extras.append((r, extra))

    if not all_rows:
        print(f"No parseable JSON files found under: {root}")
        return

    # DataFrame or plain list
    if pd is not None:
        df = pd.DataFrame(all_rows)
    else:
        # minimal fallback: write CSV without pandas
        df = None

    # Ensure numeric fields exist
    numeric_cols = [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics]
    if pd is not None:
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Rename to simpler columns for plotting
        for m in metrics:
            if f"{m}_mean" in df.columns and m not in df.columns:
                df[m] = df[f"{m}_mean"]

        # Save master CSV
        master_csv = out / "master_runs.csv"
        df.to_csv(master_csv, index=False)
        print(f"Saved: {master_csv}")

        # Robustness
        rob = compute_robustness(df.rename(columns={f"{m}_mean": f"{m}_mean" for m in metrics}))
        rob_csv = out / "robustness.csv"
        rob.to_csv(rob_csv, index=False)
        print(f"Saved: {rob_csv}")

        # Plots per phase/network
        for (phase, network, hours), g in df.groupby(["phase", "network", "hours"]):
            g2 = g.copy()

            # Prefer unique method labels (method_family::method) to avoid collisions
            g2["method"] = g2.apply(lambda r: f"{r['method_family']}::{r['method']}", axis=1)

            title = f"{phase} | {network} | {int(hours)}h"
            save_jrun_table(
                g2,
                plots_dir / f"jrun_table_{_slug(phase)}_{_slug(network)}_{int(hours)}h.png",
                title=f"J_run (mean) — {title}",
            )
            save_stacked_objective(
                g2,
                plots_dir / f"stacked_objective_{_slug(phase)}_{_slug(network)}_{int(hours)}h.png",
                title=f"Objective decomposition (mean) — {title}",
            )

        # Bandit arm plots (where available)
        for row, extra in bandit_extras:
            counts = extra.get("arm_counts", None)
            arms = extra.get("arms", None)
            if counts is None:
                continue

            phase = str(row.get("phase", ""))
            network = str(row.get("network", ""))
            scenario = str(row.get("scenario", ""))
            method = str(row.get("method", "bandit"))
            hours = row.get("hours", float("nan"))

            hours_tag = f"{int(hours)}h" if _isfinite(hours) else "unknownh"

            out_png = plots_dir / (
                f"bandit_arm_pulls_{_slug(phase)}_{_slug(network)}_{_slug(scenario)}_{_slug(method)}_{hours_tag}.png"
            )
            save_bandit_arm_pulls(
                arm_counts=list(counts) if isinstance(counts, list) else [],
                arms=arms,
                out_png=out_png,
                title=f"Bandit arm pulls — {phase} | {network} | {hours_tag} | {scenario} | {method}",
            )

        print(f"Plots saved under: {plots_dir}")

    else:
        # Fallback: write a minimal CSV
        master_csv = out / "master_runs.csv"
        keys = sorted({k for r in all_rows for k in r.keys()})
        with master_csv.open("w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in all_rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"Saved (no pandas available): {master_csv}")
        print("Install pandas to enable plots and robustness outputs.")


if __name__ == "__main__":
    main()
