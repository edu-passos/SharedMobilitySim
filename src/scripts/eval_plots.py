import argparse
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCENARIOS_ORDER = ["baseline", "event_heavy", "hetero_lambda", "hotspot_od"]


def _col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns


def _scenario_iter(df: pd.DataFrame) -> Iterator[str]:
    # keep known order first, then any extras
    scen = [s for s in SCENARIOS_ORDER if s in set(df["scenario"].astype(str))]
    extra = sorted(set(df["scenario"].astype(str)) - set(scen))
    yield from scen + extra


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def barh_metric(
    df: pd.DataFrame,
    scenario: str,
    metric: str,
    *,
    out_dir: Path,
    top_k: int,
) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if not _col(df, mean_col):
        print(f"[skip] missing column: {mean_col}")
        return

    sub = df[df["scenario"].astype(str) == scenario].copy()
    sub = sub[np.isfinite(pd.to_numeric(sub[mean_col], errors="coerce"))]
    if sub.empty:
        print(f"[skip] scenario={scenario} metric={metric}: no rows")
        return

    sub[mean_col] = pd.to_numeric(sub[mean_col], errors="coerce")
    if _col(sub, std_col):
        sub[std_col] = pd.to_numeric(sub[std_col], errors="coerce").fillna(0.0)
    else:
        sub[std_col] = 0.0

    # Sort best (lower is better for J_run and unmet_rate). If you ever plot "availability", reverse it.
    ascending = True
    sub = sub.sort_values(mean_col, ascending=ascending).head(top_k)

    methods = sub["method"].astype(str).tolist()
    means = sub[mean_col].to_numpy(dtype=float)
    stds = sub[std_col].to_numpy(dtype=float)

    fig = plt.figure(figsize=(12.8, 7.2))  # 16:9-ish
    ax = fig.add_subplot(111)

    y = np.arange(len(methods))
    ax.barh(y, means, xerr=stds)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()

    ax.set_title(f"{scenario} — {metric} (mean ± std)")
    ax.set_xlabel(metric)
    ax.grid(visible=True, axis="x", linestyle="--", alpha=0.4)

    out_path = out_dir / f"{scenario}__barh__{metric}.png"
    _save(fig, out_path)
    print(f"[ok] {out_path}")


def scatter_reloc_vs_unmet(df: pd.DataFrame, scenario: str, *, out_dir: Path, top_k: int) -> None:
    x_col = "relocation_km_total_mean"
    y_col = "unmet_rate_mean"

    if not (_col(df, x_col) and _col(df, y_col)):
        print(f"[skip] missing columns for scatter: {x_col} and/or {y_col}")
        return

    sub = df[df["scenario"].astype(str) == scenario].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")

    sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col])]
    if sub.empty:
        print(f"[skip] scenario={scenario}: no rows for scatter")
        return

    # Keep plot readable: show top_k by J_run if available, else take first top_k
    if _col(sub, "J_run_mean"):
        sub["J_run_mean"] = pd.to_numeric(sub["J_run_mean"], errors="coerce")
        sub = sub.sort_values("J_run_mean", ascending=True).head(top_k)
    else:
        sub = sub.head(top_k)

    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111)

    ax.scatter(sub[x_col].to_numpy(), sub[y_col].to_numpy())
    for _, r in sub.iterrows():
        ax.annotate(str(r["method"]), (float(r[x_col]), float(r[y_col])), fontsize=8, alpha=0.85)

    ax.set_title(f"{scenario} — relocation vs unmet (top {top_k})")
    ax.set_xlabel("relocation_km_total_mean")
    ax.set_ylabel("unmet_rate_mean")
    ax.grid(visible=True, linestyle="--", alpha=0.4)

    out_path = out_dir / f"{scenario}__scatter__reloc_vs_unmet.png"
    _save(fig, out_path)
    print(f"[ok] {out_path}")


def stacked_objective_decomp(df: pd.DataFrame, scenario: str, *, out_dir: Path, top_k: int) -> None:
    # Requires these columns from your summary CSV
    parts = ["J_avail_run", "J_reloc_run", "J_charge_run", "J_queue_run"]
    mean_cols = [f"{p}_mean" for p in parts]
    if not all(_col(df, c) for c in mean_cols):
        print(f"[skip] missing decomposition columns for stacked plot in scenario={scenario}")
        return

    sub = df[df["scenario"].astype(str) == scenario].copy()
    for c in [*mean_cols, "J_run_mean"]:
        if _col(sub, c):
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

    sub = sub[np.isfinite(sub["J_run_mean"])]
    if sub.empty:
        print(f"[skip] scenario={scenario}: no rows for decomposition")
        return

    sub = sub.sort_values("J_run_mean", ascending=True).head(top_k)

    methods = sub["method"].astype(str).tolist()
    data = np.vstack([sub[c].to_numpy(dtype=float) for c in mean_cols])  # (4, K)

    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111)

    y = np.arange(len(methods))
    left = np.zeros(len(methods), dtype=float)
    for i, p in enumerate(parts):
        ax.barh(y, data[i], left=left, label=p)
        left += data[i]

    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()

    ax.set_title(f"{scenario} — objective decomposition (per-tick mean contributions)")
    ax.set_xlabel("J components (mean)")
    ax.grid(visible=True, axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    out_path = out_dir / f"{scenario}__stacked__objective_decomp.png"
    _save(fig, out_path)
    print(f"[ok] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True, help="Path to all_results_summary.csv")
    ap.add_argument("--out_dir", default="out/eval/plots", help="Output directory for PNGs")
    ap.add_argument("--top_k", type=int, default=8, help="Max methods per plot (keeps slides readable)")
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)

    # Defensive: ensure required columns exist
    if "scenario" not in df.columns or "method" not in df.columns:
        raise SystemExit("CSV must contain 'scenario' and 'method' columns.")

    out_dir = Path(args.out_dir)

    for scen in _scenario_iter(df):
        # Slide 1: performance
        barh_metric(df, scen, "J_run", out_dir=out_dir, top_k=args.top_k)
        # Slide 2: service quality
        barh_metric(df, scen, "unmet_rate", out_dir=out_dir, top_k=args.top_k)
        # Slide 3: tradeoff
        scatter_reloc_vs_unmet(df, scen, out_dir=out_dir, top_k=args.top_k)
        # Slide 4: decomposition (if present)
        stacked_objective_decomp(df, scen, out_dir=out_dir, top_k=args.top_k)


if __name__ == "__main__":
    main()
