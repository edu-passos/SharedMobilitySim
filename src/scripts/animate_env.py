import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

from envs.porto_env import PortoMicromobilityEnv


# -----------------------------
# Scenario application (same logic you already use)
# -----------------------------
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


def _normalize_xy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = (lon - lon.min()) / max(lon.max() - lon.min(), 1e-12)
    y = (lat - lat.min()) / max(lat.max() - lat.min(), 1e-12)
    return x, y


def _tick_to_dayhour(t: int, dt_min: int) -> tuple[int, int]:
    minutes = t * dt_min
    day = minutes // (24 * 60)
    hour = (minutes % (24 * 60)) // 60
    return int(day), int(hour)


def _aggregate_reloc_flows(plans: list[list[tuple[int, int, int]]], N: int) -> np.ndarray:
    """Aggregate multiple reloc_plans into an NxN flow matrix (units moved)."""
    F = np.zeros((N, N), dtype=float)
    for plan in plans:
        if not plan:
            continue
        for i, j, k in plan:
            if k is None:
                continue
            kk = float(k)
            if kk <= 0:
                continue
            if 0 <= int(i) < N and 0 <= int(j) < N:
                F[int(i), int(j)] += kk
    return F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/network_porto10.yaml")
    ap.add_argument("--hours", type=int, default=168)
    ap.add_argument("--seed", type=int, default=42)

    # planner choices
    ap.add_argument("--reloc", default="budgeted")
    ap.add_argument("--charge", default="greedy")

    # action is the env-level 4-vector (mapped internally)
    ap.add_argument("--action", type=float, nargs=4, default=[0.5, 0.5, 0.5, 0.5])

    # scenario
    ap.add_argument("--scenario", default="baseline", help="baseline | hotspot_od | hetero_lambda | event_heavy")
    ap.add_argument("--hotspot_j", type=int, default=0)
    ap.add_argument("--hotspot_p", type=float, default=0.6)
    ap.add_argument("--hetero_strength", type=float, default=0.6)
    ap.add_argument("--event_scale", type=float, default=1.5)

    # animation controls
    ap.add_argument("--frame_every", type=int, default=6, help="ticks per frame (6 => 30 minutes if dt=5min)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default="out/sim_animation_env.mp4")

    # arrow controls
    ap.add_argument("--arrow_min_units", type=float, default=1.0, help="Only draw reloc flows >= this many units per frame.")
    ap.add_argument("--arrow_max", type=int, default=25, help="Max arrows drawn per frame (largest flows).")
    ap.add_argument("--arrow_scale", type=float, default=1.0, help="Visual scale for arrow width.")

    args = ap.parse_args()

    # --- load YAML for station coords/labels ---
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lat = np.array(cfg["network"]["lat"], dtype=float)
    lon = np.array(cfg["network"]["lon"], dtype=float)
    station_ids = cfg["network"].get("station_ids", [f"S{i + 1}" for i in range(len(lat))])

    x, y = _normalize_xy(lon, lat)
    N = len(lat)

    # --- run one episode via the ENV, collect per-tick obs + info ---
    env = PortoMicromobilityEnv(
        cfg_path=str(args.config),
        episode_hours=int(args.hours),
        seed=int(args.seed),
        reloc_name_override=str(args.reloc) if args.reloc else None,
        charge_name_override=str(args.charge) if args.charge else None,
    )

    obs = env.reset()
    apply_scenario(
        env,
        scenario=str(args.scenario),
        seed=int(args.seed),
        hotspot_j=int(args.hotspot_j),
        hotspot_p=float(args.hotspot_p),
        hetero_strength=float(args.hetero_strength),
        event_scale=float(args.event_scale),
    )

    action = np.clip(np.asarray(args.action, dtype=float).reshape(4), 0.0, 1.0)

    max_steps = int(env.max_steps)
    dt_min = int(env.dt_min)

    # Store series for animation (from OBS, not from logs)
    fill_series = np.zeros((max_steps, N), dtype=float)
    soc_series = np.zeros((max_steps, N), dtype=float)
    waiting_series = np.zeros((max_steps, N), dtype=float)

    avail_series = np.zeros(max_steps, dtype=float)
    queue_total_series = np.zeros(max_steps, dtype=float)
    reloc_km_series = np.zeros(max_steps, dtype=float)
    charge_eur_series = np.zeros(max_steps, dtype=float)

    reloc_plan_series: list[list[tuple[int, int, int]]] = []

    total_reward = 0.0

    for t in range(max_steps):
        # record current obs (state before applying action at step t)
        fill_series[t] = np.asarray(obs["fill_ratio"], dtype=float).reshape(N)
        soc_series[t] = np.asarray(obs["soc"], dtype=float).reshape(N)
        waiting_series[t] = np.asarray(obs["waiting"], dtype=float).reshape(N)

        obs, r, done, info = env.step(action)
        total_reward += float(r)

        kpi = info.get("kpi", {}) or {}
        avail_series[t] = float(kpi.get("availability", 0.0))
        queue_total_series[t] = float(kpi.get("queue_total", 0.0))
        reloc_km_series[t] = float(kpi.get("reloc_km", 0.0))
        charge_eur_series[t] = float(kpi.get("charge_cost_eur", 0.0))

        reloc_plan = info.get("reloc_plan", []) or []
        # ensure tuples are (i,j,k)
        reloc_plan_series.append([(int(a), int(b), int(c)) for (a, b, c) in reloc_plan])

        if done:
            # truncate all arrays cleanly
            T = t + 1
            fill_series = fill_series[:T]
            soc_series = soc_series[:T]
            waiting_series = waiting_series[:T]
            avail_series = avail_series[:T]
            queue_total_series = queue_total_series[:T]
            reloc_km_series = reloc_km_series[:T]
            charge_eur_series = charge_eur_series[:T]
            reloc_plan_series = reloc_plan_series[:T]
            break

    T = fill_series.shape[0]
    frames = list(range(0, T, max(1, int(args.frame_every))))

    reloc_km_cum = np.cumsum(reloc_km_series)
    charge_eur_cum = np.cumsum(charge_eur_series)

    # --- plot setup ---
    fig = plt.figure(figsize=(7.8, 7.8))
    ax = fig.add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    title = (
        f"PortoMicromobilitySim | {args.scenario} | {args.hours}h | reloc={args.reloc}, charge={args.charge}, a={action.tolist()}"
    )
    ax.set_title(title, fontsize=10)

    for i in range(N):
        ax.text(x[i] + 0.01, y[i] + 0.01, str(station_ids[i]), fontsize=8)

    # scatter: size by fill, color by soc
    fill0 = fill_series[0]
    soc0 = soc_series[0]
    sizes0 = 80 + 420 * (fill0 / max(float(np.max(fill0)), 1e-9))
    sc = ax.scatter(
        x,
        y,
        s=sizes0,
        c=soc0,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        edgecolors="k",
        linewidths=0.8,
        zorder=3,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("SoC (station avg)", fontsize=9)

    kpi_text = ax.text(
        0.01,
        -0.12,
        "",
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
    )

    # arrows are artists we update each frame
    arrow_artists: list[FancyArrowPatch] = []

    def _clear_arrows():
        nonlocal arrow_artists
        for a in arrow_artists:
            try:
                a.remove()
            except Exception:
                pass
        arrow_artists = []

    def _draw_arrows(flow: np.ndarray):
        """Draw top-k flows as arrows."""
        nonlocal arrow_artists
        _clear_arrows()

        # collect edges above threshold
        edges = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                v = float(flow[i, j])
                if v >= float(args.arrow_min_units):
                    edges.append((v, i, j))

        if not edges:
            return

        # keep top arrows
        edges.sort(reverse=True, key=lambda t: t[0])
        edges = edges[: int(args.arrow_max)]

        maxv = max(e[0] for e in edges)
        maxv = max(maxv, 1e-9)

        for v, i, j in edges:
            # arrow width scales with relative flow
            lw = float(args.arrow_scale) * (0.5 + 3.0 * (v / maxv))
            # shrink endpoints slightly so arrows don't cover node centers
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            L = math.hypot(dx, dy)
            if L < 1e-9:
                continue
            shrink = 0.025
            xi, yi = x[i] + shrink * dx / L, y[i] + shrink * dy / L
            xj, yj = x[j] - shrink * dx / L, y[j] - shrink * dy / L

            arr = FancyArrowPatch(
                (xi, yi),
                (xj, yj),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=lw,
                alpha=0.7,
                color="black",
                zorder=2,
            )
            ax.add_patch(arr)
            arrow_artists.append(arr)

    def update(fi: int):
        t = frames[fi]
        day, hour = _tick_to_dayhour(t, dt_min)

        fill_t = fill_series[t]
        soc_t = soc_series[t]

        # sizes by fill (not by stock, since obs exposes fill_ratio directly)
        sizes = 80 + 420 * (fill_t / max(float(np.max(fill_t)), 1e-9))
        sc.set_sizes(sizes)
        sc.set_array(soc_t)

        # relocation flows during this frame window
        t2 = min(t + int(args.frame_every), T)
        flow = _aggregate_reloc_flows(reloc_plan_series[t:t2], N)
        _draw_arrows(flow)

        kpi_text.set_text(
            f"t={t:4d}  (day {day}, hour {hour})\n"
            f"availability={avail_series[t]:.3f}\n"
            f"queue_total={queue_total_series[t]:.1f}\n"
            f"reloc_km_cum={reloc_km_cum[t]:.1f}\n"
            f"charge€_cum={charge_eur_cum[t]:.2f}"
        )

        return (sc, kpi_text, *arrow_artists)

    ani = FuncAnimation(fig, update, frames=len(frames), interval=1000 // int(args.fps), blit=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_path), fps=int(args.fps))

    print(f"Saved: {out_path}")
    print(f"Episode ticks: {T}, dt_min={dt_min}, total_reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
