import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

from control.baselines import plan_charging_greedy, plan_relocation_greedy
from sim.core import Sim, SimConfig
from sim.demand import effective_lambda
from sim.events import events
from sim.weather_mc import make_default_weather_mc as weather_mc


def main(cfg_path: str) -> None:
    with Path(cfg_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    time_cfg = cfg.get("time", {})
    time_step = int(time_cfg["dt_minutes"])
    sim_duration = int(time_cfg["horizon_hours"])

    network_cfg = cfg.get("network", {})
    N = int(network_cfg["n_stations"])
    C = np.full(N, int(network_cfg["capacity_default"]))
    tmin = np.array(network_cfg["travel_time_min"], dtype=float).reshape(N, N)
    km = np.array(network_cfg["distance_km"], dtype=float).reshape(N, N)

    demand_cfg = cfg.get("demand", {})
    base_lambda = np.full(N, float(demand_cfg["base_lambda_per_dt"]))

    energy = cfg.get("energy", {})
    chargers = np.array(energy.get("chargers_per_station", [2] * N), dtype=int)
    charge_rate = np.array(energy.get("charge_rate_soc_per_hour", [0.25] * N), dtype=float)
    battery_kwh = float(energy.get("battery_kwh_per_vehicle", 0.5))
    energy_cost = float(energy.get("energy_cost_per_kwh_eur", 0.20))

    simcfg = SimConfig(
        dt_min=time_step,
        horizon_h=sim_duration,
        capacity=C,
        travel_min=tmin,
        cost_km=km,
        chargers=chargers,
        charge_rate=charge_rate,
        battery_kwh=battery_kwh,
        energy_cost_per_kwh=energy_cost,
    )
    sim_seed = int(cfg.get("seed", 42))
    sim = Sim(simcfg, np.random.default_rng(sim_seed))

    P = np.full((N, N), 1.0 / N, dtype=float)  # placeholder Orig-dest prob matrix

    W = weather_mc(dt_min=simcfg.dt_min, seed=sim_seed)

    steps = math.ceil(simcfg.horizon_h * 60 / simcfg.dt_min)
    events_matrix = events(steps, N, rng=sim.rng)
    if steps <= 0:
        print(
            {
                "error": "No steps to run",
                "horizon_h": simcfg.horizon_h,
                "dt_min": simcfg.dt_min,
            },
            flush=True,
        )
        return

    total_reloc_km = 0.0
    try:
        for step in range(steps):
            hour = (step * simcfg.dt_min / 60.0) % 24

            _w_state = W.step()  # e.g., "clear", "rain", ...
            w_fac = W.factor  # numeric multiplier, e.g., 1.0, 0.6, ...

            lam_t = effective_lambda(base_lambda, hour, weather_fac=w_fac, event_fac_vec=events_matrix[step])

            reloc = plan_relocation_greedy(
                sim.x,
                simcfg.capacity,
                simcfg.travel_min,
                low=0.25,
                high=0.8,
                target=0.6,
                hysteresis=0.03,
                max_moves=50,
            )
            charge_plan = plan_charging_greedy(
                sim.x,
                sim.s,
                simcfg.chargers,
                lam_t,
                threshold_quantile=0.5,  # only plug top-50% urgent stations
            )

            sim.step(
                lam_t,
                P,
                weather_fac=1.0,
                event_fac=None,
                reloc_plan=reloc,
                charging_plan=charge_plan,
            )

            total_reloc_km += sim.logs[-1]["reloc_km"]

            # lightweight progress every simulated 6 hours
            if step % max(1, int((6 * 60) / simcfg.dt_min)) == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
    except Exception as e:
        print("\nRuntime error:", repr(e), file=sys.stderr, flush=True)
        raise

    unmet_total = int(sum(r["unmet"] for r in sim.logs))
    max_queue = max(r.get("queue_total", 0) for r in sim.logs)
    avg_queue = float(np.mean([r.get("queue_total", 0) for r in sim.logs]))
    avail_avg = float(np.mean([r["availability"] for r in sim.logs]))
    energy_kwh = float(sum(r.get("charge_energy_kwh", 0.0) for r in sim.logs))
    energy_eur = float(sum(r.get("charge_cost_eur", 0.0) for r in sim.logs))
    reloc_km_tot = float(sum(r.get("reloc_km", 0.0) for r in sim.logs))

    overflow_rerouted_total = int(sum(r.get("overflow_rerouted", 0) for r in sim.logs))
    overflow_dropped_total = int(sum(r.get("overflow_dropped", 0) for r in sim.logs))
    overflow_extra_min_tot = float(sum(r.get("overflow_extra_min", 0.0) for r in sim.logs))

    soc_mean_avg = float(np.mean([r.get("soc_mean", 0.0) for r in sim.logs]))
    full_ratio_avg = float(np.mean([r.get("full_ratio", 0.0) for r in sim.logs]))
    empty_ratio_avg = float(np.mean([r.get("empty_ratio", 0.0) for r in sim.logs]))
    stock_std_avg = float(np.mean([r.get("stock_std", 0.0) for r in sim.logs]))

    reloc_ops_total = int(sum(r.get("reloc_ops", 0) for r in sim.logs))
    charge_util_avg = float(np.mean([r.get("charge_utilization", 0.0) for r in sim.logs]))

    print("\n", flush=True)
    print(
        {
            "unmet_total": unmet_total,
            "availability_avg": round(avail_avg, 3),
            "queue_total_max": max_queue,
            "queue_total_avg": round(avg_queue, 2),
            "relocation_km_total": round(reloc_km_tot, 2),
            "reloc_ops_total": reloc_ops_total,
            "charging_energy_kwh_total": round(energy_kwh, 2),
            "charging_cost_eur_total": round(energy_eur, 2),
            "charge_utilization_avg": round(charge_util_avg, 3),
            "overflow_rerouted_total": overflow_rerouted_total,
            "overflow_dropped_total": overflow_dropped_total,  # 0 if not logged
            "overflow_extra_min_total": round(overflow_extra_min_tot, 1),
            "soc_mean_avg": round(soc_mean_avg, 3),
            "full_ratio_avg": round(full_ratio_avg, 3),
            "empty_ratio_avg": round(empty_ratio_avg, 3),
            "stock_std_avg": round(stock_std_avg, 3),
            "ticks": steps,
            "dt_min": simcfg.dt_min,
            "stations": N,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/network_porto10.yaml")
    args = parser.parse_args()
    main(args.config)
