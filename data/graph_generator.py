import argparse
import math
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import yaml


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance between two lat/lon points in kilometers."""
    R = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("../configs/network_porto10.yaml"))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--city", type=str, default="Porto, Portugal")
    parser.add_argument("--speed-kmh", type=float, default=15.0)
    parser.add_argument(
        "--min-spacing-m",
        type=float,
        default=500.0,
        help="min pairwise spacing between stations",
    )
    parser.add_argument(
        "--symmetrize",
        action="store_true",
        help="make distance/time matrices symmetric",
    )
    args = parser.parse_args()

    print(f"Building synthetic {args.n}-node network for {args.city} ...")

    # 1) Get OSM bike graph and keep the largest connected component (undirected)
    G = ox.graph_from_place(args.city, network_type="bike", simplify=True)
    print("Graph downloaded:", len(G), "nodes")
    # keep largest weakly-connected component on directed graph
    G_lcc = ox.truncate.largest_component(G, strongly=False)
    # then convert to undirected for symmetric routing
    Gu = ox.convert.to_undirected(G_lcc)
    print("Largest connected component (undirected):", len(Gu), "nodes")

    # 2) Pick N graph nodes as pseudo-stations with min spacing
    rng = np.random.default_rng(42)
    node_list = list(Gu.nodes)

    sel = []
    while len(sel) < args.n and len(sel) < len(node_list):
        n = int(rng.integers(0, len(node_list)))
        cand_node = node_list[n]
        lat_c, lon_c = Gu.nodes[cand_node]["y"], Gu.nodes[cand_node]["x"]
        # enforce min spacing in meters
        ok = True
        for s in sel:
            lat_s, lon_s = Gu.nodes[s]["y"], Gu.nodes[s]["x"]
            if haversine_km(lat_c, lon_c, lat_s, lon_s) * 1000.0 < args.min_spacing_m:
                ok = False
                break
        if ok:
            sel.append(cand_node)

    if len(sel) < args.n:
        raise RuntimeError(f"Could only place {len(sel)} stations with min-spacing {args.min_spacing_m} m. Reduce spacing or N.")

    coords = np.array([(Gu.nodes[n]["y"], Gu.nodes[n]["x"]) for n in sel])
    df = pd.DataFrame(coords, columns=["lat", "lon"])
    df["id"] = [f"S{i + 1}" for i in range(args.n)]
    df["name"] = [f"Station_{i + 1}" for i in range(args.n)]
    df["capacity"] = 12

    # 3) Compute distance and travel-time matrices
    N = len(df)
    dist_km = np.zeros((N, N), dtype=float)
    time_min = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            try:
                route = nx.shortest_path(Gu, sel[i], sel[j], weight="length")
                edges_gdf = ox.routing.route_to_gdf(Gu, route)
                Lm = float(edges_gdf["length"].sum())
                dist_km[i, j] = Lm / 1000.0
                time_min[i, j] = (dist_km[i, j] / max(args.speed_kmh, 1e-6)) * 60.0
            except Exception:
                # Fallback (should be rare after LCC): haversine distance
                d_km = haversine_km(
                    df.loc[i, "lat"],
                    df.loc[i, "lon"],
                    df.loc[j, "lat"],
                    df.loc[j, "lon"],
                )
                dist_km[i, j] = d_km
                time_min[i, j] = (d_km / max(args.speed_kmh, 1e-6)) * 60.0

    # Optional: symmetrize (useful for first experiments)
    if args.symmetrize:
        dist_km = 0.5 * (dist_km + dist_km.T)
        time_min = 0.5 * (time_min + time_min.T)

    # 4) Write YAML config
    cfg = {
        "time": {"dt_minutes": 5, "horizon_hours": 48},
        "network": {
            "n_stations": int(N),
            "capacity_default": 12,
            "distance_km": dist_km.round(2).tolist(),
            "travel_time_min": time_min.round(1).tolist(),
            "station_ids": df["id"].tolist(),
            "station_names": df["name"].tolist(),
            "lat": df["lat"].round(7).tolist(),
            "lon": df["lon"].round(7).tolist(),
        },
        "demand": {"base_lambda_per_dt": 0.8},
        "energy": {
            "chargers_per_station": [2] * N,
            "charge_rate_soc_per_hour": [0.25] * N,
            "battery_kwh_per_vehicle": 0.5,
            "energy_cost_per_kwh_eur": 0.20,
        },
        "ops": {"thresholds": {"low": 0.2, "high": 0.8}},
        "seed": 42,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    main()
