from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np

Move = tuple[int, int, int]  # (i -> j, k units)


@dataclass
class SimConfig:
    """Minimal config for a docked shared-mobility sim."""

    # Simulation parameters
    dt_min: int  # time step (delta t) in minutes
    horizon_h: int  # total hours to simulate
    # System parameters
    capacity: np.ndarray  # (N,) max vehicles per station
    travel_min: np.ndarray  # (N,N) travel time matrix (minutes)
    cost_km: np.ndarray  # (N,N) relocation distance or cost
    chargers: np.ndarray  # (N,) plugs per station
    charge_rate: np.ndarray  # (N,) SoC/hour when plugged (e.g. 0.25 => +25%/h)
    battery_kwh: float  # kWh per vehicle @ 100% SoC
    energy_cost_per_kwh: float  # €/kWh
    soc_min_depart: float = 0.00     # min SoC needed for a rental to start
    reserve_plugged: bool = True    # vehicles plugged this tick are not rentable


class Sim:
    """Discrete-time simulator for station-based e-scooters/e-bikes.

    State per station i: stock x[i], average SoC s[i], and SoC mass m[i] = x[i]*s[i].
    Trips/relocations conserve SoC mass; charging adds mass to plugged units.
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        # Split RNG into two independent streams deterministically
        seed_root = rng.integers(0, 2**32 - 1, dtype=np.uint32).item()
        self.rng_demand = np.random.default_rng(seed_root + 1)
        self.rng_route  = np.random.default_rng(seed_root + 2)

        N = cfg.capacity.shape[0]
        assert cfg.travel_min.shape == (N, N), "travel_min must be NxN"
        assert cfg.cost_km.shape == (N, N), "cost_km must be NxN"
        for arr, name in ((cfg.capacity, "capacity"), (cfg.charge_rate, "charge_rate"), (cfg.chargers, "chargers")):
            assert arr.shape == (N,), f"{name} must be shape (N,)"

        # State
        self.t = 0  # minutes since start
        self.x = np.minimum(cfg.capacity // 3, cfg.capacity).astype(int)  # initial stock ~1/3 full
        self.s = rng.uniform(0.5, 0.9, size=N)  # avg SoC in [50%, 90%]
        self.m = self.x.astype(float) * self.s  # SoC "mass" (conserved across moves)
        self._trip_heap: list[tuple[int, int, float, float]] = []  # (t_arrive, dest_j, soc_use, s_depart)

        # Accumulators / logs
        self.logs: list[dict] = []
        N = cfg.capacity.shape[0]
        self.waiting = np.zeros(N, dtype=int)

    def step(
        self,
        lam_t: np.ndarray,
        P_t: np.ndarray,
        *,
        weather_fac: float = 1.0,
        event_fac: np.ndarray | None = None,
        reloc_plan: list[Move] | None = None,
        charging_plan: np.ndarray | None = None,
    ) -> None:
        """Advance simulation by one tick."""
        self._validate_inputs(lam_t, P_t, event_fac, charging_plan)

        dt_h = self.cfg.dt_min / 60.0
        N = self.x.shape[0]

        # Resolve charging plan early so we can reserve those vehicles from rentals this tick
        plan_reserve = self._resolve_charging_plan(charging_plan)

        # 1) Demand arrivals and service
        lam_eff = lam_t * weather_fac
        if event_fac is not None:
            lam_eff = lam_eff * event_fac

        A = self.rng_demand.poisson(lam_eff)            # new arrivals per station
        q0 = self.waiting.copy()                 # backlog before arrivals
        x0 = self.x.copy()                       # physical stock before service
        m0 = self.m.copy()                       # SoC mass before service

        # total requests at station now
        self.waiting = q0 + A
        requests_total = int(self.waiting.sum())  # backlog + new arrivals
        demand_total = int(A.sum())

        # ----- rentable stock: reserve plugged + enforce SoC feasibility -----
        x_rentable0 = x0.copy()

        # Reserve vehicles that are going to be charged this tick (not rentable now)
        reserve_plugged = bool(getattr(self.cfg, "reserve_plugged", True))

        x_rentable_base = x0.copy()
        if reserve_plugged:
            x_rentable_base = np.maximum(0, x_rentable_base - plan_reserve)

        # SoC feasibility
        soc_min = float(self.cfg.soc_min_depart)
        x_rentable0 = x_rentable_base.copy()
        if soc_min > 0.0:
            rideable = np.floor(m0 / soc_min).astype(int)
            x_rentable0 = np.minimum(x_rentable0, rideable)

        rentable_frac = float(np.mean(x_rentable0 < x0))

        soc_bind_frac = 0.0
        if soc_min > 0.0:
            rideable = np.floor(m0 / soc_min).astype(int)
            soc_bind_frac = float(np.mean(rideable < x_rentable_base))

        x_total_pre = int(x0.sum())
        x_rentable_total_pre = int(x_rentable0.sum())



        # Serve as many as possible from the queue (FIFO implied), limited by rentable stock
        can_serve = np.minimum(x_rentable0, self.waiting)
        self.waiting -= can_serve

        served_total = int(can_serve.sum())
        queue_total = int(self.waiting.sum())

        # How many NEW arrivals got served immediately?
        # Capacity left for new after serving backlog, based on rentable stock
        remaining_capacity = np.maximum(0, x_rentable0 - q0)
        served_new = np.minimum(A, remaining_capacity)
        served_new_total = int(served_new.sum())

        # Immediate-service availability (set to 1 if no arrivals)
        availability = 1.0 if demand_total == 0 else (served_new_total / demand_total)

        # Backlog ratio
        queue_rate = queue_total / max(requests_total, 1)

        # Unmet NEW arrivals this tick
        unmet_tick = demand_total - served_new_total

        # Average SoC at moment of departure (BEFORE decrement), based on pre-service state (x0,m0)
        avg_at_depart = np.zeros_like(self.s, dtype=float)
        mask0 = x0 > 0
        avg_at_depart[mask0] = m0[mask0] / x0[mask0]

        # Remove departing vehicles + their SoC mass
        if can_serve.sum():
            self.x -= can_serve
            self.m -= can_serve * avg_at_depart

        # 2) Spawn trips (schedule arrivals with s_depart carried)
        if can_serve.sum():
            for i, k in enumerate(can_serve):
                if k <= 0:
                    continue
                dests = self.rng_route.choice(N, size=int(k), p=P_t[i])
                tij = self.cfg.travel_min[i, dests].astype(int)
                uses = np.clip(0.01 + 0.12 * (tij / 60.0), 0.0, 1.0)
                s_dep = float(avg_at_depart[i])
                for tt, j, u in zip(tij, dests, uses):
                    heappush(self._trip_heap, (self.t + int(tt), int(j), float(u), s_dep))

        # 3) Complete trips due by next tick
        t_next = self.t + self.cfg.dt_min
        overflow_rerouted = 0
        overflow_dropped = 0
        overflow_extra_min = 0.0

        due = []
        while self._trip_heap and self._trip_heap[0][0] <= t_next:
            due.append(heappop(self._trip_heap))

        for _, j, soc_use, s_depart in due:
            s_arrive = max(0.0, s_depart - soc_use)
            if self.x[j] < self.cfg.capacity[j]:
                self.x[j] += 1
                self.m[j] += s_arrive
            else:
                free = np.where(self.x < self.cfg.capacity)[0]
                if free.size:
                    k = free[np.argmin(self.cfg.travel_min[j, free])]
                    self.x[k] += 1
                    self.m[k] += s_arrive
                    overflow_rerouted += 1
                    overflow_extra_min += float(self.cfg.travel_min[j, k])
                else:
                    overflow_dropped += 1

        # 4) Charging (operator decision) -> add mass to plugged units, then cap
        plan_exec = np.minimum(plan_reserve, np.minimum(self.cfg.chargers, self.x)).astype(int)
        total_chargers = int(np.sum(self.cfg.chargers))
        charge_utilization = (int(plan_exec.sum()) / total_chargers) if total_chargers > 0 else 0.0

        delta_soc = self.cfg.charge_rate * dt_h
        self.m += plan_exec * delta_soc
        self.m = np.minimum(self.m, self.x.astype(float) * 1.0)

        energy_kwh = float(np.sum(plan_exec * self.cfg.battery_kwh * delta_soc))
        charge_cost = energy_kwh * self.cfg.energy_cost_per_kwh
        soc_mean_vehicles = float(self.m.sum() / max(self.x.sum(), 1))

        # 5) Apply relocation plan (move both count and mass)
        reloc_km = 0.0
        reloc_units = 0
        reloc_edges = 0

        if reloc_plan:
            for i, j, k in reloc_plan:
                if k <= 0:
                    continue
                move = int(min(k, self.x[i], self.cfg.capacity[j] - self.x[j]))
                if move <= 0:
                    continue

                reloc_edges += 1
                reloc_units += move

                avg_i = 0.0 if self.x[i] == 0 else (self.m[i] / self.x[i])
                mass_move = move * avg_i
                self.x[i] -= move
                self.m[i] -= mass_move
                self.x[j] += move
                self.m[j] += mass_move
                reloc_km += move * self.cfg.cost_km[i, j]

        # 6) Clamp, refresh averages, log
        self.x = np.clip(self.x, 0, self.cfg.capacity)
        self.m = np.clip(self.m, 0.0, self.x.astype(float))
        self._refresh_avg_soc()

        fill = self.x.astype(float) / np.maximum(self.cfg.capacity, 1)
        fill_p10 = float(np.percentile(fill, 10))
        fill_p90 = float(np.percentile(fill, 90))

        soc_station = np.zeros_like(self.s, dtype=float)
        mask_post = self.x > 0
        soc_station[mask_post] = self.m[mask_post] / self.x[mask_post]
        soc_station_min = float(soc_station[mask_post].min()) if np.any(mask_post) else 0.0
        soc_station_p10 = float(np.percentile(soc_station[mask_post], 10)) if np.any(mask_post) else 0.0

        self.logs.append({
            "t_min": self.t,

            # demand/service
            "availability": float(availability),
            "queue_rate": float(queue_rate),
            "queue_total": queue_total,
            "requests_total": requests_total,
            "demand_total": demand_total,
            "served_total": served_total,
            "served_new_total": served_new_total,
            "unmet": int(unmet_tick),

            # rentability primitives
            "x_total_pre": x_total_pre,
            "x_rentable_total_pre": x_rentable_total_pre,
            "rentable_frac": rentable_frac,
            "soc_bind_frac": soc_bind_frac,

            # operations
            "reloc_km": float(reloc_km),
            "reloc_units": int(reloc_units),
            "reloc_edges": int(reloc_edges),

            "plugged_reserve": int(plan_reserve.sum()),
            "plugged": int(plan_exec.sum()),  # executed
            "charge_energy_kwh": float(energy_kwh),
            "charge_cost_eur": float(charge_cost),
            "overflow_dropped": int(overflow_dropped),


            # SoC
            "soc_mean": float(np.mean(self.s)),
            "soc_mean_vehicles": float(soc_mean_vehicles),
            "soc_station_min": soc_station_min,
            "soc_station_p10": soc_station_p10,

            # fill distribution
            "fill_p10": fill_p10,
            "fill_p90": fill_p90,

            # system state
            "full_ratio": float(np.mean(self.x == self.cfg.capacity)),
            "empty_ratio": float(np.mean(self.x == 0)),
            "stock_std": float(np.std(self.x)),

            # overflow/charging util
            "overflow_rerouted": int(overflow_rerouted),
            "overflow_extra_min": float(overflow_extra_min),
            "charge_utilization": float(charge_utilization),
        })
        self.t = t_next

        


    # ------------------------- helpers -------------------------

    def _refresh_avg_soc(self) -> None:
        """Recompute average SoC from mass and count."""
        xnz = self.x > 0
        self.s[~xnz] = 0.0
        self.s[xnz] = self.m[xnz] / self.x[xnz]

    def _resolve_charging_plan(self, charging_plan: np.ndarray | None) -> np.ndarray:
        """Clip to [0, chargers, x]. If None, default: plug as many as possible."""
        if charging_plan is None:
            return np.minimum(self.cfg.chargers, self.x).astype(int)

        plan = np.asarray(charging_plan, dtype=int).copy()
        plan = np.clip(plan, 0, None)
        plan = np.minimum(plan, self.cfg.chargers)
        plan = np.minimum(plan, self.x)

        return plan  # noqa: RET504

    def _validate_inputs(
        self,
        lam_t: np.ndarray,
        P_t: np.ndarray,
        event_fac: np.ndarray | None,
        charging_plan: np.ndarray | None,
    ) -> None:
        N = self.x.shape[0]
        assert lam_t.shape == (N,), f"lam_t must be shape (N,), got {lam_t.shape}"
        assert P_t.shape == (N, N), f"P_t must be shape (N,N), got {P_t.shape}"
        # rows of P must sum to 1
        row_sum = P_t.sum(axis=1)
        if not np.allclose(row_sum, 1.0, atol=1e-6):
            raise ValueError("Each row of P_t must sum to 1.0")
        if event_fac is not None and event_fac.shape != (N,):
            raise ValueError("event_fac must be shape (N,) or None")
        if charging_plan is not None and charging_plan.shape != (N,):
            raise ValueError("charging_plan must be shape (N,) or None")
