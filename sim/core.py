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


class Sim:
    """Discrete-time simulator for station-based e-scooters/e-bikes.

    State per station i: stock x[i], average SoC s[i], and SoC mass m[i] = x[i]*s[i].
    Trips/relocations conserve SoC mass; charging adds mass to plugged units.
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

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

        # 1) Demand arrivals and service
        lam_eff = lam_t * weather_fac
        if event_fac is not None:
            lam_eff = lam_eff * event_fac
        A = self.rng.poisson(lam_eff)  # arrivals per station
        # add to waiting queue
        self.waiting += A

        # vehicles available
        can_serve = np.minimum(self.x, self.waiting)
        self.waiting -= can_serve  # those get served
        unmet = self.waiting  # people still waiting after service

        demand_total = int(A.sum())
        served_total = int(can_serve.sum())
        # unmet requests *for this tick* (new arrivals that didn't start)
        unmet_tick = int((A - can_serve).sum())
        # queue length after service
        queue_total = int(self.waiting.sum())

        # Average SoC at moment of departure (before decrement)
        avg_at_depart = np.zeros_like(self.s, dtype=float)
        mask_prior = self.x > 0
        avg_at_depart[mask_prior] = self.m[mask_prior] / self.x[mask_prior]

        # Remove departing vehicles + their SoC mass
        if can_serve.sum():
            self.x -= can_serve
            self.m -= can_serve * avg_at_depart

        # 2) Spawn trips (schedule arrivals with s_depart carried)
        if can_serve.sum():
            for i, k in enumerate(can_serve):
                if k <= 0:
                    continue
                dests = self.rng.choice(N, size=int(k), p=P_t[i])
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
                # dock at station j
                self.x[j] += 1
                self.m[j] += s_arrive
            else:
                # reroute to nearest station with free slot
                free = np.where(self.x < self.cfg.capacity)[0]
                if free.size:
                    k = free[np.argmin(self.cfg.travel_min[j, free])]
                    self.x[k] += 1
                    self.m[k] += s_arrive
                    overflow_rerouted += 1
                    overflow_extra_min += float(self.cfg.travel_min[j, k])
                else:
                    # nowhere to dock -> drop the trip
                    overflow_dropped += 1

        # 4) Charging (operator decision) -> add mass to plugged units, then cap
        plan = self._resolve_charging_plan(charging_plan)
        total_chargers = int(np.sum(self.cfg.chargers))
        charge_utilization = (int(plan.sum()) / total_chargers) if total_chargers > 0 else 0.0
        delta_soc = self.cfg.charge_rate * dt_h  # per plugged unit in this tick
        # Add mass for plugged vehicles
        self.m += plan * delta_soc
        # Cap mass so average SoC <= 1.0
        self.m = np.minimum(self.m, self.x.astype(float) * 1.0)

        # Energy & cost accounting for charging
        energy_kwh = float(np.sum(plan * self.cfg.battery_kwh * delta_soc))
        charge_cost = energy_kwh * self.cfg.energy_cost_per_kwh

        # 5) Apply relocation plan (move both count and mass)
        reloc_km = 0.0
        if reloc_plan:
            for i, j, k in reloc_plan:
                if k <= 0:
                    continue
                move = int(min(k, self.x[i], self.cfg.capacity[j] - self.x[j]))
                if move <= 0:
                    continue
                avg_i = 0.0 if self.x[i] == 0 else (self.m[i] / self.x[i])
                mass_move = move * avg_i
                self.x[i] -= move
                self.m[i] -= mass_move
                self.x[j] += move
                self.m[j] += mass_move
                reloc_km += move * self.cfg.cost_km[i, j]

        # 6) Clamp, refresh averages, log
        self.x = np.clip(self.x, 0, self.cfg.capacity)
        # mass cannot exceed x (avg<=1) nor be negative
        self.m = np.clip(self.m, 0.0, self.x.astype(float))
        self._refresh_avg_soc()

        self.logs.append(
            {
                "t_min": self.t,
                # demand/service
                "demand_total": demand_total,
                "served_total": served_total,
                "unmet": unmet_tick,
                "queue_total": queue_total,
                # TODO: some description for the metrics below
                "availability": float((self.x > 0).mean()),
                "reloc_km": reloc_km,
                "plugged": int(plan.sum()),
                "charge_energy_kwh": energy_kwh,
                "charge_cost_eur": charge_cost,
                "overflow_rerouted": int(overflow_rerouted),
                "overflow_extra_min": float(overflow_extra_min),
                "soc_mean": float(np.mean(self.s)),
                "full_ratio": float(np.mean(self.x == self.cfg.capacity)),
                "empty_ratio": float(np.mean(self.x == 0)),
                "stock_std": float(np.std(self.x)),
                "reloc_ops": int(sum(k for *_ij, k in (reloc_plan or []))),
                "charge_utilization": float(charge_utilization),
            }
        )
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
