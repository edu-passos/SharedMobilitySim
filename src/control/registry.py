import importlib
from collections.abc import Callable
from typing import Any

import numpy as np

PlannerFn = Callable[..., Any]


class PlannerRegistry:
    """Registry for relocation and charging planner functions."""

    def __init__(self) -> None:
        self._reloc: dict[str, PlannerFn] = {}
        self._charge: dict[str, PlannerFn] = {}

    def register_relocation(self, name: str, fn: PlannerFn) -> None:
        """Register a relocation planner function."""
        self._reloc[name.lower()] = fn

    def register_charging(self, name: str, fn: PlannerFn) -> None:
        """Register a charging planner function."""
        self._charge[name.lower()] = fn

    def get_relocation(self, name: str) -> PlannerFn:
        """Get a relocation planner function by name."""
        key = name.lower()
        if key not in self._reloc:
            raise KeyError(f"Unknown relocation planner '{name}'.")
        return self._reloc[key]

    def get_charging(self, name: str) -> PlannerFn:
        """Get a charging planner function by name."""
        key = name.lower()
        if key not in self._charge:
            raise KeyError(f"Unknown charging planner '{name}'.")
        return self._charge[key]


REGISTRY = PlannerRegistry()


# Built-in adapters
def _reloc_greedy_adapter(x, C, cost_km, *, params: dict[str, Any]) -> Any:
    from control.planners import plan_relocation_greedy

    return plan_relocation_greedy(x, C, cost_km, **params)


def _charge_greedy_adapter(x, s, chargers, lam_t, *, params: dict[str, Any]) -> Any:
    from control.planners import plan_charging_greedy

    return plan_charging_greedy(x, s, chargers, lam_t, **params)


REGISTRY.register_relocation("greedy", _reloc_greedy_adapter)
REGISTRY.register_charging("greedy", _charge_greedy_adapter)


# Optional: ML adapters
def _load_dotted(dotted: str):
    """Load a callable/class from a dotted path.

    Example:
    'my_pkg.my_mod:MyPlannerClass' or 'my_pkg.my_mod:plan_fn'
    """
    if ":" in dotted:
        mod_name, attr = dotted.split(":", 1)
    else:
        # fallback: module that itself is callable
        mod_name, attr = dotted, None
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr) if attr else mod


def _reloc_ml_adapter(x, C, move_cost, *, params: dict[str, Any]) -> Any:
    """Params example:

      loader: "ml_pkg.reloc:RelocPlanner"
      kwargs: { checkpoint: "path/to.ckpt", ... }
    The loaded object must implement: plan(x, C, move_cost) -> list[(j,i,k)]
    """
    loader = params["loader"]
    kwargs = params.get("kwargs", {})
    obj = _load_dotted(loader)
    planner = obj(**kwargs) if callable(obj) and callable(obj) else obj
    if hasattr(planner, "plan"):
        return planner.plan(x, C, move_cost)
    # fallback: assume it's a function(x,C,move_cost,**kwargs)
    return obj(x, C, move_cost, **kwargs)


def _charge_ml_adapter(x, s, chargers, lam_t, *, params: dict[str, Any]) -> Any:
    """Params example:

      loader: "ml_pkg.charge:ChargePlanner"
      kwargs: {...}
    The loaded object must implement: plan(x, s, chargers, lam_t) -> np.ndarray[int]
    """
    loader = params["loader"]
    kwargs = params.get("kwargs", {})
    obj = _load_dotted(loader)
    planner = obj(**kwargs) if callable(obj) and callable(obj) else obj
    if hasattr(planner, "plan"):
        return planner.plan(x, s, chargers, lam_t)
    return obj(x, s, chargers, lam_t, **kwargs)


REGISTRY.register_relocation("ml", _reloc_ml_adapter)
REGISTRY.register_charging("ml", _charge_ml_adapter)


# No-op baselines (explicit controls)
def _reloc_noop_adapter(x, C, move_cost, *, params: dict[str, Any]) -> Any:
    # Relocation plan: empty list of moves
    return []


def _charge_noop_adapter(x, s, chargers, lam_t, *, params: dict[str, Any]) -> Any:
    # Charging plan: plug zero vehicles everywhere
    return np.zeros_like(x, dtype=int)


REGISTRY.register_relocation("noop", _reloc_noop_adapter)
REGISTRY.register_charging("noop", _charge_noop_adapter)


def _reloc_budgeted_adapter(x, C, cost_km, *, params: dict[str, Any]) -> Any:
    from control.planners import plan_relocation_budgeted

    return plan_relocation_budgeted(x, C, cost_km, **params)


def _charge_slack_adapter(x, s, chargers, lam_t, *, params: dict[str, Any]) -> Any:
    from control.planners import plan_charging_slack

    return plan_charging_slack(x, s, chargers, lam_t, **params)


REGISTRY.register_relocation("budgeted", _reloc_budgeted_adapter)
REGISTRY.register_charging("slack", _charge_slack_adapter)
