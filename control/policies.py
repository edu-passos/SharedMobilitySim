from __future__ import annotations
from typing import Callable, Dict, Any
import importlib

PlannerFn = Callable[..., Any]

class PlannerRegistry:
    def __init__(self):
        self._reloc: Dict[str, PlannerFn] = {}
        self._charge: Dict[str, PlannerFn] = {}

    def register_relocation(self, name: str, fn: PlannerFn):
        self._reloc[name.lower()] = fn

    def register_charging(self, name: str, fn: PlannerFn):
        self._charge[name.lower()] = fn

    def get_relocation(self, name: str) -> PlannerFn:
        key = name.lower()
        if key not in self._reloc:
            raise KeyError(f"Unknown relocation planner '{name}'.")
        return self._reloc[key]

    def get_charging(self, name: str) -> PlannerFn:
        key = name.lower()
        if key not in self._charge:
            raise KeyError(f"Unknown charging planner '{name}'.")
        return self._charge[key]

REGISTRY = PlannerRegistry()

# ---------- Built-in adapters (wrap your baseline functions) ----------
def _reloc_greedy_adapter(x, C, travel_min, *, params: Dict[str, Any]) -> Any:
    from control.baselines import plan_greedy
    return plan_greedy(x, C, travel_min, **params)

def _charge_greedy_adapter(x, s, chargers, lam_t, *, params: Dict[str, Any]) -> Any:
    from control.baselines import plan_charging_greedy
    return plan_charging_greedy(x, s, chargers, lam_t, **params)

REGISTRY.register_relocation("greedy", _reloc_greedy_adapter)
REGISTRY.register_charging("greedy", _charge_greedy_adapter)

# ---------- Optional: ML adapters ----------
def _load_dotted(dotted: str):
    """
    Load a callable/class from a dotted path like:
    'my_pkg.my_mod:MyPlannerClass' or 'my_pkg.my_mod:plan_fn'
    """
    if ":" in dotted:
        mod_name, attr = dotted.split(":", 1)
    else:
        # fallback: module that itself is callable
        mod_name, attr = dotted, None
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr) if attr else mod

def _reloc_ml_adapter(x, C, travel_min, *, params: Dict[str, Any]) -> Any:
    """
    params example:
      loader: "ml_pkg.reloc:RelocPlanner"
      kwargs: { checkpoint: "path/to.ckpt", ... }
    The loaded object must implement: plan(x, C, travel_min) -> list[(j,i,k)]
    """
    loader = params["loader"]
    kwargs = params.get("kwargs", {})
    obj = _load_dotted(loader)
    planner = obj(**kwargs) if callable(obj) and hasattr(obj, "__call__") else obj
    if hasattr(planner, "plan"):
        return planner.plan(x, C, travel_min)
    # fallback: assume it’s a function(x,C,travel_min,**kwargs)
    return obj(x, C, travel_min, **kwargs)

def _charge_ml_adapter(x, s, chargers, lam_t, *, params: Dict[str, Any]) -> Any:
    """
    params example:
      loader: "ml_pkg.charge:ChargePlanner"
      kwargs: {...}
    The loaded object must implement: plan(x, s, chargers, lam_t) -> np.ndarray[int]
    """
    loader = params["loader"]
    kwargs = params.get("kwargs", {})
    obj = _load_dotted(loader)
    planner = obj(**kwargs) if callable(obj) and hasattr(obj, "__call__") else obj
    if hasattr(planner, "plan"):
        return planner.plan(x, s, chargers, lam_t)
    return obj(x, s, chargers, lam_t, **kwargs)

REGISTRY.register_relocation("ml", _reloc_ml_adapter)
REGISTRY.register_charging("ml", _charge_ml_adapter)
