"""Robustness gates shared by harvested and invented strategies.

A large backtest on one market is a hypothesis generator, not an edge.  This
module keeps the deliberately small, explainable checks used before a result
can leave research: execution-cost stress, a chronological hold-out, and
agreement between the available market/timeframe scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import median


@dataclass(frozen=True)
class ExecutionCost:
    fee: float
    slippage: float


# Round-trip taker fees and deliberately adverse fill assumptions.  These are
# research assumptions, not broker quotes; the stress case is what prevents a
# thin paper edge from becoming a promotion.
COSTS = {
    "Crypto": ExecutionCost(fee=0.00055, slippage=0.00050),
    "FX": ExecutionCost(fee=0.00010, slippage=0.00010),
    "Stocks": ExecutionCost(fee=0.00025, slippage=0.00020),
    "Futures": ExecutionCost(fee=0.00020, slippage=0.00020),
}


def stress_cost(asset: str) -> ExecutionCost:
    """A 50% worse execution scenario; no strategy passes on optimistic fills."""
    base = COSTS.get(asset, COSTS["Crypto"])
    return ExecutionCost(fee=base.fee * 1.5, slippage=base.slippage * 1.5)


def assess(scenarios: list[dict]) -> dict:
    """Return a compact, serialisable robustness verdict.

    Each scenario contains ``name``, ``pf``, ``max_dd`` and ``trades``.  A
    scenario can be unavailable (for example no licensed data); unavailable is
    visible but is not silently counted as a pass.
    """
    measured = [s for s in scenarios if s.get("available")]
    passing = [s for s in measured if s.get("pf", 0) >= 1.05 and
               s.get("max_dd", 1) <= 0.25 and s.get("trades", 0) >= 30]
    pfs = [float(s["pf"]) for s in measured]
    dds = [float(s["max_dd"]) for s in measured]
    coverage = len(measured)
    pass_rate = len(passing) / coverage if coverage else 0.0
    spread = (max(pfs) - min(pfs)) if len(pfs) > 1 else None
    stable = bool(coverage >= 2 and pass_rate >= 0.75 and
                  (spread is None or spread <= 0.75) and max(dds, default=1) <= 0.25)
    score = round(min(10.0, pass_rate * 6 + min(coverage, 3) +
                      (1.0 if spread is not None and spread <= 0.40 else 0.0)), 1)
    best = max(measured, key=lambda s: s.get("pf", -999), default=None)
    return {
        "coverage": coverage,
        "passed": len(passing),
        "pass_rate": round(pass_rate, 3),
        "pf_median": round(median(pfs), 3) if pfs else None,
        "pf_spread": round(spread, 3) if spread is not None else None,
        "worst_dd": round(max(dds), 4) if dds else None,
        "stable": stable,
        "score": score,
        "best_scenario": best.get("name") if best else None,
        "scenarios": scenarios,
    }


def cost_dict(cost: ExecutionCost) -> dict:
    return asdict(cost)
