"""The loop — stages a hypothesis through the gates, debiting budget as it goes.

    register  ->  screen (free)  ->  backtest (COSTS BUDGET)  ->  verdict

The ordering is the entire design. Screening asks "does this variable carry
information about forward returns", which is not a search over configurations
and so carries no selection bias -- it costs nothing and kills most candidates.
Only survivors reach the backtest, which is where budget is actually spent.

This inverts what the old engine did. It backtested first and screened never,
which is how 94,658 trials bought zero legs.

The verdict is NOT "is PF above 1.2". It is:
    - Deflated Sharpe, charged for every trial spent so far, and
    - the MEDIAN cell of any sweep, never the best cell, and
    - trades/day, because a high-PF/low-frequency shape cannot pass a
      challenge no matter how real the edge is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .budget import BudgetExhausted, Ledger, deflated_sharpe
from .hypothesis import Hypothesis, Register, Status

# Promotion thresholds. Deliberately in one place and deliberately strict.
MIN_DSR = 0.95          # probability the edge survives the search that found it
MIN_TPD = 1.0           # HARD RULE 10: frequency, not just PF, passes challenges
MIN_PF = 1.2
MIN_ABS_IC = 0.015      # below this a feature is not worth a backtest


@dataclass
class Verdict:
    hypothesis_id: str
    passed: bool
    stage: str
    reason: str
    detail: dict


def screen(h: Hypothesis, feature: pd.Series, forward_return: pd.Series,
           folds: int = 5) -> Verdict:
    """FREE stage. Does the feature carry information about forward returns?

    Costs no trial budget: there is no max-picking over configurations here,
    only one question asked of one variable, so there is no selection bias to
    pay for. Screen as widely as you like.

    Requires a CONSISTENT sign across folds. A feature whose IC flips sign
    between periods has no stable relationship -- averaging it to a healthy
    number hides exactly the instability that kills a leg live.
    """
    df = pd.concat([feature.rename("x"), forward_return.rename("y")],
                   axis=1).dropna()
    if len(df) < folds * 50:
        return Verdict(h.id, False, "screen",
                       f"too few overlapping observations ({len(df)})", {})

    edges = np.linspace(0, len(df), folds + 1).astype(int)
    ics = []
    for i in range(folds):
        chunk = df.iloc[edges[i]:edges[i + 1]]
        if len(chunk) < 20 or chunk["x"].std() == 0 or chunk["y"].std() == 0:
            continue
        ics.append(chunk["x"].corr(chunk["y"], method="spearman"))
    if len(ics) < folds:
        return Verdict(h.id, False, "screen",
                       f"only {len(ics)} of {folds} folds usable", {"ics": ics})

    ics = np.array(ics, float)
    mean_ic = float(np.nanmean(ics))
    same_sign = bool(np.all(ics > 0) or np.all(ics < 0))
    detail = {"ics": [round(float(i), 4) for i in ics],
              "mean_ic": round(mean_ic, 4), "consistent_sign": same_sign}

    if not same_sign:
        return Verdict(h.id, False, "screen",
                       "IC sign flips across folds -- no stable relationship",
                       detail)
    if abs(mean_ic) < MIN_ABS_IC:
        return Verdict(h.id, False, "screen",
                       f"|IC| {abs(mean_ic):.4f} below {MIN_ABS_IC}", detail)
    return Verdict(h.id, True, "screen",
                   f"IC {mean_ic:+.4f}, sign consistent across {folds} folds",
                   detail)


def evaluate(h: Hypothesis, cell_returns: dict[str, pd.Series], *,
             ledger: Ledger, independence: float = 0.3,
             bars_per_year: float = 8760.0) -> Verdict:
    """COSTS BUDGET. Judge a swept strategy on the median cell, deflated.

    `cell_returns` maps a config label to that config's per-bar return series.
    Pass every cell you ran, not the good ones -- the whole point of the
    deflation is that it charges for the search, and hiding cells from it turns
    the number back into the headline PF that fooled us before.
    """
    if not cell_returns:
        return Verdict(h.id, False, "evaluate", "no cells supplied", {})

    try:
        spend = ledger.spend(h.universe, h.id, cells=len(cell_returns),
                             independence=independence,
                             note=h.claim[:80])
    except BudgetExhausted as e:
        return Verdict(h.id, False, "evaluate", str(e), {})
    except KeyError as e:
        return Verdict(h.id, False, "evaluate", str(e), {})

    budget = ledger.budgets[h.universe]
    n_trials_total = max(budget.spent, 1.0)

    per_cell = {}
    for label, r in cell_returns.items():
        r = pd.Series(r).dropna()
        if len(r) < 30 or r.std() == 0:
            continue
        sr_bar = float(r.mean() / r.std())
        gains = r[r > 0].sum()
        losses = -r[r < 0].sum()
        per_cell[label] = {
            "sr_bar": sr_bar,
            "sr_ann": sr_bar * np.sqrt(bars_per_year),
            "pf": float(gains / losses) if losses > 0 else float("inf"),
            "n": len(r),
        }
    if not per_cell:
        return Verdict(h.id, False, "evaluate", "no usable cells", {})

    srs = sorted(c["sr_bar"] for c in per_cell.values())
    pfs = sorted(c["pf"] for c in per_cell.values())
    median_sr = srs[len(srs) // 2]
    median_pf = pfs[len(pfs) // 2]
    best_sr = srs[-1]
    n_obs = int(np.median([c["n"] for c in per_cell.values()]))

    # charge the MEDIAN cell for the FULL search, not the best cell for one test
    dsr = deflated_sharpe(observed_sr=median_sr, n_trials=n_trials_total,
                          n_obs=n_obs)

    detail = {
        "cells": len(cell_returns),
        "charged": round(spend.charged, 2),
        "budget_spent": round(budget.spent, 1),
        "budget_left": round(budget.remaining, 1),
        "median_sr_bar": round(median_sr, 5),
        "best_sr_bar": round(best_sr, 5),
        "median_pf": round(median_pf, 4),
        "dsr": round(dsr, 4),
        "n_obs": n_obs,
        "selection_gap": round(best_sr - median_sr, 5),
    }

    if dsr < MIN_DSR:
        return Verdict(h.id, False, "evaluate",
                       f"DSR {dsr:.3f} < {MIN_DSR} after charging "
                       f"{n_trials_total:.0f} trials", detail)
    if median_pf < MIN_PF:
        return Verdict(h.id, False, "evaluate",
                       f"median-cell PF {median_pf:.3f} < {MIN_PF}", detail)
    return Verdict(h.id, True, "evaluate",
                   f"DSR {dsr:.3f}, median PF {median_pf:.3f}", detail)


def run(h: Hypothesis, *, register: Register, ledger: Ledger,
        feature=None, forward_return=None, cell_returns=None,
        tpd: float | None = None) -> Verdict:
    """Take one hypothesis as far through the gates as it can go."""
    if feature is not None and forward_return is not None:
        v = screen(h, feature, forward_return)
        register.update(h.id, screen_ic=v.detail.get("mean_ic"),
                        status=Status.SCREENED if v.passed else Status.REJECTED,
                        notes=h.notes + [f"screen: {v.reason}"])
        if not v.passed:
            return v

    if cell_returns is None:
        return Verdict(h.id, True, "screen",
                       "screened only -- no cells supplied to backtest", {})

    v = evaluate(h, cell_returns, ledger=ledger)
    register.update(h.id,
                    trials_spent=v.detail.get("charged", 0.0),
                    observed_sr=v.detail.get("median_sr_bar"),
                    dsr=v.detail.get("dsr"),
                    status=Status.TESTED if v.passed else Status.REJECTED,
                    notes=h.notes + [f"evaluate: {v.reason}"])
    if not v.passed:
        return v

    if tpd is not None:
        register.update(h.id, tpd=tpd)
        if tpd < MIN_TPD:
            register.update(h.id, status=Status.REJECTED)
            return Verdict(h.id, False, "frequency",
                           f"{tpd:.2f} trades/day < {MIN_TPD} -- cannot pass a "
                           f"challenge regardless of edge quality", v.detail)

    register.update(h.id, status=Status.PROMOTED)
    return Verdict(h.id, True, "promoted", v.reason, v.detail)
