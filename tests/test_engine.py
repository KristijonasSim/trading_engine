"""Does the referee actually referee?

The one property worth testing: fed PURE NOISE and allowed to search hard, the
pipeline must REJECT. The predecessor engine's failure was that a big enough
sweep over noise always produces a great-looking headline, and nothing in the
process said no. If these tests pass, that specific failure cannot recur.

Run:  python tests/test_engine.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from engine.budget import (BudgetExhausted, Ledger, deflated_sharpe,
                           expected_max_sharpe, min_backtest_length,
                           trials_afforded)
from engine.hypothesis import (Mechanism, RejectedHypothesis, Register)
from engine.pipeline import evaluate, screen

PASS, FAIL = "  PASS", "  FAIL"
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL}  {name}{('  -- ' + detail) if detail else ''}")


# ----------------------------------------------------------------- budget
def test_budget_matches_published_table():
    """The MinBTL figures must match Bailey et al. as quoted in
    trading-bots/RESEARCH_METHOD.md, or every downstream number is wrong."""
    for n, expected in ((45, 5.0), (84, 6.1), (342, 8.6), (94658, 19.2)):
        got = min_backtest_length(n)
        check(f"MinBTL(N={n}) == {expected}y", abs(got - expected) < 0.05,
              f"got {got:.2f}")
    check("E[max SR] rises with N",
          expected_max_sharpe(10) < expected_max_sharpe(1000))


def test_allowance_binds():
    """An uncapped allowance is not a budget. 98 years must not buy infinity."""
    check("5.1y crypto affords ~47 trials", trials_afforded(5.1) == 47,
          f"got {trials_afforded(5.1)}")
    check("98y is capped, not unbounded", trials_afforded(98.0) == 1000,
          f"got {trials_afforded(98.0)}")


def test_budget_refuses_when_spent():
    with tempfile.TemporaryDirectory() as d:
        led = Ledger(Path(d) / "ledger.json")
        led.universe("crypto", years=5.1, n_eff=1.0)
        allowance = led.budgets["crypto"].allowance
        # spend it all in independent trials
        led.spend("crypto", "h1", cells=allowance + 5, independence=1.0)
        check("budget reports exhausted", led.budgets["crypto"].exhausted)
        try:
            led.spend("crypto", "h2", cells=1)
            check("further spend RAISES", False, "no exception")
        except BudgetExhausted:
            check("further spend RAISES", True)


def test_ledger_survives_restart():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ledger.json"
        led = Ledger(p)
        led.universe("crypto", years=5.1)
        led.spend("crypto", "h1", cells=10, independence=0.3)
        spent = led.budgets["crypto"].spent
        check("spend persists across reload",
              abs(Ledger(p).budgets["crypto"].spent - spent) < 1e-9)


# ------------------------------------------------------------- deflation
def test_dsr_charges_for_search():
    """Same result, more search, lower confidence. This is the core mechanic."""
    a = deflated_sharpe(0.12, n_trials=1, n_obs=1250)
    b = deflated_sharpe(0.12, n_trials=1000, n_obs=1250)
    c = deflated_sharpe(0.12, n_trials=94658, n_obs=1250)
    check("DSR falls as trials rise", a > b > c, f"{a:.3f} > {b:.3f} > {c:.3f}")
    check("94,658 trials kills a SR-0.12 result", c < 0.95, f"{c:.3f}")


# ---------------------------------------------------------------- referee
def test_register_rejects_bad_hypotheses():
    with tempfile.TemporaryDirectory() as d:
        reg = Register(Path(d) / "reg.json")
        ok = dict(mechanism=Mechanism.FORCED_FLOW, feed="cftc_cot",
                  universe="fx", prediction="net-short extremes precede a 2% "
                  "reversal within 10 sessions", null="no relationship")

        try:
            reg.register(id="a", claim="momentum works", **ok)
            check("rejects a claim too short to falsify", False)
        except RejectedHypothesis:
            check("rejects a claim too short to falsify", True)

        try:
            reg.register(id="b",
                         claim="leveraged funds at positioning extremes in FX "
                               "futures tend to mean revert over ten sessions",
                         mechanism=Mechanism.FORCED_FLOW, feed="cftc_cot",
                         universe="fx",
                         prediction="it should work pretty well",
                         null="no relationship")
            check("rejects hedged predictions", False)
        except RejectedHypothesis:
            check("rejects hedged predictions", True)

        try:
            reg.register(id="c",
                         claim="a 173 day moving average crossover on bitcoin "
                               "hourly bars produces positive expectancy",
                         mechanism=Mechanism.STRUCTURAL, feed="ohlcv",
                         universe="crypto",
                         prediction="+0.3 PF versus baseline",
                         null="no edge")
            check("rejects an already-exploited feed", False)
        except RejectedHypothesis:
            check("rejects an already-exploited feed", True)

        h = reg.register(id="d",
                         claim="leveraged funds at COT positioning extremes in "
                               "FX futures mean revert over the following ten "
                               "sessions",
                         **ok)
        check("admits a well-formed hypothesis", h.id == "d")

        try:
            reg.register(id="e",
                         claim="leveraged funds at COT positioning extremes in "
                               "FX futures mean revert over the next ten sessions",
                         **ok)
            check("rejects a near-duplicate", False)
        except RejectedHypothesis:
            check("rejects a near-duplicate", True)


# ------------------------------------------------- THE test that matters
def test_pure_noise_is_rejected():
    """Search 200 random cells on pure noise and demand the pipeline say no.

    This is precisely the 94,658-backtest failure, reproduced in miniature. The
    best cell WILL look good -- that is guaranteed by the maths, not bad luck.
    The engine must reject it anyway.
    """
    rng = np.random.default_rng(7)
    n_obs, n_cells = 1250, 200
    cells = {f"cell{i}": pd.Series(rng.normal(0, 0.01, n_obs))
             for i in range(n_cells)}

    best = max(c.mean() / c.std() for c in cells.values())
    check("noise search produces a flattering best cell", best > 0.05,
          f"best SR/bar {best:.4f}")

    with tempfile.TemporaryDirectory() as d:
        led = Ledger(Path(d) / "l.json")
        led.universe("crypto", years=5.1, n_eff=1.0)
        reg = Register(Path(d) / "r.json")
        h = reg.register(
            id="noise", mechanism=Mechanism.FORCED_FLOW, feed="cftc_cot",
            universe="crypto",
            claim="a synthetic control that should be rejected by construction",
            prediction="+0.5 Sharpe improvement over baseline",
            null="returns are noise")
        v = evaluate(h, cells, ledger=led, independence=1.0)
        check("PIPELINE REJECTS PURE NOISE", not v.passed, v.reason[:70])
        check("selection gap is reported", v.detail.get("selection_gap", 0) > 0,
              f"best-median = {v.detail.get('selection_gap')}")


def test_real_signal_survives_a_small_search():
    """The guard must not reject everything -- a genuine edge, lightly searched,
    has to get through, or the engine is just an expensive `return False`."""
    rng = np.random.default_rng(3)
    n_obs = 2000
    cells = {f"cell{i}": pd.Series(rng.normal(0.0022, 0.01, n_obs))
             for i in range(4)}
    with tempfile.TemporaryDirectory() as d:
        led = Ledger(Path(d) / "l.json")
        led.universe("multi", years=5.1, n_eff=6.39)
        reg = Register(Path(d) / "r.json")
        h = reg.register(
            id="real", mechanism=Mechanism.FORCED_FLOW, feed="cftc_cot",
            universe="multi",
            claim="a synthetic positive-drift control representing a true edge",
            prediction="+0.2 Sharpe per bar", null="returns are noise")
        v = evaluate(h, cells, ledger=led, independence=0.3)
        check("PIPELINE ACCEPTS A REAL EDGE", v.passed, v.reason[:70])


def test_screen_costs_nothing_and_catches_sign_flips():
    rng = np.random.default_rng(11)
    n = 1500
    with tempfile.TemporaryDirectory() as d:
        reg = Register(Path(d) / "r.json")
        h = reg.register(
            id="s", mechanism=Mechanism.INFORMATION, feed="fred_dgs10",
            universe="fx",
            claim="the ten year yield level predicts next day dollar direction",
            prediction="+0.03 IC, positive in every fold", null="zero IC")

        x = pd.Series(rng.normal(size=n))
        flip = pd.Series(np.r_[x[:n // 2] * 1.0, -x[n // 2:] * 1.0].ravel())
        y_flip = flip * 0.3 + rng.normal(0, 1, n) * 0.9
        check("screen rejects a sign-flipping feature",
              not screen(h, x, pd.Series(y_flip)).passed)

        y_good = x * 0.15 + rng.normal(0, 1, n)
        check("screen accepts a stable feature", screen(h, x, y_good).passed)


if __name__ == "__main__":
    print("budget")
    test_budget_matches_published_table()
    test_allowance_binds()
    test_budget_refuses_when_spent()
    test_ledger_survives_restart()
    print("\ndeflation")
    test_dsr_charges_for_search()
    print("\nreferee")
    test_register_rejects_bad_hypotheses()
    print("\nscreen")
    test_screen_costs_nothing_and_catches_sign_flips()
    print("\nthe test that matters")
    test_pure_noise_is_rejected()
    test_real_signal_survives_a_small_search()

    n, total = sum(_results), len(_results)
    print(f"\n{n}/{total} checks passed")
    sys.exit(0 if n == total else 1)
