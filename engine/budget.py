"""The trial budget — the one thing that makes every other result meaningful.

WHY THIS MODULE EXISTS
----------------------
The predecessor engine ran 94,658 backtests and shipped zero legs. That was not
bad luck or bad execution; it is the arithmetic working as designed.

Bailey, Borwein, Lopez de Prado & Zhu: if you test N independent configurations
and keep the best, the expected maximum IN-SAMPLE Sharpe of pure noise is

    E[max SR] ~= ( (1-g)*Z^-1[1 - 1/N] + g*Z^-1[1 - 1/(N*e)] ) / sqrt(T)

with g = Euler-Mascheroni. Setting the left side to 1.0 and solving for T gives
the Minimum Backtest Length: the years of data you need before an in-sample
Sharpe of 1 means anything at all.

    N =     45  ->  5.0 years needed
    N =     84  ->  6.1
    N =    342  ->  8.6
    N = 94,658  -> 19.2       <- what the old engine spent, on 5.1 years

So on 5.1 years of crypto, the honest budget is about 48 independent trials.
TOTAL. EVER. Not per session, not per idea family.

The consequence that matters: a faster engine is a WORSE engine, unless every
test it runs is debited from a budget that can run out. This module is that
budget. Nothing else here is novel; this is.

WHAT COUNTS AS A TRIAL
----------------------
One trial = one independent configuration evaluated against returns. A grid of
20 parameter cells is closer to 20 trials than to 1 -- they are correlated, not
free. `Ledger.spend()` takes the cell count and an `independence` factor for
that correlation, and refuses to pretend a sweep is a single test.

WHAT DOES *NOT* COST BUDGET
---------------------------
Feature screening. Asking "does this variable carry information about forward
returns" is not a strategy search -- there is no max-picking over configurations,
so there is no selection bias to pay for. This is Lopez de Prado's actual
position: backtesting is not a research tool, feature importance is. Screen
freely; backtest sparingly.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    from scipy.stats import norm
    _ppf = norm.ppf
except ImportError:                                              # pragma: no cover
    def _ppf(p: float) -> float:
        """Acklam's inverse-normal approximation; |error| < 1.15e-9."""
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        pl, ph = 0.02425, 1 - 0.02425
        if p < pl:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > ph:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

EULER = 0.5772156649015329


def expected_max_sharpe(n_trials: float) -> float:
    """E[max in-sample Sharpe] over `n_trials` independent trials of PURE NOISE.

    This is the number a strategy has to beat to be interesting at all. At
    N=1000 it is 3.26 -- so a backtest showing Sharpe 3 after a 1000-cell sweep
    is consistent with having found nothing whatsoever.
    """
    if n_trials < 2:
        return 0.0
    return ((1 - EULER) * _ppf(1 - 1.0 / n_trials)
            + EULER * _ppf(1 - 1.0 / (n_trials * math.e)))


def min_backtest_length(n_trials: float) -> float:
    """Years of data required before an IS Sharpe of 1 is meaningful."""
    return expected_max_sharpe(n_trials) ** 2


# Hard ceiling on any single universe's allowance. See trials_afforded().
MAX_ALLOWANCE = 1000


def trials_afforded(years: float, cap: int = MAX_ALLOWANCE) -> int:
    """Largest N whose MinBTL still fits inside `years`, CAPPED.

    The cap is not timidity, it is a correctness fix. MinBTL(N) grows like
    2*ln(N), so inverting it is EXPONENTIAL in years: N ~ exp(T/2). Measured on
    the real universes --

        5.1y  crypto, no breadth credit ->         47 trials
        10.5y crypto x sqrt(2.06)       ->        468
        12.9y multi-asset x sqrt(6.39)  ->      1,200
        98y   S&P daily                 -> 10^9 (numerically unbounded)

    -- so an uncapped budget hands a 98-year universe effectively infinite
    permission, and the guard silently stops guarding. That is worse than no
    guard, because it looks like one.

    Two things are true at once and both are respected here:
      1. More history really is worth enormously more than more tests. That is
         the honest content of the exponential, and it is why multi-market and
         long daily history matter.
      2. Past a few hundred trials the GLOBAL counter stops being the useful
         control, because the formula's assumption of independent trials on
         independent observations has long since broken down. Beyond that
         point the per-result Deflated Sharpe Ratio is the real gate, since it
         charges each individual result for the N actually spent.

    So: the counter binds early and hard, and `deflated_sharpe()` governs
    thereafter. Harvey & Liu's t>3.0 hurdle for a new factor implies the same
    order of magnitude -- hundreds, not millions.
    """
    if years <= 0:
        return 0
    lo, hi = 2.0, float(cap)
    if min_backtest_length(lo) > years:
        return 0
    if min_backtest_length(hi) <= years:
        return cap
    for _ in range(200):
        mid = (lo + hi) / 2
        if min_backtest_length(mid) <= years:
            lo = mid
        else:
            hi = mid
    return int(lo)


def deflated_sharpe(observed_sr: float, n_trials: float, n_obs: int,
                    skew: float = 0.0, kurtosis: float = 3.0,
                    sr_variance: float | None = None) -> float:
    """Probability the true Sharpe is > 0, after paying for `n_trials` of search.

    Bailey & Lopez de Prado's Deflated Sharpe Ratio. Returns a PROBABILITY in
    [0,1], not a Sharpe. Read it as: "the chance this survives having been
    cherry-picked". Conventionally you want >= 0.95.

    `observed_sr` and the result are per-observation, so pass a Sharpe computed
    on the same bar frequency as `n_obs`.
    """
    if n_obs < 2 or n_trials < 1:
        return float("nan")
    # variance of the Sharpe estimator across the trials searched
    var = sr_variance if sr_variance is not None else 1.0 / (n_obs - 1)
    sr0 = math.sqrt(var) * expected_max_sharpe(n_trials)
    denom = math.sqrt(
        max(1e-12,
            (1 - skew * observed_sr + (kurtosis - 1) / 4.0 * observed_sr ** 2)
            / (n_obs - 1))
    )
    z = (observed_sr - sr0) / denom
    # standard normal CDF
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


@dataclass
class Spend:
    """One debit against the budget."""
    at: str
    hypothesis_id: str
    cells: int
    independence: float
    charged: float
    note: str = ""


@dataclass
class Budget:
    """The budget for one DATA UNIVERSE.

    Separate universes get separate budgets, because MinBTL is a function of
    the history available. Testing on 98 years of S&P is genuinely cheaper per
    trial than testing on 5.1 years of crypto -- that is not a loophole, it is
    the actual mathematics, and it is the strongest argument for multi-market.
    """
    name: str
    years: float
    n_eff: float = 1.0
    spends: list[Spend] = field(default_factory=list)

    @property
    def effective_years(self) -> float:
        """Years x SQRT of independent breadth.

        N_eff measured 2026-08-10: crypto-only 2.06 across 11 symbols; adding
        FX, indices and commodities takes it to 6.39.

        Breadth is credited as sqrt(N_eff), NOT linearly. Grinold's fundamental
        law has information scaling with sqrt(breadth) -- IR = IC*sqrt(B) -- and
        crediting it linearly here inflated the multi-asset allowance to 99
        million trials, which is not a budget. sqrt is both the defensible
        reading and the one that keeps the guard binding.
        """
        return self.years * math.sqrt(max(self.n_eff, 1.0))

    @property
    def allowance(self) -> int:
        return trials_afforded(self.effective_years)

    @property
    def spent(self) -> float:
        return sum(s.charged for s in self.spends)

    @property
    def remaining(self) -> float:
        return self.allowance - self.spent

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def charge(self, cells: int, independence: float) -> float:
        """What `cells` configurations cost, given how correlated they are.

        independence=1.0  -> every cell is a fresh independent trial
        independence=0.0  -> the whole grid counts as ONE trial
        A parameter sweep over neighbouring values is usually 0.2-0.4: the cells
        are similar but not identical. Being generous here is how the old engine
        talked itself into 94,658 "free" tests.
        """
        if not 0.0 <= independence <= 1.0:
            raise ValueError("independence must be in [0,1]")
        if cells < 1:
            raise ValueError("cells must be >= 1")
        return 1.0 + (cells - 1) * independence


class Ledger:
    """Append-only record of every trial ever spent. Survives restarts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.budgets: dict[str, Budget] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self.path.read_text())
        for name, b in raw.items():
            self.budgets[name] = Budget(
                name=b["name"], years=b["years"], n_eff=b.get("n_eff", 1.0),
                spends=[Spend(**s) for s in b.get("spends", [])],
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {n: asdict(b) for n, b in self.budgets.items()}, indent=2))

    def universe(self, name: str, years: float, n_eff: float = 1.0) -> Budget:
        """Register or return a data universe."""
        if name not in self.budgets:
            self.budgets[name] = Budget(name=name, years=years, n_eff=n_eff)
            self.save()
        else:
            # A shorter current data window must immediately tighten the
            # allowance.  Never preserve an older, longer-history budget after
            # the research policy narrows the admissible sample.
            b = self.budgets[name]
            if years < b.years:
                b.years = years
                b.n_eff = min(b.n_eff, n_eff)
                self.save()
        return self.budgets[name]

    def spend(self, universe: str, hypothesis_id: str, cells: int,
              independence: float = 0.3, note: str = "") -> Spend:
        """Debit the budget. Raises if the universe is already exhausted.

        Deliberately raises rather than warning. A warning gets ignored; that is
        how 94,658 happened.
        """
        b = self.budgets.get(universe)
        if b is None:
            raise KeyError(f"unknown universe {universe!r} -- register it first")
        if b.exhausted:
            raise BudgetExhausted(
                f"universe {universe!r} is spent: {b.spent:.0f} of "
                f"{b.allowance} trials on {b.effective_years:.1f} effective "
                f"years. No verdict here is valid until there is more data, "
                f"a new feed, or genuinely independent markets."
            )
        s = Spend(at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  hypothesis_id=hypothesis_id, cells=cells,
                  independence=independence,
                  charged=b.charge(cells, independence), note=note)
        b.spends.append(s)
        self.save()
        return s

    def report(self) -> str:
        lines = ["universe          years  N_eff  eff.yrs  allowance   spent  left"]
        for b in self.budgets.values():
            lines.append(
                f"{b.name:16s} {b.years:6.2f} {b.n_eff:6.2f} {b.effective_years:8.2f} "
                f"{b.allowance:10d} {b.spent:7.1f} {b.remaining:6.1f}"
                + ("   EXHAUSTED" if b.exhausted else ""))
        return "\n".join(lines)


class BudgetExhausted(RuntimeError):
    """Raised when a universe has no honest trials left."""
