"""Idea 4 — regime classifier. Decide WHICH legs run now, not what they do.

THE THING THIS REPLACES
-----------------------
trading-bots gates its trend legs on one variable: DVOL percentile, off below
0.33. Measured 2026-08-10, that gate has been OFF for 100% of 132 consecutive
checks since deploy — BTC implied vol has sat at the 1st-4th percentile of 400
days for the whole period. A binary switch on a single variable is either
always on or always off for months at a time, and it silently took the trend
legs out of a nine-leg book without anyone deciding to.

A regime label built from several markets can distinguish "quiet and trending"
from "quiet and chopping", which one percentile cannot.

MULTI-ASSET IS THE POINT
------------------------
Regime is a property of the whole market, not of one coin. The features here
deliberately span asset classes — equity vol, credit, the dollar, the curve —
because measured N_eff for crypto alone is 2.06, i.e. one factor. You cannot
identify a regime from a single factor; you can only observe its level.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not predict returns, and it must never be fitted to them. A regime
model trained on forward performance is a return model wearing a hat, and it
will overfit exactly like one. This clusters OBSERVABLE STATE only. Which legs
belong in which regime is then measured separately, out of sample.

HARD RULE 8 applies with force: ADD, DON'T SUBTRACT. Using this to switch legs
OFF repeats the leg-dropping mistake that has failed blind four times. The
honest use is sizing and prioritisation — and a leg's regime fitness must be
judged at BOOK level, never on its own PF (HARD RULE 5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError as _e:                                        # pragma: no cover
    raise ImportError("regime classification needs scikit-learn") from _e


def build_features(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Observable market state from a wide close-price frame.

    Every column is backward-looking by construction — `rolling(...)` over
    closes up to and including t. Nothing here may reference t+1.
    """
    rets = prices.pct_change()
    f = pd.DataFrame(index=prices.index)

    f["vol"] = rets.std(axis=1).rolling(window).mean()
    f["vol_of_vol"] = rets.std(axis=1).rolling(window).std()
    f["dispersion"] = rets.std(axis=1) / rets.std(axis=1).rolling(252).mean()
    f["breadth"] = (rets > 0).mean(axis=1).rolling(window).mean()
    f["trend"] = (prices.pct_change(window).mean(axis=1))
    f["abs_trend"] = f["trend"].abs()

    # average pairwise correlation — the crisis tell. Everything correlating to
    # one is a different world from things moving independently, even at the
    # same volatility.
    corr = rets.rolling(window).corr()
    if isinstance(corr.index, pd.MultiIndex):
        n = prices.shape[1]
        mean_corr = corr.groupby(level=0).apply(
            lambda m: (m.values.sum() - n) / (n * n - n) if n > 1 else np.nan)
        f["mean_corr"] = mean_corr.reindex(f.index)

    return f.replace([np.inf, -np.inf], np.nan)


class RegimeModel:
    """K-means over observable state. Labels regimes, predicts nothing.

    K-means rather than an HMM on purpose: it has no transition matrix to
    overfit, it is deterministic given a seed, and it needs no extra
    dependency. Regime identification here is a labelling convenience, not a
    forecast, so the simpler estimator is the honest one.
    """

    def __init__(self, n_regimes: int = 4, random_state: int = 0):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_regimes, n_init=10,
                            random_state=random_state)
        self.columns_: list[str] = []

    def fit(self, features: pd.DataFrame) -> "RegimeModel":
        df = features.dropna()
        if len(df) < self.n_regimes * 30:
            raise ValueError(
                f"need >= {self.n_regimes * 30} clean rows, got {len(df)}")
        self.columns_ = list(df.columns)
        self.model.fit(self.scaler.fit_transform(df))
        return self

    def label(self, features: pd.DataFrame) -> pd.Series:
        """Regime id per bar. NaN where features are incomplete."""
        df = features[self.columns_]
        clean = df.dropna()
        lab = pd.Series(index=df.index, dtype=float, name="regime")
        if len(clean):
            lab.loc[clean.index] = self.model.predict(
                self.scaler.transform(clean))
        return lab

    def describe(self, features: pd.DataFrame) -> pd.DataFrame:
        """Mean of each feature per regime, so labels get readable names."""
        df = features[self.columns_].dropna()
        lab = self.label(df)
        return df.groupby(lab).mean().round(4)


def leg_fitness(regime: pd.Series, trade_times: pd.Series,
                trade_r: pd.Series, min_trades: int = 20) -> pd.DataFrame:
    """How a leg performed per regime — MEASURED, never assumed.

    `min_trades` guards the obvious trap: a regime containing 3 trades will
    show a spectacular or catastrophic PF that means nothing. Rows below the
    threshold are returned but flagged, never silently dropped, because a
    quietly missing regime looks identical to a regime that never occurred.
    """
    lab = regime.reindex(trade_times).values
    df = pd.DataFrame({"regime": lab, "r": trade_r.values}).dropna()
    rows = []
    for reg, g in df.groupby("regime"):
        gains, losses = g.r[g.r > 0].sum(), -g.r[g.r < 0].sum()
        rows.append({
            "regime": int(reg),
            "n": len(g),
            "pf": float(gains / losses) if losses > 0 else float("inf"),
            "win_pct": float((g.r > 0).mean() * 100),
            "total_r": float(g.r.sum()),
            "mean_r": float(g.r.mean()),
            "reliable": len(g) >= min_trades,
        })
    out = pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
    return out


def gate_report(fitness: pd.DataFrame) -> str:
    """Plain-language read of a fitness table, with the caveats attached."""
    if fitness.empty:
        return "no trades mapped to regimes"
    ok = fitness[fitness.reliable]
    if ok.empty:
        return (f"no regime has >= the minimum trade count "
                f"(largest is {int(fitness.n.max())}) — cannot conclude anything")
    best = ok.loc[ok.total_r.idxmax()]
    worst = ok.loc[ok.total_r.idxmin()]
    lines = [
        f"best regime  {int(best.regime)}: n={int(best.n)} PF={best.pf:.2f} "
        f"total_r={best.total_r:+.1f}",
        f"worst regime {int(worst.regime)}: n={int(worst.n)} PF={worst.pf:.2f} "
        f"total_r={worst.total_r:+.1f}",
    ]
    if len(ok) < len(fitness):
        thin = ", ".join(str(int(r)) for r in fitness[~fitness.reliable].regime)
        lines.append(f"regimes with too few trades to judge: {thin}")
    lines.append("ADD, DON'T SUBTRACT — use this to SIZE, not to switch a leg "
                 "off; leg-dropping has failed blind four times (HARD RULE 8), "
                 "and any change is decided at BOOK level (HARD RULE 5).")
    return "\n".join(lines)
