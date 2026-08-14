"""Strict BTC test of the user-supplied TradingView ORB indicator.

Rules fixed for this test:
  * BTCUSDT 5-minute bars, Mon–Fri, 09:30–09:45 America/New_York range.
  * ``breakout`` enters after the source's three-bar confirmed breakout.
  * ``retest`` waits for the source's confirmed retest after that breakout.
  * stop = the opposite edge of that day's opening range (one R);
    target = two R.  Entries fill at next open with costs.

The source only draws labels, so these execution choices are stated here rather
than being smuggled in as if the original author had backtested them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import bridge
from .robustness import COSTS, stress_cost


@dataclass
class OrbResult:
    variant: str
    scenario: str
    trades: int
    pf: float
    tpd: float
    dd_pct: float
    total_r: float


def _session(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    local = pd.to_datetime(df["start"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    weekday = local.dt.weekday < 5
    minute = local.dt.hour * 60 + local.dt.minute
    in_range = weekday & (minute >= 9 * 60 + 30) & (minute < 9 * 60 + 45)
    return local.dt.date, in_range


def intents(df: pd.DataFrame, variant: str) -> list[tuple[int, int, float, float]]:
    """Translate the source's confirmed-breakout and retest labels to orders."""
    if variant not in {"breakout", "retest"}:
        raise ValueError("variant must be breakout or retest")
    dates, in_range = _session(df)
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(df)
    high = low = np.nan
    completed: dict[object, tuple[float, float]] = {}
    pending = 0
    out: list[tuple[int, int, float, float]] = []

    for i in range(n):
        day = dates.iloc[i]
        if in_range.iloc[i]:
            if i == 0 or dates.iloc[i - 1] != day or not in_range.iloc[i - 1]:
                high, low, pending = h[i], l[i], 0
            else:
                high, low = max(high, h[i]), min(low, l[i])
            continue
        # The range is usable only after its final bar has closed.
        if i > 0 and dates.iloc[i - 1] == day and in_range.iloc[i - 1]:
            completed[day] = (high, low)
        if day not in completed or i < 2:
            continue
        orb_high, orb_low = completed[day]

        # Exact source confirmation, minus presentation-only conditions.
        long_bo = (l[i - 2] < orb_high and c[i - 2] > orb_high and
                   l[i - 1] > orb_high and c[i - 1] > orb_high and
                   c[i] > l[i - 1] and l[i] > orb_high)
        short_bo = (h[i - 2] > orb_low and c[i - 2] < orb_low and
                    h[i - 1] < orb_low and c[i - 1] < orb_low and
                    c[i] < h[i - 1] and h[i] < orb_low)
        long_retest = pending == 1 and c[i - 1] > orb_high and l[i] <= orb_high and c[i] >= orb_high
        short_retest = pending == -1 and c[i - 1] < orb_low and h[i] >= orb_low and c[i] <= orb_low
        side = 0
        if variant == "breakout":
            side = 1 if long_bo else -1 if short_bo else 0
        else:
            if long_bo:
                pending = 1
            elif short_bo:
                pending = -1
            if long_retest:
                side, pending = 1, 0
            elif short_retest:
                side, pending = -1, 0

        if side:
            stop = orb_low if side == 1 else orb_high
            # Target is based on the intended next-open entry price in the
            # backtester; it accepts a target level, so calculate using the
            # closing signal price as a conservative, known-at-signal proxy.
            risk = abs(c[i] - stop)
            target = c[i] + side * 2.0 * risk
            if (side == 1 and stop < c[i] < target) or (side == -1 and stop > c[i] > target):
                out.append((i, side, float(stop), float(target)))
    return out


def _summary(df: pd.DataFrame, orders, *, fee: float, slippage: float,
             variant: str, scenario: str) -> OrbResult:
    trades = bridge.run_backtest(df, orders, rr=2.0, fee=fee, slippage=slippage,
                                 cooldown=0, max_hold=96)
    s = bridge.stats(trades, df)
    return OrbResult(variant, scenario, s["trades"], s["profit_factor"],
                     s["trades_per_day"], s["max_dd_pct"], s["total_r"])


def evaluate(df: pd.DataFrame) -> list[OrbResult]:
    base, stressed = COSTS["Crypto"], stress_cost("Crypto")
    rows: list[OrbResult] = []
    for variant in ("breakout", "retest"):
        orders = intents(df, variant)
        rows.append(_summary(df, orders, fee=base.fee, slippage=base.slippage,
                             variant=variant, scenario="base"))
        rows.append(_summary(df, orders, fee=stressed.fee, slippage=stressed.slippage,
                             variant=variant, scenario="higher costs"))
        holdout = df.iloc[int(len(df) * 0.6):].reset_index(drop=True)
        rows.append(_summary(holdout, intents(holdout, variant), fee=base.fee,
                             slippage=base.slippage, variant=variant,
                             scenario="last 40%"))
    return rows


if __name__ == "__main__":
    data = bridge.fetch_crypto("BTCUSDT", "5m", days=int(365.25 * 5))
    for r in evaluate(data):
        print(f"{r.variant:8} {r.scenario:12} trades={r.trades:4} PF={r.pf:5.2f} "
              f"TPD={r.tpd:4.2f} DD={r.dd_pct:5.1f}% totalR={r.total_r:6.1f}")
