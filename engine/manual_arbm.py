"""Strict first-pass test for the user-supplied Adaptive Regression Breakout Map.

The published Pine is an *indicator*, not a TradingView strategy.  It records
an entry at a bar's close then immediately inspects that same bar's high/low
for targets.  That cannot be traded honestly.  This file preserves the stated
breakout/squeeze mechanics but uses the engine's execution policy: a signal at
close fills at the next open; fees and slippage apply; an intrabar stop fills at
the breaching bar's close.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import bridge
from .robustness import COSTS
from .runner import UNIVERSES, _load_bars


@dataclass
class Result:
    symbol: str
    trades: int
    pf: float | None
    tpd: float
    max_dd: float | None
    total_r: float


def _linreg_end(close: pd.Series, length: int = 50) -> pd.Series:
    """Pine ``ta.linreg(close, length, 0)``: fitted value at the latest bar."""
    x = np.arange(length, dtype=float)
    dx = x - x.mean()
    denom = float(np.dot(dx, dx))

    def endpoint(a: np.ndarray) -> float:
        if not np.isfinite(a).all():
            return np.nan
        slope = float(np.dot(dx, a - a.mean()) / denom)
        return float(a.mean() + slope * (length - 1 - x.mean()))

    return close.rolling(length).apply(endpoint, raw=True)


def _bands(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = df["close"].astype(float)
    basis = _linreg_end(close)
    # Pine's ta.stdev uses the population convention by default.
    std = close.rolling(50).std(ddof=0)
    upper, lower = basis + 2.0 * std, basis - 2.0 * std
    width = upper - lower
    lo, hi = width.rolling(100).min(), width.rolling(100).max()
    width_pct = (width - lo) / (hi - lo).clip(lower=0.000001) * 100.0
    return upper, lower, width_pct <= 20.0


def test_one(df: pd.DataFrame, *, fee: float, slippage: float) -> np.ndarray:
    """Return per-trade R, with ARBM's TP1/TP2 stop ratchet and TP3 target."""
    upper, lower, squeeze = _bands(df)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    up, down, sq = (x.to_numpy() for x in (upper, lower, squeeze.fillna(False)))
    results: list[float] = []
    free_after = 0
    n = len(df)

    for i in range(1, n - 1):
        if i < free_after or not (np.isfinite(up[i]) and np.isfinite(down[i])):
            continue
        side = 1 if c[i] > up[i] and c[i - 1] <= up[i - 1] and sq[i - 1] else \
               -1 if c[i] < down[i] and c[i - 1] >= down[i - 1] and sq[i - 1] else 0
        if side == 0:
            continue

        # Signal is known after bar i closes; fill the next bar's open.
        entry_i = i + 1
        entry = o[entry_i] * (1 + side * slippage)
        initial_stop = down[i] if side == 1 else up[i]
        risk = abs(entry - initial_stop)
        # A gap can put the next-open fill beyond the source's bracket.  It is
        # not a valid trade; treating the stop as an absolute distance would
        # accidentally turn a stop above a long entry into a plausible result.
        if (not np.isfinite(risk) or risk <= 0 or
                (side == 1 and initial_stop >= entry) or
                (side == -1 and initial_stop <= entry)):
            continue
        band_width = up[i] - down[i]
        # Preserve the author's target architecture, but do not let this bar's
        # high/low decide an outcome before the next-open fill.
        tp1 = c[i] + side * band_width
        tp2 = c[i] + side * band_width * 2
        tp3 = c[i] + side * band_width * 3
        if (side == 1 and tp3 <= entry) or (side == -1 and tp3 >= entry):
            continue
        stop = initial_stop
        tp1_hit = tp2_hit = False
        exit_px = c[-1]
        exit_i = n - 1

        for j in range(entry_i, n):
            hit_stop = (side == 1 and l[j] <= stop) or (side == -1 and h[j] >= stop)
            # Conservative bar ordering: if a bar touches both the stop and a
            # target, stop wins.  The source's optimistic order is not used.
            if hit_stop:
                exit_px, exit_i = c[j], j
                break
            hit1 = (side == 1 and h[j] >= tp1) or (side == -1 and l[j] <= tp1)
            hit2 = (side == 1 and h[j] >= tp2) or (side == -1 and l[j] <= tp2)
            hit3 = (side == 1 and h[j] >= tp3) or (side == -1 and l[j] <= tp3)
            if hit1 and not tp1_hit:
                tp1_hit, stop = True, c[i]
            if hit2 and not tp2_hit:
                tp2_hit, stop = True, tp1
            if hit3:
                exit_px, exit_i = c[j], j
                break

        gross = side * (exit_px / entry - 1.0)
        fee_cost = fee * (1.0 + exit_px / entry)
        results.append((gross - fee_cost) / (risk / entry))
        free_after = exit_i + 1
    return np.asarray(results, dtype=float)


def evaluate(asset_class: str) -> list[Result]:
    cfg, cost = UNIVERSES[asset_class], COSTS[asset_class]
    out: list[Result] = []
    for symbol in cfg["symbols"]:
        df = _load_bars(symbol, cfg)
        r = test_one(df, fee=cost.fee, slippage=cost.slippage)
        gains, losses = r[r > 0].sum(), -r[r < 0].sum()
        pf = float(gains / losses) if losses > 0 else None
        equity = np.cumsum(r)
        dd = ((np.maximum.accumulate(equity) - equity).max() /
              max(abs(equity).max(), 1.0)) if len(r) else None
        days = max((df["start"].iloc[-1] - df["start"].iloc[0]) / 86_400_000, 1)
        out.append(Result(symbol, int(len(r)), pf, len(r) / days, dd,
                          float(r.sum()) if len(r) else 0.0))
    return out


def format_results(groups: Iterable[str] = ("Crypto", "FX", "Stocks", "Futures")) -> str:
    lines = ["Adaptive Regression Breakout Map — strict five-year first pass"]
    for group in groups:
        rows = evaluate(group)
        for x in rows:
            lines.append(f"{group:7} {x.symbol:10} trades {x.trades:4}  PF "
                         f"{x.pf if x.pf is not None else '—':>5}  "
                         f"TPD {x.tpd:.3f}  DD "
                         f"{x.max_dd if x.max_dd is not None else '—'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_results())
