"""Deterministic adapters: foreign strategy code -> the `signals(df)` contract.

WHY DETERMINISTIC AND NOT AN LLM
--------------------------------
Adapting a freqtrade strategy is a SHAPE change, not a reasoning task: the class
always exposes `populate_indicators` plus an entry populator, and always marks
entries by setting a column. That is a mechanical mapping, and a mechanical
mapping belongs in code where it costs nothing per candidate and behaves the
same way every time.

The alternative — asking a model to "convert this to signals(df)" — reintroduces
exactly the failure this engine was built to avoid: a plausible function that
runs, produces a number, and quietly means something different from the source.

WHAT AN UNSUPPORTED SHAPE COSTS
-------------------------------
Nothing. It is skipped and marked. There is no repair loop and no fallback to a
model. A harvest of 5,000 files where 4,000 are unsupported is a fine outcome —
the 1,000 that adapt cleanly cost zero tokens, and the trial budget was never
the thing 4,000 extra candidates would have helped with.

STOPS
-----
`backtest.run_backtest` needs an absolute stop price per entry. Freqtrade states
risk as a fractional `stoploss` (e.g. -0.10). That converts exactly:

    long  stop = entry * (1 + stoploss)
    short stop = entry * (1 - stoploss)

When a strategy declares no usable stop we fall back to an ATR multiple rather
than inventing a tight one — HARD RULE 5, the stop-width illusion, means a
too-tight default does not merely mis-price the strategy, it flatters or damns
it depending on which side of the normalisation it lands.
"""
from __future__ import annotations

import ast
import sys
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd

# A stop this wide is almost certainly a missing stop rather than a real one.
MAX_STOP_FRAC = 0.35
DEFAULT_ATR_MULT = 2.5


@dataclass
class Adapted:
    ok: bool
    kind: str = ""
    code: str = ""
    reason: str = ""


# ----------------------------------------------------------------- shape detection
def detect(code: str) -> str:
    """Which adapter this source needs: 'native', 'freqtrade', or ''."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "signals":
            return "native"
    if "IStrategy" in code and (
            "populate_entry_trend" in code or "populate_buy_trend" in code):
        return "freqtrade"
    return ""


# ----------------------------------------------------------------- freqtrade stubs
def _stub_modules() -> dict[str, types.ModuleType]:
    """A minimal fake freqtrade surface, enough to import a strategy class.

    Deliberately narrow. Anything that reaches past this — exchange calls,
    hyperopt spaces, protections — raises on import and the candidate is
    reported unsupported instead of being half-executed.
    """
    mods: dict[str, types.ModuleType] = {}

    def mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        mods[name] = m
        return m

    class IStrategy:
        stoploss = -0.10
        timeframe = "1h"
        can_short = False

        def __init__(self, config=None):
            self.config = config or {}

    class _Param:
        """Stand-in for a hyperopt parameter, frozen at its default.

        Strategies read these BOTH ways — bare (`self.n`) and via the attribute
        (`self.n.value`) — so the stub has to be a real object with `.value`, not
        the raw number. Returning the number made every `.value` access raise
        `'int' object has no attribute 'value'`, which reads as a broken strategy
        when the only thing broken was the stub.

        The numeric dunders keep bare use working, so both spellings agree.
        """

        __slots__ = ("value",)

        def __init__(self, value):
            self.value = value

        # comparisons and arithmetic delegate to the wrapped default
        def __float__(self): return float(self.value)
        def __int__(self): return int(self.value)
        def __index__(self): return int(self.value)
        def __bool__(self): return bool(self.value)
        def __lt__(self, o): return self.value < o
        def __le__(self, o): return self.value <= o
        def __gt__(self, o): return self.value > o
        def __ge__(self, o): return self.value >= o
        def __eq__(self, o): return self.value == o
        def __ne__(self, o): return self.value != o
        def __hash__(self): return hash(self.value)
        def __add__(self, o): return self.value + o
        def __radd__(self, o): return o + self.value
        def __sub__(self, o): return self.value - o
        def __rsub__(self, o): return o - self.value
        def __mul__(self, o): return self.value * o
        def __rmul__(self, o): return o * self.value
        def __truediv__(self, o): return self.value / o
        def __rtruediv__(self, o): return o / self.value
        def __neg__(self): return -self.value
        def __repr__(self): return f"_Param({self.value!r})"

    def _param(*a, **k):
        """Freqtrade parameters resolve to their default outside hyperopt.

        The real signature is (low, high, default=..., space=...), so `default`
        must be read from the keyword — taking the first positional would silently
        substitute the search-space FLOOR for the strategy's chosen value and
        change what the code means.
        """
        if "default" in k:
            return _Param(k["default"])
        return _Param(a[0] if a else None)

    fs = mod("freqtrade.strategy")
    fs.IStrategy = IStrategy
    for n in ("IntParameter", "DecimalParameter", "RealParameter",
              "CategoricalParameter", "BooleanParameter"):
        setattr(fs, n, _param)
    fs.merge_informative_pair = lambda d, *a, **k: d
    fs.stoploss_from_open = lambda *a, **k: -0.10
    fs.informative = lambda *a, **k: (lambda f: f)

    ft = mod("freqtrade")
    ft.strategy = fs
    mod("freqtrade.persistence").Trade = object
    ex = mod("freqtrade.exchange")
    ex.timeframe_to_prev_date = lambda *a, **k: None
    ex.timeframe_to_minutes = lambda tf="1h", *a, **k: {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440
    }.get(tf, 60)
    ex.timeframe_to_seconds = lambda tf="1h", *a, **k: ex.timeframe_to_minutes(tf) * 60

    # qtpylib is vendored inside freqtrade and used by most community strategies.
    # These are the functions that actually appear in them; each is the standard
    # definition, so a strategy calling one gets the same numbers it would get
    # under real freqtrade rather than a silently different indicator.
    q = mod("freqtrade.vendor.qtpylib.indicators")
    mod("freqtrade.vendor")
    mod("freqtrade.vendor.qtpylib")

    def typical_price(bars):
        return (bars["high"] + bars["low"] + bars["close"]) / 3

    def bollinger_bands(series, window=20, stds=2):
        series = pd.Series(series)
        mid = series.rolling(window, min_periods=window).mean()
        sd = series.rolling(window, min_periods=window).std()
        return pd.DataFrame({"lower": mid - stds * sd, "mid": mid,
                             "upper": mid + stds * sd})

    def crossed_above(a, b):
        a, b = pd.Series(a), pd.Series(b)
        return (a > b) & (a.shift(1) <= b.shift(1))

    def crossed_below(a, b):
        a, b = pd.Series(a), pd.Series(b)
        return (a < b) & (a.shift(1) >= b.shift(1))

    q.typical_price = typical_price
    q.bollinger_bands = bollinger_bands
    q.crossed_above = crossed_above
    q.crossed_below = crossed_below
    q.crossed = lambda a, b: crossed_above(a, b) | crossed_below(a, b)
    q.sma = lambda s, window=20, **k: pd.Series(s).rolling(window, min_periods=window).mean()
    q.ema = lambda s, window=20, **k: pd.Series(s).ewm(span=window, adjust=False).mean()
    q.rolling_mean = q.sma
    q.rolling_std = lambda s, window=20, **k: pd.Series(s).rolling(window, min_periods=window).std()
    q.returns = lambda s, **k: pd.Series(s).pct_change()

    return mods


def _install(mods: dict[str, types.ModuleType]) -> list[str]:
    added = []
    for name, m in mods.items():
        if name not in sys.modules:
            sys.modules[name] = m
            added.append(name)
    return added


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def run_freqtrade(code: str, df: pd.DataFrame) -> list[tuple[int, int, float]]:
    """Execute a freqtrade strategy class against `df` and emit signals.

    Runs the populators once over the whole frame, which is how freqtrade itself
    backtests. `verify()` still applies the truncated-frame look-ahead check
    afterwards, so a strategy that peeks is caught here exactly as a translated
    one would be.
    """
    mods = _stub_modules()
    added = _install(mods)
    try:
        ns: dict = {}
        exec(compile(code, "<freqtrade-strategy>", "exec"), ns)  # noqa: S102
        cls = next((v for v in ns.values()
                    if isinstance(v, type)
                    and any(b.__name__ == "IStrategy" for b in v.__mro__[1:])), None)
        if cls is None:
            raise ValueError("no IStrategy subclass found")
        strat = cls()

        work = df.rename(columns={"start": "date"}).copy()
        # Strategies routinely use `dataframe['date'].dt...` for session filters.
        # `start` is epoch-ms, so it must become a real datetime or every one of
        # those raises — and a raise here reads as "strategy is broken" when the
        # only thing wrong was the column dtype.
        if "date" in work.columns and not pd.api.types.is_datetime64_any_dtype(work["date"]):
            work["date"] = pd.to_datetime(work["date"], unit="ms", utc=True)
        meta = {"pair": "BTC/USDT"}
        if hasattr(strat, "populate_indicators"):
            work = strat.populate_indicators(work, meta)
        if hasattr(strat, "populate_entry_trend"):
            work = strat.populate_entry_trend(work, meta)
        elif hasattr(strat, "populate_buy_trend"):
            work = strat.populate_buy_trend(work, meta)
        else:
            raise ValueError("no entry populator")

        long_col = next((c for c in ("enter_long", "buy") if c in work.columns), None)
        short_col = next((c for c in ("enter_short", "sell_short") if c in work.columns), None)
        if long_col is None and short_col is None:
            raise ValueError("entry populator set no entry column")

        sl = float(getattr(strat, "stoploss", -0.10) or -0.10)
        if not np.isfinite(sl) or not (-MAX_STOP_FRAC <= sl < 0):
            sl = 0.0

        atr = _atr(df)
        close = df["close"].values
        out: list[tuple[int, int, float]] = []
        for col, direction in ((long_col, 1), (short_col, -1)):
            if col is None:
                continue
            flag = (pd.to_numeric(work[col], errors="coerce")
                    .fillna(0).astype(float).values > 0)
            # RISING EDGE ONLY. Freqtrade populators mark every bar on which the
            # entry CONDITION holds, and the bot enters once because it tracks
            # whether a position is already open. `signals()` has no such state,
            # so taking every flagged bar turns "RSI below 30" into an entry on
            # each of the 40 bars it stays there — which is why these strategies
            # read as "fires on 67% of bars". The transition is the entry event.
            idx = np.flatnonzero(flag & ~np.r_[False, flag[:-1]])
            for i in idx:
                px = close[i]
                if not np.isfinite(px) or px <= 0:
                    continue
                if sl:
                    stop = px * (1 + sl) if direction == 1 else px * (1 - sl)
                else:
                    a = atr.iloc[i]
                    if not np.isfinite(a) or a <= 0:
                        continue
                    stop = (px - DEFAULT_ATR_MULT * a if direction == 1
                            else px + DEFAULT_ATR_MULT * a)
                if stop <= 0 or abs(stop - px) / px > MAX_STOP_FRAC:
                    continue
                out.append((int(i), direction, float(stop)))
        out.sort(key=lambda t: t[0])
        return out
    finally:
        for name in added:
            sys.modules.pop(name, None)


# ----------------------------------------------------------------- public entry
_WRAPPER = '''\
# auto-adapted from a freqtrade strategy — see engine/adapters.py
from engine.adapters import run_freqtrade

_SRC = {src!r}


def signals(df):
    return run_freqtrade(_SRC, df)
'''


def adapt(code: str) -> Adapted:
    """Turn foreign strategy source into something exposing `signals(df)`.

    SECURITY: adapting a freqtrade strategy means EXECUTING code downloaded from
    a stranger's repository in this process. That is a real risk and it is not
    made safe by the fact that the code looks like a trading strategy. Foreign
    source is scanned with translate's _FORBIDDEN pattern BEFORE it is stored or
    run, so filesystem, network and process access are rejected at harvest time
    rather than discovered at exec time.

    This is a filter, not a sandbox. It stops the obvious cases. Anything that
    matters more than a research box is worth running in a container.
    """
    from .translate import _FORBIDDEN
    hit = _FORBIDDEN.search(code)
    if hit:
        return Adapted(False, "", "", f"rejected unsafe construct: {hit.group(0)!r}")
    kind = detect(code)
    if kind == "native":
        return Adapted(True, "native", code)
    if kind == "freqtrade":
        return Adapted(True, "freqtrade", _WRAPPER.format(src=code))
    return Adapted(False, "", "", "unsupported shape — no signals() and not a "
                                  "recognisable freqtrade strategy")
