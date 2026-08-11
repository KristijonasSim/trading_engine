"""Transparent, deterministic strategy invention.

These are seed hypotheses made from known mechanics, not performance claims.
Every invented row writes the exact Python signal function used in testing, so
it can be inspected, reproduced, changed, or deleted like any other source.
Variants are intentionally few: creating thousands of parameter combinations
would spend the statistical budget before we learned anything.
"""
from __future__ import annotations

from pathlib import Path

from .harvest import Candidate, CandidateStore

ROOT = Path(__file__).resolve().parent.parent / "state" / "invented"


def _code(kind: str, fast: int, slow: int, threshold: float = 0.0) -> str:
    # The functions return (bar_index, side, stop) intents understood by the
    # shared trading-bots backtester. Stops use only information available at
    # the signal bar. No generated strategy gets special execution privileges.
    if kind == "trend":
        signal = f"fast = close.ewm(span={fast}, adjust=False).mean()\n    slow = close.ewm(span={slow}, adjust=False).mean()\n    long = (fast > slow) & (fast.shift(1) <= slow.shift(1))\n    short = (fast < slow) & (fast.shift(1) >= slow.shift(1))"
    elif kind == "mean_reversion":
        signal = f"mid = close.rolling({slow}).mean()\n    sd = close.rolling({slow}).std()\n    z = (close - mid) / sd.replace(0, float('nan'))\n    long = z < -{threshold or 1.8}\n    short = z > {threshold or 1.8}"
    elif kind == "breakout":
        signal = f"high_band = high.shift(1).rolling({slow}).max()\n    low_band = low.shift(1).rolling({slow}).min()\n    long = close > high_band\n    short = close < low_band"
    elif kind == "momentum":
        signal = f"mom = close.pct_change({fast})\n    long = mom > {threshold or 0.01}\n    short = mom < -{threshold or 0.01}"
    elif kind == "volume_breakout":
        signal = f"high_band = high.shift(1).rolling({slow}).max()\n    low_band = low.shift(1).rolling({slow}).min()\n    vol_ok = volume > volume.rolling({fast}).mean() * 1.2\n    long = (close > high_band) & vol_ok\n    short = (close < low_band) & vol_ok"
    else:  # structure proxy: breakout after a longer pivot range
        signal = f"high_band = high.shift(1).rolling({slow}).max()\n    low_band = low.shift(1).rolling({slow}).min()\n    long = close > high_band\n    short = close < low_band"
    return f'''import numpy as np

def signals(df):
    close, high, low, volume = (df["close"].astype(float), df["high"].astype(float),
                                df["low"].astype(float), df["volume"].astype(float))
    {signal}
    atr = (high - low).rolling(14).mean().bfill()
    out = []
    for i in range(len(df) - 1):
        if bool(long.iloc[i]):
            out.append((i, 1, close.iloc[i] - 1.5 * atr.iloc[i]))
        elif bool(short.iloc[i]):
            out.append((i, -1, close.iloc[i] + 1.5 * atr.iloc[i]))
    return out
'''


BLUEPRINTS = [
    ("trend", 20, 80, 0.0, "Trend EMA 20/80"),
    ("trend", 50, 200, 0.0, "Trend EMA 50/200"),
    ("mean_reversion", 0, 20, 1.8, "Mean reversion z-score 20"),
    ("mean_reversion", 0, 40, 2.0, "Mean reversion z-score 40"),
    ("breakout", 0, 20, 0.0, "Donchian breakout 20"),
    ("breakout", 0, 55, 0.0, "Donchian breakout 55"),
    ("momentum", 10, 0, 0.01, "Momentum 10"),
    ("volume_breakout", 20, 30, 0.0, "Volume-confirmed breakout 30"),
    ("structure", 0, 50, 0.0, "Structure range breakout 50"),
]


def seed(store: CandidateStore | None = None) -> dict:
    """Ensure the initial, auditable invented research queue exists."""
    st = store or CandidateStore()
    ROOT.mkdir(parents=True, exist_ok=True)
    added = 0
    for kind, fast, slow, threshold, name in BLUEPRINTS:
        cid = f"inv:{kind}-{fast}-{slow}-{str(threshold).replace('.', '_')}"
        path = ROOT / f"{cid.replace(':', '_')}.py"
        if not path.exists():
            path.write_text(_code(kind, fast, slow, threshold))
        c = Candidate(id=cid, source="Invented", name=name,
                      description=("Engine-generated, transparent mechanic variant. "
                                   "It is an unproven hypothesis, not an edge claim."),
                      asset_class="Crypto", popularity=0, has_source=True,
                      mechanics=["volume" if kind == "volume_breakout" else kind,
                                 "breakout" if kind in ("volume_breakout", "structure") else kind])
        # Preserve tag uniqueness and display cleanliness.
        c.mechanics = sorted(set(c.mechanics))
        added += int(st.upsert(c))
    return {"added": added, "total": len(BLUEPRINTS)}


def source(cid: str) -> str | None:
    path = ROOT / f"{cid.replace(':', '_')}.py"
    return path.read_text() if path.exists() else None
