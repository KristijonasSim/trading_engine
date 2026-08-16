"""Tests for the zero-token path: adapters, pysource, and the stage split.

The properties worth pinning here are the ones whose failure is SILENT — an
adapter that misreads a parameter, or a split that leaks the holdout, produces a
number rather than an error, and a number is what gets believed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import pysource
from engine.adapters import adapt, detect, run_freqtrade

FREQ = '''\
from freqtrade.strategy import IStrategy, IntParameter
import pandas as pd


class T(IStrategy):
    timeframe = '1h'
    stoploss = -0.05
    n = IntParameter(5, 50, default=20)

    def populate_indicators(self, dataframe, metadata):
        dataframe['sma'] = dataframe['close'].rolling(self.n.value).mean()
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[dataframe['close'] > dataframe['sma'], 'enter_long'] = 1
        return dataframe
'''


@pytest.fixture
def bars() -> pd.DataFrame:
    n = 3000
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    start = np.arange(n, dtype=np.int64) * 3_600_000 + 1_600_000_000_000
    return pd.DataFrame({
        "start": start, "open": close, "high": close * 1.004,
        "low": close * 0.996, "close": close, "volume": np.full(n, 10.0)})


def test_detect_shapes():
    assert detect(FREQ) == "freqtrade"
    assert detect("def signals(df):\n    return []\n") == "native"
    assert detect("x = 1") == ""


def test_param_default_is_read_from_keyword(bars):
    """IntParameter(5, 50, default=20) must resolve to 20, never to the floor 5.

    Taking the first positional silently substitutes the search-space floor and
    changes what the strategy computes, with no error anywhere.
    """
    out = run_freqtrade(FREQ, bars)
    assert out, "strategy produced no signals at all"
    # With default=20 the first possible entry is bar 19 (rolling window fills
    # at index n-1). A floor of 5 would allow an entry as early as bar 4.
    assert min(i for i, _, _ in out) >= 19


def test_rising_edge_only(bars):
    """A sustained condition is ONE entry, not one per bar.

    Freqtrade marks every bar the condition holds and enters once because it
    tracks position state; `signals()` has none, so without an edge filter a
    40-bar condition becomes 40 entries and the strategy reads as noise.
    """
    out = run_freqtrade(FREQ, bars)
    idx = [i for i, _, _ in out]
    assert len(idx) == len(set(idx))
    # far fewer entries than bars where close > sma
    assert len(idx) < len(bars) * 0.2


def test_stop_is_on_the_right_side_and_bounded(bars):
    for i, direction, stop in run_freqtrade(FREQ, bars):
        px = float(bars["close"].iloc[i])
        assert stop > 0
        if direction == 1:
            assert stop < px
        else:
            assert stop > px
        assert abs(stop - px) / px <= 0.35


def test_unsafe_source_is_rejected_before_execution():
    """Foreign code reaching the filesystem must never be stored or run."""
    bad = FREQ.replace("import pandas as pd", "import os\nimport pandas as pd")
    r = adapt(bad)
    assert not r.ok
    assert "unsafe" in r.reason


def test_unsupported_shape_costs_nothing():
    r = adapt("print('hello')")
    assert not r.ok
    assert "unsupported" in r.reason


def test_adapted_code_passes_verify(bars):
    from engine.translate import verify
    r = adapt(FREQ)
    assert r.ok and r.kind == "freqtrade"
    assert verify(r.code, bars).ok


def test_pysource_roundtrip_and_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(pysource, "PYSRC", tmp_path)
    assert pysource.source("gh:a/b:c.py") is None
    pysource.store("gh:a/b:c.py", "def signals(df):\n    return []\n")
    assert pysource.has_source("gh:a/b:c.py")
    assert "signals" in pysource.source("gh:a/b:c.py")
    # whitespace-insensitive: a reformatted fork is the same idea
    assert (pysource.fingerprint("def f():\n  return 1")
            == pysource.fingerprint("def f():\n    return  1"))


def test_split_is_disjoint_and_ordered():
    """The holdout must never overlap the screen — that is the whole mechanism."""
    from engine.stages import SCREEN_FRAC, _split
    df = pd.DataFrame({"start": np.arange(1000, dtype=np.int64)})
    a, b = _split(df, "screen"), _split(df, "holdout")
    assert len(a) + len(b) == len(df)
    assert set(a["start"]).isdisjoint(set(b["start"]))
    assert a["start"].max() < b["start"].min()
    assert len(a) == int(1000 * SCREEN_FRAC)
