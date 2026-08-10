"""Tests for ideas 2, 3, 4 — feeds, meta-labeling, regime.

Each module gets one test that could actually fail for a real reason:

  feeds     — does the lag guard stop a bar seeing a score published inside it?
  metalabel — does purged CV refuse to learn from a leaked label?
  regime    — does it recover regimes that genuinely exist, and refuse to
              invent ones that do not?

Run:  python tests/test_modules.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from engine.feeds import Document, LexiconScorer, TextFeed
from engine.metalabel import (MetaLabeler, PurgedSplit, Signal, apply_filter)
from engine.regime import RegimeModel, build_features, leg_fitness

_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"{'  PASS' if cond else '  FAIL'}  {name}"
          f"{('  -- ' + detail) if detail else ''}")


# -------------------------------------------------------------------- feeds
def test_feed_lag_blocks_lookahead():
    """A bar must never carry a score released inside or after it."""
    release = pd.Timestamp("2026-03-18 19:00", tz="UTC")
    s = pd.Series({release: 0.9}, name="fomc")
    bars = pd.date_range("2026-03-17", "2026-03-21", freq="D", tz="UTC")
    f = TextFeed.as_feature(s, bars, ffill_limit=10)

    before = f.loc[bars[bars < release]]
    check("bars before release are NaN", before.isna().all(),
          f"got {before.tolist()}")
    same_day = f.loc[pd.Timestamp('2026-03-18', tz='UTC')]
    check("the bar CONTAINING the release does not see it", pd.isna(same_day),
          f"got {same_day}")
    after = f.loc[pd.Timestamp('2026-03-20', tz='UTC')]
    check("a later bar does see it", after == 0.9, f"got {after}")


def test_scorer_is_deterministic_and_cached():
    docs = [Document(id="d1", released=pd.Timestamp("2026-01-01", tz="UTC"),
                     text="inflation remains elevated and the committee will "
                          "tighten policy further", source="t"),
            Document(id="d2", released=pd.Timestamp("2026-02-01", tz="UTC"),
                     text="growth is slowing and the committee will ease and "
                          "lower rates to support demand", source="t")]
    sc = LexiconScorer()
    check("hawkish text scores above dovish", sc.score(docs[0]) > sc.score(docs[1]),
          f"{sc.score(docs[0]):.2f} vs {sc.score(docs[1]):.2f}")
    check("scorer is deterministic", sc.score(docs[0]) == sc.score(docs[0]))
    with tempfile.TemporaryDirectory() as d:
        feed = TextFeed("t", sc, cache_dir=Path(d))
        a = feed.build(docs)
        b = feed.build(docs)          # second call hits cache
        check("cached rebuild is identical", a.equals(b))


# ---------------------------------------------------------------- metalabel
def _make_signals(n, rng, overlap_bars=10):
    """Overlapping trades — the situation that breaks ordinary k-fold."""
    sigs = []
    for i in range(n):
        entry = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
        sigs.append(Signal(entry=entry,
                           exit=entry + pd.Timedelta(hours=overlap_bars),
                           side=1, pnl_r=float(rng.normal(0, 1))))
    return sigs


def test_purged_split_removes_overlap():
    rng = np.random.default_rng(0)
    sigs = _make_signals(300, rng, overlap_bars=10)
    sp = PurgedSplit(n_splits=5, embargo=0.01)
    folds = list(sp.split(sigs))
    check("purged split yields folds", len(folds) == 5, f"got {len(folds)}")

    leaks = 0
    for tr, te in folds:
        t0 = min(sigs[i].entry for i in te)
        t1 = max(sigs[i].exit for i in te)
        for j in tr:
            if not (sigs[j].exit < t0 or sigs[j].entry > t1):
                leaks += 1
    check("NO training sample overlaps its test fold", leaks == 0,
          f"{leaks} overlapping samples")


def test_metalabel_refuses_to_learn_from_noise():
    """Features unrelated to outcome must give AUC ~0.5, not something flattering."""
    rng = np.random.default_rng(1)
    sigs = _make_signals(400, rng, overlap_bars=8)
    X = pd.DataFrame(rng.normal(size=(400, 6)),
                     columns=[f"f{i}" for i in range(6)])
    ml = MetaLabeler()
    res = ml.cross_validate(X, sigs, n_splits=5)
    check("noise features give AUC near 0.5",
          0.35 < res["mean_auc"] < 0.65, f"AUC {res['mean_auc']:.3f}")


def test_metalabel_finds_a_real_relationship():
    """A feature that genuinely predicts the outcome must be found — otherwise
    the purging is so aggressive the model can never learn anything."""
    rng = np.random.default_rng(2)
    n = 500
    sigs = _make_signals(n, rng, overlap_bars=4)
    signal = rng.normal(size=n)
    sigs = [Signal(s.entry, s.exit, s.side,
                   pnl_r=float(signal[i] * 1.2 + rng.normal(0, 0.6)))
            for i, s in enumerate(sigs)]
    X = pd.DataFrame({"informative": signal,
                      "noise": rng.normal(size=n)})
    res = MetaLabeler().cross_validate(X, sigs, n_splits=5)
    check("a real relationship is detected", res["mean_auc"] > 0.75,
          f"AUC {res['mean_auc']:.3f}")


def test_filter_rejects_when_frequency_collapses():
    """PF improvement bought by declining most trades must be REJECTED."""
    rng = np.random.default_rng(4)
    n = 400
    sigs = _make_signals(n, rng, overlap_bars=2)
    # keep only 3% of signals -> great PF, hopeless frequency
    proba = np.zeros(n)
    proba[:12] = 1.0
    sigs = [Signal(s.entry, s.exit, s.side,
                   pnl_r=(3.0 if i < 12 else float(rng.normal(-0.02, 1))))
            for i, s in enumerate(sigs)]
    out = apply_filter(sigs, proba, threshold=0.5)
    check("frequency collapse is rejected", "REJECT" in out["verdict"],
          out["verdict"][:60])


# ------------------------------------------------------------------- regime
def test_regime_recovers_planted_states():
    """Two synthetic worlds — calm and stressed — must land in different clusters."""
    rng = np.random.default_rng(5)
    n, k = 600, 6
    calm = rng.normal(0, 0.004, (n, k))
    stress = rng.normal(0, 0.02, (n, k)) + rng.normal(0, 0.02, (n, 1))  # correlated
    rets = np.vstack([calm, stress])
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                      index=pd.date_range("2020-01-01", periods=2 * n, freq="D"),
                      columns=[f"a{i}" for i in range(k)])
    feats = build_features(px, window=20)
    model = RegimeModel(n_regimes=2, random_state=0).fit(feats)
    lab = model.label(feats).dropna()

    first = lab.iloc[: len(lab) // 2]
    second = lab.iloc[len(lab) // 2:]
    purity = max((first == first.mode()[0]).mean(),
                 (second == second.mode()[0]).mean())
    check("planted regimes are separated", purity > 0.8,
          f"dominant-label purity {purity:.2f}")
    check("both regimes are used", lab.nunique() == 2, f"{lab.nunique()} labels")


def test_leg_fitness_flags_thin_regimes():
    rng = np.random.default_rng(6)
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    regime = pd.Series(([0] * 200) + ([1] * 100), index=idx, dtype=float)
    times = pd.Series(idx[:210])
    r = pd.Series(rng.normal(0.05, 1, 210))
    fit = leg_fitness(regime, times, r, min_trades=20)
    thin = fit[~fit.reliable]
    check("a thin regime is flagged, not dropped", len(thin) >= 1
          and set(fit.regime) == {0, 1},
          f"regimes {list(fit.regime)}, reliable {list(fit.reliable)}")


if __name__ == "__main__":
    print("feeds (idea 2)")
    test_feed_lag_blocks_lookahead()
    test_scorer_is_deterministic_and_cached()
    print("\nmeta-labeling (idea 3)")
    test_purged_split_removes_overlap()
    test_metalabel_refuses_to_learn_from_noise()
    test_metalabel_finds_a_real_relationship()
    test_filter_rejects_when_frequency_collapses()
    print("\nregime (idea 4)")
    test_regime_recovers_planted_states()
    test_leg_fitness_flags_thin_regimes()

    n, total = sum(_results), len(_results)
    print(f"\n{n}/{total} checks passed")
    sys.exit(0 if n == total else 1)
