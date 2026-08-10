"""Idea 3 — meta-labeling. Keep the leg's DIRECTION; learn only take/skip.

Lopez de Prado's construction. The primary model (an existing trading-bots leg)
decides side. A secondary classifier only answers "is this particular signal
going to work", which is a far easier question than predicting direction and
degrades gracefully: at worst it says yes to everything and you are back where
you started.

WHY IT FITS HERE
----------------
trading-bots HARD RULE 9: standalone quality is the wrong target, and improving
a leg's own PF has destroyed its book value three separate times. Meta-labeling
does not touch a leg's entry logic at all — it filters and sizes. So it cannot
break the shape diversity the book depends on.

HARD RULE 8 also applies: ADD, DON'T SUBTRACT. Leg-dropping has been tested
blind four times and the signs were inconsistent. A meta-label is not dropping
a leg — it is declining individual signals on evidence, which is a different
and reversible operation.

THE LEAKAGE PROBLEM, WHICH IS THE WHOLE DIFFICULTY
--------------------------------------------------
Trade labels OVERLAP in time: a trade opened Monday and closed Friday shares
information with one opened Wednesday. Ordinary k-fold puts those in different
folds, the model sees the answer through the overlap, and cross-validated
accuracy comes back beautiful and completely fake.

`PurgedSplit` below removes training samples whose lifetime overlaps the test
fold, then embargoes a further margin afterwards. Using plain `KFold` here is
not a small inaccuracy; it invalidates the entire result.

WHAT IT RETURNS
---------------
A probability per signal. Turn it into take/skip with a threshold, or into a
bet size — but note trading-bots HARD RULE 4: judge any sizing change at
MATCHED DRAWDOWN, never matched risk%, or you are just rewarding trade count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
except ImportError as _e:                                        # pragma: no cover
    raise ImportError("meta-labeling needs scikit-learn") from _e


@dataclass
class Signal:
    """One primary-model signal, with the window it was alive for."""
    entry: pd.Timestamp
    exit: pd.Timestamp
    side: int              # +1 long, -1 short — set by the LEG, never learned
    pnl_r: float           # realised R, the thing we are trying to predict


class PurgedSplit:
    """K-fold that respects overlapping label lifetimes.

    For each test fold: drop every training sample whose [entry, exit] window
    intersects the test window, then embargo an additional fraction of the
    sample immediately after it. Both steps are needed — purging alone still
    leaks through serial correlation just past the boundary.
    """

    def __init__(self, n_splits: int = 5, embargo: float = 0.01):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, signals: list[Signal]):
        n = len(signals)
        if n < self.n_splits * 4:
            raise ValueError(f"need >= {self.n_splits * 4} signals, got {n}")
        order = np.argsort([s.entry for s in signals])
        entries = np.array([signals[i].entry for i in order])
        exits = np.array([signals[i].exit for i in order])
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        emb = int(n * self.embargo)

        for k in range(self.n_splits):
            lo, hi = bounds[k], bounds[k + 1]
            test_idx = order[lo:hi]
            t0, t1 = entries[lo], exits[lo:hi].max()

            train = []
            for j in range(n):
                if lo <= j < hi:
                    continue
                if j < hi + emb and j >= hi:          # embargo after the fold
                    continue
                # purge: drop anything whose lifetime touches the test window
                if not (exits[j] < t0 or entries[j] > t1):
                    continue
                train.append(order[j])
            if train and len(test_idx):
                yield np.array(train), test_idx


class MetaLabeler:
    """Learns P(this signal makes money) from features known AT ENTRY.

    Every feature must be computable at the moment of entry. A feature using
    anything from the trade's own future is look-ahead, and the model will find
    it instantly and report superb accuracy.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 4,
                 min_samples_leaf: int = 20, random_state: int = 0):
        self.kw = dict(n_estimators=n_estimators, max_depth=max_depth,
                       min_samples_leaf=min_samples_leaf,
                       random_state=random_state, n_jobs=-1,
                       class_weight="balanced")
        self.model: RandomForestClassifier | None = None
        self.cv_auc_: list[float] = []
        self.feature_names_: list[str] = []

    @staticmethod
    def _labels(signals: list[Signal]) -> np.ndarray:
        return np.array([1 if s.pnl_r > 0 else 0 for s in signals])

    def cross_validate(self, X: pd.DataFrame, signals: list[Signal],
                       n_splits: int = 5, embargo: float = 0.01) -> dict:
        """Purged-CV AUC. The only number worth believing about this model.

        AUC ~0.5 means the meta-label knows nothing — which is a perfectly
        respectable finding and should be reported as such, not tuned away.
        """
        y = self._labels(signals)
        splitter = PurgedSplit(n_splits, embargo)
        aucs, sizes = [], []
        for tr, te in splitter.split(signals):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = RandomForestClassifier(**self.kw).fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            sizes.append(len(te))
        self.cv_auc_ = aucs
        return {"folds": len(aucs),
                "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
                "min_auc": float(np.min(aucs)) if aucs else float("nan"),
                "aucs": [round(float(a), 4) for a in aucs],
                "test_sizes": sizes,
                "base_rate": float(y.mean())}

    def fit(self, X: pd.DataFrame, signals: list[Signal]) -> "MetaLabeler":
        self.feature_names_ = list(X.columns)
        self.model = RandomForestClassifier(**self.kw).fit(
            X, self._labels(signals))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() first")
        return self.model.predict_proba(X)[:, 1]

    def importances(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("fit() first")
        return pd.Series(self.model.feature_importances_,
                         index=self.feature_names_).sort_values(ascending=False)


def apply_filter(signals: list[Signal], proba: np.ndarray,
                 threshold: float = 0.5) -> dict:
    """What the filter would have done, in book terms.

    Reports the KEPT book against the ORIGINAL, including trades/day, because a
    meta-label that lifts PF by declining most signals can still be useless:
    trading-bots HARD RULE 10 — win rate and frequency pass challenges, not raw
    PF. A filter that halves tpd has to be judged on that, not on PF alone.
    """
    r = np.array([s.pnl_r for s in signals], float)
    keep = proba >= threshold
    days = max((max(s.exit for s in signals)
                - min(s.entry for s in signals)).days, 1)

    def book(mask):
        x = r[mask]
        if not len(x):
            return dict(n=0, pf=float("nan"), win=float("nan"),
                        total_r=0.0, tpd=0.0)
        gains, losses = x[x > 0].sum(), -x[x < 0].sum()
        return dict(n=int(len(x)),
                    pf=float(gains / losses) if losses > 0 else float("inf"),
                    win=float((x > 0).mean() * 100),
                    total_r=float(x.sum()),
                    tpd=len(x) / days)

    before, after = book(np.ones(len(r), bool)), book(keep)
    return {"before": before, "after": after,
            "kept_pct": float(keep.mean() * 100),
            "tpd_change": after["tpd"] - before["tpd"],
            "verdict": _verdict(before, after)}


def _verdict(before: dict, after: dict) -> str:
    if after["n"] == 0:
        return "REJECT — filter declines everything"
    if after["total_r"] <= before["total_r"]:
        return "REJECT — total R did not improve"
    if after["tpd"] < 1.0 <= before["tpd"]:
        return ("REJECT — drops below 1 trade/day; cannot pass a challenge "
                "regardless of PF (HARD RULE 10)")
    return "CANDIDATE — verify at book level, matched drawdown (HARD RULE 5)"
