"""The referee — what may be tested at all, and what it must claim first.

The rule this enforces: HYPOTHESIS FIRST, BACKTEST LAST. The economic reason is
written down BEFORE the test runs. Finding a pattern and inventing the rationale
afterwards is the failure mode that produced 0 legs from 94,658 backtests.

A 173-day moving average that beats a 170-day one has no mechanism. It is a
fitted coin flip, and no amount of validation rescues it. So this module refuses
to let it into the queue at all -- which costs nothing, versus the trial budget
it would have burned.

WHAT GETS THROUGH
-----------------
Four things must be true, all checkable before any data is touched:

  1. MECHANISM. A named reason the effect exists, from a class with a non-zero
     historical hit rate in this operation: forced flow, structural constraint,
     behavioural bias, or an information-processing advantage. "Momentum works"
     is not a mechanism. "Perp funding settles 8-hourly and forces leveraged
     longs to pay, so crowded longs unwind into the settlement" is.
  2. A FEED WE DO NOT ALREADY EXPLOIT. Recorded in trading-bots/CLAUDE.md as
     the only input with a non-zero hit rate: every real leg came from a new
     data feed, or a proven mechanic pointed at new data. 0-for-351 otherwise.
  3. A PRE-STATED PREDICTION. Direction and rough magnitude, before the run.
     If the result contradicts it, that is a failed test -- not an invitation
     to flip the sign and call it a discovery.
  4. NOT A DUPLICATE. Checked against everything previously registered, so the
     same idea cannot quietly consume budget twice under a new name.

Falsification is the point. A hypothesis that cannot fail has not been stated.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class Mechanism(str, Enum):
    """Classes of cause with a non-zero historical hit rate here.

    Deliberately closed. If an idea does not fit one of these, that is
    information -- it usually means no mechanism was identified, only a pattern.
    """
    FORCED_FLOW = "forced_flow"
    """Someone MUST trade regardless of price: funding settlements, index
    rebalances, option expiry pins, liquidation cascades, margin calls, ETF
    creation/redemption. Strongest class -- the cause is observable and dated."""

    STRUCTURAL = "structural"
    """A market's plumbing creates the effect: futures roll, basis convergence,
    settlement mechanics, exchange fee tiers, session boundaries."""

    BEHAVIOURAL = "behavioural"
    """A persistent human bias: anchoring at round numbers, disposition effect,
    overreaction to headlines, weekend risk aversion."""

    INFORMATION = "information"
    """We can see something sooner or more completely than the marginal
    trader: cross-venue premium, positioning reports, on-chain flows."""


class Status(str, Enum):
    REGISTERED = "registered"
    SCREENED = "screened"          # passed the free feature screen
    TESTED = "tested"              # spent budget on a backtest
    REJECTED = "rejected"
    PROMOTED = "promoted"          # cleared every gate, ready for the book


@dataclass
class Hypothesis:
    """One falsifiable claim. Written before any data is touched."""
    id: str
    claim: str
    mechanism: Mechanism
    feed: str
    universe: str
    prediction: str
    null: str
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    status: Status = Status.REGISTERED
    notes: list[str] = field(default_factory=list)

    # results, filled in as it moves through the stages
    screen_ic: float | None = None
    trials_spent: float = 0.0
    observed_sr: float | None = None
    dsr: float | None = None
    tpd: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mechanism"] = self.mechanism.value
        d["status"] = self.status.value
        return d


class RejectedHypothesis(ValueError):
    """The claim did not meet the bar to be tested at all."""


_WEAK = re.compile(
    r"\b(works|seems|looks like|should work|is good|is profitable|"
    r"tends to|often|usually|might|maybe)\b", re.I)


class Register:
    """The hypothesis queue. Append-only, JSON-backed, survives restarts."""

    def __init__(self, path: str | Path, known_feeds: set[str] | None = None):
        self.path = Path(path)
        self.items: dict[str, Hypothesis] = {}
        # Feeds already exploited by the live book -- a hypothesis resting on
        # one of these is not new information, it is a reshuffle.
        self.known_feeds = known_feeds or {
            "ohlcv", "close", "price", "volume", "binance_klines",
            "lsr", "open_interest", "funding", "dvol",
            "coinbase_premium", "upbit_premium", "cme_basis",
        }
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        for raw in json.loads(self.path.read_text()):
            raw["mechanism"] = Mechanism(raw["mechanism"])
            raw["status"] = Status(raw["status"])
            h = Hypothesis(**raw)
            self.items[h.id] = h

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            [h.to_dict() for h in self.items.values()], indent=2))

    # ---------------------------------------------------------------- gate
    def register(self, *, id: str, claim: str, mechanism: Mechanism, feed: str,
                 universe: str, prediction: str, null: str) -> Hypothesis:
        """Admit a hypothesis, or refuse it with a reason.

        Refusing is free. Testing is not. That asymmetry is the whole design.
        """
        if id in self.items:
            raise RejectedHypothesis(f"duplicate id {id!r}")
        if not isinstance(mechanism, Mechanism):
            raise RejectedHypothesis(
                f"mechanism must be one of {[m.value for m in Mechanism]}")

        if len(claim.split()) < 8:
            raise RejectedHypothesis(
                "claim is too short to be falsifiable -- state what happens, "
                "to what, and when")
        if _WEAK.search(prediction):
            raise RejectedHypothesis(
                f"prediction contains hedging language ({_WEAK.search(prediction).group()!r}). "
                "State a direction and rough magnitude that can come out FALSE.")
        if not null.strip():
            raise RejectedHypothesis("null must be stated before the test")

        feed_key = feed.strip().lower()
        if feed_key in self.known_feeds:
            raise RejectedHypothesis(
                f"feed {feed!r} is already exploited by the live book. A new "
                f"formula on an old feed spends trial budget and adds no "
                f"information -- external idea hunts on existing data are "
                f"0-for-351 here. Bring a feed we do not have.")

        dupe = self._find_duplicate(claim, feed_key)
        if dupe:
            raise RejectedHypothesis(
                f"looks like a duplicate of {dupe.id!r} ({dupe.claim[:60]}...). "
                f"Testing it again does not make the first answer more true.")

        h = Hypothesis(id=id, claim=claim, mechanism=mechanism, feed=feed_key,
                       universe=universe, prediction=prediction, null=null)
        self.items[id] = h
        self.save()
        return h

    def _find_duplicate(self, claim: str, feed: str) -> Hypothesis | None:
        """Crude token-overlap duplicate check on same-feed hypotheses."""
        words = set(re.findall(r"[a-z]{4,}", claim.lower()))
        for h in self.items.values():
            if h.feed != feed:
                continue
            other = set(re.findall(r"[a-z]{4,}", h.claim.lower()))
            if not words or not other:
                continue
            overlap = len(words & other) / len(words | other)
            if overlap > 0.6:
                return h
        return None

    # ------------------------------------------------------------- updates
    def update(self, id: str, **fields) -> Hypothesis:
        h = self.items[id]
        for k, v in fields.items():
            if not hasattr(h, k):
                raise KeyError(f"Hypothesis has no field {k!r}")
            setattr(h, k, v)
        self.save()
        return h

    def pending(self) -> list[Hypothesis]:
        return [h for h in self.items.values()
                if h.status in (Status.REGISTERED, Status.SCREENED)]

    def report(self) -> str:
        if not self.items:
            return "register is empty"
        rows = ["id                    status      mech          feed"]
        for h in self.items.values():
            rows.append(f"{h.id:20s}  {h.status.value:10s}  "
                        f"{h.mechanism.value:12s}  {h.feed}")
        return "\n".join(rows)


def register_feeds_from_live_book() -> set[str]:
    """Feeds the deployed N5 book already trades, which are therefore NOT new.

    Kept as a function rather than a constant so it can be re-derived from the
    bots' LEGS dicts once trading-bots is importable, instead of drifting.
    """
    return {
        "ohlcv", "close", "price", "volume", "binance_klines",
        "lsr", "open_interest", "funding", "dvol",
        "coinbase_premium", "upbit_premium", "cme_basis",
    }
