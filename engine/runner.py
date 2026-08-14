"""The loop. One pass: harvest -> translate -> verify -> backtest -> emit.

Designed to be run repeatedly by a timer, not to run forever itself. A crashed
long-lived process is invisible; a systemd timer that fails is a log line and a
push notification. Each pass is short, idempotent, and safe to interrupt.

WHAT ONE PASS DOES
------------------
0. Harvest from TradingView's public endpoints (`sources/tradingview.py`), a
   rotating slice of the search vocabulary, and fetch Pine for candidates that
   do not have it yet. Free -- collecting an idea is not testing it. This is
   what makes the loop self-feeding rather than a one-shot over a fixed seed.
1. Take the next N untested candidates that HAVE Pine stored, most-popular
   first (a work order, not a ranking -- popularity is attention, not edge).
2. Translate Pine -> Python, cached, so a re-run never pays twice.
3. VERIFY the translation. Failures here are translator bugs, not strategy
   results: they mark the candidate rejected with a reason and cost NO trial
   budget, because nothing about the idea was tested.
4. Backtest the survivors on the engine's own data, through trading-bots'
   backtest.py, at taker fees with STOP_FILL=close.
5. Spend budget, deflate, score, write the dashboard.

WHEN THE BUDGET RUNS OUT IT STOPS AND SAYS SO. Per Kristijonas 2026-08-10, the
engine does not quietly switch universes to keep busy -- an exhausted budget is
information, and the honest response is to report it, not to work around it.
The dashboard shows the exhausted universe so it cannot pass unnoticed.
"""
from __future__ import annotations

import json
import math
import os
import sys
import traceback
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import bridge, report
from .budget import BudgetExhausted, Ledger, deflated_sharpe
from .harvest import CandidateStore, to_dashboard
from .robustness import COSTS, assess, cost_dict, stress_cost
from .translate import best_translator, verify

STATE = Path(__file__).resolve().parent.parent / "state"

# Symbols each asset class is tested on. Crypto has intraday; everything else is
# daily-only until a tick loader exists, and pretending otherwise would produce
# results that cannot be deployed.
#
# `min_tpd` is the frequency gate, and it is PER UNIVERSE for a reason.
#
# trading-bots HARD RULE 10 -- win rate and frequency beat raw PF -- was learned
# on intraday crypto, where a prop challenge has to be cleared in days. On DAILY
# bars 1 trade/day is not a demanding bar, it is an arithmetically impossible
# one: a strategy that entered on literally every daily bar would sit at 1.0 and
# a realistic one lands near 0.02. Applying the crypto gate to a daily universe
# would silently guarantee that no non-crypto strategy is ever promoted, no
# matter how good -- the same shape of bug as a leg that can never fire.
#
# So daily universes are judged on having enough trades to constitute a sample
# (~1/week), and `deployable` records the honest consequence: only Crypto runs
# on a timeframe this repo can actually trade a challenge on. A daily result is
# evidence about a MECHANIC, and it stays labelled as such until somebody
# writes the Dukascopy tick loader.
UNIVERSES = {
    "Crypto":  dict(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"], tf="4h",
                    years=5.0, n_eff=2.06, loader="crypto",
                    min_tpd=0.4, deployable=True),
    "FX":      dict(symbols=["EURUSD", "GBPUSD", "USDJPY"], tf="1d",
                    years=5.0, n_eff=3.77, loader="fx",
                    min_tpd=0.15, deployable=False),
    "Stocks":  dict(symbols=["SP500", "NASDAQ100"], tf="1d",
                    years=5.0, n_eff=6.39, loader="fx",
                    min_tpd=0.15, deployable=False),
    "Futures": dict(symbols=["GOLD", "OIL"], tf="1d",
                    years=5.0, n_eff=6.39, loader="fx",
                    min_tpd=0.15, deployable=False),
}

BARS_PER_YEAR = {"4h": 2190, "1h": 8760, "1d": 252, "30m": 17520}

# Passes a single candidate may fail to produce a testable translation before it
# is parked. See the retry cap in run_pass().
MAX_ATTEMPTS = 3

# Harvest only when fewer than this many workable candidates are queued. See the
# demand-driven harvest in run_pass(). ~15 passes of work at PASS_LIMIT=4, so
# the queue is refilled well before it can run dry.
HARVEST_FLOOR = int(os.environ.get("HARVEST_FLOOR", "60"))


@dataclass
class PassResult:
    considered: int = 0
    translated: int = 0
    blocked: int = 0
    verify_failed: int = 0
    tested: int = 0
    promoted: int = 0
    rejected: int = 0
    harvest: dict = None
    exhausted: list[str] = None
    errors: list[str] = None
    notified: list[str] = None
    backlog: int = 0            # workable candidates queued after this pass

    def __post_init__(self):
        self.exhausted = self.exhausted or []
        self.errors = self.errors or []
        self.harvest = self.harvest or {}
        self.notified = self.notified or []


PINE = STATE / "pine"
COUNTER = STATE / "passes.json"


def _pass_no() -> int:
    """Monotonic pass counter, persisted.

    Used only to rotate the harvest vocabulary, so that pass 40 is not
    re-querying the same six search terms as pass 1. Kept in its own file
    rather than the ledger: the ledger is the audit trail for trials SPENT and
    should not carry bookkeeping that has nothing to do with budget.
    """
    try:
        n = int(json.loads(COUNTER.read_text())["passes"])
    except Exception:                                            # noqa: BLE001
        n = 0
    COUNTER.parent.mkdir(parents=True, exist_ok=True)
    COUNTER.write_text(json.dumps({"passes": n + 1}))
    return n


def _pine_source(cid: str) -> str | None:
    """Pine for a candidate id, if it has been fetched.

    Sources live as files rather than in the DB: they are large, they never
    change once published, and keeping them on disk means a translation can be
    re-derived and diffed without touching the store.
    """
    safe = cid.replace(":", "_").replace(";", "_").replace("/", "_")
    f = PINE / f"{safe}.pine"
    return f.read_text() if f.exists() else None


def _implementation(row: dict) -> str | None:
    """Exact executable source for a row, regardless of where the idea came from."""
    if row.get("source") == "Invented":
        from .invent import source
        return source(row["id"])
    return _pine_source(row["id"])


def _load_bars(sym: str, cfg: dict) -> pd.DataFrame:
    if cfg["loader"] == "crypto":
        df = bridge.fetch_crypto(sym, cfg["tf"], days=int(cfg["years"] * 365.25))
    else:
        if bridge.fetch_fx_bars is None:
            raise RuntimeError("fetch_fx unavailable")
        df = bridge.fetch_fx_bars(sym, cfg["tf"])

    # `start` is epoch MILLISECONDS in both loaders.  Pandas otherwise reads a
    # bare int as nanoseconds, turning 2026 into 1970 and corrupting every
    # annualized statistic.  Keep a rolling, recent-only window by policy: old
    # data must never make an edge look more certain than it is today.
    stamps = pd.to_datetime(df["start"], unit="ms", utc=True, errors="coerce")
    if stamps.isna().any():
        raise RuntimeError(f"{sym}: invalid bar timestamps")
    cutoff = stamps.max() - pd.Timedelta(days=cfg["years"] * 365.25)
    out = df.loc[stamps >= cutoff].copy().reset_index(drop=True)
    if len(out) < 100:
        raise RuntimeError(f"{sym}: only {len(out)} bars inside {cfg['years']}y window")
    return out


def _score(pf, tpd, dsr, dd) -> int:
    """1-10 composite. Explicitly NOT a return forecast.

    Weighted toward the things that decide whether a strategy can pass a
    challenge, per trading-bots HARD RULE 10: frequency and reliability first,
    raw PF second. A 20%-win-rate monster with a huge PF scores poorly here on
    purpose.
    """
    s = 0.0
    if pf:
        s += 3.0 * min(max((pf - 1.0) / 0.6, 0), 1)          # PF 1.0->1.6
    if tpd:
        s += 2.5 * min(tpd / 2.0, 1)                          # up to 2 tpd
    if dsr is not None:
        s += 3.5 * min(max(dsr, 0), 1)                        # deflated confidence
    if dd:
        s += 1.0 * min(max((0.35 - dd) / 0.25, 0), 1)         # smaller DD better
    return int(round(min(max(s, 1), 10)))


def _backtest(code: str, cfg: dict, *, cost=None, holdout: bool = False):
    """Run a translation across a universe. Returns (cells, bar_from, bar_to).

    Pooled across every symbol with ZERO admission -- trading-bots HARD RULE 3.
    A per-symbol PF is not evidence; the pooled number over all symbols is.
    """
    cells: dict[str, pd.Series] = {}
    bar_from = bar_to = None
    fn_ns: dict = {}
    exec(compile(code, "<t>", "exec"), fn_ns)                    # noqa: S102
    fn = fn_ns["signals"]
    for sym in cfg["symbols"]:
        df = _load_bars(sym, cfg)
        if holdout:
            # Final 40% is never used to decide which hypothesis to create.
            # It is a chronological hold-out, not a shuffled train/test split.
            df = df.iloc[int(len(df) * 0.60):].reset_index(drop=True)
        # The DATA window, not the trade span. A strategy that stops trading
        # after year two still occupied the account for all five -- measuring
        # frequency over the trades' own span flatters exactly the sporadic
        # strategies this corpus is full of.
        lo = pd.to_datetime(df["start"].iloc[0], unit="ms", utc=True)
        hi = pd.to_datetime(df["start"].iloc[-1], unit="ms", utc=True)
        bar_from = lo if bar_from is None else min(bar_from, lo)
        bar_to = hi if bar_to is None else max(bar_to, hi)
        execution = cost or COSTS.get(cfg.get("asset", "Crypto"), COSTS["Crypto"])
        trades = bridge.run_backtest(
            df, fn(df.copy()), fee=execution.fee, slippage=execution.slippage,
            rr=2.0, max_hold=40, cooldown=1)
        r = [x.pnl_r for x in trades]
        if r:
            cells[sym] = pd.Series(r)
    return cells, bar_from, bar_to


def _scenario(cells: dict, name: str, *, available: bool = True, detail: str = "") -> dict:
    if not available or not cells:
        return {"name": name, "available": False, "detail": detail or "no trades/data"}
    rs = np.concatenate([c.values for c in cells.values()])
    gains, losses = rs[rs > 0].sum(), -rs[rs < 0].sum()
    return {"name": name, "available": True,
            "pf": round(float(gains / losses) if losses > 0 else float("inf"), 3),
            "max_dd": round(float((np.maximum.accumulate(np.cumsum(rs)) - np.cumsum(rs)).max() /
                                      max(abs(np.cumsum(rs)).max(), 1)), 4),
            "trades": int(len(rs)), "symbols": len(cells), "detail": detail}


def _judge(cells, bar_from, bar_to, cfg, asset, spent, robustness=None) -> dict:
    """Turn pooled trades into the row's stored verdict. ONE definition.

    Both the normal pass and `--recheck` come through here. When the metric
    definitions changed (trades/day moved from the trade span to the data
    window) a second copy would have left half the table on the old ruler and
    half on the new, in the same sort order, with nothing to say which was
    which.
    """
    rs = np.concatenate([c.values for c in cells.values()])
    gains, losses = rs[rs > 0].sum(), -rs[rs < 0].sum()
    pf = float(gains / losses) if losses > 0 else float("inf")
    win = float((rs > 0).mean())
    span_days = max((bar_to - bar_from).days, 1)
    years = span_days / 365.25
    tpd = len(rs) / span_days

    sr_bar = float(rs.mean() / rs.std()) if rs.std() > 0 else 0.0
    sharpe = sr_bar * np.sqrt(BARS_PER_YEAR.get(cfg["tf"], 252))
    dsr = deflated_sharpe(sr_bar, n_trials=max(spent, 1), n_obs=len(rs))

    eq = np.cumsum(rs)
    dd = float((np.maximum.accumulate(eq) - eq).max() / max(abs(eq).max(), 1))
    cagr = float(rs.sum() * 0.01 / years)                        # R at 1% risk

    robust = robustness or {"stable": False, "coverage": 0}
    promote = (dsr >= 0.95 and pf >= 1.2 and tpd >= cfg.get("min_tpd", 1.0)
               and robust["stable"])
    verdict = "pass" if promote else ("hold" if pf >= 1.2 or dsr >= 0.9 else "fail")
    pts = _points(pf, tpd, dsr, dd, win, cagr, len(rs), years, len(cells),
                  spent, asset, cfg)
    if robust["coverage"]:
        pts.append({"ok": bool(robust["stable"]),
                    "text": f"robustness: {robust['passed']}/{robust['coverage']} scenario checks pass"
                            + (" — stable across costs/timeframes" if robust["stable"]
                               else " — not stable enough to promote")})
    return dict(
        status="promoted" if promote else "tested",
        pf=round(pf, 3), tpd=round(tpd, 3), cagr=round(cagr, 4),
        max_dd=round(dd, 4), win_rate=round(win, 4), sharpe=round(sharpe, 3),
        dsr=round(dsr, 4), trades=len(rs), score=_score(pf, tpd, dsr, dd),
        verdict=verdict, note=_headline(pts, promote),
        tested_from=bar_from.date().isoformat(),
        tested_to=bar_to.date().isoformat(),
        years=round(years, 2), test_timeframe=cfg["tf"], points=json.dumps(pts),
        robustness=json.dumps(robust))


def _points(pf, tpd, dsr, dd, win, cagr, n, years, n_syms, spent, asset,
            cfg) -> list[dict]:
    """The result as plain-language findings: what is good, what is bad.

    This replaced a single dense sentence ("tested on 3 symbols, 25 trades, 4h
    bars. DSR 0.22 after 12 trials in Crypto. Gate: PF>=1.2, DSR>=0.95,
    tpd>=1."). Every fact in it was true and none of it could be SKIMMED — you
    had to already know what DSR was and what the gate meant to learn anything.
    A console that has to be read closely does not get read, and the whole point
    of this one is to make a verdict obvious at a glance.

    So each finding states the measurement, the threshold it is judged against,
    and what it means in ordinary words -- with a boolean, so the UI can mark it
    good or bad without re-deriving the gate.
    """
    min_tpd = cfg.get("min_tpd", 1.0)
    d10 = report.days_to_10pct(cagr)
    out = []

    # Frequency first: it is what kills nearly everything in this corpus, and
    # "one trade every 48 days" lands where "0.021 tpd" does not.
    every = f", one every {1 / tpd:,.0f} days" if tpd and tpd > 0 else ""
    # Frequency is a three-band decision, not merely pass/fail.  Kristijonas'
    # research preference: 0.4+/day is usable, 0.2–0.4 deserves a yellow watch
    # label, and below 0.2 is too sparse to take seriously.
    freq_level = "good" if tpd >= min_tpd else ("warn" if tpd >= 0.2 else "bad")
    out.append({"ok": bool(tpd >= min_tpd), "level": freq_level,
                "text": f"trades {tpd:.2f}/day{every} — needs {min_tpd:g}/day "
                        f"to clear a challenge in time"})

    out.append({"ok": bool(pf >= 1.2),
                "text": f"profit factor {pf:.2f} — makes ${pf:.2f} for every "
                        f"$1.00 it loses (gate: 1.20)"})

    if d10 and d10 > 0:
        out.append({"ok": bool(d10 <= 365),
                    "text": f"{d10:,.0f} days to reach +10% at 1% risk per "
                            f"trade" + (" — far too slow to fund an account"
                                        if d10 > 365 else "")})
    else:
        out.append({"ok": False,
                    "text": "loses money over the test — there is no date at "
                            "which it reaches +10%"})

    out.append({"ok": bool(dsr >= 0.95),
                "text": f"{dsr * 100:.0f}% chance the edge is real, after "
                        f"paying for {spent:.0f} trials already spent in "
                        f"{asset} (gate: 95%)"})

    out.append({"ok": bool(n >= 100),
                "text": f"{n} trades over {years:.1f} years on {n_syms} symbols, "
                        f"{cfg['tf']} bars"
                        + ("" if n >= 100 else " — too small a sample to trust")})

    out.append({"ok": bool(win >= 0.5),
                "text": f"wins {win * 100:.0f}% of trades"})

    out.append({"ok": bool(dd <= 0.35),
                "text": f"worst drawdown {dd * 100:.0f}% of accumulated profit"})

    if not cfg.get("deployable", False):
        out.append({"ok": False,
                    "text": "daily bars only — evidence about the MECHANIC, "
                            "not a leg you can trade on a challenge yet"})
    return out


def _headline(points: list[dict], promote: bool) -> str:
    """One line: the verdict and the single reason for it."""
    if promote:
        return "PASSES every gate — verify by hand before deploying."
    bad = [p["text"] for p in points if not p["ok"]]
    if not bad:
        return "Held: clears the gates but not by enough to promote."
    return "Fails on " + bad[0].split(" — ")[0] + f" ({len(bad)} issues)."


def _work_queue(st: CandidateStore, limit: int | None = None, translator=None) -> list[dict]:
    """Untested candidates that can actually be worked on, most-popular first.

    `CandidateStore.queue()` alone is not enough for a loop that runs forever.
    A candidate with no Pine stored is parked back as 'harvested', so it stays
    at the top of a popularity-ordered queue and every subsequent pass picks
    the same unworkable rows again -- the loop spins at 100% blocked while
    thousands of workable candidates sit behind it.

    So the Pine file is a precondition for entering the queue, not a thing
    discovered halfway through the pass.

    The scan is over the WHOLE harvested set, not a slice of it. Popularity and
    workability are independent, so any LIMIT applied before the Pine filter
    hides ready rows that happen to sort below it — and the store grows by ~40
    rows a pass, so that blind spot only widens. A few thousand ids and a
    directory listing is nothing next to one backtest.
    """
    have_pine = {p.stem for p in PINE.glob("*.pine")}
    ready = [r for r in st.queue(limit=None)
             if (_safe(r["id"]) in have_pine or
                 (r.get("source") == "Invented" and _implementation(r)))]

    # The old queue was pure popularity.  That is useful evidence that people
    # looked at an idea, but it is not evidence of edge, and it starves every
    # less-famous mechanic behind another EMA/RSI clone.  Priority is therefore
    # deterministic and deliberately conservative:
    #   1. only stored Pine is eligible (a real test is possible now),
    #   2. deployable intraday crypto is preferred over daily-only evidence,
    #   3. interpretable mechanics beat grid/martingale constructions,
    #   4. families already measured are down-weighted for diversity,
    #   5. popularity is only a final, weak signal.
    # It is a work order, NEVER a predicted-return score.
    tested_by_tag: dict[str, int] = {}
    for row in st.all():
        if row["status"] not in ("tested", "promoted"):
            continue
        for tag in row.get("mechanics") or []:
            tested_by_tag[tag] = tested_by_tag.get(tag, 0) + 1

    def tags(row: dict) -> list[str]:
        raw = row.get("mechanics") or []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = []
        return raw

    def priority(row: dict) -> float:
        t = tags(row)
        asset = row.get("asset_class")
        score = {"Crypto": 100.0, "FX": 45.0, "Stocks": 40.0,
                 "Futures": 40.0}.get(asset, -100.0)
        # A cached or engine-native implementation can be verified NOW. Put
        # those ahead of rows that first need a fresh Pine translation, so the
        # useful test queue is never held hostage by translator failures.
        source = _implementation(row)
        cached = getattr(translator, "cached", lambda **_: False)
        if row.get("source") == "Invented" or (translator and source and
                                                 cached(name=row["name"], source=source)):
            score += 300.0
        # Known strategy shapes. These are triage weights, not a backtest gate.
        score += sum({"structure": 28.0, "volume": 24.0, "breakout": 20.0,
                      "mean_reversion": 16.0, "momentum": 14.0,
                      "trend": 10.0, "volatility": 8.0}.get(x, 0.0) for x in t)
        score += 4.0 * (row.get("source_quality") or 1)
        if "grid_martingale" in t:
            score -= 200.0
        score -= 12.0 * sum(tested_by_tag.get(x, 0) for x in t)
        score += min(math.log1p(max(row.get("popularity") or 0, 0)) * 5.0, 55.0)
        return score

    ready.sort(key=lambda r: (-priority(r), -(r.get("popularity") or 0), r["id"]))
    return ready if limit is None else ready[:limit]


def _safe(cid: str) -> str:
    return cid.replace(":", "_").replace(";", "_").replace("/", "_")


def run_pass(limit: int = 5, use_llm: bool | None = None,
             store: CandidateStore | None = None,
             ledger: Ledger | None = None,
             harvest: bool = True, progress=None) -> PassResult:
    supplied_store = store is not None
    st = store or CandidateStore()
    led = ledger or Ledger(STATE / "ledger.json")
    res = PassResult()

    # Seed only a handful of transparent hypotheses.  This happens before the
    # queue is ranked, so invented work competes under the exact same gates as
    # TradingView work; it never receives a free pass or a synthetic score.
    if not supplied_store:
        from .invent import seed as seed_invented
        seed_invented(st)

    # API key if set, else headless Claude Code on the subscription, else None.
    # None means "park it", never "test it with placeholder logic".
    translator = best_translator() if use_llm is not False else None

    for cfg_name, cfg in UNIVERSES.items():
        led.universe(cfg_name, years=cfg["years"], n_eff=cfg["n_eff"])

    # ---- harvest, but only when the engine is actually short of work.
    #
    # Harvesting costs no trial budget and no tokens -- it is plain HTTP -- so
    # the first version ran it every pass. That was still wrong: it collected
    # ~25 new Pine sources per pass while testing 3, a 6:1 ratio, so the backlog
    # grew without bound and every pass paid ~90 seconds for candidates nobody
    # would reach for days. Collecting is not progress; measuring is.
    #
    # So harvest is now DEMAND-DRIVEN: it runs only when the workable backlog
    # (untested AND Pine stored) has fallen below a floor. With a full queue the
    # pass skips straight to testing and finishes in a third of the time.
    n_pass = _pass_no()
    ready = _work_queue(st, translator=translator)
    if harvest and len(ready) < HARVEST_FLOOR:
        # Announce the harvest BEFORE the network call, not after: it is the
        # slowest thing a pass does, and without this the console showed a
        # frozen "testing" stage for minutes with nothing apparently running.
        if progress:
            progress({"stage": "harvesting", "name": "TradingView search"})
        try:
            from .sources.tradingview import TERMS, harvest as tv_harvest
            res.harvest = tv_harvest(offset=(n_pass * 6) % len(TERMS),
                                     store=st, max_terms=6, max_fetch=25)
            ready = _work_queue(st, translator=translator)
        except Exception as e:                                   # noqa: BLE001
            res.errors.append(f"harvest: {type(e).__name__}: {e}")
    res.backlog = len(ready)

    for row in ready[:limit]:
        res.considered += 1
        cid, asset = row["id"], row["asset_class"]
        if progress:
            progress({"id": cid, "name": row["name"], "asset_class": asset,
                      "stage": "translating and backtesting", "number": res.considered,
                      "total": min(limit, len(ready))})
        cfg_base = UNIVERSES.get(asset)
        cfg = {**cfg_base, "asset": asset} if cfg_base else None
        if cfg is None:
            st.update_result(cid, status="blocked", verdict="blocked", score=None,
                             note=f"asset class {asset!r} has no data universe")
            res.blocked += 1
            continue

        if led.budgets[asset].exhausted:
            if asset not in res.exhausted:
                res.exhausted.append(asset)
            continue


        # ---- translate
        #
        # NO TRANSLATOR MEANS NO TEST. Measured 2026-08-10: running the shared
        # stub over the queue gave "SuperTrend STRATEGY  PF 1.069" and
        # "MACD + SMA 200  PF 1.069" — the same number, because both were an
        # SMA cross wearing the candidate's name. Publishing that attributes one
        # strategy's performance to another, which is fabrication regardless of
        # intent. The candidate is parked instead, no budget is spent, and the
        # dashboard says plainly why.
        invented = row.get("source") == "Invented"
        if translator is None and not invented:
            st.update_result(
                cid, status="harvested", verdict="pending", score=None,
                note="awaiting translation — ANTHROPIC_API_KEY not set, so the "
                     "Pine logic has not been converted and NOTHING about this "
                     "strategy has been measured")
            res.blocked += 1
            continue

        source = _implementation(row)
        if not source:
            st.update_result(
                cid, status="harvested", verdict="pending", score=None,
                note="awaiting Pine source — harvested metadata only, "
                     "nothing measured")
            res.blocked += 1
            continue

        # Exact-source dedupe: two TradingView listings may have different
        # titles/likes but identical logic.  Do not pay the trial budget twice.
        # This intentionally does NOT call all EMA strategies duplicates: same
        # family is useful for diversity control, identical implementation is
        # the only automatic duplicate claim we can defend.
        fingerprint = hashlib.sha256("".join(source.split()).encode()).hexdigest()
        duplicate = next((other for other in st.all() if other["id"] != cid and
                          _implementation(other) and
                          hashlib.sha256("".join(_implementation(other).split()).encode()).hexdigest() == fingerprint), None)
        if duplicate:
            st.update_result(cid, status="duplicate", verdict="duplicate", score=None,
                             duplicate_of=duplicate["id"],
                             note=f"exact duplicate of {duplicate['id']}; not retested and no trial budget spent")
            res.blocked += 1
            continue

        # ---- retry cap. Counted ONLY once the row is genuinely workable —
        # a translator that is absent, or Pine that has not been fetched yet,
        # are conditions of the ENGINE and must never convict a candidate.
        #
        # Past this line every remaining failure path raises and leaves the
        # status at 'harvested', which is right for a transient fault and wrong
        # for a permanent one: the row sits at the head of a popularity-ordered
        # queue and is retried every pass forever, so a handful of broken
        # candidates consume the whole loop and the other 300 are never reached.
        # The engine looks busy and measures nothing.
        #
        # Three attempts, then park it. Costs NO trial budget, and the note says
        # how to un-park it — the usual cause is a translator bug that a later
        # fix will clear.
        attempts = (row.get("attempts") or 0) + 1
        st.update_result(cid, attempts=attempts)
        if attempts > MAX_ATTEMPTS:
            st.update_result(
                cid, status="blocked", verdict="blocked", score=None,
                note=f"gave up after {MAX_ATTEMPTS} attempts to produce a "
                     f"testable translation. NO trial budget was spent and the "
                     f"idea was never refuted — this is a TOOLING failure. "
                     f"Un-park with: UPDATE candidates SET status='harvested', "
                     f"attempts=0 WHERE id='{cid}';")
            res.blocked += 1
            continue

        try:
            if invented:
                code = source
            else:
                code = translator.translate(
                    name=row["name"], source=source,
                    author=row.get("author") or "",
                    description=row.get("description") or "")
                res.translated += 1
        except Exception as e:                                   # noqa: BLE001
            res.errors.append(f"{cid}: translate: {e}")
            continue

        # ---- verify on the first symbol, then backtest across the universe
        try:
            df0 = _load_bars(cfg["symbols"][0], cfg)
        except Exception as e:                                   # noqa: BLE001
            res.errors.append(f"{cid}: data: {e}")
            continue

        t = verify(code, df0)
        if not t.ok:
            # A failed translation is NOT a failed strategy. No budget is spent,
            # and the note says so, so the row can be retried after a translator
            # fix without anyone thinking the idea was refuted.
            st.update_result(cid, status="blocked", verdict="blocked", score=None,
                             note="translation rejected: " + "; ".join(t.failures[:2])
                                  + " (no trial budget spent — this is a "
                                    "translator failure, not a result)")
            res.verify_failed += 1
            res.blocked += 1
            continue

        # ---- backtest, pooled across the universe with zero admission
        try:
            cells, bar_from, bar_to = _backtest(code, cfg, cost=COSTS[asset])
        except Exception as e:                                   # noqa: BLE001
            res.errors.append(f"{cid}: backtest: {type(e).__name__}: {e}")
            # The message goes in the STORE too, not just this pass's error
            # list: res.errors dies with the process, the note is what a human
            # reads days later off the dashboard.
            detail = f"{type(e).__name__}: {e}".strip().replace("\n", " ")[:300]
            st.update_result(cid, status="blocked", verdict="blocked", score=None,
                             note=f"backtest raised: {detail}")
            res.blocked += 1
            continue

        if not cells:
            st.update_result(cid, status="rejected", verdict="fail", score=1,
                             note="no trades on any symbol in the universe")
            res.rejected += 1
            continue

        # ---- spend budget, deflate, judge
        try:
            spend = led.spend(asset, cid, cells=len(cells), independence=1.0,
                              note=row["name"][:70])
        except BudgetExhausted:
            if asset not in res.exhausted:
                res.exhausted.append(asset)
            continue

        # Robustness matrix: base conditions, deliberately worse fills, a
        # chronological hold-out, plus another Crypto timeframe when the data
        # loader supports it.  A missing scenario is visible, never a pass.
        scenarios = [_scenario(cells, f"{asset} {cfg['tf']} · realistic costs",
                               detail=f"fee {COSTS[asset].fee:.04%}/side, slip {COSTS[asset].slippage:.04%}")]
        checks = [
            ("cost stress", cfg, {"cost": stress_cost(asset)}),
            ("chronological hold-out", cfg, {"cost": COSTS[asset], "holdout": True}),
        ]
        if asset == "Crypto":
            checks.append(("Crypto 1h · realistic costs", {**cfg, "tf": "1h"},
                           {"cost": COSTS[asset]}))
        for label, scenario_cfg, kwargs in checks:
            try:
                extra, _, _ = _backtest(code, scenario_cfg, **kwargs)
                scenarios.append(_scenario(extra, label))
            except Exception as e:  # unavailable data is reported, not hidden
                scenarios.append(_scenario({}, label, available=False,
                                           detail=f"unavailable: {type(e).__name__}"))
        robust = assess(scenarios)
        fields = _judge(cells, bar_from, bar_to, cfg, asset,
                        spent=led.budgets[asset].spent, robustness=robust)
        fields["trials"] = int(spend.charged)
        st.update_result(cid, **fields)
        st.append_audit(cid, "backtest completed", f"{fields['verdict']} · PF {fields['pf']}")
        res.tested += 1
        res.promoted += int(fields["status"] == "promoted")
        res.rejected += int(fields["verdict"] == "fail")

    for name, b in led.budgets.items():
        if b.exhausted and name not in res.exhausted:
            res.exhausted.append(name)

    to_dashboard(st)

    # ---- deliver. Wrapped, because the measurement is the valuable part and it
    # is already committed to the store by this point: a broken phone, a DNS
    # failure or a full disk must not turn a good pass into a failed unit.
    try:
        report.write_results(st, led, res)
        res.notified = report.notify(st, led, res)
    except Exception as e:                                       # noqa: BLE001
        res.errors.append(f"report: {type(e).__name__}: {e}")
    return res


def recheck(store: CandidateStore | None = None,
            ledger: Ledger | None = None) -> PassResult:
    """Re-measure every already-tested row on the CURRENT metric definitions.

    Spends NO trial budget, and that is not a loophole. The budget exists to
    charge for SEARCH -- for the selection bias in picking the best of N
    configurations. Re-running one already-chosen configuration on the same data
    searches nothing and cannot flatter anything; the row keeps the `trials` it
    was charged when it was chosen.

    This exists because a metric definition changed once (trades/day moved from
    the trade span to the data window, which is stricter) and half the table
    would otherwise have been on the old ruler, in the same sort order, with
    nothing marking which rows were which. Translations are cached, so this
    costs no tokens either.
    """
    st = store or CandidateStore()
    led = ledger or Ledger(STATE / "ledger.json")
    res = PassResult()
    translator = best_translator()
    if translator is None:
        res.errors.append("recheck needs a translator to rebuild signal code")
        return res

    for row in st.all():
        if row["status"] not in ("tested", "promoted"):
            continue
        cid, asset = row["id"], row["asset_class"]
        cfg_base = UNIVERSES.get(asset)
        cfg = {**cfg_base, "asset": asset} if cfg_base else None
        source = _implementation(row)
        if cfg is None or not source:
            continue
        res.considered += 1
        try:
            code = (source if row.get("source") == "Invented" else translator.translate(
                name=row["name"], source=source, author=row.get("author") or "",
                description=row.get("description") or ""))
            cells, lo, hi = _backtest(code, cfg, cost=COSTS[asset])
            if not cells:
                continue
            # `spent` is the budget ALREADY charged in this universe, exactly as
            # the original verdict saw it -- not a fresh debit.
            # A recheck does not invent a new robustness result.  Preserve the
            # existing matrix, which was charged when this hypothesis ran.
            st.update_result(cid, **_judge(cells, lo, hi, cfg, asset,
                                           spent=led.budgets[asset].spent,
                                           robustness=row.get("robustness")))
            res.tested += 1
        except Exception as e:                                   # noqa: BLE001
            res.errors.append(f"{cid}: recheck: {type(e).__name__}: {e}")

    to_dashboard(st)
    report.write_results(st, led, res)
    return res


def refresh_findings(store: CandidateStore | None = None,
                     ledger: Ledger | None = None) -> int:
    """Refresh wording/colour bands without re-running or re-charging a test."""
    st = store or CandidateStore()
    led = ledger or Ledger(STATE / "ledger.json")
    changed = 0
    for row in st.all():
        if row["status"] not in ("tested", "promoted") or row["pf"] is None:
            continue
        cfg_base = UNIVERSES.get(row["asset_class"])
        if not cfg_base:
            continue
        cfg = {**cfg_base, "asset": row["asset_class"]}
        pts = _points(row["pf"], row["tpd"], row["dsr"], row["max_dd"],
                      row["win_rate"], row["cagr"], row["trades"], row["years"],
                      len(cfg["symbols"]), led.budgets[row["asset_class"]].spent,
                      row["asset_class"], cfg)
        robust = row.get("robustness")
        if robust and robust.get("coverage"):
            pts.append({"ok": bool(robust["stable"]),
                        "text": f"robustness: {robust['passed']}/{robust['coverage']} scenario checks pass"
                                + (" — stable across costs/timeframes" if robust["stable"]
                                   else " — not stable enough to promote")})
        st.update_result(row["id"], points=json.dumps(pts),
                         note=_headline(pts, row["status"] == "promoted"))
        changed += 1
    to_dashboard(st)
    return changed


def main() -> int:
    if "--recheck" in sys.argv:
        r = recheck()
        print(f"  rechecked {r.tested} of {r.considered} tested rows "
              f"on current metric definitions (no budget spent)")
        for e in r.errors[:5]:
            print(f"  error: {e}")
        return 0
    limit = int(os.environ.get("PASS_LIMIT", "5"))
    harvest = os.environ.get("HARVEST", "1") != "0"
    from . import activity
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{started}] pass start")
    activity.write(status="running", started=started)
    try:
        r = run_pass(limit=limit, harvest=harvest,
                     progress=lambda current: activity.write(
                         status="running", started=started, current=current))
    except Exception as e:                                       # noqa: BLE001
        activity.write(status="error", started=started, error=f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    activity.write(status="idle", started=started,
                   summary={"considered": r.considered, "translated": r.translated,
                            "tested": r.tested, "promoted": r.promoted,
                            "rejected": r.rejected, "blocked": r.blocked,
                            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    if r.harvest:
        h = r.harvest
        print(f"  harvest: +{h.get('added', 0)} new of {h.get('seen', 0)} seen | "
              f"store {h.get('total', 0)} | pine {h.get('pine_total', 0)} "
              f"(+{h.get('pine_inline', 0) + h.get('pine_fetched', 0)}) | "
              f"terms {', '.join(h.get('terms', [])[:3])}…")
    print(f"  considered {r.considered} | translated {r.translated} | "
          f"verify-failed {r.verify_failed} | tested {r.tested} | "
          f"promoted {r.promoted} | rejected {r.rejected}")
    if r.blocked:
        print(f"  BLOCKED {r.blocked}: no translator, or no Pine source stored "
              f"for the candidate. Nothing was measured and no budget spent.")
    if r.exhausted:
        print(f"  BUDGET EXHAUSTED: {', '.join(r.exhausted)} — no further "
              f"verdicts are valid there without more data or a new feed")
    print(f"  results: {report.RESULTS}"
          + (f" | pushed: {', '.join(r.notified)}" if r.notified else ""))
    for e in r.errors[:5]:
        print(f"  error: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
