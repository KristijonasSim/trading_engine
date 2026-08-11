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
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import bridge, report
from .budget import BudgetExhausted, Ledger, deflated_sharpe
from .harvest import CandidateStore, to_dashboard
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
                    years=5.1, n_eff=2.06, loader="crypto",
                    min_tpd=1.0, deployable=True),
    "FX":      dict(symbols=["EURUSD", "GBPUSD", "USDJPY"], tf="1d",
                    years=20.0, n_eff=3.77, loader="fx",
                    min_tpd=0.15, deployable=False),
    "Stocks":  dict(symbols=["SP500", "NASDAQ100"], tf="1d",
                    years=25.0, n_eff=6.39, loader="fx",
                    min_tpd=0.15, deployable=False),
    "Futures": dict(symbols=["GOLD", "OIL"], tf="1d",
                    years=20.0, n_eff=6.39, loader="fx",
                    min_tpd=0.15, deployable=False),
}

BARS_PER_YEAR = {"4h": 2190, "1h": 8760, "1d": 252, "30m": 17520}

# Passes a single candidate may fail to produce a testable translation before it
# is parked. See the retry cap in run_pass().
MAX_ATTEMPTS = 3


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


def _load_bars(sym: str, cfg: dict) -> pd.DataFrame:
    if cfg["loader"] == "crypto":
        return bridge.fetch_crypto(sym, cfg["tf"], days=int(cfg["years"] * 365))
    if bridge.fetch_fx_bars is None:
        raise RuntimeError("fetch_fx unavailable")
    return bridge.fetch_fx_bars(sym, cfg["tf"])


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


def _work_queue(st: CandidateStore, limit: int) -> list[dict]:
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
    ready = [r for r in st.queue(limit=None) if _safe(r["id"]) in have_pine]
    return ready[:limit]


def _safe(cid: str) -> str:
    return cid.replace(":", "_").replace(";", "_").replace("/", "_")


def run_pass(limit: int = 5, use_llm: bool | None = None,
             store: CandidateStore | None = None,
             ledger: Ledger | None = None,
             harvest: bool = True) -> PassResult:
    st = store or CandidateStore()
    led = ledger or Ledger(STATE / "ledger.json")
    res = PassResult()

    # API key if set, else headless Claude Code on the subscription, else None.
    # None means "park it", never "test it with placeholder logic".
    translator = best_translator() if use_llm is not False else None

    for cfg_name, cfg in UNIVERSES.items():
        led.universe(cfg_name, years=cfg["years"], n_eff=cfg["n_eff"])

    # ---- harvest (free, no budget). Failures here are never fatal: the queue
    # is normally already full, and a TradingView outage must not stop testing.
    n_pass = _pass_no()
    if harvest:
        try:
            from .sources.tradingview import TERMS, harvest as tv_harvest
            res.harvest = tv_harvest(offset=(n_pass * 6) % len(TERMS),
                                     store=st, max_terms=6, max_fetch=25)
        except Exception as e:                                   # noqa: BLE001
            res.errors.append(f"harvest: {type(e).__name__}: {e}")

    for row in _work_queue(st, limit):
        res.considered += 1
        cid, asset = row["id"], row["asset_class"]
        cfg = UNIVERSES.get(asset)
        if cfg is None:
            st.update_result(cid, status="rejected", verdict="fail", score=1,
                             note=f"asset class {asset!r} has no data universe")
            res.rejected += 1
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
        if translator is None:
            st.update_result(
                cid, status="harvested", verdict="pending", score=None,
                note="awaiting translation — ANTHROPIC_API_KEY not set, so the "
                     "Pine logic has not been converted and NOTHING about this "
                     "strategy has been measured")
            res.blocked += 1
            continue

        pine = _pine_source(cid)
        if not pine:
            st.update_result(
                cid, status="harvested", verdict="pending", score=None,
                note="awaiting Pine source — harvested metadata only, "
                     "nothing measured")
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
                cid, status="rejected", verdict="fail", score=1,
                note=f"gave up after {MAX_ATTEMPTS} attempts to produce a "
                     f"testable translation. NO trial budget was spent and the "
                     f"idea was never refuted — this is a TOOLING failure. "
                     f"Un-park with: UPDATE candidates SET status='harvested', "
                     f"attempts=0 WHERE id='{cid}';")
            res.rejected += 1
            continue

        try:
            code = translator.translate(
                name=row["name"], source=pine,
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
            st.update_result(cid, status="rejected", verdict="fail", score=1,
                             note="translation rejected: " + "; ".join(t.failures[:2])
                                  + " (no trial budget spent — this is a "
                                    "translator failure, not a result)")
            res.verify_failed += 1
            res.rejected += 1
            continue

        # ---- backtest, pooled across the universe with zero admission
        cells, pooled = {}, []
        try:
            fn_ns: dict = {}
            exec(compile(code, "<t>", "exec"), fn_ns)             # noqa: S102
            fn = fn_ns["signals"]
            for sym in cfg["symbols"]:
                df = _load_bars(sym, cfg)
                intents = fn(df.copy())
                trades = bridge.run_backtest(
                    df, intents, fee=bridge.TAKER, slippage=bridge.SLIP,
                    rr=2.0, max_hold=40, cooldown=1)
                r = [x.pnl_r for x in trades]
                if r:
                    cells[sym] = pd.Series(r)
                    pooled += [(x.entry_time, x.exit_time, x.pnl_r) for x in trades]
        except Exception as e:                                   # noqa: BLE001
            res.errors.append(f"{cid}: backtest: {type(e).__name__}: {e}")
            st.update_result(cid, status="rejected", verdict="fail", score=1,
                             note=f"backtest raised: {type(e).__name__}")
            res.rejected += 1
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

        rs = np.concatenate([c.values for c in cells.values()])
        gains, losses = rs[rs > 0].sum(), -rs[rs < 0].sum()
        pf = float(gains / losses) if losses > 0 else float("inf")
        win = float((rs > 0).mean())
        span_days = max((pd.Timestamp(pooled[-1][1]) - pd.Timestamp(pooled[0][0])).days, 1)
        tpd = len(rs) / span_days

        sr_bar = float(rs.mean() / rs.std()) if rs.std() > 0 else 0.0
        bpy = BARS_PER_YEAR.get(cfg["tf"], 252)
        sharpe = sr_bar * np.sqrt(bpy)
        dsr = deflated_sharpe(sr_bar, n_trials=max(led.budgets[asset].spent, 1),
                              n_obs=len(rs))

        eq = np.cumsum(rs)
        dd = float((np.maximum.accumulate(eq) - eq).max() / max(abs(eq).max(), 1))
        cagr = float(rs.sum() * 0.01 / (span_days / 365.25))      # R at 1% risk

        min_tpd = cfg.get("min_tpd", 1.0)
        score = _score(pf, tpd, dsr, dd)
        promote = dsr >= 0.95 and pf >= 1.2 and tpd >= min_tpd
        verdict = "pass" if promote else ("hold" if pf >= 1.2 or dsr >= 0.9 else "fail")

        # Say WHY, in the row itself. A verdict without the gate that produced
        # it is the thing that gets misread three weeks later.
        why = (f"tested on {len(cells)} symbols, {len(rs)} trades, {cfg['tf']} bars. "
               f"DSR {dsr:.2f} after {led.budgets[asset].spent:.0f} trials in {asset}. "
               f"Gate: PF>=1.2, DSR>=0.95, tpd>={min_tpd:g}.")
        if not cfg.get("deployable", False):
            why += (" DAILY universe — research evidence about the mechanic, "
                    "NOT a deployable challenge leg (no intraday history here).")
        st.update_result(
            cid, status="promoted" if promote else "tested",
            pf=round(pf, 3), tpd=round(tpd, 3), cagr=round(cagr, 4),
            max_dd=round(dd, 4), win_rate=round(win, 4), sharpe=round(sharpe, 3),
            dsr=round(dsr, 4), trades=len(rs), trials=int(spend.charged),
            score=score, verdict=verdict, note=why)
        res.tested += 1
        res.promoted += int(promote)
        res.rejected += int(verdict == "fail")

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


def main() -> int:
    limit = int(os.environ.get("PASS_LIMIT", "5"))
    harvest = os.environ.get("HARVEST", "1") != "0"
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] pass start")
    try:
        r = run_pass(limit=limit, harvest=harvest)
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        return 1
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
