"""The two-stage funnel: screen on old data, confirm on data never seen.

WHY THIS MODULE EXISTS
----------------------
`budget.py` is right that picking the best of N backtests on one window is how
you manufacture a strategy out of noise. But its conclusion — "48 trials, total,
ever" — only binds when every test is scored on the SAME data you then pick from.

Split the history and the arithmetic changes. Measured on this repo's own BTC
bars, 1000 pure-noise strategies screened on 2021-01..2024-06:

    130 of 1000 passed PF >= 1.2          <- fake winners, every one
     11 of 130  passed again on 2024-06..2026-08   <- 92% of the fakes killed

A lucky run does not repeat on data it never touched; an edge does. So the
screen can be wide and cheap, and only what survives it spends real budget.

    STAGE 2  screen   first SCREEN_FRAC of history   wide, thousands allowed
    STAGE 3  holdout  the remainder, never seen      narrow, survivors only

THE ONE RULE
------------
Stage 3 data must never inform stage 2. Not for parameter choice, not for
symbol admission, not for "let me just peek". The moment it does, the holdout
stops being a holdout and this whole module is decoration. That is why the split
is computed here, once, from a fraction — never passed in by a caller who might
tune it after seeing a result.

WHAT SURVIVES IS "PROMISING", NOT "PROVEN"
------------------------------------------
11 of 1000 noise strategies still cleared both stages. Two clean windows is a
filter, not a verdict — forward testing on unseen live bars is still the thing
that settles it, which is what the promotion ladder in `promotion.py` is for.
"""
from __future__ import annotations

import json
import signal
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import activity, bridge, pysource
from .harvest import CandidateStore
from .translate import verify

# The split. A constant, deliberately: see "THE ONE RULE" above.
SCREEN_FRAC = 0.60

# Screen gate. Intentionally loose — the screen's job is to cut the obvious
# junk cheaply, not to decide anything. The holdout is where a claim is made.
SCREEN_PF = 1.10
MIN_TRADES = 30

# Holdout gate, matching trading-bots' PF >= 1.2 backtest gate.
HOLDOUT_PF = 1.20

# Wall-clock budget for ONE candidate, both stages, all symbols. A strategy that
# cannot be evaluated inside this is rejected rather than waited for — see
# _deadline().
EVAL_TIMEOUT_S = 45

DASH = Path(__file__).resolve().parent.parent / "dashboard"
FUNNEL = DASH / "pipeline.json"


@dataclass
class StageResult:
    pf: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    tpd: float = 0.0
    frm: str = ""
    to: str = ""


def _split(df: pd.DataFrame, which: str) -> pd.DataFrame:
    cut = int(len(df) * SCREEN_FRAC)
    part = df.iloc[:cut] if which == "screen" else df.iloc[cut:]
    return part.reset_index(drop=True)


# Bars, loaded once per process. Every candidate reads the same five symbols for
# both stages, so without this a pass re-parses ~53k-bar CSVs 10 times per
# strategy — 400 redundant loads for a 40-strategy pass, which was most of the
# wall time. The split frames are cached too, since they are what gets used.
_BARS: dict[tuple[str, str], pd.DataFrame] = {}


def _bars(sym: str, which: str) -> pd.DataFrame | None:
    key = (sym, which)
    if key not in _BARS:
        try:
            df = bridge.fetch_crypto(sym, "1h")
        except Exception:
            _BARS[key] = None
            return None
        if df is None or len(df) < 500:
            _BARS[key] = None
            return None
        part = _split(df, which)
        _BARS[key] = part if len(part) >= 200 else None
    return _BARS[key]


def _run(code: str, symbols: list[str], which: str,
         fee: float = bridge.TAKER) -> StageResult:
    """Pooled backtest over every symbol, zero admission (HARD RULE 3)."""
    ns: dict = {}
    exec(compile(code, "<strategy>", "exec"), ns)                # noqa: S102
    fn = ns["signals"]

    rs: list[float] = []
    lo = hi = None
    days = 0.0
    for sym in symbols:
        part = _bars(sym, which)
        if part is None:
            continue
        a = pd.to_datetime(part["start"].iloc[0], unit="ms", utc=True)
        b = pd.to_datetime(part["start"].iloc[-1], unit="ms", utc=True)
        lo = a if lo is None else min(lo, a)
        hi = b if hi is None else max(hi, b)
        days = max(days, (b - a).total_seconds() / 86400)
        try:
            trades = bridge.run_backtest(
                part, fn(part.copy()), fee=fee, slippage=bridge.SLIP,
                rr=2.0, max_hold=40, cooldown=1)
        except Exception:
            continue
        rs.extend(t.pnl_r for t in trades)

    if not rs:
        return StageResult(frm=str(lo or ""), to=str(hi or ""))
    s = pd.Series(rs)
    g = s[s > 0].sum()
    l = -s[s < 0].sum()
    return StageResult(
        pf=float(g / l) if l > 0 else 0.0,
        trades=len(s),
        win_rate=float((s > 0).mean()),
        tpd=len(s) / days if days else 0.0,
        frm=str(lo.date()) if lo is not None else "",
        to=str(hi.date()) if hi is not None else "")


@contextmanager
def _deadline(seconds: int):
    """Abort a candidate that will not finish in reasonable time.

    Some harvested strategies compute hundreds of indicators per bar and take
    minutes on a single symbol. In a 24/7 loop one of those is not slow, it is a
    STOP: the queue stalls behind it, the console's per-candidate heartbeat sits
    on the same name, and the engine looks dead while it is in fact busy on one
    hopeless file.

    SIGALRM only fires between bytecode instructions, so a call stuck inside a
    single long C-level numpy op can overrun it. That is acceptable — this is a
    guard against pathological strategies, not a hard real-time bound.
    """
    def _fire(signum, frame):
        raise TimeoutError(f"exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def evaluate(cid: str, symbols: list[str]) -> dict:
    """Full funnel for one candidate. Returns a stage-by-stage record."""
    code = pysource.source(cid)
    if code is None:
        return {"id": cid, "stage": "no-source", "passed": False}
    try:
        with _deadline(EVAL_TIMEOUT_S):
            return _evaluate(cid, code, symbols)
    except TimeoutError as e:
        return {"id": cid, "stage": "timeout", "passed": False,
                "reason": f"too slow to evaluate ({e}) — skipped"}
    except Exception as e:
        return {"id": cid, "stage": "error", "passed": False,
                "reason": f"{type(e).__name__}: {e}"[:160]}


def _evaluate(cid: str, code: str, symbols: list[str]) -> dict:

    # STAGE 1 — runnable and honest. Costs nothing, catches look-ahead.
    df = _bars(symbols[0], "screen")
    if df is None:
        return {"id": cid, "stage": "no-bars", "passed": False,
                "reason": "no bars for the reference symbol"}
    v = verify(code, df)
    if not v.ok:
        return {"id": cid, "stage": "verify", "passed": False,
                "reason": (v.failures[:1] or ["failed verification"])[0]}

    # STAGE 2 — screen.
    s2 = _run(code, symbols, "screen")
    rec = {"id": cid, "screen": asdict(s2)}
    if s2.trades < MIN_TRADES or s2.pf < SCREEN_PF:
        rec.update(stage="screen", passed=False,
                   reason=f"screen PF {s2.pf:.2f} on {s2.trades} trades")
        return rec

    # STAGE 3 — holdout. Data stage 2 never touched.
    s3 = _run(code, symbols, "holdout")
    rec["holdout"] = asdict(s3)
    passed = s3.trades >= MIN_TRADES and s3.pf >= HOLDOUT_PF
    rec.update(stage="holdout", passed=passed,
               reason=f"holdout PF {s3.pf:.2f} on {s3.trades} trades")
    return rec


def _publish(st: CandidateStore, results: list[dict], survivors: list[dict]) -> dict:
    """Write both files the console reads: the funnel and the strategy list."""
    funnel = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": SCREEN_FRAC,
        "gates": {"screen_pf": SCREEN_PF, "holdout_pf": HOLDOUT_PF,
                  "min_trades": MIN_TRADES},
        "counts": {
            "candidates": len(results),
            "evaluated": len(results),
            "failed_verify": sum(1 for r in results if r["stage"] == "verify"),
            "failed_screen": sum(1 for r in results if r["stage"] == "screen"),
            "reached_holdout": sum(1 for r in results if r["stage"] == "holdout"),
            "survivors": len(survivors),
        },
        "results": results,
    }
    DASH.mkdir(parents=True, exist_ok=True)
    tmp = FUNNEL.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(funnel, indent=1, default=str))
    tmp.replace(FUNNEL)
    try:
        from .harvest import to_dashboard
        to_dashboard(st)
    except Exception:
        pass
    return funnel["counts"]


def run(symbols: list[str] | None = None, limit: int = 25) -> dict:
    """Push the queue through both stages and write the funnel for the UI."""
    symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
    st = CandidateStore()
    rows = [r for r in st.all()
            if pysource.has_source(r["id"]) and r.get("verdict") in (None, "pending")]

    results, survivors = [], []
    batch = rows[:limit]
    for i, r in enumerate(batch, 1):
        # Heartbeat PER CANDIDATE, not once per phase. Evaluating 40 strategies
        # can take well over the console's 7-minute staleness window, and a
        # single write at the start of the loop left activity.json claiming
        # "running" while growing stale — which the console correctly, and
        # alarmingly, renders as "worker stopped" mid-pass.
        activity.write(status="running",
                       current={"stage": "screening and holdout",
                                "name": r["name"], "asset_class": "Crypto",
                                "number": i, "total": len(batch)})
        rec = evaluate(r["id"], symbols)
        rec["name"] = r["name"]
        rec["source"] = r["source"]
        rec["url"] = r.get("url")
        results.append(rec)

        # Use the vocabulary the EXISTING console already speaks: pass/fail, with
        # status rejected/tested. Writing 'promising'/'rejected' invented a second
        # vocabulary, so index.html — which counts verdict === "pass" and "fail" —
        # showed 0 rejected and 0 promoted while the store held 44 and 2.
        if rec["stage"] == "holdout" and rec["passed"]:
            survivors.append(rec)
            h = rec["holdout"]
            st.update_result(r["id"], status="tested", verdict="pass",
                             pf=h["pf"], trades=h["trades"],
                             win_rate=h["win_rate"], tpd=h["tpd"],
                             tested_from=h["frm"], tested_to=h["to"],
                             score=8,
                             note=f"survived screen + holdout — {rec['reason']}")
        else:
            s = rec.get("screen") or {}
            st.update_result(r["id"], status="rejected", verdict="fail", score=1,
                             pf=(rec.get("holdout") or s).get("pf"),
                             trades=(rec.get("holdout") or s).get("trades", 0),
                             tested_from=(rec.get("holdout") or s).get("frm"),
                             tested_to=(rec.get("holdout") or s).get("to"),
                             note=rec.get("reason", "did not pass"))
        st.append_audit(r["id"], f"stage:{rec['stage']}", rec.get("reason", ""))

        # Republish the console's data AS WE GO. index.html reads
        # strategies.json, and writing it only at the end of a pass froze every
        # counter for the pass's whole duration — 15+ minutes once the queue
        # filled with heavy strategies. The DB held 64 rejected while the page
        # still showed 44, which is indistinguishable from a stalled engine.
        _publish(st, results, survivors)

    return _publish(st, results, survivors)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
