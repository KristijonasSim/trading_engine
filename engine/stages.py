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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import bridge, pysource
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
        try:
            df = bridge.fetch_crypto(sym, "1h")
        except Exception:
            continue
        if df is None or len(df) < 500:
            continue
        part = _split(df, which)
        if len(part) < 200:
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


def evaluate(cid: str, symbols: list[str]) -> dict:
    """Full funnel for one candidate. Returns a stage-by-stage record."""
    code = pysource.source(cid)
    if code is None:
        return {"id": cid, "stage": "no-source", "passed": False}

    # STAGE 1 — runnable and honest. Costs nothing, catches look-ahead.
    df = bridge.fetch_crypto(symbols[0], "1h")
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


def run(symbols: list[str] | None = None, limit: int = 25) -> dict:
    """Push the queue through both stages and write the funnel for the UI."""
    symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
    st = CandidateStore()
    rows = [r for r in st.all()
            if pysource.has_source(r["id"]) and r.get("verdict") in (None, "pending")]

    results, survivors = [], []
    for r in rows[:limit]:
        rec = evaluate(r["id"], symbols)
        rec["name"] = r["name"]
        rec["source"] = r["source"]
        rec["url"] = r.get("url")
        results.append(rec)

        if rec["stage"] == "holdout" and rec["passed"]:
            survivors.append(rec)
            h = rec["holdout"]
            st.update_result(r["id"], status="tested", verdict="promising",
                             pf=h["pf"], trades=h["trades"],
                             win_rate=h["win_rate"], tpd=h["tpd"],
                             tested_from=h["frm"], tested_to=h["to"],
                             note=f"survived screen + holdout — {rec['reason']}")
        else:
            st.update_result(r["id"], status="tested", verdict="rejected",
                             note=rec.get("reason", "did not pass"))
        st.append_audit(r["id"], f"stage:{rec['stage']}", rec.get("reason", ""))

    funnel = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": SCREEN_FRAC,
        "gates": {"screen_pf": SCREEN_PF, "holdout_pf": HOLDOUT_PF,
                  "min_trades": MIN_TRADES},
        "counts": {
            "candidates": len(rows),
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
    return funnel["counts"]


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
