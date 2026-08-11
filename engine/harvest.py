"""Collect strategy ideas from external sources into a local, queryable store.

Harvesting is FREE. It costs no trial budget, because collecting an idea is not
testing it — there is no configuration search and nothing is being max-picked.
So this can run continuously and at volume; the funnel narrows later.

WHAT A HARVESTED ROW IS AND IS NOT
----------------------------------
A `Candidate` is a CLAIM SOMEONE ELSE MADE. It has a name, a source, an author,
a like count and possibly Pine source. It has NO performance numbers, and the
store will not invent any. Every metric stays None until this engine measures it
on this engine's data.

That distinction is the entire point. TradingView publishes strategies with
screenshots showing 90% win rates; those numbers are curve-fitted to one symbol
on one window and are worth exactly nothing. Copying them into the store would
make the dashboard a liar. `to_dashboard()` emits null, and the UI renders "—".

WHY LIKES ARE STORED BUT NOT TRUSTED
------------------------------------
Popularity is a prior on *attention*, not on edge — and if anything a negative
one, since a widely-copied edge is a crowded one. It is kept only to order the
work queue, never to score a strategy.

SOURCES
-------
`tradingview` — via the MCP corpus, which already holds 790 scripts / 566
sources locally. Ingested through `ingest_records()` so this module never
depends on the MCP being reachable at runtime; a server with no Claude Code can
still read the store.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent / "state" / "candidates.db"

# Exchange/symbol prefixes -> the asset class we will actually research it on.
_ASSET_PATTERNS = [
    (r"BTC|ETH|USDT|USD[TC]|BINANCE|BYBIT|COINBASE|BITSTAMP|BITTREX|KRAKEN", "Crypto"),
    (r"^FX:|^OANDA:|^FX_IDC:|EURUSD|GBPUSD|USDJPY|USDTRY|EURJPY|AUDUSD", "FX"),
    (r"^CME_MINI:|^CBOT:|^NYMEX:|^COMEX:|ES1!|NQ1!|CL1!|GC1!", "Futures"),
    (r"^NASDAQ:|^NYSE:|^AMEX:|^SPCFD:|^DJCFD:|SPX|SPY|DJI|NIFTY|^NSE:", "Stocks"),
]


def classify_asset(symbol: str | None) -> str:
    if not symbol:
        return "Unknown"
    s = symbol.upper()
    for pat, label in _ASSET_PATTERNS:
        if re.search(pat, s):
            return label
    return "Unknown"


# Mechanic tags, matched against name + description + Pine source. Used to spot
# families and near-duplicates, NOT to judge quality.
_MECHANIC_TAGS = {
    "trend": r"supertrend|hull|ema cross|sma ?200|golden cross|moving average|psar|ichimoku",
    "mean_reversion": r"mean revers|bollinger|rsi|oversold|overbought|z-?score|vwap revers",
    "breakout": r"breakout|donchian|opening range|orb|fractal break|channel break",
    "momentum": r"macd|momentum|awesome oscillator|roc\b|stoch",
    "volatility": r"atr|keltner|squeeze|volatility",
    "structure": r"order block|fair value gap|fvg|liquidity|support.{0,10}resistance|pivot|camarilla",
    "volume": r"volume|obv|money flow|vwap",
    "scalping": r"scalp",
    "grid_martingale": r"grid|martingale|safety order|dca",
}


def tag_mechanics(*texts: str | None) -> list[str]:
    blob = " ".join(t for t in texts if t).lower()
    return sorted(k for k, pat in _MECHANIC_TAGS.items() if re.search(pat, blob))


@dataclass
class Candidate:
    """One harvested idea. Performance fields stay None until WE measure them."""
    id: str
    source: str
    name: str
    author: str | None = None
    url: str | None = None
    description: str | None = None
    symbol_hint: str | None = None
    interval_hint: str | None = None
    asset_class: str = "Unknown"
    popularity: int = 0
    has_source: bool = False
    mechanics: list[str] = field(default_factory=list)
    harvested: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # Filled in by the pipeline, never by the harvester.
    status: str = "harvested"      # harvested | screened | tested | rejected | promoted
    pf: float | None = None
    tpd: float | None = None
    cagr: float | None = None
    max_dd: float | None = None
    win_rate: float | None = None
    sharpe: float | None = None
    dsr: float | None = None
    # `trades` is the SAMPLE SIZE the verdict rests on; `trials` is what the
    # verdict COST from the budget. Two different numbers that both have to be
    # visible — a PF of 1.4 on 11 trades and a PF of 1.4 on 900 are not the
    # same claim, and neither is one charged 3 trials against one charged 300.
    trades: int = 0
    trials: int = 0
    # The DATA WINDOW the verdict was produced on. A PF with no period attached
    # is unreadable: "PF 1.6" over 8 months and over 5 years are different
    # claims, and the second is the only one worth acting on.
    tested_from: str | None = None
    tested_to: str | None = None
    years: float | None = None
    test_timeframe: str | None = None
    # Plain-language good/bad findings, JSON. See runner._points(). The dense
    # sentence this replaced could not be skimmed, and a research console that
    # has to be read closely does not get read.
    points: str | None = None
    # Robustness is the scenario matrix (base, costly execution, hold-out,
    # and additional available timeframes/markets).  Kept separately from the
    # headline metrics so the dashboard can state exactly how broad the claim is.
    robustness: str | None = None
    duplicate_of: str | None = None
    # How many passes have picked this row up and failed to get a result out of
    # it. A row that keeps raising never changes status, so without a counter it
    # sits at the head of a popularity-ordered queue and is retried forever.
    attempts: int = 0
    implementation_attempts: int = 0
    audit: str | None = None
    score: int | None = None
    verdict: str | None = None
    note: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY, source TEXT, name TEXT, author TEXT, url TEXT,
    description TEXT, symbol_hint TEXT, interval_hint TEXT, asset_class TEXT,
    popularity INTEGER, has_source INTEGER, mechanics TEXT, harvested TEXT,
    status TEXT, pf REAL, tpd REAL, cagr REAL, max_dd REAL, win_rate REAL,
    sharpe REAL, dsr REAL, trades INTEGER, trials INTEGER, attempts INTEGER,
    score INTEGER, verdict TEXT, note TEXT, implementation_attempts INTEGER DEFAULT 0,
    tested_from TEXT, tested_to TEXT, years REAL, test_timeframe TEXT, points TEXT,
    robustness TEXT, duplicate_of TEXT, audit TEXT
);
CREATE INDEX IF NOT EXISTS idx_source ON candidates(source);
CREATE INDEX IF NOT EXISTS idx_status ON candidates(status);
"""

# Columns added after the first store was created. `CREATE TABLE IF NOT EXISTS`
# does nothing to an existing database, so a new field in `_SCHEMA` alone means
# the running engine on this box keeps a table without it and every write fails.
# Migrations are additive only: dropping or retyping a column here would discard
# measurements that cost trial budget to produce.
_MIGRATIONS = [("trades", "INTEGER DEFAULT 0"),
               ("attempts", "INTEGER DEFAULT 0"),
               ("implementation_attempts", "INTEGER DEFAULT 0"),
               ("tested_from", "TEXT"),
               ("tested_to", "TEXT"),
               ("years", "REAL"),
               ("test_timeframe", "TEXT"),
               ("points", "TEXT"),
               ("robustness", "TEXT"),
               ("duplicate_of", "TEXT")]
_MIGRATIONS.append(("audit", "TEXT"))


class CandidateStore:
    def __init__(self, path: str | Path = STORE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(candidates)")}
        for col, decl in _MIGRATIONS:
            if col not in have:
                self.db.execute(f"ALTER TABLE candidates ADD COLUMN {col} {decl}")
        self.db.commit()

    def upsert(self, c: Candidate) -> bool:
        """Insert, or refresh metadata WITHOUT clobbering measured results.

        A re-harvest must never reset a candidate that has already been tested —
        that would silently free its trial cost and let the same idea be paid for
        twice.
        """
        d = asdict(c)
        d["mechanics"] = json.dumps(c.mechanics)
        d["has_source"] = int(c.has_source)
        existing = self.db.execute(
            "SELECT status FROM candidates WHERE id=?", (c.id,)).fetchone()
        if existing:
            # A re-harvest REFRESHES metadata; it must never DOWNGRADE it.
            #
            # Sources differ in what they carry: the MCP corpus knows the
            # author's chart symbol, the public search endpoint does not. Blindly
            # writing the new record's fields let a later, thinner harvest null
            # out a symbol and reset asset_class to 'Unknown' -- which then got
            # re-routed to a different market, so an already-measured result
            # ended up filed under a universe it was never tested on. Measured
            # 2026-08-11 on SuperTrend STRATEGY: crypto numbers, 'Stocks' label.
            #
            # COALESCE/NULLIF keeps whichever side actually knows something.
            self.db.execute(
                """UPDATE candidates SET
                     source=?, name=?,
                     author=COALESCE(NULLIF(?,''), author),
                     url=COALESCE(NULLIF(?,''), url),
                     description=COALESCE(NULLIF(?,''), description),
                     symbol_hint=COALESCE(NULLIF(?,''), symbol_hint),
                     interval_hint=COALESCE(NULLIF(?,''), interval_hint),
                     asset_class=CASE WHEN ?='Unknown' THEN asset_class ELSE ? END,
                     popularity=MAX(?, popularity),
                     has_source=MAX(?, has_source),
                     mechanics=CASE WHEN ?='[]' THEN mechanics ELSE ? END
                   WHERE id=?""",
                (d["source"], d["name"], d["author"], d["url"], d["description"],
                 d["symbol_hint"], d["interval_hint"],
                 d["asset_class"], d["asset_class"],
                 d["popularity"], d["has_source"],
                 d["mechanics"], d["mechanics"], c.id))
            self.db.commit()
            return False
        cols = ", ".join(d)
        marks = ", ".join("?" * len(d))
        self.db.execute(f"INSERT INTO candidates ({cols}) VALUES ({marks})",
                        list(d.values()))
        self.db.commit()
        return True

    def update_result(self, cid: str, **fields) -> None:
        allowed = {"status", "pf", "tpd", "cagr", "max_dd", "win_rate", "sharpe",
                   "dsr", "trades", "trials", "attempts", "implementation_attempts", "score", "verdict",
                   "note", "tested_from", "tested_to", "years", "test_timeframe", "points",
                   "robustness", "duplicate_of", "audit"}
        bad = set(fields) - allowed
        if bad:
            raise KeyError(f"not result fields: {bad}")
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE candidates SET {sets} WHERE id=?",
                        [*fields.values(), cid])
        self.db.commit()

    def all(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM candidates ORDER BY popularity DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["mechanics"] = json.loads(d["mechanics"] or "[]")
            d["points"] = json.loads(d["points"] or "[]")
            d["robustness"] = json.loads(d["robustness"] or "null")
            d["audit"] = json.loads(d["audit"] or "[]")
            d["has_source"] = bool(d["has_source"])
            out.append(d)
        return out

    def counts(self) -> dict:
        q = "SELECT status, COUNT(*) n FROM candidates GROUP BY status"
        return {r["status"]: r["n"] for r in self.db.execute(q)}

    def append_audit(self, cid: str, event: str, detail: str = "") -> None:
        row = self.db.execute("SELECT audit FROM candidates WHERE id=?", (cid,)).fetchone()
        if not row:
            return
        entries = json.loads(row["audit"] or "[]")
        entries.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "event": event, "detail": detail})
        self.db.execute("UPDATE candidates SET audit=? WHERE id=?",
                        (json.dumps(entries[-20:]), cid))
        self.db.commit()

    def queue(self, limit: int | None = 50) -> list[dict]:
        """Untested candidates, most-popular first — a work order, not a ranking.

        `limit=None` returns ALL of them. The caller needs that, because being
        workable (Pine stored) and being popular are independent, and a fixed
        LIMIT here silently hides the ready rows that sort below it: measured
        2026-08-11, 88 candidates had Pine but only 85 fell inside a 160-row
        slice. That gap grows with every harvest, and the symptom is an engine
        that reports an empty queue while sitting on hundreds of testable ideas.
        """
        sql = ("SELECT * FROM candidates WHERE status='harvested' "
               "ORDER BY popularity DESC")
        rows = (self.db.execute(sql).fetchall() if limit is None
                else self.db.execute(sql + " LIMIT ?", (limit,)).fetchall())
        return [dict(r) for r in rows]


def ingest_records(records: list[dict], source: str = "TradingView",
                   store: CandidateStore | None = None) -> dict:
    """Normalise raw source records into the store.

    `records` is whatever the source hands over — for TradingView, the items
    from the MCP corpus query. Kept as a plain function taking dicts so the
    store never depends on MCP, a network, or Claude Code being present.
    """
    st = store or CandidateStore()
    added = updated = 0
    for r in records:
        sym = r.get("symbol")
        c = Candidate(
            id=f"tv:{r['script_id_part']}" if source == "TradingView"
               else f"{source.lower()}:{r.get('id') or r['name']}",
            source=source,
            name=r.get("name", "").strip(),
            author=r.get("author"),
            url=r.get("chart_url") or r.get("url"),
            description=(r.get("description_snippet") or r.get("description") or "")[:1200],
            symbol_hint=sym,
            interval_hint=r.get("interval"),
            asset_class=classify_asset(sym),
            popularity=int(r.get("likes") or 0),
            has_source=bool(r.get("has_source")),
            mechanics=tag_mechanics(r.get("name"), r.get("description_snippet")),
        )
        if st.upsert(c):
            added += 1
        else:
            updated += 1
    return {"added": added, "updated": updated, "total": len(st.all())}


def to_dashboard(store: CandidateStore | None = None,
                 out: str | Path | None = None) -> dict:
    """Emit dashboard/strategies.json.

    `sample` is False because these rows are real harvests. Every performance
    field is None until measured — the UI renders those as "—", never 0, so a
    collected-but-untested strategy can never be mistaken for a tested one.
    """
    st = store or CandidateStore()
    rows = st.all()
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample": False,
        "strategies": [{
            "id": r["id"], "name": r["name"], "source": r["source"],
            "asset_class": r["asset_class"],
            "symbols": r["symbol_hint"] or "—",
            "author": r["author"], "url": r["url"],
            "mechanics": r["mechanics"],
            "has_source": r["has_source"],
            "popularity": r["popularity"],
            "pf": r["pf"], "tpd": r["tpd"], "cagr": r["cagr"],
            "max_dd": r["max_dd"], "win_rate": r["win_rate"],
            "sharpe": r["sharpe"], "dsr": r["dsr"],
            "trades": r.get("trades") or 0,
            "trials": r["trials"] or 0,
            "implementation_attempts": r.get("implementation_attempts") or 0,
            "tested_from": r.get("tested_from"), "tested_to": r.get("tested_to"),
            "years": r.get("years"),
            "test_timeframe": r.get("test_timeframe"),
            "points": r.get("points") or [],
            "robustness": r.get("robustness"),
            "duplicate_of": r.get("duplicate_of"),
            "audit": r.get("audit") or [],
            "score": r["score"],
            "verdict": r["verdict"] or "pending",
            "note": r["note"] or _pending_note(r),
        } for r in rows],
    }
    path = Path(out or Path(__file__).resolve().parent.parent
                / "dashboard" / "strategies.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Browser refreshes can happen while a worker finishes.  Never expose a
    # half-written JSON document: it makes the UI fall back to sample data.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=1, allow_nan=False))
    os.replace(tmp, path)
    return {"written": str(path), "rows": len(rows)}


def _json_safe(value):
    """Make strict browser JSON; Python's Infinity is not valid JSON there."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _pending_note(r: dict) -> str:
    bits = []
    if r["mechanics"]:
        bits.append("mechanics: " + ", ".join(r["mechanics"]))
    if r["has_source"]:
        bits.append("Pine source available")
    bits.append("collected, not yet screened — no metrics measured")
    return ". ".join(bits)
