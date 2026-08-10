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
    trials: int = 0
    score: int | None = None
    verdict: str | None = None
    note: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY, source TEXT, name TEXT, author TEXT, url TEXT,
    description TEXT, symbol_hint TEXT, interval_hint TEXT, asset_class TEXT,
    popularity INTEGER, has_source INTEGER, mechanics TEXT, harvested TEXT,
    status TEXT, pf REAL, tpd REAL, cagr REAL, max_dd REAL, win_rate REAL,
    sharpe REAL, dsr REAL, trials INTEGER, score INTEGER, verdict TEXT, note TEXT
);
CREATE INDEX IF NOT EXISTS idx_source ON candidates(source);
CREATE INDEX IF NOT EXISTS idx_status ON candidates(status);
"""


class CandidateStore:
    def __init__(self, path: str | Path = STORE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

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
            self.db.execute(
                """UPDATE candidates SET source=?, name=?, author=?, url=?,
                   description=?, symbol_hint=?, interval_hint=?, asset_class=?,
                   popularity=?, has_source=?, mechanics=? WHERE id=?""",
                (d["source"], d["name"], d["author"], d["url"], d["description"],
                 d["symbol_hint"], d["interval_hint"], d["asset_class"],
                 d["popularity"], d["has_source"], d["mechanics"], c.id))
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
                   "dsr", "trials", "score", "verdict", "note"}
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
            d["has_source"] = bool(d["has_source"])
            out.append(d)
        return out

    def counts(self) -> dict:
        q = "SELECT status, COUNT(*) n FROM candidates GROUP BY status"
        return {r["status"]: r["n"] for r in self.db.execute(q)}

    def queue(self, limit: int = 50) -> list[dict]:
        """Untested candidates, most-popular first — a work order, not a ranking."""
        rows = self.db.execute(
            "SELECT * FROM candidates WHERE status='harvested' "
            "ORDER BY popularity DESC LIMIT ?", (limit,)).fetchall()
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
            "popularity": r["popularity"],
            "pf": r["pf"], "tpd": r["tpd"], "cagr": r["cagr"],
            "max_dd": r["max_dd"], "win_rate": r["win_rate"],
            "sharpe": r["sharpe"], "dsr": r["dsr"],
            "trials": r["trials"] or 0,
            "score": r["score"],
            "verdict": r["verdict"] or "pending",
            "note": r["note"] or _pending_note(r),
        } for r in rows],
    }
    path = Path(out or Path(__file__).resolve().parent.parent
                / "dashboard" / "strategies.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return {"written": str(path), "rows": len(rows)}


def _pending_note(r: dict) -> str:
    bits = []
    if r["mechanics"]:
        bits.append("mechanics: " + ", ".join(r["mechanics"]))
    if r["has_source"]:
        bits.append("Pine source available")
    bits.append("collected, not yet screened — no metrics measured")
    return ". ".join(bits)
