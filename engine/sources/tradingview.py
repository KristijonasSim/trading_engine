"""TradingView harvesting over the PUBLIC endpoints. No MCP, no Claude session.

WHY THIS EXISTS
---------------
The first harvest came from the TradingView MCP corpus, which is a local index
sitting on one of Kristijonas' two machines (`~/trading-robots/
tradingview-indicator-search-mcp-server`). That path does not exist on this
box, and an MCP server is reachable only from inside a Claude Code session
anyway — so an engine that depends on it cannot run unattended on a timer, and
cannot be deployed to a server at all.

This module talks to the two public endpoints directly, so the loop is
self-sufficient:

  discovery  https://www.tradingview.com/pubscripts-suggest-json/?search=<term>
             50 records per query. Carries name, author, like count, script id,
             `extra.kind` (study vs strategy) and often the Pine source inline.
  source     https://pine-facade.tradingview.com/pine-facade/get/<id>/last/
             the full Pine for open scripts, as `source`.

Both are the endpoints the site's own search box uses. Requests are paced —
this is somebody's server.

WHAT IS FILTERED OUT AND WHY
----------------------------
`extra.kind == "study"` — an indicator plots a line. It has no entry, no exit
and no stop, so there is nothing to translate into a signal function and
nothing to backtest. 153 of the first 250 records surveyed were studies.

`access != 1` — protected and invite-only scripts publish no source. Harvesting
them would produce rows that can never leave the "awaiting Pine source" state
and would pad the dashboard with permanent pending.

THE ASSET-CLASS PROBLEM, AND THE HONEST FIX
-------------------------------------------
The MCP corpus carried the author's chart symbol; the public API does not. That
matters because `runner.UNIVERSES` is keyed by asset class, and an unclassified
row is rejected before it is ever tested.

The fix is NOT to test each strategy on every universe and keep whichever
worked — that is max-picking across four markets, precisely the selection bias
this engine exists to charge for. Instead the universe is assigned
DETERMINISTICALLY FROM THE SCRIPT ID, before any data is touched, and never
revised afterwards. `assign_universe()` is a pure function of the id, so a
re-harvest lands the same strategy in the same market forever, and no result
can be improved by re-rolling it.

Two ordering rules on top, both applied before the hash:
  - a name or source that names a crypto instrument goes to Crypto, because
    testing a funding-rate or perp-specific rule on gold is not a fair test
  - everything else spreads across FX / Stocks / Futures, which is where the
    honest trial budget actually lives: 20-25 years of daily history buys a
    1000-trial allowance, against crypto's 166 on 5.1 years

That spread is a feature, not a compromise. Crypto's budget is the scarce one.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

STATE = Path(__file__).resolve().parent.parent.parent / "state"
PINE = STATE / "pine"

SEARCH_URL = "https://www.tradingview.com/pubscripts-suggest-json/?search={q}"
SOURCE_URL = "https://pine-facade.tradingview.com/pine-facade/get/{q}/last/"

# TradingView 403s an unadorned urllib agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
           "Accept": "application/json, text/plain, */*"}

PACE = 0.6          # seconds between requests. Somebody else's server.
TIMEOUT = 25

# Search vocabulary. Deliberately mechanic words rather than "best" or
# "profitable" — the point is coverage of STRATEGY FAMILIES, and the like count
# already orders the work queue. Terms overlap heavily; ids are deduped by the
# store, so overlap costs a request and nothing else.
TERMS = [
    "strategy", "backtest", "trading system", "algo",
    "supertrend", "moving average cross", "ema strategy", "macd strategy",
    "rsi strategy", "bollinger strategy", "stochastic strategy", "ichimoku",
    "breakout", "donchian", "opening range", "channel breakout",
    "mean reversion", "vwap", "pivot", "support resistance",
    "trend following", "momentum", "swing", "scalping", "intraday",
    "atr", "keltner", "squeeze", "volatility strategy",
    "order block", "fair value gap", "smart money", "liquidity",
    "grid", "dca", "martingale", "hedge",
    "volume profile", "obv", "money flow", "cumulative delta",
    "heikin ashi", "renko", "parabolic sar", "adx",
    "divergence", "fibonacci", "harmonic", "elliott",
    "range filter", "chandelier", "hull", "kalman",
    "relative strength", "pairs trading", "cointegration", "statistical arbitrage",
    "trend pullback", "breakout retest", "volatility breakout", "volume breakout",
    "gap strategy", "seasonality", "market profile", "anchored vwap",
    "rsi divergence", "macd divergence", "supply demand", "liquidity sweep",
    "nadaraya watson", "linear regression", "adaptive moving average", "vortex",
    "cmo", "williams r", "rate of change", "relative volume",
]

_CRYPTO_WORDS = re.compile(
    r"\b(btc|bitcoin|eth|ethereum|crypto|altcoin|usdt|perp|binance|bybit|"
    r"funding rate|satoshi|doge|solana|xrp)\b", re.I)

# Everything without a crypto tell spreads across the long-history universes.
_SPREAD = ("FX", "Stocks", "Futures")


def _get_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None


def safe_id(cid: str) -> str:
    """Candidate id -> filename stem. Must match runner._pine_source()."""
    return cid.replace(":", "_").replace(";", "_").replace("/", "_")


def assign_universe(script_id: str, name: str = "", source: str = "") -> str:
    """Which market this strategy will be tested on. Pure, and fixed for good.

    Deterministic on purpose: the assignment is made before any data is loaded
    and cannot be re-rolled after seeing a bad result. See the module docstring
    for why testing one strategy across all four universes would be cheating.
    """
    if _CRYPTO_WORDS.search(f"{name} {source[:4000]}"):
        return "Crypto"
    h = int(hashlib.sha256(script_id.encode()).hexdigest()[:8], 16)
    return _SPREAD[h % len(_SPREAD)]


def search(term: str) -> list[dict]:
    """Raw records for one search term, or [] if the endpoint is unreachable.

    A failed search is not an error worth stopping a pass for — the next pass
    tries again, and the queue is usually already full of untested work.
    """
    d = _get_json(SEARCH_URL.format(q=urllib.parse.quote(term)))
    return (d or {}).get("results") or []


def fetch_pine(script_id: str) -> str | None:
    """Full Pine source for an open script."""
    d = _get_json(SOURCE_URL.format(q=urllib.parse.quote(script_id, safe="")))
    if not d:
        return None
    if d.get("scriptAccess") not in (None, "open_no_auth", "open"):
        return None
    src = d.get("source")
    return src if src and "strategy(" in src else src or None


def to_record(raw: dict) -> dict | None:
    """One suggest-json result -> the dict `harvest.ingest_records` expects.

    Returns None for anything not worth storing: indicators (nothing to
    backtest) and closed-source scripts (nothing to translate).
    """
    extra = raw.get("extra") or {}
    if extra.get("kind") != "strategy":
        return None
    if raw.get("access") != 1:
        return None
    sid = raw.get("scriptIdPart")
    if not sid:
        return None
    author = (raw.get("author") or {}).get("username")
    src = raw.get("scriptSource") or ""
    name = (raw.get("scriptName") or raw.get("title") or "").strip()
    slug = raw.get("imageUrl")
    return {
        "script_id_part": sid,
        "name": name,
        "author": author,
        "likes": int(raw.get("agreeCount") or 0),
        # No symbol from this endpoint — the universe is assigned, not read.
        "symbol": None,
        "universe": assign_universe(sid, name, src),
        "interval": None,
        "chart_url": f"https://www.tradingview.com/script/{slug}/" if slug else None,
        "has_source": True,
        "description_snippet": (raw.get("shortTitle") or "")[:400],
        "_inline_source": src,
    }


def store_pine(cid: str, source: str) -> Path:
    PINE.mkdir(parents=True, exist_ok=True)
    p = PINE / f"{safe_id(cid)}.pine"
    p.write_text(source)
    return p


def harvest(terms: list[str] | None = None, *, store=None, max_terms: int = 6,
            offset: int = 0, fetch_sources: bool = True,
            max_fetch: int = 25) -> dict:
    """One harvest sweep: search a slice of the vocabulary, store what is new.

    `offset` rotates the slice so consecutive passes cover different terms
    instead of re-querying the same six forever. The runner derives it from the
    pass count.

    Harvesting costs NO trial budget — collecting somebody's published idea is
    not testing it. Only `pipeline.evaluate` spends.
    """
    from ..harvest import CandidateStore, ingest_records

    st = store or CandidateStore()
    vocab = terms or TERMS
    slice_ = [vocab[(offset + i) % len(vocab)] for i in range(max_terms)]

    records, inline = [], {}
    for term in slice_:
        for raw in search(term):
            rec = to_record(raw)
            if rec is None:
                continue
            src = rec.pop("_inline_source", "")
            if src:
                inline[f"tv:{rec['script_id_part']}"] = src
            records.append(rec)
        time.sleep(PACE)

    # Dedupe inside the sweep; the store dedupes across sweeps.
    seen, uniq = set(), []
    for r in records:
        if r["script_id_part"] in seen:
            continue
        seen.add(r["script_id_part"])
        uniq.append(r)

    res = ingest_records(uniq, source="TradingView", store=st)

    # The universe is part of the harvest decision, not a test-time choice, so
    # it is written now and never touched again.
    #
    # Only where nothing better is known. Rows harvested from the MCP corpus
    # carry the author's real chart symbol, and `classify_asset()` on a real
    # symbol beats a hash every time — clobbering those would throw away the
    # one piece of genuine routing information in the store.
    # `status='harvested'` is the important half of this clause. Once a row has
    # been TESTED, its result was produced under the universe it was assigned at
    # the time; re-pointing it afterwards would file real numbers under a market
    # they were never measured on, and would let a bad result be re-rolled into
    # a different budget. The assignment is made once, before any data is read.
    for r in uniq:
        st.db.execute(
            "UPDATE candidates SET asset_class=? WHERE id=? "
            "AND status='harvested' "
            "AND (symbol_hint IS NULL OR symbol_hint='' OR asset_class='Unknown')",
            (r["universe"], f"tv:{r['script_id_part']}"))
    st.db.commit()

    # Pine sources. Inline first (free — it came with the search), then fetch
    # the rest, newest-first by popularity, capped so a pass stays short.
    PINE.mkdir(parents=True, exist_ok=True)
    stored = 0
    for cid, src in inline.items():
        if not (PINE / f"{safe_id(cid)}.pine").exists():
            store_pine(cid, src)
            stored += 1

    fetched = 0
    if fetch_sources:
        missing = [r for r in st.all()
                   if r["status"] == "harvested"
                   and not (PINE / f"{safe_id(r['id'])}.pine").exists()]
        for r in missing[:max_fetch]:
            sid = r["id"].split(":", 1)[1]
            src = fetch_pine(sid)
            if src:
                store_pine(r["id"], src)
                fetched += 1
            time.sleep(PACE)

    return {"terms": slice_, "seen": len(uniq), **res,
            "pine_inline": stored, "pine_fetched": fetched,
            "pine_total": len(list(PINE.glob("*.pine")))}
