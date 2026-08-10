"""Idea 2 — the feed manufacturer. Turn TEXT into a numeric series.

WHY THIS IS THE HIGHEST-VALUE MODULE HERE
-----------------------------------------
trading-bots' MEASURED LIMITS: external idea hunts are 0-for-351, and every leg
that ever worked came from a NEW DATA FEED or a proven mechanic pointed at new
data. A new formula over bars we already hold spends trial budget and adds no
information.

This module is the only one in the engine that CREATES information rather than
reprocessing it. An LLM reading FOMC statements into a hawkish/dovish score is a
series that did not exist before — it costs no trial budget to produce, and it
raises the ceiling instead of consuming the allowance.

WHAT IT DOES NOT DO
-------------------
It does not generate hypotheses. Generating ideas faster is not the constraint;
the trial budget is, and an LLM proposing a thousand strategies a day makes the
arithmetic strictly worse. The LLM's job here is READING, not inventing.

THE LAG IS THE WHOLE GAME
-------------------------
A document has two timestamps and they are not the same: when it describes the
world, and when the public could read it. Scoring an FOMC statement onto the
morning of the meeting leaks the afternoon's decision. Every series this module
emits is indexed by RELEASE time and offered through `as_feature()`, which
additionally shifts by one bar so a bar can never see a score published inside
itself. This is trading-bots HARD RULE 7, which has already cost ~0.5 PF of
pure look-ahead once.

SCORERS ARE PLUGGABLE, AND THE DEFAULT WORKS OFFLINE
----------------------------------------------------
`LexiconScorer` needs no API key and is deterministic, so the pipeline is
testable and reproducible without network or spend. `ClaudeScorer` is better at
the actual task and is used when a key is present. Both satisfy the same
protocol, and scores are cached to disk keyed by (scorer, document) so a
re-run never pays twice.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (systematic-research; personal use)"}
CACHE = Path(__file__).resolve().parent.parent / "state" / "feed_cache"


# ------------------------------------------------------------------ documents
@dataclass(frozen=True)
class Document:
    """One piece of text with the moment it became PUBLIC.

    `released` is not when the document was written or what period it describes
    — it is when a trader could first have acted on it. Everything downstream
    keys off this.
    """
    id: str
    released: pd.Timestamp
    text: str
    source: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


def _get(url: str, timeout: int = 45) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf8", "ignore")


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = (html.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&#39;", "'").replace("&quot;", '"'))
    return re.sub(r"\s+", " ", html).strip()


def fomc_statements(limit: int = 40) -> list[Document]:
    """FOMC policy statements, newest first. Free, no key, ~8 per year.

    Release time: statements land at 14:00 US Eastern on the second day of the
    meeting. We record 19:00 UTC, which is 14:00 EST — deliberately the LATER
    of the two possible offsets, because being an hour pessimistic costs
    nothing while being an hour optimistic is look-ahead.
    """
    cal = _get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    links = sorted(set(re.findall(
        r"/newsevents/pressreleases/monetary(\d{8})a\.htm", cal)), reverse=True)
    docs = []
    for stamp in links[:limit]:
        url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{stamp}a.htm"
        try:
            body = _strip_html(_get(url))
        except Exception:                                        # noqa: BLE001
            continue
        # keep the policy discussion, drop nav furniture
        m = re.search(r"(?s)(Recent indicators|The Committee|Information received).{200,}",
                      body)
        text = m.group(0)[:8000] if m else body[:8000]
        released = (pd.Timestamp(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}", tz="UTC")
                    + pd.Timedelta(hours=19))
        docs.append(Document(id=f"fomc_{stamp}", released=released,
                             text=text, source="fomc"))
    return docs


# -------------------------------------------------------------------- scorers
class Scorer(Protocol):
    name: str

    def score(self, doc: Document) -> float:
        """Map a document to a number. Sign and scale must be stable."""


class LexiconScorer:
    """Offline hawkish/dovish score in [-1, 1]. Deterministic, no key.

    Crude by construction — it counts words. It exists so the pipeline is
    runnable and testable without network or spend, and as a CONTROL: if an
    LLM score cannot beat word-counting on the screen, the LLM is not adding
    anything and should not be paid for.
    """
    name = "lexicon_v1"

    HAWK = {"inflation", "tighten", "tightening", "restrictive", "raise",
            "raising", "increase", "elevated", "overheating", "firm", "robust",
            "strong", "persistent", "vigilant", "resolute"}
    DOVE = {"accommodative", "cut", "cutting", "lower", "lowering", "ease",
            "easing", "weak", "weakened", "softening", "slowing", "downside",
            "moderate", "moderated", "patient", "support"}

    def score(self, doc: Document) -> float:
        words = re.findall(r"[a-z]+", doc.text.lower())
        if not words:
            return 0.0
        h = sum(w in self.HAWK for w in words)
        d = sum(w in self.DOVE for w in words)
        if h + d == 0:
            return 0.0
        return (h - d) / (h + d)


class ClaudeScorer:
    """LLM scorer. Used only when ANTHROPIC_API_KEY is present.

    Temperature 0 and a tightly bounded output format, because a feature series
    that wobbles between runs is not a feature — it is noise with extra steps.
    """
    name = "claude_hawkdove_v1"

    PROMPT = (
        "You are scoring a central bank statement on a single axis.\n"
        "-1.0 = maximally dovish (easing, cuts, weakness emphasised)\n"
        " 0.0 = neutral / unchanged stance\n"
        "+1.0 = maximally hawkish (tightening, hikes, inflation emphasised)\n\n"
        "Judge the STANCE SHIFT relative to a normal policy statement, not the "
        "absolute level of rates. Reply with ONLY a number between -1 and 1.\n\n"
        "STATEMENT:\n{text}"
    )

    def __init__(self, model: str = "claude-opus-5"):
        self.model = model
        try:
            import anthropic
        except ImportError as e:                                 # pragma: no cover
            raise RuntimeError(
                "pip install anthropic to use ClaudeScorer") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic()

    def score(self, doc: Document) -> float:
        r = self._client.messages.create(
            model=self.model, max_tokens=8, temperature=0,
            messages=[{"role": "user",
                       "content": self.PROMPT.format(text=doc.text[:6000])}])
        raw = r.content[0].text.strip()
        m = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not m:
            raise ValueError(f"unparseable score {raw!r}")
        return max(-1.0, min(1.0, float(m.group())))


def best_available_scorer() -> Scorer:
    """ClaudeScorer when a key exists, else the offline lexicon."""
    try:
        return ClaudeScorer()
    except Exception:                                            # noqa: BLE001
        return LexiconScorer()


# --------------------------------------------------------------------- series
class TextFeed:
    """Documents + a scorer -> a cached, correctly-lagged numeric series."""

    def __init__(self, name: str, scorer: Scorer | None = None,
                 cache_dir: Path | None = None):
        self.name = name
        self.scorer = scorer or best_available_scorer()
        self.cache_dir = Path(cache_dir or CACHE)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _cache_path(self) -> Path:
        return self.cache_dir / f"{self.name}__{self.scorer.name}.json"

    def _load(self) -> dict[str, float]:
        if self._cache_path.exists():
            return json.loads(self._cache_path.read_text())
        return {}

    def build(self, docs: list[Document]) -> pd.Series:
        """Score every document, caching by content digest.

        Keyed on the digest, not the id, so an amended document is re-scored
        rather than silently keeping its old number.
        """
        cache = self._load()
        out = {}
        for d in docs:
            key = f"{d.id}:{d.digest}"
            if key not in cache:
                cache[key] = float(self.scorer.score(d))
            out[d.released] = cache[key]
        self._cache_path.write_text(json.dumps(cache, indent=2))
        return pd.Series(out, name=self.name).sort_index()

    @staticmethod
    def as_feature(series: pd.Series, bar_index: pd.DatetimeIndex,
                   ffill_limit: int | None = None) -> pd.Series:
        """Align onto bars so no bar can see a score published inside itself.

        Two guards, both necessary:
          1. `searchsorted(side='right')` — a bar takes only scores released
             STRICTLY before the bar opens.
          2. one further `.shift(1)` — the bar that contains the release still
             does not act on it, because the release lands mid-bar and a
             closed-bar system could not have traded on it any earlier.
        Dropping either one manufactures look-ahead that a backtest will
        happily report as edge.
        """
        if series.empty:
            return pd.Series(index=bar_index, dtype=float, name=series.name)
        s = series.sort_index()
        s_idx = s.index
        if getattr(s_idx, "tz", None) is not None and bar_index.tz is None:
            bar_index = bar_index.tz_localize("UTC")
        pos = s_idx.searchsorted(bar_index, side="right") - 1
        vals = [s.iloc[p] if p >= 0 else float("nan") for p in pos]
        out = pd.Series(vals, index=bar_index, name=series.name).shift(1)
        if ffill_limit is not None:
            out = out.ffill(limit=ffill_limit)
        return out
