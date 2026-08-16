"""Harvest Python strategies from GitHub — and from a local folder.

WHY GITHUB AND NOT ONLY TRADINGVIEW
-----------------------------------
Not because the ideas are better. `trading-bots/CLAUDE.md` records the measured
result on that question: external idea hunts are 0-for-351, and every leg that
actually trades live came from a new DATA FEED, not from a strategy name. The
reason to add this source is narrower and purely economic — GitHub strategies
are ALREADY PYTHON, so they enter the pipeline through `adapters.py` at zero
token cost, where a Pine script costs ~10k tokens and up to three retries.

So this module changes what harvesting costs. It does not claim to change what
harvesting finds.

RATE LIMITS
-----------
Unauthenticated GitHub allows 60 requests/hour, which is enough to walk a couple
of repos and no more. Set GITHUB_TOKEN for 5,000/hour. The harvester stops
cleanly on a 403 rather than hammering — a half-finished harvest is fine, the
queue is resumable.

DEDUPE BEFORE THE QUEUE
-----------------------
Strategies are forked and re-uploaded constantly. Two copies of the same file
are one idea, and testing the second spends hold-out budget re-answering a
settled question. Files are fingerprinted whitespace-insensitively
(`pysource.fingerprint`) and duplicates are dropped at ingest.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import pysource
from .adapters import adapt, detect
from .harvest import Candidate, CandidateStore, classify_asset, tag_mechanics

API = "https://api.github.com"
UA = "trading-engine-harvester"

# Repos worth walking. Kept as an explicit list rather than a live topic search:
# the topic index is noisy, and an explicit list is auditable — you can see what
# the engine ingested without re-running it.
SEED_REPOS = [
    "freqtrade/freqtrade-strategies",
    "nateemma/strategies",
    "TheoBrigitte/freqtrade",
]

# Paths inside a repo that hold strategies. Everything else (tests, utils,
# configs) is skipped without being fetched, which matters against a 60/hour cap.
LIKELY = ("strateg", "user_data")
SKIP = ("test", "__pycache__", "hyperopt", "setup.py", "conftest")


@dataclass
class HarvestStats:
    seen: int = 0
    stored: int = 0
    duplicate: int = 0
    unsupported: int = 0
    errors: int = 0

    def as_dict(self) -> dict:
        return {"seen": self.seen, "stored": self.stored,
                "duplicate": self.duplicate, "unsupported": self.unsupported,
                "errors": self.errors}


def _get(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "application/vnd.github+json"})
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _tree(repo: str) -> list[dict]:
    """Every file in a repo's default branch, one request."""
    info = _get(f"{API}/repos/{repo}")
    branch = info.get("default_branch", "main")
    tree = _get(f"{API}/repos/{repo}/git/trees/{branch}?recursive=1")
    return [t for t in tree.get("tree", []) if t.get("type") == "blob"]


def _wanted(path: str) -> bool:
    low = path.lower()
    if not low.endswith(".py"):
        return False
    if any(s in low for s in SKIP):
        return False
    return any(s in low for s in LIKELY)


def _blob(repo: str, sha: str) -> str:
    b = _get(f"{API}/repos/{repo}/git/blobs/{sha}")
    if b.get("encoding") != "base64":
        raise ValueError("unexpected blob encoding")
    return base64.b64decode(b["content"]).decode("utf-8", "replace")


def _ingest_code(store: CandidateStore, *, cid: str, name: str, url: str,
                 code: str, author: str, source: str,
                 seen_fp: set[str], stats: HarvestStats) -> None:
    stats.seen += 1
    kind = detect(code)
    if not kind:
        stats.unsupported += 1
        return
    fp = pysource.fingerprint(code)
    if fp in seen_fp:
        stats.duplicate += 1
        return
    result = adapt(code)
    if not result.ok:
        stats.unsupported += 1
        return
    seen_fp.add(fp)
    pysource.store(cid, result.code)
    c = Candidate(
        id=cid, source=source, name=name, author=author, url=url,
        description=f"{result.kind} strategy adapted deterministically — no LLM",
        asset_class=classify_asset(None),
        mechanics=tag_mechanics(name, code[:2000]),
        has_source=True, source_quality=2,
    )
    store.upsert(c)
    stats.stored += 1


def harvest_repo(repo: str, store: CandidateStore, *, limit: int = 40,
                 stats: HarvestStats | None = None,
                 seen_fp: set[str] | None = None) -> HarvestStats:
    stats = stats or HarvestStats()
    seen_fp = seen_fp if seen_fp is not None else set()
    try:
        files = [t for t in _tree(repo) if _wanted(t["path"])][:limit]
    except urllib.error.HTTPError as e:
        stats.errors += 1
        if e.code in (403, 429):
            raise RuntimeError(
                f"GitHub rate limit hit on {repo}. Set GITHUB_TOKEN to raise "
                f"the cap from 60/hour to 5,000/hour.") from e
        return stats
    for f in files:
        try:
            code = _blob(repo, f["sha"])
        except Exception:
            stats.errors += 1
            continue
        stem = Path(f["path"]).stem
        _ingest_code(
            store, cid=f"gh:{repo}:{f['path']}", name=stem,
            url=f"https://github.com/{repo}/blob/HEAD/{f['path']}",
            code=code, author=repo.split("/")[0], source="GitHub",
            seen_fp=seen_fp, stats=stats)
        time.sleep(0.2)
    return stats


def harvest_local(folder: str | Path, store: CandidateStore,
                  stats: HarvestStats | None = None,
                  seen_fp: set[str] | None = None) -> HarvestStats:
    """Ingest .py strategies from a local directory. No network, no limits."""
    stats = stats or HarvestStats()
    seen_fp = seen_fp if seen_fp is not None else set()
    root = Path(folder)
    for f in sorted(root.rglob("*.py")):
        try:
            code = f.read_text(errors="replace")
        except Exception:
            stats.errors += 1
            continue
        rel = f.relative_to(root)
        _ingest_code(store, cid=f"local:{rel}", name=f.stem,
                     url=str(f), code=code, author="local", source="Local",
                     seen_fp=seen_fp, stats=stats)
    return stats


def run(repos: list[str] | None = None, *, limit: int = 40) -> dict:
    store = CandidateStore()
    stats, seen = HarvestStats(), set()
    for repo in (repos or SEED_REPOS):
        try:
            harvest_repo(repo, store, limit=limit, stats=stats, seen_fp=seen)
        except RuntimeError as e:
            return {**stats.as_dict(), "stopped": str(e)}
    return stats.as_dict()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
