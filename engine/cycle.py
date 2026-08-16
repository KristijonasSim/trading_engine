"""One full pass: harvest -> adapt -> screen -> holdout. Zero tokens.

This is what the 24/7 timer runs. It deliberately does NOT call `translate.py`,
so a pass costs no model quota at all — the whole loop is scrapers, a
deterministic adapter, and `trading-bots/scalping/backtest.py`.

PACE IS NOT THE POINT
---------------------
A faster pass is not a better pass. The screen may be wide because the holdout
cleans up after it (see `stages.py`), but the holdout itself is a real claim on
real data and there is no value in making more of those per hour. The timer runs
every 30 minutes because that is often enough to keep the queue moving, not
because throughput is the goal.

WHAT A PASS DOES NOT DO
-----------------------
It does not promote anything to a bot. `promotion.py` is the only path to that,
it is gated, and it stays manual. A survivor here is a research finding: it has
cleared two clean windows and nothing more.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import activity, stages
from .harvest import CandidateStore, to_dashboard
from .harvest_github import SEED_REPOS, harvest_repo

STATE = Path(__file__).resolve().parent.parent / "state"
LOG = STATE / "cycle_log.json"

# How many new files to pull per pass. Small on purpose: unauthenticated GitHub
# allows 60 requests/hour, and a pass that trips the limit poisons the next few.
HARVEST_PER_PASS = 8
EVALUATE_PER_PASS = 10


def run(*, harvest: bool = True, evaluate: bool = True) -> dict:
    st = CandidateStore()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: dict = {"at": started}

    if harvest:
        activity.write(status="running", started=started,
                       current={"stage": "harvesting GitHub", "name": "—",
                                "asset_class": "Crypto", "number": 0, "total": 0})
        try:
            seen = set()
            total = {"seen": 0, "stored": 0, "duplicate": 0,
                     "unsupported": 0, "errors": 0}
            for repo in SEED_REPOS:
                s = harvest_repo(repo, st, limit=HARVEST_PER_PASS, seen_fp=seen)
                for k in total:
                    total[k] = getattr(s, k)
            out["harvest"] = total
        except RuntimeError as e:          # rate limit — not an error worth retrying
            out["harvest"] = {"stopped": str(e)}
        except Exception as e:
            out["harvest"] = {"error": f"{type(e).__name__}: {e}"[:200]}

    if evaluate:
        activity.write(status="running", started=started,
                       current={"stage": "screening and holdout", "name": "—",
                                "asset_class": "Crypto", "number": 0, "total": 0})
        try:
            out["stages"] = stages.run(limit=EVALUATE_PER_PASS)
        except Exception as e:
            out["stages"] = {"error": f"{type(e).__name__}: {e}"[:200]}

    try:
        to_dashboard(st)
    except Exception:
        pass

    # Publish the finished state LAST, so the UI never shows "idle" while a
    # pass is still writing results.
    s = out.get("stages") or {}
    activity.write(
        status="error" if s.get("error") else "idle",
        started=started,
        current={"stage": "last completed", "name": "pass finished",
                 "asset_class": "Crypto", "number": 0, "total": 0},
        summary={"tested": s.get("evaluated", 0),
                 "rejected": s.get("evaluated", 0) - s.get("survivors", 0),
                 "promoted": s.get("survivors", 0),
                 "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        error=s.get("error"))

    STATE.mkdir(parents=True, exist_ok=True)
    hist = []
    if LOG.exists():
        try:
            hist = json.loads(LOG.read_text())
        except json.JSONDecodeError:
            hist = []
    hist.append(out)
    LOG.write_text(json.dumps(hist[-200:], indent=1, default=str))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
