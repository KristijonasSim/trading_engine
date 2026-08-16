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

# Files per repo per pass. Generous now that contents come from
# raw.githubusercontent.com, which is NOT rate-limited — only the one tree
# listing per repo is. The old value of 8 was sized for the API blob endpoint
# and left the queue starving.
HARVEST_PER_PASS = 120

# Backtests per pass. This one stays small ON PURPOSE. Harvesting is free;
# evaluating is the part that makes claims about data, and there is no value in
# making more of those per hour. See stages.py.
EVALUATE_PER_PASS = 25


def run(*, harvest: bool = True, evaluate: bool = True) -> dict:
    st = CandidateStore()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: dict = {"at": started}

    if harvest:
        activity.write(status="running", started=started,
                       current={"stage": "harvesting GitHub", "name": "—",
                                "asset_class": "Crypto", "number": 0, "total": 0})
        # ONE shared stats object and ONE shared fingerprint set across repos.
        # Passing neither meant each call allocated its own, so the totals
        # reported the LAST repo rather than the pass, and the same strategy
        # forked into two repos was stored twice.
        from .harvest_github import HarvestStats
        agg, seen = HarvestStats(), set()
        stopped = None
        for repo in SEED_REPOS:
            try:
                harvest_repo(repo, st, limit=HARVEST_PER_PASS,
                             stats=agg, seen_fp=seen)
            except RuntimeError as e:
                # Rate limited. Stop walking further repos — the cap is global,
                # not per-repo — but keep whatever this pass already stored.
                stopped = str(e)
                break
            except Exception as e:
                agg.errors += 1
                stopped = f"{type(e).__name__}: {e}"[:200]
        out["harvest"] = agg.as_dict()
        if stopped:
            out["harvest"]["stopped"] = stopped

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
