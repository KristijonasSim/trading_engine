"""Independent Pine implementation/repair lane.

This worker never measures performance.  It turns a parked Pine strategy into
verified executable code, or leaves it parked with a concise tooling reason.
Only verifier-approved code is returned to the ordinary research queue.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import activity
from .harvest import CandidateStore, to_dashboard
from .runner import UNIVERSES, _implementation, _load_bars
from .translate import best_translator, verify

MAX_IMPLEMENTATION_ATTEMPTS = int(os.environ.get("IMPLEMENTATION_MAX_ATTEMPTS", "3"))
ACTIVITY_OUT = Path(__file__).resolve().parent.parent / "dashboard" / "implementation-activity.json"


def _queue(store: CandidateStore) -> list[dict]:
    rows = [r for r in store.all() if r["status"] == "blocked" and
            r.get("source") != "Invented" and _implementation(r)]
    # Work cheap, clear, popular mechanics first; complex failures remain
    # visible but do not monopolise the repair lane.
    def key(r: dict):
        mechanics = set(r.get("mechanics") or [])
        clarity = 20 * len(mechanics & {"trend", "breakout", "momentum", "mean_reversion", "volume", "structure"})
        complexity_penalty = 100 if "grid_martingale" in mechanics else 0
        return (r.get("implementation_attempts") or 0, -(clarity + min(r.get("popularity") or 0, 500) - complexity_penalty), r["id"])
    return sorted(rows, key=key)


def run_once(store: CandidateStore | None = None) -> dict:
    """Try one parked implementation and return a small activity summary."""
    st = store or CandidateStore()
    translator = best_translator()
    if translator is None:
        activity.write(status="waiting", error="no Pine translator available", out=ACTIVITY_OUT)
        return {"status": "waiting", "reason": "no Pine translator available"}
    for row in _queue(st):
        previous = row.get("implementation_attempts") or 0
        if previous >= MAX_IMPLEMENTATION_ATTEMPTS:
            continue
        source = _implementation(row)
        cfg = UNIVERSES.get(row["asset_class"])
        if not source or not cfg:
            continue
        attempt = previous + 1
        # Make work-in-progress visible in the shared UI before a potentially
        # long Claude invocation.  It will always leave this state as either
        # parked (still blocked) or verified (back in the test queue).
        st.update_result(row["id"], status="implementing", verdict="implementing",
                         implementation_attempts=attempt,
                         note=f"implementation attempt {attempt}/{MAX_IMPLEMENTATION_ATTEMPTS} running")
        st.append_audit(row["id"], "implementation started", f"attempt {attempt}/{MAX_IMPLEMENTATION_ATTEMPTS}")
        activity.write(status="running", current={"id": row["id"], "name": row["name"],
                       "asset_class": row["asset_class"], "stage": "translating and verifying"}, out=ACTIVITY_OUT)
        to_dashboard(st)
        try:
            # Altering only the prompt cache key permits a controlled rewrite;
            # the stored Pine itself remains untouched and auditable.
            revision_source = source + f"\n// engine implementation revision {attempt}\n"
            code = translator.translate(
                name=row["name"], source=revision_source,
                author=row.get("author") or "",
                description=(row.get("description") or "") +
                "\nImplementation repair attempt. Previous verifier note: " + (row.get("note") or "")[:400])
            df = _load_bars(cfg["symbols"][0], cfg)
            checked = verify(code, df)
        except Exception as exc:  # operational failure; do not claim a strategy result
            st.update_result(row["id"], status="blocked", verdict="blocked", implementation_attempts=attempt,
                             note=f"implementation attempt {attempt}/{MAX_IMPLEMENTATION_ATTEMPTS} failed: {type(exc).__name__}")
            st.append_audit(row["id"], "implementation failed", type(exc).__name__)
            to_dashboard(st)
            return {"status": "retry", "id": row["id"], "reason": type(exc).__name__}
        if checked.ok:
            # Make the repaired code the canonical cache entry the test worker
            # will read next time, then return it to its normal queue.
            translator.cache_code(name=row["name"], source=source, code=code)
            st.update_result(row["id"], status="harvested", verdict="pending",
                             attempts=0, implementation_attempts=attempt,
                             score=None,
                             note="implementation verified — queued for independent backtest; no performance result yet")
            st.append_audit(row["id"], "implementation verified", "returned to research queue")
            activity.write(status="ready", current={"id": row["id"], "name": row["name"],
                           "asset_class": row["asset_class"], "stage": "queued for test"}, out=ACTIVITY_OUT)
            to_dashboard(st)
            return {"status": "ready", "id": row["id"], "name": row["name"]}
        reason = "; ".join(checked.failures[:2])
        st.update_result(row["id"], status="blocked", verdict="blocked", implementation_attempts=attempt,
                         note=f"implementation attempt {attempt}/{MAX_IMPLEMENTATION_ATTEMPTS}: {reason}")
        st.append_audit(row["id"], "implementation verification failed", reason)
        activity.write(status="parked", current={"id": row["id"], "name": row["name"],
                       "asset_class": row["asset_class"], "stage": "needs repair"}, error=reason, out=ACTIVITY_OUT)
        to_dashboard(st)
        return {"status": "parked", "id": row["id"], "reason": reason}
    return {"status": "idle", "reason": "no repairable parked source"}
