"""Tiny live-status projection for the static research dashboard."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent.parent / "dashboard" / "activity.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(*, status: str, current: dict | None = None, summary: dict | None = None,
          started: str | None = None, error: str | None = None,
          out: Path | None = None) -> None:
    """Atomically publish only operational state; never research metrics."""
    target = out or OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated": _now(), "status": status, "started": started,
               "current": current, "summary": summary, "error": error}
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, target)
