"""Continuous, single-instance research worker.

One candidate is processed per iteration and committed before the next begins.
The statistical budget in ``runner`` remains the non-negotiable stop condition
for valid verdicts; this worker increases throughput, not permission to search.
"""
from __future__ import annotations

import os
import time
import traceback
from datetime import datetime, timezone

from . import activity, history
from .runner import run_pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    interval = max(float(os.environ.get("WORKER_INTERVAL", "15")), 2.0)
    harvest = os.environ.get("HARVEST", "1") != "0"
    started = _now()
    total = {"considered": 0, "translated": 0, "tested": 0,
             "promoted": 0, "rejected": 0, "blocked": 0}
    current: dict | None = None
    print(f"[{started}] continuous worker started; one candidate every >= {interval:g}s")

    while True:
        try:
            activity.write(status="running", started=started,
                           summary={**total, "finished": _now()})
            def progress(item: dict) -> None:
                nonlocal current
                current = item
                activity.write(status="running", started=started, current=current,
                               summary={**total, "finished": _now()})

            result = run_pass(limit=1, harvest=harvest, progress=progress)
            for key in total:
                total[key] += getattr(result, key)
            history.record(tested=result.tested, rejected=result.rejected, promoted=result.promoted)
            if current:
                current = {**current, "stage": "last completed"}
            activity.write(status="running", started=started, current=current,
                           summary={**total, "finished": _now()})
            print(f"  tested {result.tested} | rejected {result.rejected} | "
                  f"blocked {result.blocked} | queue {result.backlog}")
            time.sleep(interval)
        except KeyboardInterrupt:
            activity.write(status="idle", started=started,
                           summary={**total, "finished": _now()})
            return 0
        except Exception as e:  # keep the worker alive; systemd retains logs
            activity.write(status="error", started=started,
                           summary={**total, "finished": _now()},
                           error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            time.sleep(max(interval, 30.0))


if __name__ == "__main__":
    raise SystemExit(main())
