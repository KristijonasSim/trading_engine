"""Long-running, low-rate implementation worker."""
from __future__ import annotations

import os
import time

from .implementation import run_once
from . import history


def main() -> int:
    interval = max(float(os.environ.get("IMPLEMENTATION_INTERVAL", "30")), 10.0)
    print(f"implementation worker started; one repair every >= {interval:g}s")
    while True:
        result = run_once()
        history.record(implemented=int(result["status"] == "ready"))
        print(f"  implementation {result['status']} | {result.get('id', result.get('reason', ''))}")
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
