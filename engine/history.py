"""Small daily operational history for the local dashboard."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "dashboard" / "history.json"

def record(**delta: int) -> None:
    try: rows = json.loads(OUT.read_text())
    except Exception: rows = []
    day = datetime.now(timezone.utc).date().isoformat()
    row = next((x for x in rows if x["day"] == day), None)
    if row is None:
        row = {"day": day, "tested": 0, "rejected": 0, "promoted": 0, "implemented": 0}
        rows.append(row)
    for key, value in delta.items():
        if key in row: row[key] += int(value)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows[-30:], indent=1))
