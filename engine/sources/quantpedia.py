"""Import Quantpedia records from a user-provided CSV/JSON export.

Quantpedia content is licensed. This module deliberately does not scrape the
site: place an export you are allowed to use on disk, then call ``import_file``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..harvest import ingest_records


def import_file(path: str | Path, store=None) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text())
    else:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    records = []
    for r in rows:
        name = (r.get("name") or r.get("strategy") or r.get("title") or "").strip()
        if not name:
            continue
        records.append({"id": r.get("id") or name, "name": name,
                        "author": r.get("author") or "Quantpedia",
                        "url": r.get("url"),
                        "description": r.get("description") or r.get("summary"),
                        "symbol": r.get("symbol"), "interval": r.get("timeframe"),
                        "likes": 0, "has_source": bool(r.get("source"))})
    return ingest_records(records, source="Quantpedia", store=store)
