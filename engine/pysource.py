"""Python-source candidates — the no-LLM path into the pipeline.

WHY THIS EXISTS
---------------
`translate.py` pays an LLM, per script, to turn Pine into Python, and then pays
again in `verify()` to catch the bugs the LLM introduces. Measured on this repo:
~10k tokens per attempt and up to 3 attempts per script, because
`implementation.py` deliberately busts the translation cache on every repair.
That was 100% of the engine's token cost — the backtest path never called a
model at all.

A candidate whose source is ALREADY Python does not need any of that. It needs
storage, an adapter to the `signals(df)` contract, and the same verifier. Zero
tokens, zero retries, and the whole class of "the model silently changed the
logic" failures disappears — not mitigated, absent.

THE CONTRACT IS UNCHANGED
-------------------------
Whatever goes in here must end up as:

    def signals(df) -> list[tuple[int, int, float]]

(bar_index, direction, stop) with direction +1 long / -1 short and stop an
absolute price — exactly what `translate.py` produces and what
`backtest.run_backtest` consumes. One contract, so a GitHub strategy and a Pine
translation are judged by the same ruler.

VERIFY STILL RUNS. An LLM is not the only way to get look-ahead into a signal
function; `.rolling(center=True)` and `.max()` over a whole series are just as
wrong when a human on GitHub wrote them. `verify()` is kept for exactly that.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

STATE = Path(__file__).resolve().parent.parent / "state"
PYSRC = STATE / "pysource"


def _safe(cid: str) -> str:
    return cid.replace(":", "_").replace(";", "_").replace("/", "_")


def path_for(cid: str) -> Path:
    return PYSRC / f"{_safe(cid)}.py"


def store(cid: str, code: str) -> Path:
    """Persist a candidate's Python source. Returns the path written."""
    PYSRC.mkdir(parents=True, exist_ok=True)
    p = path_for(cid)
    p.write_text(code)
    return p


def source(cid: str) -> str | None:
    """The Python source for a candidate, or None if it has none."""
    p = path_for(cid)
    return p.read_text() if p.exists() else None


def has_source(cid: str) -> bool:
    return path_for(cid).exists()


def fingerprint(code: str) -> str:
    """Whitespace-insensitive digest, for duplicate detection across repos.

    The same strategy is forked and re-uploaded constantly. Testing a duplicate
    spends holdout budget on a question already answered, so dedupe before the
    queue, not after.
    """
    return hashlib.sha256("".join(code.split()).encode()).hexdigest()
