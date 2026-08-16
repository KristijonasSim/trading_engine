"""Pine -> Python. The bridge between a harvested idea and a runnable test.

`backtest.py` cannot execute Pine, so a harvested TradingView strategy is inert
until its logic exists as a Python signal function. That translation is the
bottleneck between "40 collected" and "40 tested", and it is exactly the job an
LLM should have here: MECHANICAL TRANSLATION of stated logic, not invention of
new ideas. Generating strategies faster would make the trial arithmetic worse;
translating existing ones costs no trial budget at all.

THE OUTPUT CONTRACT
-------------------
Every translation must produce:

    def signals(df) -> list[tuple[int, int, float]]

matching backtest.run_backtest's intent format: (bar_index, direction, stop),
direction +1 long / -1 short, stop an absolute price. `df` has the columns
fetch_data.fetch returns: start, open, high, low, close, volume.

WHY EVERY TRANSLATION IS DISTRUSTED BY DEFAULT
----------------------------------------------
An LLM writing code that is then judged by a backtest is a machine for producing
plausible nonsense: a subtly wrong translation does not error, it just trades
differently and reports a number. So `verify()` runs a battery BEFORE any result
is believed, and the ones that matter are the look-ahead checks:

  - determinism        same input twice, same output
  - no forward peeking  signals on df[:k] must match signals on the full frame
                        truncated to k. This catches the single most common
                        translation bug -- using a whole-series operation like
                        .max() or .rolling(center=True) where Pine only had the
                        bar's own history.
  - sane frequency      a strategy firing on 80% of bars has almost certainly
                        lost its entry condition in translation
  - bounded stops       a stop on the wrong side of entry, or 50% away, means
                        the direction or the ATR handling is inverted

A translation that fails any of these is REJECTED, never repaired silently, and
never counted against the trial budget -- it was never a test of the idea, only
of the translator.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "state" / "translations"

# Pine -> Python is mechanical: the grammar is fixed and `verify()` rejects
# anything that does not run, so Opus buys nothing here and costs ~5x. The
# translator runs on the claude.ai SUBSCRIPTION via `claude -p`, so this is a
# session-quota lever, not a billing one. Override to re-test a harder model.
#
# The cache key includes the model, so changing this re-pays every translation
# already stored under the old one.
MODEL = os.environ.get("TRANSLATOR_MODEL", "claude-sonnet-5")

@contextmanager
def _translation_lock(cache_dir: Path):
    """One Claude translation at a time across both engine workers."""
    import fcntl
    lock = cache_dir.parent / "translation.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

SYSTEM = """You translate TradingView Pine Script (versions 2 through 6) into Python signal functions.

You are TRANSLATING, not designing. Reproduce the logic that is there. Do not
add filters, do not "improve" the entry, do not substitute a better indicator.
If the Pine is ambiguous, choose the reading that trades LESS.

Output ONLY a Python module. No prose, no markdown fences, no explanation.

Required shape:

    import numpy as np
    import pandas as pd

    def signals(df):
        \"\"\"df columns: start, open, high, low, close, volume (all floats).
        Returns a list of (bar_index, direction, stop_price) tuples.
        direction: +1 long, -1 short. stop_price: absolute price level.\"\"\"
        ...
        return out

HARD CONSTRAINTS — a violation makes the translation useless:
1. Bar i may only use data from bars <= i. Never .max() over a whole column,
   never center=True, never .shift(-n), never iloc[i+1:].
2. Emit a signal on the bar where the condition CLOSES, not the bar it will be
   filled on. The backtester fills at the next bar's open itself.
3. If the Pine has a stop, translate it. If it has none, use 2.0 * ATR(14)
   beyond entry on the correct side.
4. No network, no file access, no imports beyond numpy and pandas.
5. Return an empty list rather than guessing if the logic cannot be expressed.
"""

USER = """Translate this TradingView strategy.

NAME: {name}
AUTHOR: {author}
DESCRIPTION: {description}

PINE SOURCE:
{source}
"""

# Imports a translation is allowed to make. Anything else is a rejection, not a
# warning -- generated code runs in this process.
#
# `engine` is allowed for ONE reason: `adapters.py` emits a wrapper that calls
# our own deterministic freqtrade runner. That module is ours and is in git —
# it is not generated code. The foreign strategy it carries is scanned with the
# SAME _FORBIDDEN pattern before it is ever executed (adapters.adapt), so the
# allowance widens what may be imported, not what may be run.
_ALLOWED_IMPORTS = {"numpy", "pandas", "math", "engine"}
_FORBIDDEN = re.compile(
    r"\b(open|exec|eval|compile|__import__|input|breakpoint)\s*\(|"
    r"\b(os|sys|subprocess|socket|requests|urllib|shutil|pathlib|importlib)\b")


@dataclass
class Translation:
    candidate_id: str
    code: str
    ok: bool = False
    failures: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.code.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ static gate
def static_check(code: str) -> list[str]:
    """Reject unsafe or obviously look-ahead code before running anything."""
    problems = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    problems.append(f"forbidden import: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in _ALLOWED_IMPORTS:
                problems.append(f"forbidden import: {node.module}")

    if _FORBIDDEN.search(code):
        problems.append("forbidden builtin or module reference")

    # textual look-ahead tells. Cheap, and they catch the common cases before a
    # single bar is evaluated.
    for pat, why in [
        (r"shift\(\s*-", "negative shift is look-ahead"),
        (r"center\s*=\s*True", "centred rolling window is look-ahead"),
        (r"\.iloc\[\s*i\s*\+", "indexes past the current bar"),
        (r"\[\s*i\s*\+\s*1\s*:", "slices past the current bar"),
    ]:
        if re.search(pat, code):
            problems.append(why)

    if "def signals" not in code:
        problems.append("no signals() function")
    return problems


def _load(code: str):
    ns: dict = {}
    exec(compile(code, "<translation>", "exec"), ns)      # noqa: S102
    fn = ns.get("signals")
    if not callable(fn):
        raise ValueError("signals() missing or not callable")
    return fn


# ------------------------------------------------------------------ dynamic gate
def verify(code: str, df: pd.DataFrame, *, max_signal_rate: float = 0.35,
           max_stop_frac: float = 0.5) -> Translation:
    """Run the battery. Returns a Translation with ok=False and reasons if bad."""
    t = Translation(candidate_id="", code=code)
    t.failures = static_check(code)
    if t.failures:
        return t

    try:
        fn = _load(code)
    except Exception as e:                                       # noqa: BLE001
        t.failures.append(f"import/exec failed: {type(e).__name__}: {e}")
        return t

    try:
        out = fn(df.copy())
    except Exception as e:                                       # noqa: BLE001
        t.failures.append(f"raised on real data: {type(e).__name__}: {e}")
        return t

    if not isinstance(out, list):
        t.failures.append(f"returned {type(out).__name__}, expected list")
        return t
    for s in out[:200]:
        if not (isinstance(s, tuple) and len(s) == 3):
            t.failures.append("signal is not a 3-tuple (idx, direction, stop)")
            return t
        if s[1] not in (1, -1):
            t.failures.append(f"direction {s[1]!r} is not +1/-1")
            return t

    n = len(out)
    t.stats["n_signals"] = n
    t.stats["signal_rate"] = n / max(len(df), 1)

    if n == 0:
        # HARD RULE 18: zero signals is BROKEN, not quiet.
        t.failures.append("zero signals over full history — broken, not quiet")
        return t
    if t.stats["signal_rate"] > max_signal_rate:
        t.failures.append(
            f"fires on {t.stats['signal_rate']:.0%} of bars — entry condition "
            f"probably lost in translation")
        return t

    # determinism
    if fn(df.copy()) != out:
        t.failures.append("non-deterministic: same input gave different signals")
        return t

    # look-ahead: signals on a truncated frame must match the full-frame signals
    # truncated to the same point. This is the check that actually matters.
    k = int(len(df) * 0.6)
    try:
        early = fn(df.iloc[:k].copy())
    except Exception as e:                                       # noqa: BLE001
        t.failures.append(f"raised on truncated frame: {type(e).__name__}: {e}")
        return t
    expected = [s for s in out if s[0] < k]
    # allow the last few bars to differ: indicators legitimately need lookback
    # to settle at a boundary, but anything earlier is real leakage
    margin = 50
    a = [s for s in early if s[0] < k - margin]
    b = [s for s in expected if s[0] < k - margin]
    if a != b:
        t.failures.append(
            f"LOOK-AHEAD: truncated frame gave {len(a)} signals, full frame "
            f"gave {len(b)} over the same bars")
        return t

    # stop sanity
    bad_stops = 0
    close = df["close"].values
    for idx, direction, stop in out:
        if idx >= len(close) or not np.isfinite(stop):
            bad_stops += 1
            continue
        px = close[idx]
        if px <= 0:
            continue
        if direction == 1 and stop >= px:
            bad_stops += 1
        elif direction == -1 and stop <= px:
            bad_stops += 1
        elif abs(stop - px) / px > max_stop_frac:
            bad_stops += 1
    t.stats["bad_stops"] = bad_stops
    if bad_stops > n * 0.05:
        t.failures.append(
            f"{bad_stops}/{n} stops are on the wrong side of entry or absurdly "
            f"wide — direction or ATR handling is inverted")
        return t

    t.ok = True
    return t


# ------------------------------------------------------------------ the LLM call
class Translator:
    """Claude-backed Pine translator with an on-disk cache.

    Cached by (model, prompt-digest) so a re-run never pays twice, and so a
    translation is reproducible: the same Pine always yields the same Python
    unless the model or the prompt changes.
    """

    def __init__(self, model: str = MODEL, cache_dir: Path | None = None):
        self.model = model
        self.cache_dir = Path(cache_dir or CACHE)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    "pip install anthropic to translate Pine") from e
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic()
        return self._client

    def _key(self, name: str, source: str) -> str:
        h = hashlib.sha256(f"{self.model}|{SYSTEM}|{name}|{source}".encode())
        return h.hexdigest()[:24]

    def cached(self, *, name: str, source: str) -> bool:
        """Whether this exact source already has a saved translation.

        The runner uses this only for queue priority: cached code still goes
        through the full verifier and backtest, but does not need a new
        implementation attempt first.
        """
        return (self.cache_dir / f"{self._key(name, source)}.py").exists()

    def cache_code(self, *, name: str, source: str, code: str) -> None:
        """Store a verifier-approved implementation under its canonical key."""
        (self.cache_dir / f"{self._key(name, source)}.py").write_text(code)

    def translate(self, *, name: str, source: str, author: str = "",
                  description: str = "") -> str:
        path = self.cache_dir / f"{self._key(name, source)}.py"
        if path.exists():
            return path.read_text()

        with _translation_lock(self.cache_dir):
            if path.exists():
                return path.read_text()
            client = self._client_or_raise()
            msg = client.messages.create(
                model=self.model, max_tokens=4000, temperature=0,
                system=SYSTEM,
                messages=[{"role": "user", "content": USER.format(
                    name=name, author=author, description=description or "—",
                    source=source[:24000])}])
            code = msg.content[0].text.strip()
            code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code).strip()
            path.write_text(code)
            return code


class ClaudeCLITranslator(Translator):
    """Translate via headless Claude Code (`claude -p`) instead of the API.

    Why this exists: a claude.ai subscription and API credits are billed
    SEPARATELY, and Kristijonas already pays for the former. Headless Claude
    Code authenticates against the subscription, so this path costs nothing
    extra where the API path would be ~1-2 cents per strategy.

    Trade-offs, both real:
      - subscription rate limits apply, so a 790-script run must be paced
        rather than fired off in parallel
      - it is slower per call than the API, since a full CLI session starts up

    Deployment note: this needs the `claude` binary installed AND authenticated
    on whatever machine runs the loop. On a headless server that is a one-time
    interactive login; if the engine ever reports every translation failing with
    a non-zero exit, expired auth is the first thing to check.
    """

    def __init__(self, model: str = MODEL, cache_dir: Path | None = None,
                 binary: str = "claude", timeout: int = 300):
        super().__init__(model=model, cache_dir=cache_dir)
        self.binary = binary
        self.timeout = timeout

    def _client_or_raise(self):                                  # not used here
        return None

    def translate(self, *, name: str, source: str, author: str = "",
                  description: str = "") -> str:
        import shutil
        import subprocess

        path = self.cache_dir / f"{self._key(name, source)}.py"
        if path.exists():
            return path.read_text()
        if not shutil.which(self.binary):
            raise RuntimeError(f"{self.binary!r} not on PATH")

        prompt = SYSTEM + "\n\n" + USER.format(
            name=name, author=author, description=description or "—",
            source=source[:24000])
        # --model is passed explicitly because the cache key includes it. Left
        # off, the key claims a model the CLI may not have used, and the
        # "same Pine always yields the same Python" guarantee quietly becomes
        # "same Pine yields whatever the CLI default was that week".
        with _translation_lock(self.cache_dir):
            if path.exists():
                return path.read_text()
            proc = subprocess.run(
                [self.binary, "-p", prompt, "--model", self.model],
                capture_output=True, text=True, timeout=self.timeout)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"claude -p exited {proc.returncode}: {proc.stderr[:300]}")
            code = re.sub(r"^```(?:python)?\s*|\s*```$", "",
                          proc.stdout.strip()).strip()
            if "def signals" not in code:
                raise ValueError(f"no signals() in output: {code[:200]}")
            path.write_text(code)
            return code


def best_translator(cache_dir: Path | None = None) -> Translator | None:
    """API key if present, else headless Claude Code, else None.

    Returning None is meaningful and must not be papered over: the runner parks
    the candidate rather than testing it with placeholder logic.
    """
    import shutil
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            t = Translator(cache_dir=cache_dir)
            t._client_or_raise()
            return t
        except Exception:                                        # noqa: BLE001
            pass
    if shutil.which("claude"):
        return ClaudeCLITranslator(cache_dir=cache_dir)
    return None


def stub_translation(mechanic: str = "trend") -> str:
    """A deterministic, honest translation used when no API key is present.

    It is NOT a stand-in for a real strategy and must never be scored as one —
    it exists so the loop, the verifier and the dashboard can be exercised
    end-to-end offline. `verify()` treats it like any other candidate.
    """
    return textwrap.dedent('''
        import numpy as np
        import pandas as pd

        def signals(df):
            """Placeholder: SMA(20)/SMA(50) cross with a 2*ATR(14) stop.

            Stands in for a real translation when no API key is available.
            Every value uses only bars <= i.
            """
            c = df["close"].astype(float)
            h = df["high"].astype(float)
            l = df["low"].astype(float)
            fast = c.rolling(20).mean()
            slow = c.rolling(50).mean()
            tr = pd.concat([h - l, (h - c.shift()).abs(),
                            (l - c.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()

            up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
            dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))

            out = []
            for i in np.flatnonzero(up.fillna(False).values):
                if np.isfinite(atr.iloc[i]) and atr.iloc[i] > 0:
                    out.append((int(i), 1, float(c.iloc[i] - 2.0 * atr.iloc[i])))
            for i in np.flatnonzero(dn.fillna(False).values):
                if np.isfinite(atr.iloc[i]) and atr.iloc[i] > 0:
                    out.append((int(i), -1, float(c.iloc[i] + 2.0 * atr.iloc[i])))
            return sorted(out)
    ''').strip()
