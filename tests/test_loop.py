"""Can the loop run unattended for weeks without wedging?

Correctness of a single pass is covered elsewhere. This file is about the thing
that only shows up on the 500th pass: an engine that is still running, still
logging, still reporting — and no longer measuring anything.

Two ways that happens, both fixed 2026-08-11 and pinned here:

  1. STARVATION. Work needs Pine stored; the queue is ordered by popularity.
     Those are independent, so any LIMIT applied before the Pine filter hides
     ready rows sorting below it. Measured on the live store: 88 rows had Pine,
     85 fell inside the 160-row slice the runner asked for. The store grows ~40
     rows a pass, so the hidden fraction only grows.

  2. SPIN. A row whose translation raises keeps its 'harvested' status, by
     design, so a transient fault is retried instead of convicting the idea. A
     PERMANENT fault is then retried forever, at the head of the queue, and a
     handful of broken rows consume every pass.

Run:  python tests/test_loop.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import harvest, report, runner
from engine.budget import Ledger
from engine.harvest import Candidate, CandidateStore

PASS, FAIL = "  PASS", "  FAIL"
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL}  {name}" + (f"  -- {detail}" if detail else ""))


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _silence_reporting(tmp: Path) -> None:
    """Keep a test pass off network and every real generated artifact."""
    report.RESULTS = tmp / "RESULTS.md"
    report.NOTIFIED = tmp / "notified.json"
    report.STATE = tmp
    (tmp / "pine").mkdir(exist_ok=True)
    report.push = lambda *a, **k: True
    # runner imports to_dashboard directly, so report redirection alone still
    # let a fixture row overwrite dashboard/strategies.json.  The dashboard is
    # real operator state, not a test artifact.
    runner.to_dashboard = lambda store: harvest.to_dashboard(
        store, out=tmp / "strategies.json")


# ------------------------------------------------------------------ starvation
def test_queue_scans_the_whole_store():
    print("\nstarvation: a workable row must never be hidden by popularity")
    tmp = _tmp()
    runner.PINE = tmp / "pine"
    runner.PINE.mkdir()
    st = CandidateStore(tmp / "c.db")

    # 200 popular candidates with NO Pine, and one unpopular one that has it.
    for i in range(200):
        st.upsert(Candidate(id=f"tv:pop{i}", source="TradingView",
                            name=f"Popular {i}", asset_class="Crypto",
                            popularity=10_000 + i))
    st.upsert(Candidate(id="tv:workable", source="TradingView",
                        name="Unloved but testable", asset_class="Crypto",
                        popularity=1))
    (runner.PINE / "tv_workable.pine").write_text("// strategy")

    q = runner._work_queue(st, limit=4)
    check("the one testable row is found behind 200 unworkable ones",
          [r["id"] for r in q] == ["tv:workable"], f"queue={[r['id'] for r in q]}")

    check("store.queue(limit=None) returns everything",
          len(st.queue(limit=None)) == 201)
    check("an explicit limit is still honoured", len(st.queue(limit=5)) == 5)


def test_queue_prefers_deployable_diverse_mechanics_over_likes():
    """Popularity is a weak tiebreaker, not the research policy."""
    tmp = _tmp()
    runner.PINE = tmp / "pine"
    runner.PINE.mkdir()
    st = CandidateStore(tmp / "c.db")
    st.upsert(Candidate(id="tv:popular-fx", source="TradingView", name="Popular EMA",
                        asset_class="FX", popularity=100_000, mechanics=["trend"]))
    st.upsert(Candidate(id="tv:crypto-flow", source="TradingView", name="Crypto flow",
                        asset_class="Crypto", popularity=5, mechanics=["structure", "volume"]))
    for cid in ("tv:popular-fx", "tv:crypto-flow"):
        (runner.PINE / (cid.replace(":", "_") + ".pine")).write_text("// strategy")
    q = runner._work_queue(st, limit=2)
    check("deployable clear mechanic outranks popularity", q[0]["id"] == "tv:crypto-flow")


# ------------------------------------------------------------------------ spin
class _BrokenTranslator:
    """Stands in for an expired `claude` session: always raises, never returns."""

    calls = 0

    def translate(self, **kw):
        _BrokenTranslator.calls += 1
        raise RuntimeError("claude: not authenticated")


def test_a_broken_candidate_is_parked_not_retried_forever():
    print("\nspin: a permanently broken row must stop consuming passes")
    tmp = _tmp()
    _silence_reporting(tmp)
    runner.PINE = tmp / "pine"
    runner.PINE.mkdir(exist_ok=True)
    st = CandidateStore(tmp / "c.db")
    st.upsert(Candidate(id="tv:broken", source="TradingView", name="Broken",
                        asset_class="Crypto", popularity=99))
    (runner.PINE / "tv_broken.pine").write_text("// strategy")
    led = Ledger(tmp / "ledger.json")

    runner.best_translator = lambda: _BrokenTranslator()
    _BrokenTranslator.calls = 0

    seen_status = []
    for _ in range(6):
        runner.run_pass(limit=2, store=st, ledger=led, harvest=False)
        seen_status.append(st.all()[0]["status"])

    row = st.all()[0]
    check("it is retried a few times, not once",
          _BrokenTranslator.calls == runner.MAX_ATTEMPTS,
          f"{_BrokenTranslator.calls} attempts")
    check("then it is parked for implementation", row["status"] == "blocked")
    check("and the loop stops touching it",
          seen_status[-1] == "blocked" and seen_status[-2] == "blocked")
    check("no trial budget was spent on a tooling failure",
          led.budgets["Crypto"].spent == 0,
          f"spent {led.budgets['Crypto'].spent}")
    check("the note blames the tooling, not the idea",
          "TOOLING failure" in (row["note"] or ""))
    check("the note says how to un-park it",
          "attempts=0" in (row["note"] or ""))


def test_a_missing_translator_never_convicts_anything():
    print("\nan engine fault must not be recorded as a strategy result")
    tmp = _tmp()
    _silence_reporting(tmp)
    runner.PINE = tmp / "pine"
    runner.PINE.mkdir(exist_ok=True)
    st = CandidateStore(tmp / "c.db")
    st.upsert(Candidate(id="tv:waiting", source="TradingView", name="Waiting",
                        asset_class="Crypto", popularity=99))
    (runner.PINE / "tv_waiting.pine").write_text("// strategy")
    led = Ledger(tmp / "ledger.json")

    for _ in range(6):
        runner.run_pass(limit=2, store=st, ledger=led, harvest=False,
                        use_llm=False)

    row = st.all()[0]
    check("with no translator the row stays queued forever",
          row["status"] == "harvested", row["status"])
    check("and burns no attempts", (row["attempts"] or 0) == 0)
    check("and no PF is invented for it", row["pf"] is None)


def test_bars_are_recent_and_epoch_ms_is_not_1970():
    """A naked epoch-ms integer is nanoseconds to pandas: pin the real unit."""
    import pandas as pd
    original = runner.bridge.fetch_crypto
    now = pd.Timestamp("2026-08-11", tz="UTC")
    dates = pd.date_range(now - pd.Timedelta(days=365 * 7), periods=600,
                          freq="4D", tz="UTC")
    starts = [x.value // 1_000_000 for x in dates]
    src = pd.DataFrame({"start": starts, "open": 1.0, "high": 1.0,
                        "low": 1.0, "close": 1.0, "volume": 1.0})
    try:
        runner.bridge.fetch_crypto = lambda *a, **k: src
        got = runner._load_bars("BTCUSDT", runner.UNIVERSES["Crypto"])
        first = pd.to_datetime(got["start"].iloc[0], unit="ms", utc=True)
        last = pd.to_datetime(got["start"].iloc[-1], unit="ms", utc=True)
        check("timestamps retain their millisecond unit", first.year > 2020)
        check("window contains at most five years", (last - first).days <= 365 * 5 + 2)
    finally:
        runner.bridge.fetch_crypto = original


if __name__ == "__main__":
    test_queue_scans_the_whole_store()
    test_queue_prefers_deployable_diverse_mechanics_over_likes()
    test_a_broken_candidate_is_parked_not_retried_forever()
    test_a_missing_translator_never_convicts_anything()
    test_bars_are_recent_and_epoch_ms_is_not_1970()
    n, total = sum(_results), len(_results)
    print(f"\n{n}/{total} checks passed")
    raise SystemExit(0 if n == total else 1)
