"""Does the engine actually TELL anyone what it found?

An engine that measures correctly and reports nothing is indistinguishable from
an engine that is not running. Two failure modes are locked down here, and both
have already happened once:

  1. a push that silently never arrives. The daily digest failed on 2026-08-11
     because the Title header held an em dash, and http.client encodes headers
     as latin-1. `push()` swallows exceptions by design -- so the encoding bug
     looked exactly like an unreachable network, and the notification simply
     never existed. `_ascii()` is the fix and this pins it.

  2. an alert channel that cries wolf. The timer fires 72 times a day. Without
     deduplication, one promotion becomes 72 identical pushes, the topic gets
     muted, and the next real alert is lost with it. Every branch of `notify()`
     must fire exactly once per distinct event.

Nothing here touches the network: `push` is replaced with a recorder, which is
also the only way to assert on WHAT would have been sent.

Run:  python tests/test_report.py
"""
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import report
from engine.budget import Ledger
from engine.harvest import Candidate, CandidateStore

PASS, FAIL = "  PASS", "  FAIL"
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL}  {name}" + (f"  -- {detail}" if detail else ""))


class _Res:
    """Minimal stand-in for runner.PassResult, so tests do not import the runner
    (which imports the backtester, the data cache and pandas)."""

    def __init__(self, **kw):
        self.considered = kw.get("considered", 0)
        self.tested = kw.get("tested", 0)
        self.translated = kw.get("translated", 0)
        self.verify_failed = kw.get("verify_failed", 0)
        self.promoted = kw.get("promoted", 0)
        self.rejected = kw.get("rejected", 0)
        self.blocked = kw.get("blocked", 0)
        self.harvest = kw.get("harvest", {})
        self.exhausted = kw.get("exhausted", [])
        self.errors = kw.get("errors", [])


def _sandbox(monkey_sent):
    """Point the module's state at a temp dir and capture pushes."""
    tmp = Path(tempfile.mkdtemp())
    report.NOTIFIED = tmp / "notified.json"
    report.STATE = tmp
    (tmp / "pine").mkdir()
    report.push = lambda title, body, priority="default", tags="": (
        monkey_sent.append((title, body, priority)) or True)
    return tmp


def _store_with(rows) -> CandidateStore:
    st = CandidateStore(Path(tempfile.mkdtemp()) / "c.db")
    for r in rows:
        st.upsert(Candidate(id=r["id"], source="TradingView", name=r["name"],
                            asset_class=r.get("asset_class", "Crypto")))
        fields = {k: v for k, v in r.items()
                  if k not in ("id", "name", "asset_class")}
        if fields:
            st.update_result(r["id"], **fields)
    return st


def _ledger(**universes) -> Ledger:
    led = Ledger(Path(tempfile.mkdtemp()) / "ledger.json")
    for name, (years, n_eff) in universes.items():
        led.universe(name, years=years, n_eff=n_eff)
    return led


# --------------------------------------------------------------- days to +10%
def test_days_to_target():
    print("\ndays to +10% (the prop-challenge column)")
    check("a 10%/yr strategy needs about a year",
          364 < report.days_to_10pct(0.10) < 367,
          f"{report.days_to_10pct(0.10)} days")
    check("a faster strategy needs proportionally less",
          abs(report.days_to_10pct(0.20) - report.days_to_10pct(0.10) / 2) < 0.2)
    check("a LOSING strategy reports nothing, not a negative deadline",
          report.days_to_10pct(-0.05) is None)
    check("a flat strategy reports nothing", report.days_to_10pct(0.0) is None)


# ------------------------------------------------------------------- RESULTS.md
def test_results_md():
    print("\nRESULTS.md")
    sent = []
    tmp = _sandbox(sent)
    st = _store_with([
        dict(id="tv:1", name="Loser", asset_class="Crypto", pf=0.9, tpd=0.2,
             cagr=-0.02, win_rate=0.31, dsr=0.05, trades=120, score=1,
             verdict="fail", status="tested"),
        dict(id="tv:2", name="Winner", asset_class="Crypto", pf=1.6, tpd=2.2,
             cagr=0.55, win_rate=0.54, dsr=0.97, trades=900, score=9,
             verdict="pass", status="promoted"),
        dict(id="tv:3", name="Untouched", asset_class="FX"),
    ])
    led = _ledger(Crypto=(5.1, 2.06))
    out = report.write_results(st, led, _Res(considered=3, tested=2),
                               path=tmp / "RESULTS.md")
    txt = out.read_text()

    check("a promotion becomes the headline", "1 PROMOTED" in txt)
    check("both measured rows appear", "Winner" in txt and "Loser" in txt)
    check("the unmeasured row is NOT in the results table",
          "Untouched" not in txt.split("## Trial budget")[0])
    check("trades/day is present", "tpd" in txt)
    check("days to +10% is present for the winner", "66" in txt or "67" in txt,
          f"{report.days_to_10pct(0.55):.0f} days")
    check("a losing row shows no deadline at all",
          txt.split("| Loser |")[1].split("\n")[0].count("| — |") >= 1)
    check("sample size is shown next to PF", "| 900 |" in txt)
    check("the trial budget is reported", "Crypto | 5.1" in txt)

    # Regeneration, not appending: the file must equal the store's current view.
    st.update_result("tv:2", status="tested", verdict="fail")
    txt2 = report.write_results(st, led, None, path=tmp / "RESULTS.md").read_text()
    check("a changed verdict REPLACES the old file, never appends",
          "1 PROMOTED" not in txt2 and "Nothing promoted yet" in txt2)


# ---------------------------------------------------------------- notifications
def test_promotion_pushed_once():
    print("\nnotifications")
    sent = []
    _sandbox(sent)
    st = _store_with([dict(id="tv:2", name="Winner", asset_class="Crypto",
                           pf=1.6, tpd=2.2, cagr=0.55, dsr=0.97, trades=900,
                           score=9, verdict="pass", status="promoted")])
    led = _ledger(Crypto=(5.1, 2.06))

    first = report.notify(st, led, _Res(considered=1, tested=1))
    promo = [s for s in sent if "PROMOTED" in s[0]]
    check("a promotion is pushed", len(promo) == 1)
    check("the push carries PF and trades/day",
          "1.6" in promo[0][1] and "2.2" in promo[0][1])
    check("a promotion is high priority", promo[0][2] == "high")
    check("it is reported as sent", any(s.startswith("promoted:") for s in first))

    report.notify(st, led, _Res(considered=1, tested=1))
    report.notify(st, led, _Res(considered=1, tested=1))
    check("72 passes a day do NOT mean 72 pushes",
          len([s for s in sent if "PROMOTED" in s[0]]) == 1)


def test_exhaustion_pushed_once():
    sent = []
    _sandbox(sent)
    st = _store_with([])
    led = _ledger(Crypto=(5.1, 2.06))
    res = _Res(considered=1, tested=1, exhausted=["Crypto"])
    report.notify(st, led, res)
    report.notify(st, led, res)
    ex = [s for s in sent if "budget" in s[0].lower()]
    check("budget exhaustion is announced exactly once", len(ex) == 1)
    check("it says what unblocks it",
          "new data feed" in ex[0][1] or "more history" in ex[0][1])


def test_stall_needs_three_passes():
    sent = []
    _sandbox(sent)
    st = _store_with([])
    led = _ledger(Crypto=(5.1, 2.06))
    stalled = _Res(considered=4, tested=0, errors=["translate: exit 1"])

    report.notify(st, led, stalled)
    report.notify(st, led, stalled)
    check("one quiet pass is not an incident",
          not any("stalled" in s[0].lower() for s in sent))
    report.notify(st, led, stalled)
    check("three in a row IS an incident",
          any("stalled" in s[0].lower() for s in sent))
    report.notify(st, led, stalled)
    check("and it is not repeated every pass afterwards",
          len([s for s in sent if "stalled" in s[0].lower()]) == 1)

    # A pass that tests something clears the streak, so a later stall can alarm.
    report.notify(st, led, _Res(considered=4, tested=2))
    for _ in range(3):
        report.notify(st, led, stalled)
    check("a recovery re-arms the alarm",
          len([s for s in sent if "stalled" in s[0].lower()]) == 2)


def test_digest_is_daily_not_per_pass():
    sent = []
    _sandbox(sent)
    st = _store_with([dict(id="tv:1", name="Loser", asset_class="Crypto",
                           pf=0.9, tpd=0.2, cagr=-0.02, dsr=0.05, trades=120,
                           score=1, verdict="fail", status="tested")])
    led = _ledger(Crypto=(5.1, 2.06))
    now = datetime.now(timezone.utc)

    report.notify(st, led, _Res(considered=1, tested=1), now=now)
    report.notify(st, led, _Res(considered=1, tested=1), now=now + timedelta(hours=6))
    check("proof of life is sent once, not every 20 minutes",
          len([s for s in sent if "daily" in s[0].lower()]) == 1)
    report.notify(st, led, _Res(considered=1, tested=1), now=now + timedelta(hours=23))
    check("and again the next day",
          len([s for s in sent if "daily" in s[0].lower()]) == 2)
    check("the digest names the current best",
          "Loser" in [s for s in sent if "daily" in s[0].lower()][0][1])


def test_header_is_transport_safe():
    print("\nthe bug that made a push vanish")
    check("an em dash cannot reach an HTTP header",
          report._ascii("Research engine — daily").isascii(),
          report._ascii("Research engine — daily"))
    check("an arrow cannot either", report._ascii("d→+10%").isascii(),
          report._ascii("d→+10%"))
    check("plain text is untouched",
          report._ascii("Strategy PROMOTED") == "Strategy PROMOTED")


if __name__ == "__main__":
    test_days_to_target()
    test_results_md()
    test_promotion_pushed_once()
    test_exhaustion_pushed_once()
    test_stall_needs_three_passes()
    test_digest_is_daily_not_per_pass()
    test_header_is_transport_safe()
    n, total = sum(_results), len(_results)
    print(f"\n{n}/{total} checks passed")
    raise SystemExit(0 if n == total else 1)
