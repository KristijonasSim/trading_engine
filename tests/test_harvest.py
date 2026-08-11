"""Does the store keep its promises when the SAME idea arrives twice?

The loop re-harvests forever, and sources carry different amounts of metadata:
the MCP corpus knows the author's chart symbol, the public search endpoint does
not. So every candidate gets written many times, by sources of differing
quality, across passes that may be months apart.

That makes upsert the sharp edge in this repo, and it drew blood on 2026-08-11:
a thin re-harvest nulled a row's symbol, reset its asset_class to 'Unknown',
and the universe assigner then filed it under Stocks -- a row whose 461
measured trades came from BTC/ETH/SOL. No error, no log line, just crypto
numbers under a stocks heading.

Two properties are locked down here:

  1. a re-harvest REFRESHES metadata but never DOWNGRADES it
  2. a TESTED row's universe is frozen, because its result was produced under
     that assignment and re-rolling it would both mislabel the result and let a
     failure be retried against a different budget

Run:  python tests/test_harvest.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.harvest import Candidate, CandidateStore, ingest_records
from engine.sources.tradingview import assign_universe, to_record

PASS, FAIL = "  PASS", "  FAIL"
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL}  {name}" + (f"  -- {detail}" if detail else ""))


def _store() -> CandidateStore:
    return CandidateStore(Path(tempfile.mkdtemp()) / "c.db")


def test_reharvest_does_not_downgrade_metadata():
    st = _store()
    rich = Candidate(id="tv:PUB;1", source="TradingView", name="Rich",
                     author="kivanc", symbol_hint="BITTREX:BTCUSDT",
                     asset_class="Crypto", popularity=24654,
                     mechanics=["trend"], description="full description")
    st.upsert(rich)

    # the thin source: same script, no symbol, no description, no likes
    thin = Candidate(id="tv:PUB;1", source="TradingView", name="Rich",
                     author=None, symbol_hint=None, asset_class="Unknown",
                     popularity=0, mechanics=[], description="")
    st.upsert(thin)

    r = st.all()[0]
    check("symbol survives a thin re-harvest", r["symbol_hint"] == "BITTREX:BTCUSDT",
          repr(r["symbol_hint"]))
    check("asset class is not reset to Unknown", r["asset_class"] == "Crypto",
          r["asset_class"])
    check("popularity never decreases", r["popularity"] == 24654, r["popularity"])
    check("mechanics are not emptied", r["mechanics"] == ["trend"], r["mechanics"])
    check("description is not blanked", r["description"] == "full description")


def test_reharvest_still_applies_real_updates():
    """The guard must not freeze the row -- better information must win."""
    st = _store()
    st.upsert(Candidate(id="tv:PUB;2", source="TradingView", name="Old",
                        popularity=10, asset_class="Unknown"))
    st.upsert(Candidate(id="tv:PUB;2", source="TradingView", name="New name",
                        popularity=99, asset_class="FX",
                        symbol_hint="FX:EURUSD", mechanics=["trend"]))
    r = st.all()[0]
    check("name updates", r["name"] == "New name", r["name"])
    check("popularity rises", r["popularity"] == 99, r["popularity"])
    check("Unknown is replaced by a real class", r["asset_class"] == "FX")
    check("a newly-known symbol is written", r["symbol_hint"] == "FX:EURUSD")


def test_measured_results_survive_reharvest():
    st = _store()
    st.upsert(Candidate(id="tv:PUB;3", source="TradingView", name="Tested",
                        asset_class="Crypto", symbol_hint="BINANCE:BTCUSDT"))
    st.update_result("tv:PUB;3", status="tested", pf=1.003, tpd=0.254,
                     verdict="fail", score=1, note="461 trades")
    st.upsert(Candidate(id="tv:PUB;3", source="TradingView", name="Tested",
                        asset_class="Unknown"))
    r = st.all()[0]
    check("status is not reset by re-harvest", r["status"] == "tested", r["status"])
    check("measured PF is untouched", r["pf"] == 1.003, r["pf"])
    check("universe of a tested row is frozen", r["asset_class"] == "Crypto",
          r["asset_class"])


def test_universe_assignment_is_deterministic():
    a = assign_universe("PUB;abc123", "Hull Suite Strategy", "//@version=5")
    b = assign_universe("PUB;abc123", "Hull Suite Strategy", "//@version=5")
    check("same id always lands in the same universe", a == b, a)
    check("assignment is a real universe", a in ("FX", "Stocks", "Futures", "Crypto"), a)

    crypto = assign_universe("PUB;zzz", "BTC Perp Funding Bot", "")
    check("a crypto-named strategy goes to Crypto", crypto == "Crypto", crypto)

    # spread: ids without a crypto tell must not all pile into one market
    got = {assign_universe(f"PUB;{i:04d}", "Generic Cross", "") for i in range(200)}
    check("non-crypto ideas spread across universes", got == {"FX", "Stocks", "Futures"},
          sorted(got))


def test_records_reject_what_cannot_be_tested():
    study = {"scriptIdPart": "PUB;a", "scriptName": "Nice Indicator",
             "access": 1, "extra": {"kind": "study"}, "agreeCount": 9000}
    closed = {"scriptIdPart": "PUB;b", "scriptName": "Secret Strategy",
              "access": 2, "extra": {"kind": "strategy"}, "agreeCount": 9000}
    good = {"scriptIdPart": "PUB;c", "scriptName": "Real Strategy",
            "access": 1, "extra": {"kind": "strategy"}, "agreeCount": 5,
            "author": {"username": "someone"}, "imageUrl": "abc"}
    check("an indicator is not harvested", to_record(study) is None)
    check("a closed-source strategy is not harvested", to_record(closed) is None)
    r = to_record(good)
    check("an open strategy is harvested", r is not None and r["name"] == "Real Strategy")
    check("no performance number is ever invented",
          r is not None and not any(k in r for k in ("pf", "tpd", "win_rate", "sharpe")))


def test_ingest_invents_no_metrics():
    st = _store()
    ingest_records([{"script_id_part": "PUB;x", "name": "X", "likes": 500,
                     "symbol": "BINANCE:BTCUSDT", "has_source": True}], store=st)
    r = st.all()[0]
    check("harvested row has no PF", r["pf"] is None)
    check("harvested row has no verdict", r["verdict"] is None)
    check("harvested row starts 'harvested'", r["status"] == "harvested")


if __name__ == "__main__":
    print("upsert does not downgrade")
    test_reharvest_does_not_downgrade_metadata()
    test_reharvest_still_applies_real_updates()
    print("\nmeasured results are immutable")
    test_measured_results_survive_reharvest()
    print("\nuniverse assignment")
    test_universe_assignment_is_deterministic()
    print("\nharvest filters")
    test_records_reject_what_cannot_be_tested()
    test_ingest_invents_no_metrics()

    n, total = sum(_results), len(_results)
    print(f"\n{n}/{total} checks passed")
    sys.exit(0 if n == total else 1)
