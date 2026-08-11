"""Tests for the research -> paper -> live-small -> scaled safety ladder."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.promotion import GateFailed, PromotionStore, Stage

PASS, FAIL = "  PASS", "  FAIL"
results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"{PASS if condition else FAIL}  {name}{(' -- ' + detail) if detail else ''}")


HYPOTHESIS = {
    "claim": "Extreme perpetual funding forces crowded longs to reduce exposure after settlement.",
    "mechanism": "forced_flow",
    "feed": "perp_funding_new_source",
    "prediction": "Funding above the 99th percentile precedes negative BTC returns over two days.",
    "null": "Funding has no relationship with the following two-day BTC return.",
}
VALIDATION = {
    "oos_pf": 1.31, "oos_dsr": 0.97, "oos_trades": 140,
    "walk_forward_folds": 4, "profitable_fold_fraction": 1.0,
    "stress_pf": 1.11, "parameter_sensitivity": 0.11,
}
PAPER = {"trades": 34, "execution_cost_ratio": 1.06, "drawdown_r": 4.5}
LIVE = {"trades": 31, "execution_cost_ratio": 1.12, "drawdown_r": 5.0,
        "max_pairwise_correlation": 0.42}


def store():
    tmp = tempfile.TemporaryDirectory()
    st = PromotionStore(Path(tmp.name) / "promotion.json")
    st.register(id="funding-1", name="Funding unwind", universe="Crypto",
                hypothesis=HYPOTHESIS)
    return tmp, st


def test_locked_hypothesis_required():
    with tempfile.TemporaryDirectory() as d:
        st = PromotionStore(Path(d) / "p.json")
        try:
            st.register(id="bad", name="bad", universe="Crypto", hypothesis={})
            check("missing hypothesis is refused", False)
        except GateFailed:
            check("missing hypothesis is refused", True)


def test_no_backtest_to_live_bypass():
    tmp, st = store()
    try:
        try:
            st.advance("funding-1")
            check("cannot advance without OOS evidence", False)
        except GateFailed:
            check("cannot advance without OOS evidence", True)
        check("candidate remains research", st.items["funding-1"].stage == Stage.RESEARCH)
    finally:
        tmp.cleanup()


def test_promotion_ladder_and_manifest():
    tmp, st = store()
    try:
        st.attach_evidence("funding-1", "validation", VALIDATION)
        st.advance("funding-1")
        check("strong OOS evidence reaches paper", st.items["funding-1"].stage == Stage.PAPER)
        check("paper manifest cannot place live orders", not st.manifest("funding-1")["enabled"])

        st.attach_evidence("funding-1", "paper", PAPER)
        st.advance("funding-1")
        m = st.manifest("funding-1")
        check("paper parity reaches small live", st.items["funding-1"].stage == Stage.LIVE_SMALL)
        check("small live is explicitly risk-capped", m["enabled"] and m["risk_fraction"] == 0.001)

        st.attach_evidence("funding-1", "live_small", LIVE)
        st.advance("funding-1")
        check("only live evidence permits scale", st.items["funding-1"].stage == Stage.SCALED)
        check("scaled manifest remains bounded", st.manifest("funding-1")["risk_fraction"] == 0.003)
    finally:
        tmp.cleanup()


def test_bad_execution_blocks_and_pause_is_a_kill_switch():
    tmp, st = store()
    try:
        st.attach_evidence("funding-1", "validation", VALIDATION)
        st.advance("funding-1")
        st.attach_evidence("funding-1", "paper", {**PAPER, "execution_cost_ratio": 1.8})
        try:
            st.advance("funding-1")
            check("bad paper execution blocks live", False)
        except GateFailed:
            check("bad paper execution blocks live", True)
        st.pause("funding-1", "paper slippage exceeded the risk policy")
        check("pause forces zero-risk manifest", st.manifest("funding-1")["risk_fraction"] == 0.0)
        try:
            st.advance("funding-1")
            check("paused candidate cannot advance", False)
        except GateFailed:
            check("paused candidate cannot advance", True)
    finally:
        tmp.cleanup()


def test_monitor_auto_pauses_live_risk():
    tmp, st = store()
    try:
        st.attach_evidence("funding-1", "validation", VALIDATION); st.advance("funding-1")
        st.attach_evidence("funding-1", "paper", PAPER); st.advance("funding-1")
        st.monitor("funding-1", {"execution_cost_ratio": 1.7, "drawdown_r": 5.0})
        row = st.items["funding-1"]
        check("monitor automatically pauses a cost breach", row.stage == Stage.PAUSED)
        check("auto-pause history preserves its reason", row.history[-1]["event"] == "auto_paused")
        check("auto-pause removes live risk", st.manifest("funding-1")["enabled"] is False)
    finally:
        tmp.cleanup()


def test_state_survives_restart_with_history():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.json"
        st = PromotionStore(path)
        st.register(id="funding-1", name="Funding unwind", universe="Crypto", hypothesis=HYPOTHESIS)
        st.attach_evidence("funding-1", "validation", VALIDATION)
        st.advance("funding-1")
        reloaded = PromotionStore(path)
        row = reloaded.items["funding-1"]
        check("state survives restart", row.stage == Stage.PAPER)
        check("decision history survives restart", any(x["event"] == "advanced" for x in row.history))
        check("state file declares schema", json.loads(path.read_text())["schema"] == 1)


if __name__ == "__main__":
    test_locked_hypothesis_required()
    test_no_backtest_to_live_bypass()
    test_promotion_ladder_and_manifest()
    test_bad_execution_blocks_and_pause_is_a_kill_switch()
    test_monitor_auto_pauses_live_risk()
    test_state_survives_restart_with_history()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    raise SystemExit(0 if all(results) else 1)
