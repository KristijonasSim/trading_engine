"""End-to-end demo: a real hypothesis, registered, screened, and judged.

Runs the whole pipeline on live data — FOMC statements scored into a feature,
screened against forward dollar and gold returns. Screening costs no trial
budget, so this is free to run as often as you like.

    python demo_fomc.py

It is a DEMO of the machinery, not a result to trade. Whatever it prints, the
honest reading is in the verdict text, not in the sign of the IC.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from engine import bridge
from engine.budget import Ledger
from engine.feeds import LexiconScorer, TextFeed, fomc_statements
from engine.hypothesis import Mechanism, Register, RejectedHypothesis
from engine.pipeline import screen

STATE = Path(__file__).resolve().parent / "state"


def main() -> int:
    print(bridge.describe(), "\n")

    # ---------------------------------------------------------- the claim
    reg = Register(STATE / "register.json")
    hid = "fomc_hawkdove_dxy"
    try:
        h = reg.register(
            id=hid,
            claim="a hawkish shift in the FOMC statement raises the dollar "
                  "over the following sessions, because a repricing of the "
                  "policy path is not fully absorbed on the release day",
            mechanism=Mechanism.INFORMATION,
            feed="fomc_statement_text",
            universe="multi",
            prediction="+0.03 or better Spearman IC against forward DXY "
                       "returns, same sign in every fold",
            null="the score carries no information about forward DXY returns")
        print(f"registered  {h.id}")
    except RejectedHypothesis as e:
        print(f"registration refused: {e}")
        return 1
    except Exception:
        h = reg.items[hid]
        print(f"already registered  {h.id}")

    led = Ledger(STATE / "ledger.json")
    led.universe("multi", years=5.1, n_eff=6.39)
    print(f"\nbudget\n{led.report()}\n")

    # ---------------------------------------------------------- the feature
    print("fetching FOMC statements...")
    docs = fomc_statements(limit=40)
    scorer = LexiconScorer()          # swap for ClaudeScorer when a key exists
    series = TextFeed("fomc_hawkdove", scorer).build(docs)
    print(f"  {len(series)} statements, "
          f"{series.index.min().date()} .. {series.index.max().date()}")
    print(f"  scorer: {scorer.name}  range [{series.min():.2f}, {series.max():.2f}]")

    # ------------------------------------------------------------- the test
    for sym in ("DXY", "GOLD", "SP500"):
        try:
            bars = bridge.fetch_fx_bars(sym, "1d")
        except Exception as e:                                   # noqa: BLE001
            print(f"\n{sym}: no data ({e})")
            continue
        idx = pd.to_datetime(bars["start"], unit="ms", utc=True)
        close = pd.Series(bars["close"].values, index=idx)

        # the stance persists between meetings, so hold the last score forward;
        # as_feature() has already applied the release lag and the extra shift
        feat = TextFeed.as_feature(series, pd.DatetimeIndex(idx), ffill_limit=90)
        fwd = close.pct_change(5).shift(-5)          # 5-day forward return

        v = screen(h, feat, fwd, folds=5)
        print(f"\n{sym}: {'PASS' if v.passed else 'FAIL'} — {v.reason}")
        if v.detail.get("ics"):
            print(f"  fold ICs: {v.detail['ics']}")

    print("\nno trial budget was spent — screening is free by construction.")
    print("Only a feature that survives the screen should ever reach evaluate().")
    print("\nCAVEAT on any result above: the feature is a STEP FUNCTION with ~40")
    print("distinct values held across ~1,400 daily bars, so a 5-fold split has")
    print("roughly 8 statements per fold, not 280. The IC magnitudes look large")
    print("for that reason and should not be read as strong either way. FOMC at")
    print("8 events/year is better used as a CONDITIONING variable than as a")
    print("standalone feature — or paired with ECB/BoE/BoJ to raise the count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
