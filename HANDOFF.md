# HANDOFF — pick up here

Last updated **2026-08-10**. Overwrite this file each session; it is the
"where were we" note, not a journal. Read `CLAUDE.md` first — it has the rules
and the measured facts. **Read `vendor/trading-bots/AI_RULES.md` before writing
a single reply to Kristijonas: short sentences, key points, no essays.**

---

## State right now

Nothing is running. Nothing is half-finished. Both repos are pushed and clean.

| repo | commit | status |
|---|---|---|
| `trading_engine` | pushed to `KristijonasSim/trading_engine` | 3 commits, 23/23 tests |
| `trading-bots` (submodule at `vendor/`) | `786871a` | pushed |

```bash
export TRADING_BOTS_DATA=~/trading-bots/scalping/data   # REQUIRED, see below
python tests/test_engine.py                             # expect 23/23
python -c "from engine.bridge import describe; print(describe())"
```

`TRADING_BOTS_DATA` matters: the submodule ships the loaders but NOT the 572 MB
price cache, which is gitignored. Without it every fetch silently re-downloads
years of history into a directory nothing else reads.

---

## What was built, and why it is shaped this way

Ideas **1 (referee)** and **5 (the loop)** of a five-part plan. The other three
are plug-ins and the engine runs without them.

The context that explains every design choice: **the predecessor engine ran
94,658 backtests and shipped 0 legs.** That is not bad execution — on 5.1 years
of data the honest budget is ~48 independent trials, total, ever. So this engine
exists to make each test **expensive, counted and justified**, not to run more
of them faster. A faster engine without a budget is a worse engine.

| file | job |
|---|---|
| `engine/budget.py` | MinBTL, expected-max-Sharpe, Deflated Sharpe, persistent ledger |
| `engine/hypothesis.py` | the gate — mechanism, unexploited feed, pre-stated prediction, dupe check |
| `engine/pipeline.py` | `screen` (free) → `evaluate` (spends budget, judges the MEDIAN cell) |
| `engine/bridge.py` | the only seam to trading-bots; forces `STOP_FILL=close` |

**The test that justifies the repo:** a 200-cell search over pure noise produces
a flattering best cell (SR/bar 0.092 — guaranteed by the maths, not bad luck)
and the pipeline rejects it at DSR 0.003, while a genuine edge under a small
search passes at 1.000. If that check ever fails, the engine is worthless no
matter what else is green.

---

## Do this next: idea 2, the feed manufacturer

**Why this one and not the others:** it is the only remaining idea that CREATES
information. Every real leg in trading-bots came from a new data feed; external
idea hunts over existing data are **0-for-351**. Ideas 3 and 4 reprocess what is
already there, so they cannot lift the ceiling — only the noise floor.

The shape: an LLM turns unstructured text into a numeric series that can be
screened like any other feature.

- FOMC statements / minutes → hawkish-dovish score
- ECB, BoJ, BoE statements → same, per currency
- Exchange announcements (listings, delistings, margin changes) → event flags
- Earnings-call tone for index constituents

Then: register it as a hypothesis, `screen()` it for free against forward
returns, and only backtest if the IC is stable across folds.

**Constraints that will bite:**
- The VM is 952 MB total. A local model needs ~4.5 GB. If an LLM is in the
  loop it calls the Claude API — zero RAM, pay per call.
- Generating more *hypotheses* faster is NOT the constraint; the trial budget
  is. An LLM proposing 1,000 ideas a day makes the arithmetic worse unless each
  idea arrives with new data attached. That is the whole point of this one.

The other two, whenever:
- **Idea 3, meta-labeling** — existing legs pick direction, ML only decides
  take/skip and size. `n5_metalabel.py` is in trading-bots git history.
- **Idea 4, regime classifier** — decides which legs run now. Would replace the
  DVOL gate, which has been OFF for 100% of 132 checks since deploy.

---

## Traps already paid for — do not rediscover these

1. **Do not write a backtester here.** trading-bots' `backtest.py` carries 24
   hard rules learned by being fooled. Two engines = two rulers.
2. **Breadth is credited as √N_eff, not linearly.** Linear crediting handed the
   multi-asset universe 99 million trials, which is not a budget.
3. **The allowance is capped at 1000/universe.** MinBTL inverts exponentially
   (N ~ e^(T/2)), so 98 years of S&P bought effectively infinite permission and
   the guard silently stopped guarding. Beyond a few hundred trials the
   per-result DSR is the real control.
4. **Non-crypto data is DAILY ONLY.** Yahoo caps intraday at ~730 days. The
   live engine trades 1h/4h, so nothing here is deployable to those markets
   without a Dukascopy tick loader. Do not pretend otherwise in a result.
5. **COT merges on `released`, never `report_date`** — sampled Tuesday,
   published Friday. The naive merge leaks three days.
6. **`.astype("int64") // 10**6` on a datetime is a resolution ASSUMPTION.**
   It silently yields seconds under pandas 3. Use `fetch_cme_basis.to_ms()`.
   This has now caused two separate outages.
7. **FRED and CFTC need OPPOSITE HTTP headers.** cftc.gov 403s without a
   browser User-Agent; fred.stlouisfed.org hangs to timeout *with* one.

---

## Open, unrelated to this repo

- `trading-bots` has a stale git worktree at `.claude/worktrees/n1-deep-test`
  (18 MB, branch `worktree-n1-deep-test`). Removal was blocked by a sandbox
  classifier; Kristijonas has the command.
- `bot-n5funded` went to `MAX_OPEN=4` on 2026-08-10 (was 2, real money).
  Re-measure trades/day around 2026-08-17. Backtest says this should roughly
  double it; the DVOL gate being off may mask the change.
- `bot-n5` has NO concurrency cap at all — `MAX_OPEN` does not exist in
  `bot_n5.py`. It ran 3 positions / $22.9k notional against $12.7k equity.
  Nobody has decided whether that is intended.
