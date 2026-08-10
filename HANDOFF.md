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
| `trading_engine` | pushed to `KristijonasSim/trading_engine` | all 5 ideas built, 37/37 tests |
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

**All five ideas are built.** 1 (referee) and 5 (the loop) are the core; 2
(feeds), 3 (meta-labeling) and 4 (regime) are plug-ins on top.

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
| `engine/feeds.py` | idea 2 — text to numeric series, correctly lagged |
| `engine/metalabel.py` | idea 3 — take/skip on existing legs, purged CV |
| `engine/regime.py` | idea 4 — multi-asset regime labels |
| `demo_fomc.py` | end-to-end on live data, spends no budget |

**The test that justifies the repo:** a 200-cell search over pure noise produces
a flattering best cell (SR/bar 0.092 — guaranteed by the maths, not bad luck)
and the pipeline rejects it at DSR 0.003, while a genuine edge under a small
search passes at 1.000. If that check ever fails, the engine is worthless no
matter what else is green.

---

## Do this next

All five ideas exist. What is missing is USE — nothing has produced a promoted
leg yet, and that is the only output that counts.

**1. Give the feed manufacturer more events.** `demo_fomc.py` runs clean and
correctly rejects: FOMC scores show unstable IC sign across folds against
forward DXY, GOLD and SP500. But 8 statements a year makes a step function with
~8 distinct values per fold — the screen is underpowered whichever way it lands.
Add ECB, BoE and BoJ statements (same `Document` shape, same fetch pattern) to
get to ~30 events/year, and re-screen. That is the cheapest real progress
available.

**2. Try a Claude scorer against the lexicon control.** `ClaudeScorer` is
written and unused — it needs `pip install anthropic` and `ANTHROPIC_API_KEY`.
The comparison is the point: if the LLM score does not beat word-counting on
the screen, it is not adding information and should not be paid for.

**3. Wire meta-labeling to a REAL leg.** `metalabel.py` is tested only on
synthetic signals. Pull actual trades from
`vendor/trading-bots/scalping/live_parity.py::build_book("n5")`, build entry-time
features, and run `cross_validate`. An AUC near 0.5 is a perfectly good finding
and should be reported, not tuned away.

**4. Feed the regime model real multi-asset data.** `build_features()` wants a
wide close-price frame; `fetch_fx` now supplies 19 non-crypto symbols. Then
`leg_fitness()` against N5's trades. Remember ADD, DON'T SUBTRACT — the output
is for SIZING, not for switching legs off.

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
8. **A text feature needs TWO lag guards, not one.** `searchsorted(side="right")`
   plus a further `.shift(1)`. The first stops a bar reading a score released
   after it; the second stops the bar CONTAINING the release from acting on it.
   Drop either and the backtest reports look-ahead as edge.
9. **Never use plain `KFold` on trade labels.** Trade windows overlap, so
   ordinary folds share information and cross-validated accuracy comes back
   beautiful and fake. `PurgedSplit` exists for this.
10. **Never fit the regime model to forward returns.** That makes it a return
    model, and it will overfit like one. Cluster observable state only; measure
    leg fitness per regime separately and out of sample.

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
