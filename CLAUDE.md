# trading_engine — project context

> Auto-loaded by Claude Code every session in this repo. Keep it SMALL — see
> what happened to trading-bots/CLAUDE.md (348 KB, ate 44% of the context
> window before reading the user's message).

## RULE 0 — ANSWER IN KEY POINTS

Same rule as `trading-bots`. Bullets, tables, no preamble. A long reply is a
bug. Kristijonas has asked for this 5+ times. Detail goes in code comments or
`RESEARCH_LOG.md`, never in chat. Read `../trading-bots/AI_RULES.md`.

## What this repo is

**The research engine.** It decides WHAT is worth testing and whether a result
is real. It does not trade, and it does not backtest.

**What it is NOT:** a backtester, a data loader, a fee model, or a live runner.
All of those live in `../trading-bots` and are imported through
`engine/bridge.py`. One engine, one ruler — two would make every cross-result
incomparable. If you find yourself writing a `run_backtest` here, stop.

## Why it exists

The predecessor engine ran **94,658 backtests and shipped 0 legs.** That was
not bad execution — it is the guaranteed outcome of that trial count. On 5.1
years of data the honest budget is ~48 independent trials, total, ever.

So this engine's job is the opposite of the obvious one: **not to test more
things faster, but to make each test expensive, counted, and justified.**
A faster engine without a budget is a worse engine.

## The pipeline

```
register  ->  screen (FREE)  ->  evaluate (COSTS BUDGET)  ->  verdict
```

- **register** (`hypothesis.py`) — a falsifiable claim, written before any data
  is touched: mechanism, feed, prediction, null. Rejection is free.
- **screen** (`pipeline.screen`) — does the feature carry information about
  forward returns? No configuration search, so no selection bias, so **no
  budget cost.** Screen widely; this is where most candidates should die.
- **evaluate** (`pipeline.evaluate`) — the only stage that spends budget.
  Judges the **median cell**, never the best, and deflates by every trial spent.

## Non-negotiables

1. **A hypothesis needs a MECHANISM**, from `Mechanism`: forced flow,
   structural, behavioural, information. "Momentum works" is not one. If an
   idea fits no class, that usually means no mechanism was found — only a
   pattern.
2. **A new formula on an old feed is not a new idea.** External idea hunts on
   existing data are **0-for-351** in trading-bots. Every real leg came from a
   NEW DATA FEED. `Register` refuses already-exploited feeds.
3. **Verdict on the MEDIAN cell + Deflated Sharpe.** Never a headline PF. The
   gap between the two IS the selection bias, measured — `evaluate` reports it
   as `selection_gap`.
4. **Frequency is in the gate.** `MIN_TPD = 1.0`. Every one of the old engine's
   13 "strong edge" verdicts sat at 0.05–0.15 tpd — one trade every 7–20 days,
   a shape that cannot pass a challenge regardless of edge quality.
5. **Budget exhaustion RAISES, it does not warn.** A warning gets ignored;
   that is how 94,658 happened.
6. **trading-bots' 24 HARD RULES still apply** to anything imported from there.
   `bridge.py` forces `STOP_FILL=close` at import because `backtest.py` still
   defaults to the optimistic setting.

## Measured facts — do not re-derive

- **N_eff = 2.06** across 11 crypto majors (mean pairwise corr 0.656, first PC
  = 68.9% of variance). Adding coins buys almost nothing.
- **N_eff = 6.39** for crypto + FX + indices + commodities. By Grinold
  (IR = IC·√breadth) that is **×1.76** at unchanged skill. FX alone reaches
  only 3.77 (×1.35) — the gain comes from mixing asset CLASSES.
- **Breadth is credited as √N_eff, not linearly.** Linear crediting inflated
  the multi-asset allowance to 99 million trials, which is not a budget.
- **The allowance is capped at 1000** per universe. MinBTL inverts
  exponentially (N ~ e^(T/2)), so 98 years of S&P would otherwise buy
  effectively infinite permission. Past a few hundred trials the per-result
  DSR is the real control, not the global counter.
- Daily non-crypto history: SP500 98y, DXY 56y, FRED DGS10 64y, FX 20–30y.
  **More history is worth far more than more tests** — that is the honest
  content of the exponential.

## Data

Everything comes from `../trading-bots/scalping/data` via `bridge.py`. Do not
duplicate the 572 MB cache.

| loader | covers |
|---|---|
| `fetch_data.fetch` | crypto, Binance perps, 1h/4h/30m/1d |
| `fetch_fx.fetch` | 19 FX / index / commodity / rate symbols, **daily only** |
| `fetch_macro.fred_frame` | 10 macro series, 1954+ |
| `fetch_macro.cot_frame` | CFTC positioning, weekly, 2021+ |

**Yahoo caps intraday at ~730 days**, so non-crypto research is daily-timeframe
until someone writes a Dukascopy tick loader. Do not pretend otherwise.

**COT must be merged on `released`, never `report_date`** — the report is
sampled Tuesday and published Friday. Merging on the sample date leaks three
days (trading-bots HARD RULE 7, which cost ~0.5 PF of look-ahead once already).

## Status

Built 2026-08-10: `budget.py`, `hypothesis.py`, `pipeline.py`, `bridge.py`,
`tests/test_engine.py` (23/23 passing, including "rejects pure noise").

**Not built yet** — ideas 2–4 from the plan:
- **Feed manufacturer** — LLM turning text (FOMC, filings, announcements) into
  numeric series. Highest value: it creates information rather than reshuffling
  it, and nothing in trading-bots has tried it.
- **Meta-labeling** — existing legs pick direction, ML only decides take/skip
  and size. `n5_metalabel.py` is in trading-bots git history as a start.
- **Regime classifier** — decides which legs run now. Would replace the DVOL
  gate, which has been OFF for 100% of 132 checks since deploy.
- **The autonomous loop** — build LAST. Without the budget it is just the old
  engine again.

## Testing

```
python tests/test_engine.py
```
The suite's point is one property: **fed pure noise and allowed to search hard,
the pipeline must reject.** If that check ever fails, the engine is worthless
regardless of what else passes.
