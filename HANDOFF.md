# HANDOFF — pick up here

Last updated **2026-08-10**. Overwrite each session; this is "where were we",
not a journal. Read `CLAUDE.md` for the rules and the measured facts.
**Read `vendor/trading-bots/AI_RULES.md` before replying to Kristijonas:
short sentences, key points, no essays. A long reply is a bug.**

---

## Run it

```bash
cd ~/trading_engine
export TRADING_BOTS_DATA=~/trading-bots/scalping/data     # REQUIRED

python tests/test_engine.py      # 23/23  — budget, referee, noise rejection
python tests/test_modules.py     # 14/14  — feeds, meta-labeling, regime
PASS_LIMIT=6 python -m engine.runner        # one research pass
open dashboard/index.html                   # the console
```

Fresh clone: `git clone --recurse-submodules …`, then set
`TRADING_BOTS_DATA`. The submodule carries the loaders but NOT the 572 MB
price cache, and without that var every fetch silently re-downloads years of
history into a directory nothing reads.

---

## True state — no rounding up

| | count |
|---|---|
| harvested from TradingView | **40** |
| Pine source stored | **1** |
| fully tested end to end | **1** |
| promoted | **0** |
| trial budget spent | **1** of 47 (Crypto) |

**Nothing works yet.** One strategy has been measured and it was breakeven.
That is the honest position and the dashboard shows it.

### The one real result

SuperTrend STRATEGY (KivancOzbilgic, 24,654 likes — most popular in the
corpus). Harvested → Pine translated → verified → backtested on our data,
taker fees, `STOP_FILL=close`:

| symbol | n | PF | win | total R |
|---|---|---|---|---|
| BTCUSDT | 155 | 0.917 | 43.9% | −5.8 |
| ETHUSDT | 153 | 0.928 | 42.5% | −5.2 |
| SOLUSDT | 153 | 1.182 | 47.1% | +11.5 |
| **pooled** | **461** | **1.003** | **44.5%** | **+0.5** |

Only SOL carries it — the per-coin-vs-pooled split HARD RULE 3 exists to
expose. DSR 0.20, score 1, verdict fail.

---

## Do this next, in order

**1. Pull the remaining Pine sources.** 39 of 40 candidates are parked purely
because `state/pine/<id>.pine` does not exist. The TradingView MCP corpus holds
790 scripts / 566 sources; `mcp__tradingview__query_corpus` with
`include_source=true` returns them 3 at a time. Write them to
`state/pine/{id with : ; / replaced by _}.pine` and the runner picks them up
with no code change. **This is the single highest-value next action** — it
takes the engine from 1 tested to ~40.

**2. The scheduler.** A systemd timer running `python -m engine.runner` every
N minutes. Deliberately a timer, not a daemon: a crashed long-lived process is
invisible, a failing timer is a log line. ~30 minutes of work.

**3. Deploy.** Oracle ARM free tier (4 cores / 24 GB), Kristijonas is
provisioning. **Catch:** the translator uses headless `claude -p` on his
subscription, so the box needs `claude` installed AND interactively
authenticated once. If every translation starts failing with a non-zero exit,
check auth expiry first.

**4. Quantpedia as a second source.** Same `ingest_records()` shape as
TradingView — it takes a list of dicts and does not care where they came from.

---

## Design decisions that are not up for casual revision

- **No translator means NO TEST.** Measured this session: falling back to a
  shared placeholder gave "SuperTrend PF 1.069" and "MACD+SMA 200 PF 1.069" —
  the same number, because both rows were an SMA cross wearing someone else's
  name. Those four results were rolled back and the ledger cleared. A parked
  candidate is correct; a fabricated one is not.
- **Harvested rows carry NO performance numbers.** TradingView advertises 90%
  win rates curve-fitted to one symbol on one window. Metrics stay `None` until
  measured here, and the UI renders `—`, never `0`.
- **Popularity orders the work queue and nothing else.** A widely-copied edge
  is a crowded one.
- **Budget exhaustion stops the engine and says so** — Kristijonas' explicit
  choice. It does not switch universes to stay busy.
- **A failed translation is not a refuted idea.** It spends no budget and is
  recorded as a translator failure, so it can be retried after a fix.

---

## Traps already paid for

1. **Do not write a backtester here.** `vendor/trading-bots/scalping/
   backtest.py` carries 24 hard rules learned by being fooled. Two engines
   would mean two rulers.
2. **Breadth is √N_eff, not linear.** Linear crediting handed multi-asset an
   allowance of 99 million trials.
3. **Allowance capped at 1000/universe.** MinBTL inverts exponentially
   (N ~ e^(T/2)); 98 years of S&P otherwise buys infinite permission and the
   guard silently stops guarding.
4. **Non-crypto data is DAILY ONLY.** Yahoo caps intraday at ~730 days.
5. **COT merges on `released`, never `report_date`.**
6. **`.astype("int64") // 10**6` on a datetime is a resolution ASSUMPTION** —
   silently yields seconds under pandas 3. Use `fetch_cme_basis.to_ms()`.
7. **FRED and CFTC need OPPOSITE HTTP headers.** cftc.gov 403s without a
   browser User-Agent; fred.stlouisfed.org hangs to timeout *with* one.
8. **A text feature needs TWO lag guards** — `searchsorted(side="right")` plus
   a further `.shift(1)`.
9. **Never plain `KFold` on trade labels.** Overlapping windows leak; use
   `PurgedSplit`.
10. **Never fit the regime model to forward returns.**
11. **Look-ahead detection must be BEHAVIOURAL, not textual.** Proven this
    session: a leak through a whole-series `c.max()` — no negative shift, no
    `center=True` — was invisible to the regex scan and caught only by
    re-running on a truncated frame (1192 vs 3488 signals over identical bars).

---

## Open elsewhere

- `trading-bots`: stale worktree at `.claude/worktrees/n1-deep-test` (18 MB).
  Removal was blocked by a sandbox classifier; Kristijonas has the command.
- `bot-n5funded` went `MAX_OPEN` 2 → 4 on 2026-08-10 (REAL MONEY).
  **Re-measure trades/day around 2026-08-17.** Backtest says it should roughly
  double; the DVOL gate being off may mask it.
- `bot-n5` has NO concurrency cap — `MAX_OPEN` does not exist in `bot_n5.py`.
  It ran 3 positions / $22.9k notional against $12.7k equity. Nobody has
  decided whether that is intended.
