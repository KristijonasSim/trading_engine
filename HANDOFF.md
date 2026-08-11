# HANDOFF — pick up here

Last updated **2026-08-11**. Overwrite each session; this is "where were we",
not a journal. Read `CLAUDE.md` for the rules and the measured facts.
**Read `vendor/trading-bots/AI_RULES.md` before replying to Kristijonas:
short sentences, key points, no essays. A long reply is a bug.**

---

## It is running. You do not have to start it.

`engine-pass.timer` fires every 20 minutes, under systemd `--user`, with
`Linger=yes` so it survives logout and reboot. One pass takes ~4 minutes and
~13 seconds of CPU. Nothing needs a terminal open.

```bash
systemctl --user list-timers engine-pass.timer      # next firing
journalctl --user -u engine-pass.service -n 40      # what the last pass did
cat ~/trading_engine/RESULTS.md                     # every result, always current
xdg-open http://127.0.0.1:8777                      # the console
```

One pass: harvest TradingView -> fetch Pine -> translate to Python -> verify ->
backtest across the universe -> spend trial budget -> deflate -> verdict ->
rewrite RESULTS.md and the dashboard -> push anything worth an interruption.

### Where results come out

| channel | what it carries |
|---|---|
| `RESULTS.md` | the standing picture: leaderboard, trial budget, funnel. Rewritten every pass, never appended — it cannot drift from the store. |
| dashboard `:8777` | the same rows as cards, filterable. Refresh is the whole update mechanism. |
| ntfy `kris-bots-d940e9f3814b` | a PROMOTION, a universe running out of budget, the engine stalling, and one daily proof-of-life. Nothing else — most strategies fail, and pushing failures is how a channel gets muted. |

To run one pass by hand:

```bash
cd ~/trading_engine
export TRADING_BOTS_DATA=~/trading-bots/scalping/data     # REQUIRED
PASS_LIMIT=4 .venv/bin/python -m engine.runner
for t in tests/*.py; do .venv/bin/python $t | tail -1; done   # 102 checks
```

---

## True state — no rounding up

| | count |
|---|---|
| collected | **307** |
| Pine source stored | **151** |
| fully tested end to end | **17** |
| promoted | **0** |
| trial budget spent | Crypto 12/166 · FX 18/1000 · Stocks 12/1000 · Futures 2/1000 |

**Nothing has been promoted.** Seventeen measured, all failed the gate. That is
the expected rate and the engine is built to keep it cheap.

### What 17 results already say

The best row on the board is `Bjorgum Double Tap` — Crypto, PF 1.608, 60% win
rate, and **4,566 days to make 10%** at 1% risk. 25 trades in 5.1 years.

That is the whole lesson of the corpus so far, and the `d→+10%` column is what
made it visible: **published TradingView strategies fail on FREQUENCY, not on
edge.** Several have a defensible PF. Not one trades often enough to matter —
the highest is 0.254 trades/day, against the 1.0 a crypto challenge needs. This
is trading-bots HARD RULE 10 arriving from a completely independent direction.

If that holds at 50 tested, the useful conclusion is not "keep testing this
corpus" — it is that the corpus is the wrong input, and the generator should be
pointed at DATA (feeds not yet used, events with a known mechanism), which is
the only input with a non-zero hit rate in this repo's history.

---

## Do this next, in order

**1. Let it run and read RESULTS.md.** It has 151 Pine sources queued and tests
~3 per pass. The next few days of testing are already paid for. Do not add
features to an engine that has not yet been allowed to produce its evidence.

**2. Watch the frequency finding.** If the "PF is fine, tpd is hopeless" shape
holds across ~50 results, write it up in `trading-bots/RESEARCH_LOG.md` as a
measured property of the public-strategy corpus and change the input.

**3. Quantpedia as a second source.** Same `ingest_records()` shape as
TradingView — a list of dicts; it does not care where they came from. Their
strategies come with a stated mechanism, which is what this corpus lacks.

**4. Deploy to the Oracle box, maybe.** It runs fine here and the laptop is not
the constraint. **Catch:** the translator shells out to headless `claude` on
Kristijonas' subscription, so any box needs `claude` installed and
interactively authenticated once. Expired auth is the most likely cause of a
"stalled" push.

---

## Design decisions that are not up for casual revision

- **No translator means NO TEST.** A shared placeholder once gave "SuperTrend
  PF 1.069" and "MACD + SMA 200 PF 1.069" — the same number, because both rows
  were an SMA cross wearing someone else's name. A parked candidate is correct;
  a fabricated one is not.
- **Harvested rows carry NO performance numbers.** TradingView advertises 90%
  win rates curve-fitted to one symbol on one window. Metrics stay `None` until
  measured here, and the UI renders `—`, never `0`.
- **Popularity orders the work queue and nothing else.** A widely-copied edge is
  a crowded one.
- **Budget exhaustion stops the engine and says so** — Kristijonas' explicit
  choice. It does not switch universes to stay busy.
- **A failed translation is not a refuted idea.** It spends no budget and is
  recorded as a translator failure, so it can be retried after a fix.
- **Only a promotion, an exhausted budget or a stall earns a push.** Everything
  else lives in RESULTS.md. 72 passes a day makes any other rule into spam.

---

## Traps already paid for

1. **Do not write a backtester here.** `vendor/trading-bots/scalping/
   backtest.py` carries 24 hard rules learned by being fooled. Two engines would
   mean two rulers.
2. **Breadth is √N_eff, not linear.** Linear crediting handed multi-asset an
   allowance of 99 million trials.
3. **Allowance capped at 1000/universe.** MinBTL inverts exponentially
   (N ~ e^(T/2)); 98 years of S&P otherwise buys infinite permission and the
   guard silently stops guarding.
4. **Non-crypto data is DAILY ONLY.** Yahoo caps intraday at ~730 days. Daily
   universes therefore judge a MECHANIC, not a deployable challenge leg, and
   `deployable=False` says so in the row.
5. **COT merges on `released`, never `report_date`.**
6. **`.astype("int64") // 10**6` on a datetime is a resolution ASSUMPTION** —
   silently yields seconds under pandas 3. Use `fetch_cme_basis.to_ms()`.
7. **A text feature needs TWO lag guards** — `searchsorted(side="right")` plus a
   further `.shift(1)`.
8. **Never plain `KFold` on trade labels.** Overlapping windows leak; use
   `PurgedSplit`.
9. **Never fit the regime model to forward returns.**
10. **Look-ahead detection must be BEHAVIOURAL, not textual.** A leak through a
    whole-series `c.max()` — no negative shift, no `center=True` — was invisible
    to the regex scan and caught only by re-running on a truncated frame.
11. **A swallowed exception is not an absent event (2026-08-11).** `push()`
    catches everything so a dead phone cannot fail a research pass — which made
    an em dash in an HTTP header (latin-1, raises in `http.client`) look exactly
    like an unreachable network. The digest silently never sent. Headers are
    ASCII-folded and failures now print to the journal.
12. **A LIMIT applied before a filter is a hidden queue (2026-08-11).** Work
    needs Pine stored; the queue sorts by popularity; those are independent. 88
    rows had Pine and 85 fell inside the 160-row slice the runner asked for, and
    the gap grows with every harvest. The scan covers the whole store now.
13. **A retry cap must be counted only on WORKABLE rows (2026-08-11).** Counting
    attempts before the translator/Pine checks would have marked all 307
    candidates rejected after three passes with no translator — an engine fault
    convicting every idea it holds. Pinned by `tests/test_loop.py`.
14. **The installed systemd units are COPIES, not symlinks.** Editing
    `deploy/*.service` changes nothing until you
    `cp deploy/* ~/.config/systemd/user/ && systemctl --user daemon-reload`.
    They are byte-identical as of 2026-08-11; diff them before believing what is
    deployed.

---

## Open elsewhere

- `trading-bots`: stale worktree at `.claude/worktrees/n1-deep-test` (18 MB).
  Removal was blocked by a sandbox classifier; Kristijonas has the command.
- `bot-n5funded` went `MAX_OPEN` 2 → 4 on 2026-08-10 (REAL MONEY).
  **Re-measure trades/day around 2026-08-17.** Backtest says it should roughly
  double; the DVOL gate being off may mask it.
- `bot-n5` has NO concurrency cap — `MAX_OPEN` does not exist in `bot_n5.py`.
  It ran 3 positions / $22.9k notional against $12.7k equity. Nobody has decided
  whether that is intended.
