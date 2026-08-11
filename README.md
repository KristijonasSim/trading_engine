# trading_engine

The research engine for `trading-bots`. Decides what is worth testing, and
whether a result is real.

It does **not** backtest or trade — those live in `../trading-bots` and are
imported through `engine/bridge.py`. One engine, one ruler.

## Why

The predecessor ran **94,658 backtests and shipped 0 legs**. On 5.1 years of
data the honest budget is ~48 independent trials. So this engine's job is not
to test faster — it is to make each test expensive, counted, and justified.

## Pipeline

```
register  ->  screen (free)  ->  evaluate (costs budget)  ->  verdict
```

## Quick start

```bash
export TRADING_BOTS_PATH=~/trading-bots      # optional; sibling dir is found automatically
python tests/test_engine.py                  # 23 checks
python -c "from engine.bridge import describe; print(describe())"
```

## The property that matters

Fed pure noise and allowed to search 200 cells, the pipeline must reject it —
even though the best cell always looks good. That test is in
`tests/test_engine.py` and is the reason the repo exists.

See `CLAUDE.md` for the rules.

## Promotion ladder

Research results do not automatically receive live capital.  The additive
`engine.promotion` workflow records locked hypotheses and validation evidence,
then permits only `research -> paper -> live_small -> scaled`; a pause always
emits zero risk.  See [PROMOTION.md](PROMOTION.md) and run
`python -m engine.promotion --help`.
