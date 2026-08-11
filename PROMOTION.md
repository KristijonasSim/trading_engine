# Promotion ladder

`engine.promotion` is the safety boundary between research and an execution
bot. It never places an order. It emits a small deployment manifest that a bot
may consume only when `enabled` is true.

```
research -> paper -> live_small -> scaled
                     -> paused
```

`research` is zero-risk. A candidate may enter `paper` only after a locked
hypothesis and validation evidence pass all fixed gates: out-of-sample PF/DSR,
minimum OOS trades, walk-forward consistency, fee/slippage stress, and parameter
sensitivity. `paper` can never enable live orders. Small live requires recorded
paper-execution parity; scaling requires recorded small-live parity and portfolio
correlation evidence. `paused` always emits zero risk.

The default state file is `state/promotion.json`; it is intentionally
machine-local and gitignored, like the trial ledger. The source-controlled code,
tests, and this document are what another PC needs. Export/copy the state file
separately only when you deliberately want to continue the same candidate state.

## Commands

Create a JSON file containing the five locked hypothesis fields:

```json
{
  "claim": "Extreme perp funding forces crowded longs to reduce exposure after settlement.",
  "mechanism": "forced_flow",
  "feed": "perp_funding_new_source",
  "prediction": "Funding above the 99th percentile precedes negative BTC returns over two days.",
  "null": "Funding has no relationship with the following two-day BTC return."
}
```

```bash
python -m engine.promotion register --id funding-1 --name "Funding unwind" \
  --universe Crypto --hypothesis /path/hypothesis.json
python -m engine.promotion evidence --id funding-1 --kind validation --file /path/validation.json
python -m engine.promotion advance --id funding-1       # research -> paper
python -m engine.promotion manifest --id funding-1
python -m engine.promotion monitor --id funding-1 --file /path/current_execution_health.json
python -m engine.promotion pause --id funding-1 --reason "execution drift"
```

Evidence is JSON. Required values are rejected rather than defaulted. The
complete policy is in `RiskPolicy` in `engine/promotion.py` and every successful
advance stores the exact check results and policy version in history.

`monitor` is the autonomous kill switch for a scheduled paper/live reconciliation.
If execution costs, drawdown, or—when scaled—portfolio correlation breach the
fixed policy, it moves the candidate to `paused` and its manifest disables live
execution. Resuming always needs an explicit investigation note.

## Current boundary

The current TradingView harvest runner remains research-only. Its results do not
automatically qualify for this ladder because they have not completed the locked
out-of-sample, stress, and paper-parity gates. Connecting a manifest to a live
bot is a separate, explicit deployment change.
