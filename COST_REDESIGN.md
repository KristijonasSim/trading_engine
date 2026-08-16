# Engine cost redesign — 2026-08-16

Research brief: how to stop paying LLM tokens for Pine→Python, and where the
engine should point instead.

> **BUILT 2026-08-16.** Sections 1-6 are the research. What actually shipped is
> in section 0 below. The zero-token path is live and on a 30-minute timer.

## 0. What shipped

A second, parallel path into the same pipeline that costs **no tokens at all**.
The Pine translator is untouched and still works; it is simply no longer the
only way in.

| module | what it does |
|---|---|
| `engine/pysource.py` | stores/reads Python candidates. Whitespace-insensitive fingerprint so forks dedupe. |
| `engine/adapters.py` | deterministic freqtrade → `signals(df)` adapter, plus stubs for `qtpylib`, parameters, `timeframe_to_minutes`. |
| `engine/harvest_github.py` | pulls `.py` strategies from GitHub repos or a local folder. |
| `engine/stages.py` | the two-stage funnel: screen on the first 60%, holdout on the rest. |
| `engine/cycle.py` | one full pass; what the timer runs. |
| `dashboard/pipeline.html` | the funnel UI on `127.0.0.1:8777/pipeline.html`. |
| `deploy/engine-cycle.{service,timer}`, `deploy/engine-ui.service` | 24/7, `%h`-based so both PCs can use them. |

**Measured on the first real run** (13 harvested from `freqtrade/freqtrade-strategies`):

- 4 of 13 adapt and pass `verify()` cleanly — **~31% yield, zero tokens**
- the rest fail on missing stubs (`pandas_ta`, dataprovider, `@informative`) or
  are genuinely broken. They are skipped, never repaired by a model.

**The funnel doing its job**, first pass:

| strategy | screen PF | holdout PF |
|---|---|---|
| Heracles | 1.24 | **0.94** |
| MultiMa | 1.20 | **0.90** |
| DemoStrat | 1.16 | **0.87** |

Three strategies that looked like edges on the screen and died on data they had
never seen. That is the mechanism working, on real harvested code.

**One survivor so far:** `BinHV45` — screen PF 1.46 (n=95), holdout PF 3.72
(n=34). **Treat with suspicion.** n=34 barely clears the 30-trade floor, and a
PF of 3.7 on 34 trades is the rare-huge-winner shape that `CLAUDE.md` rule 10
warns is the WRONG shape for challenges. It is a lead, not a result.

### Known limits of what shipped

- **~31% adapter yield.** Raising it means more stubs (`pandas_ta`, talib,
  dataprovider), not an LLM.
- **`adapt()` executes third-party code.** It is scanned with translate's
  `_FORBIDDEN` pattern first, which stops filesystem/network/process access —
  but that is a filter, not a sandbox. Worth a container if this ever runs
  somewhere that matters.
- **Screen and holdout share one 60/40 split.** Fine for a filter; a proper
  walk-forward would rotate it.
- Survivors are marked `promising` in the store. **Nothing auto-promotes** —
  `promotion.py` is still the only path to a bot and is still manual.

## 1. The core mistake

**Pine→Python is a compiler problem, not a reasoning problem.** The grammar is
fixed and published. We are paying a frontier model per-script to do lexing and
codegen, and then paying again — in the `verify()` battery — to catch the bugs
the model introduces.

Look at what `translate.py` defends against:

| verify() check | exists because |
|---|---|
| determinism | an LLM may emit non-deterministic code |
| no forward peeking | an LLM substitutes `.max()` / `rolling(center=True)` |
| sane frequency | an LLM drops an entry condition |
| bounded stops | an LLM inverts direction or ATR handling |

**A deterministic compiler cannot make any of these four mistakes.** The entire
verification layer is insurance against a tool we chose unnecessarily. Keep it
as a safety net, but the rejection rate should collapse to near zero.

## 2. The replacement: PyneSys + PyneCore

- **PyneCore** — Apache 2.0, free, open source. A Python runtime that reproduces
  Pine's execution model (bar-by-bar, series semantics, `ta`/`math`/`array`/
  `strategy` namespaces, v5 + v6). Measured **99.56% bit-exact vs TradingView**.
- **PyneSys** — the Pine→Python compiler. Classic `lexical parser → AST →
  transform → codegen`, explicitly **not** LLM-based. Published accuracy:
  **99.69% of 28.8M compared bars bit-exact, 0 trade divergences across 106,351
  compared trades**, over 250 real TradingView scripts.

Pricing (verify at pynesys.io before committing):

| plan | $/mo | scripts/day | scripts/mo |
|---|---|---|---|
| free (Discord bot) | 0 | 3 total | — |
| Seed | 8 | 5 | 150 |
| Sprout | 20 | 30 | 900 |
| **Grower** | **28** | **100** | **3,000** |
| Forest | 45 | 300 | 9,000 |

All paid plans include API access. The PyneCore CLI caches locally — unchanged
scripts skip the API entirely, same as our current translation cache.

**Recommendation: prove it on the free Discord tier with 3 scripts we already
have LLM translations for.** Diff the two implementations against the same bars.
If PyneSys matches, buy Grower and delete the LLM translator path.

Fully-open fallbacks if we refuse to pay anything: `pine2py`, `PinePyConvert`,
`pyine`. All lower maturity and none publish bit-exactness numbers — treat as
unverified.

## 3. Local LLMs — not viable, and the wrong tool anyway

| box | RAM | GPU | verdict |
|---|---|---|---|
| VM 89.168.78.138 | **952 MB total** | none | cannot run any LLM. Already the binding constraint with 2 bots. |
| home PC | 14 GB (~7 GB free) | AMD Rembrandt iGPU, no CUDA | caps at ~7B quantized on CPU |

A 7B CPU model is *worse* at Pine semantics than Sonnet, not better. Trading
money for silently-wrong translations is the worst available outcome — it
manufactures exactly the fake results HARD RULE 16 exists to prevent. **Do not
run a local model for translation.** The deterministic compiler is both cheaper
and strictly more correct.

## 4. Free LLM APIs — use them, but not for translation

Free tiers worth routing to (limits change constantly, verify before relying):

- **Cerebras** — ~1M tokens/day, no credit card
- **Google AI Studio / Gemini** — high daily request cap, 1M context
- **Groq** — Llama 3.3 70B, very fast, low per-minute cap
- **OpenRouter** — aggregates several free models

Each provider has independent limits, so routing across them multiplies free
capacity. Point these at the *judgement* steps that survive the redesign:
ranking harvest candidates, summarising script descriptions, drafting hypothesis
text. Those are cheap, low-stakes, and tolerant of a weaker model.

## 5. The harder question: is the engine pointed at the right thing?

`trading-bots/CLAUDE.md`, MEASURED LIMITS:

> **External idea hunts are 0-for-351.** Every real leg came from a NEW DATA
> FEED, or a mechanic already proven here pointed at new data — never from
> working down a list of strategy names.

Harvesting TradingView Pine is an external idea hunt, industrialised. The prior
research engine verdicted 84 templates — 57 likely overfit, 13 strong edge, 13
edge-not-robust, 1 inconclusive — and **none reached deployment.**

Making the translation step 100x cheaper makes us fail faster, not succeed. The
cost fix is worth doing regardless, but it does not address the premise.

**Proposed repointing:** the engine's scarce resource should be spent on data it
has never seen, not on restating strategies in a new syntax. Two feeds are
already listed in CLAUDE.md as *"verified available but never turned into a
leg"*:

- DeFiLlama stablecoin supply
- CFTC COT

Both are free, both are new information rather than a new arrangement of price.
That matches the only pattern that has ever produced a live leg here.

Note also the well-documented failure mode of automated generation: generate
10,000 candidates, take the top 1%, and 100 strategies look extraordinary
in-sample even when the generator emitted pure noise. Our Bailey / López de
Prado trial budget is the right defence and must survive any redesign.

## 6. Recommended architecture

```
harvest (pure Python, no LLM)
   -> PyneSys compile  [deterministic, cached, ~$0.01/script]
   -> PyneCore run     [free, Apache 2.0]
   -> verify()         [keep as safety net; rejections should approach zero]
   -> backtest via trading-bots/scalping/backtest.py   [ONE engine, unchanged]
   -> Bailey/DSR trial budget
   -> verdict
```

LLM calls in the redesigned loop: **one per promoted candidate**, to turn a
finding into a written hypothesis. Not one per harvested script.

Where it runs: **the home PC, not the VM.** Backtests span 46k bars × 21 symbols
against a 572 MB cache — that is GB-scale pandas work and the VM has 952 MB
total. Keep the VM for execution only.

## 7. "Are there LLMs already trained to do this?"

Short answer: **no trained model, but several agent frameworks.** Nothing is a
fine-tuned "quant strategy LLM" you can download — they are all orchestration
layers that still call a general model. So none of them cut token cost by
themselves; they change *what the tokens are spent on*.

| project | license | what it does | fit for us |
|---|---|---|---|
| **RD-Agent(Q)** (Microsoft) | MIT | automates the full factor/model R&D cycle; **~2x ARR vs benchmark factor libraries with 70% fewer factors**; experiments cost "under $10" | **best fit.** LiteLLM backend → can route to DeepSeek / free tiers. Supports custom datasets, not just Qlib. Needs Docker. |
| XAlpha | research | memory-driven hypothesis→code alpha loop | same shape as RD-Agent, less mature tooling |
| TradingAgents | open source | multi-agent analysts (news, sentiment, fundamentals) | poor fit — equities/news driven, not crypto microstructure |
| AI Hedge Fund | open source | multi-agent portfolio sim, 45k stars | poor fit — same reason |

**The important distinction:** RD-Agent mines *factors from data*. Our engine
translates *strategies from scripts*. Per the 0-for-351 measured limit, factor
mining on a new feed is the pattern that has actually produced live legs here;
script translation is the pattern that has produced none.

If we spend an LLM budget at all, RD-Agent(Q) pointed at DeFiLlama stablecoin
supply or CFTC COT is a far better bet than any number of harvested Pine
scripts — and at "under $10 per experiment" it is cheaper than what the
translator currently burns.

Caveat worth carrying: the backtest-to-live gap is the standing complaint
against every one of these frameworks (50% backtested → 10-15% live after
frictions). That is precisely what our HARD RULES and `live_parity.py` already
defend against, so any framework we adopt must feed *our* backtester, not its
own.

## 8. Open items

- Engine state lives at `/home/kris/trading_engine/state/` on the *other* PC.
  This machine has no `state/`, no timer, no `RESULTS.md` — the engine is not
  running here, contrary to `trading-bots/CLAUDE.md`.
- `vendor/trading-bots` submodule is uninitialised and pinned at `786871a`,
  4 commits behind and predating the 2026-08-10 cleanup.
- Verify PyneSys pricing and the PyneCore accuracy claims independently before
  committing a subscription.

## Sources

- PyneCore — https://pynecore.org/
- PyneSys — https://pynesys.io/
- pine2py — https://github.com/xtoor/pine2py
- PinePyConvert — https://github.com/LotfiAghel/PinePyConvert
- Free LLM API tiers — https://openrouter.ai/blog/tutorials/free-llm-apis-compared/
- Local model sizing — https://www.promptquorum.com/local-llms
- Overfitting in automated generation — https://quanttradingtools.com/automated-trading-strategy-generation/
- Deflated Sharpe Ratio — https://arxiv.org/pdf/2101.07217
- RD-Agent (Microsoft, MIT) — https://github.com/microsoft/RD-Agent
- XAlpha — https://arxiv.org/pdf/2607.08332
- TradingAgents — https://github.com/tauricresearch/tradingagents
- LLM quant trading paper collection — https://github.com/Tom-roujiang/Awesome-LLM-Quantitative-Trading-Papers
