"""The single import point for everything that lives in `trading-bots`.

ONE BACKTEST ENGINE, ON PURPOSE. This repo does not own a backtester, a data
loader, or a fee model, and must never grow one. `trading-bots/scalping/
backtest.py` carries 24 hard rules that were each learned by being fooled once
-- stop pricing, limit fills, matched drawdown, the stop-width illusion. A
second engine here would mean two rulers, and every cross-result would become
incomparable.

So this module is a thin, deliberate seam: it puts trading-bots on sys.path and
re-exports what the engine needs. Everything else imports from here, so when
trading-bots moves there is exactly one path to fix.

STOP_FILL is forced to 'close' at import. backtest.py still DEFAULTS to the
optimistic 'stop' for backward compatibility, and that artifact has fooled two
separate sessions. It is not left to the caller here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# HARD RULE 1: software stops price at the breaching bar's CLOSE, never at the
# stop level. Set before backtest.py is imported, since it reads the env var
# at call time and defaults to the optimistic setting.
os.environ.setdefault("STOP_FILL", "close")

# Resolve trading-bots. Vendored submodule first, then env override, then the
# conventional sibling checkout.
_ENV = os.environ.get("TRADING_BOTS_PATH")
_HERE = Path(__file__).resolve().parent.parent
_CANDIDATES = [_HERE / "vendor" / "trading-bots",
               Path(_ENV) if _ENV else None,
               Path.home() / "trading-bots",
               _HERE.parent / "trading-bots"]

TRADING_BOTS: Path | None = None
for _c in _CANDIDATES:
    if _c and (_c / "scalping" / "backtest.py").exists():
        TRADING_BOTS = _c.resolve()
        break

if TRADING_BOTS is None:
    raise ImportError(
        "cannot find trading-bots. Set TRADING_BOTS_PATH, or clone it beside "
        "this repo. This engine deliberately has no backtester of its own -- "
        "see the module docstring."
    )

# CODE and DATA are resolved SEPARATELY, and they usually differ.
#
# The price cache is ~572 MB and gitignored, so a vendored submodule ships the
# loaders WITHOUT the bars they read. Bind DATA to the working checkout that
# actually holds the cache, or every fetch silently re-downloads years of
# history into a directory nobody else looks at.
_DATA_ENV = os.environ.get("TRADING_BOTS_DATA")
_DATA_CANDIDATES = [Path(_DATA_ENV) if _DATA_ENV else None,
                    TRADING_BOTS / "scalping" / "data",
                    Path.home() / "trading-bots" / "scalping" / "data"]
_DATA = next((p for p in _DATA_CANDIDATES if p and p.is_dir()
              and any(p.glob("*.csv"))), TRADING_BOTS / "scalping" / "data")

for _p in (str(TRADING_BOTS), str(TRADING_BOTS / "scalping")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------- re-exports
import backtest as BT                                            # noqa: E402
from fetch_data import fetch as fetch_crypto                     # noqa: E402

run_backtest = BT.run_backtest
stats = BT.stats
Trade = BT.Trade

try:
    import fetch_fx                                              # noqa: E402
    fetch_fx_bars = fetch_fx.fetch
    FX_SYMBOLS = fetch_fx.SYMBOLS
except ImportError:                                              # pragma: no cover
    fetch_fx_bars, FX_SYMBOLS = None, {}

try:
    import fetch_macro                                           # noqa: E402
    fred_frame = fetch_macro.fred_frame
    cot_frame = fetch_macro.cot_frame
except ImportError:                                              # pragma: no cover
    fred_frame = cot_frame = None

DATA = _DATA

# Fee regimes, matching trading-bots' conventions so numbers stay comparable.
TAKER = 0.00055
MAKER = 0.0002
SLIP = 0.0002


def describe() -> str:
    n = len(list(DATA.glob("*.csv"))) if DATA.exists() else 0
    warn = ""
    if n == 0:
        warn = ("\nWARNING      : data cache is EMPTY. If trading-bots is a "
                "submodule, set TRADING_BOTS_DATA to the checkout holding the "
                "cache -- otherwise every fetch re-downloads years of history.")
    return (f"code         : {TRADING_BOTS}\n"
            f"data cache   : {DATA} ({n} files)\n"
            f"STOP_FILL    : {os.environ['STOP_FILL']}\n"
            f"fx loader    : {'yes' if fetch_fx_bars else 'MISSING'}\n"
            f"macro loader : {'yes' if fred_frame else 'MISSING'}" + warn)
