"""Deliver the results. A research engine nobody hears from is not running.

The loop already stores everything it measures — SQLite for the rows,
`ledger.json` for the trials, `dashboard/` for the console. That is storage, not
delivery: all three require somebody to go and look, and the whole point of a
timer-driven engine is that nobody is watching it at 04:00.

So this module is the OUTPUT side, and it has exactly two channels:

  RESULTS.md   the full standing picture, rewritten at the end of every pass.
               One file, always current, readable in a terminal. It is
               regenerated from the store rather than appended to, so it can
               never drift from what the database actually says.

  ntfy         a PUSH, for the three things worth interrupting somebody about:
               a promotion, a universe running out of trial budget, and the
               engine stalling. Notifications are deduplicated through
               `state/notified.json`, because a timer that fires 72 times a day
               will happily send the same alert 72 times a day, and an alert
               channel that cries wolf gets muted — after which the ONE message
               that mattered is also lost.

WHAT IS DELIBERATELY *NOT* PUSHED
---------------------------------
Individual failures. Most strategies fail; that is the expected outcome and the
reason the engine exists. Pushing every `fail` verdict would be pushing noise,
and would train exactly the muting described above. Failures land in RESULTS.md,
which is where you go when you want to read, not be told.

DAYS TO +10% IS IN EVERY TABLE
------------------------------
Kristijonas' standing requirement, alongside trades/day: a strategy's job is to
clear a prop firm's profit target FAST. PF alone hides that — a PF of 1.4 at
0.01 trades/day takes years to make 10%, and is useless for a challenge no
matter how good the ratio looks. The column is computed at 1% risk per trade
from the same pooled R the verdict used.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

STATE = Path(__file__).resolve().parent.parent / "state"
RESULTS = Path(__file__).resolve().parent.parent / "RESULTS.md"
NOTIFIED = STATE / "notified.json"

# Kristijonas' existing phone topic — the same one the live bots push to. Reused
# on purpose: one channel he already watches beats a second one he has to
# remember to subscribe to.
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "kris-bots-d940e9f3814b")
NTFY_URL = "https://ntfy.sh/{topic}"

DIGEST_EVERY = timedelta(hours=22)      # at most one "still alive" push per day


# --------------------------------------------------------------------- helpers
def days_to_10pct(cagr: float | None) -> float | None:
    """Calendar days to +10% at 1% risk per trade, from the measured R rate.

    `cagr` as the runner stores it is already "fraction of equity per year at 1%
    risk", so this is a unit conversion, not a new estimate. Returns None for a
    losing strategy rather than a negative number: "-412 days" reads like a
    result, and it is not one.
    """
    if not cagr or cagr <= 0:
        return None
    return round(0.10 / cagr * 365.25, 1)


def _fmt(v, nd=3, dash="—"):
    return dash if v is None else f"{v:.{nd}f}"


def _load_notified() -> dict:
    try:
        return json.loads(NOTIFIED.read_text())
    except Exception:                                            # noqa: BLE001
        return {}


def _save_notified(d: dict) -> None:
    NOTIFIED.parent.mkdir(parents=True, exist_ok=True)
    NOTIFIED.write_text(json.dumps(d, indent=1))


# ----------------------------------------------------------------------- ntfy
def _ascii(s: str) -> str:
    """Fold a string down to something an HTTP HEADER can carry.

    Headers are latin-1 in http.client, and this codebase's prose is full of em
    dashes and arrows. A single "—" in a Title raises UnicodeEncodeError INSIDE
    the try below, which `push` then reports as an ordinary network failure —
    so the notification silently never arrives and nothing says why. Measured
    2026-08-11: the daily digest failed exactly this way while a plain-ASCII
    test push to the same topic returned HTTP 200.
    """
    return (s.replace("—", "-").replace("–", "-").replace("→", "->")
             .replace("≥", ">=").replace("’", "'")
             .encode("ascii", "replace").decode("ascii"))


def push(title: str, body: str, priority: str = "default",
         tags: str = "microscope") -> bool:
    """Fire-and-forget notification. NEVER raises.

    A failed push must not fail a research pass: the measurement is the valuable
    part and it is already committed to the store by the time this is called.
    Network errors are swallowed and reported as False.
    """
    try:
        req = urllib.request.Request(
            NTFY_URL.format(topic=NTFY_TOPIC),
            data=body.encode("utf-8"),
            headers={"Title": _ascii(title), "Priority": priority,
                     "Tags": _ascii(tags)},
            method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return 200 <= r.status < 300
    except Exception as e:                                       # noqa: BLE001
        # Loud in the journal, harmless to the pass. A push that fails silently
        # is indistinguishable from a push that was never due, and that is how
        # an alerting channel dies without anybody noticing.
        print(f"[report] push failed: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def notify(store, ledger, res, now: datetime | None = None) -> list[str]:
    """Decide what — if anything — is worth waking somebody for. Returns what was sent.

    Every branch is deduplicated against `state/notified.json` by a STABLE key,
    so the same promotion, the same exhausted universe and the same stall are
    announced exactly once each.
    """
    now = now or datetime.now(timezone.utc)
    seen = _load_notified()
    sent = []

    # ---- 1. a promotion. The only genuinely good news this engine can produce,
    # and it has never fired yet.
    for r in store.all():
        if r["status"] != "promoted":
            continue
        key = f"promoted:{r['id']}"
        if key in seen:
            continue
        d10 = days_to_10pct(r["cagr"])
        if push("Strategy PROMOTED",
                f"{r['name']}\n{r['asset_class']} — PF {_fmt(r['pf'])}, "
                f"{_fmt(r['tpd'], 2)} trades/day, DSR {_fmt(r['dsr'], 2)}, "
                f"{'—' if d10 is None else f'{d10:.0f}d'} to +10%\n"
                f"score {r['score']}/10. Verify before deploying.",
                priority="high", tags="rocket"):
            seen[key] = now.isoformat(timespec="seconds")
            sent.append(key)

    # ---- 2. a universe out of trial budget. This is the engine's designed
    # stopping condition, not a fault — but it means no verdict there is valid
    # again without new data, so it must not pass unnoticed.
    for name in (res.exhausted or []):
        key = f"exhausted:{name}"
        if key in seen:
            continue
        b = ledger.budgets.get(name)
        if b and push(
                f"Trial budget spent: {name}",
                f"{b.spent:.0f} of {b.allowance} trials used on "
                f"{b.effective_years:.1f} effective years.\n"
                f"No further verdict in {name} is valid without more history "
                f"or a new data feed. The engine will keep working the other "
                f"universes.", priority="default", tags="warning"):
            seen[key] = now.isoformat(timespec="seconds")
            sent.append(key)

    # ---- 3. the engine is stalled. Distinguished from "quiet" on purpose: a
    # pass that considers candidates and tests none of them looks identical in
    # the journal to a pass with nothing to do. Three in a row means the
    # translator is down (expired `claude` auth is the usual cause) and the loop
    # is spinning without measuring anything.
    stalled = res.considered > 0 and res.tested == 0
    streak = int(seen.get("_stall_streak", 0)) + 1 if stalled else 0
    seen["_stall_streak"] = streak
    if streak == 3:
        detail = (res.errors or ["no error recorded"])[0]
        if push("Research engine stalled",
                f"3 passes in a row considered candidates and tested none.\n"
                f"Most likely the Pine translator: check `claude` auth on this "
                f"box.\nfirst error: {detail[:180]}",
                priority="high", tags="warning"):
            sent.append("stalled")

    # ---- 4. the daily digest. Proof of life plus the current best, at most
    # once every 22 hours, so a silent phone means silence and not a dead timer.
    last = seen.get("_digest")
    due = last is None or (now - datetime.fromisoformat(last)) > DIGEST_EVERY
    if due:
        rows = [r for r in store.all() if r["pf"] is not None]
        best = max(rows, key=lambda r: (r["score"] or 0, r["pf"] or 0),
                   default=None)
        counts = store.counts()
        line = ("nothing measured yet" if best is None else
                f"best: {best['name'][:44]} — PF {_fmt(best['pf'])}, "
                f"{_fmt(best['tpd'], 2)} tpd, score {best['score']}/10")
        if push("Research engine — daily",
                f"{len(rows)} strategies tested, "
                f"{counts.get('promoted', 0)} promoted, "
                f"{counts.get('harvested', 0)} queued.\n{line}\n"
                f"{budget_line(ledger)}", priority="low", tags="microscope"):
            seen["_digest"] = now.isoformat(timespec="seconds")
            sent.append("digest")

    _save_notified(seen)
    return sent


def budget_line(ledger) -> str:
    return " | ".join(f"{b.name} {b.spent:.0f}/{b.allowance}"
                      for b in ledger.budgets.values())


# ------------------------------------------------------------------ RESULTS.md
def write_results(store, ledger, res=None, path: Path | None = None) -> Path:
    """Rewrite RESULTS.md from the store. Regenerated, never appended.

    An appended log drifts from the database the moment a row is re-tested or a
    verdict changes. This file is a projection of state, so it is always exactly
    what the engine currently believes.
    """
    path = Path(path or RESULTS)
    rows = store.all()
    tested = sorted([r for r in rows if r["pf"] is not None],
                    key=lambda r: (-(r["score"] or 0), -(r["pf"] or 0)))
    counts = store.counts()
    pine = len(list((STATE / "pine").glob("*.pine")))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    L = [f"# Research results — {now}",
         "",
         "Rewritten by the engine at the end of every pass. Do not edit; edits "
         "are overwritten within 20 minutes.",
         ""]

    promoted = [r for r in tested if r["status"] == "promoted"]
    if promoted:
        L += [f"## {len(promoted)} PROMOTED — verify before deploying", ""]
    else:
        L += ["## Nothing promoted yet",
              "",
              "Every strategy measured so far has failed the gate "
              "(PF >= 1.2, DSR >= 0.95, and the universe's frequency floor). "
              "That is the expected rate; the engine exists to keep the "
              "failures cheap and honest.",
              ""]

    # ---- the leaderboard
    L += ["## Measured strategies",
          "",
          "`d→+10%` is calendar days to a 10% gain at 1% risk per trade — the "
          "prop-challenge question. `DSR` is the deflated Sharpe: the "
          "probability the edge survives having been cherry-picked out of the "
          "trials already spent in that universe.",
          "",
          "| # | strategy | universe | period tested | trades | PF | win | tpd | "
          "d→+10% | DSR | score | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(tested, 1):
        d10 = days_to_10pct(r["cagr"])
        win = "—" if r["win_rate"] is None else f"{r['win_rate'] * 100:.1f}%"
        span = ("—" if not r.get("tested_from") else
                f"{r['tested_from']} → {r['tested_to']} ({r.get('years')}y)")
        L.append(
            f"| {i} | {r['name'][:44]} | {r['asset_class']} | {span} | "
            f"{r.get('trades') or '—'} | {_fmt(r['pf'])} | {win} | "
            f"{_fmt(r['tpd'], 3)} | {'—' if d10 is None else format(d10, ',.0f')} | "
            f"{_fmt(r['dsr'], 2)} | {r['score'] or '—'}/10 | "
            f"**{r['verdict'] or 'pending'}** |")
    if not tested:
        L.append("| — | *nothing measured yet* | | | | | | | | | | |")

    # ---- the same results in words. The table answers "how did it score"; this
    # answers "what is actually wrong with it", which is the question you have
    # when deciding whether an idea is worth another look.
    scored = [r for r in tested if r.get("points")]
    if scored:
        L += ["", "## What is good and what is bad, per strategy", ""]
        for r in scored[:12]:
            L += [f"**{r['name']}** — {r['asset_class']}, {r.get('years')} years "
                  f"({r.get('tested_from')} → {r.get('tested_to')})", ""]
            L.append(f"*{r['note']}*")
            L.append("")
            for p in r["points"]:
                L.append(f"- {'✅' if p['ok'] else '❌'} {p['text']}")
            L.append("")

    # ---- budget. The number that makes every row above mean anything.
    L += ["",
          "## Trial budget",
          "",
          "Bailey/Lopez de Prado minimum backtest length. Spending trials is "
          "what makes an in-sample result meaningless, so the engine debits "
          "them and STOPS when a universe runs out — it does not switch "
          "markets to stay busy.",
          "",
          "| universe | years | N_eff | effective years | allowance | spent | "
          "left |",
          "|---|---|---|---|---|---|---|"]
    for b in ledger.budgets.values():
        L.append(f"| {b.name} | {b.years:g} | {b.n_eff:g} | "
                 f"{b.effective_years:.1f} | {b.allowance} | {b.spent:.0f} | "
                 f"{b.remaining:.0f}"
                 + (" **EXHAUSTED**" if b.exhausted else "") + " |")

    # ---- funnel
    L += ["",
          "## Pipeline",
          "",
          "| stage | count |",
          "|---|---|",
          f"| harvested (queued) | {counts.get('harvested', 0)} |",
          f"| Pine source stored | {pine} |",
          f"| tested | {counts.get('tested', 0)} |",
          f"| promoted | {counts.get('promoted', 0)} |",
          f"| rejected | {counts.get('rejected', 0)} |",
          f"| **total collected** | **{len(rows)}** |"]

    if res is not None:
        L += ["",
              "## Last pass",
              "",
              f"considered {res.considered} · translated {res.translated} · "
              f"verify-failed {res.verify_failed} · tested {res.tested} · "
              f"promoted {res.promoted} · rejected {res.rejected}"]
        if res.harvest:
            h = res.harvest
            L.append(f"harvest: +{h.get('added', 0)} new of "
                     f"{h.get('seen', 0)} seen, "
                     f"+{h.get('pine_inline', 0) + h.get('pine_fetched', 0)} "
                     f"Pine sources")
        if res.blocked:
            L.append(f"blocked {res.blocked} — no translator or no Pine "
                     f"source. Nothing measured, no budget spent.")
        for e in (res.errors or [])[:5]:
            L.append(f"- error: {e}")

    L += ["",
          "---",
          "",
          "A `fail` here is a real measurement on this repo's own data at taker "
          "fees with `STOP_FILL=close`, pooled across the universe's symbols "
          "with zero admission (trading-bots HARD RULE 3). A `rejected` row "
          "with no numbers is usually a TRANSLATOR failure, not a refuted idea "
          "— it costs no trial budget and can be retried.",
          ""]

    path.write_text("\n".join(L))
    return path
