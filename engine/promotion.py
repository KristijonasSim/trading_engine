"""A conservative, durable promotion ladder for research candidates.

This module deliberately does *not* place orders.  It decides the most risk a
candidate is allowed to request from an execution system:

    research -> paper -> live_small -> scaled
                         -> paused

The record is append-only in spirit: each decision keeps the exact evidence
and policy version that made it.  This makes a second machine or a later AI
session able to resume from source-controlled code plus an exported state file,
without treating a favourable backtest as permission to trade.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


STATE = Path(__file__).resolve().parent.parent / "state" / "promotion.json"
POLICY_VERSION = "2026-08-11.1"


class Stage(str, Enum):
    RESEARCH = "research"
    PAPER = "paper"
    LIVE_SMALL = "live_small"
    SCALED = "scaled"
    PAUSED = "paused"
    REJECTED = "rejected"


class GateFailed(ValueError):
    """Raised when evidence does not justify the requested stage."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RiskPolicy:
    """All automatic decisions are made against these fixed, auditable limits."""

    version: str = POLICY_VERSION
    min_oos_pf: float = 1.20
    min_oos_dsr: float = 0.95
    min_oos_trades: int = 100
    min_walk_forward_folds: int = 3
    min_profitable_fold_fraction: float = 0.80
    min_stress_pf: float = 1.05
    max_parameter_sensitivity: float = 0.25
    min_paper_trades: int = 30
    min_live_small_trades: int = 30
    max_execution_cost_ratio: float = 1.25
    max_drawdown_r: float = 12.0
    max_pairwise_correlation: float = 0.70
    paper_risk_fraction: float = 0.0
    live_small_risk_fraction: float = 0.001
    scaled_risk_fraction: float = 0.003


@dataclass
class Candidate:
    id: str
    name: str
    universe: str
    hypothesis: dict[str, str]
    stage: Stage = Stage.RESEARCH
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Candidate":
        raw = dict(raw)
        raw["stage"] = Stage(raw["stage"])
        return cls(**raw)


class PromotionStore:
    """JSON state store with atomic writes, suitable for timer-driven use."""

    def __init__(self, path: str | Path = STATE, policy: RiskPolicy | None = None):
        self.path = Path(path)
        self.policy = policy or RiskPolicy()
        self.items: dict[str, Candidate] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self.path.read_text())
        if raw.get("schema") != 1:
            raise ValueError("unsupported promotion state schema")
        self.items = {x["id"]: Candidate.from_dict(x) for x in raw.get("candidates", [])}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps({"schema": 1, "policy": asdict(self.policy),
                           "candidates": [x.to_dict() for x in self.items.values()]},
                          indent=2, sort_keys=True) + "\n"
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(blob)
        os.replace(tmp, self.path)

    def register(self, *, id: str, name: str, universe: str,
                 hypothesis: dict[str, str]) -> Candidate:
        if id in self.items:
            raise ValueError(f"candidate already exists: {id}")
        required = {"claim", "mechanism", "feed", "prediction", "null"}
        missing = sorted(k for k in required if not hypothesis.get(k, "").strip())
        if missing:
            raise GateFailed("locked hypothesis missing: " + ", ".join(missing))
        c = Candidate(id=id, name=name, universe=universe, hypothesis=dict(hypothesis))
        c.history.append({"at": _now(), "event": "registered", "stage": c.stage.value})
        self.items[id] = c
        self.save()
        return c

    def attach_evidence(self, id: str, kind: str, evidence: dict[str, Any]) -> Candidate:
        c = self.items[id]
        if c.stage in (Stage.REJECTED,):
            raise GateFailed("rejected candidates cannot receive new evidence")
        c.evidence[kind] = dict(evidence)
        c.updated = _now()
        c.history.append({"at": c.updated, "event": "evidence", "kind": kind})
        self.save()
        return c

    def advance(self, id: str) -> Candidate:
        c = self.items[id]
        if c.stage == Stage.PAUSED:
            raise GateFailed("paused candidate needs explicit resume after investigation")
        target = {Stage.RESEARCH: Stage.PAPER, Stage.PAPER: Stage.LIVE_SMALL,
                  Stage.LIVE_SMALL: Stage.SCALED}.get(c.stage)
        if target is None:
            raise GateFailed(f"cannot advance from {c.stage.value}")
        checks = _checks(c, target, self.policy)
        failed = [x["reason"] for x in checks if not x["passed"]]
        if failed:
            raise GateFailed("; ".join(failed))
        c.stage, c.updated = target, _now()
        c.history.append({"at": c.updated, "event": "advanced", "stage": target.value,
                          "policy": self.policy.version, "checks": checks})
        self.save()
        return c

    def pause(self, id: str, reason: str) -> Candidate:
        if not reason.strip():
            raise ValueError("pause reason is required")
        c = self.items[id]
        c.stage, c.updated = Stage.PAUSED, _now()
        c.history.append({"at": c.updated, "event": "paused", "reason": reason})
        self.save()
        return c

    def monitor(self, id: str, evidence: dict[str, Any]) -> Candidate:
        """Record current execution health and automatically pause on a breach.

        A scheduler can call this after every paper/live reconciliation.  It is
        intentionally one-way: a breach can remove risk without human input,
        but re-enabling a candidate still requires ``resume(..., reason=...)``.
        """
        c = self.items[id]
        if c.stage not in (Stage.PAPER, Stage.LIVE_SMALL, Stage.SCALED):
            raise GateFailed("only paper or live candidates can be monitored")
        c.evidence["monitor"] = dict(evidence)
        checks = [_at_most(evidence, "execution_cost_ratio", self.policy.max_execution_cost_ratio,
                           "execution health"),
                  _at_most(evidence, "drawdown_r", self.policy.max_drawdown_r,
                           "execution health")]
        if c.stage == Stage.SCALED:
            checks.append(_at_most(evidence, "max_pairwise_correlation",
                                   self.policy.max_pairwise_correlation,
                                   "portfolio health"))
        c.updated = _now()
        failed = [x["reason"] for x in checks if not x["passed"]]
        if failed:
            prior, c.stage = c.stage, Stage.PAUSED
            c.history.append({"at": c.updated, "event": "auto_paused", "from": prior.value,
                              "reason": "; ".join(failed), "checks": checks})
        else:
            c.history.append({"at": c.updated, "event": "health_ok", "checks": checks})
        self.save()
        return c

    def resume(self, id: str, reason: str) -> Candidate:
        c = self.items[id]
        if c.stage != Stage.PAUSED:
            raise GateFailed("only paused candidates can resume")
        if not reason.strip():
            raise ValueError("resume investigation note is required")
        c.stage, c.updated = Stage.RESEARCH, _now()
        c.history.append({"at": c.updated, "event": "resumed", "reason": reason})
        self.save()
        return c

    def manifest(self, id: str) -> dict[str, Any]:
        c = self.items[id]
        risk = {Stage.RESEARCH: 0.0, Stage.PAPER: self.policy.paper_risk_fraction,
                Stage.LIVE_SMALL: self.policy.live_small_risk_fraction,
                Stage.SCALED: self.policy.scaled_risk_fraction,
                Stage.PAUSED: 0.0, Stage.REJECTED: 0.0}[c.stage]
        return {"schema": 1, "candidate_id": c.id, "name": c.name,
                "universe": c.universe, "stage": c.stage.value,
                "enabled": c.stage in (Stage.LIVE_SMALL, Stage.SCALED),
                "paper": c.stage == Stage.PAPER, "risk_fraction": risk,
                "policy_version": self.policy.version,
                "updated": c.updated}


def _num(d: dict[str, Any], key: str) -> float | None:
    value = d.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _at_least(d: dict[str, Any], key: str, threshold: float, label: str) -> dict[str, Any]:
    value = _num(d, key)
    return {"name": key, "passed": value is not None and value >= threshold,
            "reason": f"{label} requires {key}>={threshold:g}; got {value}"}


def _at_most(d: dict[str, Any], key: str, threshold: float, label: str) -> dict[str, Any]:
    value = _num(d, key)
    return {"name": key, "passed": value is not None and value <= threshold,
            "reason": f"{label} requires {key}<={threshold:g}; got {value}"}


def _checks(c: Candidate, target: Stage, p: RiskPolicy) -> list[dict[str, Any]]:
    if target == Stage.PAPER:
        e = c.evidence.get("validation", {})
        checks = [_at_least(e, "oos_pf", p.min_oos_pf, "paper promotion"),
                  _at_least(e, "oos_dsr", p.min_oos_dsr, "paper promotion"),
                  _at_least(e, "oos_trades", p.min_oos_trades, "paper promotion"),
                  _at_least(e, "walk_forward_folds", p.min_walk_forward_folds, "paper promotion"),
                  _at_least(e, "profitable_fold_fraction", p.min_profitable_fold_fraction, "paper promotion"),
                  _at_least(e, "stress_pf", p.min_stress_pf, "paper promotion"),
                  _at_most(e, "parameter_sensitivity", p.max_parameter_sensitivity, "paper promotion")]
        return checks
    if target == Stage.LIVE_SMALL:
        e = c.evidence.get("paper", {})
        return [_at_least(e, "trades", p.min_paper_trades, "small-live promotion"),
                _at_most(e, "execution_cost_ratio", p.max_execution_cost_ratio, "small-live promotion"),
                _at_most(e, "drawdown_r", p.max_drawdown_r, "small-live promotion")]
    if target == Stage.SCALED:
        e = c.evidence.get("live_small", {})
        return [_at_least(e, "trades", p.min_live_small_trades, "scale-up"),
                _at_most(e, "execution_cost_ratio", p.max_execution_cost_ratio, "scale-up"),
                _at_most(e, "drawdown_r", p.max_drawdown_r, "scale-up"),
                _at_most(e, "max_pairwise_correlation", p.max_pairwise_correlation, "scale-up")]
    raise ValueError(f"no checks for {target}")


def _read_json(path: str) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("JSON must be an object")
    return raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Conservative research-to-deployment promotion ladder")
    ap.add_argument("--state", default=str(STATE), help="machine-local promotion state JSON")
    sub = ap.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register", help="create a locked research candidate")
    reg.add_argument("--id", required=True); reg.add_argument("--name", required=True)
    reg.add_argument("--universe", required=True); reg.add_argument("--hypothesis", required=True)
    ev = sub.add_parser("evidence", help="attach validation, paper, or live_small evidence")
    ev.add_argument("--id", required=True); ev.add_argument("--kind", required=True,
                    choices=("validation", "paper", "live_small")); ev.add_argument("--file", required=True)
    for command in ("advance", "manifest"):
        x = sub.add_parser(command); x.add_argument("--id", required=True)
    for command in ("pause", "resume"):
        x = sub.add_parser(command); x.add_argument("--id", required=True); x.add_argument("--reason", required=True)
    monitor = sub.add_parser("monitor", help="record execution health; breaches auto-pause")
    monitor.add_argument("--id", required=True); monitor.add_argument("--file", required=True)
    sub.add_parser("status", help="show all candidates and current stage")
    args = ap.parse_args(argv)
    st = PromotionStore(args.state)
    try:
        if args.command == "register":
            c = st.register(id=args.id, name=args.name, universe=args.universe,
                            hypothesis=_read_json(args.hypothesis)); print(json.dumps(c.to_dict(), indent=2))
        elif args.command == "evidence":
            c = st.attach_evidence(args.id, args.kind, _read_json(args.file)); print(json.dumps(c.to_dict(), indent=2))
        elif args.command == "advance":
            c = st.advance(args.id); print(json.dumps(c.to_dict(), indent=2))
        elif args.command == "pause":
            print(json.dumps(st.pause(args.id, args.reason).to_dict(), indent=2))
        elif args.command == "resume":
            print(json.dumps(st.resume(args.id, args.reason).to_dict(), indent=2))
        elif args.command == "monitor":
            print(json.dumps(st.monitor(args.id, _read_json(args.file)).to_dict(), indent=2))
        elif args.command == "manifest":
            print(json.dumps(st.manifest(args.id), indent=2))
        else:
            print(json.dumps([x.to_dict() for x in st.items.values()], indent=2))
    except (KeyError, ValueError, GateFailed) as e:
        ap.error(str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
