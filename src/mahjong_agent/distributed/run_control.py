"""Run control request + status aggregation for distributed runs.

`run_config.json` は immutable、`control/state.json` は learner single-writer の
ままに保ちつつ、ユーザー操作（drain / stop）を **request file** 経由で渡す。

- `control/run_request.json`: ユーザー CLI が atomic write する drain/stop 要求。
  learner だけがこれを読み、安全なタイミングで `state.json` を遷移させる
  （ユーザーは state.json を直接書かない＝single-writer を維持）。
- `collect_status`: run 全体（config / state / checkpoints / shard counts /
  heartbeats / journal / request / metrics tail）を 1 dict に集約して status CLI に渡す。

hidden info は扱わない（request/status は run の制御・観測メタのみ）。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mahjong_agent.distributed.manifest import read_json, write_json_atomic
from mahjong_agent.distributed.run_state import (
    read_run_config,
    read_run_state,
    run_config_path,
    state_path,
)

RUN_REQUEST_REL = "control/run_request.json"
PROCESSED_REQUESTS_REL = "control/processed_requests"
LEARNER_JOURNAL_REL = "control/learner_transaction.json"
REQUEST_VERSION = 1

VALID_COMMANDS = ("drain", "stop", "clear")
# learner が actionable とみなす command（clear は CLI 側で file 削除する補助）。
LEARNER_COMMANDS = ("drain", "stop")

_STALE_HEARTBEAT_SEC = 120.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_ts() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# run request
# ---------------------------------------------------------------------------


def run_request_path(run_root: str | Path) -> Path:
    return Path(run_root) / RUN_REQUEST_REL


def write_run_request(
    run_root: str | Path,
    command: str,
    *,
    requested_by: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """drain/stop/clear request を atomic write する（ユーザー CLI 用）。"""
    if command not in VALID_COMMANDS:
        raise ValueError(f"invalid run command: {command!r} (valid: {VALID_COMMANDS})")
    payload = {
        "request_version": REQUEST_VERSION,
        "command": str(command),
        "requested_at": _utc_now_iso(),
        "requested_by": str(requested_by) if requested_by else _hostname(),
        "message": str(message) if message else "",
    }
    path = run_request_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)
    return payload


def read_run_request(run_root: str | Path) -> dict[str, Any] | None:
    path = run_request_path(run_root)
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except ValueError:
        return None


def clear_run_request(run_root: str | Path) -> bool:
    """request file を削除する（CLI の clear-request / 学習側の補助）。"""
    path = run_request_path(run_root)
    if path.exists():
        path.unlink()
        return True
    return False


def archive_run_request(run_root: str | Path, *, disposition: str) -> Path | None:
    """処理済 request を ``control/processed_requests/`` へ移動する。"""
    import os

    path = run_request_path(run_root)
    if not path.is_file():
        return None
    try:
        req = read_json(path)
        cmd = str(req.get("command", "unknown"))
    except ValueError:
        cmd = "unparseable"
    dest_dir = Path(run_root) / PROCESSED_REQUESTS_REL
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{_safe_ts()}_{cmd}_{disposition}.json"
    i = 1
    while dest.exists():
        dest = dest_dir / f"{_safe_ts()}_{cmd}_{disposition}_{i}.json"
        i += 1
    os.replace(path, dest)
    return dest


def _hostname() -> str:
    import socket

    return socket.gethostname()


# ---------------------------------------------------------------------------
# status aggregation
# ---------------------------------------------------------------------------


def _count_manifests(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("manifest.json"))


def _read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            last = line
    if last is None:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def _heartbeat_stats(hb_dir: Path) -> dict[str, int]:
    if not hb_dir.is_dir():
        return {"count": 0, "stale": 0}
    now = _utc_now()
    count = 0
    stale = 0
    for p in hb_dir.glob("*.json"):
        count += 1
        try:
            row = read_json(p)
            ts = datetime.strptime(
                str(row.get("last_update_at", "")), "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
            if (now - ts).total_seconds() > _STALE_HEARTBEAT_SEC:
                stale += 1
        except (ValueError, KeyError, OSError):
            stale += 1
    return {"count": count, "stale": stale}


def collect_status(run_root: str | Path) -> dict[str, Any]:
    """run 全体の status を 1 dict に集約する。"""
    root = Path(run_root)
    status: dict[str, Any] = {"run_root": str(root)}

    # config / state
    try:
        cfg = read_run_config(run_config_path(root))
        status["run_id"] = cfg.run_id
        status["seed_namespace"] = cfg.seed_namespace
    except (FileNotFoundError, KeyError, ValueError):
        status["run_id"] = None
    try:
        st = read_run_state(state_path(root))
        status["phase"] = st.phase.value
        status["phase_generation"] = st.phase_generation
        status["policy_version"] = st.policy_version
        status["message"] = st.message
        status["failure_reason"] = st.failure_reason
    except (FileNotFoundError, KeyError, ValueError):
        status["phase"] = None

    # latest checkpoint
    latest = root / "checkpoints" / "latest.json"
    if latest.is_file():
        try:
            reg = read_json(latest)
            status["latest_checkpoint"] = {
                "policy_version": reg.get("policy_version"),
                "checkpoint_path": reg.get("checkpoint_path"),
            }
        except ValueError:
            status["latest_checkpoint"] = None
    else:
        status["latest_checkpoint"] = None

    # best
    best = root / "best" / "best.json"
    if best.is_file():
        try:
            b = read_json(best)
            status["best_checkpoint"] = {
                "policy_version": b.get("policy_version"),
                "metric": b.get("metric"),
                "metric_value": b.get("metric_value"),
                "mode": b.get("mode"),
            }
        except ValueError:
            status["best_checkpoint"] = None
    else:
        status["best_checkpoint"] = None

    # shard counts
    status["shards"] = {
        "imitation": {
            "pending": _count_manifests(root / "imitation" / "pending"),
            "ready": _count_manifests(root / "imitation" / "ready"),
            "consumed": _count_manifests(root / "imitation" / "consumed"),
            "rejected": _count_manifests(root / "imitation" / "rejected"),
        },
        "rollouts": {
            "pending": _count_manifests(root / "rollouts" / "pending"),
            "ready": _count_manifests(root / "rollouts" / "ready"),
            "consumed": _count_manifests(root / "rollouts" / "consumed"),
            "rejected": _count_manifests(root / "rollouts" / "rejected"),
        },
    }

    # heartbeats
    status["heartbeats"] = _heartbeat_stats(root / "metrics" / "heartbeats")

    # learner transaction journal
    status["learner_journal_present"] = (root / LEARNER_JOURNAL_REL).is_file()

    # run request
    req = read_run_request(root)
    status["run_request"] = (
        {"command": req.get("command"), "requested_at": req.get("requested_at")}
        if req
        else None
    )

    # metrics tail
    learner_tail = _read_last_jsonl(root / "metrics" / "learner.jsonl")
    eval_tail = _read_last_jsonl(root / "metrics" / "eval.jsonl")
    status["learner_tail"] = (
        {
            k: learner_tail.get(k)
            for k in (
                "event", "policy_version_after", "num_samples",
                "approx_kl_mean", "clip_fraction",
            )
        }
        if learner_tail
        else None
    )
    status["eval_tail"] = (
        {
            k: eval_tail.get(k)
            for k in ("event", "policy_version", "mode", "avg_rank", "win_rate")
        }
        if eval_tail
        else None
    )
    return status


def format_status(status: dict[str, Any]) -> str:
    """``collect_status`` を human-readable text にする。"""
    lines: list[str] = []
    lines.append(f"run_id:           {status.get('run_id')}")
    lines.append(
        f"phase:            {status.get('phase')} "
        f"(gen {status.get('phase_generation')}, "
        f"policy_version {status.get('policy_version')})"
    )
    if status.get("failure_reason"):
        lines.append(f"failure_reason:   {status['failure_reason']}")
    lc = status.get("latest_checkpoint")
    lines.append(
        f"latest_checkpoint: v{lc['policy_version']} {lc['checkpoint_path']}"
        if lc else "latest_checkpoint: (none)"
    )
    bc = status.get("best_checkpoint")
    lines.append(
        f"best_checkpoint:  v{bc['policy_version']} {bc['metric']}={bc['metric_value']}"
        f" ({bc['mode']})" if bc else "best_checkpoint:  (none)"
    )
    sh = status.get("shards", {})
    for q in ("imitation", "rollouts"):
        c = sh.get(q, {})
        lines.append(
            f"{q:9} shards: pending={c.get('pending', 0)} ready={c.get('ready', 0)} "
            f"consumed={c.get('consumed', 0)} rejected={c.get('rejected', 0)}"
        )
    hb = status.get("heartbeats", {})
    lines.append(
        f"heartbeats:       {hb.get('count', 0)} (stale {hb.get('stale', 0)})"
    )
    lines.append(f"learner_journal:  {status.get('learner_journal_present')}")
    req = status.get("run_request")
    lines.append(f"run_request:      {req['command'] if req else '(none)'}")
    if status.get("learner_tail"):
        lines.append(f"learner_tail:     {status['learner_tail']}")
    if status.get("eval_tail"):
        lines.append(f"eval_tail:        {status['eval_tail']}")
    return "\n".join(lines)


__all__ = [
    "REQUEST_VERSION",
    "VALID_COMMANDS",
    "LEARNER_COMMANDS",
    "run_request_path",
    "write_run_request",
    "read_run_request",
    "clear_run_request",
    "archive_run_request",
    "collect_status",
    "format_status",
]
