"""Phase-aware evaluator supervisor.

run state を見て checkpoint を継続評価する evaluator。learner / actor とは独立な
process で、`state.json` には **書き込まない**（single-writer 原則を壊さない）。既存
``distributed.evaluator.run_evaluator`` を内部 API として呼ぶ。

phase 別:
- initializing / imitation_collect / imitation_train: 待機。
- ppo: 未評価 checkpoint を順に eval（`checkpoint_version=unevaluated`）。
- draining: 未評価 checkpoint をできるだけ eval してから終了。
- stopped / failed: 終了。

hidden info は扱わない（評価は public self-play、成果物は metrics/eval.jsonl と
best/）。設計は `docs/distributed_actor_learner.md`。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mahjong_agent.distributed.evaluator import EvaluatorConfig, run_evaluator
from mahjong_agent.distributed.run_state import (
    RunPhase,
    read_run_state,
    state_path,
)


def resolve_device(name: str | None) -> str:
    import torch

    if name:
        return str(name)
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class EvaluatorSupervisorConfig:
    run_root: str
    eval_games_per_seat: int = 50
    mode_names: tuple[str, ...] = ("greedy", "stochastic")
    low_temp: float = 0.2
    opponent: str = "rule_based"
    device: str | None = None
    game_type_name: str = "YON_TONPUSEN"
    poll_interval_sec: float = 5.0
    checkpoints_per_round: int = 1
    max_idle_polls: int = 0  # 0 = unlimited（待機/未評価なしの上限。test 用）
    max_eval_rounds: int = 0  # 0 = unlimited（eval 実行回数の上限。test 用）
    max_runtime_sec: float | None = None
    seed: int = 42


def _eval_config(config: EvaluatorSupervisorConfig, device: str) -> EvaluatorConfig:
    return EvaluatorConfig(
        run_root=str(config.run_root),
        checkpoint_version="unevaluated",
        eval_games_per_seat=int(config.eval_games_per_seat),
        mode_names=tuple(config.mode_names),
        low_temp=float(config.low_temp),
        opponent=str(config.opponent),
        device=device,
        game_type_name=str(config.game_type_name),
        stop_after_checkpoints=int(config.checkpoints_per_round),
        seed=int(config.seed),
    )


_WAIT_PHASES = frozenset(
    {RunPhase.INITIALIZING, RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN}
)
_TERMINAL_PHASES = frozenset({RunPhase.STOPPED, RunPhase.FAILED})


def run_evaluator_supervisor(config: EvaluatorSupervisorConfig) -> dict[str, Any]:
    """run state に追従して未評価 checkpoint を継続評価する。"""
    device = resolve_device(config.device)
    run_root = Path(config.run_root)
    start = time.time()
    idle_polls = 0
    eval_rounds = 0
    evaluated_total = 0
    exit_reason = "loop_end"

    while True:
        if (
            config.max_runtime_sec is not None
            and (time.time() - start) >= float(config.max_runtime_sec)
        ):
            exit_reason = "max_runtime"
            break
        state = read_run_state(state_path(run_root))
        phase = state.phase

        if phase in _TERMINAL_PHASES:
            exit_reason = phase.value
            break

        if phase == RunPhase.DRAINING:
            # 終了前に未評価 checkpoint をできるだけ評価する。
            summary = run_evaluator(_drain_eval_config(config, device))
            evaluated_total += int(summary.get("evaluated_checkpoints", 0))
            exit_reason = "draining"
            break

        if phase in _WAIT_PHASES:
            idle_polls += 1
            if config.max_idle_polls and idle_polls >= config.max_idle_polls:
                exit_reason = "max_idle_polls"
                break
            time.sleep(config.poll_interval_sec)
            continue

        # ppo: 未評価 checkpoint を評価する
        summary = run_evaluator(_eval_config(config, device))
        evaluated = int(summary.get("evaluated_checkpoints", 0))
        evaluated_total += evaluated
        eval_rounds += 1
        if config.max_eval_rounds and eval_rounds >= int(config.max_eval_rounds):
            exit_reason = "max_eval_rounds"
            break
        if evaluated == 0:
            # 未評価が無い → poll 待ち
            idle_polls += 1
            if config.max_idle_polls and idle_polls >= config.max_idle_polls:
                exit_reason = "max_idle_polls"
                break
            time.sleep(config.poll_interval_sec)
            continue
        idle_polls = 0

    return {
        "ok": True,
        "run_root": str(run_root),
        "device": device,
        "eval_rounds": eval_rounds,
        "evaluated_checkpoints": evaluated_total,
        "exit_reason": exit_reason,
    }


def _drain_eval_config(config: EvaluatorSupervisorConfig, device: str) -> EvaluatorConfig:
    # draining では残り未評価をできるだけ拾うため上限を大きめにする。
    cfg = _eval_config(config, device)
    return EvaluatorConfig(
        run_root=cfg.run_root,
        checkpoint_version="unevaluated",
        eval_games_per_seat=cfg.eval_games_per_seat,
        mode_names=cfg.mode_names,
        low_temp=cfg.low_temp,
        opponent=cfg.opponent,
        device=cfg.device,
        game_type_name=cfg.game_type_name,
        stop_after_checkpoints=10_000,
        seed=cfg.seed,
    )


__all__ = [
    "EvaluatorSupervisorConfig",
    "resolve_device",
    "run_evaluator_supervisor",
]
