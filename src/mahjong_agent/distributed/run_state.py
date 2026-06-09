"""Distributed run の immutable config + 可変 phase state の基盤。

ローカル PC と大規模サーバ群で同じ運用手順を使えるよう、distributed run 全体の
**固定設定** (`<run-root>/run_config.json`) と **現在 phase** (`<run-root>/
control/state.json`) を管理する。

- `run_config.json`: 一度書いたら不変 (同一内容の再書込のみ許容、異なる内容は
  fail-fast)。run 全体の seed namespace / imitation / PPO / evaluator /
  operations 設定。
- `control/state.json`: learner だけが書く single-writer な可変状態。現在 phase /
  phase_generation / policy_version 等。`write_json_atomic` で atomic 更新。

本 module は config/state の schema・validation・atomic helper・phase transition
の検証を提供するだけで、actor/learner/evaluator の phase-aware 化自体は後続の
supervisor 実装で行う。

hidden info は扱わない (config/state は run の設定とメタのみ)。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from mahjong_agent.distributed.manifest import read_json, write_json_atomic

CONFIG_VERSION: int = 1
STATE_VERSION: int = 1

_KNOWN_EVAL_MODES = frozenset({"greedy", "stochastic", "low_temp"})

RUN_CONFIG_NAME = "run_config.json"
STATE_REL_PATH = "control/state.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------


class RunPhase(str, Enum):
    INITIALIZING = "initializing"
    IMITATION_COLLECT = "imitation_collect"
    IMITATION_TRAIN = "imitation_train"
    PPO = "ppo"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


# phase ごとに許可する遷移先。FAILED/STOPPED -> INITIALIZING は resume 時のみ。
_ALLOWED_TRANSITIONS: dict[RunPhase, frozenset[RunPhase]] = {
    RunPhase.INITIALIZING: frozenset({RunPhase.IMITATION_COLLECT, RunPhase.FAILED}),
    RunPhase.IMITATION_COLLECT: frozenset(
        {RunPhase.IMITATION_TRAIN, RunPhase.DRAINING, RunPhase.FAILED}
    ),
    RunPhase.IMITATION_TRAIN: frozenset({RunPhase.PPO, RunPhase.FAILED}),
    RunPhase.PPO: frozenset({RunPhase.DRAINING, RunPhase.FAILED}),
    RunPhase.DRAINING: frozenset({RunPhase.STOPPED, RunPhase.FAILED}),
    RunPhase.FAILED: frozenset({RunPhase.INITIALIZING}),
    RunPhase.STOPPED: frozenset({RunPhase.INITIALIZING}),
}

# resume 専用の遷移 (allow_resume=True を必須にする)。
_RESUME_ONLY: frozenset[tuple[RunPhase, RunPhase]] = frozenset(
    {
        (RunPhase.FAILED, RunPhase.INITIALIZING),
        (RunPhase.STOPPED, RunPhase.INITIALIZING),
    }
)


def is_transition_allowed(
    current: RunPhase, nxt: RunPhase, *, allow_resume: bool = False
) -> bool:
    """``current -> nxt`` の phase 遷移が許可されるか (same-phase は対象外)。"""
    if nxt not in _ALLOWED_TRANSITIONS.get(current, frozenset()):
        return False
    if (current, nxt) in _RESUME_ONLY and not allow_resume:
        return False
    return True


# ---------------------------------------------------------------------------
# immutable run config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImitationRunConfig:
    target_games: int = 3000
    chunk_games: int = 100
    teacher: str = "rule_based"
    epochs: int = 8


@dataclass(frozen=True)
class PPORunConfig:
    actor_chunk_games: int = 100
    game_type: str = "YON_TONPUSEN"
    temperature: float = 1.0
    min_samples_per_update: int = 200_000
    max_samples_per_update: int = 500_000
    max_policy_lag: int = 1
    learning_rate: float = 5e-4
    epochs: int = 1
    target_kl: float = 0.01


@dataclass(frozen=True)
class EvaluatorRunConfig:
    enabled: bool = True
    eval_games_per_seat: int = 100
    modes: tuple[str, ...] = ("greedy", "stochastic")
    opponent: str = "rule_based"


@dataclass(frozen=True)
class OperationsRunConfig:
    actor_poll_interval_sec: float = 5.0
    learner_poll_interval_sec: float = 5.0
    max_ready_shards: int = 10_000
    checkpoint_keep_every: int = 50


@dataclass(frozen=True)
class DistributedRunConfig:
    run_id: str
    seed_namespace: str
    config_version: int = CONFIG_VERSION
    imitation: ImitationRunConfig = field(default_factory=ImitationRunConfig)
    ppo: PPORunConfig = field(default_factory=PPORunConfig)
    evaluator: EvaluatorRunConfig = field(default_factory=EvaluatorRunConfig)
    operations: OperationsRunConfig = field(default_factory=OperationsRunConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # tuple modes を JSON 安定のため list 化 (read 側で tuple に戻す)。
        d["evaluator"]["modes"] = list(self.evaluator.modes)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DistributedRunConfig:
        imitation = d.get("imitation", {})
        ppo = d.get("ppo", {})
        evaluator = dict(d.get("evaluator", {}))
        operations = d.get("operations", {})
        if "modes" in evaluator and evaluator["modes"] is not None:
            evaluator["modes"] = tuple(evaluator["modes"])
        return cls(
            run_id=str(d["run_id"]),
            seed_namespace=str(d["seed_namespace"]),
            config_version=int(d.get("config_version", CONFIG_VERSION)),
            imitation=ImitationRunConfig(**imitation),
            ppo=PPORunConfig(**ppo),
            evaluator=EvaluatorRunConfig(**evaluator),
            operations=OperationsRunConfig(**operations),
        )


def default_run_config(run_id: str, *, seed_namespace: str | None = None) -> DistributedRunConfig:
    """合理的な default を持つ run config を作る (smoke / test 用)。"""
    return DistributedRunConfig(
        run_id=str(run_id),
        seed_namespace=str(seed_namespace or run_id),
    )


# ---------------------------------------------------------------------------
# config hashing / validation
# ---------------------------------------------------------------------------


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_sha256(config: DistributedRunConfig) -> str:
    """run config の canonical JSON SHA-256 hex digest。"""
    return hashlib.sha256(
        _canonical_json(config.to_dict()).encode("utf-8")
    ).hexdigest()


def validate_run_config(config: DistributedRunConfig) -> None:
    """fail-fast validation。不正なら ``ValueError``。"""
    if not str(config.run_id).strip():
        raise ValueError("run_id must be non-empty")
    if not str(config.seed_namespace).strip():
        raise ValueError("seed_namespace must be non-empty")

    im = config.imitation
    if im.target_games <= 0 or im.chunk_games <= 0 or im.epochs <= 0:
        raise ValueError("imitation games/epochs must be positive")
    if im.chunk_games > im.target_games:
        raise ValueError(
            f"imitation.chunk_games ({im.chunk_games}) must be <= target_games "
            f"({im.target_games})"
        )

    ppo = config.ppo
    if ppo.actor_chunk_games <= 0 or ppo.epochs <= 0:
        raise ValueError("ppo actor_chunk_games/epochs must be positive")
    if ppo.min_samples_per_update <= 0 or ppo.max_samples_per_update <= 0:
        raise ValueError("ppo min/max samples must be positive")
    if ppo.min_samples_per_update > ppo.max_samples_per_update:
        raise ValueError(
            f"ppo.min_samples_per_update ({ppo.min_samples_per_update}) must be "
            f"<= max_samples_per_update ({ppo.max_samples_per_update})"
        )
    if float(ppo.temperature) != 1.0:
        raise ValueError(
            "ppo.temperature must be 1.0 (only temperature==1.0 stays PPO eligible)"
        )
    if ppo.max_policy_lag < 0:
        raise ValueError("ppo.max_policy_lag must be >= 0")
    if ppo.learning_rate <= 0:
        raise ValueError("ppo.learning_rate must be positive")
    if ppo.target_kl <= 0:
        raise ValueError("ppo.target_kl must be positive")

    ev = config.evaluator
    if ev.enabled:
        if ev.eval_games_per_seat <= 0:
            raise ValueError("evaluator.eval_games_per_seat must be positive")
        if not ev.modes:
            raise ValueError("evaluator.modes must be non-empty when enabled")
        unknown = set(ev.modes) - _KNOWN_EVAL_MODES
        if unknown:
            raise ValueError(f"evaluator.modes has unknown values: {sorted(unknown)}")

    ops = config.operations
    if ops.actor_poll_interval_sec <= 0 or ops.learner_poll_interval_sec <= 0:
        raise ValueError("operations poll intervals must be positive")
    if ops.max_ready_shards <= 0:
        raise ValueError("operations.max_ready_shards must be > 0")
    if ops.checkpoint_keep_every <= 0:
        raise ValueError("operations.checkpoint_keep_every must be positive")


# ---------------------------------------------------------------------------
# run state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunState:
    run_id: str
    phase: RunPhase
    phase_generation: int
    config_sha256: str
    updated_at: str
    policy_version: int | None = None
    message: str = ""
    previous_phase: RunPhase | None = None
    failure_reason: str = ""
    state_version: int = STATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": int(self.state_version),
            "run_id": str(self.run_id),
            "phase": self.phase.value,
            "phase_generation": int(self.phase_generation),
            "policy_version": (
                int(self.policy_version) if self.policy_version is not None else None
            ),
            "config_sha256": str(self.config_sha256),
            "updated_at": str(self.updated_at),
            "message": str(self.message),
            "previous_phase": (
                self.previous_phase.value if self.previous_phase is not None else None
            ),
            "failure_reason": str(self.failure_reason),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunState:
        prev = d.get("previous_phase")
        pv = d.get("policy_version")
        return cls(
            run_id=str(d["run_id"]),
            phase=RunPhase(str(d["phase"])),
            phase_generation=int(d["phase_generation"]),
            config_sha256=str(d["config_sha256"]),
            updated_at=str(d["updated_at"]),
            policy_version=(int(pv) if pv is not None else None),
            message=str(d.get("message", "")),
            previous_phase=(RunPhase(str(prev)) if prev is not None else None),
            failure_reason=str(d.get("failure_reason", "")),
            state_version=int(d.get("state_version", STATE_VERSION)),
        )


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def run_config_path(run_root: str | Path) -> Path:
    return Path(run_root) / RUN_CONFIG_NAME


def state_path(run_root: str | Path) -> Path:
    return Path(run_root) / STATE_REL_PATH


# ---------------------------------------------------------------------------
# config / state read-write helpers
# ---------------------------------------------------------------------------


def write_run_config(path: str | Path, config: DistributedRunConfig) -> None:
    """immutable に run config を書く。

    既に存在する場合、内容が完全一致なら no-op、異なるなら ``ValueError``。
    """
    validate_run_config(config)
    path = Path(path)
    if path.is_file():
        existing = DistributedRunConfig.from_dict(read_json(path))
        if config_sha256(existing) != config_sha256(config):
            raise ValueError(
                f"run config already exists with different content: {path}"
            )
        return
    write_json_atomic(path, config.to_dict())


def read_run_config(path: str | Path) -> DistributedRunConfig:
    return DistributedRunConfig.from_dict(read_json(path))


def write_run_state(path: str | Path, state: RunState) -> None:
    write_json_atomic(path, state.to_dict())


def read_run_state(path: str | Path) -> RunState:
    return RunState.from_dict(read_json(path))


# ---------------------------------------------------------------------------
# directory layout
# ---------------------------------------------------------------------------


_RUN_SUBDIRS = (
    "control",
    "imitation",
    "rollouts/pending",
    "rollouts/ready",
    "rollouts/consumed",
    "rollouts/rejected",
    "checkpoints",
    "metrics/actors",
    "eval",
    "best",
)


def ensure_run_dirs(run_root: str | Path) -> None:
    root = Path(run_root)
    for sub in _RUN_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# init / transition
# ---------------------------------------------------------------------------


def initialize_run_state(
    run_root: str | Path, config: DistributedRunConfig
) -> RunState:
    """run-root を初期化する (idempotent)。

    - directory layout を作成。
    - immutable ``run_config.json`` を書く (異なる内容なら fail-fast)。
    - ``control/state.json`` が無ければ ``initializing`` で作成。既にあれば
      **壊さず** 既存 state を返す (init 再実行で run を破壊しない)。
    """
    validate_run_config(config)
    root = Path(run_root)
    ensure_run_dirs(root)
    write_run_config(run_config_path(root), config)

    sp = state_path(root)
    if sp.is_file():
        existing = read_run_state(sp)
        # config hash 不一致 (= 別 config で再 init しようとした) は fail-fast。
        if existing.config_sha256 != config_sha256(config):
            raise ValueError(
                "existing run state was created with a different run config"
            )
        return existing

    state = RunState(
        run_id=config.run_id,
        phase=RunPhase.INITIALIZING,
        phase_generation=0,
        config_sha256=config_sha256(config),
        updated_at=_utc_now_iso(),
        policy_version=None,
        message="initialized",
        previous_phase=None,
        failure_reason="",
    )
    write_run_state(sp, state)
    return state


# policy_version を「更新しない」ことを表す sentinel。
_UNSET = object()


def transition_run_state(
    run_root: str | Path,
    *,
    expected_phase: RunPhase,
    next_phase: RunPhase,
    config: DistributedRunConfig | None = None,
    policy_version: Any = _UNSET,
    message: str = "",
    failure_reason: str = "",
    allow_resume: bool = False,
) -> RunState:
    """phase state を atomic に遷移/更新する (learner single-writer)。

    - ``expected_phase`` が現在 state と不一致なら fail-fast (race 防止)。
    - config hash 不一致なら fail-fast。
    - ``next_phase != expected_phase`` のとき: 遷移表で検証し
      ``phase_generation`` を +1 する。
    - ``next_phase == expected_phase`` のとき: 同 phase 内 update とみなし
      ``phase_generation`` は据え置き (message / policy_version の更新用)。
    """
    root = Path(run_root)
    state = read_run_state(state_path(root))
    if state.phase != expected_phase:
        raise ValueError(
            f"expected_phase mismatch: state={state.phase.value} "
            f"expected={expected_phase.value}"
        )
    cfg = config if config is not None else read_run_config(run_config_path(root))
    if state.config_sha256 != config_sha256(cfg):
        raise ValueError("config hash mismatch: run_config.json changed since init")

    if next_phase == expected_phase:
        new_generation = state.phase_generation
        previous_phase = state.previous_phase
    else:
        if not is_transition_allowed(
            expected_phase, next_phase, allow_resume=allow_resume
        ):
            raise ValueError(
                f"illegal phase transition: {expected_phase.value} -> "
                f"{next_phase.value} (allow_resume={allow_resume})"
            )
        new_generation = state.phase_generation + 1
        previous_phase = expected_phase

    new_policy_version = (
        state.policy_version if policy_version is _UNSET else policy_version
    )
    new_state = RunState(
        run_id=state.run_id,
        phase=next_phase,
        phase_generation=new_generation,
        config_sha256=state.config_sha256,
        updated_at=_utc_now_iso(),
        policy_version=(
            int(new_policy_version) if new_policy_version is not None else None
        ),
        message=str(message),
        previous_phase=previous_phase,
        failure_reason=str(failure_reason),
    )
    write_run_state(state_path(root), new_state)
    return new_state


__all__ = [
    "CONFIG_VERSION",
    "STATE_VERSION",
    "RunPhase",
    "ImitationRunConfig",
    "PPORunConfig",
    "EvaluatorRunConfig",
    "OperationsRunConfig",
    "DistributedRunConfig",
    "RunState",
    "default_run_config",
    "validate_run_config",
    "config_sha256",
    "is_transition_allowed",
    "write_run_config",
    "read_run_config",
    "write_run_state",
    "read_run_state",
    "run_config_path",
    "state_path",
    "ensure_run_dirs",
    "initialize_run_state",
    "transition_run_state",
]
