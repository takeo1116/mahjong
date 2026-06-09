"""Unified learner supervisor: bootstrap (imitation) → PPO with crash-safe resume.

GPU サーバで 1 コマンド起動し、`run_config.json` / `control/state.json` に従って

    initializing → imitation_collect → imitation_train → (policy_000001 publish)
    → ppo → draining / stopped

を人手なしで進める learner。`control/state.json` の **single-writer** であり、
singleton lock で二重起動を防ぐ。

### crash 安全性（transaction journal）

checkpoint/latest publish・shard consume・state 更新の間に crash すると
registry / state / shard lifecycle が不整合になり自動 resume できない。これを防ぐ
ため `control/learner_transaction.json` に commit protocol の進行段階を残し、起動
直後に reconcile して publish 後の処理を **idempotent に完了**させる。

commit protocol（imitation / PPO 共通）:

    train/update 成功
    → journal stage=prepared
    → checkpoint + latest.json publish
    → journal stage=published（checkpoint_sha256 を記録）
    → source shard を idempotent に consume
    → journal stage=shards_consumed
    → state を更新（policy_version / phase）
    → journal stage=state_committed
    → journal 削除

publish 後の任意地点で crash しても、次回起動時の reconcile が published 以降を
再開して整合させる（再 training しない、二重 consume しない）。hash mismatch /
shard 消失など自動復旧不能は state=failed + failure_reason。

既存 distributed learner CLI / imitation・PPO loss / DecisionSample schema /
RiichiEnv は変更しない。hidden info は扱わない。設計は
`docs/distributed_actor_learner.md`。
"""
from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mahjong_agent.data import read_decision_shard
from mahjong_agent.distributed.manifest import (
    CheckpointRegistryEntry,
    RolloutShardManifest,
    is_manifest_compatible,
    read_json,
    sha256_file,
    write_json_atomic,
)
from mahjong_agent.distributed.run_control import (
    LEARNER_COMMANDS,
    archive_run_request,
    read_run_request,
)
from mahjong_agent.distributed.run_state import (
    DistributedRunConfig,
    RunPhase,
    config_sha256,
    initialize_run_state,
    read_run_config,
    read_run_state,
    run_config_path,
    state_path,
    transition_run_state,
    validate_run_config,
)
from mahjong_agent.encoders import PublicObservationEncoder
from mahjong_agent.models import Stage03Model, Stage03ModelConfig
from mahjong_agent.training import (
    ImitationConfig,
    PPOConfig,
    compute_returns_and_advantages,
    fit_imitation,
    fit_ppo,
)
from mahjong_agent.training.ppo import make_default_ppo_optimizer

SCHEMA_VERSION = 2  # DecisionSample schema (= manifest schema_version)
TRANSACTION_VERSION = 1
JOURNAL_REL = "control/learner_transaction.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_device(name: str | None) -> str:
    if name:
        return str(name)
    return "cuda" if torch.cuda.is_available() else "cpu"


class LearnerRecoveryError(RuntimeError):
    """journal reconcile が自動復旧不能（hash mismatch / shard 消失 等）。"""


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnerSupervisorConfig:
    run_root: str
    config_path: str | None = None
    device: str | None = None
    poll_interval_sec: float = 5.0
    max_runtime_sec: float | None = None
    max_idle_polls: int = 0
    stop_after_ppo_updates: int = 0
    consume_policy: str = "move"
    force_unlock: bool = False


# ---------------------------------------------------------------------------
# directory layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LSDirs:
    root: Path
    control: Path
    checkpoints: Path
    imitation_ready: Path
    imitation_consumed: Path
    rollouts_ready: Path
    rollouts_consumed: Path
    rollouts_rejected: Path
    metrics: Path

    @classmethod
    def from_root(cls, run_root: str | Path) -> _LSDirs:
        root = Path(run_root)
        return cls(
            root=root,
            control=root / "control",
            checkpoints=root / "checkpoints",
            imitation_ready=root / "imitation" / "ready",
            imitation_consumed=root / "imitation" / "consumed",
            rollouts_ready=root / "rollouts" / "ready",
            rollouts_consumed=root / "rollouts" / "consumed",
            rollouts_rejected=root / "rollouts" / "rejected",
            metrics=root / "metrics",
        )

    def ensure(self) -> None:
        for p in (
            self.control, self.checkpoints, self.imitation_ready,
            self.imitation_consumed, self.rollouts_ready, self.rollouts_consumed,
            self.rollouts_rejected, self.metrics,
        ):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def latest_json(self) -> Path:
        return self.checkpoints / "latest.json"

    def rel(self, p: Path) -> str:
        return str(Path(p).resolve().relative_to(self.root.resolve()))


# ---------------------------------------------------------------------------
# singleton lock
# ---------------------------------------------------------------------------


class LearnerLock:
    """`control/learner.lock/` の directory 作成による atomic singleton lock。"""

    def __init__(self, run_root: str | Path, *, force: bool = False):
        self.dir = Path(run_root) / "control" / "learner.lock"
        self.force = bool(force)
        self._held = False

    def acquire(self) -> None:
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(self.dir)
        except FileExistsError:
            if not self.force:
                raise RuntimeError(
                    f"learner already locked: {self.dir} owner={self._read_owner()}"
                ) from None
            shutil.rmtree(self.dir, ignore_errors=True)
            os.mkdir(self.dir)
        write_json_atomic(
            self.dir / "owner.json",
            {"hostname": _hostname(), "pid": os.getpid(), "started_at": _utc_now_iso()},
        )
        self._held = True

    def _read_owner(self) -> dict[str, Any] | None:
        try:
            return read_json(self.dir / "owner.json")
        except (FileNotFoundError, ValueError):
            return None

    def release(self) -> None:
        if self._held:
            shutil.rmtree(self.dir, ignore_errors=True)
            self._held = False

    def __enter__(self) -> LearnerLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _hostname() -> str:
    import socket

    return socket.gethostname()


# ---------------------------------------------------------------------------
# transaction journal
# ---------------------------------------------------------------------------


def journal_path(run_root: str | Path) -> Path:
    return Path(run_root) / JOURNAL_REL


def read_journal(dirs: _LSDirs) -> dict[str, Any] | None:
    p = journal_path(dirs.root)
    if not p.is_file():
        return None
    try:
        return read_json(p)
    except ValueError:
        return None


def write_journal(dirs: _LSDirs, journal: dict[str, Any]) -> None:
    journal["updated_at"] = _utc_now_iso()
    write_json_atomic(journal_path(dirs.root), journal)


def delete_journal(dirs: _LSDirs) -> None:
    p = journal_path(dirs.root)
    if p.exists():
        p.unlink()


def _latest_policy_version(dirs: _LSDirs) -> int | None:
    if not dirs.latest_json.is_file():
        return None
    try:
        return int(read_json(dirs.latest_json)["policy_version"])
    except (ValueError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# bootstrap (init or validate run config)
# ---------------------------------------------------------------------------


def bootstrap_run(config: LearnerSupervisorConfig) -> DistributedRunConfig:
    dirs = _LSDirs.from_root(config.run_root)
    rc_path = run_config_path(config.run_root)
    if config.config_path is not None:
        import json

        with open(config.config_path, encoding="utf-8") as f:
            provided = DistributedRunConfig.from_dict(json.load(f))
        validate_run_config(provided)
        if not rc_path.is_file():
            initialize_run_state(config.run_root, provided)
        else:
            existing = read_run_config(rc_path)
            if config_sha256(existing) != config_sha256(provided):
                raise ValueError(
                    "provided --config does not match existing run_config.json"
                )
    if not rc_path.is_file():
        raise FileNotFoundError(
            f"run not initialized and no --config provided: {rc_path}"
        )
    dirs.ensure()
    return read_run_config(rc_path)


# ---------------------------------------------------------------------------
# encoder metadata / checkpoint publish
# ---------------------------------------------------------------------------


def _build_encoder_metadata() -> dict[str, Any]:
    enc = PublicObservationEncoder(enable_hints=True)
    meta = enc.metadata()
    return {
        "observation_dim": int(meta.observation_dim),
        "candidate_dim": int(meta.candidate_dim),
        "enable_hints": bool(enc.enable_hints),
    }


def publish_policy_checkpoint(
    dirs: _LSDirs,
    *,
    model: Stage03Model,
    new_version: int,
    phase: RunPhase,
    phase_generation: int,
    config_sha: str,
    encoder_metadata: dict[str, Any],
    learner_metrics: dict[str, Any],
    source_policy_versions: list[int],
    num_samples: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> CheckpointRegistryEntry:
    """checkpoint を atomic publish して registry entry を返す。"""
    rel = f"checkpoints/policy_{new_version:06d}.pt"
    ckpt_path = dirs.root / rel
    tmp = ckpt_path.with_name(ckpt_path.name + ".tmp")
    model_config = asdict(model.config)
    payload: dict[str, Any] = {
        "model_state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        },
        "model_config": model_config,
        "policy_version": int(new_version),
        "phase": phase.value,
        "phase_generation": int(phase_generation),
        "run_config_sha256": config_sha,
        "learner_metrics": learner_metrics,
        "source_policy_versions": [int(v) for v in source_policy_versions],
        "num_samples": int(num_samples),
        "created_at": _utc_now_iso(),
        "encoder_metadata": encoder_metadata,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    with tmp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ckpt_path)

    entry = CheckpointRegistryEntry(
        policy_version=int(new_version),
        checkpoint_path=rel,
        checkpoint_sha256=sha256_file(ckpt_path),
        created_at=_utc_now_iso(),
        schema_version=SCHEMA_VERSION,
        observation_dim=int(model_config["observation_dim"]),
        candidate_dim=int(model_config["candidate_dim"]),
        model_config=model_config,
        encoder_metadata=encoder_metadata,
    )
    write_json_atomic(dirs.latest_json, entry.to_dict())
    return entry


def read_registry(dirs: _LSDirs) -> CheckpointRegistryEntry:
    if not dirs.latest_json.is_file():
        raise FileNotFoundError(f"registry not found: {dirs.latest_json}")
    return CheckpointRegistryEntry.from_dict(read_json(dirs.latest_json))


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for st in optimizer.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)


def load_model_and_optimizer(
    dirs: _LSDirs,
    entry: CheckpointRegistryEntry,
    device: str,
    ppo_config: PPOConfig,
) -> tuple[Stage03Model, torch.optim.Optimizer, bool]:
    """latest checkpoint から model と PPO optimizer を復元する。"""
    ckpt = (dirs.root / entry.checkpoint_path).resolve()
    actual = sha256_file(ckpt)
    if actual != entry.checkpoint_sha256:
        raise ValueError(
            f"checkpoint sha256 mismatch: registry={entry.checkpoint_sha256} "
            f"actual={actual}"
        )
    payload = torch.load(ckpt, map_location="cpu")
    model = Stage03Model(Stage03ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device))
    optimizer = make_default_ppo_optimizer(model, ppo_config)
    had_opt = "optimizer_state_dict" in payload
    if had_opt:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
    return model, optimizer, had_opt


# ---------------------------------------------------------------------------
# transaction commit protocol (publish -> consume -> state -> delete)
# ---------------------------------------------------------------------------


def _source_entries(dirs: _LSDirs, shards, *, consumed_root: Path) -> list[dict[str, str]]:
    """consumed destination を **transaction 作成時に確定**（idempotent 化）。"""
    out: list[dict[str, str]] = []
    for shard_dir, _manifest in shards:
        name = shard_dir.name
        out.append(
            {
                "ready_path": dirs.rel(shard_dir),
                "consumed_path": dirs.rel(consumed_root / name),
            }
        )
    return out


def _begin_transaction(
    dirs: _LSDirs,
    *,
    kind: str,
    phase: RunPhase,
    phase_generation: int,
    before: int,
    after: int,
    source_entries: list[dict[str, str]],
    checkpoint_rel: str,
    consume_policy: str,
) -> dict[str, Any]:
    journal = {
        "transaction_version": TRANSACTION_VERSION,
        "kind": kind,
        "stage": "prepared",
        "phase": phase.value,
        "phase_generation": int(phase_generation),
        "policy_version_before": int(before),
        "policy_version_after": int(after),
        "source_shards": source_entries,
        "checkpoint_path": checkpoint_rel,
        "checkpoint_sha256": "",
        "consume_policy": str(consume_policy),
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }
    write_journal(dirs, journal)
    return journal


def publish_transaction(
    dirs: _LSDirs,
    journal: dict[str, Any],
    *,
    model: Stage03Model,
    optimizer: torch.optim.Optimizer | None,
    encoder_metadata: dict[str, Any],
    learner_metrics: dict[str, Any],
    source_policy_versions: list[int],
    num_samples: int,
    run_config: DistributedRunConfig,
) -> CheckpointRegistryEntry:
    """checkpoint publish 後、journal を published（+sha）へ進める。"""
    entry = publish_policy_checkpoint(
        dirs,
        model=model,
        new_version=int(journal["policy_version_after"]),
        phase=RunPhase(journal["phase"]),
        phase_generation=int(journal["phase_generation"]),
        config_sha=config_sha256(run_config),
        encoder_metadata=encoder_metadata,
        learner_metrics=learner_metrics,
        source_policy_versions=source_policy_versions,
        num_samples=num_samples,
        optimizer=optimizer,
    )
    journal["checkpoint_sha256"] = entry.checkpoint_sha256
    journal["stage"] = "published"
    write_journal(dirs, journal)
    return entry


def _verify_published(dirs: _LSDirs, journal: dict[str, Any]) -> None:
    ckpt = dirs.root / journal["checkpoint_path"]
    if not ckpt.is_file():
        raise LearnerRecoveryError(f"checkpoint missing: {ckpt}")
    if sha256_file(ckpt) != journal["checkpoint_sha256"]:
        raise LearnerRecoveryError(f"checkpoint hash mismatch: {ckpt}")
    if _latest_policy_version(dirs) != int(journal["policy_version_after"]):
        raise LearnerRecoveryError("latest.json does not point to after version")


def _idempotent_consume_one(dirs: _LSDirs, e: dict[str, str], consume_policy: str) -> None:
    ready = dirs.root / e["ready_path"]
    consumed = dirs.root / e["consumed_path"]
    if consume_policy == "delete":
        if ready.exists():
            shutil.rmtree(ready, ignore_errors=True)
        return
    if consumed.exists():
        return  # 既に consume 済み
    if ready.exists():
        consumed.parent.mkdir(parents=True, exist_ok=True)
        os.replace(ready, consumed)
        return
    raise LearnerRecoveryError(f"source shard disappeared: {e['ready_path']}")


def _commit_state(dirs: _LSDirs, run_config: DistributedRunConfig, journal: dict[str, Any]) -> None:
    kind = journal["kind"]
    after = int(journal["policy_version_after"])
    state = read_run_state(state_path(dirs.root))
    if kind == "ppo_update":
        if state.phase == RunPhase.PPO and int(state.policy_version or -1) == after:
            return  # 既に commit 済み
        if state.phase != RunPhase.PPO:
            raise LearnerRecoveryError(
                f"unexpected phase for ppo commit: {state.phase.value}"
            )
        transition_run_state(
            dirs.root, expected_phase=RunPhase.PPO, next_phase=RunPhase.PPO,
            config=run_config, policy_version=after, message="ppo update (committed)",
        )
    elif kind == "imitation_publish":
        if state.phase == RunPhase.PPO:
            return  # 既に遷移済み
        if state.phase != RunPhase.IMITATION_TRAIN:
            raise LearnerRecoveryError(
                f"unexpected phase for imitation commit: {state.phase.value}"
            )
        transition_run_state(
            dirs.root, expected_phase=RunPhase.IMITATION_TRAIN, next_phase=RunPhase.PPO,
            config=run_config, policy_version=after,
            message="initial checkpoint committed",
        )
    else:
        raise LearnerRecoveryError(f"unknown transaction kind: {kind}")


def finish_transaction(
    dirs: _LSDirs, run_config: DistributedRunConfig, journal: dict[str, Any]
) -> None:
    """published 以降を idempotent に完了させる（consume → state → delete）。"""
    stage = journal["stage"]
    if stage == "published":
        _verify_published(dirs, journal)
        for e in journal["source_shards"]:
            _idempotent_consume_one(dirs, e, journal["consume_policy"])
        journal["stage"] = "shards_consumed"
        write_journal(dirs, journal)
        stage = "shards_consumed"
    if stage == "shards_consumed":
        _commit_state(dirs, run_config, journal)
        journal["stage"] = "state_committed"
        write_journal(dirs, journal)
        stage = "state_committed"
    if stage == "state_committed":
        delete_journal(dirs)


def reconcile_journal(
    dirs: _LSDirs, run_config: DistributedRunConfig
) -> str | None:
    """起動直後に journal があれば reconcile する。"""
    journal = read_journal(dirs)
    if journal is None:
        return None
    stage = journal.get("stage")
    if stage == "prepared":
        after = int(journal["policy_version_after"])
        ckpt = dirs.root / journal["checkpoint_path"]
        if _latest_policy_version(dirs) == after:
            # publish は完了していた（latest 更新済）。sha を latest から採用。
            entry = read_registry(dirs)
            if not ckpt.is_file() or sha256_file(ckpt) != entry.checkpoint_sha256:
                raise LearnerRecoveryError("prepared/after: checkpoint hash mismatch")
            journal["checkpoint_sha256"] = entry.checkpoint_sha256
            journal["stage"] = "published"
            write_journal(dirs, journal)
            finish_transaction(dirs, run_config, journal)
            return "reconciled_from_prepared"
        # publish 未完了: leftover checkpoint を消し journal 破棄（次 loop で再実行）。
        if ckpt.is_file():
            ckpt.unlink()
        delete_journal(dirs)
        return "discarded_prepared"
    if stage in ("published", "shards_consumed", "state_committed"):
        finish_transaction(dirs, run_config, journal)
        return f"reconciled_from_{stage}"
    # 未知 stage は安全側で破棄しない（fail-fast）。
    raise LearnerRecoveryError(f"unknown journal stage: {stage}")


# ---------------------------------------------------------------------------
# manifest scan helpers
# ---------------------------------------------------------------------------


def _scan_manifests(ready_root: Path):
    if not ready_root.is_dir():
        return
    for shard_dir in sorted(p for p in ready_root.iterdir() if p.is_dir()):
        mpath = shard_dir / "manifest.json"
        if not mpath.is_file():
            yield shard_dir, None, "manifest_missing"
            continue
        try:
            yield shard_dir, RolloutShardManifest.from_dict(read_json(mpath)), None
        except Exception:  # noqa: BLE001
            yield shard_dir, None, "manifest_parse_error"


def count_collect_games(dirs: _LSDirs, collect_generation: int) -> int:
    total = 0
    for _shard_dir, manifest, err in _scan_manifests(dirs.imitation_ready):
        if err or manifest is None:
            continue
        if (
            manifest.phase == RunPhase.IMITATION_COLLECT.value
            and manifest.phase_generation == int(collect_generation)
        ):
            total += int(manifest.num_games)
    return total


def _collect_teacher_shards(dirs: _LSDirs, collect_generation: int):
    out = []
    for shard_dir, manifest, err in _scan_manifests(dirs.imitation_ready):
        if err or manifest is None:
            continue
        if (
            manifest.phase == RunPhase.IMITATION_COLLECT.value
            and manifest.phase_generation == int(collect_generation)
        ):
            out.append((shard_dir, manifest))
    return out


# ---------------------------------------------------------------------------
# imitation phase
# ---------------------------------------------------------------------------


def run_imitation_train(
    dirs: _LSDirs,
    run_config: DistributedRunConfig,
    *,
    device: str,
    collect_generation: int,
    phase: RunPhase,
    phase_generation: int,
) -> dict[str, Any]:
    """teacher shard で学習し policy_000001 を journaled に publish/commit。"""
    shards = _collect_teacher_shards(dirs, collect_generation)
    samples: list[Any] = []
    for shard_dir, manifest in shards:
        samples.extend(read_decision_shard(shard_dir / manifest.shard_path))
    if not samples:
        raise RuntimeError(
            f"imitation_train: no teacher samples for generation {collect_generation}"
        )

    encoder = PublicObservationEncoder(enable_hints=True)
    model = Stage03Model(Stage03ModelConfig.from_encoder_metadata(encoder.metadata()))
    model.to(torch.device(device))
    imitation_config = ImitationConfig(
        num_epochs=int(run_config.imitation.epochs),
        batch_size=256, learning_rate=1e-3, device=device,
        tie_aware_discard=True, exclude_post_riichi_discards=True,
        per_player_round_weighting=False,
    )
    result = fit_imitation(model, samples, imitation_config)
    final = result.to_dict().get("final", {})

    source_entries = _source_entries(dirs, shards, consumed_root=dirs.imitation_consumed)
    journal = _begin_transaction(
        dirs, kind="imitation_publish", phase=phase, phase_generation=phase_generation,
        before=0, after=1, source_entries=source_entries,
        checkpoint_rel="checkpoints/policy_000001.pt", consume_policy="move",
    )
    publish_transaction(
        dirs, journal, model=model, optimizer=None,
        encoder_metadata=_build_encoder_metadata(),
        learner_metrics={"imitation_final": final, "num_samples": len(samples)},
        source_policy_versions=[], num_samples=len(samples), run_config=run_config,
    )
    finish_transaction(dirs, run_config, journal)
    _append_learner_metric(
        dirs,
        {
            "event": "imitation_train",
            "policy_version_after": 1,
            "num_samples": len(samples),
            "num_shards": len(shards),
            "created_at": _utc_now_iso(),
        },
    )
    return {
        "policy_version": 1, "num_samples": len(samples), "num_shards": len(shards),
        "imitation_final": final,
    }


# ---------------------------------------------------------------------------
# ppo phase
# ---------------------------------------------------------------------------


def _reason_key(reason: str) -> str:
    return str(reason).split("(", 1)[0] or "rejected"


def _move_reject(dirs: _LSDirs, shard_dir: Path, reason: str) -> None:
    parent = dirs.rollouts_rejected / reason
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / shard_dir.name
    if dest.exists():
        i = 1
        while (parent / f"{shard_dir.name}_dup{i}").exists():
            i += 1
        dest = parent / f"{shard_dir.name}_dup{i}"
    os.replace(shard_dir, dest)


@dataclass
class _PPOScan:
    compatible: list[tuple[Path, RolloutShardManifest]] = field(default_factory=list)
    rejects: list[tuple[Path, str]] = field(default_factory=list)


def scan_ppo_ready(
    dirs: _LSDirs,
    *,
    entry: CheckpointRegistryEntry,
    ppo_generation: int,
    max_policy_lag: int,
) -> _PPOScan:
    scan = _PPOScan()
    for shard_dir, manifest, err in _scan_manifests(dirs.rollouts_ready):
        if err or manifest is None:
            scan.rejects.append((shard_dir, err or "manifest_parse_error"))
            continue
        if manifest.phase is None or manifest.phase_generation is None:
            scan.rejects.append((shard_dir, "missing_phase"))
            continue
        if manifest.phase != RunPhase.PPO.value:
            scan.rejects.append((shard_dir, "phase_mismatch"))
            continue
        if int(manifest.phase_generation) != int(ppo_generation):
            scan.rejects.append((shard_dir, "generation_mismatch"))
            continue
        if not (shard_dir / manifest.shard_path).is_file():
            scan.rejects.append((shard_dir, "missing_shard"))
            continue
        compat = is_manifest_compatible(
            manifest, schema_version=int(entry.schema_version),
            observation_dim=int(entry.observation_dim),
            candidate_dim=int(entry.candidate_dim),
            current_policy_version=int(entry.policy_version),
            max_policy_lag=int(max_policy_lag),
        )
        if not compat.ok:
            scan.rejects.append((shard_dir, _reason_key(compat.reason)))
            continue
        scan.compatible.append((shard_dir, manifest))
    return scan


def _make_ppo_config(run_config: DistributedRunConfig, device: str) -> PPOConfig:
    return PPOConfig(
        learning_rate=float(run_config.ppo.learning_rate),
        batch_size=256, num_epochs=int(run_config.ppo.epochs), device=device,
        target_kl_enabled=True, target_kl=float(run_config.ppo.target_kl),
        target_kl_stop_multiplier=1.5, target_kl_skip_minibatch_on_exceed=True,
        exclude_post_riichi_discards=True, value_loss_includes_excluded=False,
        per_player_round_weighting=False, include_actor_types=("policy",),
    )


@dataclass
class _PPOContext:
    model: Stage03Model
    optimizer: torch.optim.Optimizer
    entry: CheckpointRegistryEntry


def _count_reasons(rejects) -> dict[str, int]:
    out: dict[str, int] = {}
    for _d, reason in rejects:
        out[reason] = out.get(reason, 0) + 1
    return out


def _append_learner_metric(dirs: _LSDirs, row: dict[str, Any]) -> None:
    """``metrics/learner.jsonl`` に 1 update 1 行を追記する（status / pilot 監視用）。"""
    import json

    path = dirs.metrics / "learner.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_ppo_update(
    dirs: _LSDirs,
    run_config: DistributedRunConfig,
    *,
    device: str,
    ppo_generation: int,
    ctx: _PPOContext,
    consume_policy: str,
) -> dict[str, Any]:
    """1 PPO update（generation filtered + journaled commit）。"""
    entry = ctx.entry
    scan = scan_ppo_ready(
        dirs, entry=entry, ppo_generation=ppo_generation,
        max_policy_lag=int(run_config.ppo.max_policy_lag),
    )
    for shard_dir, reason in scan.rejects:
        _move_reject(dirs, shard_dir, reason)

    used: list[tuple[Path, RolloutShardManifest]] = []
    samples: list[Any] = []
    corrupt: list[tuple[Path, str]] = []
    for shard_dir, manifest in scan.compatible:
        if samples and len(samples) >= int(run_config.ppo.max_samples_per_update):
            break
        try:
            shard_samples = read_decision_shard(shard_dir / manifest.shard_path)
        except Exception:  # noqa: BLE001
            corrupt.append((shard_dir, "corrupt_shard"))
            continue
        if len(shard_samples) != int(manifest.num_samples):
            corrupt.append((shard_dir, "num_samples_mismatch"))
            continue
        used.append((shard_dir, manifest))
        samples.extend(shard_samples)
    for shard_dir, reason in corrupt:
        _move_reject(dirs, shard_dir, reason)

    if len(samples) < int(run_config.ppo.min_samples_per_update):
        return {"need_more": True, "available": len(samples)}

    ppo_config = _make_ppo_config(run_config, device)
    data = compute_returns_and_advantages(samples, ppo_config)
    eligible = int(data.eligible.sum())
    if eligible <= 0:
        # eligible が無い batch は publish せず skip（run を failed にしない）。
        # used shard は ready に残し、後続 shard と合わせて再評価する。
        return {"need_more": True, "available": len(samples), "zero_eligible": True}
    result = fit_ppo(ctx.model, samples, ppo_config, optimizer=ctx.optimizer)
    final = result.final
    if final is None or int(final.num_updates) <= 0:
        raise RuntimeError(
            f"ppo update: no updates (early_stopped={result.early_stopped})"
        )

    before = int(entry.policy_version)
    after = before + 1
    source_versions = sorted({int(m.policy_version) for _d, m in used})
    source_entries = _source_entries(
        dirs, used, consumed_root=dirs.rollouts_consumed / f"policy_{after:06d}"
    )
    journal = _begin_transaction(
        dirs, kind="ppo_update", phase=RunPhase.PPO, phase_generation=ppo_generation,
        before=before, after=after, source_entries=source_entries,
        checkpoint_rel=f"checkpoints/policy_{after:06d}.pt",
        consume_policy=str(consume_policy),
    )
    new_entry = publish_transaction(
        dirs, journal, model=ctx.model, optimizer=ctx.optimizer,
        encoder_metadata=entry.encoder_metadata or _build_encoder_metadata(),
        learner_metrics={"ppo_final": final.to_dict(), "eligible": eligible},
        source_policy_versions=source_versions, num_samples=len(samples),
        run_config=run_config,
    )
    finish_transaction(dirs, run_config, journal)
    ctx.entry = new_entry
    ppo_final = final.to_dict()
    _append_learner_metric(
        dirs,
        {
            "event": "learner_update",
            "policy_version_before": before,
            "policy_version_after": after,
            "phase_generation": int(ppo_generation),
            "num_shards": len(used),
            "num_samples": len(samples),
            "eligible": eligible,
            "source_policy_versions": source_versions,
            "rejected_counts": _count_reasons(scan.rejects + corrupt),
            "loss": ppo_final.get("loss"),
            "policy_loss": ppo_final.get("policy_loss"),
            "value_loss": ppo_final.get("value_loss"),
            "entropy": ppo_final.get("entropy"),
            "approx_kl_mean": ppo_final.get("approx_kl_mean"),
            "clip_fraction": ppo_final.get("clip_fraction"),
            "num_updates": ppo_final.get("num_updates"),
            "checkpoint_path": new_entry.checkpoint_path,
            "created_at": _utc_now_iso(),
        },
    )
    return {
        "updated": True, "policy_version_before": before, "policy_version_after": after,
        "num_shards": len(used), "num_samples": len(samples), "eligible": eligible,
        "source_policy_versions": source_versions,
        "rejected_counts": _count_reasons(scan.rejects + corrupt),
        "ppo_final": ppo_final,
    }


# ---------------------------------------------------------------------------
# top-level driver
# ---------------------------------------------------------------------------


# active phase で drain/stop request を draining へ反映できる phase
# (run_state の遷移表で *->draining が許可されるのは collect / ppo のみ)。
_DRAINABLE_PHASES = frozenset({RunPhase.IMITATION_COLLECT, RunPhase.PPO})


def _handle_run_request(dirs: _LSDirs, run_config: DistributedRunConfig, state) -> str | None:
    """drain/stop request を **learner だけが** state へ反映する（single-writer 維持）。

    - drainable phase（imitation_collect / ppo）: draining へ遷移し request を archive。
    - draining / stopped / failed: no-op で archive。
    - initializing / imitation_train: まだ draining に直接遷移できないので request を
      **残したまま** False を返す（drainable phase 到達後に適用）。
    - 未知 command / clear: ignored として archive。
    戻り値は遷移ラベル（遷移したとき）または None。
    """
    req = read_run_request(dirs.root)
    if req is None:
        return None
    cmd = str(req.get("command", ""))
    if cmd in LEARNER_COMMANDS:
        if state.phase in _DRAINABLE_PHASES:
            transition_run_state(
                dirs.root, expected_phase=state.phase, next_phase=RunPhase.DRAINING,
                config=run_config, message=f"run request: {cmd}",
            )
            archive_run_request(dirs.root, disposition="applied")
            return f"request:{cmd}->draining"
        if state.phase in (RunPhase.DRAINING, RunPhase.STOPPED, RunPhase.FAILED):
            archive_run_request(dirs.root, disposition="noop")
            return None
        # initializing / imitation_train: drainable phase 到達まで pending。
        return None
    # 未知 command（clear 含む）は state を壊さず ignored archive。
    archive_run_request(dirs.root, disposition="ignored")
    return None


def run_learner_supervisor(config: LearnerSupervisorConfig) -> dict[str, Any]:
    import time

    device = resolve_device(config.device)
    run_config = bootstrap_run(config)
    dirs = _LSDirs.from_root(config.run_root)
    dirs.ensure()
    lock = LearnerLock(config.run_root, force=config.force_unlock)
    lock.acquire()

    start = time.time()
    idle_polls = 0
    ppo_updates = 0
    ppo_ctx: _PPOContext | None = None
    transitions: list[str] = []
    exit_reason = "loop_end"

    try:
        # crash reconciliation（lock 取得後・phase loop 前）
        recon = reconcile_journal(dirs, run_config)
        if recon:
            transitions.append(f"reconcile:{recon}")

        while True:
            if (
                config.max_runtime_sec is not None
                and (time.time() - start) >= float(config.max_runtime_sec)
            ):
                exit_reason = "max_runtime"
                break
            state = read_run_state(state_path(config.run_root))

            # run request (drain/stop) を learner だけが state へ反映する。
            # reconcile は loop 前に済んでいるので、request 処理は journal 整合後。
            req_label = _handle_run_request(dirs, run_config, state)
            if req_label:
                transitions.append(req_label)
                continue

            phase = state.phase
            gen = int(state.phase_generation)

            if phase in (RunPhase.STOPPED, RunPhase.FAILED):
                exit_reason = phase.value
                break
            if phase == RunPhase.DRAINING:
                transition_run_state(
                    config.run_root, expected_phase=RunPhase.DRAINING,
                    next_phase=RunPhase.STOPPED, config=run_config,
                    message="learner drained",
                )
                transitions.append("draining->stopped")
                exit_reason = "stopped"
                break

            if phase == RunPhase.INITIALIZING:
                transition_run_state(
                    config.run_root, expected_phase=RunPhase.INITIALIZING,
                    next_phase=RunPhase.IMITATION_COLLECT, config=run_config,
                    message="begin imitation collect",
                )
                transitions.append("initializing->imitation_collect")
                continue

            if phase == RunPhase.IMITATION_COLLECT:
                collected = count_collect_games(dirs, gen)
                if collected >= int(run_config.imitation.target_games):
                    transition_run_state(
                        config.run_root, expected_phase=RunPhase.IMITATION_COLLECT,
                        next_phase=RunPhase.IMITATION_TRAIN, config=run_config,
                        message=f"collected {collected} games",
                    )
                    transitions.append("imitation_collect->imitation_train")
                    continue
                idle_polls += 1
                if config.max_idle_polls and idle_polls >= config.max_idle_polls:
                    exit_reason = "max_idle_polls"
                    break
                time.sleep(config.poll_interval_sec)
                continue

            if phase == RunPhase.IMITATION_TRAIN:
                run_imitation_train(
                    dirs, run_config, device=device, collect_generation=gen - 1,
                    phase=phase, phase_generation=gen,
                )
                transitions.append("imitation_train->ppo")
                idle_polls = 0
                continue

            if phase == RunPhase.PPO:
                entry = read_registry(dirs)
                if state.policy_version is not None and int(
                    state.policy_version
                ) != int(entry.policy_version):
                    raise RuntimeError(
                        f"state.policy_version {state.policy_version} != registry "
                        f"{entry.policy_version} (no journal; refuse to proceed)"
                    )
                if ppo_ctx is None:
                    model, optimizer, _had = load_model_and_optimizer(
                        dirs, entry, device, _make_ppo_config(run_config, device)
                    )
                    ppo_ctx = _PPOContext(model=model, optimizer=optimizer, entry=entry)
                else:
                    ppo_ctx.entry = entry
                res = run_ppo_update(
                    dirs, run_config, device=device, ppo_generation=gen,
                    ctx=ppo_ctx, consume_policy=config.consume_policy,
                )
                if res.get("need_more"):
                    idle_polls += 1
                    if config.max_idle_polls and idle_polls >= config.max_idle_polls:
                        exit_reason = "max_idle_polls"
                        break
                    time.sleep(config.poll_interval_sec)
                    continue
                ppo_updates += 1
                idle_polls = 0
                if (
                    config.stop_after_ppo_updates
                    and ppo_updates >= int(config.stop_after_ppo_updates)
                ):
                    exit_reason = "stop_after_ppo_updates"
                    break
                continue

            raise RuntimeError(f"unhandled phase: {phase.value}")
    except LearnerRecoveryError as exc:
        _fail_state(config.run_root, run_config, reason=str(exc))
        lock.release()
        raise
    except Exception as exc:  # noqa: BLE001
        # publish 後 (= journal が published/shards_consumed) の失敗は recoverable。
        # state を failed にせず journal を残し、再起動で reconcile させる。
        j = read_journal(dirs)
        if not (j and j.get("stage") in ("published", "shards_consumed")):
            _fail_state(config.run_root, run_config, reason=str(exc))
        lock.release()
        raise
    finally:
        lock.release()

    return {
        "ok": True, "run_root": str(dirs.root), "device": device,
        "transitions": transitions, "ppo_updates": ppo_updates,
        "exit_reason": exit_reason,
        "final_policy_version": (
            int(ppo_ctx.entry.policy_version) if ppo_ctx is not None else None
        ),
    }


def _fail_state(run_root: str, run_config: DistributedRunConfig, *, reason: str) -> None:
    try:
        state = read_run_state(state_path(run_root))
        transition_run_state(
            run_root, expected_phase=state.phase, next_phase=RunPhase.FAILED,
            config=run_config, failure_reason=reason, message="learner failed",
        )
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "LearnerSupervisorConfig",
    "LearnerLock",
    "LearnerRecoveryError",
    "bootstrap_run",
    "count_collect_games",
    "delete_journal",
    "finish_transaction",
    "journal_path",
    "load_model_and_optimizer",
    "publish_policy_checkpoint",
    "read_journal",
    "read_registry",
    "reconcile_journal",
    "resolve_device",
    "run_imitation_train",
    "run_learner_supervisor",
    "run_ppo_update",
    "scan_ppo_ready",
    "write_journal",
]
