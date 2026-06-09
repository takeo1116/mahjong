"""File-based distributed learner consumer (GPU PPO updater + checkpoint publisher).

learner は CPU actor が ``rollouts/ready/`` に publish した rollout shard を拾い、
policy lag が許容範囲内の shard だけで PPO update を行い、新 checkpoint を
``checkpoints/`` へ atomic に publish する。``checkpoints/latest.json`` を書くのは
learner だけ (single-writer)。

1 update のライフサイクル:

1. ``checkpoints/latest.json`` を ``CheckpointRegistryEntry`` として読む
   (current policy)。
2. ``rollouts/ready/*/manifest.json`` を scan し、``is_manifest_compatible`` で
   採用可否を判定 (schema / dim / policy lag / old_log_prob)。incompatible は
   ``rollouts/rejected/<reason>/`` へ move。
3. compatible shard を ``max_samples_per_update`` まで集めて読む。
   ``min_samples_per_update`` 未満なら poll 待ち (dry-run は summary を返す)。
4. current checkpoint から model を復元 → ``compute_returns_and_advantages`` +
   ``fit_ppo`` で 1 phase update。
5. ``policy_version+1`` で checkpoint を tmp+rename で publish し、``latest.json``
   を ``write_json_atomic`` で更新。
6. 使用済み shard を ``rollouts/consumed/policy_NNNNNN/`` へ move。

failure ordering: PPO update → checkpoint publish → consume の順なので、update /
publish が失敗した時点では ready shard は移動されず、partial checkpoint /
壊れた latest.json も残らない。

hidden info は扱わない (manifest は dim / version / hash のメタのみ、sample は
public observation feature のみ)。全体設計は ``docs/distributed_actor_learner.md``。
evaluator / best checkpoint selection は本 issue の範囲外。
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mahjong_agent.data import read_decision_shard
from mahjong_agent.data.types import SCHEMA_VERSION
from mahjong_agent.distributed.manifest import (
    CheckpointRegistryEntry,
    RolloutShardManifest,
    is_manifest_compatible,
    read_json,
    sha256_file,
    write_json_atomic,
)
from mahjong_agent.models import Stage03Model, Stage03ModelConfig
from mahjong_agent.training import (
    PPOConfig,
    compute_returns_and_advantages,
    fit_ppo,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_device(name: str | None) -> str:
    """device 名を解決する (None なら cuda が使えれば cuda、無ければ cpu)。"""
    if name:
        return str(name)
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnerConfig:
    """1 回の learner invocation の設定。"""

    run_root: str
    device: str | None = None
    min_samples_per_update: int = 100_000
    max_samples_per_update: int = 250_000
    max_policy_lag: int = 1
    ppo_lr: float = 5e-4
    ppo_epochs: int = 1
    target_kl: float = 0.01
    batch_size: int = 256
    poll_interval_sec: float = 5.0
    stop_after_updates: int = 1
    consume_policy: str = "move"  # "move" | "delete"
    exclude_post_riichi_discards: bool = True
    fail_on_target_kl: bool = False
    dry_run: bool = False
    metrics_jsonl: str | None = None
    seed: int = 42
    max_poll_iterations: int = 0  # 0 = unlimited (CLI 用); test は enough を渡す


# ---------------------------------------------------------------------------
# directory layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RunDirs:
    root: Path
    checkpoints: Path
    ready: Path
    consumed: Path
    rejected: Path
    metrics: Path

    @classmethod
    def from_root(cls, run_root: str | Path) -> _RunDirs:
        root = Path(run_root)
        return cls(
            root=root,
            checkpoints=root / "checkpoints",
            ready=root / "rollouts" / "ready",
            consumed=root / "rollouts" / "consumed",
            rejected=root / "rollouts" / "rejected",
            metrics=root / "metrics",
        )

    def ensure(self) -> None:
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.ready.mkdir(parents=True, exist_ok=True)
        self.consumed.mkdir(parents=True, exist_ok=True)
        self.rejected.mkdir(parents=True, exist_ok=True)
        self.metrics.mkdir(parents=True, exist_ok=True)

    @property
    def latest_json(self) -> Path:
        return self.checkpoints / "latest.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _reason_key(reason: str) -> str:
    """CompatResult.reason (例 ``stale_policy(...)``) を directory 名に縮める。"""
    return str(reason).split("(", 1)[0] or "rejected"


def _unique_dir(parent: Path, name: str) -> Path:
    """``parent/name`` が衝突したら suffix を足して回避する。"""
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / name
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        alt = parent / f"{name}_dup{i}"
        if not alt.exists():
            return alt
        i += 1


def read_registry(dirs: _RunDirs) -> CheckpointRegistryEntry:
    if not dirs.latest_json.is_file():
        raise FileNotFoundError(
            f"checkpoint registry not found: {dirs.latest_json} "
            "(learner には初期 checkpoint / latest.json が必要)"
        )
    return CheckpointRegistryEntry.from_dict(read_json(dirs.latest_json))


def load_model(dirs: _RunDirs, entry: CheckpointRegistryEntry, device: str) -> Stage03Model:
    """registry entry の checkpoint を hash 検証してロードする。"""
    ckpt = (dirs.root / entry.checkpoint_path).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {ckpt}")
    actual_sha = sha256_file(ckpt)
    if actual_sha != entry.checkpoint_sha256:
        raise ValueError(
            f"checkpoint sha256 mismatch for {ckpt}: "
            f"registry={entry.checkpoint_sha256} actual={actual_sha}"
        )
    payload = torch.load(ckpt, map_location="cpu")
    if "model_config" not in payload or "model_state_dict" not in payload:
        raise ValueError(f"invalid checkpoint payload: {ckpt}")
    model = Stage03Model(Stage03ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device))
    return model


@dataclass
class _ShardEntry:
    shard_dir: Path
    manifest: RolloutShardManifest


@dataclass
class _RejectEntry:
    shard_dir: Path
    reason: str


@dataclass
class _ScanResult:
    compatible: list[_ShardEntry] = field(default_factory=list)
    rejects: list[_RejectEntry] = field(default_factory=list)
    future_policy_version_count: int = 0


def scan_ready(dirs: _RunDirs, entry: CheckpointRegistryEntry, config: LearnerConfig) -> _ScanResult:
    """``rollouts/ready/`` を scan し compatible / reject に分類する。

    ``rollouts/pending/`` は **絶対に読まない** (ready のみ走査する)。
    """
    res = _ScanResult()
    if not dirs.ready.is_dir():
        return res
    for shard_dir in sorted(p for p in dirs.ready.iterdir() if p.is_dir()):
        manifest_path = shard_dir / "manifest.json"
        if not manifest_path.is_file():
            res.rejects.append(_RejectEntry(shard_dir, "manifest_missing"))
            continue
        try:
            manifest = RolloutShardManifest.from_dict(read_json(manifest_path))
        except Exception:  # noqa: BLE001
            res.rejects.append(_RejectEntry(shard_dir, "manifest_parse_error"))
            continue
        shard_file = shard_dir / manifest.shard_path
        if not shard_file.is_file():
            res.rejects.append(_RejectEntry(shard_dir, "missing_shard"))
            continue
        compat = is_manifest_compatible(
            manifest,
            schema_version=int(entry.schema_version),
            observation_dim=int(entry.observation_dim),
            candidate_dim=int(entry.candidate_dim),
            current_policy_version=int(entry.policy_version),
            max_policy_lag=int(config.max_policy_lag),
        )
        if not compat.ok:
            res.rejects.append(_RejectEntry(shard_dir, _reason_key(compat.reason)))
            continue
        if int(manifest.policy_version) > int(entry.policy_version):
            res.future_policy_version_count += 1
        res.compatible.append(_ShardEntry(shard_dir, manifest))
    return res


def _move_reject(dirs: _RunDirs, rej: _RejectEntry) -> None:
    dest_parent = dirs.rejected / rej.reason
    dest = _unique_dir(dest_parent, rej.shard_dir.name)
    os.replace(rej.shard_dir, dest)


def _consume_shard(dirs: _RunDirs, shard_dir: Path, new_version: int, *, policy: str) -> str | None:
    """使用済み shard を consumed へ move (or delete)。consumed 先 path を返す。"""
    if policy == "delete":
        shutil.rmtree(shard_dir, ignore_errors=True)
        return None
    dest_parent = dirs.consumed / f"policy_{new_version:06d}"
    dest = _unique_dir(dest_parent, shard_dir.name)
    os.replace(shard_dir, dest)
    return str(dest)


# ---------------------------------------------------------------------------
# checkpoint publish
# ---------------------------------------------------------------------------


def publish_checkpoint(
    dirs: _RunDirs,
    *,
    model: Stage03Model,
    new_version: int,
    prev_entry: CheckpointRegistryEntry,
    ppo_config: PPOConfig,
    learner_metrics: dict[str, Any],
    source_policy_versions: list[int],
    num_samples: int,
) -> CheckpointRegistryEntry:
    """新 checkpoint を atomic publish し、registry entry を返す。

    1. ``policy_NNNNNN.pt.tmp`` に保存 → fsync → ``os.replace``。
    2. sha256 を計算。
    3. ``latest.json`` を ``write_json_atomic`` で更新 (= tmp + os.replace)。
    """
    rel = f"checkpoints/policy_{new_version:06d}.pt"
    ckpt_path = dirs.root / rel
    tmp_path = ckpt_path.with_name(ckpt_path.name + ".tmp")
    model_config = asdict(model.config)
    payload = {
        "model_state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        },
        "model_config": model_config,
        "policy_version": int(new_version),
        "ppo_config": asdict(ppo_config),
        "learner_metrics": learner_metrics,
        "source_policy_versions": [int(v) for v in source_policy_versions],
        "num_samples": int(num_samples),
        "created_at": _utc_now_iso(),
    }
    with tmp_path.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, ckpt_path)

    entry = CheckpointRegistryEntry(
        policy_version=int(new_version),
        checkpoint_path=rel,
        checkpoint_sha256=sha256_file(ckpt_path),
        created_at=_utc_now_iso(),
        schema_version=SCHEMA_VERSION,
        observation_dim=int(model_config["observation_dim"]),
        candidate_dim=int(model_config["candidate_dim"]),
        model_config=model_config,
        encoder_metadata=prev_entry.encoder_metadata,
    )
    write_json_atomic(dirs.latest_json, entry.to_dict())
    return entry


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _metrics_path(dirs: _RunDirs, config: LearnerConfig) -> Path:
    if config.metrics_jsonl:
        return Path(config.metrics_jsonl)
    return dirs.metrics / "learner.jsonl"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _reject_counts(rejects: list[_RejectEntry]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rejects:
        out[r.reason] = out.get(r.reason, 0) + 1
    return out


# ---------------------------------------------------------------------------
# sample collection
# ---------------------------------------------------------------------------


@dataclass
class _Collected:
    used: list[_ShardEntry] = field(default_factory=list)
    samples: list[Any] = field(default_factory=list)
    corrupt: list[_RejectEntry] = field(default_factory=list)


def _collect_samples(scan: _ScanResult, config: LearnerConfig) -> _Collected:
    """compatible shard を ``max_samples_per_update`` まで読み集める。

    shard 単位でしか consume しない (partial consume しない)。read 失敗や
    manifest との num_samples 不一致は corrupt として弾く。
    """
    out = _Collected()
    for entry in scan.compatible:
        if out.samples and len(out.samples) >= int(config.max_samples_per_update):
            break
        shard_file = entry.shard_dir / entry.manifest.shard_path
        try:
            shard_samples = read_decision_shard(shard_file)
        except Exception:  # noqa: BLE001
            out.corrupt.append(_RejectEntry(entry.shard_dir, "corrupt_shard"))
            continue
        if len(shard_samples) != int(entry.manifest.num_samples):
            out.corrupt.append(
                _RejectEntry(entry.shard_dir, "num_samples_mismatch")
            )
            continue
        out.used.append(entry)
        out.samples.extend(shard_samples)
    return out


# ---------------------------------------------------------------------------
# top-level run
# ---------------------------------------------------------------------------


def _make_ppo_config(config: LearnerConfig, device: str) -> PPOConfig:
    return PPOConfig(
        learning_rate=float(config.ppo_lr),
        batch_size=int(config.batch_size),
        num_epochs=int(config.ppo_epochs),
        device=device,
        target_kl_enabled=True,
        target_kl=float(config.target_kl),
        target_kl_stop_multiplier=1.5,
        target_kl_skip_minibatch_on_exceed=True,
        exclude_post_riichi_discards=bool(config.exclude_post_riichi_discards),
        value_loss_includes_excluded=False,
        per_player_round_weighting=False,
        include_actor_types=("policy",),
        seed=int(config.seed),
    )


def _do_update(
    dirs: _RunDirs,
    config: LearnerConfig,
    device: str,
    metrics_path: Path,
) -> dict[str, Any]:
    """1 update を実行して row を返す (update 不可なら ``need_more`` を立てる)。"""
    entry = read_registry(dirs)
    scan = scan_ready(dirs, entry, config)

    if config.dry_run:
        collectable = sum(int(e.manifest.num_samples) for e in scan.compatible)
        row = {
            "event": "learner_dry_run",
            "policy_version": int(entry.policy_version),
            "compatible_shards": len(scan.compatible),
            "compatible_samples": int(collectable),
            "rejected_counts": _reject_counts(scan.rejects),
            "future_policy_version_count": int(scan.future_policy_version_count),
            "device": device,
            "created_at": _utc_now_iso(),
        }
        return {"row": row, "done": True, "updated": False}

    # incompatible shard は (再 scan を避けるため) 即 reject move する。
    for rej in scan.rejects:
        _move_reject(dirs, rej)

    collected = _collect_samples(scan, config)
    for rej in collected.corrupt:
        _move_reject(dirs, rej)

    if len(collected.samples) < int(config.min_samples_per_update):
        return {
            "need_more": True,
            "available": len(collected.samples),
            "rejected_counts": _reject_counts(scan.rejects + collected.corrupt),
        }

    start = time.time()
    model = load_model(dirs, entry, device)
    ppo_config = _make_ppo_config(config, device)

    ppo_data = compute_returns_and_advantages(collected.samples, ppo_config)
    eligible = int(ppo_data.eligible.sum())
    if eligible <= 0:
        raise RuntimeError("learner update: PPO eligible sample count is zero")

    result = fit_ppo(model, collected.samples, ppo_config)
    final = result.final
    if final is None or int(final.num_updates) <= 0:
        raise RuntimeError(
            f"learner update: PPO produced no updates (early_stopped="
            f"{result.early_stopped})"
        )
    if bool(result.early_stopped) and bool(config.fail_on_target_kl):
        raise RuntimeError(
            "learner update: PPO target_kl early stop and --fail-on-target-kl set"
        )

    new_version = int(entry.policy_version) + 1
    source_versions = sorted(
        {int(e.manifest.policy_version) for e in collected.used}
    )
    ppo_final = final.to_dict()
    elapsed = round(time.time() - start, 3)
    learner_metrics = {
        "num_shards": len(collected.used),
        "num_samples": len(collected.samples),
        "eligible": eligible,
        "source_policy_versions": source_versions,
        "ppo_final": ppo_final,
        "early_stopped": bool(result.early_stopped),
    }

    # publish checkpoint (atomic)。publish 成功後にのみ consume する。
    new_entry = publish_checkpoint(
        dirs,
        model=model,
        new_version=new_version,
        prev_entry=entry,
        ppo_config=ppo_config,
        learner_metrics=learner_metrics,
        source_policy_versions=source_versions,
        num_samples=len(collected.samples),
    )

    consumed_paths: list[str] = []
    for e in collected.used:
        dest = _consume_shard(
            dirs, e.shard_dir, new_version, policy=str(config.consume_policy)
        )
        if dest is not None:
            consumed_paths.append(dest)

    elapsed_total = max(time.time() - start, 1e-9)
    row = {
        "event": "learner_update",
        "policy_version_before": int(entry.policy_version),
        "policy_version_after": int(new_version),
        "num_shards": len(collected.used),
        "num_samples": len(collected.samples),
        "eligible": eligible,
        "source_policy_versions": source_versions,
        "rejected_counts": _reject_counts(scan.rejects + collected.corrupt),
        "future_policy_version_count": int(scan.future_policy_version_count),
        "elapsed_sec": elapsed,
        "samples_per_sec": round(len(collected.samples) / elapsed_total, 4),
        "device": device,
        "loss": ppo_final.get("loss"),
        "policy_loss": ppo_final.get("policy_loss"),
        "value_loss": ppo_final.get("value_loss"),
        "entropy": ppo_final.get("entropy"),
        "approx_kl_mean": ppo_final.get("approx_kl_mean"),
        "approx_kl_max": ppo_final.get("approx_kl_max"),
        "clip_fraction": ppo_final.get("clip_fraction"),
        "target_kl_checked_minibatches": ppo_final.get(
            "target_kl_checked_minibatches"
        ),
        "target_kl_applied_minibatches": ppo_final.get(
            "target_kl_applied_minibatches"
        ),
        "target_kl_skipped_minibatches": ppo_final.get(
            "target_kl_skipped_minibatches"
        ),
        "num_updates": ppo_final.get("num_updates"),
        "early_stopped": bool(result.early_stopped),
        "checkpoint_path": new_entry.checkpoint_path,
        "checkpoint_sha256": new_entry.checkpoint_sha256,
        "created_at": _utc_now_iso(),
    }
    return {"row": row, "done": False, "updated": True}


def run_learner(config: LearnerConfig) -> dict[str, Any]:
    """learner を実行する。``stop_after_updates`` 回 update したら終了。

    dry-run のときは 1 回 scan して summary を返す (update / publish / consume
    をしない)。
    """
    device = resolve_device(config.device)
    dirs = _RunDirs.from_root(config.run_root)
    dirs.ensure()
    metrics_path = _metrics_path(dirs, config)

    if config.dry_run:
        out = _do_update(dirs, config, device, metrics_path)
        _append_jsonl(metrics_path, out["row"])
        return {
            "ok": True,
            "dry_run": True,
            "run_root": str(dirs.root),
            "device": device,
            "metrics_jsonl": str(metrics_path),
            **out["row"],
        }

    updates: list[dict[str, Any]] = []
    poll_iters = 0
    final_policy_version: int | None = None
    while len(updates) < int(config.stop_after_updates):
        out = _do_update(dirs, config, device, metrics_path)
        if out.get("need_more"):
            poll_iters += 1
            if (
                int(config.max_poll_iterations) > 0
                and poll_iters >= int(config.max_poll_iterations)
            ):
                break
            time.sleep(float(config.poll_interval_sec))
            continue
        row = out["row"]
        _append_jsonl(metrics_path, row)
        updates.append(row)
        final_policy_version = int(row["policy_version_after"])

    return {
        "ok": True,
        "dry_run": False,
        "run_root": str(dirs.root),
        "device": device,
        "updates_done": len(updates),
        "final_policy_version": final_policy_version,
        "poll_iterations": poll_iters,
        "metrics_jsonl": str(metrics_path),
        "updates": updates,
    }


__all__ = [
    "LearnerConfig",
    "load_model",
    "publish_checkpoint",
    "read_registry",
    "resolve_device",
    "run_learner",
    "scan_ready",
]
