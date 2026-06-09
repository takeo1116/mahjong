"""Manifest / registry dataclasses + atomic file helpers for the file-based
distributed actor / learner setup.

責務:
- ``CheckpointRegistryEntry``: ``checkpoints/latest.json`` の中身。learner が
  publish し actor が polling する現行 policy のポインタ。
- ``RolloutShardManifest``: ``rollouts/ready/<id>/manifest.json`` の中身。
  actor が生成した shard の policy_version / dim / hash 等のメタ。
- ``write_json_atomic`` / ``read_json`` / ``sha256_file``: partial read を
  避ける atomic 書き込みと整合性検証。
- ``is_manifest_compatible``: learner が ready shard を採用してよいかの判定
  (schema / dim / policy lag / old_log_prob)。理由文字列付きで返す。

hidden info は扱わない。dim / version / hash / id といったメタのみ。
全体設計は ``docs/distributed_actor_learner.md`` 参照。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

MANIFEST_VERSION: int = 1


class ShardState(str, Enum):
    """rollout shard の lifecycle 状態 (= directory 名に対応)。"""

    PENDING = "pending"     # actor が書き込み中。learner は触らない。
    READY = "ready"         # actor が書き終えた。learner が消費してよい。
    CONSUMED = "consumed"   # learner が採用・使用済み。
    REJECTED = "rejected"   # stale / incompatible / corrupted で未使用。


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointRegistryEntry:
    """``checkpoints/latest.json`` の 1 エントリ (現行 policy のポインタ)。

    ``checkpoint_path`` は run ディレクトリからの相対パス (マシン間で絶対パスを
    共有しないため)。``model_config`` / ``encoder_metadata`` は actor が encoder
    / model を再構築し、shard の dim が learner と一致することを保証するために
    含める。
    """

    policy_version: int
    checkpoint_path: str
    checkpoint_sha256: str
    created_at: str
    schema_version: int
    observation_dim: int
    candidate_dim: int
    model_config: dict[str, Any] | None = None
    encoder_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckpointRegistryEntry:
        return cls(
            policy_version=int(d["policy_version"]),
            checkpoint_path=str(d["checkpoint_path"]),
            checkpoint_sha256=str(d["checkpoint_sha256"]),
            created_at=str(d["created_at"]),
            schema_version=int(d["schema_version"]),
            observation_dim=int(d["observation_dim"]),
            candidate_dim=int(d["candidate_dim"]),
            model_config=d.get("model_config"),
            encoder_metadata=d.get("encoder_metadata"),
        )


@dataclass(frozen=True)
class RolloutShardManifest:
    """``rollouts/ready/<id>/manifest.json`` の中身。

    ``old_log_prob_available`` は PPO ratio 用 ``old_log_prob`` を sample が持つ
    か。learner はこれが False の shard を policy gradient に使わない。
    ``shard_path`` は manifest と同一ディレクトリからの相対 (= ``"shard.npz"``)。
    """

    policy_version: int
    checkpoint_sha256: str
    actor_id: str
    created_at: str
    num_games: int
    num_samples: int
    schema_version: int
    observation_dim: int
    candidate_dim: int
    shard_path: str = "shard.npz"
    seed_start: int | None = None
    seed_end: int | None = None
    hostname: str | None = None
    pid: int | None = None
    old_log_prob_available: bool = True
    manifest_version: int = MANIFEST_VERSION
    role: str = "actor_rollout"
    # phase-aware supervisor で記録する識別情報。
    # 後方互換のため optional (旧 manifest は None で読める)。
    phase: str | None = None
    phase_generation: int | None = None
    worker_id: str | None = None
    chunk_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RolloutShardManifest:
        return cls(
            policy_version=int(d["policy_version"]),
            checkpoint_sha256=str(d["checkpoint_sha256"]),
            actor_id=str(d["actor_id"]),
            created_at=str(d["created_at"]),
            num_games=int(d["num_games"]),
            num_samples=int(d["num_samples"]),
            schema_version=int(d["schema_version"]),
            observation_dim=int(d["observation_dim"]),
            candidate_dim=int(d["candidate_dim"]),
            shard_path=str(d.get("shard_path", "shard.npz")),
            seed_start=(
                int(d["seed_start"]) if d.get("seed_start") is not None else None
            ),
            seed_end=(
                int(d["seed_end"]) if d.get("seed_end") is not None else None
            ),
            hostname=d.get("hostname"),
            pid=(int(d["pid"]) if d.get("pid") is not None else None),
            old_log_prob_available=bool(d.get("old_log_prob_available", True)),
            manifest_version=int(d.get("manifest_version", MANIFEST_VERSION)),
            role=str(d.get("role", "actor_rollout")),
            phase=(str(d["phase"]) if d.get("phase") is not None else None),
            phase_generation=(
                int(d["phase_generation"])
                if d.get("phase_generation") is not None
                else None
            ),
            worker_id=(str(d["worker_id"]) if d.get("worker_id") is not None else None),
            chunk_index=(
                int(d["chunk_index"]) if d.get("chunk_index") is not None else None
            ),
        )


@dataclass(frozen=True)
class CompatResult:
    """``is_manifest_compatible`` の判定結果。

    ``ok`` が False のとき ``reason`` に理由 (learner metrics の
    ``rejected_reason_counts`` 用) が入る。``ok`` が True のとき ``reason`` は
    ``"ok"``。
    """

    ok: bool
    reason: str

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------------
# atomic file helpers
# ---------------------------------------------------------------------------


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """``payload`` を ``path`` に **atomic** に書く (tmp + ``os.replace``)。

    同一 filesystem 上で ``os.replace`` は atomic。reader は常に「完全な旧版」か
    「完全な新版」のどちらかを読み、partial write を観測しない。tmp は同一
    ディレクトリに作る (cross-device rename を避けるため)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: str | Path) -> dict[str, Any]:
    """JSON file を dict として読む。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """file の SHA-256 hex digest を返す (checkpoint / shard 整合性検証用)。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# compatibility check (learner consumption rule)
# ---------------------------------------------------------------------------


def is_manifest_compatible(
    manifest: RolloutShardManifest,
    *,
    schema_version: int,
    observation_dim: int,
    candidate_dim: int,
    current_policy_version: int,
    max_policy_lag: int,
    require_old_log_prob: bool = True,
) -> CompatResult:
    """learner が ready shard を採用してよいかを判定する。

    採用条件 (全て満たす):
      1. ``schema_version`` 一致。
      2. ``observation_dim`` / ``candidate_dim`` 一致。
      3. ``manifest.policy_version >= current_policy_version - max_policy_lag``
         (= policy staleness が許容 lag 内)。未来 version
         (``> current_policy_version``) も整合性上は許容する (learner より新しい
         actor checkpoint は通常起きないが、起きても reject しない)。
      4. ``require_old_log_prob`` のとき ``old_log_prob_available`` が True。

    Returns
    -------
    CompatResult:
        ``ok`` と ``reason``。``reason`` は不採用時に learner metrics へ記録する。
    """
    if int(manifest.schema_version) != int(schema_version):
        return CompatResult(
            False,
            f"schema_mismatch(manifest={manifest.schema_version},"
            f"learner={schema_version})",
        )
    if int(manifest.observation_dim) != int(observation_dim):
        return CompatResult(
            False,
            f"observation_dim_mismatch(manifest={manifest.observation_dim},"
            f"learner={observation_dim})",
        )
    if int(manifest.candidate_dim) != int(candidate_dim):
        return CompatResult(
            False,
            f"candidate_dim_mismatch(manifest={manifest.candidate_dim},"
            f"learner={candidate_dim})",
        )
    min_allowed = int(current_policy_version) - int(max_policy_lag)
    if int(manifest.policy_version) < min_allowed:
        return CompatResult(
            False,
            f"stale_policy(manifest={manifest.policy_version},"
            f"min_allowed={min_allowed},max_lag={max_policy_lag})",
        )
    if require_old_log_prob and not bool(manifest.old_log_prob_available):
        return CompatResult(False, "old_log_prob_unavailable")
    return CompatResult(True, "ok")


__all__ = [
    "MANIFEST_VERSION",
    "CheckpointRegistryEntry",
    "CompatResult",
    "RolloutShardManifest",
    "ShardState",
    "is_manifest_compatible",
    "read_json",
    "sha256_file",
    "write_json_atomic",
]
