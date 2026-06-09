"""File-based distributed actor / learner helpers.

CPU actor 群 + GPU learner の file-based 構成 (shared filesystem + manifest +
atomic rename) のための薄い helper を提供する。actor / learner CLI 本体は
後続で実装する。本 package は manifest / registry の dataclass と、atomic な
JSON 書き込み・shard 互換判定などの純粋 helper に限定する。

設計の全体像は ``docs/distributed_actor_learner.md`` 参照。hidden info は扱わ
ない (manifest は dim / version / hash のメタのみ)。
"""
from __future__ import annotations

from mahjong_agent.distributed.manifest import (
    CheckpointRegistryEntry,
    CompatResult,
    RolloutShardManifest,
    ShardState,
    is_manifest_compatible,
    read_json,
    sha256_file,
    write_json_atomic,
)

__all__ = [
    "CheckpointRegistryEntry",
    "CompatResult",
    "RolloutShardManifest",
    "ShardState",
    "is_manifest_compatible",
    "read_json",
    "sha256_file",
    "write_json_atomic",
]
