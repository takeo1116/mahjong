"""Tests for the file-based distributed actor worker.

重い self-play を避けるため、registry / checkpoint load 系は tiny checkpoint で
unit test し、publish 経路は YON_IKKYOKU (single round) 1 game の最小
integration で確認する。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mahjong_agent.distributed.actor import (
    ActorConfig,
    _chunk_seed_ranges,
    load_policy,
    resolve_registry_entry,
    run_actor,
)
from mahjong_agent.distributed.manifest import (
    CheckpointRegistryEntry,
    read_json,
    sha256_file,
    write_json_atomic,
)
from mahjong_agent.encoders import PublicObservationEncoder
from mahjong_agent.models import Stage03Model, Stage03ModelConfig


def _make_checkpoint(run_root: Path, *, policy_version: int = 1) -> CheckpointRegistryEntry:
    """tiny Stage03 checkpoint + latest.json を run_root 下に作る。"""
    encoder = PublicObservationEncoder(enable_hints=True)
    cfg = Stage03ModelConfig.from_encoder_metadata(
        encoder.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16,
    )
    model = Stage03Model(cfg).eval()
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict

    ckpt_rel = f"checkpoints/policy_{policy_version:06d}.pt"
    ckpt_path = run_root / ckpt_rel
    torch.save(
        {
            "model_state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "model_config": asdict(model.config),
        },
        ckpt_path,
    )
    entry = CheckpointRegistryEntry(
        policy_version=policy_version,
        checkpoint_path=ckpt_rel,
        checkpoint_sha256=sha256_file(ckpt_path),
        created_at="2026-06-04T00:00:00Z",
        schema_version=2,
        observation_dim=int(cfg.observation_dim),
        candidate_dim=int(cfg.candidate_dim),
        model_config=asdict(model.config),
        encoder_metadata={
            "observation_dim": int(cfg.observation_dim),
            "candidate_dim": int(cfg.candidate_dim),
            "enable_hints": True,
        },
    )
    write_json_atomic(ckpt_dir / "latest.json", entry.to_dict())
    return entry


# ----------------------------------------------------------------------
# seed range helper
# ----------------------------------------------------------------------


def test_chunk_seed_ranges_even_split():
    r = _chunk_seed_ranges(seed_base=1000, num_games=4, chunk_games=2)
    assert r == [(1000, 1001), (1002, 1003)]


def test_chunk_seed_ranges_uneven_tail():
    r = _chunk_seed_ranges(seed_base=0, num_games=5, chunk_games=2)
    assert r == [(0, 1), (2, 3), (4, 4)]


# ----------------------------------------------------------------------
# registry / checkpoint load
# ----------------------------------------------------------------------


def test_actor_reads_latest_registry_and_validates_checkpoint_hash(tmp_path: Path):
    run_root = tmp_path / "run"
    entry = _make_checkpoint(run_root, policy_version=7)
    from mahjong_agent.distributed.actor import _RunDirs

    dirs = _RunDirs.from_root(run_root)
    loaded_entry = resolve_registry_entry(dirs, "latest")
    assert loaded_entry.policy_version == 7
    assert loaded_entry.checkpoint_sha256 == entry.checkpoint_sha256
    policy = load_policy(dirs, loaded_entry)
    assert int(policy.entry.policy_version) == 7
    assert int(policy.model.config.observation_dim) == int(entry.observation_dim)


def test_actor_rejects_checkpoint_hash_mismatch(tmp_path: Path):
    run_root = tmp_path / "run"
    _make_checkpoint(run_root, policy_version=1)
    # latest.json の hash を壊す
    latest = run_root / "checkpoints" / "latest.json"
    payload = read_json(latest)
    payload["checkpoint_sha256"] = "0" * 64
    write_json_atomic(latest, payload)

    from mahjong_agent.distributed.actor import _RunDirs

    dirs = _RunDirs.from_root(run_root)
    entry = resolve_registry_entry(dirs, "latest")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_policy(dirs, entry)


def test_actor_missing_registry_fail_fast(tmp_path: Path):
    from mahjong_agent.distributed.actor import _RunDirs

    dirs = _RunDirs.from_root(tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        resolve_registry_entry(dirs, "latest")


def test_actor_dry_run_loads_without_publishing(tmp_path: Path):
    run_root = tmp_path / "run"
    _make_checkpoint(run_root, policy_version=3)
    summary = run_actor(
        ActorConfig(
            run_root=str(run_root),
            actor_id="actor_dry",
            num_games=2,
            chunk_games=1,
            dry_run=True,
        )
    )
    assert summary["dry_run"] is True
    assert summary["policy_version"] == 3
    # publish していない
    ready = run_root / "rollouts" / "ready"
    assert not ready.exists() or not any(ready.iterdir())


# ----------------------------------------------------------------------
# publish integration (tiny YON_IKKYOKU rollout)
# ----------------------------------------------------------------------


def _run_publish(tmp_path: Path, *, actor_id="actor_0001") -> tuple[Path, dict]:
    run_root = tmp_path / "run"
    _make_checkpoint(run_root, policy_version=5)
    summary = run_actor(
        ActorConfig(
            run_root=str(run_root),
            actor_id=actor_id,
            num_games=1,
            chunk_games=1,
            seed_base=1234500,
            game_type_name="YON_IKKYOKU",
            max_steps_per_game=2000,
        )
    )
    return run_root, summary


def test_actor_publishes_ready_directory_after_pending_write(tmp_path: Path):
    run_root, summary = _run_publish(tmp_path)
    assert summary["published_shards"] == 1
    assert summary["crash_chunks"] == 0

    ready = run_root / "rollouts" / "ready"
    pending = run_root / "rollouts" / "pending"
    shard_dirs = list(ready.iterdir())
    assert len(shard_dirs) == 1
    shard_dir = shard_dirs[0]
    assert (shard_dir / "shard.npz").is_file()
    assert (shard_dir / "manifest.json").is_file()
    # pending は空 (atomic rename 後)
    assert list(pending.iterdir()) == []


def test_actor_manifest_matches_shard_dims(tmp_path: Path):
    from mahjong_agent.data import read_shard_metadata

    run_root, _ = _run_publish(tmp_path)
    shard_dir = next((run_root / "rollouts" / "ready").iterdir())
    manifest = read_json(shard_dir / "manifest.json")
    shard_meta = read_shard_metadata(shard_dir / "shard.npz")

    assert manifest["schema_version"] == shard_meta["schema_version"] == 2
    assert manifest["observation_dim"] == shard_meta["observation_dim"]
    assert manifest["candidate_dim"] == shard_meta["candidate_dim"]
    assert manifest["num_samples"] == shard_meta["num_samples"]
    assert manifest["policy_version"] == 5
    assert manifest["old_log_prob_available"] is True
    assert manifest["actor_id"] == "actor_0001"
    assert manifest["seed_start"] == 1234500
    assert manifest["seed_end"] == 1234500
    assert "checkpoint_sha256" in manifest and manifest["checkpoint_sha256"]


def test_actor_metrics_jsonl_has_shard_published_row(tmp_path: Path):
    import json

    run_root, summary = _run_publish(tmp_path, actor_id="actor_m")
    metrics = run_root / "metrics" / "actors" / "actor_m.jsonl"
    assert metrics.is_file()
    rows = [json.loads(ln) for ln in metrics.read_text().splitlines() if ln.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "shard_published"
    assert row["actor_id"] == "actor_m"
    assert row["policy_version"] == 5
    assert row["num_samples"] > 0
    assert row["games_per_sec"] >= 0.0


def test_actor_generated_shard_is_readable(tmp_path: Path):
    from mahjong_agent.data import read_decision_shard

    run_root, _ = _run_publish(tmp_path)
    shard_dir = next((run_root / "rollouts" / "ready").iterdir())
    samples = read_decision_shard(shard_dir / "shard.npz")
    assert len(samples) > 0
    # 全 sample が policy actor 由来 + observation は public feature のみ
    assert all(str(s.actor_type) == "policy" for s in samples)
    obs_dim = int(samples[0].observation.shape[0])
    assert obs_dim == int(samples[0].observation.reshape(-1).shape[0])


# ----------------------------------------------------------------------
# hidden-info guard
# ----------------------------------------------------------------------


def test_actor_source_does_not_reference_hidden_state():
    import mahjong_agent.distributed.actor as actor_mod

    text = Path(actor_mod.__file__).read_text(encoding="utf-8")
    for tok in (
        "env.hands",
        "env.wall",
        "env.state",
        "obs.hands[",
        "full_state",
        "private_hand",
        "mjai_log",
    ):
        assert tok not in text, f"actor source references {tok!r}"
