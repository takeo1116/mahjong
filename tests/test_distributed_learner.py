"""Tests for the file-based distributed learner consumer.

scan / compatibility は hand-crafted manifest で軽く検証し、PPO update /
checkpoint publish / consume は actor で作った tiny ready shard
(YON_IKKYOKU 2 game) を 1 update 回して確認する。
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from mahjong_agent.distributed import learner as learner_mod
from mahjong_agent.distributed.actor import ActorConfig, run_actor
from mahjong_agent.distributed.learner import (
    LearnerConfig,
    _RunDirs,
    run_learner,
    scan_ready,
)
from mahjong_agent.distributed.manifest import (
    CheckpointRegistryEntry,
    RolloutShardManifest,
    read_json,
    sha256_file,
    write_json_atomic,
)
from mahjong_agent.encoders import PublicObservationEncoder
from mahjong_agent.models import Stage03Model, Stage03ModelConfig


def _build_checkpoint(run_root: Path, *, policy_version: int = 1) -> CheckpointRegistryEntry:
    encoder = PublicObservationEncoder(enable_hints=True)
    cfg = Stage03ModelConfig.from_encoder_metadata(
        encoder.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16,
    )
    model = Stage03Model(cfg).eval()
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rel = f"checkpoints/policy_{policy_version:06d}.pt"
    ckpt_path = run_root / rel
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
        checkpoint_path=rel,
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


def _actor_fill(run_root: Path, *, num_games: int = 2) -> dict:
    return run_actor(
        ActorConfig(
            run_root=str(run_root),
            actor_id="actor_t",
            num_games=num_games,
            chunk_games=1,
            seed_base=1234500,
            game_type_name="YON_IKKYOKU",
            max_steps_per_game=2000,
        )
    )


def _update_config(run_root: Path, **kw) -> LearnerConfig:
    base = dict(
        run_root=str(run_root),
        device="cpu",
        min_samples_per_update=1,
        max_samples_per_update=10_000,
        max_policy_lag=1,
        ppo_lr=5e-4,
        ppo_epochs=1,
        target_kl=1.0,  # early stop しないよう大きめ
        batch_size=256,
        stop_after_updates=1,
        max_poll_iterations=1,
    )
    base.update(kw)
    return LearnerConfig(**base)


def _write_crafted_shard(
    ready: Path,
    shard_id: str,
    *,
    policy_version: int,
    schema_version: int = 2,
    observation_dim: int = 608,
    candidate_dim: int = 91,
    old_log_prob: bool = True,
) -> None:
    d = ready / shard_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "shard.npz").write_bytes(b"dummy")
    m = RolloutShardManifest(
        policy_version=policy_version,
        checkpoint_sha256="x",
        actor_id="a",
        created_at="t",
        num_games=1,
        num_samples=1,
        schema_version=schema_version,
        observation_dim=observation_dim,
        candidate_dim=candidate_dim,
        old_log_prob_available=old_log_prob,
    )
    write_json_atomic(d / "manifest.json", m.to_dict())


# ----------------------------------------------------------------------
# scan / compatibility
# ----------------------------------------------------------------------


def test_learner_scans_ready_and_rejects_incompatible(tmp_path: Path):
    run_root = tmp_path / "run"
    entry = _build_checkpoint(run_root, policy_version=10)
    dirs = _RunDirs.from_root(run_root)
    dirs.ensure()
    _write_crafted_shard(dirs.ready, "ok", policy_version=10)
    _write_crafted_shard(dirs.ready, "stale", policy_version=8)
    _write_crafted_shard(dirs.ready, "dim", policy_version=10, observation_dim=506)
    _write_crafted_shard(dirs.ready, "schema", policy_version=10, schema_version=1)
    _write_crafted_shard(dirs.ready, "nolp", policy_version=10, old_log_prob=False)

    scan = scan_ready(dirs, entry, _update_config(run_root))
    assert len(scan.compatible) == 1
    assert scan.compatible[0].shard_dir.name == "ok"
    reasons = {r.reason for r in scan.rejects}
    assert reasons == {
        "stale_policy",
        "observation_dim_mismatch",
        "schema_mismatch",
        "old_log_prob_unavailable",
    }


def test_learner_does_not_read_pending(tmp_path: Path):
    run_root = tmp_path / "run"
    entry = _build_checkpoint(run_root, policy_version=1)
    dirs = _RunDirs.from_root(run_root)
    dirs.ensure()
    # pending に置いた shard は scan されない
    pending = run_root / "rollouts" / "pending"
    _write_crafted_shard(pending, "p1", policy_version=1)
    scan = scan_ready(dirs, entry, _update_config(run_root))
    assert scan.compatible == []
    assert scan.rejects == []


# ----------------------------------------------------------------------
# dry-run
# ----------------------------------------------------------------------


def test_learner_dry_run_does_not_publish_checkpoint(tmp_path: Path):
    run_root = tmp_path / "run"
    entry = _build_checkpoint(run_root, policy_version=1)
    _actor_fill(run_root)
    before = read_json(run_root / "checkpoints" / "latest.json")

    summary = run_learner(_update_config(run_root, dry_run=True))
    assert summary["dry_run"] is True
    assert summary["event"] == "learner_dry_run"
    assert summary["compatible_shards"] >= 1
    assert summary["compatible_samples"] >= 1

    after = read_json(run_root / "checkpoints" / "latest.json")
    assert after == before
    assert not (run_root / "checkpoints" / "policy_000002.pt").exists()
    # ready は残ったまま
    assert any((run_root / "rollouts" / "ready").iterdir())
    assert entry.policy_version == 1


# ----------------------------------------------------------------------
# update / publish / consume
# ----------------------------------------------------------------------


def test_learner_publishes_checkpoint_and_updates_latest(tmp_path: Path):
    run_root = tmp_path / "run"
    _build_checkpoint(run_root, policy_version=1)
    _actor_fill(run_root)

    summary = run_learner(_update_config(run_root))
    assert summary["updates_done"] == 1
    assert summary["final_policy_version"] == 2

    new_ckpt = run_root / "checkpoints" / "policy_000002.pt"
    assert new_ckpt.is_file()
    latest = read_json(run_root / "checkpoints" / "latest.json")
    assert latest["policy_version"] == 2
    assert latest["checkpoint_path"] == "checkpoints/policy_000002.pt"
    # 新 checkpoint は再ロード可能
    payload = torch.load(new_ckpt, map_location="cpu")
    assert payload["policy_version"] == 2
    assert payload["source_policy_versions"] == [1]


def test_learner_moves_consumed_shards_after_success(tmp_path: Path):
    run_root = tmp_path / "run"
    _build_checkpoint(run_root, policy_version=1)
    actor_summary = _actor_fill(run_root)
    num_shards = int(actor_summary["published_shards"])
    assert num_shards >= 1

    run_learner(_update_config(run_root))
    ready = run_root / "rollouts" / "ready"
    consumed = run_root / "rollouts" / "consumed" / "policy_000002"
    assert list(ready.iterdir()) == []
    assert consumed.is_dir()
    assert len(list(consumed.iterdir())) == num_shards


def test_checkpoint_publish_is_hash_consistent(tmp_path: Path):
    run_root = tmp_path / "run"
    _build_checkpoint(run_root, policy_version=1)
    _actor_fill(run_root)
    run_learner(_update_config(run_root))
    new_ckpt = run_root / "checkpoints" / "policy_000002.pt"
    latest = read_json(run_root / "checkpoints" / "latest.json")
    assert latest["checkpoint_sha256"] == sha256_file(new_ckpt)


def test_learner_metrics_jsonl_row(tmp_path: Path):
    import json

    run_root = tmp_path / "run"
    _build_checkpoint(run_root, policy_version=1)
    _actor_fill(run_root)
    run_learner(_update_config(run_root))
    metrics = run_root / "metrics" / "learner.jsonl"
    rows = [json.loads(ln) for ln in metrics.read_text().splitlines() if ln.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "learner_update"
    assert row["policy_version_before"] == 1
    assert row["policy_version_after"] == 2
    assert row["num_samples"] > 0
    assert row["num_updates"] >= 1
    assert row["source_policy_versions"] == [1]
    assert "checkpoint_sha256" in row


# ----------------------------------------------------------------------
# failure handling
# ----------------------------------------------------------------------


def test_learner_does_not_consume_on_training_failure(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "run"
    entry = _build_checkpoint(run_root, policy_version=1)
    _actor_fill(run_root)
    before = read_json(run_root / "checkpoints" / "latest.json")
    ready_before = sorted(p.name for p in (run_root / "rollouts" / "ready").iterdir())

    def _boom(*args, **kwargs):
        raise RuntimeError("induced training failure")

    monkeypatch.setattr(learner_mod, "fit_ppo", _boom)

    with pytest.raises(RuntimeError, match="induced training failure"):
        run_learner(_update_config(run_root))

    # checkpoint は publish されず latest.json も不変
    after = read_json(run_root / "checkpoints" / "latest.json")
    assert after == before
    assert not (run_root / "checkpoints" / "policy_000002.pt").exists()
    # ready shard は残る (consume されない)
    ready_after = sorted(p.name for p in (run_root / "rollouts" / "ready").iterdir())
    assert ready_after == ready_before
    assert entry.policy_version == 1


# ----------------------------------------------------------------------
# hidden-info guard
# ----------------------------------------------------------------------


def test_learner_source_does_not_reference_hidden_state():
    text = Path(learner_mod.__file__).read_text(encoding="utf-8")
    for tok in (
        "env.hands",
        "env.wall",
        "env.state",
        "obs.hands[",
        "full_state",
        "private_hand",
        "mjai_log",
    ):
        assert tok not in text, f"learner source references {tok!r}"
