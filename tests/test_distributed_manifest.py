"""Tests for ``mahjong_agent.distributed.manifest`` helpers."""
from __future__ import annotations

from pathlib import Path

from mahjong_agent.distributed import (
    CheckpointRegistryEntry,
    RolloutShardManifest,
    ShardState,
    is_manifest_compatible,
    read_json,
    sha256_file,
    write_json_atomic,
)

# ----------------------------------------------------------------------
# atomic json helpers
# ----------------------------------------------------------------------


def test_write_json_atomic_leaves_no_tmp(tmp_path: Path):
    p = tmp_path / "sub" / "latest.json"
    write_json_atomic(p, {"a": 1, "b": "x"})
    assert p.is_file()
    assert read_json(p) == {"a": 1, "b": "x"}
    # tmp が残っていない
    assert not (p.with_name(p.name + ".tmp")).exists()
    assert list(p.parent.glob("*.tmp")) == []


def test_write_json_atomic_overwrites_existing(tmp_path: Path):
    p = tmp_path / "latest.json"
    write_json_atomic(p, {"v": 1})
    write_json_atomic(p, {"v": 2})
    assert read_json(p) == {"v": 2}


def test_sha256_file_stable_and_distinct(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")
    assert sha256_file(a) == sha256_file(a)
    assert sha256_file(a) != sha256_file(b)
    # 既知 SHA-256("hello")
    assert sha256_file(a) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


# ----------------------------------------------------------------------
# dataclass round-trip
# ----------------------------------------------------------------------


def test_rollout_manifest_round_trip(tmp_path: Path):
    m = RolloutShardManifest(
        policy_version=123,
        checkpoint_sha256="deadbeef",
        actor_id="actor_0007",
        created_at="2026-06-04T12:30:00Z",
        num_games=1000,
        num_samples=213760,
        schema_version=2,
        observation_dim=608,
        candidate_dim=91,
        seed_start=1000000,
        seed_end=1000999,
        hostname="cpu-node-07",
        pid=12345,
        old_log_prob_available=True,
    )
    p = tmp_path / "manifest.json"
    write_json_atomic(p, m.to_dict())
    loaded = RolloutShardManifest.from_dict(read_json(p))
    assert loaded == m
    assert loaded.shard_path == "shard.npz"
    assert loaded.manifest_version == 1
    assert loaded.role == "actor_rollout"


def test_checkpoint_registry_round_trip(tmp_path: Path):
    e = CheckpointRegistryEntry(
        policy_version=123,
        checkpoint_path="checkpoints/policy_000123.pt",
        checkpoint_sha256="abc123",
        created_at="2026-06-04T12:34:56Z",
        schema_version=2,
        observation_dim=608,
        candidate_dim=91,
        model_config={"hidden_dim": 256},
        encoder_metadata={"observation_dim": 608, "candidate_dim": 91},
    )
    p = tmp_path / "latest.json"
    write_json_atomic(p, e.to_dict())
    loaded = CheckpointRegistryEntry.from_dict(read_json(p))
    assert loaded == e


def test_shard_state_values():
    assert ShardState.PENDING.value == "pending"
    assert ShardState.READY.value == "ready"
    assert ShardState.CONSUMED.value == "consumed"
    assert ShardState.REJECTED.value == "rejected"


# ----------------------------------------------------------------------
# compatibility / policy lag
# ----------------------------------------------------------------------


def _manifest(policy_version=10, schema_version=2, observation_dim=608,
              candidate_dim=91, old_log_prob_available=True):
    return RolloutShardManifest(
        policy_version=policy_version,
        checkpoint_sha256="x",
        actor_id="a",
        created_at="t",
        num_games=1,
        num_samples=1,
        schema_version=schema_version,
        observation_dim=observation_dim,
        candidate_dim=candidate_dim,
        old_log_prob_available=old_log_prob_available,
    )


def _compat(m, *, current=10, max_lag=1):
    return is_manifest_compatible(
        m, schema_version=2, observation_dim=608, candidate_dim=91,
        current_policy_version=current, max_policy_lag=max_lag,
    )


def test_policy_lag_equal_is_ok():
    r = _compat(_manifest(policy_version=10), current=10, max_lag=1)
    assert r.ok
    assert r.reason == "ok"
    assert bool(r) is True


def test_policy_lag_within_is_ok():
    assert _compat(_manifest(policy_version=9), current=10, max_lag=1).ok


def test_policy_lag_exceeded_is_rejected():
    r = _compat(_manifest(policy_version=8), current=10, max_lag=1)
    assert not r.ok
    assert "stale_policy" in r.reason
    assert bool(r) is False


def test_future_policy_version_is_allowed():
    # learner より新しい actor checkpoint 由来でも reject しない
    assert _compat(_manifest(policy_version=12), current=10, max_lag=1).ok


def test_schema_mismatch_rejected():
    r = _compat(_manifest(schema_version=1), current=10, max_lag=1)
    assert not r.ok
    assert "schema_mismatch" in r.reason


def test_observation_dim_mismatch_rejected():
    r = _compat(_manifest(observation_dim=506), current=10, max_lag=1)
    assert not r.ok
    assert "observation_dim_mismatch" in r.reason


def test_candidate_dim_mismatch_rejected():
    r = _compat(_manifest(candidate_dim=86), current=10, max_lag=1)
    assert not r.ok
    assert "candidate_dim_mismatch" in r.reason


def test_missing_old_log_prob_rejected():
    r = _compat(
        _manifest(old_log_prob_available=False), current=10, max_lag=1
    )
    assert not r.ok
    assert "old_log_prob" in r.reason


def test_old_log_prob_not_required_when_flag_off():
    m = _manifest(old_log_prob_available=False)
    r = is_manifest_compatible(
        m, schema_version=2, observation_dim=608, candidate_dim=91,
        current_policy_version=10, max_policy_lag=1,
        require_old_log_prob=False,
    )
    assert r.ok


def test_max_lag_zero_only_accepts_current():
    assert _compat(_manifest(policy_version=10), current=10, max_lag=0).ok
    assert not _compat(_manifest(policy_version=9), current=10, max_lag=0).ok


# ----------------------------------------------------------------------
# hidden info guard
# ----------------------------------------------------------------------


def test_distributed_source_does_not_reference_hidden_state():
    import mahjong_agent.distributed as pkg

    pkg_dir = Path(pkg.__file__).parent
    forbidden = (
        "env.hands",
        "env.wall",
        "env.state",
        "obs.hands[",
        "full_state",
        "private_hand",
        "mjai_log",
    )
    for f in pkg_dir.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, (
                f"distributed source {f.name} references {tok!r}"
            )


def test_manifest_signature_takes_only_metadata():
    import inspect

    params = set(inspect.signature(is_manifest_compatible).parameters.keys())
    forbidden = {"env", "obs", "observation", "hands", "wall", "state"}
    assert forbidden.isdisjoint(params)
