"""Tests for the unified learner supervisor (imitation bootstrap -> PPO + resume)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from mahjong_agent.distributed import learner_supervisor as ls
from mahjong_agent.distributed.actor_supervisor import (
    SupervisorConfig as ActorSupConfig,
)
from mahjong_agent.distributed.actor_supervisor import (
    run_worker_loop as actor_worker,
)
from mahjong_agent.distributed.learner_supervisor import (
    LearnerLock,
    LearnerRecoveryError,
    LearnerSupervisorConfig,
    _LSDirs,
    bootstrap_run,
    finish_transaction,
    journal_path,
    load_model_and_optimizer,
    publish_policy_checkpoint,
    read_journal,
    read_registry,
    reconcile_journal,
    run_learner_supervisor,
    scan_ppo_ready,
    write_journal,
)
from mahjong_agent.distributed.manifest import (
    CheckpointRegistryEntry,
    RolloutShardManifest,
    read_json,
    sha256_file,
    write_json_atomic,
)
from mahjong_agent.distributed.run_state import (
    DistributedRunConfig,
    ImitationRunConfig,
    OperationsRunConfig,
    PPORunConfig,
    RunPhase,
    config_sha256,
    initialize_run_state,
    read_run_state,
    state_path,
    transition_run_state,
)
from mahjong_agent.encoders import PublicObservationEncoder
from mahjong_agent.models import Stage03Model, Stage03ModelConfig


def _config() -> DistributedRunConfig:
    return DistributedRunConfig(
        run_id="run_ls",
        seed_namespace="ns_ls",
        imitation=ImitationRunConfig(target_games=2, chunk_games=1, epochs=1),
        ppo=PPORunConfig(
            actor_chunk_games=1, game_type="YON_IKKYOKU",
            min_samples_per_update=1, max_samples_per_update=10_000,
            max_policy_lag=1, epochs=1, target_kl=1.0, learning_rate=5e-4,
        ),
        operations=OperationsRunConfig(),
    )


def _init(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    initialize_run_state(run_root, _config())
    return run_root


def _ls_cfg(run_root: Path, **kw) -> LearnerSupervisorConfig:
    base = dict(run_root=str(run_root), device="cpu", poll_interval_sec=0.01)
    base.update(kw)
    return LearnerSupervisorConfig(**base)


def _write_ckpt_v1(run_root: Path) -> None:
    encoder = PublicObservationEncoder(enable_hints=True)
    cfg = Stage03ModelConfig.from_encoder_metadata(
        encoder.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16,
    )
    model = Stage03Model(cfg).eval()
    ckdir = run_root / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    rel = "checkpoints/policy_000001.pt"
    p = run_root / rel
    torch.save(
        {"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
         "model_config": asdict(model.config), "policy_version": 1},
        p,
    )
    entry = CheckpointRegistryEntry(
        policy_version=1, checkpoint_path=rel, checkpoint_sha256=sha256_file(p),
        created_at="t", schema_version=2,
        observation_dim=int(cfg.observation_dim), candidate_dim=int(cfg.candidate_dim),
        model_config=asdict(model.config),
        encoder_metadata={"observation_dim": int(cfg.observation_dim),
                          "candidate_dim": int(cfg.candidate_dim), "enable_hints": True},
    )
    write_json_atomic(ckdir / "latest.json", entry.to_dict())


def _gen_actor_shards(run_root: Path, n: int) -> None:
    actor_worker(
        ActorSupConfig(run_root=str(run_root), actor_id="a", workers=1, max_chunks=n),
        0,
    )


def _advance(run_root: Path, *phases, policy_version_on_last=None):
    cur = RunPhase.INITIALIZING
    for i, nxt in enumerate(phases):
        kw = {}
        if policy_version_on_last is not None and i == len(phases) - 1:
            kw["policy_version"] = policy_version_on_last
        transition_run_state(run_root, expected_phase=cur, next_phase=nxt, **kw)
        cur = nxt


# ----------------------------------------------------------------------
# singleton lock
# ----------------------------------------------------------------------


def test_lock_acquire_double_and_release(tmp_path: Path):
    run_root = _init(tmp_path)
    lock1 = LearnerLock(run_root)
    lock1.acquire()
    lock2 = LearnerLock(run_root)
    with pytest.raises(RuntimeError, match="already locked"):
        lock2.acquire()
    lock1.release()
    # release 後は再取得できる
    lock3 = LearnerLock(run_root)
    lock3.acquire()
    lock3.release()


def test_lock_force_unlock(tmp_path: Path):
    run_root = _init(tmp_path)
    LearnerLock(run_root).acquire()  # 解放しない（stale lock を模倣）
    forced = LearnerLock(run_root, force=True)
    forced.acquire()
    forced.release()


# ----------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------


def test_bootstrap_init_and_validate(tmp_path: Path):
    run_root = tmp_path / "run"
    cfg_path = tmp_path / "rc.json"
    cfg_path.write_text(json.dumps(_config().to_dict()), encoding="utf-8")
    rc = bootstrap_run(_ls_cfg(run_root, config_path=str(cfg_path)))
    assert rc.run_id == "run_ls"
    assert (run_root / "run_config.json").is_file()
    # 同一 config で再 bootstrap は ok
    bootstrap_run(_ls_cfg(run_root, config_path=str(cfg_path)))
    # 別 config は reject
    other = _config()
    other = DistributedRunConfig(
        run_id="run_ls", seed_namespace="ns_ls",
        ppo=PPORunConfig(learning_rate=9e-4),
    )
    bad = tmp_path / "rc2.json"
    bad.write_text(json.dumps(other.to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        bootstrap_run(_ls_cfg(run_root, config_path=str(bad)))


# ----------------------------------------------------------------------
# phase progression
# ----------------------------------------------------------------------


def test_initializing_to_collect_and_wait(tmp_path: Path):
    run_root = _init(tmp_path)
    out = run_learner_supervisor(_ls_cfg(run_root, max_idle_polls=1))
    assert "initializing->imitation_collect" in out["transitions"]
    assert out["exit_reason"] == "max_idle_polls"
    state = read_run_state(state_path(run_root))
    assert state.phase == RunPhase.IMITATION_COLLECT


def test_collect_target_unmet_waits(tmp_path: Path):
    run_root = _init(tmp_path)
    transition_run_state(
        run_root, expected_phase=RunPhase.INITIALIZING,
        next_phase=RunPhase.IMITATION_COLLECT,
    )
    # 1 shard だけ（target=2 未満）
    _gen_actor_shards(run_root, 1)
    out = run_learner_supervisor(_ls_cfg(run_root, max_idle_polls=1))
    assert out["exit_reason"] == "max_idle_polls"
    assert read_run_state(state_path(run_root)).phase == RunPhase.IMITATION_COLLECT


def test_collect_to_train_publishes_v1_and_enters_ppo(tmp_path: Path):
    run_root = _init(tmp_path)
    transition_run_state(
        run_root, expected_phase=RunPhase.INITIALIZING,
        next_phase=RunPhase.IMITATION_COLLECT,
    )
    _gen_actor_shards(run_root, 2)  # 2 games == target
    out = run_learner_supervisor(_ls_cfg(run_root, max_idle_polls=1))
    assert "imitation_collect->imitation_train" in out["transitions"]
    assert "imitation_train->ppo" in out["transitions"]
    # initial checkpoint published
    assert (run_root / "checkpoints" / "policy_000001.pt").is_file()
    latest = read_json(run_root / "checkpoints" / "latest.json")
    assert latest["policy_version"] == 1
    # imitation shards consumed
    assert list((run_root / "imitation" / "ready").iterdir()) == []
    assert len(list((run_root / "imitation" / "consumed").iterdir())) == 2
    state = read_run_state(state_path(run_root))
    assert state.phase == RunPhase.PPO
    assert state.policy_version == 1


# ----------------------------------------------------------------------
# ppo update + optimizer persistence
# ----------------------------------------------------------------------


def _setup_ppo(run_root: Path, n_shards: int = 2):
    _write_ckpt_v1(run_root)
    _advance(
        run_root, RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN, RunPhase.PPO,
        policy_version_on_last=1,
    )
    _gen_actor_shards(run_root, n_shards)


def test_ppo_update_publishes_v2_with_optimizer(tmp_path: Path):
    run_root = _init(tmp_path)
    _setup_ppo(run_root, n_shards=2)
    out = run_learner_supervisor(_ls_cfg(run_root, stop_after_ppo_updates=1))
    assert out["ppo_updates"] == 1
    assert out["final_policy_version"] == 2
    new_ckpt = run_root / "checkpoints" / "policy_000002.pt"
    assert new_ckpt.is_file()
    payload = torch.load(new_ckpt, map_location="cpu")
    assert "optimizer_state_dict" in payload
    assert payload["phase"] == "ppo"
    assert payload["policy_version"] == 2
    state = read_run_state(state_path(run_root))
    assert state.policy_version == 2
    assert state.phase == RunPhase.PPO
    # shards consumed
    assert list((run_root / "rollouts" / "ready").iterdir()) == []
    assert (run_root / "rollouts" / "consumed" / "policy_000002").is_dir()


def test_optimizer_state_restored_after_restart(tmp_path: Path):
    run_root = _init(tmp_path)
    _setup_ppo(run_root, n_shards=2)
    run_learner_supervisor(_ls_cfg(run_root, stop_after_ppo_updates=1))
    # restart: latest (v2) から model+optimizer 復元
    dirs = _LSDirs.from_root(run_root)
    entry = read_registry(dirs)
    assert entry.policy_version == 2
    from mahjong_agent.distributed.learner_supervisor import _make_ppo_config

    _model, optimizer, had_opt = load_model_and_optimizer(
        dirs, entry, "cpu", _make_ppo_config(_config(), "cpu")
    )
    assert had_opt is True
    assert len(optimizer.state) > 0  # AdamW state が復元されている


# ----------------------------------------------------------------------
# generation filtering
# ----------------------------------------------------------------------


def _craft_shard(ready: Path, name: str, *, phase, phase_generation, policy_version=1,
                 schema_version=2, observation_dim=608, candidate_dim=91):
    d = ready / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "shard.npz").write_bytes(b"dummy")
    m = RolloutShardManifest(
        policy_version=policy_version, checkpoint_sha256="x", actor_id="a",
        created_at="t", num_games=1, num_samples=1, schema_version=schema_version,
        observation_dim=observation_dim, candidate_dim=candidate_dim,
        old_log_prob_available=True, role="actor_rollout",
        phase=phase, phase_generation=phase_generation,
    )
    write_json_atomic(d / "manifest.json", m.to_dict())


def test_scan_ppo_ready_generation_and_phase_filter(tmp_path: Path):
    run_root = _init(tmp_path)
    _write_ckpt_v1(run_root)
    dirs = _LSDirs.from_root(run_root)
    dirs.ensure()
    entry = read_registry(dirs)
    ready = dirs.rollouts_ready
    _craft_shard(ready, "ok", phase="ppo", phase_generation=3)
    _craft_shard(ready, "oldgen", phase="ppo", phase_generation=2)
    _craft_shard(ready, "wrongphase", phase="imitation_collect", phase_generation=3)
    _craft_shard(ready, "nophase", phase=None, phase_generation=None)
    _craft_shard(ready, "dimbad", phase="ppo", phase_generation=3, observation_dim=506)

    scan = scan_ppo_ready(dirs, entry=entry, ppo_generation=3, max_policy_lag=1)
    assert [d.name for d, _m in scan.compatible] == ["ok"]
    reasons = {d.name: r for d, r in scan.rejects}
    assert reasons["oldgen"] == "generation_mismatch"
    assert reasons["wrongphase"] == "phase_mismatch"
    assert reasons["nophase"] == "missing_phase"
    assert reasons["dimbad"] == "observation_dim_mismatch"


# ----------------------------------------------------------------------
# failure semantics
# ----------------------------------------------------------------------


def test_training_failure_no_consume_and_failed_state(tmp_path: Path, monkeypatch):
    run_root = _init(tmp_path)
    _setup_ppo(run_root, n_shards=2)
    ready_before = sorted(p.name for p in (run_root / "rollouts" / "ready").iterdir())

    def _boom(*a, **k):
        raise RuntimeError("induced ppo failure")

    monkeypatch.setattr(ls, "fit_ppo", _boom)
    with pytest.raises(RuntimeError, match="induced ppo failure"):
        run_learner_supervisor(_ls_cfg(run_root, stop_after_ppo_updates=1))

    # shard 未 consume・checkpoint 未 publish・state=failed
    ready_after = sorted(p.name for p in (run_root / "rollouts" / "ready").iterdir())
    assert ready_after == ready_before
    assert not (run_root / "checkpoints" / "policy_000002.pt").exists()
    state = read_run_state(state_path(run_root))
    assert state.phase == RunPhase.FAILED
    assert "induced ppo failure" in state.failure_reason
    # lock も解放されている（再取得できる）
    LearnerLock(run_root).acquire()


def test_draining_transitions_to_stopped(tmp_path: Path):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT, RunPhase.DRAINING)
    out = run_learner_supervisor(_ls_cfg(run_root, max_idle_polls=1))
    assert out["exit_reason"] == "stopped"
    assert read_run_state(state_path(run_root)).phase == RunPhase.STOPPED


# ----------------------------------------------------------------------
# transaction journal: finish_transaction reconciliation (PPO)
# ----------------------------------------------------------------------


def _fresh_model():
    enc = PublicObservationEncoder(enable_hints=True)
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16,
    )
    return Stage03Model(cfg).eval()


def _craft_rollout(ready: Path, name: str, *, gen=3, pv=1):
    d = ready / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "shard.npz").write_bytes(b"x")
    m = RolloutShardManifest(
        policy_version=pv, checkpoint_sha256="x", actor_id="a", created_at="t",
        num_games=1, num_samples=1, schema_version=2, observation_dim=608,
        candidate_dim=91, old_log_prob_available=True, role="actor_rollout",
        phase="ppo", phase_generation=gen,
    )
    write_json_atomic(d / "manifest.json", m.to_dict())


def _make_published_ppo_txn(tmp_path: Path, n: int = 2):
    """v2 を実 publish し、stage=published の journal + source shards を作る。"""
    run_root = _init(tmp_path)
    _write_ckpt_v1(run_root)
    _advance(
        run_root, RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN, RunPhase.PPO,
        policy_version_on_last=1,
    )
    dirs = _LSDirs.from_root(run_root)
    dirs.ensure()
    entry = publish_policy_checkpoint(
        dirs, model=_fresh_model(), new_version=2, phase=RunPhase.PPO,
        phase_generation=3, config_sha=config_sha256(_config()),
        encoder_metadata={"observation_dim": 608, "candidate_dim": 91, "enable_hints": True},
        learner_metrics={}, source_policy_versions=[1], num_samples=1, optimizer=None,
    )
    src = []
    for i in range(n):
        name = f"shard{i}"
        _craft_rollout(dirs.rollouts_ready, name)
        src.append({
            "ready_path": f"rollouts/ready/{name}",
            "consumed_path": f"rollouts/consumed/policy_000002/{name}",
        })
    journal = {
        "transaction_version": 1, "kind": "ppo_update", "stage": "published",
        "phase": "ppo", "phase_generation": 3, "policy_version_before": 1,
        "policy_version_after": 2, "source_shards": src,
        "checkpoint_path": "checkpoints/policy_000002.pt",
        "checkpoint_sha256": entry.checkpoint_sha256, "consume_policy": "move",
        "created_at": "t", "updated_at": "t",
    }
    write_journal(dirs, journal)
    return run_root, dirs, journal


def test_finish_published_consumes_and_commits(tmp_path: Path):
    run_root, dirs, journal = _make_published_ppo_txn(tmp_path, n=2)
    finish_transaction(dirs, _config(), journal)
    assert list((run_root / "rollouts" / "ready").iterdir()) == []
    consumed = run_root / "rollouts" / "consumed" / "policy_000002"
    assert len(list(consumed.iterdir())) == 2
    assert read_run_state(state_path(run_root)).policy_version == 2
    assert not journal_path(run_root).exists()


def test_finish_partial_consume_idempotent(tmp_path: Path):
    run_root, dirs, journal = _make_published_ppo_txn(tmp_path, n=2)
    # shard0 を事前に consumed へ move（部分 consume 後 crash を模倣）
    src0 = run_root / journal["source_shards"][0]["consumed_path"]
    src0.parent.mkdir(parents=True, exist_ok=True)
    import os as _os
    _os.replace(run_root / journal["source_shards"][0]["ready_path"], src0)
    finish_transaction(dirs, _config(), journal)
    consumed = run_root / "rollouts" / "consumed" / "policy_000002"
    assert sorted(p.name for p in consumed.iterdir()) == ["shard0", "shard1"]
    assert list((run_root / "rollouts" / "ready").iterdir()) == []
    assert not journal_path(run_root).exists()


def test_finish_hash_mismatch_is_fatal(tmp_path: Path):
    run_root, dirs, journal = _make_published_ppo_txn(tmp_path, n=1)
    # checkpoint を破損させる
    (run_root / "checkpoints" / "policy_000002.pt").write_bytes(b"corrupt")
    with pytest.raises(LearnerRecoveryError, match="hash mismatch"):
        finish_transaction(dirs, _config(), journal)


def test_finish_missing_shard_is_fatal(tmp_path: Path):
    run_root, dirs, journal = _make_published_ppo_txn(tmp_path, n=2)
    # source shard を ready からも consumed からも消す
    import shutil as _sh
    _sh.rmtree(run_root / journal["source_shards"][0]["ready_path"])
    with pytest.raises(LearnerRecoveryError, match="disappeared"):
        finish_transaction(dirs, _config(), journal)


def test_reconcile_state_committed_deletes_journal(tmp_path: Path):
    run_root, dirs, journal = _make_published_ppo_txn(tmp_path, n=1)
    # state は既に v2 に commit 済みとし、journal を state_committed に進める
    transition_run_state(
        run_root, expected_phase=RunPhase.PPO, next_phase=RunPhase.PPO, policy_version=2,
    )
    journal["stage"] = "state_committed"
    write_journal(dirs, journal)
    status = reconcile_journal(dirs, _config())
    assert status == "reconciled_from_state_committed"
    assert not journal_path(run_root).exists()


# ----------------------------------------------------------------------
# transaction journal: live publish-crash then restart reconcile
# ----------------------------------------------------------------------


def test_ppo_publish_crash_then_restart_reconciles(tmp_path: Path, monkeypatch):
    run_root = _init(tmp_path)
    _setup_ppo(run_root, n_shards=2)

    ppo_calls = {"n": 0}
    real_fit = ls.fit_ppo

    def counting_fit(*a, **k):
        ppo_calls["n"] += 1
        return real_fit(*a, **k)

    monkeypatch.setattr(ls, "fit_ppo", counting_fit)

    real_finish = ls.finish_transaction
    crashed = {"done": False}

    def crash_once(*a, **k):
        if not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("crash after publish")
        return real_finish(*a, **k)

    monkeypatch.setattr(ls, "finish_transaction", crash_once)
    with pytest.raises(RuntimeError, match="crash after publish"):
        run_learner_supervisor(_ls_cfg(run_root, stop_after_ppo_updates=1))

    # publish 済み・state 未 commit・shard 未 consume・state=ppo(v1)
    j = read_journal(_LSDirs.from_root(run_root))
    assert j is not None and j["stage"] == "published"
    assert (run_root / "checkpoints" / "policy_000002.pt").is_file()
    assert read_json(run_root / "checkpoints" / "latest.json")["policy_version"] == 2
    state = read_run_state(state_path(run_root))
    assert state.phase == RunPhase.PPO and state.policy_version == 1
    assert len(list((run_root / "rollouts" / "ready").iterdir())) == 2

    # restart: reconcile が完了させる（再 training しない）
    out = run_learner_supervisor(_ls_cfg(run_root, max_idle_polls=1))
    assert any("reconcile" in t for t in out["transitions"])
    assert read_journal(_LSDirs.from_root(run_root)) is None
    assert read_run_state(state_path(run_root)).policy_version == 2
    assert list((run_root / "rollouts" / "ready").iterdir()) == []
    assert (run_root / "rollouts" / "consumed" / "policy_000002").is_dir()
    assert ppo_calls["n"] == 1  # 二重 update していない


def test_imitation_publish_crash_then_restart_reconciles(tmp_path: Path, monkeypatch):
    run_root = _init(tmp_path)
    transition_run_state(
        run_root, expected_phase=RunPhase.INITIALIZING,
        next_phase=RunPhase.IMITATION_COLLECT,
    )
    _gen_actor_shards(run_root, 2)

    imit_calls = {"n": 0}
    real_imit = ls.fit_imitation

    def counting_imit(*a, **k):
        imit_calls["n"] += 1
        return real_imit(*a, **k)

    monkeypatch.setattr(ls, "fit_imitation", counting_imit)
    real_finish = ls.finish_transaction
    crashed = {"done": False}

    def crash_once(*a, **k):
        if not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("crash after imitation publish")
        return real_finish(*a, **k)

    monkeypatch.setattr(ls, "finish_transaction", crash_once)
    with pytest.raises(RuntimeError, match="crash after imitation publish"):
        run_learner_supervisor(_ls_cfg(run_root, max_idle_polls=1))

    j = read_journal(_LSDirs.from_root(run_root))
    assert j is not None and j["stage"] == "published"
    assert (run_root / "checkpoints" / "policy_000001.pt").is_file()
    assert read_run_state(state_path(run_root)).phase == RunPhase.IMITATION_TRAIN
    assert len(list((run_root / "imitation" / "ready").iterdir())) == 2

    # restart reconcile
    run_learner_supervisor(_ls_cfg(run_root, max_idle_polls=1))
    assert read_journal(_LSDirs.from_root(run_root)) is None
    state = read_run_state(state_path(run_root))
    assert state.phase == RunPhase.PPO and state.policy_version == 1
    assert list((run_root / "imitation" / "ready").iterdir()) == []
    assert len(list((run_root / "imitation" / "consumed").iterdir())) == 2
    assert imit_calls["n"] == 1  # 再 training していない


# ----------------------------------------------------------------------
# hidden-info guard
# ----------------------------------------------------------------------


def test_learner_supervisor_source_does_not_reference_hidden_state():
    text = Path(ls.__file__).read_text(encoding="utf-8")
    for tok in (
        "env.hands", "env.wall", "env.state", "obs.hands[",
        "full_state", "private_hand", "mjai_log",
    ):
        assert tok not in text, f"learner_supervisor source references {tok!r}"
