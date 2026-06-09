"""Tests for the phase-aware actor supervisor."""
from __future__ import annotations

import socket
from dataclasses import asdict
from pathlib import Path

import torch

from mahjong_agent.distributed.actor_supervisor import (
    Supervisor,
    SupervisorConfig,
    _SupDirs,
    _supervise_step,
    _WorkerProc,
    compute_backoff,
    derive_chunk_base_seed,
    read_next_chunk_index,
    reserve_chunk_index,
    run_worker_loop,
    safe_worker_id,
    worker_id,
)
from mahjong_agent.distributed.manifest import (
    CheckpointRegistryEntry,
    read_json,
    sha256_file,
    write_json_atomic,
)
from mahjong_agent.distributed.run_state import (
    DistributedRunConfig,
    EvaluatorRunConfig,
    ImitationRunConfig,
    OperationsRunConfig,
    PPORunConfig,
    RunPhase,
    initialize_run_state,
    transition_run_state,
)
from mahjong_agent.encoders import PublicObservationEncoder
from mahjong_agent.models import Stage03Model, Stage03ModelConfig


def _config(*, max_ready_shards: int = 10_000) -> DistributedRunConfig:
    return DistributedRunConfig(
        run_id="run_sup",
        seed_namespace="ns_sup",
        imitation=ImitationRunConfig(target_games=10, chunk_games=1, epochs=1),
        ppo=PPORunConfig(actor_chunk_games=1, game_type="YON_IKKYOKU"),
        evaluator=EvaluatorRunConfig(),
        operations=OperationsRunConfig(max_ready_shards=max_ready_shards),
    )


def _init(tmp_path: Path, *, max_ready_shards: int = 10_000) -> Path:
    run_root = tmp_path / "run"
    initialize_run_state(run_root, _config(max_ready_shards=max_ready_shards))
    return run_root


def _advance(run_root: Path, *phases: RunPhase) -> None:
    cur = RunPhase.INITIALIZING
    for nxt in phases:
        transition_run_state(run_root, expected_phase=cur, next_phase=nxt)
        cur = nxt


def _write_ckpt(run_root: Path, *, policy_version: int = 1) -> None:
    encoder = PublicObservationEncoder(enable_hints=True)
    cfg = Stage03ModelConfig.from_encoder_metadata(
        encoder.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16,
    )
    model = Stage03Model(cfg).eval()
    ckdir = run_root / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    rel = f"checkpoints/policy_{policy_version:06d}.pt"
    p = run_root / rel
    torch.save(
        {"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
         "model_config": asdict(model.config)},
        p,
    )
    entry = CheckpointRegistryEntry(
        policy_version=policy_version, checkpoint_path=rel,
        checkpoint_sha256=sha256_file(p), created_at="t", schema_version=2,
        observation_dim=int(cfg.observation_dim), candidate_dim=int(cfg.candidate_dim),
        model_config=asdict(model.config),
        encoder_metadata={"observation_dim": int(cfg.observation_dim),
                          "candidate_dim": int(cfg.candidate_dim), "enable_hints": True},
    )
    write_json_atomic(ckdir / "latest.json", entry.to_dict())


def _sup(run_root: Path, **kw) -> SupervisorConfig:
    base = dict(run_root=str(run_root), actor_id="host_a", workers=1,
                poll_interval_sec=0.01)
    base.update(kw)
    return SupervisorConfig(**base)


# ----------------------------------------------------------------------
# identity / seed
# ----------------------------------------------------------------------


def test_actor_id_defaults_to_hostname():
    assert SupervisorConfig(run_root="x").resolved_actor_id() == socket.gethostname()
    assert SupervisorConfig(run_root="x", actor_id="node7").resolved_actor_id() == "node7"


def test_worker_id_numbering():
    assert worker_id("host", 0) == "host/w000"
    assert worker_id("host", 12) == "host/w012"
    assert safe_worker_id("host/w012") == "host_w012"


def test_derive_seed_stable_and_distinct():
    def s(**kw):
        base = dict(seed_namespace="ns", actor_id="a", wid="a/w000",
                    phase_generation=1, chunk_index=0)
        base.update(kw)
        return derive_chunk_base_seed(**base)

    assert s() == s()  # 安定
    assert s(wid="a/w001") != s()  # worker 差
    assert s(phase_generation=2) != s()  # generation 差
    assert s(chunk_index=1) != s()  # chunk 差
    assert s(actor_id="b") != s()  # actor 差
    assert s(seed_namespace="ns2") != s()  # namespace 差


def test_compute_backoff():
    assert compute_backoff(0) == 0.0
    assert compute_backoff(1) == 1.0
    assert compute_backoff(2) == 2.0
    assert compute_backoff(3) == 4.0
    assert compute_backoff(100, cap=30.0) == 30.0


# ----------------------------------------------------------------------
# waiting phases
# ----------------------------------------------------------------------


def test_initializing_waits_no_shards(tmp_path: Path):
    run_root = _init(tmp_path)
    out = run_worker_loop(_sup(run_root, max_poll_iterations=2), 0)
    assert out["exit_reason"] == "max_poll_iterations"
    assert out["chunks_published"] == 0
    assert not any((run_root / "imitation" / "ready").iterdir())
    assert not any((run_root / "rollouts" / "ready").iterdir())


def test_imitation_train_waits(tmp_path: Path):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN)
    out = run_worker_loop(_sup(run_root, max_poll_iterations=2), 0)
    assert out["exit_reason"] == "max_poll_iterations"
    assert out["chunks_published"] == 0


# ----------------------------------------------------------------------
# imitation_collect generation
# ----------------------------------------------------------------------


def test_imitation_collect_generates_ready_shard(tmp_path: Path):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT)
    out = run_worker_loop(_sup(run_root, max_chunks=1), 0)
    assert out["chunks_published"] == 1
    ready = list((run_root / "imitation" / "ready").iterdir())
    assert len(ready) == 1
    shard_dir = ready[0]
    assert (shard_dir / "shard.npz").is_file()
    manifest = read_json(shard_dir / "manifest.json")
    assert manifest["phase"] == "imitation_collect"
    assert manifest["phase_generation"] == 1
    assert manifest["worker_id"] == "host_a/w000"
    assert manifest["chunk_index"] == 0
    assert manifest["role"] == "imitation_teacher"
    assert manifest["old_log_prob_available"] is False
    # rollouts には何も出ない
    assert not any((run_root / "rollouts" / "ready").iterdir())


def test_heartbeat_written(tmp_path: Path):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT)
    run_worker_loop(_sup(run_root, max_chunks=1), 0)
    hb_path = run_root / "metrics" / "heartbeats" / "host_a_w000.json"
    assert hb_path.is_file()
    hb = read_json(hb_path)
    assert hb["actor_id"] == "host_a"
    assert hb["worker_id"] == "host_a/w000"
    assert hb["chunk_counter"] == 1
    assert hb["phase"] == "imitation_collect"
    assert "status" in hb


# ----------------------------------------------------------------------
# ppo generation
# ----------------------------------------------------------------------


def test_ppo_generates_rollout_shard(tmp_path: Path):
    run_root = _init(tmp_path)
    _write_ckpt(run_root, policy_version=1)
    _advance(run_root, RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN, RunPhase.PPO)
    out = run_worker_loop(_sup(run_root, max_chunks=1), 0)
    assert out["chunks_published"] == 1
    ready = list((run_root / "rollouts" / "ready").iterdir())
    assert len(ready) == 1
    manifest = read_json(ready[0] / "manifest.json")
    assert manifest["phase"] == "ppo"
    assert manifest["phase_generation"] == 3
    assert manifest["role"] == "actor_rollout"
    assert manifest["policy_version"] == 1
    assert manifest["old_log_prob_available"] is True
    assert manifest["worker_id"] == "host_a/w000"


# ----------------------------------------------------------------------
# draining / stopped follow
# ----------------------------------------------------------------------


def test_draining_exits_without_new_chunk(tmp_path: Path):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT, RunPhase.DRAINING)
    out = run_worker_loop(_sup(run_root, max_chunks=5), 0)
    assert out["exit_reason"] == "draining"
    assert out["chunks_published"] == 0


def test_stopped_exits(tmp_path: Path):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT, RunPhase.DRAINING, RunPhase.STOPPED)
    out = run_worker_loop(_sup(run_root, max_chunks=5), 0)
    assert out["exit_reason"] == "stopped"
    assert out["chunks_published"] == 0


def test_phase_change_followed_across_iterations(tmp_path: Path):
    # collect で 1 chunk 出してから draining に変えると、次 iteration で停止する。
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT)
    out1 = run_worker_loop(_sup(run_root, max_chunks=1), 0)
    assert out1["chunks_published"] == 1
    transition_run_state(
        run_root, expected_phase=RunPhase.IMITATION_COLLECT,
        next_phase=RunPhase.DRAINING,
    )
    out2 = run_worker_loop(_sup(run_root, max_chunks=5), 0)
    assert out2["exit_reason"] == "draining"
    assert out2["chunks_published"] == 0


# ----------------------------------------------------------------------
# backpressure
# ----------------------------------------------------------------------


def test_backpressure_stops_generation(tmp_path: Path):
    run_root = _init(tmp_path, max_ready_shards=1)
    _advance(run_root, RunPhase.IMITATION_COLLECT)
    # ready に dummy shard を置いて backpressure を誘発
    dummy = run_root / "imitation" / "ready" / "dummy"
    dummy.mkdir(parents=True, exist_ok=True)
    write_json_atomic(dummy / "manifest.json", {"x": 1})
    out = run_worker_loop(_sup(run_root, max_chunks=5, max_poll_iterations=2), 0)
    assert out["exit_reason"] == "max_poll_iterations"
    assert out["chunks_published"] == 0


# ----------------------------------------------------------------------
# supervisor subprocess integration
# ----------------------------------------------------------------------


def test_supervisor_spawns_worker_and_generates(tmp_path: Path):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT)
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "local" / "stage3" / "distributed_actor_supervisor.py"
    )
    config = SupervisorConfig(
        run_root=str(run_root), actor_id="host_int", workers=1,
        max_chunks=1, max_runtime_sec=120.0, worker_script=str(script),
    )
    summary = Supervisor(config).run()
    assert summary["ok"] is True
    ready = list((run_root / "imitation" / "ready").iterdir())
    assert len(ready) >= 1


def test_supervisor_requires_worker_script(tmp_path: Path):
    run_root = _init(tmp_path)
    import pytest

    with pytest.raises(ValueError, match="worker_script"):
        Supervisor(SupervisorConfig(run_root=str(run_root)))


# ----------------------------------------------------------------------
# persistent chunk counter (restart seed safety)
# ----------------------------------------------------------------------


def test_chunk_counter_initial_and_atomic_increment(tmp_path: Path):
    run_root = _init(tmp_path)
    dirs = _SupDirs.from_root(run_root)
    dirs.ensure()
    wid = worker_id("host_a", 0)
    assert read_next_chunk_index(dirs, wid) == 0
    assert reserve_chunk_index(dirs, wid) == 0
    assert read_next_chunk_index(dirs, wid) == 1
    assert reserve_chunk_index(dirs, wid) == 1
    assert reserve_chunk_index(dirs, wid) == 2
    assert read_next_chunk_index(dirs, wid) == 3
    # counter file は worker ごとに独立
    other = worker_id("host_a", 1)
    assert read_next_chunk_index(dirs, other) == 0


def test_chunk_index_increases_after_worker_restart(tmp_path: Path):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT)
    # 1 回目の worker invocation
    run_worker_loop(_sup(run_root, max_chunks=1), 0)
    m1 = read_json(next((run_root / "imitation" / "ready").iterdir()) / "manifest.json")
    assert m1["chunk_index"] == 0
    # worker 再起動 (= 同 run_root で再度 run_worker_loop) → chunk_index は増える
    run_worker_loop(_sup(run_root, max_chunks=1), 0)
    indices = sorted(
        read_json(d / "manifest.json")["chunk_index"]
        for d in (run_root / "imitation" / "ready").iterdir()
    )
    assert indices == [0, 1]


def test_derived_seed_differs_after_restart(tmp_path: Path):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT)
    run_worker_loop(_sup(run_root, max_chunks=1), 0)
    run_worker_loop(_sup(run_root, max_chunks=1), 0)
    seed_starts = sorted(
        read_json(d / "manifest.json")["seed_start"]
        for d in (run_root / "imitation" / "ready").iterdir()
    )
    # 2 chunk の base seed (= seed_start) が異なる = 同じ対局を再生成していない
    assert seed_starts[0] != seed_starts[1]


def test_crashed_chunk_index_not_reused(tmp_path: Path, monkeypatch):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT)
    import mahjong_agent.distributed.actor_supervisor as mod

    real = mod.generate_imitation_chunk
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            # 予約済み index 0 で crash (publish しない)
            from mahjong_agent.distributed.actor_supervisor import _ChunkResult
            return _ChunkResult(False, None, 1, 0, 1, 0.0)
        return real(**kw)

    monkeypatch.setattr(mod, "generate_imitation_chunk", flaky)
    run_worker_loop(_sup(run_root, max_chunks=2), 0)
    # crash した index 0 は欠番。publish された shard は index 1。
    ready = list((run_root / "imitation" / "ready").iterdir())
    assert len(ready) == 1
    assert read_json(ready[0] / "manifest.json")["chunk_index"] == 1
    dirs = _SupDirs.from_root(run_root)
    assert read_next_chunk_index(dirs, worker_id("host_a", 0)) == 2


# ----------------------------------------------------------------------
# supervisor backoff (does not exit while waiting for restart)
# ----------------------------------------------------------------------


class _FakeProc:
    def __init__(self, rc):
        self._rc = rc

    def poll(self):
        return self._rc

    def terminate(self):
        pass


def test_supervisor_does_not_exit_while_waiting_for_restart():
    spawned = []

    def spawn(index):
        spawned.append(index)
        return _FakeProc(None)  # 新 process は alive

    # worker が crash 終了 (rc=1)
    workers = {0: _WorkerProc(index=0, proc=_FakeProc(1))}
    keep = _supervise_step(workers, terminal=False, now=0.0, spawn=spawn)
    assert keep is True  # backoff 待機中なので終了しない
    assert workers[0].status == "waiting_for_restart"
    assert workers[0].next_restart_at == compute_backoff(1)
    assert spawned == []  # まだ再起動していない


def test_supervisor_restarts_after_backoff():
    spawned = []

    def spawn(index):
        spawned.append(index)
        return _FakeProc(None)

    workers = {0: _WorkerProc(index=0, proc=_FakeProc(1))}
    _supervise_step(workers, terminal=False, now=0.0, spawn=spawn)
    # backoff 到達後
    keep = _supervise_step(
        workers, terminal=False, now=workers[0].next_restart_at + 1.0, spawn=spawn
    )
    assert keep is True
    assert workers[0].status == "running"
    assert workers[0].restarts == 1
    assert spawned == [0]


def test_supervisor_stops_on_all_done():
    workers = {
        0: _WorkerProc(index=0, proc=_FakeProc(0)),
        1: _WorkerProc(index=1, proc=_FakeProc(0)),
    }
    keep = _supervise_step(
        workers, terminal=False, now=0.0, spawn=lambda i: _FakeProc(None)
    )
    assert keep is False
    assert all(wp.status == "done" for wp in workers.values())


def test_supervisor_stops_on_terminal_even_if_crashed():
    workers = {0: _WorkerProc(index=0, proc=_FakeProc(1))}
    keep = _supervise_step(
        workers, terminal=True, now=0.0, spawn=lambda i: _FakeProc(None)
    )
    assert keep is False
    assert workers[0].status == "done"


# ----------------------------------------------------------------------
# hidden-info guard
# ----------------------------------------------------------------------


def test_supervisor_source_does_not_reference_hidden_state():
    import mahjong_agent.distributed.actor_supervisor as mod

    text = Path(mod.__file__).read_text(encoding="utf-8")
    for tok in (
        "env.hands", "env.wall", "env.state", "obs.hands[",
        "full_state", "private_hand", "mjai_log",
    ):
        assert tok not in text, f"supervisor source references {tok!r}"
