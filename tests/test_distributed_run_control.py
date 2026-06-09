"""Tests for run control requests, status aggregation, and learner drain handling."""
from __future__ import annotations

from pathlib import Path

import pytest

from mahjong_agent.distributed import run_control as rc
from mahjong_agent.distributed.learner_supervisor import (
    LearnerSupervisorConfig,
    run_learner_supervisor,
)
from mahjong_agent.distributed.run_control import (
    archive_run_request,
    clear_run_request,
    collect_status,
    read_run_request,
    run_request_path,
    write_run_request,
)
from mahjong_agent.distributed.run_state import (
    DistributedRunConfig,
    ImitationRunConfig,
    PPORunConfig,
    RunPhase,
    initialize_run_state,
    read_run_state,
    state_path,
    transition_run_state,
)


def _config() -> DistributedRunConfig:
    return DistributedRunConfig(
        run_id="run_rc",
        seed_namespace="ns_rc",
        imitation=ImitationRunConfig(target_games=2, chunk_games=1, epochs=1),
        ppo=PPORunConfig(game_type="YON_IKKYOKU"),
    )


def _init(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    initialize_run_state(run_root, _config())
    return run_root


# ----------------------------------------------------------------------
# run request
# ----------------------------------------------------------------------


def test_write_and_read_run_request(tmp_path: Path):
    run_root = _init(tmp_path)
    req = write_run_request(run_root, "drain", message="going down")
    assert req["command"] == "drain"
    assert run_request_path(run_root).is_file()
    loaded = read_run_request(run_root)
    assert loaded["command"] == "drain"
    assert loaded["message"] == "going down"


def test_write_invalid_command_rejected(tmp_path: Path):
    run_root = _init(tmp_path)
    with pytest.raises(ValueError, match="invalid run command"):
        write_run_request(run_root, "explode")


def test_clear_run_request(tmp_path: Path):
    run_root = _init(tmp_path)
    write_run_request(run_root, "stop")
    assert clear_run_request(run_root) is True
    assert read_run_request(run_root) is None
    assert clear_run_request(run_root) is False


def test_archive_run_request_moves_file(tmp_path: Path):
    run_root = _init(tmp_path)
    write_run_request(run_root, "drain")
    dest = archive_run_request(run_root, disposition="applied")
    assert dest is not None and dest.is_file()
    assert not run_request_path(run_root).exists()
    assert "applied" in dest.name


# ----------------------------------------------------------------------
# status
# ----------------------------------------------------------------------


def test_collect_status_reads_core_fields(tmp_path: Path):
    run_root = _init(tmp_path)
    write_run_request(run_root, "drain")
    status = collect_status(run_root)
    assert status["run_id"] == "run_rc"
    assert status["phase"] == "initializing"
    assert status["latest_checkpoint"] is None
    assert status["best_checkpoint"] is None
    assert status["shards"]["imitation"]["ready"] == 0
    assert status["shards"]["rollouts"]["ready"] == 0
    assert status["heartbeats"]["count"] == 0
    assert status["learner_journal_present"] is False
    assert status["run_request"]["command"] == "drain"
    # format does not crash
    text = rc.format_status(status)
    assert "run_id:" in text and "phase:" in text


def test_collect_status_counts_shards_and_journal(tmp_path: Path):
    run_root = _init(tmp_path)
    # craft a ready rollout shard dir + a learner journal
    ready = run_root / "rollouts" / "ready" / "s0"
    ready.mkdir(parents=True)
    (ready / "manifest.json").write_text("{}", encoding="utf-8")
    (run_root / "control" / "learner_transaction.json").write_text("{}", encoding="utf-8")
    status = collect_status(run_root)
    assert status["shards"]["rollouts"]["ready"] == 1
    assert status["learner_journal_present"] is True


# ----------------------------------------------------------------------
# learner drain handling (single-writer preserved)
# ----------------------------------------------------------------------


def _ls_cfg(run_root: Path, **kw) -> LearnerSupervisorConfig:
    base = dict(run_root=str(run_root), device="cpu", poll_interval_sec=0.01)
    base.update(kw)
    return LearnerSupervisorConfig(**base)


def test_learner_applies_drain_request_in_ppo(tmp_path: Path):
    run_root = _init(tmp_path)
    # 手で ppo phase (gen3, pv1) にする
    cur = RunPhase.INITIALIZING
    for nxt in (RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN, RunPhase.PPO):
        kw = {"policy_version": 1} if nxt == RunPhase.PPO else {}
        transition_run_state(run_root, expected_phase=cur, next_phase=nxt, **kw)
        cur = nxt
    write_run_request(run_root, "drain")

    out = run_learner_supervisor(_ls_cfg(run_root, max_idle_polls=1))
    assert any("request:drain->draining" in t for t in out["transitions"])
    # learner が drain → draining → stopped まで進める
    assert read_run_state(state_path(run_root)).phase == RunPhase.STOPPED
    # request は archive 済み（残っていない）
    assert read_run_request(run_root) is None
    assert (run_root / "control" / "processed_requests").is_dir()


def test_learner_drain_request_pending_in_imitation_train(tmp_path: Path):
    run_root = _init(tmp_path)
    cur = RunPhase.INITIALIZING
    for nxt in (RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN):
        transition_run_state(run_root, expected_phase=cur, next_phase=nxt)
        cur = nxt
    write_run_request(run_root, "drain")
    # imitation_train では直接 draining に遷移できないので request は pending のまま。
    # （learner は imitation_train で train を試みるが teacher shard が無いので失敗する。
    #   ここでは request が残ることだけ確認するため、train を呼ばずに handler を直接検証）
    state = read_run_state(state_path(run_root))
    label = rc.LEARNER_COMMANDS  # noqa: F841 (確認用)
    from mahjong_agent.distributed.learner_supervisor import _handle_run_request, _LSDirs

    dirs = _LSDirs.from_root(run_root)
    res = _handle_run_request(dirs, _config(), state)
    assert res is None
    assert read_run_request(run_root) is not None  # pending のまま


def test_learner_ignores_unknown_command(tmp_path: Path):
    run_root = _init(tmp_path)
    transition_run_state(
        run_root, expected_phase=RunPhase.INITIALIZING,
        next_phase=RunPhase.IMITATION_COLLECT,
    )
    # request file を不正 command で直接書く（write_run_request は弾くので手書き）
    from mahjong_agent.distributed.manifest import write_json_atomic

    write_json_atomic(
        run_request_path(run_root),
        {"request_version": 1, "command": "bogus", "requested_at": "t",
         "requested_by": "x", "message": ""},
    )
    from mahjong_agent.distributed.learner_supervisor import _handle_run_request, _LSDirs

    dirs = _LSDirs.from_root(run_root)
    state = read_run_state(state_path(run_root))
    res = _handle_run_request(dirs, _config(), state)
    assert res is None
    # ignored archive 済み（state は collect のまま）
    assert read_run_request(run_root) is None
    assert read_run_state(state_path(run_root)).phase == RunPhase.IMITATION_COLLECT


# ----------------------------------------------------------------------
# hidden-info guard
# ----------------------------------------------------------------------


def test_run_control_source_does_not_reference_hidden_state():
    text = Path(rc.__file__).read_text(encoding="utf-8")
    for tok in (
        "env.hands", "env.wall", "env.state", "obs.hands[",
        "full_state", "private_hand", "mjai_log",
    ):
        assert tok not in text, f"run_control source references {tok!r}"
