"""Tests for the phase-aware evaluator supervisor."""
from __future__ import annotations

from pathlib import Path

from mahjong_agent.distributed import evaluator_supervisor as es
from mahjong_agent.distributed.evaluator_supervisor import (
    EvaluatorSupervisorConfig,
    run_evaluator_supervisor,
)
from mahjong_agent.distributed.run_state import (
    DistributedRunConfig,
    ImitationRunConfig,
    PPORunConfig,
    RunPhase,
    initialize_run_state,
    transition_run_state,
)


def _config() -> DistributedRunConfig:
    return DistributedRunConfig(
        run_id="run_es",
        seed_namespace="ns_es",
        imitation=ImitationRunConfig(target_games=2, chunk_games=1, epochs=1),
        ppo=PPORunConfig(game_type="YON_IKKYOKU"),
    )


def _init(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    initialize_run_state(run_root, _config())
    return run_root


def _advance(run_root: Path, *phases, policy_version_on_last=None):
    cur = RunPhase.INITIALIZING
    for i, nxt in enumerate(phases):
        kw = {}
        if policy_version_on_last is not None and i == len(phases) - 1:
            kw["policy_version"] = policy_version_on_last
        transition_run_state(run_root, expected_phase=cur, next_phase=nxt, **kw)
        cur = nxt


def _cfg(run_root: Path, **kw) -> EvaluatorSupervisorConfig:
    base = dict(run_root=str(run_root), device="cpu", poll_interval_sec=0.01)
    base.update(kw)
    return EvaluatorSupervisorConfig(**base)


def _patch_eval(monkeypatch, calls: list, *, evaluated=1):
    def fake(cfg):
        calls.append(cfg.checkpoint_version)
        return {"evaluated_checkpoints": evaluated, "evaluated_rows": []}

    monkeypatch.setattr(es, "run_evaluator", fake)


# ----------------------------------------------------------------------
# wait phases
# ----------------------------------------------------------------------


def test_waits_in_initializing(tmp_path: Path, monkeypatch):
    run_root = _init(tmp_path)
    calls: list = []
    _patch_eval(monkeypatch, calls)
    out = run_evaluator_supervisor(_cfg(run_root, max_idle_polls=2))
    assert out["exit_reason"] == "max_idle_polls"
    assert out["evaluated_checkpoints"] == 0
    assert calls == []  # eval は呼ばれない


def test_waits_in_imitation_train(tmp_path: Path, monkeypatch):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN)
    calls: list = []
    _patch_eval(monkeypatch, calls)
    out = run_evaluator_supervisor(_cfg(run_root, max_idle_polls=2))
    assert out["exit_reason"] == "max_idle_polls"
    assert calls == []


# ----------------------------------------------------------------------
# ppo: evaluate
# ----------------------------------------------------------------------


def test_evaluates_in_ppo(tmp_path: Path, monkeypatch):
    run_root = _init(tmp_path)
    _advance(
        run_root, RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN, RunPhase.PPO,
        policy_version_on_last=1,
    )
    calls: list = []
    _patch_eval(monkeypatch, calls, evaluated=1)
    out = run_evaluator_supervisor(_cfg(run_root, max_eval_rounds=2))
    assert out["exit_reason"] == "max_eval_rounds"
    assert out["eval_rounds"] == 2
    assert out["evaluated_checkpoints"] == 2
    assert calls == ["unevaluated", "unevaluated"]


def test_ppo_no_unevaluated_waits(tmp_path: Path, monkeypatch):
    run_root = _init(tmp_path)
    _advance(
        run_root, RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN, RunPhase.PPO,
        policy_version_on_last=1,
    )
    calls: list = []
    _patch_eval(monkeypatch, calls, evaluated=0)  # 未評価なし
    out = run_evaluator_supervisor(_cfg(run_root, max_idle_polls=1, max_eval_rounds=0))
    assert out["exit_reason"] == "max_idle_polls"
    # 1 回評価試行 (0 件) してから idle exit
    assert out["evaluated_checkpoints"] == 0
    assert len(calls) >= 1


# ----------------------------------------------------------------------
# draining / terminal
# ----------------------------------------------------------------------


def test_draining_runs_final_eval_then_exits(tmp_path: Path, monkeypatch):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT, RunPhase.DRAINING)
    calls: list = []

    def fake(cfg):
        calls.append(cfg.stop_after_checkpoints)
        return {"evaluated_checkpoints": 3}

    monkeypatch.setattr(es, "run_evaluator", fake)
    out = run_evaluator_supervisor(_cfg(run_root))
    assert out["exit_reason"] == "draining"
    assert out["evaluated_checkpoints"] == 3
    # draining では大きめ stop_after_checkpoints で 1 回まとめて評価
    assert calls == [10_000]


def test_stopped_exits_without_eval(tmp_path: Path, monkeypatch):
    run_root = _init(tmp_path)
    _advance(run_root, RunPhase.IMITATION_COLLECT, RunPhase.DRAINING, RunPhase.STOPPED)
    calls: list = []
    _patch_eval(monkeypatch, calls)
    out = run_evaluator_supervisor(_cfg(run_root))
    assert out["exit_reason"] == "stopped"
    assert calls == []


# ----------------------------------------------------------------------
# hidden-info guard
# ----------------------------------------------------------------------


def test_evaluator_supervisor_source_does_not_reference_hidden_state():
    text = Path(es.__file__).read_text(encoding="utf-8")
    for tok in (
        "env.hands", "env.wall", "env.state", "obs.hands[",
        "full_state", "private_hand", "mjai_log",
    ):
        assert tok not in text, f"evaluator_supervisor source references {tok!r}"
