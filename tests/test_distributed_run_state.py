"""Tests for distributed run config + phase state machine."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mahjong_agent.distributed.run_state import (
    DistributedRunConfig,
    EvaluatorRunConfig,
    PPORunConfig,
    RunPhase,
    config_sha256,
    default_run_config,
    initialize_run_state,
    is_transition_allowed,
    read_run_config,
    read_run_state,
    run_config_path,
    state_path,
    transition_run_state,
    validate_run_config,
    write_run_config,
)


def _cfg(**ppo_kw) -> DistributedRunConfig:
    base = default_run_config("run_001", seed_namespace="ns_a")
    if ppo_kw:
        base = DistributedRunConfig(
            run_id=base.run_id,
            seed_namespace=base.seed_namespace,
            imitation=base.imitation,
            ppo=PPORunConfig(**{**base.ppo.__dict__, **ppo_kw}),
            evaluator=base.evaluator,
            operations=base.operations,
        )
    return base


# ----------------------------------------------------------------------
# config round-trip / validation
# ----------------------------------------------------------------------


def test_config_round_trip(tmp_path: Path):
    cfg = _cfg()
    d = cfg.to_dict()
    assert isinstance(d["evaluator"]["modes"], list)
    back = DistributedRunConfig.from_dict(json.loads(json.dumps(d)))
    assert back == cfg
    assert isinstance(back.evaluator.modes, tuple)
    assert config_sha256(back) == config_sha256(cfg)


def test_validate_rejects_empty_run_id():
    cfg = DistributedRunConfig(run_id="", seed_namespace="ns")
    with pytest.raises(ValueError, match="run_id"):
        validate_run_config(cfg)


def test_validate_rejects_chunk_gt_target():
    cfg = default_run_config("r")
    cfg = DistributedRunConfig(
        run_id="r", seed_namespace="r",
        imitation=type(cfg.imitation)(target_games=10, chunk_games=20, epochs=1),
    )
    with pytest.raises(ValueError, match="chunk_games"):
        validate_run_config(cfg)


def test_validate_rejects_min_gt_max_samples():
    with pytest.raises(ValueError, match="min_samples_per_update"):
        validate_run_config(_cfg(min_samples_per_update=500, max_samples_per_update=100))


def test_validate_rejects_non_unit_temperature():
    with pytest.raises(ValueError, match="temperature"):
        validate_run_config(_cfg(temperature=0.5))


def test_validate_rejects_negative_policy_lag():
    with pytest.raises(ValueError, match="max_policy_lag"):
        validate_run_config(_cfg(max_policy_lag=-1))


def test_validate_rejects_unknown_eval_mode():
    cfg = default_run_config("r")
    cfg = DistributedRunConfig(
        run_id="r", seed_namespace="r",
        evaluator=EvaluatorRunConfig(modes=("greedy", "bogus")),
    )
    with pytest.raises(ValueError, match="unknown"):
        validate_run_config(cfg)


# ----------------------------------------------------------------------
# immutable config write
# ----------------------------------------------------------------------


def test_write_run_config_same_content_is_allowed(tmp_path: Path):
    p = run_config_path(tmp_path)
    cfg = _cfg()
    write_run_config(p, cfg)
    write_run_config(p, cfg)  # 同一内容の再書込は許容
    assert read_run_config(p) == cfg


def test_write_run_config_different_content_rejected(tmp_path: Path):
    p = run_config_path(tmp_path)
    write_run_config(p, _cfg())
    with pytest.raises(ValueError, match="different content"):
        write_run_config(p, _cfg(learning_rate=1e-3))


# ----------------------------------------------------------------------
# state round-trip
# ----------------------------------------------------------------------


def test_initialize_creates_dirs_and_state(tmp_path: Path):
    cfg = _cfg()
    state = initialize_run_state(tmp_path, cfg)
    assert state.phase == RunPhase.INITIALIZING
    assert state.phase_generation == 0
    assert state.policy_version is None
    assert state.config_sha256 == config_sha256(cfg)
    # directory layout
    for sub in (
        "control", "imitation", "rollouts/pending", "rollouts/ready",
        "rollouts/consumed", "rollouts/rejected", "checkpoints",
        "metrics/actors", "eval", "best",
    ):
        assert (tmp_path / sub).is_dir()
    assert run_config_path(tmp_path).is_file()
    assert state_path(tmp_path).is_file()
    # round trip
    assert read_run_state(state_path(tmp_path)) == state


def test_initialize_idempotent_does_not_reset(tmp_path: Path):
    cfg = _cfg()
    initialize_run_state(tmp_path, cfg)
    # phase を進める
    transition_run_state(
        tmp_path, expected_phase=RunPhase.INITIALIZING,
        next_phase=RunPhase.IMITATION_COLLECT, config=cfg,
    )
    # 再 init しても state は壊れない (collect のまま)
    again = initialize_run_state(tmp_path, cfg)
    assert again.phase == RunPhase.IMITATION_COLLECT
    assert again.phase_generation == 1


def test_initialize_with_different_config_rejected(tmp_path: Path):
    initialize_run_state(tmp_path, _cfg())
    with pytest.raises(ValueError):
        initialize_run_state(tmp_path, _cfg(learning_rate=1e-3))


# ----------------------------------------------------------------------
# transitions
# ----------------------------------------------------------------------


def test_is_transition_allowed_table():
    assert is_transition_allowed(RunPhase.INITIALIZING, RunPhase.IMITATION_COLLECT)
    assert is_transition_allowed(RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN)
    assert is_transition_allowed(RunPhase.IMITATION_TRAIN, RunPhase.PPO)
    assert is_transition_allowed(RunPhase.PPO, RunPhase.DRAINING)
    assert is_transition_allowed(RunPhase.DRAINING, RunPhase.STOPPED)
    # illegal
    assert not is_transition_allowed(RunPhase.INITIALIZING, RunPhase.PPO)
    assert not is_transition_allowed(RunPhase.PPO, RunPhase.IMITATION_TRAIN)
    # resume-only
    assert not is_transition_allowed(RunPhase.FAILED, RunPhase.INITIALIZING)
    assert is_transition_allowed(
        RunPhase.FAILED, RunPhase.INITIALIZING, allow_resume=True
    )
    assert is_transition_allowed(
        RunPhase.STOPPED, RunPhase.INITIALIZING, allow_resume=True
    )


def test_valid_transition_increments_generation(tmp_path: Path):
    cfg = _cfg()
    initialize_run_state(tmp_path, cfg)
    s1 = transition_run_state(
        tmp_path, expected_phase=RunPhase.INITIALIZING,
        next_phase=RunPhase.IMITATION_COLLECT, config=cfg,
    )
    assert s1.phase == RunPhase.IMITATION_COLLECT
    assert s1.phase_generation == 1
    assert s1.previous_phase == RunPhase.INITIALIZING
    s2 = transition_run_state(
        tmp_path, expected_phase=RunPhase.IMITATION_COLLECT,
        next_phase=RunPhase.IMITATION_TRAIN, config=cfg,
    )
    assert s2.phase_generation == 2


def test_same_phase_update_keeps_generation(tmp_path: Path):
    cfg = _cfg()
    initialize_run_state(tmp_path, cfg)
    transition_run_state(
        tmp_path, expected_phase=RunPhase.INITIALIZING,
        next_phase=RunPhase.IMITATION_COLLECT, config=cfg,
    )
    # 同 phase 内で policy_version / message を更新
    s = transition_run_state(
        tmp_path, expected_phase=RunPhase.IMITATION_COLLECT,
        next_phase=RunPhase.IMITATION_COLLECT, config=cfg,
        policy_version=5, message="progress",
    )
    assert s.phase == RunPhase.IMITATION_COLLECT
    assert s.phase_generation == 1  # 不変
    assert s.policy_version == 5
    assert s.message == "progress"


def test_invalid_transition_rejected(tmp_path: Path):
    cfg = _cfg()
    initialize_run_state(tmp_path, cfg)
    with pytest.raises(ValueError, match="illegal phase transition"):
        transition_run_state(
            tmp_path, expected_phase=RunPhase.INITIALIZING,
            next_phase=RunPhase.PPO, config=cfg,
        )


def test_expected_phase_mismatch_rejected(tmp_path: Path):
    cfg = _cfg()
    initialize_run_state(tmp_path, cfg)
    with pytest.raises(ValueError, match="expected_phase mismatch"):
        transition_run_state(
            tmp_path, expected_phase=RunPhase.PPO,
            next_phase=RunPhase.DRAINING, config=cfg,
        )


def test_config_hash_mismatch_rejected(tmp_path: Path):
    cfg = _cfg()
    initialize_run_state(tmp_path, cfg)
    # 別 config を渡すと hash mismatch
    with pytest.raises(ValueError, match="config hash mismatch"):
        transition_run_state(
            tmp_path, expected_phase=RunPhase.INITIALIZING,
            next_phase=RunPhase.IMITATION_COLLECT, config=_cfg(learning_rate=1e-3),
        )


def test_resume_transition_requires_allow_resume(tmp_path: Path):
    cfg = _cfg()
    initialize_run_state(tmp_path, cfg)
    transition_run_state(
        tmp_path, expected_phase=RunPhase.INITIALIZING,
        next_phase=RunPhase.FAILED, config=cfg, failure_reason="boom",
    )
    with pytest.raises(ValueError, match="illegal phase transition"):
        transition_run_state(
            tmp_path, expected_phase=RunPhase.FAILED,
            next_phase=RunPhase.INITIALIZING, config=cfg,
        )
    resumed = transition_run_state(
        tmp_path, expected_phase=RunPhase.FAILED,
        next_phase=RunPhase.INITIALIZING, config=cfg, allow_resume=True,
    )
    assert resumed.phase == RunPhase.INITIALIZING


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def test_init_show_cli(tmp_path: Path, capsys):
    import importlib.util
    import sys

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "local" / "stage3" / "distributed_run.py"
    )
    spec = importlib.util.spec_from_file_location("distributed_run", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    cfg_path = tmp_path / "run_config.json"
    cfg_path.write_text(
        json.dumps(default_run_config("run_cli", seed_namespace="ns").to_dict()),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    rc = mod.main(["init", "--run-root", str(run_root), "--config", str(cfg_path)])
    assert rc == 0
    assert (run_root / "run_config.json").is_file()
    assert (run_root / "control" / "state.json").is_file()
    capsys.readouterr()

    rc2 = mod.main(["show", "--run-root", str(run_root)])
    assert rc2 == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run_id"] == "run_cli"
    assert out["config_hash_matches_state"] is True
    assert out["state"]["phase"] == "initializing"


# ----------------------------------------------------------------------
# hidden-info guard
# ----------------------------------------------------------------------


def test_run_state_source_does_not_reference_hidden_state():
    import mahjong_agent.distributed.run_state as mod

    text = Path(mod.__file__).read_text(encoding="utf-8")
    for tok in (
        "env.hands",
        "env.wall",
        "env.state",
        "obs.hands[",
        "full_state",
        "private_hand",
        "mjai_log",
    ):
        assert tok not in text, f"run_state source references {tok!r}"
