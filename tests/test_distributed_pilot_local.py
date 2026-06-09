"""Tests for the single-PC imitation→PPO pilot driver."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "local" / "stage3" / "distributed_pilot_local.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("distributed_pilot_local", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pilot = _load()


# ----------------------------------------------------------------------
# config / command construction
# ----------------------------------------------------------------------


def test_config_from_args():
    args = pilot.parse_args([
        "--run-root", "/tmp/x", "--device", "cuda", "--actor-workers", "8",
        "--imitation-teacher-games", "30", "--ppo-updates", "3",
        "--eval-games-per-seat", "5", "--modes", "greedy,low_temp",
        "--game-type", "YON_TONPUSEN", "--timeout-sec", "120", "--cleanup",
    ])
    cfg = pilot.config_from_args(args)
    assert cfg.device == "cuda"
    assert cfg.actor_workers == 8
    assert cfg.imitation_teacher_games == 30
    assert cfg.ppo_updates == 3
    assert cfg.modes == ("greedy", "low_temp")
    assert cfg.cleanup is True


def test_build_run_config():
    cfg = pilot.PilotConfig(run_root="/tmp/x", imitation_teacher_games=12,
                            game_type="YON_IKKYOKU", eval_games_per_seat=3)
    rc = pilot.build_run_config(cfg)
    assert rc.imitation.target_games == 12
    assert rc.ppo.game_type == "YON_IKKYOKU"
    assert rc.ppo.temperature == 1.0
    assert rc.evaluator.eval_games_per_seat == 3


def test_command_construction():
    cfg = pilot.PilotConfig(run_root="/tmp/x", device="cpu", actor_workers=4,
                            modes=("greedy", "stochastic"), timeout_sec=300)
    lc = pilot.learner_cmd(cfg)
    assert "distributed_learner_supervisor.py" in lc[1]
    assert "--run-root" in lc and "/tmp/x" in lc
    assert "--device" in lc and "cpu" in lc
    ac = pilot.actor_cmd(cfg)
    assert "distributed_actor_supervisor.py" in ac[1]
    assert "--workers" in ac and "4" in ac
    assert "--actor-id" in ac and "pilot" in ac
    ec = pilot.evaluator_cmd(cfg)
    assert "distributed_evaluator_supervisor.py" in ec[1]
    assert "--modes" in ec and "greedy,stochastic" in ec


# ----------------------------------------------------------------------
# progress counting / summary aggregation
# ----------------------------------------------------------------------


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_ppo_update_count_and_eval_count(tmp_path: Path):
    _write_jsonl(
        tmp_path / "metrics" / "learner.jsonl",
        [{"event": "learner_update"}, {"event": "learner_dry_run"},
         {"event": "learner_update"}],
    )
    _write_jsonl(
        tmp_path / "metrics" / "eval.jsonl",
        [{"event": "checkpoint_eval"}, {"event": "eval_dry_run"}],
    )
    assert pilot.ppo_update_count(tmp_path) == 2
    assert pilot.eval_count(tmp_path) == 1


def test_build_summary_ok(tmp_path: Path):
    from mahjong_agent.distributed.run_state import (
        RunPhase,
        initialize_run_state,
        transition_run_state,
    )

    # tiny run_root
    rc = pilot.build_run_config(pilot.PilotConfig(run_root=str(tmp_path)))
    initialize_run_state(tmp_path, rc)
    cur = RunPhase.INITIALIZING
    for nxt in (RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN, RunPhase.PPO):
        kw = {"policy_version": 1} if nxt == RunPhase.PPO else {}
        transition_run_state(tmp_path, expected_phase=cur, next_phase=nxt, **kw)
        cur = nxt
    _write_jsonl(tmp_path / "metrics" / "learner.jsonl", [{"event": "learner_update"}])
    _write_jsonl(tmp_path / "metrics" / "eval.jsonl", [{"event": "checkpoint_eval"}])

    summary = pilot.build_summary(
        tmp_path, returncodes={"learner": 0, "actor": 0, "evaluator": 0},
        elapsed_sec=1.0, drained=True, timed_out=False,
    )
    assert summary["ok"] is True
    assert summary["final_phase"] == "ppo"
    assert summary["final_policy_version"] == 1
    assert summary["learner_update_count"] == 1
    assert summary["eval_count"] == 1
    assert summary["learner_journal_present"] is False
    assert summary["returncodes"]["learner"] == 0


def test_build_summary_not_ok_on_journal_or_rc(tmp_path: Path):
    from mahjong_agent.distributed.run_state import initialize_run_state

    initialize_run_state(tmp_path, pilot.build_run_config(pilot.PilotConfig(run_root=str(tmp_path))))
    # journal 残存 → ok=False
    (tmp_path / "control" / "learner_transaction.json").write_text("{}", encoding="utf-8")
    s1 = pilot.build_summary(
        tmp_path, returncodes={"learner": 0}, elapsed_sec=1.0, drained=True, timed_out=False,
    )
    assert s1["ok"] is False
    (tmp_path / "control" / "learner_transaction.json").unlink()
    # 非 0 returncode → ok=False
    s2 = pilot.build_summary(
        tmp_path, returncodes={"learner": 1}, elapsed_sec=1.0, drained=True, timed_out=False,
    )
    assert s2["ok"] is False
    # timed_out → ok=False
    s3 = pilot.build_summary(
        tmp_path, returncodes={"learner": 0}, elapsed_sec=1.0, drained=True, timed_out=True,
    )
    assert s3["ok"] is False


# ----------------------------------------------------------------------
# wait_or_kill (timeout kill)
# ----------------------------------------------------------------------


class _FakeProc:
    def __init__(self, rc, *, hang=False):
        self._rc = rc
        self._hang = hang
        self.killed = False

    def wait(self, timeout=None):
        if self._hang and not self.killed:
            import subprocess
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
        return self._rc

    def kill(self):
        self.killed = True

    def poll(self):
        return self._rc


def test_wait_or_kill_normal(monkeypatch):
    procs = {"a": _FakeProc(0), "b": _FakeProc(0)}
    deadline = __import__("time").time() + 60
    rcs = pilot._wait_or_kill(procs, deadline)
    assert rcs == {"a": 0, "b": 0}


def test_wait_or_kill_timeout_kills():
    procs = {"hung": _FakeProc(0, hang=True)}
    deadline = __import__("time").time() + 0.1
    rcs = pilot._wait_or_kill(procs, deadline)
    assert procs["hung"].killed is True
    assert rcs["hung"] in (0, -9)  # kill 後 wait は 0 を返す fake


# ----------------------------------------------------------------------
# hidden-info guard
# ----------------------------------------------------------------------


def test_pilot_source_does_not_reference_hidden_state():
    text = SCRIPT.read_text(encoding="utf-8")
    for tok in (
        "env.hands", "env.wall", "env.state", "obs.hands[",
        "full_state", "private_hand", "mjai_log",
    ):
        assert tok not in text, f"pilot source references {tok!r}"
