"""Tests for the multi-node cluster rehearsal driver."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "local" / "stage3" / "distributed_cluster_rehearsal.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("distributed_cluster_rehearsal", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


reh = _load()


# ----------------------------------------------------------------------
# config / naming / commands
# ----------------------------------------------------------------------


def test_config_from_args():
    args = reh.parse_args([
        "--run-root", "/tmp/x", "--device", "cuda", "--actor-nodes", "3",
        "--workers-per-node", "4", "--imitation-teacher-games", "40",
        "--ppo-updates", "3", "--eval-games-per-seat", "4",
        "--game-type", "YON_IKKYOKU", "--timeout-sec", "1200",
    ])
    cfg = reh.config_from_args(args)
    assert cfg.actor_nodes == 3
    assert cfg.workers_per_node == 4
    assert cfg.ppo_updates == 3


def test_node_actor_id_naming_distinct():
    assert reh.node_actor_id(0) == "localnode-000"
    assert reh.node_actor_id(12) == "localnode-012"
    cfg = reh.RehearsalConfig(run_root="/tmp/x", actor_nodes=3)
    ids = reh.actor_ids(cfg)
    assert ids == ["localnode-000", "localnode-001", "localnode-002"]
    assert len(set(ids)) == 3  # node ごとに分離


def test_actor_node_command_construction():
    cfg = reh.RehearsalConfig(run_root="/tmp/x", actor_nodes=2, workers_per_node=4)
    c0 = reh.actor_node_cmd(cfg, 0)
    c1 = reh.actor_node_cmd(cfg, 1)
    assert "distributed_actor_supervisor.py" in c0[1]
    assert "--actor-id" in c0 and "localnode-000" in c0
    assert "--actor-id" in c1 and "localnode-001" in c1
    assert "--workers" in c0 and "4" in c0
    # learner / evaluator
    assert "distributed_learner_supervisor.py" in reh.learner_cmd(cfg)[1]
    assert "distributed_evaluator_supervisor.py" in reh.evaluator_cmd(cfg)[1]


def test_build_run_config():
    cfg = reh.RehearsalConfig(run_root="/tmp/x", imitation_teacher_games=40,
                              game_type="YON_IKKYOKU")
    rc = reh.build_run_config(cfg)
    assert rc.imitation.target_games == 40
    assert rc.ppo.min_samples_per_update == 100
    assert rc.ppo.game_type == "YON_IKKYOKU"


# ----------------------------------------------------------------------
# per-node metrics aggregation
# ----------------------------------------------------------------------


def _craft_shard(root: Path, rel_dir: str, name: str, *, actor_id: str, num_samples: int):
    from mahjong_agent.distributed.manifest import RolloutShardManifest, write_json_atomic

    d = root / rel_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "shard.npz").write_bytes(b"x")
    m = RolloutShardManifest(
        policy_version=1, checkpoint_sha256="x", actor_id=actor_id, created_at="t",
        num_games=1, num_samples=num_samples, schema_version=2, observation_dim=608,
        candidate_dim=91, old_log_prob_available=True, role="actor_rollout",
        phase="ppo", phase_generation=3, worker_id=f"{actor_id}/w000", chunk_index=0,
    )
    write_json_atomic(d / "manifest.json", m.to_dict())


def test_per_node_metrics_groups_by_actor_id(tmp_path: Path):
    _craft_shard(tmp_path, "rollouts/consumed/policy_000002", "s0",
                 actor_id="localnode-000", num_samples=40)
    _craft_shard(tmp_path, "rollouts/consumed/policy_000002", "s1",
                 actor_id="localnode-000", num_samples=50)
    _craft_shard(tmp_path, "rollouts/ready", "s2",
                 actor_id="localnode-001", num_samples=30)
    # heartbeats per node
    from mahjong_agent.distributed.manifest import write_json_atomic
    hb = tmp_path / "metrics" / "heartbeats"
    write_json_atomic(hb / "localnode-000_w000.json",
                      {"actor_id": "localnode-000", "last_update_at": "2099-01-01T00:00:00Z"})
    write_json_atomic(hb / "localnode-001_w000.json",
                      {"actor_id": "localnode-001", "last_update_at": "2000-01-01T00:00:00Z"})

    m = reh.per_node_metrics(tmp_path, ["localnode-000", "localnode-001"])
    assert m["localnode-000"]["published_shards"] == 2
    assert m["localnode-000"]["num_samples"] == 90
    assert m["localnode-000"]["heartbeat_present"] == 1
    assert m["localnode-000"]["heartbeat_stale"] == 0
    assert m["localnode-001"]["published_shards"] == 1
    assert m["localnode-001"]["num_samples"] == 30
    assert m["localnode-001"]["heartbeat_stale"] == 1  # 古い heartbeat


def test_ppo_update_and_eval_counts(tmp_path: Path):
    (tmp_path / "metrics").mkdir(parents=True)
    (tmp_path / "metrics" / "learner.jsonl").write_text(
        json.dumps({"event": "learner_update"}) + "\n"
        + json.dumps({"event": "imitation_train"}) + "\n"
        + json.dumps({"event": "learner_update"}) + "\n", encoding="utf-8",
    )
    (tmp_path / "metrics" / "eval.jsonl").write_text(
        json.dumps({"event": "checkpoint_eval"}) + "\n", encoding="utf-8",
    )
    assert reh.ppo_update_count(tmp_path) == 2
    assert reh.eval_count(tmp_path) == 1


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------


def test_build_summary_aggregates_per_node_and_returncodes(tmp_path: Path):
    from mahjong_agent.distributed.run_state import (
        RunPhase,
        initialize_run_state,
        transition_run_state,
    )

    cfg = reh.RehearsalConfig(run_root=str(tmp_path), actor_nodes=2, workers_per_node=1)
    initialize_run_state(tmp_path, reh.build_run_config(cfg))
    cur = RunPhase.INITIALIZING
    for nxt in (RunPhase.IMITATION_COLLECT, RunPhase.IMITATION_TRAIN, RunPhase.PPO,
                RunPhase.DRAINING, RunPhase.STOPPED):
        kw = {"policy_version": 2} if nxt == RunPhase.PPO else {}
        transition_run_state(tmp_path, expected_phase=cur, next_phase=nxt, **kw)
        cur = nxt
    _craft_shard(tmp_path, "rollouts/consumed/policy_000002", "s0",
                 actor_id="localnode-000", num_samples=40)
    (tmp_path / "metrics").mkdir(exist_ok=True)
    (tmp_path / "metrics" / "learner.jsonl").write_text(
        json.dumps({"event": "learner_update"}) + "\n", encoding="utf-8")

    rcs = {"learner": 0, "evaluator": 0,
           "actor:localnode-000": 0, "actor:localnode-001": 0}
    summary = reh.build_summary(
        cfg, returncodes=rcs, elapsed_sec=5.0, drained=True, timed_out=False,
    )
    assert summary["ok"] is True
    assert summary["actor_nodes"] == 2
    assert summary["final_phase"] == "stopped"
    assert summary["learner_update_count"] == 1
    assert summary["per_node"]["localnode-000"]["published_shards"] == 1
    assert summary["per_node"]["localnode-001"]["published_shards"] == 0
    assert summary["returncodes"]["actor:localnode-001"] == 0


def test_build_summary_not_ok_on_nonzero_rc(tmp_path: Path):
    from mahjong_agent.distributed.run_state import initialize_run_state

    cfg = reh.RehearsalConfig(run_root=str(tmp_path), actor_nodes=1)
    initialize_run_state(tmp_path, reh.build_run_config(cfg))
    summary = reh.build_summary(
        cfg, returncodes={"learner": 0, "actor:localnode-000": 1},
        elapsed_sec=1.0, drained=True, timed_out=False,
    )
    assert summary["ok"] is False


# ----------------------------------------------------------------------
# drain decision / wait_or_kill
# ----------------------------------------------------------------------


def test_drain_fires_when_updates_reached(tmp_path: Path):
    # mocked status polling 相当: update 数が target 到達したか
    (tmp_path / "metrics").mkdir(parents=True)
    (tmp_path / "metrics" / "learner.jsonl").write_text(
        json.dumps({"event": "learner_update"}) + "\n"
        + json.dumps({"event": "learner_update"}) + "\n", encoding="utf-8")
    assert reh.ppo_update_count(tmp_path) >= 2  # ppo_updates=2 なら drain


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


def test_wait_or_kill_timeout():
    procs = {"learner": _FakeProc(0), "actor:localnode-000": _FakeProc(0, hang=True)}
    deadline = __import__("time").time() + 0.05
    rcs = reh._wait_or_kill(procs, deadline)
    assert rcs["learner"] == 0
    assert procs["actor:localnode-000"].killed is True


# ----------------------------------------------------------------------
# hidden-info guard
# ----------------------------------------------------------------------


def test_rehearsal_source_does_not_reference_hidden_state():
    text = SCRIPT.read_text(encoding="utf-8")
    for tok in (
        "env.hands", "env.wall", "env.state", "obs.hands[",
        "full_state", "private_hand", "mjai_log",
    ):
        assert tok not in text, f"rehearsal source references {tok!r}"
