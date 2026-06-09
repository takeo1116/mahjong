"""Tests for the distributed actor/learner local smoke driver.

実 subprocess を起動するので軽量設定 (actor 1, games 2, learner update 1,
YON_IKKYOKU) で 1 回だけ完走させる。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "local" / "stage3" / "distributed_smoke.py"
)


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("distributed_smoke", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # dataclass の string annotation 解決のため exec 前に sys.modules へ登録する。
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_smoke_runs_actor_and_learner_and_advances_version(tmp_path: Path):
    smoke = _load_smoke_module()
    config = smoke.SmokeConfig(
        run_root=str(tmp_path / "run"),
        actor_count=1,
        games_per_actor=2,
        chunk_games=1,
        learner_updates=1,
        device="cpu",
        game_type_name="YON_IKKYOKU",
        timeout_sec=180.0,
    )
    summary = smoke.run_smoke(config)

    assert summary["ok"] is True
    assert all(rc == 0 for rc in summary["returncodes"].values())
    assert summary["initial_policy_version"] == 1
    # learner が 1 update 進めて policy_version が +1
    assert summary["learner_updates"] == 1
    assert summary["final_policy_version"] == 2
    # actor が ready shard を生成し、learner が消費した
    assert summary["published_shards"] >= 1
    assert summary["consumed_shards"] >= 1
    assert summary["pending_remaining"] == 0
    assert summary["total_samples_consumed"] > 0

    # latest.json が新 checkpoint を指す
    run_root = Path(summary["run_root"])
    latest = json.loads(
        (run_root / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    assert latest["policy_version"] == 2
    assert latest["checkpoint_path"] == "checkpoints/policy_000002.pt"
    assert (run_root / "checkpoints" / "policy_000002.pt").is_file()


def test_create_initial_checkpoint_writes_registry(tmp_path: Path):
    smoke = _load_smoke_module()
    entry = smoke.create_initial_checkpoint(tmp_path / "run")
    assert entry.policy_version == 1
    latest = json.loads(
        (tmp_path / "run" / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    assert latest["policy_version"] == 1
    assert latest["observation_dim"] == entry.observation_dim
    assert (tmp_path / "run" / "checkpoints" / "policy_000001.pt").is_file()


def test_smoke_source_does_not_reference_hidden_state():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    for tok in (
        "env.hands",
        "env.wall",
        "env.state",
        "obs.hands[",
        "full_state",
        "private_hand",
        "mjai_log",
    ):
        assert tok not in text, f"smoke source references {tok!r}"
