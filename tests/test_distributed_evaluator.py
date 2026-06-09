"""Tests for the distributed evaluator + best checkpoint selection.

discovery / skip / best-update のロジックは ``evaluate_policy`` を monkeypatch
した軽量 unit test で確認し、実 self-play は YON_IKKYOKU 1 game/seat の tiny
eval を 1 本だけ回す。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from mahjong_agent.distributed import evaluator as ev
from mahjong_agent.distributed.evaluator import (
    EvaluatorConfig,
    _RunDirs,
    discover_targets,
    run_evaluator,
)
from mahjong_agent.distributed.manifest import (
    CheckpointRegistryEntry,
    read_json,
    sha256_file,
    write_json_atomic,
)
from mahjong_agent.encoders import PublicObservationEncoder
from mahjong_agent.models import Stage03Model, Stage03ModelConfig


def _write_ckpt(run_root: Path, *, policy_version: int, update_latest: bool = True) -> None:
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
            "encoder_metadata": {
                "observation_dim": int(cfg.observation_dim),
                "candidate_dim": int(cfg.candidate_dim),
                "enable_hints": True,
            },
        },
        ckpt_path,
    )
    if update_latest:
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


def _fake_result(avg_rank: float) -> dict:
    return {
        "avg_rank": avg_rank,
        "rank_counts": {"1": 1, "2": 0, "3": 0, "4": 0},
        "mean_score": 25000.0,
        "win_rate": 0.2,
        "deal_in_rate": 0.1,
        "riichi_rate": 0.15,
        "call_rate": 0.05,
        "num_games": 4,
        "crash_count": 0,
        "elapsed_sec": 0.01,
    }


def _patch_eval(monkeypatch, avg_rank: float):
    monkeypatch.setattr(
        ev, "evaluate_policy", lambda *a, **k: _fake_result(avg_rank)
    )


def _config(run_root: Path, **kw) -> EvaluatorConfig:
    base = dict(
        run_root=str(run_root),
        checkpoint_version="latest",
        eval_games_per_seat=2,
        mode_names=("greedy",),
        device="cpu",
        seed_start=700_000,
    )
    base.update(kw)
    return EvaluatorConfig(**base)


# ----------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------


def test_discover_targets_latest(tmp_path: Path):
    run_root = tmp_path / "run"
    _write_ckpt(run_root, policy_version=3)
    dirs = _RunDirs.from_root(run_root)
    targets = discover_targets(dirs, _config(run_root), [])
    assert len(targets) == 1
    assert targets[0].policy_version == 3
    assert targets[0].checkpoint_path == "checkpoints/policy_000003.pt"


def test_discover_targets_unevaluated_skips_done(tmp_path: Path):
    run_root = tmp_path / "run"
    _write_ckpt(run_root, policy_version=1)
    _write_ckpt(run_root, policy_version=2)
    dirs = _RunDirs.from_root(run_root)
    cfg = _config(run_root, checkpoint_version="unevaluated", stop_after_checkpoints=10)
    # pv=1 を評価済みとする eval_rows を渡す
    done_row = {
        "event": "checkpoint_eval",
        "policy_version": 1,
        "mode": "greedy",
        "opponent": "rule_based",
        "eval_games_per_seat": 2,
    }
    targets = discover_targets(dirs, cfg, [done_row])
    assert [t.policy_version for t in targets] == [2]


# ----------------------------------------------------------------------
# skip already evaluated
# ----------------------------------------------------------------------


def test_evaluator_skips_already_evaluated(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "run"
    _write_ckpt(run_root, policy_version=1)
    _patch_eval(monkeypatch, 2.3)

    first = run_evaluator(_config(run_root))
    assert first["evaluated_checkpoints"] == 1
    assert len(first["evaluated_rows"]) == 1

    second = run_evaluator(_config(run_root))
    assert second["evaluated_checkpoints"] == 0
    assert second["skipped"] == 1
    assert second["evaluated_rows"] == []
    # eval.jsonl には 1 行のみ
    rows = (run_root / "metrics" / "eval.jsonl").read_text().splitlines()
    assert len([r for r in rows if r.strip()]) == 1


# ----------------------------------------------------------------------
# best selection
# ----------------------------------------------------------------------


def test_best_update_on_lower_avg_rank(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "run"
    _write_ckpt(run_root, policy_version=1)
    _write_ckpt(run_root, policy_version=2)

    _patch_eval(monkeypatch, 2.5)
    run_evaluator(_config(run_root, checkpoint_version="1"))
    best1 = read_json(run_root / "best" / "best.json")
    assert best1["policy_version"] == 1
    assert best1["metric"] == "avg_rank"
    assert best1["metric_value"] == 2.5

    _patch_eval(monkeypatch, 2.0)
    run_evaluator(_config(run_root, checkpoint_version="2"))
    best2 = read_json(run_root / "best" / "best.json")
    assert best2["policy_version"] == 2
    assert best2["metric_value"] == 2.0
    assert (run_root / "best" / "policy_000002.pt").is_file()


def test_best_not_updated_when_worse(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "run"
    _write_ckpt(run_root, policy_version=1)
    _write_ckpt(run_root, policy_version=2)

    _patch_eval(monkeypatch, 2.0)
    run_evaluator(_config(run_root, checkpoint_version="1"))
    _patch_eval(monkeypatch, 2.6)
    run_evaluator(_config(run_root, checkpoint_version="2"))

    best = read_json(run_root / "best" / "best.json")
    assert best["policy_version"] == 1
    assert best["metric_value"] == 2.0
    assert not (run_root / "best" / "policy_000002.pt").exists()


def test_best_tie_keeps_existing(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "run"
    _write_ckpt(run_root, policy_version=1)
    _write_ckpt(run_root, policy_version=2)
    _patch_eval(monkeypatch, 2.0)
    run_evaluator(_config(run_root, checkpoint_version="1"))
    run_evaluator(_config(run_root, checkpoint_version="2"))
    best = read_json(run_root / "best" / "best.json")
    assert best["policy_version"] == 1  # 同点は既存維持


# ----------------------------------------------------------------------
# dry-run
# ----------------------------------------------------------------------


def test_dry_run_does_not_write_best_or_eval(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "run"
    _write_ckpt(run_root, policy_version=1)
    # evaluate_policy が呼ばれたら fail させる
    monkeypatch.setattr(
        ev, "evaluate_policy",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not eval")),
    )
    summary = run_evaluator(_config(run_root, dry_run=True))
    assert summary["dry_run"] is True
    assert len(summary["dry_run_rows"]) == 1
    assert not (run_root / "best" / "best.json").exists()
    eval_jsonl = run_root / "metrics" / "eval.jsonl"
    if eval_jsonl.exists():
        rows = [r for r in eval_jsonl.read_text().splitlines() if r.strip()]
        assert all(json.loads(r).get("event") != "checkpoint_eval" for r in rows)


# ----------------------------------------------------------------------
# real tiny eval
# ----------------------------------------------------------------------


def test_real_tiny_eval_writes_row_and_best(tmp_path: Path):
    run_root = tmp_path / "run"
    _write_ckpt(run_root, policy_version=1)
    summary = run_evaluator(
        _config(
            run_root,
            eval_games_per_seat=1,
            game_type_name="YON_IKKYOKU",
            max_steps_per_game=2000,
        )
    )
    assert summary["evaluated_checkpoints"] == 1
    rows = summary["evaluated_rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "checkpoint_eval"
    assert row["policy_version"] == 1
    assert row["mode"] == "greedy"
    assert 1.0 <= row["avg_rank"] <= 4.0
    assert row["num_games"] >= 1
    assert sum(row["rank_counts"].values()) == row["num_games"]
    best = read_json(run_root / "best" / "best.json")
    assert best["policy_version"] == 1


# ----------------------------------------------------------------------
# hidden-info guard
# ----------------------------------------------------------------------


def test_evaluator_source_does_not_reference_hidden_state():
    text = Path(ev.__file__).read_text(encoding="utf-8")
    for tok in (
        "env.hands",
        "env.wall",
        "env.state",
        "obs.hands[",
        "full_state",
        "private_hand",
        "mjai_log",
    ):
        assert tok not in text, f"evaluator source references {tok!r}"
