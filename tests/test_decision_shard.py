"""Tests for decision sample schema and shard IO.

確認内容:
- DecisionSample dataclass construction (default / discard / candidate)
- writer / reader roundtrip
- mixed family (discard + candidate) roundtrip
- schema_version mismatch / missing は fail-fast
- C=0 candidate の roundtrip + batch
- variable C candidates の padding / candidate_mask
- terminal / yaku target roundtrip
- teacher info roundtrip
- collate tensor dtype / shape
- hidden info を schema に含まないことの guard
- source に repo 外 local docs / issue ID 参照が混入していないこと
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# DecisionSample construction
# ---------------------------------------------------------------------------


def test_decision_sample_default_schema_version():
    from mahjong_agent.data import SCHEMA_VERSION, DecisionSample

    s = DecisionSample()
    assert s.schema_version == SCHEMA_VERSION
    assert isinstance(s.observation, np.ndarray)
    assert isinstance(s.discard_mask, np.ndarray)
    assert isinstance(s.candidate_features, np.ndarray)
    assert s.discard_mask.shape == (34,)


def test_decision_sample_discard_construction():
    from mahjong_agent.data import DecisionSample

    s = DecisionSample(
        decision_family="normal_discard",
        observation=np.zeros(128, dtype=np.float32),
        discard_mask=np.ones(34, dtype=np.float32),
        selected_discard_tile_type=5,
        selected_candidate_index=-1,
        old_log_prob=-0.123,
        value=0.5,
    )
    assert s.selected_discard_tile_type == 5
    assert s.selected_candidate_index == -1


def test_decision_sample_candidate_construction():
    from mahjong_agent.data import DecisionSample

    s = DecisionSample(
        decision_family="chi",
        observation=np.zeros(128, dtype=np.float32),
        candidate_features=np.zeros((3, 86), dtype=np.float32),
        selected_discard_tile_type=-1,
        selected_candidate_index=1,
    )
    assert s.selected_candidate_index == 1
    assert s.candidate_features.shape == (3, 86)


# ---------------------------------------------------------------------------
# Writer / Reader roundtrip
# ---------------------------------------------------------------------------


def _make_sample(*, discard: bool, idx: int):
    """テスト用 sample factory (discard vs candidate)。"""
    from mahjong_agent.data import DecisionSample
    from mahjong_agent.targets import NUM_YAKU

    obs = np.full(64, fill_value=float(idx), dtype=np.float32)
    discard_mask = np.zeros(34, dtype=np.float32)
    discard_mask[idx % 34] = 1.0
    yaku_target = np.zeros(NUM_YAKU, dtype=np.float32)
    if idx % 2 == 0:
        yaku_target[idx % NUM_YAKU] = 1.0
    if discard:
        return DecisionSample(
            episode_id=f"e{idx}",
            round_id=idx,
            step_id=idx * 10,
            player_id=idx % 4,
            decision_family="normal_discard",
            actor_type="policy",
            observation=obs,
            discard_mask=discard_mask,
            candidate_features=np.zeros((0, 12), dtype=np.float32),
            selected_discard_tile_type=idx % 34,
            selected_candidate_index=-1,
            old_log_prob=-0.5 - idx * 0.01,
            value=float(idx) * 0.1,
            reward=float(idx),
            terminated=(idx % 5 == 0),
            round_over=(idx % 3 == 0),
            terminal_class=idx % 5,
            yaku_target=yaku_target,
            yaku_loss_mask=1.0 if idx % 2 == 0 else 0.0,
            han=idx if idx % 2 == 0 else -1,
            fu=30 if idx % 2 == 0 else -1,
            score_delta=idx * 1000,
            teacher_discard_tile_type=(idx + 1) % 34,
            teacher_candidate_index=-1,
            teacher_available=True,
            metadata={"note": f"sample-{idx}"},
        )
    else:
        C = (idx % 4) + 1  # 1..4
        cand_feat = np.full((C, 12), fill_value=float(idx), dtype=np.float32)
        return DecisionSample(
            episode_id=f"e{idx}",
            round_id=idx,
            step_id=idx * 10 + 1,
            player_id=idx % 4,
            decision_family="chi",
            actor_type="baseline",
            observation=obs,
            discard_mask=np.zeros(34, dtype=np.float32),
            candidate_features=cand_feat,
            selected_discard_tile_type=-1,
            selected_candidate_index=idx % C,
            old_log_prob=-0.7,
            value=0.0,
            reward=0.0,
            terminated=False,
            round_over=False,
            terminal_class=-1,
            yaku_target=yaku_target,
            yaku_loss_mask=0.0,
            han=-1,
            fu=-1,
            score_delta=0,
            teacher_discard_tile_type=-1,
            teacher_candidate_index=0,
            teacher_available=False,
            metadata={"variant": "chi-test"},
        )


def _assert_samples_equal(a, b) -> None:
    assert a.schema_version == b.schema_version
    assert a.episode_id == b.episode_id
    assert a.round_id == b.round_id
    assert a.step_id == b.step_id
    assert a.player_id == b.player_id
    assert a.decision_family == b.decision_family
    assert a.actor_type == b.actor_type
    np.testing.assert_array_equal(a.observation, b.observation)
    np.testing.assert_array_equal(a.discard_mask, b.discard_mask)
    np.testing.assert_array_equal(a.candidate_features, b.candidate_features)
    assert a.selected_discard_tile_type == b.selected_discard_tile_type
    assert a.selected_candidate_index == b.selected_candidate_index
    assert pytest.approx(a.old_log_prob) == b.old_log_prob
    assert pytest.approx(a.value) == b.value
    assert pytest.approx(a.reward) == b.reward
    assert a.terminated == b.terminated
    assert a.round_over == b.round_over
    assert a.terminal_class == b.terminal_class
    np.testing.assert_array_equal(a.yaku_target, b.yaku_target)
    assert pytest.approx(a.yaku_loss_mask) == b.yaku_loss_mask
    assert a.han == b.han
    assert a.fu == b.fu
    assert a.score_delta == b.score_delta
    assert a.teacher_discard_tile_type == b.teacher_discard_tile_type
    assert a.teacher_candidate_index == b.teacher_candidate_index
    assert a.teacher_available == b.teacher_available
    assert a.metadata == b.metadata


def test_writer_reader_roundtrip_discard_only(tmp_path):
    from mahjong_agent.data import read_decision_shard, write_decision_shard

    samples = [_make_sample(discard=True, idx=i) for i in range(5)]
    path = tmp_path / "shard.npz"
    write_decision_shard(path, samples)
    loaded = read_decision_shard(path)
    assert len(loaded) == 5
    for a, b in zip(samples, loaded, strict=True):
        _assert_samples_equal(a, b)


def test_writer_reader_roundtrip_candidate_only(tmp_path):
    from mahjong_agent.data import read_decision_shard, write_decision_shard

    samples = [_make_sample(discard=False, idx=i) for i in range(4)]
    path = tmp_path / "shard.npz"
    write_decision_shard(path, samples)
    loaded = read_decision_shard(path)
    for a, b in zip(samples, loaded, strict=True):
        _assert_samples_equal(a, b)


def test_writer_reader_roundtrip_mixed_family(tmp_path):
    """discard と candidate sample が混在しても roundtrip できる。"""
    from mahjong_agent.data import read_decision_shard, write_decision_shard

    samples: list = []
    for i in range(6):
        samples.append(_make_sample(discard=(i % 2 == 0), idx=i))
    path = tmp_path / "shard.npz"
    write_decision_shard(path, samples)
    loaded = read_decision_shard(path)
    assert len(loaded) == 6
    for a, b in zip(samples, loaded, strict=True):
        _assert_samples_equal(a, b)


def test_writer_empty_list(tmp_path):
    """sample 0 件でも shard は書け、reader が空 list を返す。"""
    from mahjong_agent.data import (
        read_decision_shard,
        read_shard_metadata,
        write_decision_shard,
    )

    path = tmp_path / "empty.npz"
    write_decision_shard(path, [])
    meta = read_shard_metadata(path)
    assert meta["num_samples"] == 0
    samples = read_decision_shard(path)
    assert samples == []


def test_writer_metadata_passthrough(tmp_path):
    from mahjong_agent.data import read_shard_metadata, write_decision_shard

    write_decision_shard(
        tmp_path / "s.npz",
        [_make_sample(discard=True, idx=0)],
        metadata={"experiment": "stage03_smoke", "seed": 42},
    )
    meta = read_shard_metadata(tmp_path / "s.npz")
    assert meta["user_metadata"] == {"experiment": "stage03_smoke", "seed": 42}


# ---------------------------------------------------------------------------
# schema_version mismatch / missing fail-fast
# ---------------------------------------------------------------------------


def test_writer_rejects_wrong_schema_version(tmp_path):
    from mahjong_agent.data import DecisionSample, write_decision_shard

    bad = DecisionSample(schema_version=999)
    with pytest.raises(ValueError, match="schema_version"):
        write_decision_shard(tmp_path / "bad.npz", [bad])


def test_reader_rejects_missing_shard_meta(tmp_path):
    """``_shard_meta`` を含まない npz は ValueError。"""
    from mahjong_agent.data import read_decision_shard

    path = tmp_path / "nometa.npz"
    np.savez_compressed(path, observation=np.zeros((0, 0)))
    with pytest.raises(ValueError, match="shard meta"):
        read_decision_shard(path)


def test_reader_rejects_wrong_schema_version_in_meta(tmp_path):
    """``_shard_meta`` 内の schema_version が reader の値と違うと fail-fast。"""
    import json

    from mahjong_agent.data import read_decision_shard

    path = tmp_path / "wrong_ver.npz"
    meta = {"schema_version": 999, "observation_dim": 0,
            "candidate_dim": 0, "num_samples": 0}
    np.savez_compressed(
        path,
        _shard_meta=np.array(json.dumps(meta), dtype=object),
    )
    with pytest.raises(ValueError, match="schema_version"):
        read_decision_shard(path)


# ---------------------------------------------------------------------------
# C=0 candidate roundtrip
# ---------------------------------------------------------------------------


def test_zero_candidate_roundtrip(tmp_path):
    from mahjong_agent.data import (
        DecisionSample,
        read_decision_shard,
        write_decision_shard,
    )

    s = DecisionSample(
        decision_family="normal_discard",
        observation=np.zeros(16, dtype=np.float32),
        candidate_features=np.zeros((0, 12), dtype=np.float32),
    )
    path = tmp_path / "z.npz"
    write_decision_shard(path, [s])
    loaded = read_decision_shard(path)
    assert loaded[0].candidate_features.shape == (0, 12)


def test_zero_candidate_in_collate_yields_zero_pad():
    """C=0 sample のみでも collate は (N, 0, candidate_dim) を返す。"""
    from mahjong_agent.data import DecisionSample, collate_decision_samples

    sample = DecisionSample(
        observation=np.zeros(16, dtype=np.float32),
        candidate_features=np.zeros((0, 12), dtype=np.float32),
    )
    batch = collate_decision_samples([sample, sample])
    assert batch.candidate_features.shape == (2, 0, 12)
    assert batch.candidate_mask.shape == (2, 0)
    assert batch.candidate_count.tolist() == [0, 0]


# ---------------------------------------------------------------------------
# Variable C padding
# ---------------------------------------------------------------------------


def test_collate_padding_variable_candidates():
    """C=1, 3, 0 が混在 → Cmax=3 で padding + candidate_mask が正しい。"""
    from mahjong_agent.data import DecisionSample, collate_decision_samples

    samples = [
        DecisionSample(
            observation=np.zeros(16, dtype=np.float32),
            candidate_features=np.full((1, 4), 1.0, dtype=np.float32),
        ),
        DecisionSample(
            observation=np.zeros(16, dtype=np.float32),
            candidate_features=np.full((3, 4), 2.0, dtype=np.float32),
        ),
        DecisionSample(
            observation=np.zeros(16, dtype=np.float32),
            candidate_features=np.zeros((0, 4), dtype=np.float32),
        ),
    ]
    batch = collate_decision_samples(samples)
    assert batch.candidate_features.shape == (3, 3, 4)
    assert batch.candidate_mask.shape == (3, 3)
    # sample0: 1 valid, 2 pad
    assert batch.candidate_mask[0].tolist() == [1.0, 0.0, 0.0]
    # sample1: 3 valid
    assert batch.candidate_mask[1].tolist() == [1.0, 1.0, 1.0]
    # sample2: 0 valid (all padded)
    assert batch.candidate_mask[2].tolist() == [0.0, 0.0, 0.0]
    # values: padding 部分は 0
    assert batch.candidate_features[0, 0, 0] == 1.0
    assert batch.candidate_features[0, 1, 0] == 0.0  # padding
    assert batch.candidate_features[1, 2, 0] == 2.0
    assert batch.candidate_count.tolist() == [1, 3, 0]


def test_collate_roundtrip_after_write_read(tmp_path):
    """shard write → read → collate が end-to-end で動く。"""
    from mahjong_agent.data import (
        collate_decision_samples,
        read_decision_shard,
        write_decision_shard,
    )

    samples = [_make_sample(discard=(i % 2 == 0), idx=i) for i in range(5)]
    write_decision_shard(tmp_path / "s.npz", samples)
    loaded = read_decision_shard(tmp_path / "s.npz")
    batch = collate_decision_samples(loaded)
    assert batch.batch_size == 5


# ---------------------------------------------------------------------------
# Terminal / yaku target roundtrip
# ---------------------------------------------------------------------------


def test_terminal_and_yaku_target_roundtrip(tmp_path):
    from mahjong_agent.data import (
        DecisionSample,
        read_decision_shard,
        write_decision_shard,
    )
    from mahjong_agent.targets import NUM_YAKU

    yaku_target = np.zeros(NUM_YAKU, dtype=np.float32)
    yaku_target[0] = 1.0  # Menzen Tsumo
    yaku_target[11] = 1.0  # Tanyao
    s = DecisionSample(
        observation=np.zeros(8, dtype=np.float32),
        terminal_class=2,  # draw_tenpai
        yaku_target=yaku_target,
        yaku_loss_mask=1.0,
        han=3,
        fu=30,
        score_delta=5800,
    )
    path = tmp_path / "yaku.npz"
    write_decision_shard(path, [s])
    loaded = read_decision_shard(path)[0]
    assert loaded.terminal_class == 2
    np.testing.assert_array_equal(loaded.yaku_target, yaku_target)
    assert loaded.yaku_loss_mask == 1.0
    assert loaded.han == 3
    assert loaded.fu == 30
    assert loaded.score_delta == 5800


# ---------------------------------------------------------------------------
# Teacher info roundtrip
# ---------------------------------------------------------------------------


def test_teacher_info_roundtrip(tmp_path):
    from mahjong_agent.data import (
        DecisionSample,
        read_decision_shard,
        write_decision_shard,
    )

    s = DecisionSample(
        observation=np.zeros(4, dtype=np.float32),
        teacher_discard_tile_type=17,
        teacher_candidate_index=2,
        teacher_available=True,
    )
    write_decision_shard(tmp_path / "t.npz", [s])
    loaded = read_decision_shard(tmp_path / "t.npz")[0]
    assert loaded.teacher_discard_tile_type == 17
    assert loaded.teacher_candidate_index == 2
    assert loaded.teacher_available is True


# ---------------------------------------------------------------------------
# Tensor dtype / shape
# ---------------------------------------------------------------------------


def test_collate_tensor_dtypes():
    from mahjong_agent.data import DecisionSample, collate_decision_samples
    from mahjong_agent.targets import NUM_YAKU

    samples = [
        DecisionSample(
            observation=np.zeros(8, dtype=np.float32),
            candidate_features=np.zeros((2, 4), dtype=np.float32),
            yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
        )
        for _ in range(3)
    ]
    b = collate_decision_samples(samples)
    assert b.observation.dtype == torch.float32
    assert b.observation.shape == (3, 8)
    assert b.discard_mask.dtype == torch.float32
    assert b.discard_mask.shape == (3, 34)
    assert b.candidate_features.dtype == torch.float32
    assert b.candidate_features.shape == (3, 2, 4)
    assert b.candidate_mask.dtype == torch.float32
    assert b.candidate_mask.shape == (3, 2)
    assert b.candidate_count.dtype == torch.int64
    assert b.candidate_count.shape == (3,)
    assert b.selected_discard_tile_type.dtype == torch.int64
    assert b.selected_candidate_index.dtype == torch.int64
    assert b.old_log_prob.dtype == torch.float32
    assert b.terminated.dtype == torch.float32
    assert b.round_over.dtype == torch.float32
    assert b.terminal_class.dtype == torch.int64
    assert b.yaku_target.dtype == torch.float32
    assert b.yaku_target.shape == (3, NUM_YAKU)
    assert b.yaku_loss_mask.dtype == torch.float32
    assert b.teacher_available.dtype == torch.float32


def test_collate_empty_list_raises():
    from mahjong_agent.data import collate_decision_samples

    with pytest.raises(ValueError):
        collate_decision_samples([])


def test_collate_rejects_inconsistent_obs_dim():
    """sample 間で observation dim が異なると ValueError。"""
    from mahjong_agent.data import DecisionSample, collate_decision_samples

    samples = [
        DecisionSample(observation=np.zeros(8, dtype=np.float32)),
        DecisionSample(observation=np.zeros(9, dtype=np.float32)),
    ]
    with pytest.raises(ValueError, match="observation"):
        collate_decision_samples(samples)


# ---------------------------------------------------------------------------
# Hidden info guard: schema does not contain raw env / state / wall / hands
# ---------------------------------------------------------------------------


def test_decision_sample_does_not_carry_hidden_state_fields():
    """``DecisionSample`` の field 名に env / wall / state / hands が含まれない。"""
    from mahjong_agent.data import DecisionSample

    field_names = {f.name for f in dataclasses.fields(DecisionSample)}
    forbidden = {"hands", "wall", "state", "env", "full_state",
                 "other_hands", "dead_wall"}
    leaked = field_names & forbidden
    assert leaked == set(), (
        f"DecisionSample carries hidden-info-like fields: {leaked}"
    )
    # observation は public-only encoder の出力なので OK (1 field のみ)
    assert "observation" in field_names


def test_shard_file_does_not_contain_hidden_keys(tmp_path):
    from mahjong_agent.data import write_decision_shard

    samples = [_make_sample(discard=True, idx=0)]
    path = tmp_path / "leak_test.npz"
    write_decision_shard(path, samples)
    with np.load(path, allow_pickle=True) as npz:
        keys = set(npz.files)
    forbidden = {"hands", "wall", "state", "env", "full_state"}
    assert keys & forbidden == set(), (
        f"shard file leaks hidden keys: {keys & forbidden}"
    )


# ---------------------------------------------------------------------------
# Repo-external local docs / issue ID guard
# ---------------------------------------------------------------------------


def test_data_source_does_not_reference_local_docs_or_issues():
    src_dir = Path(__file__).resolve().parent.parent / "src" / "mahjong_agent"
    forbidden_patterns = [
        r"PROJECT_RULE",
        r"ISSUE_BOARD",
        r"\bISSUE-\d",
        r"\bISSUE-L\d",
        r"\bAGENTS\.md",
        r"\bCLAUDE\.md",
        r"\bmajong-rl",
        r"CHANGE_QUEUE",
    ]
    pattern = re.compile("|".join(forbidden_patterns))
    offenders = []
    for path in src_dir.rglob("*.py"):
        text = path.read_text()
        for m in pattern.finditer(text):
            offenders.append((str(path.relative_to(src_dir.parent.parent)), m.group(0)))
    assert offenders == [], (
        f"package source references workspace-local docs / issue IDs: "
        f"{offenders}"
    )
