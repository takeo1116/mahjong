"""Tests for the public observation encoder v1.

確認内容:
- metadata の dim / feature_ranges が encoder の出力 shape と一致する
- observation feature の固定長
- discard legal mask の shape / values
- candidate feature の shape (per candidate / batch)
- 可変 candidate 数
- 実 RiichiEnv reset 後 observation での smoke
- hidden information を参照していないことの guard
- package source / README が repo 外 local docs / local issue を参照しない
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# metadata / shape
# ---------------------------------------------------------------------------


def test_metadata_consistent_with_feature_ranges():
    """metadata.observation_dim が feature_ranges の合計と一致する。"""
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    meta = enc.metadata()
    total = sum(e - s for s, e in meta.feature_ranges.values())
    assert total == meta.observation_dim
    cand_total = sum(e - s for s, e in meta.candidate_feature_ranges.values())
    assert cand_total == meta.candidate_dim
    assert meta.discard_mask_dim == 34


def test_feature_ranges_are_contiguous_and_sorted():
    """feature_ranges が 0 から始まり、隙間なく順に並んでいる。"""
    from mahjong_agent.encoders import PublicObservationEncoder

    meta = PublicObservationEncoder().metadata()
    # observation
    cursor = 0
    for name, (s, e) in meta.feature_ranges.items():
        assert s == cursor, (
            f"feature {name!r} starts at {s}, expected {cursor}")
        assert e > s
        cursor = e
    assert cursor == meta.observation_dim
    # candidate
    cursor = 0
    for name, (s, e) in meta.candidate_feature_ranges.items():
        assert s == cursor, (
            f"candidate feature {name!r} starts at {s}, expected {cursor}")
        assert e > s
        cursor = e
    assert cursor == meta.candidate_dim


# ---------------------------------------------------------------------------
# observation feature
# ---------------------------------------------------------------------------


def _real_observation(seed: int = 42):
    """RiichiEnv reset 直後の observation を返す helper。"""
    import riichienv

    env = riichienv.RiichiEnv(riichienv.GameType.YON_TONPUSEN)
    env.reset(seed=seed)
    return env, env.get_observation(env.current_player)


def test_encode_observation_fixed_size_and_dtype():
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    meta = enc.metadata()
    _, obs = _real_observation()
    feat = enc.encode_observation(obs)
    assert isinstance(feat, np.ndarray)
    assert feat.shape == (meta.observation_dim,)
    assert feat.dtype == np.float32
    # 含まれる値はすべて有限
    assert np.all(np.isfinite(feat))


def test_self_hand_counts_match_observation_hand():
    """self_hand_counts のスライスが obs.hand の tile_type 集計と一致する。"""
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    meta = enc.metadata()
    _, obs = _real_observation()
    feat = enc.encode_observation(obs)
    s, e = meta.feature_ranges["self_hand_counts"]
    hand_counts = feat[s:e]
    # 手動集計と一致
    expected = np.zeros(34, dtype=np.float32)
    for tid in obs.hand:
        expected[int(tid) // 4] += 1.0
    assert np.array_equal(hand_counts, expected)
    # 起家初手なので 14 牌
    assert int(hand_counts.sum()) == len(obs.hand)


def test_self_riichi_flag_zero_at_reset():
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    _, obs = _real_observation()
    feat = enc.encode_observation(obs)
    s, e = enc.metadata().feature_ranges["self_riichi_flag"]
    assert feat[s:e].tolist() == [0.0]


# ---------------------------------------------------------------------------
# discard legal mask
# ---------------------------------------------------------------------------


def test_discard_legal_mask_shape_and_values():
    """legal mask shape (34,), values 0/1, sum == number of legal tile_types。"""
    from mahjong_agent.actions import legal_actions_to_model_set
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    env, obs = _real_observation()
    legal_set = legal_actions_to_model_set(
        list(obs.legal_actions()),
        actor=obs.player_id,
        num_players=env.num_players,
        env_for_riichi=env,
    )
    mask = enc.discard_legal_mask(legal_set)
    assert mask.shape == (34,)
    assert mask.dtype == np.float32
    assert set(mask.tolist()) <= {0.0, 1.0}
    assert int(mask.sum()) == len(legal_set.normal_discard)


# ---------------------------------------------------------------------------
# candidate feature
# ---------------------------------------------------------------------------


def test_encode_candidate_shape():
    from mahjong_agent.actions.types import ActionFamily, ActionKey, ModelAction
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    meta = enc.metadata()
    cand = ModelAction(
        key=ActionKey(
            family=ActionFamily.CHI,
            tile_type=16,
            consume_tile_types=(14, 15),
            target_rel_seat=3,
        ),
        actor=0,
        _raw_actions=(),
    )
    feat = enc.encode_candidate(cand)
    assert feat.shape == (meta.candidate_dim,)
    assert feat.dtype == np.float32
    # family one-hot
    fs, fe = meta.candidate_feature_ranges["family_one_hot"]
    assert int(feat[fs:fe].sum()) == 1
    # tile_type present
    ps, pe = meta.candidate_feature_ranges["tile_type_present_flag"]
    assert feat[ps:pe].tolist() == [1.0]
    # target_rel_seat one-hot
    rs, re_ = meta.candidate_feature_ranges["target_rel_seat_one_hot"]
    assert int(feat[rs:re_].sum()) == 1


def test_encode_candidate_with_no_tile_type():
    """Pass / Tsumo / Ron / KyushuKyuhai は tile_type=None → present_flag=0。"""
    from mahjong_agent.actions.types import ActionFamily, ActionKey, ModelAction
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    cand = ModelAction(
        key=ActionKey(family=ActionFamily.PASS),
        actor=0,
        _raw_actions=(),
    )
    feat = enc.encode_candidate(cand)
    meta = enc.metadata()
    ps, pe = meta.candidate_feature_ranges["tile_type_present_flag"]
    assert feat[ps:pe].tolist() == [0.0]
    # target_rel_seat one-hot の "none" slot が立つ
    rs, re_ = meta.candidate_feature_ranges["target_rel_seat_one_hot"]
    one_hot = feat[rs:re_]
    assert one_hot[-1] == 1.0  # last index = none


def test_encode_candidates_variable_count():
    from mahjong_agent.actions import legal_actions_to_model_set
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    env, obs = _real_observation()
    legal_set = legal_actions_to_model_set(
        list(obs.legal_actions()),
        actor=obs.player_id,
        num_players=env.num_players,
        env_for_riichi=env,
    )
    feats = enc.encode_candidates(legal_set)
    assert feats.ndim == 2
    assert feats.shape == (len(legal_set.candidates), enc.metadata().candidate_dim)
    assert feats.dtype == np.float32


def test_encode_candidates_zero_count_returns_empty_array():
    from mahjong_agent.actions.types import LegalActionSet
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    empty_set = LegalActionSet(decision_player=0, normal_discard={}, candidates=())
    feats = enc.encode_candidates(empty_set)
    assert feats.shape == (0, enc.metadata().candidate_dim)


# ---------------------------------------------------------------------------
# real env smoke
# ---------------------------------------------------------------------------


def test_real_env_smoke_encode_observation_and_legal_mask_and_candidates():
    """RiichiEnv reset → encode → 全 output が consistent shape を持つ。"""
    from mahjong_agent.actions import legal_actions_to_model_set
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    meta = enc.metadata()
    env, obs = _real_observation()
    legal_set = legal_actions_to_model_set(
        list(obs.legal_actions()),
        actor=obs.player_id,
        num_players=env.num_players,
        env_for_riichi=env,
    )
    feat = enc.encode_observation(obs)
    mask = enc.discard_legal_mask(legal_set)
    cand_feats = enc.encode_candidates(legal_set)
    assert feat.shape == (meta.observation_dim,)
    assert mask.shape == (34,)
    assert cand_feats.shape == (len(legal_set.candidates), meta.candidate_dim)


# ---------------------------------------------------------------------------
# hidden leak guard
# ---------------------------------------------------------------------------


def test_encoder_does_not_use_hidden_observation_fields(monkeypatch):
    """encoder が ``obs.hands`` (他家 slot) を読まないことを property 経由で確認する。

    Observation の hands を読もうとすると test fixture 側で trap が走り、test が
    fail する。
    """
    from mahjong_agent.encoders import PublicObservationEncoder

    # Observation オブジェクトの代替に minimal stub を使う。
    # encoder が hidden field を読みに行ったら AttributeError で検知。
    class _Trap:
        def __init__(self, real_obs):
            self._real = real_obs
            self.touched_forbidden = []

        def __getattr__(self, name):
            if name in ("hands", "wall", "state"):
                self.touched_forbidden.append(name)
                raise AssertionError(
                    f"encoder accessed forbidden field {name!r}"
                )
            return getattr(self._real, name)

    _, real_obs = _real_observation()
    trap = _Trap(real_obs)
    enc = PublicObservationEncoder()
    feat = enc.encode_observation(trap)  # should not touch hands/wall/state
    assert feat.shape == (enc.metadata().observation_dim,)
    assert trap.touched_forbidden == []


# ---------------------------------------------------------------------------
# repo-external local docs / issue ID references must not leak into source.
# ---------------------------------------------------------------------------


def test_encoder_source_does_not_reference_local_docs_or_issues():
    """encoders / actions / envs の source に repo 外 local docs / issue ID への
    参照が含まれないことを確認する。"""
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


# pytest がこの module を import するときに riichienv が無いと skip しない
# ようにするため、最低限の import smoke を最初に置く。
def test_module_imports():
    from mahjong_agent.encoders import EncoderMetadata, PublicObservationEncoder
    assert EncoderMetadata is not None
    assert PublicObservationEncoder is not None


# linter 用に未使用 import を避ける
_ = pytest


# ----------------------------------------------------------------------
# candidate safety scalars (ISSUE-0023)
# ----------------------------------------------------------------------

_CAND_SAFETY_NAMES = (
    "candidate_shanten_delta",
    "candidate_ukeire_norm",
    "candidate_safe_vs_all_riichi",
    "candidate_suji_vs_all_riichi",
    "candidate_kabe_suji",
)
_CAND_SAFETY_SRC = {
    "candidate_shanten_delta": "shanten_delta_per_discard",
    "candidate_ukeire_norm": "discard_ukeire_per_tile",
    "candidate_safe_vs_all_riichi": "safe_vs_all_riichi_mask",
    "candidate_suji_vs_all_riichi": "suji_vs_all_riichi_mask",
    "candidate_kabe_suji": "kabe_suji_mask",
}


def _cand(family, tile_type=None):
    from mahjong_agent.actions.types import ActionFamily, ActionKey, ModelAction

    return ModelAction(
        key=ActionKey(family=getattr(ActionFamily, family), tile_type=tile_type),
        actor=0,
        _raw_actions=(),
    )


def test_candidate_safety_scalars_in_ranges_and_dim():
    from mahjong_agent.encoders import PublicObservationEncoder

    meta = PublicObservationEncoder().metadata()
    cr = meta.candidate_feature_ranges
    for name in _CAND_SAFETY_NAMES:
        assert name in cr
        s, e = cr[name]
        assert e - s == 1
    # 末尾 append: 5 scalar が target_rel_seat_one_hot の直後から連続
    after = cr["target_rel_seat_one_hot"][1]
    cur = after
    for name in _CAND_SAFETY_NAMES:
        assert cr[name] == (cur, cur + 1)
        cur += 1
    assert cur == meta.candidate_dim
    # candidate_dim == old(86) + 5
    assert meta.candidate_dim == 91


def test_candidate_safety_lookup_matches_observation_feat():
    from mahjong_agent.actions.types import LegalActionSet
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    meta = enc.metadata()
    # known observation_feat: hint range の index 5 に distinct value
    feat = np.zeros(meta.observation_dim, dtype=np.float32)
    expected = {
        "shanten_delta_per_discard": 0.7,
        "discard_ukeire_per_tile": 0.3,
        "safe_vs_all_riichi_mask": 1.0,
        "suji_vs_all_riichi_mask": 1.0,
        "kabe_suji_mask": 1.0,
    }
    for hint, val in expected.items():
        s, _e = meta.feature_ranges[hint]
        feat[s + 5] = val
    cand = _cand("RIICHI_DISCARD", tile_type=5)
    lset = LegalActionSet(decision_player=0, normal_discard={}, candidates=(cand,))
    out = enc.encode_candidates(lset, observation_feat=feat)
    cr = meta.candidate_feature_ranges
    for cand_name, src in _CAND_SAFETY_SRC.items():
        s, _e = cr[cand_name]
        assert out[0, s] == pytest.approx(expected[src], abs=1e-6)


def test_candidate_safety_zero_when_no_tile_type():
    from mahjong_agent.actions.types import LegalActionSet
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    meta = enc.metadata()
    feat = np.ones(meta.observation_dim, dtype=np.float32)  # 全 1
    cand = _cand("PASS", tile_type=None)
    lset = LegalActionSet(decision_player=0, normal_discard={}, candidates=(cand,))
    out = enc.encode_candidates(lset, observation_feat=feat)
    cr = meta.candidate_feature_ranges
    for name in _CAND_SAFETY_NAMES:
        s, _e = cr[name]
        assert out[0, s] == 0.0


def test_candidate_safety_zero_when_no_observation():
    """observation_feat / observation どちらも無い従来呼び出しは all-zero fallback。"""
    from mahjong_agent.actions.types import LegalActionSet
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    meta = enc.metadata()
    cand = _cand("RIICHI_DISCARD", tile_type=5)
    lset = LegalActionSet(decision_player=0, normal_discard={}, candidates=(cand,))
    out = enc.encode_candidates(lset)  # 従来 API
    cr = meta.candidate_feature_ranges
    for name in _CAND_SAFETY_NAMES:
        s, _e = cr[name]
        assert out[0, s] == 0.0


def test_candidate_safety_observation_feat_shape_mismatch_raises():
    from mahjong_agent.actions.types import LegalActionSet
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    cand = _cand("RIICHI_DISCARD", tile_type=5)
    lset = LegalActionSet(decision_player=0, normal_discard={}, candidates=(cand,))
    with pytest.raises(ValueError):
        enc.encode_candidates(
            lset, observation_feat=np.zeros(10, dtype=np.float32)
        )


def test_candidate_safety_hints_off_zero_fallback():
    """enable_hints=False では hint range が無く全 scalar 0.0、dim は同じ。"""
    from mahjong_agent.actions.types import LegalActionSet
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder(enable_hints=False)
    meta = enc.metadata()
    assert meta.candidate_dim == 91  # candidate_dim は hints 有無で不変
    feat = np.ones(meta.observation_dim, dtype=np.float32)
    cand = _cand("RIICHI_DISCARD", tile_type=5)
    lset = LegalActionSet(decision_player=0, normal_discard={}, candidates=(cand,))
    out = enc.encode_candidates(lset, observation_feat=feat)
    cr = meta.candidate_feature_ranges
    for name in _CAND_SAFETY_NAMES:
        s, _e = cr[name]
        assert out[0, s] == 0.0


def test_candidate_safety_observation_arg_path():
    """observation_feat の代わりに observation を渡しても lookup される。"""
    import riichienv

    from mahjong_agent.actions import legal_actions_to_model_set
    from mahjong_agent.encoders import PublicObservationEncoder

    enc = PublicObservationEncoder()
    env = riichienv.RiichiEnv(riichienv.GameType.YON_TONPUSEN)
    env.reset(seed=3)
    cp = env.current_player
    obs = env.get_observation(cp)
    legal_set = legal_actions_to_model_set(
        list(obs.legal_actions()), actor=cp, num_players=env.num_players,
        env_for_riichi=env,
    )
    # observation 引数経路と observation_feat 経路が一致すること
    feat = enc.encode_observation(obs)
    out_a = enc.encode_candidates(legal_set, observation=obs)
    out_b = enc.encode_candidates(legal_set, observation_feat=feat)
    assert np.array_equal(out_a, out_b)
    assert out_a.shape == (len(legal_set.candidates), enc.metadata().candidate_dim)
