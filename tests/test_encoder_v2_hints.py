"""Batch 3: Encoder v2 hints tests."""
from __future__ import annotations

import numpy as np
import riichienv
import torch

from mahjong_agent.encoders.public_observation import PublicObservationEncoder
from mahjong_agent.models import Stage03Model, Stage03ModelConfig

# ----------------------------------------------------------------------
# Dim / metadata / feature_ranges
# ----------------------------------------------------------------------


def test_encoder_default_enables_hints():
    enc = PublicObservationEncoder()
    assert enc.enable_hints is True
    meta = enc.metadata()
    assert meta.observation_dim > 363
    # 363 + hint dims (1 + 34 + 34 + 1 + 1 + 6 + shape_hint 66) = 506
    # (旧仕様にあった riichi_discard_mask=34 は削除済み、shape_hint 66 を追加)
    assert meta.observation_dim == 506


def test_encoder_with_hints_disabled_returns_legacy_dim():
    enc = PublicObservationEncoder(enable_hints=False)
    assert enc.enable_hints is False
    meta = enc.metadata()
    assert meta.observation_dim == 363


def test_encoder_feature_ranges_include_new_hints():
    enc = PublicObservationEncoder(enable_hints=True)
    ranges = enc.metadata().feature_ranges
    expected_hint_names = {
        "current_shanten_norm",
        "shanten_delta_per_discard",
        "discard_ukeire_per_tile",
        "remaining_draws_norm",
        "turn_progress_norm",
        "tile_presence_flags",
        "shape_hint",
    }
    assert expected_hint_names <= set(ranges.keys())
    # shape_hint は 66 dim
    s, e = ranges["shape_hint"]
    assert e - s == 66
    # contiguous な layout (= 全 range の合計 = observation_dim)
    total = sum(e - s for s, e in ranges.values())
    assert total == enc.metadata().observation_dim


def test_encoder_feature_ranges_no_longer_include_riichi_discard_mask():
    """``riichi_discard_mask`` は Stage03 で常時 all-zero になるため削除した
    (dead feature)。RIICHI_DISCARD candidate 自体は ``ModelAction`` /
    ``LegalActionSet`` 経路で env clone 経由で正確に生成される。"""
    enc = PublicObservationEncoder(enable_hints=True)
    ranges = enc.metadata().feature_ranges
    assert "riichi_discard_mask" not in ranges


def test_encoder_legacy_off_omits_hint_ranges():
    enc = PublicObservationEncoder(enable_hints=False)
    ranges = enc.metadata().feature_ranges
    hint_names = {
        "current_shanten_norm",
        "shanten_delta_per_discard",
        "discard_ukeire_per_tile",
        "remaining_draws_norm",
        "turn_progress_norm",
        "tile_presence_flags",
        "shape_hint",
    }
    assert not (hint_names & set(ranges.keys()))


# ----------------------------------------------------------------------
# encode_observation against real env
# ----------------------------------------------------------------------


def _real_obs(seed: int = 42):
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=seed)
    obs = env.get_observation(env.current_player)
    return env, obs


def test_encode_observation_shape_matches_metadata():
    enc = PublicObservationEncoder()
    _, obs = _real_obs(seed=1)
    feat = enc.encode_observation(obs)
    assert feat.shape == (enc.metadata().observation_dim,)
    assert feat.dtype == np.float32
    # finite values
    assert np.isfinite(feat).all()


def test_encoder_hints_finite_within_expected_ranges():
    enc = PublicObservationEncoder()
    _, obs = _real_obs(seed=3)
    feat = enc.encode_observation(obs)
    ranges = enc.metadata().feature_ranges

    def _slice(name):
        s, e = ranges[name]
        return feat[s:e]

    # current_shanten_norm in [0, 1]
    cs = _slice("current_shanten_norm")
    assert 0.0 <= float(cs[0]) <= 1.0
    # shanten_delta_per_discard in [-1, 1]
    sd = _slice("shanten_delta_per_discard")
    assert sd.min() >= -1.0 - 1e-6 and sd.max() <= 1.0 + 1e-6
    # discard_ukeire_per_tile in [0, 1]
    uk = _slice("discard_ukeire_per_tile")
    assert uk.min() >= 0.0 and uk.max() <= 1.0 + 1e-6
    # (riichi_discard_mask は dead feature だったので削除済み — slice しない)
    # remaining_draws_norm / turn_progress_norm in [0, 1]
    rd = _slice("remaining_draws_norm")
    tp = _slice("turn_progress_norm")
    assert 0.0 <= float(rd[0]) <= 1.0
    assert 0.0 <= float(tp[0]) <= 1.0
    # tile_presence_flags in {0, 1}, length 6
    tpf = _slice("tile_presence_flags")
    assert tpf.shape == (6,)
    assert set(tpf.tolist()) <= {0.0, 1.0}


# ----------------------------------------------------------------------
# shanten / ukeire hint sanity on known hand
# ----------------------------------------------------------------------


class _FakeObs:
    """encode_observation を呼べる程度の stub observation。"""

    def __init__(self, hand, player_id=0):
        self.hand = hand
        self.player_id = player_id
        self.melds = [[], [], [], []]
        self.discards = [[], [], [], []]
        self.dora_indicators = []
        self.scores = [25000, 25000, 25000, 25000]
        self.oya = 0
        self.round_wind = 0
        self.honba = 0
        self.riichi_sticks = 0
        self.kyoku_index = 0
        self.riichi_declared = [False, False, False, False]
        self.is_tenpai = False

    def legal_actions(self):
        return []


def test_shanten_hint_on_complete_hand():
    # 11122233344499m = 4 mentsu + 1 pair = 和了 → shanten=-1
    # tile_ids: tile_type 0 → tile_id 0..3, tile_type 1 → 4..7, ...
    hand = (
        [0, 1, 2]     # 1m × 3 (tile_type 0)
        + [4, 5, 6]   # 2m × 3
        + [8, 9, 10]  # 3m × 3
        + [12, 13, 14]  # 4m × 3
        + [32, 33]    # 9m × 2 (tile_type 8)
    )
    enc = PublicObservationEncoder()
    feat = enc.encode_observation(_FakeObs(hand))
    s, e = enc.metadata().feature_ranges["current_shanten_norm"]
    # shanten = -1 → -1/8 = -0.125
    cs = feat[s]
    assert cs == np.float32(-1.0 / 8.0)


def test_shanten_hint_negative_for_useful_tile():
    """tenpai 形を崩す useful tile を切ると shanten が悪化する (delta < 0)。

    14 牌: 1m×3 + 4m×3 + 7m×3 + 2p×2 + 5p + 8m + 9m
    = 1m刻 + 4m刻 + 7m対 + 789m順 + 2p雀頭 + 5p
    → tenpai (shanten=0, 5p 単騎 or 7m シャンポン待ち)。

    8m (= tile_type 7) を切ると 789m順子が崩れ、7m刻に戻って 2p雀頭 + 5p
    + 9m 単独で 1 shanten 悪化 → delta[7] < 0。
    """
    hand = (
        [0, 1, 2]            # 1m × 3 (tile_type 0)
        + [12, 13, 14]       # 4m × 3 (tile_type 3)
        + [24, 25, 26]       # 7m × 3 (tile_type 6)
        + [40, 41]           # 2p × 2 (tile_type 10)
        + [52]               # 5p × 1 (tile_type 13)
        + [28]               # 8m × 1 (tile_type 7)
        + [32]               # 9m × 1 (tile_type 8)
    )
    assert len(hand) == 14
    enc = PublicObservationEncoder()
    feat = enc.encode_observation(_FakeObs(hand))
    s, e = enc.metadata().feature_ranges["shanten_delta_per_discard"]
    deltas = feat[s:e]
    # 8m を切ると tenpai → 1 shanten。delta = 0 - 1 = -1 → -0.25 (norm)
    assert deltas[7] < 0


def test_shanten_hint_zero_for_tiles_not_in_hand():
    """手牌に無い tile_type の delta は 0 (= 切れないので評価対象外)。"""
    hand = [0, 1, 2] + [4, 5, 6] + [8, 9, 10] + [12, 13, 14] + [32, 33]
    enc = PublicObservationEncoder()
    feat = enc.encode_observation(_FakeObs(hand))
    s, e = enc.metadata().feature_ranges["shanten_delta_per_discard"]
    deltas = feat[s:e]
    # tile_type 20 (= 6s) は手に居ない → delta = 0
    assert deltas[20] == 0


# ----------------------------------------------------------------------
# Stage03Model.from_encoder_metadata follows new dim
# ----------------------------------------------------------------------


def test_stage03_model_config_follows_encoder_dim():
    enc = PublicObservationEncoder(enable_hints=True)
    cfg = Stage03ModelConfig.from_encoder_metadata(enc.metadata())
    assert cfg.observation_dim == enc.metadata().observation_dim
    assert cfg.candidate_dim == enc.metadata().candidate_dim


def test_model_forward_with_v2_encoder_runs():
    enc = PublicObservationEncoder()
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16,
    )
    model = Stage03Model(cfg).eval()
    _, obs = _real_obs(seed=5)
    feat = enc.encode_observation(obs)
    feat_t = torch.from_numpy(feat).unsqueeze(0)
    with torch.no_grad():
        out = model(feat_t)
    assert out.discard_logits.shape == (1, 34)
    assert out.value.shape == (1,)


def test_legacy_encoder_off_still_works_with_model():
    enc = PublicObservationEncoder(enable_hints=False)
    assert enc.metadata().observation_dim == 363
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16,
    )
    model = Stage03Model(cfg).eval()
    _, obs = _real_obs(seed=7)
    feat = enc.encode_observation(obs)
    feat_t = torch.from_numpy(feat).unsqueeze(0)
    with torch.no_grad():
        out = model(feat_t)
    assert out.discard_logits.shape == (1, 34)


# ----------------------------------------------------------------------
# Hidden-info guard for hints
# ----------------------------------------------------------------------


def test_encoder_hints_dont_touch_other_hands():
    """obs.hands (他家手牌) を読まずに hint が計算できることを確認する。

    riichienv の Observation 仕様では obs.hands[other] は空 list だが、
    hint feature 経路がそこを誤って読まない (= AttributeError にならない)。
    """
    enc = PublicObservationEncoder()
    _, obs = _real_obs(seed=9)
    # obs.hands は他家側 slot が空。encode が pass する。
    feat = enc.encode_observation(obs)
    assert feat.shape == (enc.metadata().observation_dim,)
    # obs.hands 他家を覗くと空 list なので、それを誤って使えば 0 dim feature
    # になる。ここでは hint が non-zero になり得る (自手由来) ことを assert。
    s, e = enc.metadata().feature_ranges["self_hand_counts"]
    hand_counts = feat[s:e]
    assert float(hand_counts.sum()) > 0  # 自手 13/14 牌は必ず居る


def test_encoder_source_does_not_reference_hands_other():
    """encoder source に ``obs.hands[`` への参照が無い (自手 .hand のみ)。"""
    from pathlib import Path

    import mahjong_agent.encoders as pkg

    src = Path(pkg.__file__).parent.rglob("*.py")
    forbidden = ("obs.hands[", "obs.hands [")
    for f in src:
        text = f.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, (
                f"encoder source {f.name} references {tok!r} "
                f"(hidden hands of other players)"
            )


# ----------------------------------------------------------------------
# Integration: SelfPlayRunner uses new encoder dim transparently
# ----------------------------------------------------------------------


def test_self_play_runner_records_new_obs_dim():
    """SelfPlayRunner が default で v2 encoder を使うので、sample の
    observation shape が新 dim になる。"""
    from mahjong_agent.agents import RandomAgent
    from mahjong_agent.evaluation import (
        SeatAgents,
        SelfPlayConfig,
        SelfPlayRunner,
    )

    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    sa = SeatAgents.homogeneous(
        RandomAgent(seed=0), actor_type="random", num_players=4
    )
    res = runner.run_episode(sa, seed=0)
    assert res.crash_context is None
    assert res.samples
    enc = PublicObservationEncoder()
    expected_dim = enc.metadata().observation_dim
    for s in res.samples[:5]:
        assert s.observation.shape == (expected_dim,)


# ----------------------------------------------------------------------
# Grep guard: no hard-coded 363 in source
# ----------------------------------------------------------------------


def test_no_legacy_363_hardcode_in_source():
    """src/ 内に「観測 dim = 363」を前提とした hard-code が残っていないこと。"""
    from pathlib import Path

    import mahjong_agent

    pkg_root = Path(mahjong_agent.__file__).parent
    bad_hits: list[tuple[str, int, str]] = []
    for path in pkg_root.rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "363" in line and "observation_dim" not in line:
                # コメントで「v1 base 363」のような言及は許容するが、
                # コード literal としては許容しない
                if line.lstrip().startswith("#"):
                    continue
                bad_hits.append((str(path), i, line))
    assert not bad_hits, f"363 hard-codes found: {bad_hits}"
