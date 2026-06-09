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
    # 506 (= 363 base + 旧 hints + shape_hint 66) + defensive direct hints
    # (safe 34 + suji 34 + kabe 34 = 102) = 608
    assert meta.observation_dim == 608


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
        "safe_vs_all_riichi_mask",
        "suji_vs_all_riichi_mask",
        "kabe_suji_mask",
    }
    assert expected_hint_names <= set(ranges.keys())
    # shape_hint は 66 dim
    s, e = ranges["shape_hint"]
    assert e - s == 66
    # defensive direct hints は各 34 dim
    for name in (
        "safe_vs_all_riichi_mask",
        "suji_vs_all_riichi_mask",
        "kabe_suji_mask",
    ):
        s, e = ranges[name]
        assert e - s == 34
    # contiguous な layout (= 全 range の合計 = observation_dim)
    total = sum(e - s for s, e in ranges.values())
    assert total == enc.metadata().observation_dim


def test_encoder_defensive_hints_appended_after_existing_hints():
    """3 defensive feature は hint 末尾 append で、既存 range が動いていない。"""
    enc = PublicObservationEncoder(enable_hints=True)
    ranges = enc.metadata().feature_ranges
    # 既存 shape_hint の直後から defensive が始まる (末尾 append)
    assert ranges["safe_vs_all_riichi_mask"][0] == ranges["shape_hint"][1]
    assert ranges["suji_vs_all_riichi_mask"][0] == ranges["safe_vs_all_riichi_mask"][1]
    assert ranges["kabe_suji_mask"][0] == ranges["suji_vs_all_riichi_mask"][1]
    assert ranges["kabe_suji_mask"][1] == enc.metadata().observation_dim
    # 既存 base feature の range は不変 (先頭 self_hand_counts は 0..34)
    assert ranges["self_hand_counts"] == (0, 34)


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
        "safe_vs_all_riichi_mask",
        "suji_vs_all_riichi_mask",
        "kabe_suji_mask",
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


# ----------------------------------------------------------------------
# defensive direct hints (ISSUE-0022): semantics
# ----------------------------------------------------------------------

from mahjong_agent.encoders.public_observation import (  # noqa: E402
    _compute_kabe_suji_mask,
    _compute_riichi_safe_masks,
)


def _ids(tile_types):
    """tile_type list -> tile_id list (各 type の代表 id = type*4)。"""
    return [int(tt) * 4 for tt in tile_types]


class _DefenseObs:
    """守備 feature 用の薄い observation stub。"""

    def __init__(self, discards, riichi_declared, hand=None, melds=None,
                 dora=None, player_id=0):
        self.discards = discards
        self.riichi_declared = riichi_declared
        self.hand = hand or []
        self.melds = melds or [[], [], [], []]
        self.dora_indicators = dora or []
        self.player_id = player_id


def test_defense_no_active_riichi_is_all_zero():
    o = _DefenseObs([[], [], [], []], [False, False, False, False])
    safe, suji = _compute_riichi_safe_masks(o, 0, 4)
    assert float(safe.sum()) == 0.0
    assert float(suji.sum()) == 0.0


def test_defense_genbutsu_from_opp_river():
    # opp seat1 riichi, 河に 5m(tt=4) -> safe[4]==1
    o = _DefenseObs([[], _ids([4]), [], []], [False, True, False, False])
    safe, _ = _compute_riichi_safe_masks(o, 0, 4)
    assert safe[4] == 1.0


def test_defense_outer_suji_from_center_safe():
    # opp safe に 4m(tt=3) -> suji 1m(0), 7m(6)
    o = _DefenseObs([[], _ids([3]), [], []], [False, True, False, False])
    _, suji = _compute_riichi_safe_masks(o, 0, 4)
    assert suji[0] == 1.0
    assert suji[6] == 1.0


def test_defense_single_suji_not_adopted():
    # 1m(0) だけ safe で 7m が safe でない -> suji[4m]==0 (片スジ不採用)
    o = _DefenseObs([[], _ids([0]), [], []], [False, True, False, False])
    _, suji = _compute_riichi_safe_masks(o, 0, 4)
    assert suji[3] == 0.0


def test_defense_naka_suji_both_outers_safe():
    # 1m(0) と 7m(6) が両方 safe -> suji[4m]==1 (中筋)
    o = _DefenseObs([[], _ids([0, 6]), [], []], [False, True, False, False])
    _, suji = _compute_riichi_safe_masks(o, 0, 4)
    assert suji[3] == 1.0


def test_defense_multi_riichi_uses_and():
    # opp1(seat1) 河に 5m、opp2(seat2) 河に 1m。5m は opp2 に対して safe でない
    # -> AND なので safe[5m]==0
    o = _DefenseObs(
        [[], _ids([4]), _ids([0]), []], [False, True, True, False]
    )
    safe, _ = _compute_riichi_safe_masks(o, 0, 4)
    assert safe[4] == 0.0
    # 共通して safe な牌種は無い
    assert float(safe.sum()) == 0.0


def test_defense_self_riichi_not_counted_as_opponent():
    # 自分(player_id=0)が riichi でも、自分は active riichi opponent ではない
    o = _DefenseObs([_ids([4]), [], [], []], [True, False, False, False])
    safe, suji = _compute_riichi_safe_masks(o, 0, 4)
    assert float(safe.sum()) == 0.0
    assert float(suji.sum()) == 0.0


def test_kabe_center_wall_outer_suji():
    # visible 4m(tt=3) が4枚 (手牌に4枚) -> kabe[1m]==1, [7m]==1
    o = _DefenseObs([[], [], [], []], [False, False, False, False],
                    hand=_ids([3, 3, 3, 3]))
    k = _compute_kabe_suji_mask(o, 4)
    assert k[0] == 1.0
    assert k[6] == 1.0


def test_kabe_terminal_wall_not_adopted():
    # visible 1m(tt=0) が4枚 -> kabe[4m]==0 (端牌壁からの片側筋は不採用)
    o = _DefenseObs([[], [], [], []], [False, False, False, False],
                    hand=_ids([0, 0, 0, 0]))
    k = _compute_kabe_suji_mask(o, 4)
    assert k[3] == 0.0


def test_kabe_visible_counts_aggregate_sources():
    # 4m を 手牌2 + 河1 + ドラ表示1 = 4枚見え -> kabe[1m]==1
    o = _DefenseObs(
        discards=[_ids([3]), [], [], []],
        riichi_declared=[False, False, False, False],
        hand=_ids([3, 3]),
        dora=_ids([3]),
    )
    k = _compute_kabe_suji_mask(o, 4)
    assert k[0] == 1.0
    assert k[6] == 1.0


def test_encoder_integration_defensive_masks_written():
    """encoder 経由で defensive feature が range に正しく書かれる (integration)。"""
    enc = PublicObservationEncoder(enable_hints=True)
    ranges = enc.metadata().feature_ranges
    # seat1 riichi、河に 5m(tt=4)。自手は適当な 13 枚。
    obs = _FakeObs(hand=_ids([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 28, 28]))
    obs.discards = [[], _ids([4, 3]), [], []]
    obs.riichi_declared = [False, True, False, False]
    feat = enc.encode_observation(obs)
    s, e = ranges["safe_vs_all_riichi_mask"]
    safe = feat[s:e]
    # 5m(4) と 4m(3) は opp 河の現物 -> safe
    assert safe[4] == 1.0
    assert safe[3] == 1.0
    s, e = ranges["suji_vs_all_riichi_mask"]
    suji = feat[s:e]
    # 4m(3) が safe -> 1m(0),7m(6) は suji
    assert suji[0] == 1.0
    assert suji[6] == 1.0
