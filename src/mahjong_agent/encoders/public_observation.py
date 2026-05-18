"""Public-only observation / candidate / legal-mask encoder (v1).

このモジュールは ``riichienv.Observation`` と
``mahjong_agent.actions.LegalActionSet`` を入力にして、policy / value /
auxiliary heads に渡せる固定長 numpy feature を生成する。

公開する情報は public-only:

- 自手 (``obs.hand``)
- 4 player の河 (``obs.discards``)
- 4 player の副露 (``obs.melds``、ただし当人視点の公開副露)
- ドラ表示牌 (``obs.dora_indicators``)
- 点数 (``obs.scores``、self-relative 順で並べ替え)
- 局情報 (``obs.oya`` / ``obs.round_wind`` / ``obs.honba`` /
  ``obs.riichi_sticks`` / ``obs.kyoku_index``)
- riichi 状態 (``obs.riichi_declared``)
- 自家テンパイ flag (``obs.is_tenpai``、自手から計算される public 情報)

利用しない情報:

- ``obs.hands[other_player]`` (riichienv 仕様上、他家 slot は空 list で
  来るがガードとして touch しない)
- ``env.hands`` / ``env.wall`` / ``env.state`` (engine 内部 hidden state)
- 山 / 裏ドラ / future draw
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mahjong_agent.actions.types import (
    ActionFamily,
    LegalActionSet,
    ModelAction,
    tile_id_to_type,
)
from mahjong_agent.encoders.metadata import EncoderMetadata

# tile_type / family vocab
_NUM_TILE_TYPES = 34
_NUM_PLAYERS = 4
_NUM_OPPONENTS = _NUM_PLAYERS - 1  # 3 (shimo / toimen / kamicha)
_NUM_FAMILIES = len(ActionFamily)  # 12
_NUM_REL_SEATS_PLUS_NONE = _NUM_PLAYERS + 1  # 4 seats + 1 for "none"


# ---------------------------------------------------------------------------
# observation feature layout (decided once; metadata exposes ranges)
#
# layout (順):
#   self_hand_counts        : 34
#   self_meld_counts        : 34
#   self_open_meld_flag     :  1
#   self_riichi_flag        :  1
#   self_tenpai_flag        :  1
#   discards_self           : 34
#   discards_shimo          : 34
#   discards_toimen         : 34
#   discards_kamicha        : 34
#   melds_shimo             : 34
#   melds_toimen            : 34
#   melds_kamicha           : 34
#   open_meld_flag_opp      :  3
#   riichi_flag_opp         :  3
#   dora_counts             : 34
#   scores_normalized       :  4 (self-relative order)
#   oya_rel_one_hot         :  4
#   round_wind_one_hot      :  4
#   honba_norm              :  1
#   riichi_sticks_norm      :  1
# ---------------------------------------------------------------------------


def _build_observation_layout() -> tuple[int, dict[str, tuple[int, int]]]:
    """observation feature の dim と feature_ranges を組み立てる。"""
    spec: list[tuple[str, int]] = [
        ("self_hand_counts", _NUM_TILE_TYPES),
        ("self_meld_counts", _NUM_TILE_TYPES),
        ("self_open_meld_flag", 1),
        ("self_riichi_flag", 1),
        ("self_tenpai_flag", 1),
        ("discards_self", _NUM_TILE_TYPES),
        ("discards_shimo", _NUM_TILE_TYPES),
        ("discards_toimen", _NUM_TILE_TYPES),
        ("discards_kamicha", _NUM_TILE_TYPES),
        ("melds_shimo", _NUM_TILE_TYPES),
        ("melds_toimen", _NUM_TILE_TYPES),
        ("melds_kamicha", _NUM_TILE_TYPES),
        ("open_meld_flag_opp", _NUM_OPPONENTS),
        ("riichi_flag_opp", _NUM_OPPONENTS),
        ("dora_counts", _NUM_TILE_TYPES),
        ("scores_normalized", _NUM_PLAYERS),
        ("oya_rel_one_hot", _NUM_PLAYERS),
        ("round_wind_one_hot", 4),
        ("honba_norm", 1),
        ("riichi_sticks_norm", 1),
    ]
    ranges: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, dim in spec:
        ranges[name] = (cursor, cursor + dim)
        cursor += dim
    return cursor, ranges


def _build_candidate_layout() -> tuple[int, dict[str, tuple[int, int]]]:
    """candidate feature の dim と feature_ranges を組み立てる。"""
    spec: list[tuple[str, int]] = [
        ("family_one_hot", _NUM_FAMILIES),
        ("tile_type_one_hot", _NUM_TILE_TYPES),
        ("tile_type_present_flag", 1),
        ("consume_tile_type_counts", _NUM_TILE_TYPES),
        ("target_rel_seat_one_hot", _NUM_REL_SEATS_PLUS_NONE),
    ]
    ranges: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, dim in spec:
        ranges[name] = (cursor, cursor + dim)
        cursor += dim
    return cursor, ranges


_OBS_DIM, _OBS_RANGES = _build_observation_layout()
_CAND_DIM, _CAND_RANGES = _build_candidate_layout()


# ---------------------------------------------------------------------------
# tile-count helpers
# ---------------------------------------------------------------------------


def _tile_id_list_to_counts(tile_ids) -> np.ndarray:
    """tile_id (0..135) list を 34-dim tile_type counts に変換する。"""
    counts = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
    if not tile_ids:
        return counts
    for tid in tile_ids:
        tt = tile_id_to_type(int(tid))
        if 0 <= tt < _NUM_TILE_TYPES:
            counts[tt] += 1.0
    return counts


def _meld_tile_ids(meld: Any) -> list[int]:
    """meld オブジェクトから構成 tile_id list を取り出す。

    PyPI 版 ``riichienv.Meld`` は ``tiles`` field を持つ。空の場合は []。
    """
    tiles = getattr(meld, "tiles", None)
    if tiles is None:
        return []
    return list(tiles)


def _opponent_rel_seat_order(num_players: int, player_id: int) -> list[int]:
    """対戦相手 (shimo, toimen, kamicha) の絶対 player_id list を返す。"""
    return [(player_id + off) % num_players for off in range(1, num_players)]


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class PublicObservationEncoder:
    """Public-only observation / candidate / legal-mask encoder (v1).

    Output は np.float32 で固定長。``metadata()`` で各部分の dim / 範囲を
    取得できる。
    """

    def __init__(self, num_players: int = _NUM_PLAYERS):
        if num_players != _NUM_PLAYERS:
            # 3 人麻雀の場合 layout が変わるが、v1 は 4 人麻雀のみ対応。
            raise NotImplementedError(
                "PublicObservationEncoder v1 supports 4-player only "
                f"(got num_players={num_players})"
            )
        self._num_players = num_players

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def metadata(self) -> EncoderMetadata:
        return EncoderMetadata(
            observation_dim=_OBS_DIM,
            candidate_dim=_CAND_DIM,
            discard_mask_dim=_NUM_TILE_TYPES,
            feature_ranges=dict(_OBS_RANGES),
            candidate_feature_ranges=dict(_CAND_RANGES),
        )

    # ------------------------------------------------------------------
    # observation feature
    # ------------------------------------------------------------------

    def encode_observation(self, obs: Any) -> np.ndarray:
        """``riichienv.Observation`` を固定長 (``observation_dim``,) feature に変換する。

        Hidden information (他家手牌 / 山 / engine internal full state) は
        参照しない。``obs.hands`` の他家 slot (= 仕様上空 list) も touch しない。
        """
        feat = np.zeros(_OBS_DIM, dtype=np.float32)
        player_id = int(obs.player_id)
        num_players = self._num_players
        opp_order = _opponent_rel_seat_order(num_players, player_id)

        # self hand
        self_hand_counts = _tile_id_list_to_counts(obs.hand)
        _set_range(feat, _OBS_RANGES, "self_hand_counts", self_hand_counts)

        # self meld counts + open meld flag
        self_melds = obs.melds[player_id] if obs.melds else []
        self_meld_counts = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
        self_has_open_meld = 0.0
        for meld in self_melds:
            self_meld_counts += _tile_id_list_to_counts(_meld_tile_ids(meld))
            if _meld_is_open(meld):
                self_has_open_meld = 1.0
        _set_range(feat, _OBS_RANGES, "self_meld_counts", self_meld_counts)
        _set_range(feat, _OBS_RANGES, "self_open_meld_flag",
                   np.array([self_has_open_meld], dtype=np.float32))

        # self riichi flag + tenpai flag
        self_riichi = 1.0 if obs.riichi_declared[player_id] else 0.0
        _set_range(feat, _OBS_RANGES, "self_riichi_flag",
                   np.array([self_riichi], dtype=np.float32))
        self_tenpai = 1.0 if obs.is_tenpai else 0.0
        _set_range(feat, _OBS_RANGES, "self_tenpai_flag",
                   np.array([self_tenpai], dtype=np.float32))

        # discards (self / shimo / toimen / kamicha)
        _set_range(feat, _OBS_RANGES, "discards_self",
                   _tile_id_list_to_counts(obs.discards[player_id]))
        for rel_idx, opp_pid in enumerate(opp_order, start=1):
            name = {1: "discards_shimo", 2: "discards_toimen",
                    3: "discards_kamicha"}[rel_idx]
            _set_range(feat, _OBS_RANGES, name,
                       _tile_id_list_to_counts(obs.discards[opp_pid]))

        # opponent melds + flags
        open_meld_opp = np.zeros(_NUM_OPPONENTS, dtype=np.float32)
        riichi_opp = np.zeros(_NUM_OPPONENTS, dtype=np.float32)
        for rel_idx, opp_pid in enumerate(opp_order, start=1):
            name = {1: "melds_shimo", 2: "melds_toimen",
                    3: "melds_kamicha"}[rel_idx]
            opp_melds = obs.melds[opp_pid] if obs.melds else []
            opp_meld_counts = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
            for meld in opp_melds:
                opp_meld_counts += _tile_id_list_to_counts(_meld_tile_ids(meld))
                if _meld_is_open(meld):
                    open_meld_opp[rel_idx - 1] = 1.0
            _set_range(feat, _OBS_RANGES, name, opp_meld_counts)
            if obs.riichi_declared[opp_pid]:
                riichi_opp[rel_idx - 1] = 1.0
        _set_range(feat, _OBS_RANGES, "open_meld_flag_opp", open_meld_opp)
        _set_range(feat, _OBS_RANGES, "riichi_flag_opp", riichi_opp)

        # dora indicators
        _set_range(feat, _OBS_RANGES, "dora_counts",
                   _tile_id_list_to_counts(obs.dora_indicators))

        # scores (self / shimo / toimen / kamicha 順、100k で正規化)
        scores_rel = np.zeros(num_players, dtype=np.float32)
        scores_rel[0] = float(obs.scores[player_id]) / 100_000.0
        for rel_idx, opp_pid in enumerate(opp_order, start=1):
            scores_rel[rel_idx] = float(obs.scores[opp_pid]) / 100_000.0
        _set_range(feat, _OBS_RANGES, "scores_normalized", scores_rel)

        # oya: self-relative one-hot
        oya_rel = (int(obs.oya) - player_id) % num_players
        oya_one_hot = np.zeros(num_players, dtype=np.float32)
        oya_one_hot[oya_rel] = 1.0
        _set_range(feat, _OBS_RANGES, "oya_rel_one_hot", oya_one_hot)

        # round wind one-hot
        wind = int(obs.round_wind)
        wind_one_hot = np.zeros(4, dtype=np.float32)
        if 0 <= wind < 4:
            wind_one_hot[wind] = 1.0
        _set_range(feat, _OBS_RANGES, "round_wind_one_hot", wind_one_hot)

        # honba + riichi sticks (norm)
        _set_range(feat, _OBS_RANGES, "honba_norm",
                   np.array([float(obs.honba) / 10.0], dtype=np.float32))
        _set_range(feat, _OBS_RANGES, "riichi_sticks_norm",
                   np.array([float(obs.riichi_sticks) / 4.0], dtype=np.float32))

        return feat

    # ------------------------------------------------------------------
    # legal mask
    # ------------------------------------------------------------------

    def discard_legal_mask(self, legal_set: LegalActionSet) -> np.ndarray:
        """通常打牌の legal mask を ``(34,)`` float32 で返す。"""
        mask = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
        for tt in legal_set.normal_discard.keys():
            if 0 <= tt < _NUM_TILE_TYPES:
                mask[tt] = 1.0
        return mask

    # ------------------------------------------------------------------
    # candidate feature
    # ------------------------------------------------------------------

    def encode_candidate(self, candidate: ModelAction) -> np.ndarray:
        """1 candidate を ``(candidate_dim,)`` float32 に変換する。"""
        feat = np.zeros(_CAND_DIM, dtype=np.float32)
        # family one-hot
        family_one_hot = np.zeros(_NUM_FAMILIES, dtype=np.float32)
        family_one_hot[_family_index(candidate.key.family)] = 1.0
        _set_range(feat, _CAND_RANGES, "family_one_hot", family_one_hot)
        # tile_type one-hot + present flag
        tile_one_hot = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
        present = np.zeros(1, dtype=np.float32)
        tt = candidate.key.tile_type
        if tt is not None and 0 <= int(tt) < _NUM_TILE_TYPES:
            tile_one_hot[int(tt)] = 1.0
            present[0] = 1.0
        _set_range(feat, _CAND_RANGES, "tile_type_one_hot", tile_one_hot)
        _set_range(feat, _CAND_RANGES, "tile_type_present_flag", present)
        # consume tile_type counts
        consume_counts = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
        for cstt in candidate.key.consume_tile_types:
            if 0 <= int(cstt) < _NUM_TILE_TYPES:
                consume_counts[int(cstt)] += 1.0
        _set_range(feat, _CAND_RANGES, "consume_tile_type_counts",
                   consume_counts)
        # target_rel_seat one-hot (4 + 1 for none)
        rel_one_hot = np.zeros(_NUM_REL_SEATS_PLUS_NONE, dtype=np.float32)
        if candidate.key.target_rel_seat is None:
            rel_one_hot[_NUM_PLAYERS] = 1.0  # last index = "none"
        else:
            rel = int(candidate.key.target_rel_seat)
            if 0 <= rel < _NUM_PLAYERS:
                rel_one_hot[rel] = 1.0
            else:
                rel_one_hot[_NUM_PLAYERS] = 1.0
        _set_range(feat, _CAND_RANGES, "target_rel_seat_one_hot", rel_one_hot)
        return feat

    def encode_candidates(self, legal_set: LegalActionSet) -> np.ndarray:
        """``legal_set.candidates`` を ``(num_candidates, candidate_dim)`` に変換する。

        candidate が 0 個のときは ``(0, candidate_dim)`` を返す。
        """
        n = len(legal_set.candidates)
        out = np.zeros((n, _CAND_DIM), dtype=np.float32)
        for i, c in enumerate(legal_set.candidates):
            out[i] = self.encode_candidate(c)
        return out


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _set_range(
    feat: np.ndarray,
    ranges: dict[str, tuple[int, int]],
    name: str,
    values: np.ndarray,
) -> None:
    s, e = ranges[name]
    if values.shape != (e - s,):
        raise ValueError(
            f"feature {name!r} shape mismatch: expected ({e - s},), "
            f"got {values.shape}"
        )
    feat[s:e] = values


def _meld_is_open(meld: Any) -> bool:
    """meld が公開副露 (= Chi/Pon/Daiminkan/Kakan) かを判定する。

    ``riichienv.Meld.opened`` 属性を優先的に使う。属性が無ければ
    ``meld_type`` から Ankan のみ閉じた副露と判定。
    """
    opened = getattr(meld, "opened", None)
    if opened is not None:
        return bool(opened)
    mtype = getattr(meld, "meld_type", None)
    if mtype is None:
        return False
    return str(mtype).rsplit(".", 1)[-1] != "Ankan"


# Stable family -> index map (固定順)
_FAMILY_ORDER = (
    ActionFamily.NORMAL_DISCARD,
    ActionFamily.RIICHI_DISCARD,
    ActionFamily.TSUMO,
    ActionFamily.RON,
    ActionFamily.CHI,
    ActionFamily.PON,
    ActionFamily.DAIMINKAN,
    ActionFamily.ANKAN,
    ActionFamily.KAKAN,
    ActionFamily.KYUSHU_KYUHAI,
    ActionFamily.KITA,
    ActionFamily.PASS,
)
_FAMILY_INDEX = {f: i for i, f in enumerate(_FAMILY_ORDER)}


def _family_index(family: ActionFamily) -> int:
    return _FAMILY_INDEX[family]


__all__ = ["PublicObservationEncoder"]
