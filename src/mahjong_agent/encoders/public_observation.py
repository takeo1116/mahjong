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

- ``obs.hands`` の他家 slot (riichienv 仕様上、他家 slot は空 list で
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
from mahjong_agent.baseline import _fast
from mahjong_agent.baseline.shanten import compute_shanten
from mahjong_agent.baseline.shape import compute_shape_hint
from mahjong_agent.baseline.ukeire import count_acceptance
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
# v1 (base) layout:
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
#
# v2 hints (enable_hints=True, default):
#   current_shanten_norm        :  1  (shanten / 8, clamp [-1/8, 1])
#   shanten_delta_per_discard   : 34  (per-tile_type の打牌後 shanten 変化 / 4)
#   discard_ukeire_per_tile     : 34  (per-tile_type の打牌後 ukeire / 64)
#   remaining_draws_norm        :  1  (山残り tiles / 70)
#   turn_progress_norm          :  1  (sum(all discards) / 70)
#   tile_presence_flags         :  6
#       has_honor / has_terminal / has_simple / has_man / has_pin / has_sou
#   shape_hint                  : 66  (closed chi21 + outside_wait24 + inside_wait21)
#       自手の閉じた順子 / 塔子 / 嵌張 の binary multihot (Stage02 parity)。
#
# Note: 旧仕様にあった ``riichi_discard_mask`` (34 dim) は、Stage03 で使う
# PyPI ``riichienv`` の ``ActionType.RIICHI`` が ``.tile = None`` を返すため
# 常に all-zero になっていた (dead feature)。Stage03 では Riichi 候補の
# tile_type は ``ModelAction``/``LegalActionSet`` 経路で ``env.clone()`` +
# Riichi step 経由で正確に生成しているので、encoder からは削除した。
# ---------------------------------------------------------------------------

_BASE_SPEC: tuple[tuple[str, int], ...] = (
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
)

_SHAPE_HINT_DIM = 66

_HINT_SPEC: tuple[tuple[str, int], ...] = (
    ("current_shanten_norm", 1),
    ("shanten_delta_per_discard", _NUM_TILE_TYPES),
    ("discard_ukeire_per_tile", _NUM_TILE_TYPES),
    ("remaining_draws_norm", 1),
    ("turn_progress_norm", 1),
    ("tile_presence_flags", 6),
    ("shape_hint", _SHAPE_HINT_DIM),
)

# tile_presence_flags の内訳 (固定順)
_TILE_PRESENCE_NAMES: tuple[str, ...] = (
    "has_honor",
    "has_terminal",
    "has_simple",
    "has_man",
    "has_pin",
    "has_sou",
)

# 最大ツモ可能枚数 (4 人麻雀: 136 - 14 dead wall - 13*4 配牌 = 70)
_MAX_DRAWS_PER_KYOKU: int = 70
# shanten clamping / normalization
_SHANTEN_MAX: int = 8
# ukeire 正規化 denom
_UKEIRE_DENOM: float = 64.0


def _build_observation_layout(
    enable_hints: bool,
) -> tuple[int, dict[str, tuple[int, int]]]:
    """observation feature の dim と feature_ranges を組み立てる。"""
    spec: list[tuple[str, int]] = list(_BASE_SPEC)
    if enable_hints:
        spec.extend(_HINT_SPEC)
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


_OBS_DIM_NO_HINTS, _OBS_RANGES_NO_HINTS = _build_observation_layout(False)
_OBS_DIM_WITH_HINTS, _OBS_RANGES_WITH_HINTS = _build_observation_layout(True)
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
    """Public-only observation / candidate / legal-mask encoder (v2)。

    ``enable_hints=True`` (default) で shanten / ukeire / riichi-discard-mask
    / 残り山 / 牌種分布 等の public hint feature を追加する。off にすると
    base layout のみで legacy compat 互換動作。

    Output は np.float32 固定長。``metadata()`` で各 feature の range を
    返す。``Stage03ModelConfig.from_encoder_metadata(encoder.metadata())``
    で model の input dim が自動追従する。
    """

    def __init__(
        self,
        num_players: int = _NUM_PLAYERS,
        *,
        enable_hints: bool = True,
    ):
        if num_players != _NUM_PLAYERS:
            # 3 人麻雀の場合 layout が変わるが、v1 は 4 人麻雀のみ対応。
            raise NotImplementedError(
                "PublicObservationEncoder v1 supports 4-player only "
                f"(got num_players={num_players})"
            )
        self._num_players = num_players
        self._enable_hints = bool(enable_hints)
        if self._enable_hints:
            self._obs_dim = _OBS_DIM_WITH_HINTS
            self._obs_ranges = _OBS_RANGES_WITH_HINTS
        else:
            self._obs_dim = _OBS_DIM_NO_HINTS
            self._obs_ranges = _OBS_RANGES_NO_HINTS

    @property
    def enable_hints(self) -> bool:
        return self._enable_hints

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def metadata(self) -> EncoderMetadata:
        return EncoderMetadata(
            observation_dim=self._obs_dim,
            candidate_dim=_CAND_DIM,
            discard_mask_dim=_NUM_TILE_TYPES,
            feature_ranges=dict(self._obs_ranges),
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
        feat = np.zeros(self._obs_dim, dtype=np.float32)
        player_id = int(obs.player_id)
        num_players = self._num_players
        opp_order = _opponent_rel_seat_order(num_players, player_id)

        # self hand
        self_hand_counts = _tile_id_list_to_counts(obs.hand)
        _set_range(feat, self._obs_ranges, "self_hand_counts", self_hand_counts)

        # self meld counts + open meld flag
        self_melds = obs.melds[player_id] if obs.melds else []
        self_meld_counts = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
        self_has_open_meld = 0.0
        for meld in self_melds:
            self_meld_counts += _tile_id_list_to_counts(_meld_tile_ids(meld))
            if _meld_is_open(meld):
                self_has_open_meld = 1.0
        _set_range(feat, self._obs_ranges, "self_meld_counts", self_meld_counts)
        _set_range(feat, self._obs_ranges, "self_open_meld_flag",
                   np.array([self_has_open_meld], dtype=np.float32))

        # self riichi flag + tenpai flag
        self_riichi = 1.0 if obs.riichi_declared[player_id] else 0.0
        _set_range(feat, self._obs_ranges, "self_riichi_flag",
                   np.array([self_riichi], dtype=np.float32))
        self_tenpai = 1.0 if obs.is_tenpai else 0.0
        _set_range(feat, self._obs_ranges, "self_tenpai_flag",
                   np.array([self_tenpai], dtype=np.float32))

        # discards (self / shimo / toimen / kamicha)
        _set_range(feat, self._obs_ranges, "discards_self",
                   _tile_id_list_to_counts(obs.discards[player_id]))
        for rel_idx, opp_pid in enumerate(opp_order, start=1):
            name = {1: "discards_shimo", 2: "discards_toimen",
                    3: "discards_kamicha"}[rel_idx]
            _set_range(feat, self._obs_ranges, name,
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
            _set_range(feat, self._obs_ranges, name, opp_meld_counts)
            if obs.riichi_declared[opp_pid]:
                riichi_opp[rel_idx - 1] = 1.0
        _set_range(feat, self._obs_ranges, "open_meld_flag_opp", open_meld_opp)
        _set_range(feat, self._obs_ranges, "riichi_flag_opp", riichi_opp)

        # dora indicators
        _set_range(feat, self._obs_ranges, "dora_counts",
                   _tile_id_list_to_counts(obs.dora_indicators))

        # scores (self / shimo / toimen / kamicha 順、100k で正規化)
        scores_rel = np.zeros(num_players, dtype=np.float32)
        scores_rel[0] = float(obs.scores[player_id]) / 100_000.0
        for rel_idx, opp_pid in enumerate(opp_order, start=1):
            scores_rel[rel_idx] = float(obs.scores[opp_pid]) / 100_000.0
        _set_range(feat, self._obs_ranges, "scores_normalized", scores_rel)

        # oya: self-relative one-hot
        oya_rel = (int(obs.oya) - player_id) % num_players
        oya_one_hot = np.zeros(num_players, dtype=np.float32)
        oya_one_hot[oya_rel] = 1.0
        _set_range(feat, self._obs_ranges, "oya_rel_one_hot", oya_one_hot)

        # round wind one-hot
        wind = int(obs.round_wind)
        wind_one_hot = np.zeros(4, dtype=np.float32)
        if 0 <= wind < 4:
            wind_one_hot[wind] = 1.0
        _set_range(feat, self._obs_ranges, "round_wind_one_hot", wind_one_hot)

        # honba + riichi sticks (norm)
        _set_range(feat, self._obs_ranges, "honba_norm",
                   np.array([float(obs.honba) / 10.0], dtype=np.float32))
        _set_range(feat, self._obs_ranges, "riichi_sticks_norm",
                   np.array([float(obs.riichi_sticks) / 4.0], dtype=np.float32))

        if self._enable_hints:
            self._encode_hints(
                feat, obs, self_hand_counts, self_melds,
            )

        return feat

    # ------------------------------------------------------------------
    # v2 hint features
    # ------------------------------------------------------------------

    def _encode_hints(
        self,
        feat: np.ndarray,
        obs: Any,
        self_hand_counts: np.ndarray,
        self_melds: list,
    ) -> None:
        """v2 hint features を ``feat`` の対応 range に書き込む (in-place)。

        hidden info を一切使わず、``obs.hand`` (自手) と ``obs.legal_actions()``
        (公開された legal mask) と ``obs.discards`` (公開河) のみを参照する。
        """
        # 自手 counts (int list 化)
        hand_counts_list = [int(x) for x in self_hand_counts.tolist()]
        meld_count = len(self_melds)
        # 1) current shanten
        try:
            current_shanten = compute_shanten(hand_counts_list, meld_count)
        except ValueError:
            current_shanten = _SHANTEN_MAX
        _set_range(
            feat, self._obs_ranges, "current_shanten_norm",
            np.array([float(current_shanten) / float(_SHANTEN_MAX)],
                     dtype=np.float32),
        )

        # 2) shanten_delta_per_discard / 3) discard_ukeire_per_tile
        shanten_delta = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
        ukeire_per_tile = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
        if _fast.FAST_AVAILABLE:
            # C++ fast path: 全 tile の shanten_after / acceptance を 1 回で取得。
            # legal_mask は「手牌にある全 tile」(= analyze_discards が
            # counts<1 を自動 skip するので all-ones で渡す)。Python loop と
            # 同一 semantics (seen_counts=None = 手牌側のみ既見扱い)。
            try:
                analysis = _fast.analyze_discards(
                    hand_counts_list, [1] * _NUM_TILE_TYPES, meld_count
                )
                sh_after_arr = analysis["shanten_after"]
                acc_arr = analysis["acceptance"]
            except (ValueError, RuntimeError):
                sh_after_arr = None
                acc_arr = None
            if sh_after_arr is not None:
                for t in range(_NUM_TILE_TYPES):
                    if hand_counts_list[t] <= 0:
                        continue
                    sh_after = int(sh_after_arr[t])
                    acc = int(acc_arr[t])
                    d = current_shanten - sh_after
                    shanten_delta[t] = max(-1.0, min(1.0, float(d) / 4.0))
                    ukeire_per_tile[t] = min(1.0, float(acc) / _UKEIRE_DENOM)
                _set_range(
                    feat, self._obs_ranges,
                    "shanten_delta_per_discard", shanten_delta,
                )
                _set_range(
                    feat, self._obs_ranges,
                    "discard_ukeire_per_tile", ukeire_per_tile,
                )
                return self._encode_hints_tail(
                    feat, obs, hand_counts_list
                )
        # Python fallback: per-tile に shanten + ukeire を計算する。
        for t in range(_NUM_TILE_TYPES):
            if hand_counts_list[t] <= 0:
                continue
            hand_counts_list[t] -= 1
            try:
                sh_after = compute_shanten(hand_counts_list, meld_count)
                acc = count_acceptance(hand_counts_list, sh_after, meld_count)
            except ValueError:
                sh_after = current_shanten
                acc = 0
            hand_counts_list[t] += 1
            # delta > 0 ⇒ 打牌すると shanten が下がる (= 良い)。/4 で正規化、
            # [-1, 1] にクランプ。
            d = current_shanten - sh_after
            shanten_delta[t] = max(-1.0, min(1.0, float(d) / 4.0))
            ukeire_per_tile[t] = min(
                1.0, float(acc) / _UKEIRE_DENOM
            )
        _set_range(feat, self._obs_ranges, "shanten_delta_per_discard", shanten_delta)
        _set_range(feat, self._obs_ranges, "discard_ukeire_per_tile", ukeire_per_tile)
        self._encode_hints_tail(feat, obs, hand_counts_list)
        return None

    def _encode_hints_tail(
        self,
        feat: np.ndarray,
        obs: Any,
        hand_counts_list: list[int],
    ) -> None:
        """hint feature の残り (remaining_draws / turn_progress /
        tile_presence_flags) を書き込む。fast / fallback 両 path から呼ぶ。"""
        # 4) remaining_draws_norm / 5) turn_progress_norm
        # 公開河の枚数合計から残り山を概算する。
        total_discarded = 0
        for d in obs.discards if obs.discards else []:
            total_discarded += len(d)
        remaining = max(0, _MAX_DRAWS_PER_KYOKU - total_discarded)
        _set_range(
            feat, self._obs_ranges, "remaining_draws_norm",
            np.array(
                [min(1.0, float(remaining) / float(_MAX_DRAWS_PER_KYOKU))],
                dtype=np.float32,
            ),
        )
        _set_range(
            feat, self._obs_ranges, "turn_progress_norm",
            np.array(
                [min(1.0,
                     float(total_discarded) / float(_MAX_DRAWS_PER_KYOKU))],
                dtype=np.float32,
            ),
        )

        # 6) tile_presence_flags
        flags = _tile_presence_flags(hand_counts_list)
        _set_range(feat, self._obs_ranges, "tile_presence_flags", flags)

        # 7) shape_hint (Stage02 parity): 閉じた順子 / 塔子 / 嵌張 の multihot
        shape = compute_shape_hint(hand_counts_list)
        _set_range(feat, self._obs_ranges, "shape_hint", shape)

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


# tile_presence_flags 用の tile_type set 定義
_HONOR_TT: tuple[int, ...] = tuple(range(27, 34))
_TERMINAL_TT: tuple[int, ...] = (0, 8, 9, 17, 18, 26)
_MAN_TT: tuple[int, ...] = tuple(range(0, 9))
_PIN_TT: tuple[int, ...] = tuple(range(9, 18))
_SOU_TT: tuple[int, ...] = tuple(range(18, 27))


def _tile_presence_flags(counts: list[int]) -> np.ndarray:
    """自手 counts から (6,) tile-presence flag を作る。

    順: has_honor / has_terminal / has_simple / has_man / has_pin / has_sou。
    """
    has_honor = any(counts[t] > 0 for t in _HONOR_TT)
    has_terminal = any(counts[t] > 0 for t in _TERMINAL_TT)
    # has_simple = 数牌 2..8 が 1 枚でもある
    has_simple = False
    for suit_base in (0, 9, 18):
        for offset in range(1, 8):
            if counts[suit_base + offset] > 0:
                has_simple = True
                break
        if has_simple:
            break
    has_man = any(counts[t] > 0 for t in _MAN_TT)
    has_pin = any(counts[t] > 0 for t in _PIN_TT)
    has_sou = any(counts[t] > 0 for t in _SOU_TT)
    return np.array(
        [
            float(has_honor),
            float(has_terminal),
            float(has_simple),
            float(has_man),
            float(has_pin),
            float(has_sou),
        ],
        dtype=np.float32,
    )


__all__ = ["PublicObservationEncoder"]
