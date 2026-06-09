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
    # defensive direct hints: 立直相手への守備 mask (現物 / 保守的 suji / 壁筋)。末尾 append。
    ("safe_vs_all_riichi_mask", _NUM_TILE_TYPES),
    ("suji_vs_all_riichi_mask", _NUM_TILE_TYPES),
    ("kabe_suji_mask", _NUM_TILE_TYPES),
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


# candidate safety scalars: (candidate feature 名, 参照する observation hint 名)。
# candidate の tile_type に対応する observation hint value を action-local
# scalar として candidate feature 末尾に append する (discard direct hint と
# 同種情報を candidate scorer にも与える)。observation hint が無い
# (enable_hints=False / tile_type=None) 場合は 0.0 fallback。
_CANDIDATE_SAFETY_SPEC: tuple[tuple[str, str], ...] = (
    ("candidate_shanten_delta", "shanten_delta_per_discard"),
    ("candidate_ukeire_norm", "discard_ukeire_per_tile"),
    ("candidate_safe_vs_all_riichi", "safe_vs_all_riichi_mask"),
    ("candidate_suji_vs_all_riichi", "suji_vs_all_riichi_mask"),
    ("candidate_kabe_suji", "kabe_suji_mask"),
)


def _build_candidate_layout() -> tuple[int, dict[str, tuple[int, int]]]:
    """candidate feature の dim と feature_ranges を組み立てる。"""
    spec: list[tuple[str, int]] = [
        ("family_one_hot", _NUM_FAMILIES),
        ("tile_type_one_hot", _NUM_TILE_TYPES),
        ("tile_type_present_flag", 1),
        ("consume_tile_type_counts", _NUM_TILE_TYPES),
        ("target_rel_seat_one_hot", _NUM_REL_SEATS_PLUS_NONE),
    ]
    # candidate safety scalars (末尾 append、各 1 dim)
    spec.extend((name, 1) for name, _src in _CANDIDATE_SAFETY_SPEC)
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


# tile_type index: man 0-8 (1m-9m), pin 9-17, sou 18-26, honor 27-33。
_SUIT_BASES: tuple[int, ...] = (0, 9, 18)


def _genbutsu_set(discard_tile_ids) -> set[int]:
    """その相手の **自分の河** に出ている tile_type 集合 (= 現物)。

    立直者は自分が捨てた牌種ではロンできない (永続フリテン) ため、相手の河に
    ある牌種はその相手に対して 100% 安全。public-only (obs.discards[opp])。

    Note: 「立直宣言後に第三者が通した牌」も理論上は安全だが、PyPI riichienv の
    Stage03 経路では立直宣言巡目を確実に取得できない (``riichi_sutehais`` が
    二段階 (Riichi, Discard) action 経由では None のまま; riichi 宣言牌が観測
    から復元できない既知の制約と同根)。曖昧推測で hidden state を使わない
    ため、ここでは own-river genbutsu に限定する。これは安全側に過小評価する
    だけで、危険牌を安全と誤判定することはない (= direct hint に入れて安全)。
    """
    out: set[int] = set()
    if not discard_tile_ids:
        return out
    for tid in discard_tile_ids:
        tt = tile_id_to_type(int(tid))
        if 0 <= tt < _NUM_TILE_TYPES:
            out.add(tt)
    return out


def _suji_from_safe_set(safe: set[int]) -> np.ndarray:
    """1 相手の safe 集合から保守的 suji mask ``(34,)`` を作る。

    数牌のみ。**片スジは採用しない** (両面待ちに対して安全と言い切れないため)。
    - 外側筋: 中央牌 (4/5/6) が safe → 同色の外側 (1,7 / 2,8 / 3,9) を suji。
    - 中筋: 外側 2 枚が **両方** safe (1&7 / 2&8 / 3&9) → 中央 (4/5/6) を suji。
    """
    mask = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
    for base in _SUIT_BASES:
        # 外側筋 (center safe -> outers suji)
        if (base + 3) in safe:  # 4
            mask[base + 0] = 1.0  # 1
            mask[base + 6] = 1.0  # 7
        if (base + 4) in safe:  # 5
            mask[base + 1] = 1.0  # 2
            mask[base + 7] = 1.0  # 8
        if (base + 5) in safe:  # 6
            mask[base + 2] = 1.0  # 3
            mask[base + 8] = 1.0  # 9
        # 中筋 (both outers safe -> center suji)
        if (base + 0) in safe and (base + 6) in safe:  # 1 & 7
            mask[base + 3] = 1.0  # 4
        if (base + 1) in safe and (base + 7) in safe:  # 2 & 8
            mask[base + 4] = 1.0  # 5
        if (base + 2) in safe and (base + 8) in safe:  # 3 & 9
            mask[base + 5] = 1.0  # 6
    return mask


def _active_riichi_opponents(obs: Any, player_id: int, num_players: int) -> list[int]:
    """自分以外で立直宣言済みの絶対 seat list。"""
    riichi = obs.riichi_declared if obs.riichi_declared else []
    out: list[int] = []
    for pid in range(num_players):
        if pid == player_id:
            continue
        if pid < len(riichi) and bool(riichi[pid]):
            out.append(pid)
    return out


def _compute_riichi_safe_masks(
    obs: Any, player_id: int, num_players: int
) -> tuple[np.ndarray, np.ndarray]:
    """active riichi opponent **全員に対して** 安全な現物 / 保守的 suji mask。

    Returns
    -------
    (safe_mask, suji_mask):
        各 ``(34,) float32``。active riichi opponent がいなければ両方 all-zero。
        複数立直者がいる場合は **全員に対する AND** (OR で過大評価しない)。
    """
    safe_mask = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
    suji_mask = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
    opps = _active_riichi_opponents(obs, player_id, num_players)
    if not opps:
        return safe_mask, suji_mask
    discards = obs.discards if obs.discards else []
    per_opp_safe: list[np.ndarray] = []
    per_opp_suji: list[np.ndarray] = []
    for opp in opps:
        opp_discards = discards[opp] if opp < len(discards) else []
        safe_set = _genbutsu_set(opp_discards)
        s = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
        for tt in safe_set:
            s[tt] = 1.0
        per_opp_safe.append(s)
        per_opp_suji.append(_suji_from_safe_set(safe_set))
    # AND across all active riichi opponents
    safe_and = per_opp_safe[0].copy()
    suji_and = per_opp_suji[0].copy()
    for s in per_opp_safe[1:]:
        safe_and *= s
    for s in per_opp_suji[1:]:
        suji_and *= s
    return safe_and, suji_and


def _visible_tile_counts(obs: Any, player_id: int, num_players: int) -> np.ndarray:
    """見えている牌の tile_type counts (自手 + 全河 + 全副露 + ドラ表示)。

    public-only。自手は ``obs.hand`` のみ参照し、他家手牌 slot には触らない。
    """
    counts = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
    counts += _tile_id_list_to_counts(obs.hand)
    discards = obs.discards if obs.discards else []
    melds = obs.melds if obs.melds else []
    for pid in range(num_players):
        if pid < len(discards):
            counts += _tile_id_list_to_counts(discards[pid])
        if pid < len(melds):
            for meld in melds[pid]:
                counts += _tile_id_list_to_counts(_meld_tile_ids(meld))
    if obs.dora_indicators:
        counts += _tile_id_list_to_counts(obs.dora_indicators)
    return counts


def _compute_kabe_suji_mask(obs: Any, num_players: int) -> np.ndarray:
    """壁筋 mask ``(34,)``。

    中央数牌 4/5/6 が **4 枚見え** のとき、同色の外側筋 (1,7 / 2,8 / 3,9) を 1。
    完全安全ではなく「両面待ちに対して比較的安全な壁筋」を表す。**2/3/7/8 等
    単独壁からの片側筋は採用しない** (保守的)。字牌は常に 0。
    """
    mask = np.zeros(_NUM_TILE_TYPES, dtype=np.float32)
    visible = _visible_tile_counts(obs, int(obs.player_id), num_players)
    for base in _SUIT_BASES:
        if visible[base + 3] >= 4:  # 4 が4枚見え
            mask[base + 0] = 1.0  # 1
            mask[base + 6] = 1.0  # 7
        if visible[base + 4] >= 4:  # 5
            mask[base + 1] = 1.0  # 2
            mask[base + 7] = 1.0  # 8
        if visible[base + 5] >= 4:  # 6
            mask[base + 2] = 1.0  # 3
            mask[base + 8] = 1.0  # 9
    return mask


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class PublicObservationEncoder:
    """Public-only observation / candidate / legal-mask encoder (v2)。

    ``enable_hints=True`` (default) で shanten / ukeire / shape / defensive
    direct hints / 残り山 / 牌種分布 等の public hint feature を追加する。off
    にすると base layout のみで legacy compat 互換動作。

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

        # 8) defensive direct hints: 立直相手への守備 mask
        safe_mask, suji_mask = _compute_riichi_safe_masks(
            obs, int(obs.player_id), self._num_players
        )
        kabe_mask = _compute_kabe_suji_mask(obs, self._num_players)
        _set_range(feat, self._obs_ranges, "safe_vs_all_riichi_mask", safe_mask)
        _set_range(feat, self._obs_ranges, "suji_vs_all_riichi_mask", suji_mask)
        _set_range(feat, self._obs_ranges, "kabe_suji_mask", kabe_mask)

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

    def _resolve_observation_feat(
        self,
        observation: Any | None,
        observation_feat: np.ndarray | None,
    ) -> np.ndarray | None:
        """candidate safety scalar lookup 用の observation feature を解決する。

        - ``observation_feat`` が渡されたら shape を検証して使う (再 encode しない)。
        - 無く ``observation`` が渡されたら ``encode_observation`` で作る。
        - どちらも無ければ ``None`` (= 5 scalar all-zero fallback)。
        - ``observation_feat`` の shape が ``observation_dim`` と不一致なら
          ``ValueError`` で fail-fast (silent corruption を防ぐ)。
        """
        if observation_feat is not None:
            arr = np.asarray(observation_feat, dtype=np.float32).reshape(-1)
            if arr.shape != (self._obs_dim,):
                raise ValueError(
                    f"observation_feat shape {arr.shape} mismatches "
                    f"observation_dim ({self._obs_dim},)"
                )
            return arr
        if observation is not None:
            return self.encode_observation(observation)
        return None

    def _candidate_safety_scalars(
        self, tile_type: int | None, obs_feat: np.ndarray | None
    ) -> dict[str, float]:
        """candidate tile_type に対応する observation hint scalar を引く。

        tile_type が無い / obs_feat が無い / hint range が無い (enable_hints=False)
        場合は 0.0。hidden info には触れず、既存 public observation hint の
        該当 index を読むだけ。
        """
        out: dict[str, float] = {name: 0.0 for name, _src in _CANDIDATE_SAFETY_SPEC}
        if obs_feat is None:
            return out
        if tile_type is None or not (0 <= int(tile_type) < _NUM_TILE_TYPES):
            return out
        tt = int(tile_type)
        for cand_name, hint_name in _CANDIDATE_SAFETY_SPEC:
            rng = self._obs_ranges.get(hint_name)
            if rng is None:
                continue  # enable_hints=False では hint range が無い → 0.0
            s, _e = rng
            out[cand_name] = float(obs_feat[s + tt])
        return out

    def encode_candidate(
        self,
        candidate: ModelAction,
        *,
        observation: Any | None = None,
        observation_feat: np.ndarray | None = None,
    ) -> np.ndarray:
        """1 candidate を ``(candidate_dim,)`` float32 に変換する。

        ``observation`` / ``observation_feat`` のいずれかが渡された場合、
        candidate の ``tile_type`` に対応する observation hint
        (shanten_delta / ukeire / safe / suji / kabe) を action-local scalar
        として末尾に埋める。どちらも無ければ 5 scalar は 0.0 fallback。
        """
        obs_feat = self._resolve_observation_feat(observation, observation_feat)
        return self._encode_candidate_with_feat(candidate, obs_feat)

    def _encode_candidate_with_feat(
        self, candidate: ModelAction, obs_feat: np.ndarray | None
    ) -> np.ndarray:
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
        # candidate safety scalars (tile_type に対応する observation hint lookup)
        scalars = self._candidate_safety_scalars(candidate.key.tile_type, obs_feat)
        for cand_name, _src in _CANDIDATE_SAFETY_SPEC:
            _set_range(
                feat, _CAND_RANGES, cand_name,
                np.array([scalars[cand_name]], dtype=np.float32),
            )
        return feat

    def encode_candidates(
        self,
        legal_set: LegalActionSet,
        *,
        observation: Any | None = None,
        observation_feat: np.ndarray | None = None,
    ) -> np.ndarray:
        """``legal_set.candidates`` を ``(num_candidates, candidate_dim)`` に変換する。

        candidate が 0 個のときは ``(0, candidate_dim)`` を返す。
        ``observation`` / ``observation_feat`` を渡すと candidate safety scalar
        が埋まる (1 回だけ resolve して全 candidate で再利用)。
        """
        obs_feat = self._resolve_observation_feat(observation, observation_feat)
        n = len(legal_set.candidates)
        out = np.zeros((n, _CAND_DIM), dtype=np.float32)
        for i, c in enumerate(legal_set.candidates):
            out[i] = self._encode_candidate_with_feat(c, obs_feat)
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
