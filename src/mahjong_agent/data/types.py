"""Decision sample / batch dataclasses + schema version.

Stage03 の self-play / imitation / PPO で扱う最小単位の sample 表現と、
学習時の minibatch 表現を定義する。

設計:
- 1 つの ``DecisionSample`` で discard decision と candidate decision の
  両方を表現できる (使わない field は ``None`` / ``-1`` / 0-len array)。
- ``observation`` は public-only な encoder 出力ベクトル。生 env state を
  保持しない。
- ``candidate_features`` は ``(C, candidate_dim)``。``C=0`` でも OK。
- ``yaku_target`` は固定長 multi-hot、``yaku_loss_mask`` は winner-only
  yaku loss を mask out するための float scalar。
- ``metadata`` は forward-compat 用の dict (free-form)。本体 schema を
  壊さず後方互換に追加情報を持たせる。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from mahjong_agent.targets.yaku import NUM_YAKU

# Sample schema バージョン。後方互換が壊れる変更ごとに +1 する。
SCHEMA_VERSION: int = 1

# discard mask の固定長 (4-player tile types)
_DISCARD_MASK_DIM: int = 34


@dataclass
class DecisionSample:
    """1 decision を表す sample。

    Attributes
    ----------
    schema_version:
        Sample schema バージョン。``write_decision_shard`` / ``read_decision_shard``
        は version 一致を verify する。
    episode_id:
        match 内識別子 (例: ``"match_42"``)。
    round_id:
        game 内 round index (0 始まり)。
    step_id:
        match 内連番 step (player_id を跨いで increment)。
    player_id:
        この decision を行う player id (0..3)。
    decision_family:
        ``ActionFamily`` の string 値 (``"normal_discard"`` / ``"chi"`` / ...)。
    actor_type:
        どの actor がこの decision を行ったか (``"policy"`` / ``"baseline"`` /
        ``"random"`` / ``"teacher"`` 等、free-form)。
    observation:
        encoder の固定長 public observation feature。``(observation_dim,)``
        float32。
    discard_mask:
        ``(34,)`` float32 の通常打牌 legal mask。``1.0`` = legal。
        discard 以外の decision でも shape は固定で保持。
    candidate_features:
        ``(C, candidate_dim)`` float32 の candidate feature。``C=0`` のときは
        ``(0, candidate_dim)`` または ``(0, 0)`` のどちらも受け付ける。
    selected_discard_tile_type:
        discard decision のとき、選んだ tile_type (0..33)。それ以外は ``-1``。
    selected_candidate_index:
        candidate decision のとき、選んだ candidate index。それ以外は ``-1``。
    old_log_prob:
        PPO 用 ``log π_old(a|s)``。imitation のみで使わないなら 0.0。
    value:
        value head 推定値 (生 scalar)。
    reward:
        この sample に attribute された (round-level / step-level) reward。
        詳細な reward 設計は learner 側で決める。
    terminated:
        この step で game が終了したか。
    round_over:
        この step で round (kyoku) が終了したか。
    terminal_class:
        terminal auxiliary 5-class index (0..4)。未確定なら ``-1``。
    yaku_target:
        ``(NUM_YAKU,)`` float32 multi-hot target。winner-only。
        non-winner では all-zero。
    yaku_loss_mask:
        BCE loss を mask out する scalar。1.0 で適用、0.0 で無視。
        non-winner や yaku target が無い sample では 0.0。
    han / fu:
        winner の和了点情報。non-winner では ``-1``。
    score_delta:
        round 中に player_id が得失した点数。未確定なら 0。
    teacher_discard_tile_type:
        teacher (rule_base) の discard 推薦 tile_type。無効なら ``-1``。
    teacher_candidate_index:
        teacher の candidate 推薦 index。無効なら ``-1``。
    teacher_available:
        teacher 情報が attach されているか。imitation 学習で mask に使う。
    metadata:
        forward-compat 用 free-form dict。JSON serialize できる値のみ。
    """

    # identity
    schema_version: int = SCHEMA_VERSION
    episode_id: str = ""
    round_id: int = 0
    step_id: int = 0
    player_id: int = 0

    # decision metadata
    decision_family: str = "normal_discard"
    actor_type: str = "policy"

    # observation
    observation: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    discard_mask: np.ndarray = field(
        default_factory=lambda: np.zeros(_DISCARD_MASK_DIM, dtype=np.float32)
    )
    candidate_features: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )

    # decision
    selected_discard_tile_type: int = -1
    selected_candidate_index: int = -1

    # PPO / imitation training info
    old_log_prob: float = 0.0
    value: float = 0.0
    reward: float = 0.0
    terminated: bool = False
    round_over: bool = False

    # auxiliary targets
    terminal_class: int = -1
    yaku_target: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_YAKU, dtype=np.float32)
    )
    yaku_loss_mask: float = 0.0

    # outcome metadata
    han: int = -1
    fu: int = -1
    score_delta: int = 0

    # teacher info
    teacher_discard_tile_type: int = -1
    teacher_candidate_index: int = -1
    teacher_available: bool = False

    # free-form metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionBatch:
    """Collate された minibatch (torch tensors + 補助 list)。

    ``observation`` / ``discard_mask`` / 各 scalar field は (N, ...) tensor。
    ``candidate_features`` は ``(N, Cmax, candidate_dim)`` に padding 済み、
    ``candidate_mask`` で有効 candidate を示す。

    string field (``decision_family`` / ``actor_type`` / ``episode_id``) は
    tensor 化せずに長さ N の list として保持する (downstream で必要なら
    embedding 化する想定)。
    """

    # identity (string lists, length N)
    schema_version: int
    episode_id: list[str]
    round_id: torch.Tensor              # (N,) int64
    step_id: torch.Tensor               # (N,) int64
    player_id: torch.Tensor             # (N,) int64

    # decision metadata
    decision_family: list[str]
    actor_type: list[str]

    # observation
    observation: torch.Tensor           # (N, observation_dim) float32
    discard_mask: torch.Tensor          # (N, 34) float32

    # candidates (padded)
    candidate_features: torch.Tensor    # (N, Cmax, candidate_dim) float32
    candidate_mask: torch.Tensor        # (N, Cmax) float32 (1=valid)
    candidate_count: torch.Tensor       # (N,) int64
    candidate_dim: int

    # decision
    selected_discard_tile_type: torch.Tensor  # (N,) int64
    selected_candidate_index: torch.Tensor    # (N,) int64

    # PPO / imitation
    old_log_prob: torch.Tensor          # (N,) float32
    value: torch.Tensor                 # (N,) float32
    reward: torch.Tensor                # (N,) float32
    terminated: torch.Tensor            # (N,) float32 (1.0 / 0.0)
    round_over: torch.Tensor            # (N,) float32

    # auxiliary
    terminal_class: torch.Tensor        # (N,) int64; -1 if absent
    yaku_target: torch.Tensor           # (N, NUM_YAKU) float32
    yaku_loss_mask: torch.Tensor        # (N,) float32

    # outcome
    han: torch.Tensor                   # (N,) int64
    fu: torch.Tensor                    # (N,) int64
    score_delta: torch.Tensor           # (N,) int64

    # teacher
    teacher_discard_tile_type: torch.Tensor  # (N,) int64
    teacher_candidate_index: torch.Tensor    # (N,) int64
    teacher_available: torch.Tensor          # (N,) float32

    # forward-compat
    metadata: list[dict[str, Any]]

    @property
    def batch_size(self) -> int:
        return self.observation.size(0)


__all__ = [
    "SCHEMA_VERSION",
    "DecisionSample",
    "DecisionBatch",
]
