"""Stage03 multi-head model v1.

Multi-head policy/value/auxiliary model for the public-only encoder.

Heads:
- 34-way discard policy logits
- variable-count candidate scorer
- value head (scalar per sample)
- terminal auxiliary head (5-class)
- yaku auxiliary head (49-class multi-label)

すべての head は public-only の shared trunk から分岐する。default では
semantic summary を policy 経路に押し込まない (= terminal / yaku auxiliary は
value side で別 head として学ぶだけ)。Stage02 parity 用に、terminal / yaku
出力を detach して policy 経路へ戻す opt-in 経路も持つ。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from mahjong_agent.encoders.metadata import EncoderMetadata
from mahjong_agent.targets.terminal import NUM_TERMINAL_CLASSES
from mahjong_agent.targets.yaku import NUM_YAKU

# tile_type の数 (discard head の出力 dim)
_NUM_TILE_TYPES = 34


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage03ModelConfig:
    """Stage03 v1 model の hyperparameters。

    Attributes
    ----------
    observation_dim:
        encoder ``encode_observation()`` の output dim (= ``trunk`` の入力)。
    candidate_dim:
        encoder ``encode_candidate()`` の output dim (= candidate scorer の
        candidate-side 入力)。
    hidden_dim:
        shared trunk hidden 幅。
    trunk_layers:
        shared trunk の Linear+ReLU 段数 (>=1)。
    candidate_hidden_dim:
        candidate scorer 内 MLP hidden 幅。
    dropout:
        trunk / candidate scorer dropout 率 (0.0 で無効)。
    num_terminal_classes:
        terminal auxiliary head の出力 class 数。default 5。
    num_yaku:
        yaku auxiliary head の出力 class 数。default ``NUM_YAKU`` (= 49)。
    num_tile_types:
        discard head の出力 dim。default 34。
    semantic_summary_in_policy:
        ``False`` (default) で旧 architecture と完全互換。``True`` のとき、
        terminal head と yaku head の出力から作った detached summary feature
        を discard head / candidate scorer の入力に追加する (Stage02
        semantic auxiliary injection の Stage03 移植)。policy 経路に流す際は
        必ず detach する (= summary は terminal/yaku loss だけで学ぶ)。
        既存 checkpoint との互換性を壊さないため default は False。
    semantic_summary_detach:
        ``True`` (default, 推奨)。``False`` は debugging 用で、summary に
        gradient を逆流させる (Stage02 知見に反するので production では
        使わない)。
    """

    observation_dim: int
    candidate_dim: int
    hidden_dim: int = 256
    trunk_layers: int = 2
    candidate_hidden_dim: int = 128
    dropout: float = 0.0
    num_terminal_classes: int = NUM_TERMINAL_CLASSES
    num_yaku: int = NUM_YAKU
    num_tile_types: int = _NUM_TILE_TYPES
    semantic_summary_in_policy: bool = False
    semantic_summary_detach: bool = True

    @classmethod
    def from_encoder_metadata(
        cls,
        metadata: EncoderMetadata,
        *,
        hidden_dim: int = 256,
        trunk_layers: int = 2,
        candidate_hidden_dim: int = 128,
        dropout: float = 0.0,
        semantic_summary_in_policy: bool = False,
        semantic_summary_detach: bool = True,
    ) -> Stage03ModelConfig:
        """``EncoderMetadata`` から observation_dim / candidate_dim を引き継いで config を作る。"""
        return cls(
            observation_dim=int(metadata.observation_dim),
            candidate_dim=int(metadata.candidate_dim),
            hidden_dim=int(hidden_dim),
            trunk_layers=int(trunk_layers),
            candidate_hidden_dim=int(candidate_hidden_dim),
            dropout=float(dropout),
            semantic_summary_in_policy=bool(semantic_summary_in_policy),
            semantic_summary_detach=bool(semantic_summary_detach),
        )


# ---------------------------------------------------------------------------
# Output dataclasses (NamedTuple for torch.jit-friendliness & immutability)
# ---------------------------------------------------------------------------


class Stage03ForwardOutput(NamedTuple):
    """``Stage03Model.forward`` の戻り値。

    Attributes
    ----------
    discard_logits:
        ``(B, num_tile_types)``。discard_mask が渡された場合は illegal idx に
        ``-1e9`` 相当の負値を入れた masked logits。
    value:
        ``(B,)``。value head の出力 (raw scalar、squashing なし)。
    terminal_logits:
        ``(B, num_terminal_classes)``。CE loss 用 raw logits。
    yaku_logits:
        ``(B, num_yaku)``。multi-label BCE 用 raw logits。
    """

    discard_logits: torch.Tensor
    value: torch.Tensor
    terminal_logits: torch.Tensor
    yaku_logits: torch.Tensor


class CandidateScoreOutput(NamedTuple):
    """``Stage03Model.score_candidates`` の戻り値。

    Attributes
    ----------
    candidate_scores:
        ``(B, C)``。各 candidate に対する scalar score。``C=0`` のときは
        ``(B, 0)``。softmax / log-softmax は呼び出し側で適用する想定。
    """

    candidate_scores: torch.Tensor


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _build_trunk(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    """MLP trunk: Linear -> ReLU [-> Dropout] を ``num_layers`` 段重ねる。"""
    if num_layers < 1:
        raise ValueError(
            f"trunk_layers must be >= 1, got {num_layers}"
        )
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        in_dim = hidden_dim
    return nn.Sequential(*layers)


class Stage03Model(nn.Module):
    """Stage03 multi-head policy / value / auxiliary model (v1).

    public-only feature ``obs_features`` のみを入力にし、5 つの head を持つ:

    - shared trunk -> discard policy logits (34-way)
    - shared trunk + candidate features -> candidate scores
    - shared trunk -> value
    - shared trunk -> terminal auxiliary logits (5-class)
    - shared trunk -> yaku auxiliary logits (49-class multi-label)

    candidate scorer は trunk hidden を candidate 数だけ broadcast して
    candidate feature と結合し、MLP で per-candidate scalar score を計算する。
    """

    def __init__(self, config: Stage03ModelConfig) -> None:
        super().__init__()
        self._config = config

        self.trunk = _build_trunk(
            input_dim=config.observation_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.trunk_layers,
            dropout=config.dropout,
        )
        # value head (scalar -> squeeze)
        self.value_head = nn.Linear(config.hidden_dim, 1)
        # terminal auxiliary head
        self.terminal_head = nn.Linear(
            config.hidden_dim, config.num_terminal_classes
        )
        # yaku auxiliary head
        self.yaku_head = nn.Linear(config.hidden_dim, config.num_yaku)
        # semantic summary を policy に流すかどうかで discard / candidate head
        # の入力 dim が変わる。default off では従来 architecture と一致する。
        if config.semantic_summary_in_policy:
            self._semantic_summary_dim = int(
                config.num_terminal_classes + config.num_yaku
            )
        else:
            self._semantic_summary_dim = 0
        discard_in = config.hidden_dim + self._semantic_summary_dim
        self.discard_head = nn.Linear(discard_in, config.num_tile_types)
        # candidate scorer MLP: input = trunk hidden + semantic summary (opt) +
        # candidate feature。
        scorer_in = (
            config.hidden_dim + self._semantic_summary_dim + config.candidate_dim
        )
        scorer_layers: list[nn.Module] = [
            nn.Linear(scorer_in, config.candidate_hidden_dim),
            nn.ReLU(),
        ]
        if config.dropout > 0.0:
            scorer_layers.append(nn.Dropout(p=config.dropout))
        scorer_layers.append(nn.Linear(config.candidate_hidden_dim, 1))
        self.candidate_scorer = nn.Sequential(*scorer_layers)

    @property
    def config(self) -> Stage03ModelConfig:
        return self._config

    # ------------------------------------------------------------------
    # main forward (4 heads)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs_features: torch.Tensor,
        discard_mask: torch.Tensor | None = None,
    ) -> Stage03ForwardOutput:
        """obs feature から discard / value / terminal / yaku を一括 forward する。

        Parameters
        ----------
        obs_features:
            ``(B, observation_dim)`` float tensor。
        discard_mask:
            ``(B, num_tile_types)`` 0/1 tensor (float or bool)。illegal idx は
            出力 logits に ``-1e9`` を足す。``None`` のときは mask 無し。

        Returns
        -------
        Stage03ForwardOutput
        """
        if obs_features.dim() != 2:
            raise ValueError(
                f"obs_features must be 2D (B, observation_dim), got shape "
                f"{tuple(obs_features.shape)}"
            )
        if obs_features.size(-1) != self._config.observation_dim:
            raise ValueError(
                f"obs_features last dim must be {self._config.observation_dim}, "
                f"got {obs_features.size(-1)}"
            )
        h = self.trunk(obs_features.float())
        value = self.value_head(h).squeeze(-1)
        terminal_logits = self.terminal_head(h)
        yaku_logits = self.yaku_head(h)
        discard_input = self._augment_with_semantic_summary(
            h, terminal_logits, yaku_logits
        )
        discard_logits = self.discard_head(discard_input)
        if discard_mask is not None:
            discard_logits = _apply_discard_mask(discard_logits, discard_mask)
        return Stage03ForwardOutput(
            discard_logits=discard_logits,
            value=value,
            terminal_logits=terminal_logits,
            yaku_logits=yaku_logits,
        )

    # ------------------------------------------------------------------
    # candidate scorer (variable C)
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        obs_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> CandidateScoreOutput:
        """obs と candidate features から ``(B, C)`` の per-candidate score を出す。

        Parameters
        ----------
        obs_features:
            ``(B, observation_dim)`` float tensor。
        candidate_features:
            ``(B, C, candidate_dim)`` float tensor。``C=0`` でも crash しない。

        Returns
        -------
        CandidateScoreOutput
        """
        if obs_features.dim() != 2:
            raise ValueError(
                f"obs_features must be 2D, got shape "
                f"{tuple(obs_features.shape)}"
            )
        if candidate_features.dim() != 3:
            raise ValueError(
                f"candidate_features must be 3D (B, C, candidate_dim), got "
                f"shape {tuple(candidate_features.shape)}"
            )
        if candidate_features.size(0) != obs_features.size(0):
            raise ValueError(
                f"batch size mismatch: obs={obs_features.size(0)} vs "
                f"candidates={candidate_features.size(0)}"
            )
        if candidate_features.size(-1) != self._config.candidate_dim:
            raise ValueError(
                f"candidate_features last dim must be "
                f"{self._config.candidate_dim}, got "
                f"{candidate_features.size(-1)}"
            )
        B, C, _ = candidate_features.shape
        h = self.trunk(obs_features.float())  # (B, H)
        if C == 0:
            return CandidateScoreOutput(
                candidate_scores=torch.zeros(
                    B, 0,
                    dtype=h.dtype,
                    device=h.device,
                )
            )
        # h を (B, C, H) に expand
        h_expanded = h.unsqueeze(1).expand(B, C, h.size(-1))
        if self._semantic_summary_dim > 0:
            terminal_logits = self.terminal_head(h)
            yaku_logits = self.yaku_head(h)
            summary = self._compute_semantic_summary(terminal_logits, yaku_logits)
            summary_expanded = summary.unsqueeze(1).expand(B, C, summary.size(-1))
            scorer_input = torch.cat(
                [h_expanded, summary_expanded, candidate_features.float()],
                dim=-1,
            )
        else:
            scorer_input = torch.cat(
                [h_expanded, candidate_features.float()], dim=-1
            )
        scores = self.candidate_scorer(scorer_input).squeeze(-1)  # (B, C)
        return CandidateScoreOutput(candidate_scores=scores)

    # ------------------------------------------------------------------
    # semantic summary helpers (Stage02 CQ-0256 移植)
    # ------------------------------------------------------------------

    def _compute_semantic_summary(
        self,
        terminal_logits: torch.Tensor,
        yaku_logits: torch.Tensor,
    ) -> torch.Tensor:
        """terminal / yaku logits を policy 入力用に確率化して連結する。

        Stage02 仕様: terminal は softmax、yaku は sigmoid を取り、policy 経路に
        流すときは detach する。``config.semantic_summary_detach=False`` のときだけ
        gradient を逆流させる (debug 用)。
        """
        terminal_prob = torch.softmax(terminal_logits, dim=-1)
        yaku_prob = torch.sigmoid(yaku_logits)
        summary = torch.cat([terminal_prob, yaku_prob], dim=-1)
        if self._config.semantic_summary_detach:
            summary = summary.detach()
        return summary

    def _augment_with_semantic_summary(
        self,
        trunk_hidden: torch.Tensor,
        terminal_logits: torch.Tensor,
        yaku_logits: torch.Tensor,
    ) -> torch.Tensor:
        """discard head 入力を semantic summary で拡張する (opt-in)。"""
        if self._semantic_summary_dim == 0:
            return trunk_hidden
        summary = self._compute_semantic_summary(terminal_logits, yaku_logits)
        return torch.cat([trunk_hidden, summary], dim=-1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _apply_discard_mask(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """``(B, K)`` logits に ``(B, K)`` mask (1=legal, 0=illegal) を適用する。

    illegal idx に ``-1e9`` を足し、softmax 前に挙動を整える。mask は float
    でも bool でも可。
    """
    if mask.shape != logits.shape:
        raise ValueError(
            f"discard_mask shape {tuple(mask.shape)} does not match "
            f"discard_logits shape {tuple(logits.shape)}"
        )
    if mask.dtype == torch.bool:
        legal = mask.to(logits.dtype)
    else:
        legal = mask.to(logits.dtype)
    # illegal idx に大きな負値を加算
    return logits + (1.0 - legal) * (-1.0e9)


__all__ = [
    "Stage03ModelConfig",
    "Stage03ForwardOutput",
    "CandidateScoreOutput",
    "Stage03Model",
]
