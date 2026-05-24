"""PPO learner v1 with Stage02 stabilizers.

Stage03 用 PPO trainer。``DecisionSample`` (= public-only encoder 出力 +
reward / value / old_log_prob 等) を入力に、``Stage03Model`` の discard /
candidate policy + value + auxiliary heads を更新する。

Stage02 で有効だった以下を **default で** 入れる:

- target_kl による minibatch / epoch early stop
- entropy bonus
- decision_family (discard / candidate) ごとの diagnostics
- mixed off-policy baseline を PPO ratio に **混ぜない** (``actor_type ==
  "policy"`` の sample のみ PPO 対象、それ以外は ratio から除外)
- gradient norm diagnostics は default off (速度影響を避ける)
- skipped minibatch を applied diagnostics に **混ぜない**

注意:
- ``RiichiEnv`` / encoder には触らない。trainer は ``Stage03Model`` の public
  API (``forward`` / ``score_candidates``) と ``DecisionBatch`` だけを使う。
- hidden info critic は導入しない (value head は public-only feature から
  predict する ``Stage03Model.value`` をそのまま使う)。
- reward target は ``DecisionSample.reward`` (= round_score_delta * 1e-4) を
  そのまま使う。trainer 側で reward scale を変えない。
"""
from __future__ import annotations

import json
import math
import random as _random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn, optim

from mahjong_agent.actions.types import ActionFamily
from mahjong_agent.data.collate import collate_decision_samples
from mahjong_agent.data.types import DecisionBatch, DecisionSample
from mahjong_agent.models.stage03_model import Stage03Model
from mahjong_agent.training.optimizer_groups import (
    LRGroupConfig,
    build_lr_grouped_optimizer,
)
from mahjong_agent.training.sample_weighting import compute_per_player_round_weights

_NORMAL_DISCARD_FAMILY: str = ActionFamily.NORMAL_DISCARD.value
_LARGE_NEG: float = -1.0e9


# ---------------------------------------------------------------------------
# Config / metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PPOConfig:
    """PPO learner の hyperparameters。

    Default 値の根拠
    ----------------
    - ``gamma=0.95``: project rule に固定値は無いので Stage03 初期値として
      設定。round-level reward (round_score_delta * 1e-4) に対して discount
      が強すぎないバランスを取った。
    - ``gae_lambda=0.95``: 標準的な PPO 設定。
    - ``clip_epsilon=0.2``: 標準的な PPO 設定。
    - ``value_loss_coef=0.5``: PPO 標準。
    - ``entropy_coef=0.01``: 過剰な entropy collapse を防ぐ程度の小さい coef。
    - ``terminal_loss_coef=0.2`` / ``yaku_loss_coef=0.1``: imitation trainer
      と同じ conservative default。
    - ``max_grad_norm=1.0``: gradient clipping を default で適用する (Stage02
      の知見)。``None`` で無効化可能。
    - ``policy_only=False`` / ``separated=True``: spec の推奨 default。
      policy_only=True にすると value / aux loss を抑える (= warm-up policy
      update に使う想定)。separated は discard / candidate を metrics 上で
      別に出すモード。default で diagnostics は family 別に出すため、フラグ
      自体は将来の loss 分離 / 別 optimizer step に向けた slot として保持。
    - ``target_kl=0.05`` / ``target_kl_stop_multiplier=1.5``: Stage02 で
      安定だった範囲。``target_kl_skip_minibatch_on_exceed=True`` で対象
      minibatch を skip して applied diagnostics に混ぜない。
    - ``gradient_norms_enabled=False``: 速度影響を default で避けるため
      off。enabled 時は per-component grad norm を ``torch.autograd.grad``
      で計測するが、``gradient_norms_max_batches_per_epoch=4`` で限定。
    - ``advantage_normalize=True``: PPO は advantage scale に敏感なので、
      minibatch ごとに mean/std で正規化する。
    - ``include_actor_types=("policy",)``: PPO ratio に baseline / imitation
      の off-policy sample を **混ぜない**。actor_type が含まれない sample
      は ratio 計算から除外し、metrics の ``ppo_excluded_by_actor_type`` に
      集計する。
    """

    learning_rate: float = 3e-4
    batch_size: int = 256
    num_epochs: int = 1
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    terminal_loss_coef: float = 0.2
    yaku_loss_coef: float = 0.1
    max_grad_norm: float | None = 1.0
    weight_decay: float = 0.0
    device: str = "cpu"
    shuffle: bool = True
    seed: int | None = None
    policy_only: bool = False
    separated: bool = True
    target_kl_enabled: bool = True
    target_kl: float = 0.05
    target_kl_stop_multiplier: float = 1.5
    target_kl_skip_minibatch_on_exceed: bool = True
    gradient_norms_enabled: bool = False
    gradient_norms_max_batches_per_epoch: int = 4
    advantage_normalize: bool = True
    include_actor_types: tuple[str, ...] = ("policy",)
    # Stage02 由来の learner stabilizer (opt-in)。default は全て off で
    # 現行挙動と互換。
    lr_group_config: LRGroupConfig | None = None
    """Stage02 由来の policy / value_semantic / trunk lr 分離。``None`` または
    ``LRGroupConfig(enabled=False)`` で single group optimizer (現行互換)。"""
    exclude_post_riichi_discards: bool = False
    """``DecisionSample.metadata["is_post_riichi_discard"]==True`` を policy /
    value 両方の loss から除外する (Stage02 CQ-0164 移植)。default off。"""
    value_loss_includes_excluded: bool = False
    """``ppo_exclude=True`` (= TSUMO / RON / KYUSHU shortcut sample) を value
    loss に **だけ** 含める。policy loss には絶対に混ぜない。default off。"""
    per_player_round_weighting: bool = False
    """同一 ``(episode_id, round_id, player_id)`` の sample 重み合計が 1.0 に
    なるよう normalize する weighting (Stage02 CQ-0268 移植)。default off。"""


@dataclass(frozen=True)
class PPOMetrics:
    """1 batch または 1 epoch の PPO metrics (JSON serializable)。

    Note
    ----
    minibatch を target_kl で **skip した場合**、その minibatch は
    ``target_kl_skipped_minibatches`` にだけ加算され、``applied`` / loss /
    grad_norm 等 ``更新に使った diagnostics`` には混ぜない。
    """

    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    terminal_loss: float = 0.0
    yaku_loss: float = 0.0
    discard_policy_loss: float = 0.0
    candidate_policy_loss: float = 0.0
    discard_count: int = 0
    candidate_count: int = 0
    value_count: int = 0
    terminal_count: int = 0
    yaku_count: int = 0
    num_samples: int = 0
    num_batches: int = 0
    ppo_included_count: int = 0
    ppo_excluded_count: int = 0
    ppo_excluded_by_actor_type: int = 0
    clip_fraction: float = 0.0
    approx_kl_mean: float = 0.0
    approx_kl_max: float = 0.0
    target_kl_checked_minibatches: int = 0
    target_kl_skipped_minibatches: int = 0
    target_kl_applied_minibatches: int = 0
    num_updates: int = 0
    grad_norm: float = 0.0
    # Stage02 stabilizer diagnostics (opt-in flags の効果を可視化)
    post_riichi_excluded_count: int = 0
    """``exclude_post_riichi_discards`` で policy / value 両方から除外した
    sample 数。flag off では常に 0。"""
    value_loss_extra_shortcut_count: int = 0
    """``value_loss_includes_excluded=True`` のとき、value loss に追加で含めた
    shortcut (= ``ppo_exclude=True``) sample 数。flag off では常に 0。"""
    decision_family: dict[str, dict[str, float]] = field(default_factory=dict)
    actor_type_counts: dict[str, int] = field(default_factory=dict)
    grad_norms_per_component: dict[str, float] = field(default_factory=dict)
    lr_groups_info: dict[str, Any] = field(default_factory=dict)
    """``build_lr_grouped_optimizer`` から得た JSON serializable diagnostics。
    optimizer を共有する epoch 集約では同じ dict を再 attach する。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": float(self.loss),
            "policy_loss": float(self.policy_loss),
            "value_loss": float(self.value_loss),
            "entropy": float(self.entropy),
            "terminal_loss": float(self.terminal_loss),
            "yaku_loss": float(self.yaku_loss),
            "discard_policy_loss": float(self.discard_policy_loss),
            "candidate_policy_loss": float(self.candidate_policy_loss),
            "discard_count": int(self.discard_count),
            "candidate_count": int(self.candidate_count),
            "value_count": int(self.value_count),
            "terminal_count": int(self.terminal_count),
            "yaku_count": int(self.yaku_count),
            "num_samples": int(self.num_samples),
            "num_batches": int(self.num_batches),
            "ppo_included_count": int(self.ppo_included_count),
            "ppo_excluded_count": int(self.ppo_excluded_count),
            "ppo_excluded_by_actor_type": int(self.ppo_excluded_by_actor_type),
            "clip_fraction": float(self.clip_fraction),
            "approx_kl_mean": float(self.approx_kl_mean),
            "approx_kl_max": float(self.approx_kl_max),
            "target_kl_checked_minibatches": int(
                self.target_kl_checked_minibatches
            ),
            "target_kl_skipped_minibatches": int(
                self.target_kl_skipped_minibatches
            ),
            "target_kl_applied_minibatches": int(
                self.target_kl_applied_minibatches
            ),
            "num_updates": int(self.num_updates),
            "grad_norm": float(self.grad_norm),
            "post_riichi_excluded_count": int(self.post_riichi_excluded_count),
            "value_loss_extra_shortcut_count": int(
                self.value_loss_extra_shortcut_count
            ),
            "decision_family": {
                str(k): {str(kk): float(vv) for kk, vv in v.items()}
                for k, v in self.decision_family.items()
            },
            "actor_type_counts": {
                str(k): int(v) for k, v in self.actor_type_counts.items()
            },
            "grad_norms_per_component": {
                str(k): float(v) for k, v in self.grad_norms_per_component.items()
            },
            "lr_groups_info": dict(self.lr_groups_info),
        }


@dataclass
class PPORunResult:
    """``fit_ppo`` の戻り値: per-epoch metrics + final aggregate。"""

    epochs: list[PPOMetrics] = field(default_factory=list)
    final: PPOMetrics | None = None
    early_stopped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": [m.to_dict() for m in self.epochs],
            "final": self.final.to_dict() if self.final is not None else None,
            "early_stopped": bool(self.early_stopped),
        }


# ---------------------------------------------------------------------------
# Returns / advantages
# ---------------------------------------------------------------------------


@dataclass
class PPOTrainingData:
    """``compute_returns_and_advantages`` の戻り値。

    Attributes
    ----------
    samples:
        入力と同じ ``DecisionSample`` の list (順序は変えない)。
    returns:
        ``(N,) float32``。samples と同 order。
    advantages:
        ``(N,) float32``。samples と同 order。
    eligible:
        ``(N,) bool``。policy loss 対象 (= ``actor_type`` が
        ``config.include_actor_types`` に含まれる、かつ ``decision_family``
        が discard か候補で ``selected_*`` が有効、shortcut / post-riichi 除外)
        フラグ。
    value_eligible:
        ``(N,) bool``。value loss 対象。``eligible`` を base に、
        ``config.value_loss_includes_excluded`` が True なら shortcut
        (``ppo_exclude=True``) sample を追加で含める。``config.exclude_post_riichi_discards``
        が True の post-riichi sample は **policy / value 両方** から除外。
    sample_weight:
        ``(N,) float32``。**常に全 1.0**。per-player-round weighting は
        ``_iter_minibatches`` で minibatch-local に計算する設計に統一した
        ため、trajectory 全体スコープのこの field では重みを持たない
        (= imitation と同じ minibatch-local semantics に揃える)。
        後方互換のため field 自体は残してある。
    """

    samples: list[DecisionSample]
    returns: np.ndarray
    advantages: np.ndarray
    eligible: np.ndarray
    value_eligible: np.ndarray
    sample_weight: np.ndarray


def compute_returns_and_advantages(
    samples: list[DecisionSample],
    config: PPOConfig,
) -> PPOTrainingData:
    """GAE で returns / advantages を計算する。

    Trajectory definition
    ---------------------
    同 ``episode_id`` かつ同 ``player_id`` の sample を 1 player trajectory
    として扱い、``step_id`` 昇順で並べる。

    Boundary handling
    -----------------
    - ``terminated[t] == True``: game 終端。次状態の value で bootstrap せず
      ``not_done=0``。
    - ``round_over[t] == True``: round 終端。v1 では **round boundary でも
      bootstrap 0** (= ``not_done=0``) として扱う。理由: ``reward`` は
      round-level (round_score_delta * 1e-4) で round 内全 sample に同じ値
      が backfill されており、round 単位の credit を value head に押し付ける
      ことで cross-round の noisy bootstrap を避けるため。Stage02 でも
      round-level baseline を切るのが安定だった。

    Eligibility
    -----------
    sample が PPO 対象かどうかは:
    - ``actor_type`` が ``config.include_actor_types`` に含まれる
    - かつ discard なら ``selected_discard_tile_type >= 0``、candidate なら
      ``selected_candidate_index >= 0``
    の AND。``eligible`` 配列に bool で書き込む。
    """
    n = len(samples)
    returns = np.zeros(n, dtype=np.float32)
    advantages = np.zeros(n, dtype=np.float32)
    eligible = np.zeros(n, dtype=bool)
    value_eligible = np.zeros(n, dtype=bool)
    sample_weight = np.ones(n, dtype=np.float32)

    if n == 0:
        return PPOTrainingData(
            samples=samples,
            returns=returns,
            advantages=advantages,
            eligible=eligible,
            value_eligible=value_eligible,
            sample_weight=sample_weight,
        )

    # group by trajectory key
    traj_indices: dict[tuple[str, int], list[int]] = {}
    for i, s in enumerate(samples):
        key = (str(s.episode_id), int(s.player_id))
        traj_indices.setdefault(key, []).append(i)
    # sort each trajectory by step_id
    for idxs in traj_indices.values():
        idxs.sort(key=lambda i: int(samples[i].step_id))

    gamma = float(config.gamma)
    lam = float(config.gae_lambda)
    for idxs in traj_indices.values():
        gae = 0.0
        next_value = 0.0
        for j in reversed(range(len(idxs))):
            i = idxs[j]
            s = samples[i]
            terminated = bool(s.terminated)
            round_over = bool(s.round_over)
            not_done = 0.0 if (terminated or round_over) else 1.0
            r = float(s.reward)
            v = float(s.value)
            delta = r + gamma * next_value * not_done - v
            gae = delta + gamma * lam * not_done * gae
            advantages[i] = gae
            returns[i] = gae + v
            next_value = v

    include_set = set(str(x) for x in config.include_actor_types)
    exclude_post_riichi = bool(config.exclude_post_riichi_discards)
    include_shortcut_in_value = bool(config.value_loss_includes_excluded)
    for i, s in enumerate(samples):
        if str(s.actor_type) not in include_set:
            continue
        meta = s.metadata or {}
        is_shortcut = bool(meta.get("ppo_exclude", False))
        is_post_riichi = bool(meta.get("is_post_riichi_discard", False))
        # post-riichi 除外は policy / value 両方に効く (Stage02 と同方針)。
        if exclude_post_riichi and is_post_riichi:
            continue
        # selected_* validity
        if str(s.decision_family) == _NORMAL_DISCARD_FAMILY:
            if int(s.selected_discard_tile_type) < 0:
                continue
        else:
            if int(s.selected_candidate_index) < 0:
                continue
        if is_shortcut:
            # value loss にだけ含める opt-in (policy には絶対入れない)。
            if include_shortcut_in_value:
                value_eligible[i] = True
            continue
        eligible[i] = True
        value_eligible[i] = True

    # per-player-round weighting は **minibatch-local** に計算する
    # (= imitation と同じ semantics)。ここでは全 1.0 のまま置き、
    # 実際の重みは ``_iter_minibatches`` が sub_samples から計算する。
    # これにより PPO / imitation で flag の意味が一致する。

    return PPOTrainingData(
        samples=samples,
        returns=returns,
        advantages=advantages,
        eligible=eligible,
        value_eligible=value_eligible,
        sample_weight=sample_weight,
    )


# ---------------------------------------------------------------------------
# PPO batch wrapper
# ---------------------------------------------------------------------------


@dataclass
class PPOBatch:
    """1 minibatch 分の collate 済み batch + PPO 補助 tensor。

    Attributes
    ----------
    batch:
        ``DecisionBatch`` (= ``collate_decision_samples`` の戻り値)。
    returns:
        ``(N,)`` float32 tensor。
    advantages:
        ``(N,)`` float32 tensor。
    eligible:
        ``(N,)`` bool tensor。policy loss 対象。
    value_eligible:
        ``(N,)`` bool tensor。value loss 対象。``eligible`` の superset で、
        opt-in 時 shortcut sample を追加で含む。default ``None`` は
        ``eligible`` をそのままコピーして使う (= 旧テスト・旧呼び出し互換)。
    sample_weight:
        ``(N,)`` float32 tensor。default 全 1.0、weighting 有効時は
        per-(episode, round, player) 正規化重み。default ``None`` は
        全 1.0 を自動生成する (= 旧呼び出し互換)。
    """

    batch: DecisionBatch
    returns: torch.Tensor
    advantages: torch.Tensor
    eligible: torch.Tensor
    value_eligible: torch.Tensor | None = None
    sample_weight: torch.Tensor | None = None

    def __post_init__(self) -> None:
        # default fill: value_eligible は eligible と同じ、sample_weight は全 1.0
        if self.value_eligible is None:
            self.value_eligible = self.eligible.clone()
        if self.sample_weight is None:
            self.sample_weight = torch.ones_like(
                self.eligible, dtype=torch.float32
            )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_ppo_loss(
    model: Stage03Model,
    ppo_batch: PPOBatch,
    config: PPOConfig,
) -> tuple[torch.Tensor, PPOMetrics]:
    """1 batch 分の PPO loss と metrics を返す。

    Combined log_prob 仕様 (Stage03 specific)
    ----------------------------------------
    Stage03 の ``ModelPolicyAgent`` は normal discard logits (34) と
    candidate scores (Cmax) を **1 本のベクトル (34 + Cmax)** に concat
    した combined softmax で action を sample する。``compute_ppo_loss``
    でも同じ combined distribution から ``new_log_prob`` を計算しないと、
    同一 model rollout 直後の ratio が 1 にならない (= PPO の前提崩壊)。

    本実装は combined ベクトルを再構築する:

    - ``combined_logits = [masked_discard_logits (B, 34), masked_cand_scores (B, Cmax)]``
    - ``masked_discard_logits`` は ``model.forward(obs, discard_mask)`` が既に
      illegal idx を ``-1e9`` 化したもの。
    - ``masked_cand_scores`` は ``model.score_candidates(...)`` の出力に
      ``(1 - candidate_mask) * -1e9`` を加えた padding-masked 版。
    - 選択 index:
        - discard sample: ``selected_discard_tile_type``
        - candidate sample: ``34 + selected_candidate_index``
    - ``new_log_prob = log_softmax(combined_logits, dim=-1)[combined_idx]``
    - ratio / clip / KL / entropy も combined distribution 上で計算する。

    Per-family diagnostics (``decision_family`` subdict) は combined loss を
    sample 単位で計算した後、family ごとに集計するだけ。

    Stage03 では env の 1 つの ``WaitAct`` で normal_discard と candidate
    (TSUMO / Ankan / Riichi 等) が **同時に legal** になるため、combined
    softmax が必要。

    Loss 全体
    --------
    - policy_loss = - min(ratio * adv, clip(ratio) * adv) の eligible 平均
    - value loss: ``policy_only=False`` のとき eligible sample で
      ``0.5 * MSE(value_pred, returns)``。
    - entropy bonus: combined categorical entropy の eligible 平均。
    - terminal / yaku aux: ``policy_only=False`` のとき imitation trainer
      と同じ semantics (terminal_class >= 0 / yaku_loss_mask > 0)。

    Deterministic shortcut (TSUMO/RON/KYUSHU) は
    ``compute_returns_and_advantages`` で ``metadata["ppo_exclude"]=True``
    に基づき eligibility から除外する設計。本関数内では特別扱いしない。

    Total
    -----
    ``total = policy - entropy_coef * entropy + value_coef * value
            + terminal_coef * terminal + yaku_coef * yaku``
    """
    device = torch.device(config.device)
    n = int(ppo_batch.batch.observation.size(0))
    obs = ppo_batch.batch.observation.to(device).float()
    discard_mask = ppo_batch.batch.discard_mask.to(device).float()
    cand_feat = ppo_batch.batch.candidate_features.to(device).float()
    cand_mask = ppo_batch.batch.candidate_mask.to(device).float()
    sel_disc = ppo_batch.batch.selected_discard_tile_type.to(device).long()
    sel_cand = ppo_batch.batch.selected_candidate_index.to(device).long()
    cand_count = ppo_batch.batch.candidate_count.to(device).long()
    old_log_prob = ppo_batch.batch.old_log_prob.to(device).float()
    terminal_class = ppo_batch.batch.terminal_class.to(device).long()
    yaku_target = ppo_batch.batch.yaku_target.to(device).float()
    yaku_loss_mask = ppo_batch.batch.yaku_loss_mask.to(device).float()
    eligible = ppo_batch.eligible.to(device).bool()
    value_eligible = ppo_batch.value_eligible.to(device).bool()
    sample_weight = ppo_batch.sample_weight.to(device).float()
    advantages = ppo_batch.advantages.to(device).float()
    returns = ppo_batch.returns.to(device).float()

    # advantage normalize (eligible だけで mean/std)
    if config.advantage_normalize and bool(eligible.any().item()):
        adv_pool = advantages[eligible]
        if adv_pool.numel() > 1:
            mean = adv_pool.mean()
            std = adv_pool.std(unbiased=False).clamp_min(1e-8)
            advantages = (advantages - mean) / std

    is_normal = torch.tensor(
        [fam == _NORMAL_DISCARD_FAMILY for fam in ppo_batch.batch.decision_family],
        dtype=torch.bool,
        device=device,
    )
    is_candidate_family = ~is_normal

    # ---- Forward (both heads) ----
    fwd = model(obs, discard_mask=discard_mask)
    # ``fwd.discard_logits`` は model 側で discard_mask 適用済み (illegal は
    # -1e9 が加算されている)。
    cand_out = model.score_candidates(obs, cand_feat)
    cand_scores = cand_out.candidate_scores  # (B, Cmax)
    cmax = int(cand_scores.size(1))
    if cmax > 0:
        masked_cand = cand_scores + (1.0 - cand_mask) * _LARGE_NEG
    else:
        masked_cand = cand_scores  # (B, 0)

    # ---- Combined logits = [masked discard (B, 34), masked candidate (B, Cmax)] ----
    combined_logits = torch.cat([fwd.discard_logits, masked_cand], dim=-1)
    # (B, 34 + Cmax)
    log_softmax_combined = F.log_softmax(combined_logits, dim=-1)

    # Combined index per sample: discard sample → sel_disc, candidate sample
    # → 34 + sel_cand。Eligibility check は別途下で行うので、無効値は
    # 0 にクランプして gather index out-of-range を防ぐ。
    combined_dim = combined_logits.size(-1)
    combined_idx_disc = sel_disc.clamp(min=0)
    combined_idx_cand = (sel_cand + 34).clamp(min=0)
    combined_idx = torch.where(
        is_normal, combined_idx_disc, combined_idx_cand
    ).clamp(0, combined_dim - 1)

    # per-sample new_log_prob
    new_log_prob_all = log_softmax_combined.gather(
        -1, combined_idx.unsqueeze(-1)
    ).squeeze(-1)  # (B,)

    # ---- per-family validity ----
    discard_valid = (
        is_normal & (sel_disc >= 0) & eligible
    )
    cand_valid = (
        is_candidate_family
        & (sel_cand >= 0)
        & (sel_cand < cand_count)
        & (cand_count > 0)
        & eligible
    )
    valid_mask = discard_valid | cand_valid  # (B,) bool
    discard_count = int(discard_valid.sum().item())
    candidate_count = int(cand_valid.sum().item())

    # ---- Combined PPO loss ----
    eps = float(config.clip_epsilon)
    log_ratio_all = new_log_prob_all - old_log_prob
    ratio_all = torch.exp(log_ratio_all)
    clipped_all = torch.clamp(ratio_all, 1.0 - eps, 1.0 + eps)
    # per-sample policy loss tensor (eligibility なし; aggregation 側で mask)
    per_sample_loss = -torch.min(ratio_all * advantages, clipped_all * advantages)
    # entropy per sample (combined categorical entropy)
    combined_probs = torch.exp(log_softmax_combined)
    entropy_per_sample = -(combined_probs * log_softmax_combined).sum(dim=-1)
    # KL approx per sample
    kl_per_sample = (ratio_all - 1.0) - log_ratio_all
    # clip indicator
    clip_per_sample = (
        (ratio_all < 1.0 - eps) | (ratio_all > 1.0 + eps)
    )

    def _masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """weighted masked mean。weights が None なら旧来の equal weight。"""
        m = mask.float()
        if weights is None:
            denom = m.sum().clamp_min(1.0)
            return (values * m).sum() / denom
        w = (m * weights).clamp_min(0.0)
        denom = w.sum().clamp_min(1e-8)
        return (values * w).sum() / denom

    weights_for_loss = (
        sample_weight if bool(config.per_player_round_weighting) else None
    )
    if int(valid_mask.sum().item()) > 0:
        policy_loss_t = _masked_mean(
            per_sample_loss, valid_mask, weights=weights_for_loss
        )
    else:
        policy_loss_t = torch.zeros((), device=device, dtype=torch.float32)

    # ---- per-family decomposition (diagnostics only) ----
    discard_policy_loss_t = torch.zeros((), device=device, dtype=torch.float32)
    discard_clip_count = 0
    discard_kl_sum = torch.zeros((), device=device, dtype=torch.float32)
    discard_kl_max = 0.0
    discard_entropy_sum = torch.zeros((), device=device, dtype=torch.float32)
    if discard_count > 0:
        discard_policy_loss_t = _masked_mean(
            per_sample_loss, discard_valid, weights=weights_for_loss
        )
        with torch.no_grad():
            discard_clip_count = int(
                (clip_per_sample & discard_valid).sum().item()
            )
            discard_kl_sum = (kl_per_sample * discard_valid.float()).sum()
            kl_only_disc = kl_per_sample.masked_select(discard_valid)
            discard_kl_max = (
                float(kl_only_disc.max().item())
                if int(kl_only_disc.numel()) > 0
                else 0.0
            )
            discard_entropy_sum = (
                entropy_per_sample * discard_valid.float()
            ).sum()

    candidate_policy_loss_t = torch.zeros(
        (), device=device, dtype=torch.float32
    )
    cand_clip_count = 0
    cand_kl_sum = torch.zeros((), device=device, dtype=torch.float32)
    cand_kl_max = 0.0
    cand_entropy_sum = torch.zeros((), device=device, dtype=torch.float32)
    if candidate_count > 0:
        candidate_policy_loss_t = _masked_mean(
            per_sample_loss, cand_valid, weights=weights_for_loss
        )
        with torch.no_grad():
            cand_clip_count = int((clip_per_sample & cand_valid).sum().item())
            cand_kl_sum = (kl_per_sample * cand_valid.float()).sum()
            kl_only_cand = kl_per_sample.masked_select(cand_valid)
            cand_kl_max = (
                float(kl_only_cand.max().item())
                if int(kl_only_cand.numel()) > 0
                else 0.0
            )
            cand_entropy_sum = (
                entropy_per_sample * cand_valid.float()
            ).sum()

    # ---------- value / aux (policy_only=False のみ更新) ----------
    value_loss_t = torch.zeros((), device=device, dtype=torch.float32)
    value_count = 0
    terminal_loss_t = torch.zeros((), device=device, dtype=torch.float32)
    terminal_count = 0
    yaku_loss_t = torch.zeros((), device=device, dtype=torch.float32)
    yaku_count = 0
    if not config.policy_only:
        # value loss: value_eligible (= eligible ∪ opt-in shortcut)
        # の sample で MSE。per-player-round weighting も適用する。
        if bool(value_eligible.any().item()):
            idx_v = value_eligible.nonzero(as_tuple=False).flatten()
            v_pred = fwd.value[idx_v]
            v_target = returns[idx_v]
            v_sq = (v_pred - v_target).pow(2)
            if bool(config.per_player_round_weighting):
                w_v = sample_weight[idx_v].clamp_min(0.0)
                denom = w_v.sum().clamp_min(1e-8)
                value_loss_t = 0.5 * (v_sq * w_v).sum() / denom
            else:
                value_loss_t = 0.5 * v_sq.mean()
            value_count = int(idx_v.numel())
        # terminal aux
        term_valid = terminal_class >= 0
        if bool(term_valid.any().item()):
            idx_t = term_valid.nonzero(as_tuple=False).flatten()
            t_logits = fwd.terminal_logits[idx_t]
            t_target = terminal_class[idx_t]
            terminal_loss_t = F.cross_entropy(
                t_logits, t_target, reduction="mean"
            )
            terminal_count = int(idx_t.numel())
        # yaku aux (winner-only)
        yaku_valid = yaku_loss_mask > 0
        if bool(yaku_valid.any().item()):
            idx_y = yaku_valid.nonzero(as_tuple=False).flatten()
            y_logits = fwd.yaku_logits[idx_y]
            y_target = yaku_target[idx_y]
            yaku_loss_t = F.binary_cross_entropy_with_logits(
                y_logits, y_target, reduction="mean"
            )
            yaku_count = int(idx_y.numel())

    # ---------- entropy aggregation ----------
    entropy_total = discard_entropy_sum + cand_entropy_sum
    entropy_count = discard_count + candidate_count
    if entropy_count > 0:
        entropy_value = entropy_total / float(entropy_count)
    else:
        entropy_value = torch.zeros((), device=device, dtype=torch.float32)

    # entropy bonus は total から **引く** (= - coef * entropy)
    total_loss = (
        policy_loss_t
        - float(config.entropy_coef) * entropy_value
        + float(config.value_loss_coef) * value_loss_t
        + float(config.terminal_loss_coef) * terminal_loss_t
        + float(config.yaku_loss_coef) * yaku_loss_t
    )

    # ---------- metrics ----------
    total_pol = discard_count + candidate_count
    clip_total = discard_clip_count + cand_clip_count
    clip_fraction = (
        float(clip_total) / float(total_pol) if total_pol > 0 else 0.0
    )
    kl_sum = discard_kl_sum + cand_kl_sum
    approx_kl_mean = (
        float((kl_sum / total_pol).item()) if total_pol > 0 else 0.0
    )
    approx_kl_max = max(discard_kl_max, cand_kl_max) if total_pol > 0 else 0.0

    # PPO included / excluded counts
    ppo_included_count = total_pol
    # excluded: actor_type が include_actor_types に含まれない sample 数
    include_set = set(str(x) for x in config.include_actor_types)
    ppo_excluded_by_actor = sum(
        1 for at in ppo_batch.batch.actor_type if str(at) not in include_set
    )
    # 合計 excluded: non-eligible sample 数 (selected_* が -1 の sample 含む)
    ppo_excluded_total = n - ppo_included_count

    # actor_type counts (集計のため全 sample を数える)
    actor_type_counter: dict[str, int] = {}
    for at in ppo_batch.batch.actor_type:
        actor_type_counter[str(at)] = actor_type_counter.get(str(at), 0) + 1

    # Stage02 stabilizer diagnostics: post-riichi 除外 / shortcut value 加算
    post_riichi_excluded_count = 0
    value_loss_extra_shortcut_count = 0
    if bool(config.exclude_post_riichi_discards):
        post_riichi_excluded_count = sum(
            1
            for md in ppo_batch.batch.metadata
            if bool((md or {}).get("is_post_riichi_discard", False))
        )
    if bool(config.value_loss_includes_excluded):
        # eligible では無いが value_eligible には入っている sample 数
        extra_mask = value_eligible & (~eligible)
        value_loss_extra_shortcut_count = int(extra_mask.sum().item())

    # decision_family subdict
    family_stats: dict[str, dict[str, float]] = {}
    if discard_count > 0:
        family_stats["normal_discard"] = {
            "count": float(discard_count),
            "policy_loss": float(discard_policy_loss_t.detach().item()),
            "clip_count": float(discard_clip_count),
            "entropy_sum": float(discard_entropy_sum.detach().item()),
            "approx_kl_mean": (
                float((discard_kl_sum / discard_count).item())
                if discard_count > 0 else 0.0
            ),
        }
    if candidate_count > 0:
        family_stats["candidate"] = {
            "count": float(candidate_count),
            "policy_loss": float(candidate_policy_loss_t.detach().item()),
            "clip_count": float(cand_clip_count),
            "entropy_sum": float(cand_entropy_sum.detach().item()),
            "approx_kl_mean": (
                float((cand_kl_sum / candidate_count).item())
                if candidate_count > 0 else 0.0
            ),
        }

    metrics = PPOMetrics(
        loss=float(total_loss.detach().item()),
        policy_loss=float(policy_loss_t.detach().item()),
        value_loss=float(value_loss_t.detach().item()),
        entropy=float(entropy_value.detach().item())
        if isinstance(entropy_value, torch.Tensor) else 0.0,
        terminal_loss=float(terminal_loss_t.detach().item()),
        yaku_loss=float(yaku_loss_t.detach().item()),
        discard_policy_loss=float(discard_policy_loss_t.detach().item()),
        candidate_policy_loss=float(candidate_policy_loss_t.detach().item()),
        discard_count=discard_count,
        candidate_count=candidate_count,
        value_count=value_count,
        terminal_count=terminal_count,
        yaku_count=yaku_count,
        num_samples=n,
        num_batches=1,
        ppo_included_count=ppo_included_count,
        ppo_excluded_count=ppo_excluded_total,
        ppo_excluded_by_actor_type=ppo_excluded_by_actor,
        clip_fraction=clip_fraction,
        approx_kl_mean=approx_kl_mean,
        approx_kl_max=approx_kl_max,
        target_kl_checked_minibatches=0,
        target_kl_skipped_minibatches=0,
        target_kl_applied_minibatches=0,
        num_updates=0,
        grad_norm=0.0,
        post_riichi_excluded_count=post_riichi_excluded_count,
        value_loss_extra_shortcut_count=value_loss_extra_shortcut_count,
        decision_family=family_stats,
        actor_type_counts=actor_type_counter,
        grad_norms_per_component={},
        lr_groups_info={},
    )
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Epoch / fit
# ---------------------------------------------------------------------------


def make_default_ppo_optimizer(
    model: nn.Module, config: PPOConfig
) -> optim.Optimizer:
    """AdamW を ``PPOConfig`` 設定で作る convenience。

    ``config.lr_group_config.enabled=True`` のときは Stage02 由来の lr group
    分離を有効化する。group ごとの parameter count / lr 情報は
    ``build_lr_grouped_optimizer`` の戻り値を捨てて optimizer のみ返す。
    diagnostic として info dict が欲しい呼び出し側は
    ``build_lr_grouped_optimizer`` を直接使う。
    """
    opt, _info = build_lr_grouped_optimizer(
        model,
        base_lr=float(config.learning_rate),
        base_weight_decay=float(config.weight_decay),
        lr_group_config=config.lr_group_config,
        optimizer_cls=optim.AdamW,
    )
    return opt


def make_ppo_optimizer_with_info(
    model: nn.Module, config: PPOConfig
) -> tuple[optim.Optimizer, dict[str, Any]]:
    """``make_default_ppo_optimizer`` と同等だが、lr group diagnostics 情報も
    返す。runbook / experiment report で param group breakdown が必要なときに使う。
    """
    return build_lr_grouped_optimizer(
        model,
        base_lr=float(config.learning_rate),
        base_weight_decay=float(config.weight_decay),
        lr_group_config=config.lr_group_config,
        optimizer_cls=optim.AdamW,
    )


def _iter_minibatches(
    data: PPOTrainingData,
    batch_size: int,
    *,
    shuffle: bool,
    rng: _random.Random,
    per_player_round_weighting: bool = False,
) -> Iterable[PPOBatch]:
    """``PPOTrainingData`` を minibatch 化して ``PPOBatch`` を yield する。

    ``per_player_round_weighting=True`` のとき、各 minibatch の
    ``sub_samples`` から **minibatch-local** に per-(episode, round, player)
    weight を計算して ``PPOBatch.sample_weight`` に渡す (= imitation と同じ
    minibatch-local semantics)。``data.sample_weight`` は使わない (常に全 1.0
    のため)。
    """
    n = len(data.samples)
    if n == 0:
        return
    order = list(range(n))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, n, batch_size):
        idxs = order[start : start + batch_size]
        sub_samples = [data.samples[i] for i in idxs]
        sub_returns = data.returns[idxs]
        sub_advs = data.advantages[idxs]
        sub_elig = data.eligible[idxs]
        sub_value_elig = data.value_eligible[idxs]
        if per_player_round_weighting:
            sub_weights = compute_per_player_round_weights(
                episode_ids=[str(s.episode_id) for s in sub_samples],
                round_ids=[int(s.round_id) for s in sub_samples],
                player_ids=[int(s.player_id) for s in sub_samples],
            )
        else:
            sub_weights = np.ones(len(sub_samples), dtype=np.float32)
        batch = collate_decision_samples(sub_samples)
        yield PPOBatch(
            batch=batch,
            returns=torch.from_numpy(sub_returns.copy()),
            advantages=torch.from_numpy(sub_advs.copy()),
            eligible=torch.from_numpy(sub_elig.copy()),
            value_eligible=torch.from_numpy(sub_value_elig.copy()),
            sample_weight=torch.from_numpy(sub_weights.copy()),
        )


def _compute_component_grad_norms(
    loss_components: dict[str, torch.Tensor],
    parameters: list[nn.Parameter],
) -> dict[str, float]:
    """``torch.autograd.grad`` を使って per-component grad norm を計測する。

    速度コストを払うため、呼び出し側で batch 数を制限すること。
    """
    out: dict[str, float] = {}
    for name, comp in loss_components.items():
        if not torch.is_tensor(comp) or not comp.requires_grad:
            continue
        if float(comp.item()) == 0.0:
            out[name] = 0.0
            continue
        try:
            grads = torch.autograd.grad(
                comp,
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
        except RuntimeError:
            # gradient が一部 disconnected の場合は skip
            continue
        sq = 0.0
        for g in grads:
            if g is None:
                continue
            sq += float(g.float().pow(2).sum().item())
        out[name] = float(math.sqrt(sq))
    return out


def _compute_total_grad_norm(parameters) -> float:
    total = 0.0
    has_grad = False
    for p in parameters:
        if p.grad is None:
            continue
        has_grad = True
        total += float(p.grad.detach().float().pow(2).sum().item())
    if not has_grad:
        return 0.0
    return float(math.sqrt(total))


def train_ppo_epoch(
    model: Stage03Model,
    data_or_batches: Any,
    optimizer: optim.Optimizer,
    config: PPOConfig,
    *,
    rng: _random.Random | None = None,
) -> tuple[PPOMetrics, bool]:
    """1 epoch 分の PPO update を実行する。

    Parameters
    ----------
    model:
        ``Stage03Model``。
    data_or_batches:
        ``PPOTrainingData`` または事前構築した ``PPOBatch`` の iterable。
    optimizer:
        ``torch.optim.Optimizer``。
    config:
        ``PPOConfig``。
    rng:
        shuffle 用 ``random.Random``。

    Returns
    -------
    (epoch_metrics, early_stopped):
        epoch metrics と target_kl による early stop の flag。
    """
    if rng is None:
        rng = _random.Random(config.seed)
    model.train()
    if isinstance(data_or_batches, PPOTrainingData):
        batches_iter: Iterable[PPOBatch] = _iter_minibatches(
            data_or_batches,
            batch_size=int(config.batch_size),
            shuffle=bool(config.shuffle),
            rng=rng,
            per_player_round_weighting=bool(config.per_player_round_weighting),
        )
    else:
        batches_iter = data_or_batches  # type: ignore[assignment]

    applied_metrics: list[PPOMetrics] = []
    checked = 0
    skipped = 0
    applied = 0
    early_stopped = False
    grad_norm_batches_used = 0

    for batch in batches_iter:
        if early_stopped:
            break
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_ppo_loss(model, batch, config)
        if not loss.requires_grad or (
            metrics.discard_count + metrics.candidate_count == 0
            and (config.policy_only
                 or (metrics.value_count + metrics.terminal_count
                     + metrics.yaku_count == 0))
        ):
            # 何も backward できる branch が無い → skip (applied に混ぜない)
            continue

        # target_kl check (backward 前: previous batch までの状態とは独立)
        check_kl = config.target_kl_enabled and (
            metrics.discard_count + metrics.candidate_count > 0
        )
        if check_kl:
            checked += 1
            limit = float(config.target_kl) * float(config.target_kl_stop_multiplier)
            if metrics.approx_kl_mean > limit:
                if config.target_kl_skip_minibatch_on_exceed:
                    # skip: backward/step せず early stop
                    skipped += 1
                    early_stopped = True
                    continue
                # skip しないなら step してから early stop
                early_stopped = True

        loss.backward()
        grad_norm = _compute_total_grad_norm(model.parameters())
        if config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.max_grad_norm)
            )
        optimizer.step()

        applied += 1
        # grad_norm 補強 component (optional)
        per_comp: dict[str, float] = {}
        if (
            config.gradient_norms_enabled
            and grad_norm_batches_used < int(
                config.gradient_norms_max_batches_per_epoch
            )
        ):
            per_comp = _component_grad_norms(model, batch, config)
            grad_norm_batches_used += 1
        metrics = _with_extras(
            metrics,
            grad_norm=grad_norm,
            target_kl_checked=1 if check_kl else 0,
            target_kl_skipped=0,
            target_kl_applied=1,
            num_updates=1,
            grad_norms_per_component=per_comp,
        )
        applied_metrics.append(metrics)

    epoch = _aggregate_epoch_metrics(
        applied_metrics,
        target_kl_checked=checked,
        target_kl_skipped=skipped,
        target_kl_applied=applied,
    )
    return epoch, early_stopped


def _component_grad_norms(
    model: Stage03Model,
    batch: PPOBatch,
    config: PPOConfig,
) -> dict[str, float]:
    """gradient norm diagnostics 用に loss component 別の grad norm を計算する。

    backward から独立した再 forward + ``torch.autograd.grad`` を使う。速度
    コストが大きいので caller 側で batch 数を限定する。
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return {}
    # 再 forward して components を取り直す
    device = torch.device(config.device)
    obs = batch.batch.observation.to(device).float()
    discard_mask = batch.batch.discard_mask.to(device).float()
    cand_feat = batch.batch.candidate_features.to(device).float()
    cand_mask = batch.batch.candidate_mask.to(device).float()
    sel_disc = batch.batch.selected_discard_tile_type.to(device).long()
    sel_cand = batch.batch.selected_candidate_index.to(device).long()
    cand_count = batch.batch.candidate_count.to(device).long()
    old_log_prob = batch.batch.old_log_prob.to(device).float()
    returns = batch.returns.to(device).float()
    advantages = batch.advantages.to(device).float()
    eligible = batch.eligible.to(device).bool()
    value_eligible = batch.value_eligible.to(device).bool()
    is_normal = torch.tensor(
        [fam == _NORMAL_DISCARD_FAMILY for fam in batch.batch.decision_family],
        dtype=torch.bool,
        device=device,
    )
    is_candidate_family = ~is_normal
    discard_valid = is_normal & (sel_disc >= 0) & eligible
    cand_valid = (
        is_candidate_family
        & (sel_cand >= 0)
        & (sel_cand < cand_count)
        & (cand_count > 0)
        & eligible
    )

    if config.advantage_normalize and bool(eligible.any().item()):
        adv_pool = advantages[eligible]
        if adv_pool.numel() > 1:
            advantages = (advantages - adv_pool.mean()) / adv_pool.std(
                unbiased=False
            ).clamp_min(1e-8)

    fwd = model(obs, discard_mask=discard_mask)
    cand_out = model.score_candidates(obs, cand_feat)
    cand_scores = cand_out.candidate_scores
    cmax = int(cand_scores.size(1))
    if cmax > 0:
        masked_cand = cand_scores + (1.0 - cand_mask) * _LARGE_NEG
    else:
        masked_cand = cand_scores

    components: dict[str, torch.Tensor] = {}

    # policy (combined logits を再現)
    combined_logits = torch.cat([fwd.discard_logits, masked_cand], dim=-1)
    combined_dim = combined_logits.size(-1)
    log_softmax_combined = F.log_softmax(combined_logits, dim=-1)
    combined_idx_disc = sel_disc.clamp(min=0)
    combined_idx_cand = (sel_cand + 34).clamp(min=0)
    combined_idx = torch.where(
        is_normal, combined_idx_disc, combined_idx_cand
    ).clamp(0, combined_dim - 1)
    new_log_prob_all = log_softmax_combined.gather(
        -1, combined_idx.unsqueeze(-1)
    ).squeeze(-1)
    log_ratio_all = new_log_prob_all - old_log_prob
    ratio_all = torch.exp(log_ratio_all)
    clipped_all = torch.clamp(
        ratio_all, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
    )
    per_sample_loss = -torch.min(
        ratio_all * advantages, clipped_all * advantages
    )
    valid_mask = discard_valid | cand_valid
    if int(valid_mask.sum().item()) > 0:
        m = valid_mask.float()
        pol_loss = (per_sample_loss * m).sum() / m.sum().clamp_min(1.0)
    else:
        pol_loss = torch.zeros((), device=device, dtype=torch.float32)
    components["policy"] = pol_loss

    # value
    if not config.policy_only and value_eligible.any():
        idx_v = value_eligible.nonzero(as_tuple=False).flatten()
        v_pred = fwd.value[idx_v]
        v_tgt = returns[idx_v]
        components["value"] = 0.5 * (v_pred - v_tgt).pow(2).mean()
    # aux (terminal + yaku をまとめて 1 component)
    if not config.policy_only:
        aux_loss = torch.zeros((), device=device, dtype=torch.float32)
        terminal_class = batch.batch.terminal_class.to(device).long()
        yaku_target = batch.batch.yaku_target.to(device).float()
        yaku_loss_mask = batch.batch.yaku_loss_mask.to(device).float()
        tv = terminal_class >= 0
        if tv.any():
            idx_t = tv.nonzero(as_tuple=False).flatten()
            aux_loss = aux_loss + F.cross_entropy(
                fwd.terminal_logits[idx_t],
                terminal_class[idx_t],
                reduction="mean",
            )
        yv = yaku_loss_mask > 0
        if yv.any():
            idx_y = yv.nonzero(as_tuple=False).flatten()
            aux_loss = aux_loss + F.binary_cross_entropy_with_logits(
                fwd.yaku_logits[idx_y],
                yaku_target[idx_y],
                reduction="mean",
            )
        components["aux"] = aux_loss

    return _compute_component_grad_norms(components, params)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _with_extras(
    metrics: PPOMetrics,
    *,
    grad_norm: float,
    target_kl_checked: int,
    target_kl_skipped: int,
    target_kl_applied: int,
    num_updates: int,
    grad_norms_per_component: dict[str, float],
) -> PPOMetrics:
    return PPOMetrics(
        loss=metrics.loss,
        policy_loss=metrics.policy_loss,
        value_loss=metrics.value_loss,
        entropy=metrics.entropy,
        terminal_loss=metrics.terminal_loss,
        yaku_loss=metrics.yaku_loss,
        discard_policy_loss=metrics.discard_policy_loss,
        candidate_policy_loss=metrics.candidate_policy_loss,
        discard_count=metrics.discard_count,
        candidate_count=metrics.candidate_count,
        value_count=metrics.value_count,
        terminal_count=metrics.terminal_count,
        yaku_count=metrics.yaku_count,
        num_samples=metrics.num_samples,
        num_batches=metrics.num_batches,
        ppo_included_count=metrics.ppo_included_count,
        ppo_excluded_count=metrics.ppo_excluded_count,
        ppo_excluded_by_actor_type=metrics.ppo_excluded_by_actor_type,
        clip_fraction=metrics.clip_fraction,
        approx_kl_mean=metrics.approx_kl_mean,
        approx_kl_max=metrics.approx_kl_max,
        target_kl_checked_minibatches=int(target_kl_checked),
        target_kl_skipped_minibatches=int(target_kl_skipped),
        target_kl_applied_minibatches=int(target_kl_applied),
        num_updates=int(num_updates),
        grad_norm=float(grad_norm),
        post_riichi_excluded_count=metrics.post_riichi_excluded_count,
        value_loss_extra_shortcut_count=metrics.value_loss_extra_shortcut_count,
        decision_family=dict(metrics.decision_family),
        actor_type_counts=dict(metrics.actor_type_counts),
        grad_norms_per_component=dict(grad_norms_per_component),
        lr_groups_info=dict(metrics.lr_groups_info),
    )


def _aggregate_epoch_metrics(
    applied: list[PPOMetrics],
    *,
    target_kl_checked: int,
    target_kl_skipped: int,
    target_kl_applied: int,
) -> PPOMetrics:
    """applied minibatch metrics の sample-weighted aggregate。

    skipped minibatch は ``applied`` list に含めず、target_kl_* count だけを
    保持する。
    """
    if not applied:
        return PPOMetrics(
            target_kl_checked_minibatches=int(target_kl_checked),
            target_kl_skipped_minibatches=int(target_kl_skipped),
            target_kl_applied_minibatches=int(target_kl_applied),
            post_riichi_excluded_count=0,
            value_loss_extra_shortcut_count=0,
        )

    total_samples = sum(m.num_samples for m in applied)
    total_batches = len(applied)

    def _w_sum(field_name: str, count_name: str) -> tuple[float, int]:
        s = 0.0
        c = 0
        for m in applied:
            cnt = int(getattr(m, count_name))
            if cnt > 0:
                s += float(getattr(m, field_name)) * cnt
                c += cnt
        return s, c

    def _w_avg(field_name: str, count_name: str) -> float:
        s, c = _w_sum(field_name, count_name)
        return s / c if c > 0 else 0.0

    # weighted by ppo_included_count
    pol_loss_w, pol_count = _w_sum("policy_loss", "ppo_included_count")
    discard_loss_w, discard_count = _w_sum("discard_policy_loss",
                                            "discard_count")
    cand_loss_w, cand_count = _w_sum("candidate_policy_loss",
                                      "candidate_count")
    value_loss_w, value_count = _w_sum("value_loss", "value_count")
    terminal_loss_w, terminal_count = _w_sum("terminal_loss",
                                              "terminal_count")
    yaku_loss_w, yaku_count = _w_sum("yaku_loss", "yaku_count")
    entropy_w, ent_count = _w_sum("entropy", "ppo_included_count")
    clip_fraction = _w_avg("clip_fraction", "ppo_included_count")
    approx_kl_mean = _w_avg("approx_kl_mean", "ppo_included_count")
    approx_kl_max = max((m.approx_kl_max for m in applied), default=0.0)

    policy_loss = pol_loss_w / pol_count if pol_count > 0 else 0.0
    discard_policy_loss = (
        discard_loss_w / discard_count if discard_count > 0 else 0.0
    )
    candidate_policy_loss = (
        cand_loss_w / cand_count if cand_count > 0 else 0.0
    )
    value_loss = value_loss_w / value_count if value_count > 0 else 0.0
    terminal_loss = (
        terminal_loss_w / terminal_count if terminal_count > 0 else 0.0
    )
    yaku_loss = yaku_loss_w / yaku_count if yaku_count > 0 else 0.0
    entropy = entropy_w / ent_count if ent_count > 0 else 0.0
    total_loss_w = sum(float(m.loss) * m.num_samples for m in applied)
    total_loss = (
        total_loss_w / total_samples if total_samples > 0 else 0.0
    )

    # decision_family aggregate
    family_keys: set[str] = set()
    for m in applied:
        family_keys.update(m.decision_family.keys())
    family_agg: dict[str, dict[str, float]] = {}
    for k in family_keys:
        count_sum = 0.0
        ploss_w = 0.0
        clip_w = 0.0
        ent_w = 0.0
        kl_w = 0.0
        for m in applied:
            sub = m.decision_family.get(k)
            if sub is None:
                continue
            c = float(sub.get("count", 0.0))
            count_sum += c
            ploss_w += float(sub.get("policy_loss", 0.0)) * c
            clip_w += float(sub.get("clip_count", 0.0))
            ent_w += float(sub.get("entropy_sum", 0.0))
            kl_w += float(sub.get("approx_kl_mean", 0.0)) * c
        family_agg[k] = {
            "count": count_sum,
            "policy_loss": (ploss_w / count_sum) if count_sum > 0 else 0.0,
            "clip_count": clip_w,
            "entropy_sum": ent_w,
            "approx_kl_mean": (kl_w / count_sum) if count_sum > 0 else 0.0,
        }

    # actor_type counts aggregate
    actor_agg: dict[str, int] = {}
    for m in applied:
        for k, v in m.actor_type_counts.items():
            actor_agg[k] = actor_agg.get(k, 0) + int(v)

    # grad_norm: applied batch 平均
    grad_norms = [m.grad_norm for m in applied if m.grad_norm > 0]
    grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

    # grad_norms_per_component: 平均
    comp_keys: set[str] = set()
    for m in applied:
        comp_keys.update(m.grad_norms_per_component.keys())
    comp_agg: dict[str, float] = {}
    for k in comp_keys:
        vals = [
            float(m.grad_norms_per_component[k])
            for m in applied if k in m.grad_norms_per_component
        ]
        comp_agg[k] = sum(vals) / len(vals) if vals else 0.0

    return PPOMetrics(
        loss=float(total_loss),
        policy_loss=float(policy_loss),
        value_loss=float(value_loss),
        entropy=float(entropy),
        terminal_loss=float(terminal_loss),
        yaku_loss=float(yaku_loss),
        discard_policy_loss=float(discard_policy_loss),
        candidate_policy_loss=float(candidate_policy_loss),
        discard_count=int(discard_count),
        candidate_count=int(cand_count),
        value_count=int(value_count),
        terminal_count=int(terminal_count),
        yaku_count=int(yaku_count),
        num_samples=int(total_samples),
        num_batches=int(total_batches),
        ppo_included_count=int(pol_count),
        ppo_excluded_count=sum(int(m.ppo_excluded_count) for m in applied),
        ppo_excluded_by_actor_type=sum(
            int(m.ppo_excluded_by_actor_type) for m in applied
        ),
        clip_fraction=float(clip_fraction),
        approx_kl_mean=float(approx_kl_mean),
        approx_kl_max=float(approx_kl_max),
        target_kl_checked_minibatches=int(target_kl_checked),
        target_kl_skipped_minibatches=int(target_kl_skipped),
        target_kl_applied_minibatches=int(target_kl_applied),
        num_updates=sum(int(m.num_updates) for m in applied),
        grad_norm=float(grad_norm),
        post_riichi_excluded_count=sum(
            int(m.post_riichi_excluded_count) for m in applied
        ),
        value_loss_extra_shortcut_count=sum(
            int(m.value_loss_extra_shortcut_count) for m in applied
        ),
        decision_family=family_agg,
        actor_type_counts=actor_agg,
        grad_norms_per_component=comp_agg,
        lr_groups_info=(
            dict(applied[0].lr_groups_info) if applied else {}
        ),
    )


def fit_ppo(
    model: Stage03Model,
    samples: list[DecisionSample],
    config: PPOConfig,
    *,
    optimizer: optim.Optimizer | None = None,
) -> PPORunResult:
    """``samples`` 上で ``config.num_epochs`` epoch 回す convenience。"""
    if not samples:
        raise ValueError("fit_ppo: empty samples")
    lr_groups_info: dict[str, Any] = {}
    if optimizer is None:
        optimizer, lr_groups_info = make_ppo_optimizer_with_info(model, config)
    data = compute_returns_and_advantages(samples, config)
    rng = _random.Random(config.seed)
    result = PPORunResult()
    for _ in range(int(config.num_epochs)):
        epoch_metrics, early_stopped = train_ppo_epoch(
            model, data, optimizer, config, rng=rng
        )
        # epoch metrics に lr_groups_info を attach (optimizer ごとに固定値)。
        if lr_groups_info:
            epoch_metrics = _attach_lr_groups_info(
                epoch_metrics, lr_groups_info
            )
        result.epochs.append(epoch_metrics)
        if early_stopped:
            result.early_stopped = True
            break
    result.final = result.epochs[-1] if result.epochs else None
    return result


def _attach_lr_groups_info(
    metrics: PPOMetrics, info: dict[str, Any]
) -> PPOMetrics:
    return _with_extras(
        PPOMetrics(
            loss=metrics.loss,
            policy_loss=metrics.policy_loss,
            value_loss=metrics.value_loss,
            entropy=metrics.entropy,
            terminal_loss=metrics.terminal_loss,
            yaku_loss=metrics.yaku_loss,
            discard_policy_loss=metrics.discard_policy_loss,
            candidate_policy_loss=metrics.candidate_policy_loss,
            discard_count=metrics.discard_count,
            candidate_count=metrics.candidate_count,
            value_count=metrics.value_count,
            terminal_count=metrics.terminal_count,
            yaku_count=metrics.yaku_count,
            num_samples=metrics.num_samples,
            num_batches=metrics.num_batches,
            ppo_included_count=metrics.ppo_included_count,
            ppo_excluded_count=metrics.ppo_excluded_count,
            ppo_excluded_by_actor_type=metrics.ppo_excluded_by_actor_type,
            clip_fraction=metrics.clip_fraction,
            approx_kl_mean=metrics.approx_kl_mean,
            approx_kl_max=metrics.approx_kl_max,
            target_kl_checked_minibatches=metrics.target_kl_checked_minibatches,
            target_kl_skipped_minibatches=metrics.target_kl_skipped_minibatches,
            target_kl_applied_minibatches=metrics.target_kl_applied_minibatches,
            num_updates=metrics.num_updates,
            grad_norm=metrics.grad_norm,
            post_riichi_excluded_count=metrics.post_riichi_excluded_count,
            value_loss_extra_shortcut_count=metrics.value_loss_extra_shortcut_count,
            decision_family=dict(metrics.decision_family),
            actor_type_counts=dict(metrics.actor_type_counts),
            grad_norms_per_component=dict(metrics.grad_norms_per_component),
            lr_groups_info=dict(info),
        ),
        grad_norm=metrics.grad_norm,
        target_kl_checked=metrics.target_kl_checked_minibatches,
        target_kl_skipped=metrics.target_kl_skipped_minibatches,
        target_kl_applied=metrics.target_kl_applied_minibatches,
        num_updates=metrics.num_updates,
        grad_norms_per_component=dict(metrics.grad_norms_per_component),
    )


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def ppo_metrics_to_json(metrics: PPOMetrics) -> str:
    return json.dumps(metrics.to_dict())


__all__ = [
    "LRGroupConfig",
    "PPOConfig",
    "PPOMetrics",
    "PPORunResult",
    "PPOTrainingData",
    "PPOBatch",
    "compute_returns_and_advantages",
    "compute_ppo_loss",
    "train_ppo_epoch",
    "fit_ppo",
    "make_default_ppo_optimizer",
    "make_ppo_optimizer_with_info",
    "ppo_metrics_to_json",
]
