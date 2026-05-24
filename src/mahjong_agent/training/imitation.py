"""Imitation warm-start trainer (v1).

``DecisionSample`` / ``DecisionBatch`` と ``Stage03Model`` を使い、教師あり
(imitation) で policy + auxiliary heads を学習する最小 trainer を提供する。

設計メモ:
- PPO / advantage / GAE / target_kl / entropy regularization にはまだ踏み込まない。
- 入力は ``DecisionSample`` の list / iterable または事前に collate された
  ``DecisionBatch``。``collate_decision_samples`` で minibatch 化する。
- loss は 4 つに分岐: discard / candidate / terminal / yaku。
  詳細は ``compute_imitation_loss`` の docstring 参照。
- ``RiichiEnv`` / encoder / agent には触らない。``DecisionSample.observation``
  は public-only encoder の出力なので、trainer は env を import しない。
"""
from __future__ import annotations

import json
import math
import random as _random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

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

# ---------------------------------------------------------------------------
# Config / metrics dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImitationConfig:
    """Imitation warm-start trainer の hyperparameters。

    Attributes
    ----------
    learning_rate:
        AdamW の lr。
    batch_size:
        sample list を minibatch 化するときの batch size (collate 後)。
    num_epochs:
        epoch 数 (``fit_imitation`` 用)。
    terminal_loss_coef:
        ``terminal_loss`` の係数。Stage02 の知見に合わせ小さめ default。
    yaku_loss_coef:
        ``yaku_loss`` の係数。winner-only なので contribution が薄い前提。
    weight_decay:
        AdamW weight decay。
    max_grad_norm:
        ``None`` (default) で gradient clipping 無し。値を渡すと
        ``torch.nn.utils.clip_grad_norm_`` を適用する。
    device:
        ``"cpu"`` / ``"cuda"`` / ``"cuda:0"`` 等。default ``"cpu"``。
    shuffle:
        epoch 開始時に sample 順を shuffle する。
    seed:
        ``shuffle`` 用 PRNG seed。``None`` で system entropy。

    Note
    ----
    Stage02 で reward scale / auxiliary coef を雑に大きくすると policy が
    歪む知見があったため、auxiliary coef は default 0.2 / 0.1 と保守的に
    抑えてある。学習対象 sample 数によって調整する。
    """

    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 1
    terminal_loss_coef: float = 0.2
    yaku_loss_coef: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float | None = None
    device: str = "cpu"
    shuffle: bool = True
    seed: int | None = None
    # Stage02 由来の stabilizer (opt-in)。default off で旧挙動互換。
    lr_group_config: LRGroupConfig | None = None
    """policy / value_semantic / trunk lr 分離。``None`` で single group。"""
    exclude_post_riichi_discards: bool = False
    """``DecisionSample.metadata["is_post_riichi_discard"]==True`` の discard
    sample を loss / accuracy 集計から除外する (Stage02 CQ-0164 移植)。
    default off。candidate / terminal / yaku branch には影響しない。"""
    per_player_round_weighting: bool = False
    """同一 ``(episode_id, round_id, player_id)`` の sample 重み合計が 1.0
    になるよう normalize する。discard / candidate / terminal / yaku の全
    branch loss に乗算で効く。default off。

    weight は **minibatch-local** (= collate された 1 batch 内の sample で
    正規化) に計算する。PPO 側も同じ minibatch-local semantics に揃えてある。"""
    # tie-aware imitation
    tie_aware_discard: bool = False
    """discard loss の target を ``teacher_best_mask`` (multi-hot) で扱う。

    True のとき:
      - sample.teacher_best_mask が非ゼロの discard sample は
        ``-log(sum_{i in mask} softmax(logits)[i])`` で soft target CE。
      - mask が全 0 の sample は hard top1 CE (= ``selected_discard_tile_type``
        を target) に fallback。
    False (default) のとき:
      - 全 discard sample で hard top1 CE。Stage03 v1 互換。
    """


@dataclass(frozen=True)
class ImitationMetrics:
    """1 batch あるいは 1 epoch の集計 metrics。

    Attributes
    ----------
    loss / policy_loss / discard_loss / candidate_loss / terminal_loss /
    yaku_loss:
        scalar float。対象 sample が 0 の branch は 0.0。
    discard_count / candidate_count / terminal_count / yaku_count:
        各 branch の有効 sample 数。
    num_samples:
        batch 全体の sample 数 (N)。
    num_batches:
        epoch 集計のときに minibatch 数を保持する (single batch のときは 1)。
    accuracy_discard / accuracy_candidate / accuracy_terminal:
        各 branch の hit ratio (有効 sample 0 のときは 0.0)。
    accuracy_yaku_micro:
        yaku BCE 後の (sigmoid > 0.5) と target の micro accuracy。
        winner-only なので winner sample 数 * NUM_YAKU が denominator。
    grad_norm:
        backward 後の gradient L2 norm。training 経路でしか出ない。
    """

    loss: float = 0.0
    policy_loss: float = 0.0
    discard_loss: float = 0.0
    candidate_loss: float = 0.0
    terminal_loss: float = 0.0
    yaku_loss: float = 0.0
    discard_count: int = 0
    candidate_count: int = 0
    terminal_count: int = 0
    yaku_count: int = 0
    num_samples: int = 0
    num_batches: int = 0
    accuracy_discard: float = 0.0
    accuracy_candidate: float = 0.0
    accuracy_terminal: float = 0.0
    accuracy_yaku_micro: float = 0.0
    grad_norm: float = 0.0
    post_riichi_excluded_count: int = 0
    """``exclude_post_riichi_discards`` で discard branch から除外した sample 数。
    flag off では常に 0。"""

    def to_dict(self) -> dict[str, Any]:
        """JSON serializable dict を返す。"""
        return {
            "loss": float(self.loss),
            "policy_loss": float(self.policy_loss),
            "discard_loss": float(self.discard_loss),
            "candidate_loss": float(self.candidate_loss),
            "terminal_loss": float(self.terminal_loss),
            "yaku_loss": float(self.yaku_loss),
            "discard_count": int(self.discard_count),
            "candidate_count": int(self.candidate_count),
            "terminal_count": int(self.terminal_count),
            "yaku_count": int(self.yaku_count),
            "num_samples": int(self.num_samples),
            "num_batches": int(self.num_batches),
            "accuracy_discard": float(self.accuracy_discard),
            "accuracy_candidate": float(self.accuracy_candidate),
            "accuracy_terminal": float(self.accuracy_terminal),
            "accuracy_yaku_micro": float(self.accuracy_yaku_micro),
            "grad_norm": float(self.grad_norm),
            "post_riichi_excluded_count": int(self.post_riichi_excluded_count),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_NORMAL_DISCARD_FAMILY: str = ActionFamily.NORMAL_DISCARD.value
_LARGE_NEG: float = -1.0e9


def _zero_loss(device: torch.device) -> torch.Tensor:
    """0.0 scalar tensor (requires_grad=False)。empty branch 用。"""
    return torch.zeros((), dtype=torch.float32, device=device)


def _weighted_mean(
    per_sample_loss: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """``per_sample_loss`` を ``weights`` で重み付き平均する。

    両者は同 length 1D tensor。``weights.sum() == 0`` の極端ケースでも
    backward に支障が無いよう小さな ``clamp_min`` で割る。
    """
    w = weights.clamp_min(0.0)
    denom = w.sum().clamp_min(1e-8)
    return (per_sample_loss * w).sum() / denom


def _move_tensor(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t.device == device:
        return t
    return t.to(device)


# ---------------------------------------------------------------------------
# Core loss
# ---------------------------------------------------------------------------


def compute_imitation_loss(
    model: Stage03Model,
    batch: DecisionBatch,
    config: ImitationConfig,
) -> tuple[torch.Tensor, ImitationMetrics]:
    """1 batch 分の imitation loss と metrics を返す。

    Loss 仕様
    ---------
    - **discard branch**: ``decision_family == "normal_discard"`` かつ
      ``selected_discard_tile_type >= 0`` の sample のみ対象。
      ``model.forward(obs, discard_mask)`` の ``discard_logits`` に対して
      CE loss。``discard_mask`` (=legal tile_type 1.0) は model 側で
      illegal index に大きな負値を加える。
    - **candidate branch**: ``decision_family != "normal_discard"`` かつ
      ``selected_candidate_index >= 0`` かつ ``candidate_count > 0`` の
      sample が対象。``model.score_candidates(obs, candidate_features)`` の
      ``(B, Cmax)`` scores に対し、padding position (``candidate_mask=0``)
      に ``-1e9`` を加えて mask し、CE loss。``C=0`` または invalid index は
      除外する (fail-fast ではなく silent 除外、ただし全 sample 無効なら
      ``fit_imitation`` 側で fail-fast)。
    - **terminal aux**: ``terminal_class >= 0`` の sample のみ対象。
      ``terminal_logits`` と target で CE loss。
    - **yaku aux**: ``yaku_loss_mask > 0`` の sample のみ対象 (winner-only)。
      ``yaku_logits`` と ``yaku_target`` (multi-hot) で BCEWithLogits。
    - **total**:
      ``policy_loss = discard_loss + candidate_loss``
      ``total_loss = policy_loss + terminal_coef * terminal_loss + yaku_coef * yaku_loss``

    Empty branch
    ------------
    対象 sample 0 の branch は loss=0.0 / count=0 として **crash しない**。
    backward 経路でも勾配は 0 (no-op) になるよう、``loss`` を 0-tensor で
    返す。``fit_imitation`` 側で全 branch 0 の epoch を検出して fail-fast
    する。

    Parameters
    ----------
    model:
        ``Stage03Model``。``forward`` と ``score_candidates`` を持つ。
    batch:
        ``DecisionBatch`` (collate 済み)。
    config:
        ``ImitationConfig``。``device`` / ``terminal_loss_coef`` /
        ``yaku_loss_coef`` を読む。

    Returns
    -------
    (loss_tensor, metrics):
        ``loss_tensor`` は scalar tensor (requires_grad は内部 head の
        grad に従う)。``metrics`` は ``ImitationMetrics`` (grad_norm は
        この段では 0; ``train_imitation_epoch`` 側で埋める)。
    """
    device = torch.device(config.device)
    n = int(batch.observation.size(0))
    obs = _move_tensor(batch.observation, device).float()
    discard_mask = _move_tensor(batch.discard_mask, device).float()
    cand_feat = _move_tensor(batch.candidate_features, device).float()
    cand_mask = _move_tensor(batch.candidate_mask, device).float()
    sel_disc = _move_tensor(batch.selected_discard_tile_type, device).long()
    sel_cand = _move_tensor(batch.selected_candidate_index, device).long()
    terminal_class = _move_tensor(batch.terminal_class, device).long()
    yaku_target = _move_tensor(batch.yaku_target, device).float()
    yaku_loss_mask = _move_tensor(batch.yaku_loss_mask, device).float()
    teacher_best_mask = _move_tensor(batch.teacher_best_mask, device).float()
    # Stage02 stabilizer: post-riichi sample 検出 (discard branch のみ除外)。
    is_post_riichi = torch.tensor(
        [
            bool((md or {}).get("is_post_riichi_discard", False))
            for md in batch.metadata
        ],
        dtype=torch.bool,
        device=device,
    )
    # per-(episode, round, player) weighting (default 全 1.0)。
    # minibatch-local: collate 済みの 1 batch 内 sample でのみ正規化する
    # (PPO 側も _iter_minibatches で同じ minibatch-local 計算に統一)。
    if bool(config.per_player_round_weighting):
        weight_np = compute_per_player_round_weights(
            episode_ids=batch.episode_id,
            round_ids=[int(x) for x in batch.round_id.tolist()],
            player_ids=[int(x) for x in batch.player_id.tolist()],
        )
        sample_weight = torch.from_numpy(weight_np).to(device)
    else:
        sample_weight = torch.ones(n, dtype=torch.float32, device=device)

    # forward (4 heads)
    fwd = model(obs, discard_mask=discard_mask)
    # candidate scorer
    cand_out = model.score_candidates(obs, cand_feat)
    cand_scores = cand_out.candidate_scores  # (B, Cmax)

    # ---- discard branch ----
    is_normal_discard = torch.tensor(
        [fam == _NORMAL_DISCARD_FAMILY for fam in batch.decision_family],
        dtype=torch.bool,
        device=device,
    )
    discard_mask_valid = is_normal_discard & (sel_disc >= 0)
    # Stage02 stabilizer: post-riichi 強制 tsumogiri を除外 (opt-in)。
    post_riichi_excluded_count = 0
    if bool(config.exclude_post_riichi_discards):
        excluded = discard_mask_valid & is_post_riichi
        post_riichi_excluded_count = int(excluded.sum().item())
        discard_mask_valid = discard_mask_valid & (~is_post_riichi)
    discard_valid_idx = discard_mask_valid.nonzero(as_tuple=False).flatten()
    discard_count = int(discard_valid_idx.numel())
    discard_correct = 0
    use_weighting = bool(config.per_player_round_weighting)
    if discard_count > 0:
        d_logits = fwd.discard_logits[discard_valid_idx]
        d_targets = sel_disc[discard_valid_idx]
        d_weights = sample_weight[discard_valid_idx]
        if config.tie_aware_discard:
            d_masks = teacher_best_mask[discard_valid_idx]  # (M, 34)
            has_best = d_masks.sum(dim=-1) > 0  # (M,) bool
            # mask の中で hot な index を target に使うため、softmax 後の
            # 確率を合計して -log を取る。
            log_softmax = F.log_softmax(d_logits, dim=-1)  # (M, 34)
            losses = torch.zeros(
                d_logits.size(0), device=d_logits.device, dtype=d_logits.dtype
            )
            # mask あり: tie-aware soft target
            if bool(has_best.any().item()):
                # log_sum_exp over hot indices = log(sum p_i)
                #   = log( sum exp(log_softmax) where mask=1 )
                # mask=0 の位置を -inf に倒して logsumexp する。
                logp = log_softmax[has_best]
                m = d_masks[has_best]
                # 0 の位置を -inf 相当に
                neg_inf = torch.full_like(logp, -1.0e9)
                masked_logp = torch.where(m > 0, logp, neg_inf)
                tie_loss = -torch.logsumexp(masked_logp, dim=-1)
                losses[has_best] = tie_loss
            # mask 無し: hard top1 fallback
            if bool((~has_best).any().item()):
                idx_hard = (~has_best).nonzero(as_tuple=False).flatten()
                tgts_hard = d_targets[idx_hard]
                hard_loss = F.cross_entropy(
                    d_logits[idx_hard], tgts_hard, reduction="none"
                )
                losses[idx_hard] = hard_loss
            if use_weighting:
                discard_loss = _weighted_mean(losses, d_weights)
            else:
                discard_loss = losses.mean()
            with torch.no_grad():
                # tie-aware accuracy: best_mask を持つ sample では
                # argmax が best_set 内のどれかに当たれば correct、
                # mask 無し sample は hard top1 一致で correct (= loss semantics と整合)。
                argmax_idx = d_logits.argmax(dim=-1)  # (M,)
                correct = torch.zeros(
                    argmax_idx.size(0),
                    device=argmax_idx.device,
                    dtype=torch.bool,
                )
                if bool(has_best.any().item()):
                    # gather mask 値: d_masks[i, argmax_idx[i]] が > 0 なら correct
                    chosen_mask_val = d_masks.gather(
                        1, argmax_idx.unsqueeze(1)
                    ).squeeze(1)  # (M,)
                    correct |= (has_best & (chosen_mask_val > 0))
                if bool((~has_best).any().item()):
                    correct |= ((~has_best) & (argmax_idx == d_targets))
                discard_correct = int(correct.sum().item())
        else:
            per_sample = F.cross_entropy(d_logits, d_targets, reduction="none")
            if use_weighting:
                discard_loss = _weighted_mean(per_sample, d_weights)
            else:
                discard_loss = per_sample.mean()
            with torch.no_grad():
                discard_correct = int(
                    (d_logits.argmax(dim=-1) == d_targets).sum().item()
                )
    else:
        discard_loss = _zero_loss(device)

    # ---- candidate branch ----
    cmax = int(cand_scores.size(1))
    candidate_count = 0
    candidate_correct = 0
    if cmax > 0:
        is_candidate_family = ~is_normal_discard
        # selected_candidate_index が範囲内 + candidate が 1 つ以上ある sample
        cand_count_per_sample = _move_tensor(batch.candidate_count, device).long()
        cand_valid_mask = (
            is_candidate_family
            & (sel_cand >= 0)
            & (sel_cand < cand_count_per_sample)
            & (cand_count_per_sample > 0)
        )
        candidate_valid_idx = cand_valid_mask.nonzero(as_tuple=False).flatten()
        candidate_count = int(candidate_valid_idx.numel())
        if candidate_count > 0:
            scores = cand_scores[candidate_valid_idx]  # (M, Cmax)
            cmask = cand_mask[candidate_valid_idx]     # (M, Cmax)
            masked = scores + (1.0 - cmask) * _LARGE_NEG
            tgt = sel_cand[candidate_valid_idx]
            per_sample_cand = F.cross_entropy(masked, tgt, reduction="none")
            if use_weighting:
                cand_weights = sample_weight[candidate_valid_idx]
                candidate_loss = _weighted_mean(per_sample_cand, cand_weights)
            else:
                candidate_loss = per_sample_cand.mean()
            with torch.no_grad():
                candidate_correct = int(
                    (masked.argmax(dim=-1) == tgt).sum().item()
                )
        else:
            candidate_loss = _zero_loss(device)
    else:
        candidate_loss = _zero_loss(device)

    # ---- terminal aux ----
    terminal_valid_idx = (terminal_class >= 0).nonzero(as_tuple=False).flatten()
    terminal_count = int(terminal_valid_idx.numel())
    terminal_correct = 0
    if terminal_count > 0:
        t_logits = fwd.terminal_logits[terminal_valid_idx]
        t_target = terminal_class[terminal_valid_idx]
        per_sample_term = F.cross_entropy(t_logits, t_target, reduction="none")
        if use_weighting:
            t_weights = sample_weight[terminal_valid_idx]
            terminal_loss = _weighted_mean(per_sample_term, t_weights)
        else:
            terminal_loss = per_sample_term.mean()
        with torch.no_grad():
            terminal_correct = int(
                (t_logits.argmax(dim=-1) == t_target).sum().item()
            )
    else:
        terminal_loss = _zero_loss(device)

    # ---- yaku aux (winner-only) ----
    yaku_valid_idx = (yaku_loss_mask > 0).nonzero(as_tuple=False).flatten()
    yaku_count = int(yaku_valid_idx.numel())
    yaku_micro_correct = 0
    yaku_micro_total = 0
    if yaku_count > 0:
        y_logits = fwd.yaku_logits[yaku_valid_idx]
        y_target = yaku_target[yaku_valid_idx]
        per_sample_yaku = F.binary_cross_entropy_with_logits(
            y_logits, y_target, reduction="none"
        ).mean(dim=-1)  # per-sample average over yaku dim
        if use_weighting:
            y_weights = sample_weight[yaku_valid_idx]
            yaku_loss = _weighted_mean(per_sample_yaku, y_weights)
        else:
            yaku_loss = per_sample_yaku.mean()
        with torch.no_grad():
            pred = (torch.sigmoid(y_logits) > 0.5).float()
            yaku_micro_correct = int((pred == y_target).sum().item())
            yaku_micro_total = int(y_target.numel())
    else:
        yaku_loss = _zero_loss(device)

    # ---- combine ----
    policy_loss = discard_loss + candidate_loss
    total_loss = (
        policy_loss
        + float(config.terminal_loss_coef) * terminal_loss
        + float(config.yaku_loss_coef) * yaku_loss
    )

    metrics = ImitationMetrics(
        loss=float(total_loss.detach().item()),
        policy_loss=float(policy_loss.detach().item()),
        discard_loss=float(discard_loss.detach().item()),
        candidate_loss=float(candidate_loss.detach().item()),
        terminal_loss=float(terminal_loss.detach().item()),
        yaku_loss=float(yaku_loss.detach().item()),
        discard_count=discard_count,
        candidate_count=candidate_count,
        terminal_count=terminal_count,
        yaku_count=yaku_count,
        num_samples=n,
        num_batches=1,
        accuracy_discard=(
            float(discard_correct) / discard_count if discard_count > 0 else 0.0
        ),
        accuracy_candidate=(
            float(candidate_correct) / candidate_count
            if candidate_count > 0
            else 0.0
        ),
        accuracy_terminal=(
            float(terminal_correct) / terminal_count
            if terminal_count > 0
            else 0.0
        ),
        accuracy_yaku_micro=(
            float(yaku_micro_correct) / yaku_micro_total
            if yaku_micro_total > 0
            else 0.0
        ),
        grad_norm=0.0,
        post_riichi_excluded_count=post_riichi_excluded_count,
    )
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Epoch / fit helpers
# ---------------------------------------------------------------------------


def make_default_optimizer(
    model: nn.Module, config: ImitationConfig
) -> optim.Optimizer:
    """AdamW を ``ImitationConfig`` 設定で作る convenience。

    ``config.lr_group_config.enabled=True`` のときは Stage02 由来の lr group
    分離を適用する (= policy / value_semantic / trunk / default の 4 group)。
    """
    opt, _info = build_lr_grouped_optimizer(
        model,
        base_lr=float(config.learning_rate),
        base_weight_decay=float(config.weight_decay),
        lr_group_config=config.lr_group_config,
        optimizer_cls=optim.AdamW,
    )
    return opt


def make_imitation_optimizer_with_info(
    model: nn.Module, config: ImitationConfig
) -> tuple[optim.Optimizer, dict[str, Any]]:
    """``make_default_optimizer`` と同等だが、lr group diagnostics 情報も返す。"""
    return build_lr_grouped_optimizer(
        model,
        base_lr=float(config.learning_rate),
        base_weight_decay=float(config.weight_decay),
        lr_group_config=config.lr_group_config,
        optimizer_cls=optim.AdamW,
    )


def _iter_minibatches(
    samples: list[DecisionSample],
    batch_size: int,
    *,
    shuffle: bool,
    rng: _random.Random,
) -> Iterable[DecisionBatch]:
    """sample list を minibatch 化して yield する。"""
    if not samples:
        return
    order = list(range(len(samples)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        yield collate_decision_samples([samples[i] for i in idxs])


def _aggregate_epoch_metrics(
    batch_metrics: list[ImitationMetrics],
) -> ImitationMetrics:
    """batch metrics list を sample-weighted な epoch metrics に集計する。

    accuracy は per-branch sample 数で加重、loss は per-branch sample 数で
    加重する (count=0 の branch は skip)。``grad_norm`` は batch 平均。
    """
    if not batch_metrics:
        return ImitationMetrics()
    total_samples = sum(m.num_samples for m in batch_metrics)
    total_batches = len(batch_metrics)

    def _w_sum(field_name: str, count_name: str) -> tuple[float, int]:
        s = 0.0
        c = 0
        for m in batch_metrics:
            cnt = int(getattr(m, count_name))
            if cnt > 0:
                s += float(getattr(m, field_name)) * cnt
                c += cnt
        return s, c

    def _w_avg(field_name: str, count_name: str) -> float:
        s, c = _w_sum(field_name, count_name)
        return s / c if c > 0 else 0.0

    # discard
    d_loss, d_count = _w_sum("discard_loss", "discard_count")
    c_loss, c_count = _w_sum("candidate_loss", "candidate_count")
    t_loss, t_count = _w_sum("terminal_loss", "terminal_count")
    y_loss, y_count = _w_sum("yaku_loss", "yaku_count")

    discard_loss = d_loss / d_count if d_count > 0 else 0.0
    candidate_loss = c_loss / c_count if c_count > 0 else 0.0
    terminal_loss = t_loss / t_count if t_count > 0 else 0.0
    yaku_loss = y_loss / y_count if y_count > 0 else 0.0
    policy_loss = discard_loss + candidate_loss

    # accuracy は epoch 全体での hit-rate (loss と同じ branch sample 数で加重)
    accuracy_discard = _w_avg("accuracy_discard", "discard_count")
    accuracy_candidate = _w_avg("accuracy_candidate", "candidate_count")
    accuracy_terminal = _w_avg("accuracy_terminal", "terminal_count")
    accuracy_yaku_micro = _w_avg("accuracy_yaku_micro", "yaku_count")

    # total loss: branch loss の reweighted sum (config coef は metric には
    # 直接持たず、loss 値そのもの = batch ごとの total_loss を sample 平均)
    total_loss_weighted = 0.0
    total_weight = 0
    for m in batch_metrics:
        if m.num_samples > 0:
            total_loss_weighted += float(m.loss) * m.num_samples
            total_weight += m.num_samples
    total_loss_avg = (
        total_loss_weighted / total_weight if total_weight > 0 else 0.0
    )

    # grad_norm: batch 平均 (training 経路のみ)
    grad_norms = [float(m.grad_norm) for m in batch_metrics if m.grad_norm > 0]
    grad_norm_avg = (
        sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    )

    return ImitationMetrics(
        loss=float(total_loss_avg),
        policy_loss=float(policy_loss),
        discard_loss=float(discard_loss),
        candidate_loss=float(candidate_loss),
        terminal_loss=float(terminal_loss),
        yaku_loss=float(yaku_loss),
        discard_count=int(d_count),
        candidate_count=int(c_count),
        terminal_count=int(t_count),
        yaku_count=int(y_count),
        num_samples=int(total_samples),
        num_batches=int(total_batches),
        accuracy_discard=float(accuracy_discard),
        accuracy_candidate=float(accuracy_candidate),
        accuracy_terminal=float(accuracy_terminal),
        accuracy_yaku_micro=float(accuracy_yaku_micro),
        grad_norm=float(grad_norm_avg),
        post_riichi_excluded_count=sum(
            int(m.post_riichi_excluded_count) for m in batch_metrics
        ),
    )


def train_imitation_epoch(
    model: Stage03Model,
    samples_or_batches: Iterable[Any],
    optimizer: optim.Optimizer,
    config: ImitationConfig,
    *,
    rng: _random.Random | None = None,
) -> ImitationMetrics:
    """1 epoch 分の training を実行する。

    Parameters
    ----------
    model:
        ``Stage03Model`` (``model.train()`` mode で training する想定)。
    samples_or_batches:
        ``DecisionSample`` の list / iterable、または事前に collate された
        ``DecisionBatch`` の iterable。``list[DecisionSample]`` を渡した
        場合は内部で ``collate_decision_samples`` を呼ぶ。
    optimizer:
        既に作られた ``torch.optim.Optimizer``。``make_default_optimizer``
        で作るのが推奨。
    config:
        ``ImitationConfig``。
    rng:
        shuffle 用 ``random.Random``。``None`` なら ``config.seed`` から作る。

    Returns
    -------
    ImitationMetrics:
        epoch 集計 metrics (sample-weighted loss / accuracy / grad_norm)。
    """
    if rng is None:
        rng = _random.Random(config.seed)
    model.train()
    batch_metrics: list[ImitationMetrics] = []

    batches_iter: Iterable[DecisionBatch]
    if isinstance(samples_or_batches, list) and (
        not samples_or_batches
        or isinstance(samples_or_batches[0], DecisionSample)
    ):
        batches_iter = _iter_minibatches(
            list(samples_or_batches),
            batch_size=int(config.batch_size),
            shuffle=bool(config.shuffle),
            rng=rng,
        )
    else:
        # already collated batches (iterable)
        batches_iter = samples_or_batches  # type: ignore[assignment]

    for batch in batches_iter:
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_imitation_loss(model, batch, config)
        if not torch.is_tensor(loss):
            continue
        if loss.requires_grad and (
            metrics.discard_count + metrics.candidate_count
            + metrics.terminal_count + metrics.yaku_count
            > 0
        ):
            loss.backward()
            grad_norm = _compute_grad_norm(model.parameters())
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config.max_grad_norm)
                )
            optimizer.step()
        else:
            grad_norm = 0.0
        # grad_norm を metrics に埋め直す (immutable dataclass のため
        # field 上書きは to_dict 経由でやる代わりに、新規 instance を作る)
        metrics = _with_grad_norm(metrics, grad_norm)
        batch_metrics.append(metrics)
    if not batch_metrics:
        raise ValueError(
            "train_imitation_epoch: no batches were processed (empty input)"
        )
    return _aggregate_epoch_metrics(batch_metrics)


def _with_grad_norm(
    metrics: ImitationMetrics, grad_norm: float
) -> ImitationMetrics:
    return ImitationMetrics(
        loss=metrics.loss,
        policy_loss=metrics.policy_loss,
        discard_loss=metrics.discard_loss,
        candidate_loss=metrics.candidate_loss,
        terminal_loss=metrics.terminal_loss,
        yaku_loss=metrics.yaku_loss,
        discard_count=metrics.discard_count,
        candidate_count=metrics.candidate_count,
        terminal_count=metrics.terminal_count,
        yaku_count=metrics.yaku_count,
        num_samples=metrics.num_samples,
        num_batches=metrics.num_batches,
        accuracy_discard=metrics.accuracy_discard,
        accuracy_candidate=metrics.accuracy_candidate,
        accuracy_terminal=metrics.accuracy_terminal,
        accuracy_yaku_micro=metrics.accuracy_yaku_micro,
        grad_norm=float(grad_norm),
        post_riichi_excluded_count=metrics.post_riichi_excluded_count,
    )


def _compute_grad_norm(parameters) -> float:
    total = 0.0
    has_grad = False
    for p in parameters:
        if p.grad is None:
            continue
        has_grad = True
        g = p.grad.detach()
        total += float(g.float().pow(2).sum().item())
    if not has_grad:
        return 0.0
    return float(math.sqrt(total))


@dataclass
class ImitationRunResult:
    """``fit_imitation`` の戻り値 (per-epoch metrics + final aggregate)。

    Attributes
    ----------
    epochs:
        ``ImitationMetrics`` の list (epoch 順)。
    final:
        最後の epoch の metrics。
    """

    epochs: list[ImitationMetrics] = field(default_factory=list)
    final: ImitationMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": [m.to_dict() for m in self.epochs],
            "final": self.final.to_dict() if self.final is not None else None,
        }


def fit_imitation(
    model: Stage03Model,
    samples: list[DecisionSample],
    config: ImitationConfig,
    *,
    optimizer: optim.Optimizer | None = None,
) -> ImitationRunResult:
    """``samples`` 上で ``config.num_epochs`` epoch 回す convenience。

    sample list を内部で minibatch 化して per-epoch metrics を集計する。
    ``optimizer`` が ``None`` のときは ``make_default_optimizer`` で AdamW
    を作る。
    """
    if not samples:
        raise ValueError("fit_imitation: empty samples")
    if optimizer is None:
        optimizer = make_default_optimizer(model, config)
    rng = _random.Random(config.seed)
    result = ImitationRunResult()
    for _ in range(int(config.num_epochs)):
        epoch_metrics = train_imitation_epoch(
            model, samples, optimizer, config, rng=rng
        )
        result.epochs.append(epoch_metrics)
    result.final = result.epochs[-1] if result.epochs else None
    return result


# ---------------------------------------------------------------------------
# Convenience JSON helpers
# ---------------------------------------------------------------------------


def metrics_to_json(metrics: ImitationMetrics) -> str:
    """``ImitationMetrics`` を JSON 文字列に直す convenience。"""
    return json.dumps(metrics.to_dict())


__all__ = [
    "ImitationConfig",
    "ImitationMetrics",
    "ImitationRunResult",
    "LRGroupConfig",
    "compute_imitation_loss",
    "train_imitation_epoch",
    "fit_imitation",
    "make_default_optimizer",
    "make_imitation_optimizer_with_info",
    "metrics_to_json",
]
