"""Read-only shard / sample audit for Stage03 diagnostics.

本 module は **training behavior を変えない** offline audit。``DecisionSample``
shard または sample list を入力に取り、teacher / family / policy diagnostics
を JSON / Markdown summary に丸める。

設計方針:

- engine 由来の hidden state (player の手牌や山牌、内部 phase) には触らない。
  入力は ``DecisionSample`` の public encoder output と target label のみ。
- model が渡された場合のみ policy-side diagnostics を出す。
- numpy / torch scalar は最終 dict に残さず、すべて Python primitive に変換する。
- audit が hidden state を model に渡さないこと (= 入力に env / replay 等を
  取らない) を signature で固定する。

Stage02 (``optional_family_audit.py``) から参考にした点:

- Counter + percentile ベースの軽量集計関数の構造。
- family ごとに count / actor / teacher agreement / entropy / max_prob を出す
  layout。

Stage03 と Stage02 の違い:

- Stage03 では NORMAL_DISCARD / candidate (TSUMO / RON / CHI / PON / ANKAN
  / KAKAN / DAIMINKAN / KYUSHU_KYUHAI / KITA / PASS) を ``ActionFamily`` で
  表現する。本 audit は ``ActionFamily`` の値そのまま family キーに使う。
- combined softmax policy なので、entropy / max_prob も combined distribution
  上で計算する。
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from mahjong_agent.actions.types import ActionFamily
from mahjong_agent.data.collate import collate_decision_samples
from mahjong_agent.data.shard_io import read_decision_shard
from mahjong_agent.data.types import DecisionBatch, DecisionSample
from mahjong_agent.models.stage03_model import Stage03Model

_NORMAL_DISCARD_FAMILY: str = ActionFamily.NORMAL_DISCARD.value
_LARGE_NEG: float = -1.0e9
_NUM_TILE_TYPES: int = 34


# ---------------------------------------------------------------------------
# Config / summary dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardAuditConfig:
    """audit 動作 config。

    Attributes
    ----------
    device:
        ``"cpu"`` / ``"cuda"`` 等。model forward で使う。
    max_model_batches:
        model forward を流す minibatch 数の上限。``None`` で全 sample。
    batch_size:
        model forward 用 minibatch サイズ。
    sanity_max_illegal_prob:
        illegal discard / candidate padding に確率が乗っていないことを判定する
        閾値。``max_illegal_prob`` がこの値を超えたら ``illegal_prob_ok=False``。
    """

    device: str = "cpu"
    max_model_batches: int | None = None
    batch_size: int = 256
    sanity_max_illegal_prob: float = 1e-4


@dataclass(frozen=True)
class ShardAuditSummary:
    """audit 結果。``to_dict`` で JSON-serializable dict にする。"""

    num_samples: int
    schema_version: int
    observation_dim: int
    candidate_dim: int
    discard_count: int
    candidate_count: int
    decision_family_counts: dict[str, int]
    actor_type_counts: dict[str, int]
    player_id_counts: dict[int, int]
    round_id_distribution: dict[str, float]
    candidate_count_distribution: dict[str, float]
    selected_discard_tile_distribution: dict[int, int]
    selected_candidate_index_distribution: dict[int, int]
    terminal_class_counts: dict[int, int]
    yaku_loss_mask_count: int
    yaku_positive_total: int
    han_distribution: dict[str, float]
    fu_distribution: dict[str, float]
    score_delta_summary: dict[str, float]
    reward_summary: dict[str, float]
    teacher: dict[str, Any]
    family_audit: dict[str, dict[str, Any]]
    policy: dict[str, Any]
    model_evaluated: bool = False
    metadata_keys: dict[str, int] = field(default_factory=dict)
    """audit-time に観測した ``metadata`` キー出現頻度。``ppo_exclude`` /
    ``is_post_riichi_discard`` 等の boolean flag は別個に集計する。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": int(self.num_samples),
            "schema_version": int(self.schema_version),
            "observation_dim": int(self.observation_dim),
            "candidate_dim": int(self.candidate_dim),
            "discard_count": int(self.discard_count),
            "candidate_count": int(self.candidate_count),
            "decision_family_counts": _str_int_dict(self.decision_family_counts),
            "actor_type_counts": _str_int_dict(self.actor_type_counts),
            "player_id_counts": _int_int_dict(self.player_id_counts),
            "round_id_distribution": _str_float_dict(self.round_id_distribution),
            "candidate_count_distribution": _str_float_dict(
                self.candidate_count_distribution
            ),
            "selected_discard_tile_distribution": _int_int_dict(
                self.selected_discard_tile_distribution
            ),
            "selected_candidate_index_distribution": _int_int_dict(
                self.selected_candidate_index_distribution
            ),
            "terminal_class_counts": _int_int_dict(self.terminal_class_counts),
            "yaku_loss_mask_count": int(self.yaku_loss_mask_count),
            "yaku_positive_total": int(self.yaku_positive_total),
            "han_distribution": _str_float_dict(self.han_distribution),
            "fu_distribution": _str_float_dict(self.fu_distribution),
            "score_delta_summary": _str_float_dict(self.score_delta_summary),
            "reward_summary": _str_float_dict(self.reward_summary),
            "teacher": _to_jsonable(self.teacher),
            "family_audit": {
                str(k): _to_jsonable(v) for k, v in self.family_audit.items()
            },
            "policy": _to_jsonable(self.policy),
            "model_evaluated": bool(self.model_evaluated),
            "metadata_keys": _str_int_dict(self.metadata_keys),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _str_int_dict(d: dict[Any, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in d.items()}


def _int_int_dict(d: dict[Any, Any]) -> dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def _str_float_dict(d: dict[Any, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in d.items()}


def _to_jsonable(obj: Any) -> Any:
    """numpy / torch scalar を Python primitive に降ろす再帰 helper。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return [float(x) for x in obj.detach().cpu().reshape(-1).tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def _basic_stats(values: Sequence[float]) -> dict[str, float]:
    """count / mean / std / min / max / p50 / p90 を返す軽量 helper。

    values が空のときは全 0 を返す (downstream で穴埋めが楽になる)。
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


# ---------------------------------------------------------------------------
# Per-sample collectors (non-model)
# ---------------------------------------------------------------------------


def _aggregate_base(samples: Sequence[DecisionSample]) -> dict[str, Any]:
    """shard summary の core counts / distributions を集計する。"""
    n = len(samples)
    if n == 0:
        return {
            "num_samples": 0,
            "schema_version": 0,
            "observation_dim": 0,
            "candidate_dim": 0,
            "discard_count": 0,
            "candidate_count": 0,
            "decision_family_counts": {},
            "actor_type_counts": {},
            "player_id_counts": {},
            "round_id_distribution": _basic_stats([]),
            "candidate_count_distribution": _basic_stats([]),
            "selected_discard_tile_distribution": {},
            "selected_candidate_index_distribution": {},
            "terminal_class_counts": {},
            "yaku_loss_mask_count": 0,
            "yaku_positive_total": 0,
            "han_distribution": _basic_stats([]),
            "fu_distribution": _basic_stats([]),
            "score_delta_summary": _basic_stats([]),
            "reward_summary": _basic_stats([]),
            "metadata_keys": {},
        }
    family_counter: Counter[str] = Counter()
    actor_counter: Counter[str] = Counter()
    player_counter: Counter[int] = Counter()
    terminal_counter: Counter[int] = Counter()
    sel_discard_counter: Counter[int] = Counter()
    sel_candidate_counter: Counter[int] = Counter()
    metadata_key_counter: Counter[str] = Counter()
    round_ids: list[int] = []
    cand_counts: list[int] = []
    han_values: list[int] = []
    fu_values: list[int] = []
    score_deltas: list[int] = []
    rewards: list[float] = []
    yaku_loss_mask_count = 0
    yaku_positive_total = 0
    discard_count = 0
    candidate_count = 0
    obs_dim = int(np.asarray(samples[0].observation).shape[-1])
    cand_dim = (
        int(samples[0].candidate_features.shape[-1])
        if samples[0].candidate_features.ndim == 2
        and samples[0].candidate_features.shape[-1] > 0
        else 0
    )
    schema_version = int(samples[0].schema_version)
    for s in samples:
        family = str(s.decision_family)
        family_counter[family] += 1
        actor_counter[str(s.actor_type)] += 1
        player_counter[int(s.player_id)] += 1
        round_ids.append(int(s.round_id))
        c = int(s.candidate_features.shape[0]) if s.candidate_features.ndim == 2 else 0
        cand_counts.append(c)
        if family == _NORMAL_DISCARD_FAMILY:
            discard_count += 1
            if int(s.selected_discard_tile_type) >= 0:
                sel_discard_counter[int(s.selected_discard_tile_type)] += 1
        else:
            candidate_count += 1
            if int(s.selected_candidate_index) >= 0:
                sel_candidate_counter[int(s.selected_candidate_index)] += 1
        # candidate_dim is inferred per-sample; samples with C=0 may have shape (0,0)
        if s.candidate_features.ndim == 2 and s.candidate_features.shape[-1] > 0:
            cand_dim = max(cand_dim, int(s.candidate_features.shape[-1]))
        if int(s.terminal_class) >= 0:
            terminal_counter[int(s.terminal_class)] += 1
        if float(s.yaku_loss_mask) > 0:
            yaku_loss_mask_count += 1
            yaku_positive_total += int(
                (np.asarray(s.yaku_target) > 0.5).sum()
            )
        if int(s.han) >= 0:
            han_values.append(int(s.han))
        if int(s.fu) >= 0:
            fu_values.append(int(s.fu))
        score_deltas.append(int(s.score_delta))
        rewards.append(float(s.reward))
        if s.metadata:
            for k in s.metadata.keys():
                metadata_key_counter[str(k)] += 1

    return {
        "num_samples": int(n),
        "schema_version": int(schema_version),
        "observation_dim": int(obs_dim),
        "candidate_dim": int(cand_dim),
        "discard_count": int(discard_count),
        "candidate_count": int(candidate_count),
        "decision_family_counts": dict(family_counter),
        "actor_type_counts": dict(actor_counter),
        "player_id_counts": dict(player_counter),
        "round_id_distribution": _basic_stats(round_ids),
        "candidate_count_distribution": _basic_stats(cand_counts),
        "selected_discard_tile_distribution": dict(sel_discard_counter),
        "selected_candidate_index_distribution": dict(sel_candidate_counter),
        "terminal_class_counts": dict(terminal_counter),
        "yaku_loss_mask_count": int(yaku_loss_mask_count),
        "yaku_positive_total": int(yaku_positive_total),
        "han_distribution": _basic_stats(han_values),
        "fu_distribution": _basic_stats(fu_values),
        "score_delta_summary": _basic_stats(score_deltas),
        "reward_summary": _basic_stats(rewards),
        "metadata_keys": dict(metadata_key_counter),
    }


def _aggregate_teacher(samples: Sequence[DecisionSample]) -> dict[str, Any]:
    """teacher 情報の有無を含めて teacher diagnostics を集計する。"""
    teacher_available = 0
    teacher_discard_counter: Counter[int] = Counter()
    teacher_candidate_counter: Counter[int] = Counter()
    best_mask_nonzero_count = 0
    best_mask_sizes: list[int] = []
    selected_in_best_mask = 0
    selected_in_best_mask_total = 0
    teacher_discard_agree = 0
    teacher_discard_agree_total = 0
    post_riichi_discard = 0
    for s in samples:
        if bool(s.teacher_available):
            teacher_available += 1
        if int(s.teacher_discard_tile_type) >= 0:
            teacher_discard_counter[int(s.teacher_discard_tile_type)] += 1
        if int(s.teacher_candidate_index) >= 0:
            teacher_candidate_counter[int(s.teacher_candidate_index)] += 1
        mask = np.asarray(s.teacher_best_mask).reshape(-1)
        if mask.size and (mask > 0).any():
            nz = int((mask > 0).sum())
            best_mask_nonzero_count += 1
            best_mask_sizes.append(nz)
            if (
                str(s.decision_family) == _NORMAL_DISCARD_FAMILY
                and int(s.selected_discard_tile_type) >= 0
            ):
                selected_in_best_mask_total += 1
                if mask[int(s.selected_discard_tile_type)] > 0:
                    selected_in_best_mask += 1
        if (
            str(s.decision_family) == _NORMAL_DISCARD_FAMILY
            and int(s.selected_discard_tile_type) >= 0
            and int(s.teacher_discard_tile_type) >= 0
        ):
            teacher_discard_agree_total += 1
            if (
                int(s.selected_discard_tile_type)
                == int(s.teacher_discard_tile_type)
            ):
                teacher_discard_agree += 1
        if bool((s.metadata or {}).get("is_post_riichi_discard", False)):
            post_riichi_discard += 1

    return {
        "teacher_available_count": int(teacher_available),
        "teacher_discard_tile_type_distribution": dict(teacher_discard_counter),
        "teacher_candidate_index_distribution": dict(teacher_candidate_counter),
        "teacher_best_mask_nonzero_count": int(best_mask_nonzero_count),
        "teacher_best_mask_size_distribution": _basic_stats(best_mask_sizes),
        "selected_in_teacher_best_mask_count": int(selected_in_best_mask),
        "selected_in_teacher_best_mask_total": int(selected_in_best_mask_total),
        "selected_in_teacher_best_mask_rate": (
            float(selected_in_best_mask) / float(selected_in_best_mask_total)
            if selected_in_best_mask_total > 0
            else 0.0
        ),
        "teacher_discard_agreement_count": int(teacher_discard_agree),
        "teacher_discard_agreement_total": int(teacher_discard_agree_total),
        "teacher_discard_agreement_rate": (
            float(teacher_discard_agree) / float(teacher_discard_agree_total)
            if teacher_discard_agree_total > 0
            else 0.0
        ),
        "post_riichi_discard_count": int(post_riichi_discard),
    }


def _aggregate_family_audit(
    samples: Sequence[DecisionSample],
    *,
    policy_per_sample: dict[int, dict[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """family ごとに count / candidate_count / actor_type / teacher agreement /
    policy entropy + max_prob を集計する (Stage02 optional_family_audit 相当)。

    Stage03 の family は ``ActionFamily`` 値 (= decision_family 文字列) を直接
    キーに使う。teacher / policy diagnostics は available なら mean を出す。
    """
    by_family: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[tuple[int, DecisionSample]]] = {}
    for i, s in enumerate(samples):
        family = str(s.decision_family)
        grouped.setdefault(family, []).append((i, s))
    policy_per_sample = policy_per_sample or {}
    for family, pairs in grouped.items():
        cand_counts = [int(s.candidate_features.shape[0]) if s.candidate_features.ndim == 2 else 0
                        for _, s in pairs]
        actor_counter: Counter[str] = Counter()
        selected_counter: Counter[int] = Counter()
        teacher_agree = 0
        teacher_agree_total = 0
        post_riichi = 0
        policy_entropies: list[float] = []
        policy_max_probs: list[float] = []
        policy_selected_probs: list[float] = []
        # candidate 局面で model が call を放棄して discard に倒した回数
        # (= teacher=call なのに model argmax が discard 領域)。
        abandon_count = 0  # argmax が discard 領域だった candidate sample 数
        pred_teacher_cand_count = 0  # argmax が teacher の選んだ candidate
        abandon_total = 0  # policy 評価できた candidate sample 数
        for i, s in pairs:
            actor_counter[str(s.actor_type)] += 1
            if family == _NORMAL_DISCARD_FAMILY:
                sel = int(s.selected_discard_tile_type)
                if sel >= 0:
                    selected_counter[sel] += 1
                    if int(s.teacher_discard_tile_type) >= 0:
                        teacher_agree_total += 1
                        if sel == int(s.teacher_discard_tile_type):
                            teacher_agree += 1
                if bool((s.metadata or {}).get("is_post_riichi_discard", False)):
                    post_riichi += 1
            else:
                sel = int(s.selected_candidate_index)
                if sel >= 0:
                    selected_counter[sel] += 1
                    if int(s.teacher_candidate_index) >= 0:
                        teacher_agree_total += 1
                        if sel == int(s.teacher_candidate_index):
                            teacher_agree += 1
            ps = policy_per_sample.get(i)
            if ps is not None:
                policy_entropies.append(float(ps.get("entropy", 0.0)))
                policy_max_probs.append(float(ps.get("max_prob", 0.0)))
                policy_selected_probs.append(
                    float(ps.get("selected_prob", 0.0))
                )
                if family != _NORMAL_DISCARD_FAMILY:
                    abandon_total += 1
                    if float(ps.get("argmax_in_candidate_region", 1.0)) < 0.5:
                        abandon_count += 1
                    if float(ps.get("argmax_is_teacher_candidate", 0.0)) > 0.5:
                        pred_teacher_cand_count += 1
        entry: dict[str, Any] = {
            "count": int(len(pairs)),
            "candidate_count_stats": _basic_stats(cand_counts),
            "actor_type_counts": dict(actor_counter),
            "selected_distribution": dict(selected_counter),
            "teacher_agreement_count": int(teacher_agree),
            "teacher_agreement_total": int(teacher_agree_total),
            "teacher_agreement_rate": (
                float(teacher_agree) / float(teacher_agree_total)
                if teacher_agree_total > 0
                else 0.0
            ),
            "post_riichi_count": int(post_riichi),
        }
        if policy_per_sample:
            entry["policy_entropy_stats"] = _basic_stats(policy_entropies)
            entry["policy_max_prob_stats"] = _basic_stats(policy_max_probs)
            entry["policy_selected_prob_stats"] = _basic_stats(
                policy_selected_probs
            )
            # candidate family のみ: call 放棄 / teacher candidate 一致の率。
            if family != _NORMAL_DISCARD_FAMILY and abandon_total > 0:
                entry["policy_abandon_call_count"] = int(abandon_count)
                entry["policy_abandon_call_total"] = int(abandon_total)
                entry["policy_abandon_call_rate"] = (
                    float(abandon_count) / float(abandon_total)
                )
                entry["policy_pred_teacher_candidate_count"] = int(
                    pred_teacher_cand_count
                )
                entry["policy_pred_teacher_candidate_rate"] = (
                    float(pred_teacher_cand_count) / float(abandon_total)
                )
        by_family[family] = entry
    return by_family


# ---------------------------------------------------------------------------
# Model-based policy diagnostics
# ---------------------------------------------------------------------------


def _evaluate_policy(
    samples: Sequence[DecisionSample],
    *,
    model: Stage03Model,
    config: ShardAuditConfig,
) -> tuple[dict[str, Any], dict[int, dict[str, float]]]:
    """model を eval mode で走らせて combined-policy diagnostics を計算する。

    Stage03 ``ModelPolicyAgent`` と同じ combined softmax (discard logits 34 +
    candidate scores Cmax) 上で entropy / max_prob / selected probability /
    teacher 重み等を per-sample に出す。

    Returns
    -------
    (agg, per_sample_policy):
        ``agg`` は family-agnostic な aggregated diagnostics dict。
        ``per_sample_policy`` は sample 全体 index → {entropy, max_prob,
        selected_prob} の dict (family audit に使う)。
    """
    device = torch.device(config.device)
    model_was_training = model.training
    model.eval()
    n = len(samples)
    per_sample: dict[int, dict[str, float]] = {}
    discard_entropies: list[float] = []
    discard_max_probs: list[float] = []
    candidate_entropies: list[float] = []
    candidate_max_probs: list[float] = []
    selected_probs: list[float] = []
    teacher_selected_probs: list[float] = []
    teacher_best_mass: list[float] = []
    max_illegal_prob = 0.0
    max_candidate_padding_prob = 0.0
    samples_evaluated = 0
    max_batches = config.max_model_batches
    bs = max(1, int(config.batch_size))
    try:
        with torch.no_grad():
            for batch_idx, start in enumerate(range(0, n, bs)):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                end = min(n, start + bs)
                sub = list(samples[start:end])
                if not sub:
                    continue
                batch: DecisionBatch = collate_decision_samples(sub)
                obs = batch.observation.to(device).float()
                discard_mask = batch.discard_mask.to(device).float()
                fwd = model(obs, discard_mask=discard_mask)
                cand_out = model.score_candidates(
                    obs, batch.candidate_features.to(device).float()
                )
                cand_scores = cand_out.candidate_scores
                cmax = int(cand_scores.size(1))
                cand_mask = batch.candidate_mask.to(device).float()
                if cmax > 0:
                    masked_cand = cand_scores + (1.0 - cand_mask) * _LARGE_NEG
                else:
                    masked_cand = cand_scores
                combined_logits = torch.cat(
                    [fwd.discard_logits, masked_cand], dim=-1
                )
                log_softmax = F.log_softmax(combined_logits, dim=-1)
                probs = torch.exp(log_softmax)  # (B, 34 + Cmax)
                # sanity: illegal discard / candidate padding に確率が乗らない
                # discard mask=0 の位置の確率合計
                illegal_discard_mask = (
                    1.0 - discard_mask
                )  # (B, 34) 1.0 if illegal
                illegal_discard_prob = (
                    probs[:, :_NUM_TILE_TYPES] * illegal_discard_mask
                ).sum(dim=-1)
                if cmax > 0:
                    pad_mask = 1.0 - cand_mask  # (B, Cmax) 1.0 if pad
                    pad_prob = (probs[:, _NUM_TILE_TYPES:] * pad_mask).sum(
                        dim=-1
                    )
                else:
                    pad_prob = torch.zeros(
                        probs.size(0), device=probs.device, dtype=probs.dtype
                    )
                max_illegal_prob = max(
                    max_illegal_prob, float(illegal_discard_prob.max().item())
                )
                max_candidate_padding_prob = max(
                    max_candidate_padding_prob, float(pad_prob.max().item())
                )
                # per-sample
                #
                # 設計方針:
                # - entropy / max_prob は family 内分布を見たいので
                #   family 部分を renormalize した値を使う (= 旧挙動)。
                # - selected_prob / teacher_selected_prob / teacher_best_mass
                #   は「実 policy がその action を選ぶ確率」を意味するため、
                #   combined softmax (`probs[k]`) 上の確率をそのまま使う。
                # - candidate 部分が空 (cmax==0) でも crash しないよう、
                #   empty tensor 経路を分岐して 0.0 を埋める。
                for k, s in enumerate(sub):
                    global_idx = start + k
                    family = str(s.decision_family)
                    combined_row = probs[k]  # (34 + Cmax,)
                    if family == _NORMAL_DISCARD_FAMILY:
                        # discard-only renormalize: entropy / max_prob 用。
                        d_probs = combined_row[:_NUM_TILE_TYPES]
                        d_sum = d_probs.sum().clamp_min(1e-12)
                        d_norm = d_probs / d_sum
                        d_log = torch.log(d_norm.clamp_min(1e-12))
                        d_entropy = float(-(d_norm * d_log).sum().item())
                        d_maxp = float(d_norm.max().item())
                        discard_entropies.append(d_entropy)
                        discard_max_probs.append(d_maxp)
                        sel = int(s.selected_discard_tile_type)
                        # selected_prob は combined softmax 上の確率。
                        sel_prob = (
                            float(combined_row[sel].item())
                            if 0 <= sel < _NUM_TILE_TYPES
                            else 0.0
                        )
                        per_sample[global_idx] = {
                            "entropy": d_entropy,
                            "max_prob": d_maxp,
                            "selected_prob": sel_prob,
                        }
                        if 0 <= sel < _NUM_TILE_TYPES:
                            selected_probs.append(sel_prob)
                        # teacher diagnostics: combined probability。
                        teacher_tt = int(s.teacher_discard_tile_type)
                        if 0 <= teacher_tt < _NUM_TILE_TYPES:
                            teacher_selected_probs.append(
                                float(combined_row[teacher_tt].item())
                            )
                        tbm = np.asarray(s.teacher_best_mask).reshape(-1)
                        if tbm.size and (tbm > 0).any():
                            mask_t = torch.from_numpy(
                                (tbm > 0).astype(np.float32)
                            ).to(combined_row.device)
                            teacher_best_mass.append(
                                float(
                                    (combined_row[:_NUM_TILE_TYPES] * mask_t)
                                    .sum()
                                    .item()
                                )
                            )
                    else:
                        # candidate-only renormalize: entropy / max_prob 用。
                        # cmax==0 の sample (= candidate_features.shape[0]==0
                        # の family != normal_discard) でも crash しないよう、
                        # 空 tensor 経路を 0.0 で埋める。
                        cprobs = combined_row[_NUM_TILE_TYPES:]
                        sel = int(s.selected_candidate_index)
                        if cprobs.numel() > 0:
                            c_sum = cprobs.sum().clamp_min(1e-12)
                            c_norm = cprobs / c_sum
                            c_log = torch.log(c_norm.clamp_min(1e-12))
                            c_entropy = float(-(c_norm * c_log).sum().item())
                            c_maxp = float(c_norm.max().item())
                        else:
                            c_entropy = 0.0
                            c_maxp = 0.0
                        candidate_entropies.append(c_entropy)
                        candidate_max_probs.append(c_maxp)
                        # selected_prob は combined softmax 上の確率
                        # (= 34 + sel index)。
                        sel_prob = 0.0
                        if 0 <= sel < cprobs.numel():
                            sel_prob = float(
                                combined_row[_NUM_TILE_TYPES + sel].item()
                            )
                        # combined argmax が discard 領域 (<34) に落ちると、
                        # candidate 局面で model が「call を取らず打牌に倒す」
                        # ことを意味する (= teacher=chi なのに model=discard/pass
                        # の検出に使う)。
                        argmax_idx = int(combined_row.argmax().item())
                        argmax_in_candidate = argmax_idx >= _NUM_TILE_TYPES
                        argmax_is_teacher_cand = (
                            argmax_in_candidate
                            and (argmax_idx - _NUM_TILE_TYPES) == sel
                            and 0 <= sel < cprobs.numel()
                        )
                        per_sample[global_idx] = {
                            "entropy": c_entropy,
                            "max_prob": c_maxp,
                            "selected_prob": sel_prob,
                            "argmax_in_candidate_region": float(
                                argmax_in_candidate
                            ),
                            "argmax_is_teacher_candidate": float(
                                argmax_is_teacher_cand
                            ),
                        }
                        if 0 <= sel < cprobs.numel():
                            selected_probs.append(sel_prob)
                samples_evaluated += end - start
    finally:
        if model_was_training:
            model.train()

    agg: dict[str, Any] = {
        "samples_evaluated": int(samples_evaluated),
        "discard_entropy_stats": _basic_stats(discard_entropies),
        "discard_max_prob_stats": _basic_stats(discard_max_probs),
        "candidate_entropy_stats": _basic_stats(candidate_entropies),
        "candidate_max_prob_stats": _basic_stats(candidate_max_probs),
        "selected_prob_stats": _basic_stats(selected_probs),
        "teacher_selected_prob_stats": _basic_stats(teacher_selected_probs),
        "teacher_best_mass_stats": _basic_stats(teacher_best_mass),
        "max_illegal_discard_prob": float(max_illegal_prob),
        "max_candidate_padding_prob": float(max_candidate_padding_prob),
        "illegal_prob_ok": bool(
            max_illegal_prob <= float(config.sanity_max_illegal_prob)
            and max_candidate_padding_prob
            <= float(config.sanity_max_illegal_prob)
        ),
    }
    return agg, per_sample


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_decision_samples(
    samples: Sequence[DecisionSample],
    *,
    model: Stage03Model | None = None,
    config: ShardAuditConfig | None = None,
) -> ShardAuditSummary:
    """``DecisionSample`` 列を audit して ``ShardAuditSummary`` を返す。

    - ``model`` が ``None`` の場合は base + teacher + family audit のみを集計し、
      ``model_evaluated=False`` をセットする。
    - engine 由来の hidden state や replay event log には触らない。
      ``DecisionSample`` の public field のみ参照する。
    """
    cfg = config or ShardAuditConfig()
    base = _aggregate_base(samples)
    teacher = _aggregate_teacher(samples)
    policy_per_sample: dict[int, dict[str, float]] | None = None
    policy_agg: dict[str, Any] = {}
    model_evaluated = False
    if model is not None and len(samples) > 0:
        policy_agg, policy_per_sample = _evaluate_policy(
            samples, model=model, config=cfg
        )
        model_evaluated = True
    family_audit = _aggregate_family_audit(
        samples, policy_per_sample=policy_per_sample
    )
    return ShardAuditSummary(
        num_samples=int(base["num_samples"]),
        schema_version=int(base["schema_version"]),
        observation_dim=int(base["observation_dim"]),
        candidate_dim=int(base["candidate_dim"]),
        discard_count=int(base["discard_count"]),
        candidate_count=int(base["candidate_count"]),
        decision_family_counts=base["decision_family_counts"],
        actor_type_counts=base["actor_type_counts"],
        player_id_counts=base["player_id_counts"],
        round_id_distribution=base["round_id_distribution"],
        candidate_count_distribution=base["candidate_count_distribution"],
        selected_discard_tile_distribution=base[
            "selected_discard_tile_distribution"
        ],
        selected_candidate_index_distribution=base[
            "selected_candidate_index_distribution"
        ],
        terminal_class_counts=base["terminal_class_counts"],
        yaku_loss_mask_count=int(base["yaku_loss_mask_count"]),
        yaku_positive_total=int(base["yaku_positive_total"]),
        han_distribution=base["han_distribution"],
        fu_distribution=base["fu_distribution"],
        score_delta_summary=base["score_delta_summary"],
        reward_summary=base["reward_summary"],
        teacher=teacher,
        family_audit=family_audit,
        policy=policy_agg,
        model_evaluated=model_evaluated,
        metadata_keys=base["metadata_keys"],
    )


def audit_decision_shard(
    path: str | Path,
    *,
    model: Stage03Model | None = None,
    config: ShardAuditConfig | None = None,
) -> ShardAuditSummary:
    """``.npz`` shard を読み込んで audit する high-level API。"""
    samples = read_decision_shard(Path(path))
    return audit_decision_samples(samples, model=model, config=config)


# ---------------------------------------------------------------------------
# JSON / Markdown rendering
# ---------------------------------------------------------------------------


def summary_to_json(summary: ShardAuditSummary) -> str:
    """``ShardAuditSummary`` を JSON 文字列に変換する。"""
    return json.dumps(summary.to_dict(), ensure_ascii=False)


def summary_to_markdown(summary: ShardAuditSummary) -> str:
    """人間が一目で読める Markdown summary を返す。

    diagnostics 用の軽量出力。``mahjong-experiments/`` の report ではない。
    """
    d = summary.to_dict()
    lines: list[str] = []
    lines.append("# Shard Audit Summary")
    lines.append("")
    lines.append(f"- num_samples: {d['num_samples']}")
    lines.append(f"- schema_version: {d['schema_version']}")
    lines.append(f"- observation_dim: {d['observation_dim']}")
    lines.append(f"- candidate_dim: {d['candidate_dim']}")
    lines.append(f"- discard_count: {d['discard_count']}")
    lines.append(f"- candidate_count: {d['candidate_count']}")
    lines.append(f"- model_evaluated: {d['model_evaluated']}")
    lines.append("")
    lines.append("## decision_family_counts")
    for k, v in d["decision_family_counts"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## actor_type_counts")
    for k, v in d["actor_type_counts"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## teacher")
    teacher = d["teacher"]
    lines.append(
        f"- teacher_available_count: {teacher['teacher_available_count']}"
    )
    lines.append(
        f"- selected_in_teacher_best_mask_rate: "
        f"{teacher['selected_in_teacher_best_mask_rate']:.4f}"
        f" ({teacher['selected_in_teacher_best_mask_count']}/"
        f"{teacher['selected_in_teacher_best_mask_total']})"
    )
    lines.append(
        f"- teacher_discard_agreement_rate: "
        f"{teacher['teacher_discard_agreement_rate']:.4f}"
        f" ({teacher['teacher_discard_agreement_count']}/"
        f"{teacher['teacher_discard_agreement_total']})"
    )
    lines.append(
        f"- post_riichi_discard_count: {teacher['post_riichi_discard_count']}"
    )
    lines.append("")
    lines.append("## family_audit")
    for family, info in d["family_audit"].items():
        lines.append(f"### {family}")
        lines.append(f"- count: {info['count']}")
        lines.append(
            "- candidate_count mean/max: "
            f"{info['candidate_count_stats']['mean']:.2f}/"
            f"{info['candidate_count_stats']['max']:.0f}"
        )
        lines.append(
            f"- teacher_agreement_rate: {info['teacher_agreement_rate']:.4f}"
            f" ({info['teacher_agreement_count']}/"
            f"{info['teacher_agreement_total']})"
        )
        lines.append(f"- post_riichi_count: {info['post_riichi_count']}")
        if "policy_entropy_stats" in info:
            lines.append(
                "- policy_entropy mean/p90: "
                f"{info['policy_entropy_stats']['mean']:.4f}/"
                f"{info['policy_entropy_stats']['p90']:.4f}"
            )
            lines.append(
                "- policy_max_prob mean/p50: "
                f"{info['policy_max_prob_stats']['mean']:.4f}/"
                f"{info['policy_max_prob_stats']['p50']:.4f}"
            )
            lines.append(
                "- policy_selected_prob mean: "
                f"{info['policy_selected_prob_stats']['mean']:.4f}"
            )
    if d["model_evaluated"]:
        p = d["policy"]
        lines.append("")
        lines.append("## policy (model)")
        lines.append(f"- samples_evaluated: {p['samples_evaluated']}")
        lines.append(
            f"- discard_entropy mean/p90: {p['discard_entropy_stats']['mean']:.4f}/"
            f"{p['discard_entropy_stats']['p90']:.4f}"
        )
        lines.append(
            f"- candidate_entropy mean/p90: "
            f"{p['candidate_entropy_stats']['mean']:.4f}/"
            f"{p['candidate_entropy_stats']['p90']:.4f}"
        )
        lines.append(
            f"- max_illegal_discard_prob: {p['max_illegal_discard_prob']:.6f}"
        )
        lines.append(
            f"- max_candidate_padding_prob: "
            f"{p['max_candidate_padding_prob']:.6f}"
        )
        lines.append(f"- illegal_prob_ok: {p['illegal_prob_ok']}")
    return "\n".join(lines) + "\n"


__all__ = [
    "ShardAuditConfig",
    "ShardAuditSummary",
    "audit_decision_samples",
    "audit_decision_shard",
    "summary_to_json",
    "summary_to_markdown",
]
