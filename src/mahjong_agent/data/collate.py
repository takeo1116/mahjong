"""Minibatch collation for ``DecisionSample``.

可変長 candidate を持つ sample list を、torch tensor 主体の
``DecisionBatch`` に変換する。padding と ``candidate_mask`` を付ける。
"""
from __future__ import annotations

import numpy as np
import torch

from mahjong_agent.data.types import (
    SCHEMA_VERSION,
    DecisionBatch,
    DecisionSample,
)
from mahjong_agent.targets.yaku import NUM_YAKU

_DISCARD_MASK_DIM = 34


def _infer_obs_dim(samples: list[DecisionSample]) -> int:
    for s in samples:
        arr = np.asarray(s.observation)
        if arr.size > 0:
            return int(arr.reshape(-1).shape[0])
    return 0


def _infer_cand_dim(samples: list[DecisionSample]) -> int:
    for s in samples:
        arr = np.asarray(s.candidate_features)
        if arr.ndim == 2 and arr.shape[0] > 0:
            return int(arr.shape[1])
        if arr.ndim == 2 and arr.shape[1] > 0:
            # (0, candidate_dim) でも cand_dim が分かる
            return int(arr.shape[1])
        if arr.ndim == 1 and arr.size > 0:
            return int(arr.size)
    return 0


def collate_decision_samples(samples: list[DecisionSample]) -> DecisionBatch:
    """``DecisionSample`` の list を ``DecisionBatch`` に変換する。

    可変長の ``candidate_features`` は ``Cmax = max(Ci)`` で padding し、
    ``candidate_mask`` (1=valid) を返す。``Cmax == 0`` のときは
    ``(N, 0, candidate_dim)`` を返す。

    全 sample で ``schema_version`` が ``SCHEMA_VERSION`` と一致することを
    verify する。
    """
    if not samples:
        raise ValueError("collate_decision_samples requires non-empty list")
    for i, s in enumerate(samples):
        if int(s.schema_version) != SCHEMA_VERSION:
            raise ValueError(
                f"sample {i} has schema_version={s.schema_version}; "
                f"collate expects {SCHEMA_VERSION}"
            )
    n = len(samples)
    obs_dim = _infer_obs_dim(samples)
    cand_dim = _infer_cand_dim(samples)

    # observation
    obs = np.zeros((n, obs_dim), dtype=np.float32)
    for i, s in enumerate(samples):
        arr = np.asarray(s.observation, dtype=np.float32).reshape(-1)
        if obs_dim > 0:
            if arr.size != obs_dim:
                raise ValueError(
                    f"sample {i}: observation size {arr.size} mismatches "
                    f"inferred obs_dim {obs_dim}"
                )
            obs[i] = arr

    # discard_mask
    discard_mask = np.zeros((n, _DISCARD_MASK_DIM), dtype=np.float32)
    for i, s in enumerate(samples):
        arr = np.asarray(s.discard_mask, dtype=np.float32).reshape(-1)
        if arr.size != _DISCARD_MASK_DIM:
            raise ValueError(
                f"sample {i}: discard_mask size {arr.size} mismatches "
                f"expected {_DISCARD_MASK_DIM}"
            )
        discard_mask[i] = arr

    # candidate_features: pad to Cmax
    candidate_counts = np.zeros(n, dtype=np.int64)
    raw_cands: list[np.ndarray] = []
    for s in samples:
        arr = np.asarray(s.candidate_features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, cand_dim) if cand_dim > 0 else arr.reshape(0, 0)
        if arr.ndim != 2:
            raise ValueError(
                f"candidate_features must be 2D, got shape {arr.shape}"
            )
        raw_cands.append(arr)
    candidate_counts = np.asarray(
        [arr.shape[0] for arr in raw_cands], dtype=np.int64
    )
    cmax = int(candidate_counts.max(initial=0))
    cand_feat = np.zeros((n, cmax, cand_dim), dtype=np.float32)
    cand_mask = np.zeros((n, cmax), dtype=np.float32)
    for i, arr in enumerate(raw_cands):
        c = arr.shape[0]
        if c == 0:
            continue
        if cand_dim > 0 and arr.shape[1] != cand_dim:
            raise ValueError(
                f"sample {i}: candidate_features dim {arr.shape[1]} mismatches "
                f"inferred candidate_dim {cand_dim}"
            )
        cand_feat[i, :c, :cand_dim] = arr
        cand_mask[i, :c] = 1.0

    # yaku_target
    yaku_target = np.zeros((n, NUM_YAKU), dtype=np.float32)
    for i, s in enumerate(samples):
        arr = np.asarray(s.yaku_target, dtype=np.float32).reshape(-1)
        if arr.size != NUM_YAKU:
            raise ValueError(
                f"sample {i}: yaku_target size {arr.size} mismatches "
                f"NUM_YAKU={NUM_YAKU}"
            )
        yaku_target[i] = arr

    # teacher_best_mask
    teacher_best_mask = np.zeros((n, _DISCARD_MASK_DIM), dtype=np.float32)
    for i, s in enumerate(samples):
        arr = np.asarray(s.teacher_best_mask, dtype=np.float32).reshape(-1)
        if arr.size != _DISCARD_MASK_DIM:
            raise ValueError(
                f"sample {i}: teacher_best_mask size {arr.size} mismatches "
                f"expected {_DISCARD_MASK_DIM}"
            )
        teacher_best_mask[i] = arr

    def _int_t(getter):
        return torch.tensor(
            [int(getter(s)) for s in samples], dtype=torch.int64
        )

    def _float_t(getter):
        return torch.tensor(
            [float(getter(s)) for s in samples], dtype=torch.float32
        )

    return DecisionBatch(
        schema_version=SCHEMA_VERSION,
        episode_id=[str(s.episode_id) for s in samples],
        round_id=_int_t(lambda s: s.round_id),
        step_id=_int_t(lambda s: s.step_id),
        player_id=_int_t(lambda s: s.player_id),
        decision_family=[str(s.decision_family) for s in samples],
        actor_type=[str(s.actor_type) for s in samples],
        observation=torch.from_numpy(obs),
        discard_mask=torch.from_numpy(discard_mask),
        candidate_features=torch.from_numpy(cand_feat),
        candidate_mask=torch.from_numpy(cand_mask),
        candidate_count=torch.from_numpy(candidate_counts),
        candidate_dim=int(cand_dim),
        selected_discard_tile_type=_int_t(lambda s: s.selected_discard_tile_type),
        selected_candidate_index=_int_t(lambda s: s.selected_candidate_index),
        old_log_prob=_float_t(lambda s: s.old_log_prob),
        value=_float_t(lambda s: s.value),
        reward=_float_t(lambda s: s.reward),
        terminated=_float_t(lambda s: 1.0 if s.terminated else 0.0),
        round_over=_float_t(lambda s: 1.0 if s.round_over else 0.0),
        terminal_class=_int_t(lambda s: s.terminal_class),
        yaku_target=torch.from_numpy(yaku_target),
        yaku_loss_mask=_float_t(lambda s: s.yaku_loss_mask),
        han=_int_t(lambda s: s.han),
        fu=_int_t(lambda s: s.fu),
        score_delta=_int_t(lambda s: s.score_delta),
        teacher_discard_tile_type=_int_t(
            lambda s: s.teacher_discard_tile_type
        ),
        teacher_candidate_index=_int_t(lambda s: s.teacher_candidate_index),
        teacher_best_mask=torch.from_numpy(teacher_best_mask),
        teacher_available=_float_t(
            lambda s: 1.0 if s.teacher_available else 0.0
        ),
        metadata=[dict(s.metadata) for s in samples],
    )


__all__ = ["collate_decision_samples"]
