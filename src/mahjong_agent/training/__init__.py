"""Imitation / PPO training loops."""
from __future__ import annotations

from mahjong_agent.training.imitation import (
    ImitationConfig,
    ImitationMetrics,
    ImitationRunResult,
    compute_imitation_loss,
    fit_imitation,
    make_default_optimizer,
    make_imitation_optimizer_with_info,
    metrics_to_json,
    train_imitation_epoch,
)
from mahjong_agent.training.optimizer_groups import (
    LRGroupConfig,
    build_lr_grouped_optimizer,
    classify_parameters_by_group,
)
from mahjong_agent.training.ppo import (
    PPOBatch,
    PPOConfig,
    PPOMetrics,
    PPORunResult,
    PPOTrainingData,
    compute_ppo_loss,
    compute_returns_and_advantages,
    fit_ppo,
    make_default_ppo_optimizer,
    make_ppo_optimizer_with_info,
    ppo_metrics_to_json,
    train_ppo_epoch,
)
from mahjong_agent.training.sample_weighting import compute_per_player_round_weights

__all__ = [
    "ImitationConfig",
    "ImitationMetrics",
    "ImitationRunResult",
    "LRGroupConfig",
    "PPOBatch",
    "PPOConfig",
    "PPOMetrics",
    "PPORunResult",
    "PPOTrainingData",
    "build_lr_grouped_optimizer",
    "classify_parameters_by_group",
    "compute_imitation_loss",
    "compute_ppo_loss",
    "compute_per_player_round_weights",
    "compute_returns_and_advantages",
    "fit_imitation",
    "fit_ppo",
    "make_default_optimizer",
    "make_default_ppo_optimizer",
    "make_imitation_optimizer_with_info",
    "make_ppo_optimizer_with_info",
    "metrics_to_json",
    "ppo_metrics_to_json",
    "train_imitation_epoch",
    "train_ppo_epoch",
]
