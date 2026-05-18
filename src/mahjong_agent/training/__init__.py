"""Imitation / PPO training loops."""
from __future__ import annotations

from mahjong_agent.training.imitation import (
    ImitationConfig,
    ImitationMetrics,
    ImitationRunResult,
    compute_imitation_loss,
    fit_imitation,
    make_default_optimizer,
    metrics_to_json,
    train_imitation_epoch,
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
    ppo_metrics_to_json,
    train_ppo_epoch,
)

__all__ = [
    "ImitationConfig",
    "ImitationMetrics",
    "ImitationRunResult",
    "PPOBatch",
    "PPOConfig",
    "PPOMetrics",
    "PPORunResult",
    "PPOTrainingData",
    "compute_imitation_loss",
    "compute_ppo_loss",
    "compute_returns_and_advantages",
    "fit_imitation",
    "fit_ppo",
    "make_default_optimizer",
    "make_default_ppo_optimizer",
    "metrics_to_json",
    "ppo_metrics_to_json",
    "train_imitation_epoch",
    "train_ppo_epoch",
]
