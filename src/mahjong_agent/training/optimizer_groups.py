"""LR groups for Stage03 multi-head training.

Stage02 で有効だった「policy / value_semantic / default の lr を分離する」
optimizer 構成を Stage03Model の module 階層に合わせて移植する。default は
single group (全 parameter を 1 group) で現行挙動と完全互換。

Stage03 の module 階層:

- ``trunk.*``: shared MLP trunk (discard / candidate / value / terminal / yaku
  すべての head が ReLU 経由でここから分岐)。
- ``discard_head.*`` / ``candidate_scorer.*``: policy 側 head。
- ``direct_hint_*`` (opt-in, direct hint branch 有効時のみ): per-tile hint を
  discard logits に直接効かせる tile_embedding / local_scorer / context_gate。
  policy 経路なので policy group に分類する。
- ``value_head.*`` / ``terminal_head.*`` / ``yaku_head.*``: value + semantic 側 head。
- ``semantic_summary_proj.*`` (opt-in, semantic summary injection 有効時のみ):
  policy 経路に summary を入れる射影層。``trunk`` 側に分類する (= 学習速度は
  shared trunk と同じ扱い)。

分類規則 (top-level module 名 → group):

- ``policy``: ``discard_head``, ``candidate_scorer``,
  ``direct_hint_tile_embedding``, ``direct_hint_local_scorer``,
  ``direct_hint_context_gate``
- ``value_semantic``: ``value_head``, ``terminal_head``, ``yaku_head``
- ``trunk``: ``trunk``, ``semantic_summary_proj``
- ``default``: 上記いずれにも該当しない trainable parameter (forward-compat)

default group は将来 head が増えたときの fall-back。Stage03 v1 では普段空に
なる。
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from torch import nn, optim

# top-level module 名による group 分類規則。
# direct hint branch (tile_embedding / local_scorer / context_gate) は
# discard logits に直接寄与する policy 経路なので policy group に入れる。
_POLICY_PREFIXES: tuple[str, ...] = (
    "discard_head",
    "candidate_scorer",
    "direct_hint_tile_embedding",
    "direct_hint_local_scorer",
    "direct_hint_context_gate",
)
_VALUE_SEMANTIC_PREFIXES: tuple[str, ...] = (
    "value_head",
    "terminal_head",
    "yaku_head",
)
_TRUNK_PREFIXES: tuple[str, ...] = ("trunk", "semantic_summary_proj")

_GROUP_NAMES: tuple[str, ...] = ("policy", "value_semantic", "trunk", "default")


@dataclass(frozen=True)
class LRGroupConfig:
    """lr group 設定。

    Attributes
    ----------
    enabled:
        ``False`` (default) で single group optimizer を返す。``True`` で
        Stage02 由来の policy / value_semantic / trunk / default 分離を有効化。
    policy_lr / value_semantic_lr / trunk_lr / default_lr:
        各 group の lr。``None`` で base_lr に倒す。
    policy_weight_decay / value_semantic_weight_decay / trunk_weight_decay /
    default_weight_decay:
        各 group の weight_decay。``None`` で base_weight_decay に倒す。
    """

    enabled: bool = False
    policy_lr: float | None = None
    value_semantic_lr: float | None = None
    trunk_lr: float | None = None
    default_lr: float | None = None
    policy_weight_decay: float | None = None
    value_semantic_weight_decay: float | None = None
    trunk_weight_decay: float | None = None
    default_weight_decay: float | None = None


def _classify_top_level(name: str) -> str:
    """``module.sub.weight`` のような名前を top-level 名で group 分類する。"""
    top = name.split(".", 1)[0]
    if top in _POLICY_PREFIXES:
        return "policy"
    if top in _VALUE_SEMANTIC_PREFIXES:
        return "value_semantic"
    if top in _TRUNK_PREFIXES:
        return "trunk"
    return "default"


def _resolve(value: float | None, fallback: float) -> float:
    return float(fallback if value is None else value)


def build_lr_grouped_optimizer(
    model: nn.Module,
    *,
    base_lr: float,
    base_weight_decay: float = 0.0,
    lr_group_config: LRGroupConfig | None = None,
    optimizer_cls: type[optim.Optimizer] = optim.AdamW,
) -> tuple[optim.Optimizer, dict[str, Any]]:
    """``model.named_parameters()`` を group 分類して optimizer を構築する。

    Parameters
    ----------
    model:
        対象 model (typically ``Stage03Model``)。
    base_lr:
        single group optimizer / 未指定 group の fallback lr。
    base_weight_decay:
        single group optimizer / 未指定 group の fallback weight_decay。
    lr_group_config:
        ``LRGroupConfig``。``None`` または ``enabled=False`` で single group。
    optimizer_cls:
        ``torch.optim.Optimizer`` subclass。default ``AdamW``。

    Returns
    -------
    (optimizer, info):
        ``info`` は JSON serializable diagnostics。
        keys: ``"enabled"``, ``"groups"``。各 group には ``"lr"``,
        ``"weight_decay"``, ``"param_count"``, ``"tensor_count"`` を含む。

    Raises
    ------
    ValueError:
        trainable parameter が 0 件、または分類 group 全部に 1 件も
        parameter が落ちなかった場合。
    """
    cfg = lr_group_config or LRGroupConfig()
    base_lr_f = float(base_lr)
    base_wd_f = float(base_weight_decay)

    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named_trainable:
        raise ValueError(
            "build_lr_grouped_optimizer: model has no trainable parameters"
        )

    if not cfg.enabled:
        opt = optimizer_cls(
            (p for _, p in named_trainable),
            lr=base_lr_f,
            weight_decay=base_wd_f,
        )
        info: dict[str, Any] = {
            "enabled": False,
            "groups": {
                "all": {
                    "lr": base_lr_f,
                    "weight_decay": base_wd_f,
                    "param_count": int(sum(p.numel() for _, p in named_trainable)),
                    "tensor_count": int(len(named_trainable)),
                }
            },
        }
        return opt, info

    # 分類
    buckets: dict[str, list[tuple[str, nn.Parameter]]] = {
        g: [] for g in _GROUP_NAMES
    }
    for n, p in named_trainable:
        buckets[_classify_top_level(n)].append((n, p))

    # group ごとの lr / weight_decay 解決
    lr_map = {
        "policy": _resolve(cfg.policy_lr, base_lr_f),
        "value_semantic": _resolve(cfg.value_semantic_lr, base_lr_f),
        "trunk": _resolve(cfg.trunk_lr, base_lr_f),
        "default": _resolve(cfg.default_lr, base_lr_f),
    }
    wd_map = {
        "policy": _resolve(cfg.policy_weight_decay, base_wd_f),
        "value_semantic": _resolve(cfg.value_semantic_weight_decay, base_wd_f),
        "trunk": _resolve(cfg.trunk_weight_decay, base_wd_f),
        "default": _resolve(cfg.default_weight_decay, base_wd_f),
    }

    param_groups: list[dict[str, Any]] = []
    groups_info: dict[str, dict[str, Any]] = {}
    total_params = 0
    for g in _GROUP_NAMES:
        items = buckets[g]
        param_count = int(sum(p.numel() for _, p in items))
        tensor_count = int(len(items))
        groups_info[g] = {
            "lr": float(lr_map[g]),
            "weight_decay": float(wd_map[g]),
            "param_count": param_count,
            "tensor_count": tensor_count,
        }
        if items:
            param_groups.append(
                {
                    "params": [p for _, p in items],
                    "lr": float(lr_map[g]),
                    "weight_decay": float(wd_map[g]),
                    "name": g,
                }
            )
        total_params += param_count

    if not param_groups:
        raise ValueError(
            "build_lr_grouped_optimizer: no parameters fell into any group "
            "(this should never happen with non-empty model)"
        )

    opt = optimizer_cls(param_groups, lr=base_lr_f, weight_decay=base_wd_f)
    info = {
        "enabled": True,
        "groups": groups_info,
    }
    return opt, info


def classify_parameters_by_group(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> dict[str, list[str]]:
    """diagnostics 用: ``named_parameters`` を group ごとの name list に分類する。"""
    out: dict[str, list[str]] = {g: [] for g in _GROUP_NAMES}
    for n, _ in named_parameters:
        out[_classify_top_level(n)].append(n)
    return out


__all__ = [
    "LRGroupConfig",
    "build_lr_grouped_optimizer",
    "classify_parameters_by_group",
]
