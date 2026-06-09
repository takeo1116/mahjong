"""Stage03 multi-head model v1 tests.

確認内容:
- encoder metadata から model config が作れる
- forward の output shape (discard / value / terminal / yaku)
- discard mask の適用 (illegal idx の logit が softmax 後 ~0 になる)
- candidate scorer の output shape
- candidate count 0 でも crash しない
- batch size 1 / >1 両方で動く
- 実 encoder の出力を model に通す CPU smoke
- source に repo 外 local docs / issue ID 参照が混入していないこと
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# config from encoder metadata
# ---------------------------------------------------------------------------


def test_model_config_from_encoder_metadata():
    from mahjong_agent.encoders import PublicObservationEncoder
    from mahjong_agent.models import Stage03ModelConfig

    enc = PublicObservationEncoder()
    cfg = Stage03ModelConfig.from_encoder_metadata(enc.metadata())
    assert cfg.observation_dim == enc.metadata().observation_dim
    assert cfg.candidate_dim == enc.metadata().candidate_dim
    assert cfg.num_terminal_classes == 5
    assert cfg.num_yaku == 49
    assert cfg.num_tile_types == 34


def test_model_config_overrides():
    from mahjong_agent.encoders import PublicObservationEncoder
    from mahjong_agent.models import Stage03ModelConfig

    enc = PublicObservationEncoder()
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(),
        hidden_dim=64,
        trunk_layers=1,
        candidate_hidden_dim=32,
        dropout=0.1,
    )
    assert cfg.hidden_dim == 64
    assert cfg.trunk_layers == 1
    assert cfg.candidate_hidden_dim == 32
    assert cfg.dropout == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# main forward shapes
# ---------------------------------------------------------------------------


def _make_model(batch_size=2, hidden_dim=32, trunk_layers=1):
    """軽量 model を構築する helper。"""
    from mahjong_agent.encoders import PublicObservationEncoder
    from mahjong_agent.models import Stage03Model, Stage03ModelConfig

    enc = PublicObservationEncoder()
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(),
        hidden_dim=hidden_dim,
        trunk_layers=trunk_layers,
        candidate_hidden_dim=hidden_dim,
    )
    model = Stage03Model(cfg)
    obs = torch.zeros(batch_size, cfg.observation_dim, dtype=torch.float32)
    return enc, cfg, model, obs


def test_forward_returns_expected_shapes_batch_one():
    _, cfg, model, _ = _make_model(batch_size=1)
    obs = torch.zeros(1, cfg.observation_dim, dtype=torch.float32)
    out = model(obs)
    assert out.discard_logits.shape == (1, cfg.num_tile_types)
    assert out.value.shape == (1,)
    assert out.terminal_logits.shape == (1, cfg.num_terminal_classes)
    assert out.yaku_logits.shape == (1, cfg.num_yaku)


def test_forward_returns_expected_shapes_batch_many():
    _, cfg, model, obs = _make_model(batch_size=8)
    out = model(obs)
    assert out.discard_logits.shape == (8, 34)
    assert out.value.shape == (8,)
    assert out.terminal_logits.shape == (8, 5)
    assert out.yaku_logits.shape == (8, 49)


def test_forward_dtype_is_float32_on_cpu():
    _, cfg, model, obs = _make_model(batch_size=2)
    out = model(obs)
    assert out.discard_logits.dtype == torch.float32
    assert out.value.dtype == torch.float32
    assert out.terminal_logits.dtype == torch.float32
    assert out.yaku_logits.dtype == torch.float32
    assert out.discard_logits.device.type == "cpu"


def test_forward_rejects_wrong_obs_dim():
    _, cfg, model, _ = _make_model()
    bad = torch.zeros(2, cfg.observation_dim + 1, dtype=torch.float32)
    with pytest.raises(ValueError):
        model(bad)


def test_forward_rejects_1d_obs():
    _, cfg, model, _ = _make_model()
    bad = torch.zeros(cfg.observation_dim, dtype=torch.float32)
    with pytest.raises(ValueError):
        model(bad)


# ---------------------------------------------------------------------------
# discard mask
# ---------------------------------------------------------------------------


def test_discard_mask_zeroes_softmax_for_illegal_indices():
    _, cfg, model, obs = _make_model(batch_size=2)
    mask = torch.zeros(2, cfg.num_tile_types, dtype=torch.float32)
    mask[:, [0, 5, 33]] = 1.0  # 3 牌だけ legal
    out = model(obs, discard_mask=mask)
    probs = torch.softmax(out.discard_logits, dim=-1)
    # legal idx の合計 ≈ 1.0、illegal idx は ~ 0
    legal_sum = probs[:, [0, 5, 33]].sum(dim=-1)
    illegal_max = probs[:, [1, 2, 3, 4, 6, 32]].max(dim=-1).values
    assert torch.allclose(legal_sum, torch.ones(2), atol=1e-5)
    assert (illegal_max < 1e-6).all()


def test_discard_mask_accepts_bool_dtype():
    _, cfg, model, obs = _make_model(batch_size=2)
    mask = torch.zeros(2, cfg.num_tile_types, dtype=torch.bool)
    mask[:, 0] = True
    out = model(obs, discard_mask=mask)
    # mask 通すと illegal idx の logit が -1e9 相当
    assert out.discard_logits[:, 1].max().item() < -1e8


def test_discard_mask_shape_mismatch_raises():
    _, cfg, model, obs = _make_model(batch_size=2)
    mask = torch.zeros(2, 33, dtype=torch.float32)  # 33 != 34
    with pytest.raises(ValueError):
        model(obs, discard_mask=mask)


def test_discard_mask_none_passes_through():
    _, cfg, model, obs = _make_model(batch_size=2)
    out_no_mask = model(obs)
    # raw logits を softmax したら全 34 idx に確率が分散
    probs = torch.softmax(out_no_mask.discard_logits, dim=-1)
    assert (probs > 0.0).all()


# ---------------------------------------------------------------------------
# candidate scorer
# ---------------------------------------------------------------------------


def test_candidate_scorer_basic_shape():
    _, cfg, model, obs = _make_model(batch_size=3)
    C = 5
    cand = torch.zeros(3, C, cfg.candidate_dim, dtype=torch.float32)
    out = model.score_candidates(obs, cand)
    assert out.candidate_scores.shape == (3, C)
    assert out.candidate_scores.dtype == torch.float32


def test_candidate_scorer_zero_candidates():
    _, cfg, model, obs = _make_model(batch_size=2)
    cand = torch.zeros(2, 0, cfg.candidate_dim, dtype=torch.float32)
    out = model.score_candidates(obs, cand)
    assert out.candidate_scores.shape == (2, 0)


def test_candidate_scorer_variable_count_smoke():
    """C=1 / C=3 / C=10 で順に評価しても crash しない。"""
    _, cfg, model, obs = _make_model(batch_size=2)
    for C in (1, 3, 10):
        cand = torch.zeros(2, C, cfg.candidate_dim, dtype=torch.float32)
        out = model.score_candidates(obs, cand)
        assert out.candidate_scores.shape == (2, C)


def test_candidate_scorer_rejects_wrong_candidate_dim():
    _, cfg, model, obs = _make_model(batch_size=2)
    cand = torch.zeros(2, 3, cfg.candidate_dim + 1, dtype=torch.float32)
    with pytest.raises(ValueError):
        model.score_candidates(obs, cand)


def test_candidate_scorer_rejects_batch_mismatch():
    _, cfg, model, obs = _make_model(batch_size=2)
    cand = torch.zeros(3, 5, cfg.candidate_dim, dtype=torch.float32)
    with pytest.raises(ValueError):
        model.score_candidates(obs, cand)


def test_candidate_scorer_rejects_2d_input():
    _, cfg, model, obs = _make_model(batch_size=2)
    bad = torch.zeros(2, cfg.candidate_dim, dtype=torch.float32)
    with pytest.raises(ValueError):
        model.score_candidates(obs, bad)


# ---------------------------------------------------------------------------
# real encoder -> model smoke
# ---------------------------------------------------------------------------


def test_real_encoder_to_model_cpu_smoke():
    """RiichiEnv reset → encoder → model forward + candidate scoring smoke。"""
    import riichienv

    from mahjong_agent.actions import legal_actions_to_model_set
    from mahjong_agent.encoders import PublicObservationEncoder
    from mahjong_agent.models import Stage03Model, Stage03ModelConfig

    enc = PublicObservationEncoder()
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=32
    )
    model = Stage03Model(cfg).eval()

    env = riichienv.RiichiEnv(riichienv.GameType.YON_TONPUSEN)
    env.reset(seed=42)
    cp = env.current_player
    obs = env.get_observation(cp)
    legal_set = legal_actions_to_model_set(
        list(obs.legal_actions()),
        actor=cp,
        num_players=env.num_players,
        env_for_riichi=env,
    )

    obs_np = enc.encode_observation(obs)
    mask_np = enc.discard_legal_mask(legal_set)
    cand_np = enc.encode_candidates(legal_set)

    obs_t = torch.from_numpy(obs_np).unsqueeze(0)            # (1, obs_dim)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0)          # (1, 34)
    cand_t = torch.from_numpy(
        cand_np.reshape(1, *cand_np.shape) if cand_np.ndim == 2
        else cand_np
    )                                                          # (1, C, cand_dim)

    with torch.no_grad():
        out = model(obs_t, discard_mask=mask_t)
        cand_out = model.score_candidates(obs_t, cand_t)
    assert out.discard_logits.shape == (1, 34)
    assert out.value.shape == (1,)
    assert out.terminal_logits.shape == (1, 5)
    assert out.yaku_logits.shape == (1, 49)
    # mask が効いていれば、legal な tile_type の softmax 合計が ~1
    probs = torch.softmax(out.discard_logits, dim=-1)
    legal_indices = np.where(mask_np > 0.5)[0]
    legal_sum = probs[0, legal_indices].sum().item()
    assert abs(legal_sum - 1.0) < 1e-5
    # candidate output shape
    assert cand_out.candidate_scores.shape == (1, cand_np.shape[0])


# ---------------------------------------------------------------------------
# direct hint branch (Stage02 parity, opt-in)
# ---------------------------------------------------------------------------


def test_stage03_model_config_direct_hint_default_on():
    """default config + default encoder metadata で direct hint branch が有効。"""
    from mahjong_agent.encoders import PublicObservationEncoder
    from mahjong_agent.models import Stage03Model, Stage03ModelConfig

    enc = PublicObservationEncoder()
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16
    )
    assert cfg.direct_hint_in_policy is True
    assert len(cfg.direct_hint_ranges) >= 1
    model = Stage03Model(cfg).eval()
    assert hasattr(model, "direct_hint_local_scorer")
    out = model(torch.randn(2, cfg.observation_dim))
    assert out.discard_logits.shape == (2, 34)


def test_stage03_model_config_can_disable_direct_hint():
    """``direct_hint_in_policy=False`` を明示すると branch 無し (旧 architecture)。"""
    from mahjong_agent.encoders import PublicObservationEncoder
    from mahjong_agent.models import Stage03Model, Stage03ModelConfig

    enc = PublicObservationEncoder()
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16,
        direct_hint_in_policy=False,
    )
    assert cfg.direct_hint_in_policy is False
    assert cfg.direct_hint_ranges == ()
    model = Stage03Model(cfg).eval()
    assert not hasattr(model, "direct_hint_local_scorer")
    out = model(torch.randn(2, cfg.observation_dim))
    assert out.discard_logits.shape == (2, 34)


def test_direct_hint_with_legacy_encoder_metadata_does_not_crash():
    """``enable_hints=False`` の legacy metadata では per-tile hint range が
    無いため direct hint branch は空 range で無効化され、forward が通る。"""
    from mahjong_agent.encoders import PublicObservationEncoder
    from mahjong_agent.models import Stage03Model, Stage03ModelConfig

    enc = PublicObservationEncoder(enable_hints=False)
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16
    )
    # default on でも 34 幅 hint が無いので ranges は空 → branch 無効。
    assert cfg.direct_hint_in_policy is True
    assert cfg.direct_hint_ranges == ()
    model = Stage03Model(cfg).eval()
    assert not hasattr(model, "direct_hint_local_scorer")
    out = model(torch.randn(2, cfg.observation_dim))
    assert out.discard_logits.shape == (2, 34)


def test_direct_hint_params_are_policy_lr_group():
    """direct hint branch の parameter が lr group の policy に入り、
    全 parameter が exactly once で分類されること。"""
    from mahjong_agent.encoders import PublicObservationEncoder
    from mahjong_agent.models import Stage03Model, Stage03ModelConfig
    from mahjong_agent.training.optimizer_groups import (
        LRGroupConfig,
        build_lr_grouped_optimizer,
        classify_parameters_by_group,
    )

    enc = PublicObservationEncoder()
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16
    )
    model = Stage03Model(cfg)
    assert hasattr(model, "direct_hint_local_scorer")

    groups = classify_parameters_by_group(model.named_parameters())
    direct_hint_names = [
        n for n, _ in model.named_parameters() if n.startswith("direct_hint_")
    ]
    assert direct_hint_names  # 存在する
    # すべて policy group に入り、default には落ちない
    for n in direct_hint_names:
        assert n in groups["policy"]
        assert n not in groups["default"]

    # exactly-once invariant: 全 trainable param が 1 group に厳密に 1 回
    all_names = [m for lst in groups.values() for m in lst]
    expected = [n for n, p in model.named_parameters() if p.requires_grad]
    assert sorted(all_names) == sorted(expected)
    assert len(all_names) == len(set(all_names))

    # param_count にも反映される
    _opt, info = build_lr_grouped_optimizer(
        model, base_lr=1e-3, lr_group_config=LRGroupConfig(enabled=True)
    )
    expected_total = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    actual_total = sum(g["param_count"] for g in info["groups"].values())
    assert actual_total == expected_total
    assert info["groups"]["policy"]["param_count"] > 0


def test_direct_hint_from_encoder_metadata_picks_per_tile_ranges():
    from mahjong_agent.encoders import PublicObservationEncoder
    from mahjong_agent.models import Stage03ModelConfig

    enc = PublicObservationEncoder()
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(), direct_hint_in_policy=True
    )
    names = [n for n, _s, _e in cfg.direct_hint_ranges]
    assert "shanten_delta_per_discard" in names
    assert "discard_ukeire_per_tile" in names
    # defensive direct hints (ISSUE-0022) も default で含まれる
    assert "safe_vs_all_riichi_mask" in names
    assert "suji_vs_all_riichi_mask" in names
    assert "kabe_suji_mask" in names
    for _n, s, e in cfg.direct_hint_ranges:
        assert e - s == 34


def test_direct_hint_forward_smoke_and_no_illegal_leak():
    from mahjong_agent.encoders import PublicObservationEncoder
    from mahjong_agent.models import Stage03Model, Stage03ModelConfig

    torch.manual_seed(0)
    enc = PublicObservationEncoder()
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(), hidden_dim=32, trunk_layers=1, candidate_hidden_dim=16,
        direct_hint_in_policy=True,
    )
    model = Stage03Model(cfg).eval()
    assert hasattr(model, "direct_hint_local_scorer")
    obs = torch.randn(4, cfg.observation_dim)
    dmask = torch.ones(4, 34)
    dmask[:, :12] = 0.0  # illegal
    with torch.no_grad():
        out = model(obs, discard_mask=dmask)
    assert out.discard_logits.shape == (4, 34)
    # direct hint delta は mask 適用前に加算されるので illegal idx は -1e9 のまま
    assert float(out.discard_logits[:, :12].max()) < -1e8
    # legal idx は finite で softmax 合計 ~1
    probs = torch.softmax(out.discard_logits, dim=-1)
    assert torch.isfinite(probs).all()
    assert abs(float(probs[:, 12:].sum(dim=-1).mean()) - 1.0) < 1e-5


def test_direct_hint_on_off_differ_with_same_weights_seeded():
    """direct hint on の方が discard logits に delta が乗り、off と異なること。"""
    from mahjong_agent.encoders import PublicObservationEncoder
    from mahjong_agent.models import Stage03Model, Stage03ModelConfig

    enc = PublicObservationEncoder()
    meta = enc.metadata()
    torch.manual_seed(1)
    cfg_off = Stage03ModelConfig.from_encoder_metadata(
        meta, hidden_dim=16, trunk_layers=1, candidate_hidden_dim=8
    )
    model_off = Stage03Model(cfg_off).eval()
    cfg_on = Stage03ModelConfig.from_encoder_metadata(
        meta, hidden_dim=16, trunk_layers=1, candidate_hidden_dim=8,
        direct_hint_in_policy=True,
    )
    model_on = Stage03Model(cfg_on).eval()
    # shared part (trunk / discard_head) の重みを揃えて delta の効果だけを見る
    model_on.trunk.load_state_dict(model_off.trunk.state_dict())
    model_on.discard_head.load_state_dict(model_off.discard_head.state_dict())
    obs = torch.randn(2, meta.observation_dim)
    with torch.no_grad():
        d_off = model_off(obs).discard_logits
        d_on = model_on(obs).discard_logits
    # direct hint delta が非ゼロなら off/on は一致しない
    assert not torch.allclose(d_off, d_on)


def test_direct_hint_candidate_scorer_unaffected():
    """direct hint は discard 専用。candidate scorer の出力 shape は不変。"""
    from mahjong_agent.encoders import PublicObservationEncoder
    from mahjong_agent.models import Stage03Model, Stage03ModelConfig

    enc = PublicObservationEncoder()
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(), hidden_dim=16, trunk_layers=1, candidate_hidden_dim=8,
        direct_hint_in_policy=True,
    )
    model = Stage03Model(cfg).eval()
    obs = torch.randn(2, cfg.observation_dim)
    cand = torch.randn(2, 5, cfg.candidate_dim)
    with torch.no_grad():
        out = model.score_candidates(obs, cand)
    assert out.candidate_scores.shape == (2, 5)


# ---------------------------------------------------------------------------
# repo-external local docs / issue ID references must not leak
# ---------------------------------------------------------------------------


def test_models_source_does_not_reference_local_docs_or_issues():
    src_dir = Path(__file__).resolve().parent.parent / "src" / "mahjong_agent"
    forbidden_patterns = [
        r"PROJECT_RULE",
        r"ISSUE_BOARD",
        r"\bISSUE-\d",
        r"\bISSUE-L\d",
        r"\bAGENTS\.md",
        r"\bCLAUDE\.md",
        r"\bmajong-rl",
        r"CHANGE_QUEUE",
    ]
    pattern = re.compile("|".join(forbidden_patterns))
    offenders = []
    for path in src_dir.rglob("*.py"):
        text = path.read_text()
        for m in pattern.finditer(text):
            offenders.append((str(path.relative_to(src_dir.parent.parent)), m.group(0)))
    assert offenders == [], (
        f"package source references workspace-local docs / issue IDs: "
        f"{offenders}"
    )
