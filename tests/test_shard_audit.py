"""Tests for ``mahjong_agent.diagnostics.shard_audit``.

- 合成 sample で `audit_decision_samples` の counts / distribution が正しいこと。
- teacher 情報の有無で crash しないこと。
- model 渡し / 非渡しでの挙動差。
- shard roundtrip 経路。
- JSON / Markdown serialization。
- hidden info / local docs guard。
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mahjong_agent.actions.types import ActionFamily
from mahjong_agent.data import (
    SCHEMA_VERSION,
    DecisionSample,
    write_decision_shard,
)
from mahjong_agent.diagnostics import (
    ShardAuditConfig,
    ShardAuditSummary,
    audit_decision_samples,
    audit_decision_shard,
    summary_to_json,
    summary_to_markdown,
)
from mahjong_agent.models import Stage03Model, Stage03ModelConfig
from mahjong_agent.targets.yaku import NUM_YAKU

_OBS_DIM = 16
_CAND_DIM = 6


def _make_discard_sample(
    *,
    tile_type: int = 5,
    legal_tiles: tuple[int, ...] = (3, 5, 7),
    episode_id: str = "ep",
    round_id: int = 0,
    step_id: int = 0,
    player_id: int = 0,
    actor_type: str = "policy",
    teacher_tt: int = -1,
    teacher_best: tuple[int, ...] = (),
    metadata: dict | None = None,
    terminal_class: int = -1,
    yaku_indices: tuple[int, ...] = (),
    han: int = -1,
    fu: int = -1,
    score_delta: int = 0,
    reward: float = 0.0,
) -> DecisionSample:
    obs = np.full(_OBS_DIM, 1.0, dtype=np.float32)
    mask = np.zeros(34, dtype=np.float32)
    for t in legal_tiles:
        mask[int(t)] = 1.0
    tbm = np.zeros(34, dtype=np.float32)
    for t in teacher_best:
        tbm[int(t)] = 1.0
    yt = np.zeros(NUM_YAKU, dtype=np.float32)
    for y in yaku_indices:
        yt[int(y)] = 1.0
    return DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        round_id=int(round_id),
        step_id=int(step_id),
        player_id=int(player_id),
        decision_family=ActionFamily.NORMAL_DISCARD.value,
        actor_type=actor_type,
        observation=obs,
        discard_mask=mask,
        candidate_features=np.zeros((0, _CAND_DIM), dtype=np.float32),
        selected_discard_tile_type=int(tile_type),
        selected_candidate_index=-1,
        yaku_target=yt,
        yaku_loss_mask=1.0 if yaku_indices else 0.0,
        han=int(han),
        fu=int(fu),
        score_delta=int(score_delta),
        reward=float(reward),
        terminal_class=int(terminal_class),
        teacher_discard_tile_type=int(teacher_tt),
        teacher_best_mask=tbm,
        teacher_available=bool(teacher_tt >= 0 or len(teacher_best) > 0),
        metadata=dict(metadata or {}),
    )


def _make_candidate_sample(
    *,
    family: ActionFamily = ActionFamily.PASS,
    num_cands: int = 3,
    selected_idx: int = 1,
    actor_type: str = "policy",
    teacher_ci: int = -1,
    episode_id: str = "ep",
    round_id: int = 0,
    step_id: int = 0,
    player_id: int = 0,
) -> DecisionSample:
    obs = np.full(_OBS_DIM, 0.5, dtype=np.float32)
    cand = np.zeros((num_cands, _CAND_DIM), dtype=np.float32)
    for i in range(num_cands):
        cand[i, i % _CAND_DIM] = 1.0
    return DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        round_id=int(round_id),
        step_id=int(step_id),
        player_id=int(player_id),
        decision_family=family.value,
        actor_type=actor_type,
        observation=obs,
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=cand,
        selected_discard_tile_type=-1,
        selected_candidate_index=int(selected_idx),
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
        teacher_candidate_index=int(teacher_ci),
        teacher_available=bool(teacher_ci >= 0),
    )


def _build_model() -> Stage03Model:
    cfg = Stage03ModelConfig(
        observation_dim=_OBS_DIM,
        candidate_dim=_CAND_DIM,
        hidden_dim=32,
        trunk_layers=1,
        candidate_hidden_dim=16,
    )
    return Stage03Model(cfg)


# ----------------------------------------------------------------------
# basic counts / distribution
# ----------------------------------------------------------------------


def test_audit_does_not_crash_on_empty_list():
    summary = audit_decision_samples([])
    assert isinstance(summary, ShardAuditSummary)
    assert summary.num_samples == 0
    assert summary.discard_count == 0
    assert summary.model_evaluated is False


def test_audit_counts_discard_vs_candidate_correctly():
    samples = [
        _make_discard_sample(tile_type=5, step_id=0),
        _make_discard_sample(tile_type=7, step_id=1),
        _make_candidate_sample(
            family=ActionFamily.PASS, num_cands=3, selected_idx=1, step_id=2
        ),
        _make_candidate_sample(
            family=ActionFamily.PON, num_cands=2, selected_idx=0, step_id=3
        ),
    ]
    summary = audit_decision_samples(samples)
    assert summary.num_samples == 4
    assert summary.discard_count == 2
    assert summary.candidate_count == 2
    fc = summary.decision_family_counts
    assert fc[ActionFamily.NORMAL_DISCARD.value] == 2
    assert fc[ActionFamily.PASS.value] == 1
    assert fc[ActionFamily.PON.value] == 1


def test_audit_actor_type_and_player_counts():
    samples = [
        _make_discard_sample(actor_type="policy", player_id=0, step_id=0),
        _make_discard_sample(actor_type="rule_based", player_id=1, step_id=1),
        _make_discard_sample(actor_type="rule_based", player_id=1, step_id=2),
    ]
    summary = audit_decision_samples(samples)
    assert summary.actor_type_counts == {"policy": 1, "rule_based": 2}
    assert summary.player_id_counts == {0: 1, 1: 2}


def test_audit_candidate_count_distribution():
    samples = [
        _make_candidate_sample(num_cands=2, selected_idx=0, step_id=0),
        _make_candidate_sample(num_cands=4, selected_idx=2, step_id=1),
        _make_candidate_sample(num_cands=4, selected_idx=3, step_id=2),
    ]
    summary = audit_decision_samples(samples)
    stats = summary.candidate_count_distribution
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx((2 + 4 + 4) / 3.0)
    assert stats["max"] == pytest.approx(4.0)


def test_audit_handles_candidate_count_zero_sample():
    """C=0 の candidate sample (例: pass 候補が無い) でも crash しない。"""
    s = DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id="ep",
        decision_family=ActionFamily.PASS.value,
        observation=np.zeros(_OBS_DIM, dtype=np.float32),
        candidate_features=np.zeros((0, 0), dtype=np.float32),
        selected_candidate_index=-1,
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
    )
    summary = audit_decision_samples([s])
    assert summary.num_samples == 1
    assert summary.candidate_count == 1


# ----------------------------------------------------------------------
# teacher diagnostics
# ----------------------------------------------------------------------


def test_audit_teacher_best_mask_agreement():
    """selected が teacher_best_mask に含まれるかが正しく計算される。"""
    samples = [
        _make_discard_sample(
            tile_type=5,
            teacher_best=(5, 7),  # selected 5 ∈ {5,7} → hit
            step_id=0,
        ),
        _make_discard_sample(
            tile_type=3,
            teacher_best=(5, 7),  # selected 3 ∉ {5,7} → miss
            step_id=1,
        ),
        _make_discard_sample(
            tile_type=5,
            teacher_best=(5,),  # selected 5 ∈ {5} → hit
            step_id=2,
        ),
        _make_discard_sample(
            tile_type=5,
            teacher_best=(),  # mask empty → not counted
            step_id=3,
        ),
    ]
    summary = audit_decision_samples(samples)
    t = summary.teacher
    assert t["teacher_best_mask_nonzero_count"] == 3
    assert t["selected_in_teacher_best_mask_total"] == 3
    assert t["selected_in_teacher_best_mask_count"] == 2
    assert t["selected_in_teacher_best_mask_rate"] == pytest.approx(2 / 3)


def test_audit_teacher_discard_tile_type_agreement():
    samples = [
        _make_discard_sample(
            tile_type=5, teacher_tt=5, step_id=0
        ),  # agree
        _make_discard_sample(
            tile_type=5, teacher_tt=7, step_id=1
        ),  # disagree
        _make_discard_sample(
            tile_type=7, teacher_tt=-1, step_id=2
        ),  # no teacher → not counted
    ]
    summary = audit_decision_samples(samples)
    t = summary.teacher
    assert t["teacher_discard_agreement_total"] == 2
    assert t["teacher_discard_agreement_count"] == 1
    assert t["teacher_discard_agreement_rate"] == pytest.approx(0.5)


def test_audit_post_riichi_count_picked_up():
    samples = [
        _make_discard_sample(tile_type=5, step_id=0),
        _make_discard_sample(
            tile_type=5,
            step_id=1,
            metadata={"is_post_riichi_discard": True},
        ),
        _make_discard_sample(
            tile_type=3,
            step_id=2,
            metadata={"is_post_riichi_discard": True},
        ),
    ]
    summary = audit_decision_samples(samples)
    assert summary.teacher["post_riichi_discard_count"] == 2


def test_audit_handles_no_teacher_info():
    samples = [
        _make_discard_sample(tile_type=5, step_id=0, teacher_tt=-1),
        _make_discard_sample(tile_type=7, step_id=1, teacher_tt=-1),
    ]
    summary = audit_decision_samples(samples)
    assert summary.teacher["teacher_available_count"] == 0
    assert summary.teacher["teacher_discard_agreement_total"] == 0
    assert summary.teacher["selected_in_teacher_best_mask_total"] == 0


# ----------------------------------------------------------------------
# family audit
# ----------------------------------------------------------------------


def test_family_audit_counts_per_family():
    samples = [
        _make_discard_sample(actor_type="policy", step_id=0),
        _make_discard_sample(actor_type="rule_based", step_id=1),
        _make_candidate_sample(
            family=ActionFamily.PASS, actor_type="policy", step_id=2
        ),
        _make_candidate_sample(
            family=ActionFamily.PON, actor_type="rule_based", step_id=3
        ),
    ]
    summary = audit_decision_samples(samples)
    fa = summary.family_audit
    assert fa[ActionFamily.NORMAL_DISCARD.value]["count"] == 2
    assert fa[ActionFamily.PASS.value]["count"] == 1
    assert fa[ActionFamily.PON.value]["count"] == 1
    nd = fa[ActionFamily.NORMAL_DISCARD.value]
    assert nd["actor_type_counts"] == {"policy": 1, "rule_based": 1}


def test_family_audit_teacher_agreement_per_family():
    samples = [
        _make_discard_sample(tile_type=5, teacher_tt=5, step_id=0),  # agree
        _make_discard_sample(tile_type=3, teacher_tt=5, step_id=1),  # disagree
        _make_candidate_sample(
            family=ActionFamily.PASS,
            selected_idx=1,
            teacher_ci=1,
            step_id=2,
        ),  # agree
    ]
    summary = audit_decision_samples(samples)
    nd = summary.family_audit[ActionFamily.NORMAL_DISCARD.value]
    assert nd["teacher_agreement_total"] == 2
    assert nd["teacher_agreement_count"] == 1
    pass_fam = summary.family_audit[ActionFamily.PASS.value]
    assert pass_fam["teacher_agreement_total"] == 1
    assert pass_fam["teacher_agreement_count"] == 1


# ----------------------------------------------------------------------
# shard roundtrip
# ----------------------------------------------------------------------


def test_audit_decision_shard_roundtrip(tmp_path: Path):
    samples = [
        _make_discard_sample(tile_type=5, step_id=i) for i in range(4)
    ]
    samples += [
        _make_candidate_sample(num_cands=2, selected_idx=0, step_id=10 + i)
        for i in range(2)
    ]
    path = tmp_path / "shard.npz"
    write_decision_shard(path, samples)
    summary = audit_decision_shard(path)
    assert summary.num_samples == 6
    assert summary.discard_count == 4
    assert summary.candidate_count == 2


# ----------------------------------------------------------------------
# model-based diagnostics
# ----------------------------------------------------------------------


def test_audit_without_model_sets_flag_and_skips_policy():
    samples = [_make_discard_sample(tile_type=5)]
    summary = audit_decision_samples(samples)
    assert summary.model_evaluated is False
    assert summary.policy == {}


def test_audit_with_model_emits_entropy_max_prob_selected():
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(tile_type=5, step_id=0),
        _make_discard_sample(tile_type=7, step_id=1),
        _make_candidate_sample(num_cands=3, selected_idx=1, step_id=2),
    ]
    summary = audit_decision_samples(samples, model=model)
    assert summary.model_evaluated is True
    p = summary.policy
    assert "discard_entropy_stats" in p
    assert "candidate_entropy_stats" in p
    assert "selected_prob_stats" in p
    assert p["samples_evaluated"] == 3
    assert p["illegal_prob_ok"] is True


def test_audit_with_model_family_audit_has_policy_stats():
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(tile_type=5, step_id=0),
        _make_discard_sample(tile_type=7, step_id=1),
    ]
    summary = audit_decision_samples(samples, model=model)
    nd = summary.family_audit[ActionFamily.NORMAL_DISCARD.value]
    assert "policy_entropy_stats" in nd
    assert "policy_max_prob_stats" in nd
    assert "policy_selected_prob_stats" in nd


def test_audit_candidate_abandon_call_diagnostic():
    """candidate family の family_audit に call 放棄率 diagnostic が出て、
    JSON serializable であること。"""
    import json

    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_candidate_sample(
            family=ActionFamily.CHI, num_cands=3, selected_idx=1, step_id=0
        ),
        _make_candidate_sample(
            family=ActionFamily.PON, num_cands=2, selected_idx=0, step_id=1
        ),
    ]
    summary = audit_decision_samples(samples, model=model)
    chi = summary.family_audit[ActionFamily.CHI.value]
    assert "policy_abandon_call_count" in chi
    assert "policy_abandon_call_total" in chi
    assert "policy_abandon_call_rate" in chi
    assert "policy_pred_teacher_candidate_rate" in chi
    assert chi["policy_abandon_call_total"] == 1
    assert 0.0 <= chi["policy_abandon_call_rate"] <= 1.0
    # NORMAL_DISCARD family には abandon 系を出さない
    json.loads(summary_to_json(summary))


def test_audit_with_model_handles_candidate_count_zero_sample():
    """candidate family かつ ``candidate_features.shape[0]==0`` の sample が
    含まれていても model あり audit が crash しない (regression for follow-up #1)。"""
    torch.manual_seed(0)
    model = _build_model()
    # cmax=0 の "PASS" sample (selected_candidate_index=-1)
    cmax0_sample = DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id="ep",
        decision_family=ActionFamily.PASS.value,
        observation=np.zeros(_OBS_DIM, dtype=np.float32),
        candidate_features=np.zeros((0, _CAND_DIM), dtype=np.float32),
        selected_candidate_index=-1,
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
    )
    samples = [_make_discard_sample(tile_type=5, step_id=0), cmax0_sample]
    summary = audit_decision_samples(samples, model=model)
    assert summary.model_evaluated is True
    # samples_evaluated には cmax=0 sample も含まれる
    assert summary.policy["samples_evaluated"] == 2
    # candidate entropy / max_prob は 0.0 fallback で finite
    cstats = summary.policy["candidate_entropy_stats"]
    assert cstats["count"] == 1
    assert cstats["mean"] == pytest.approx(0.0)
    # family_audit に PASS family は count=1 で残る
    pass_fam = summary.family_audit[ActionFamily.PASS.value]
    assert pass_fam["count"] == 1
    # cmax=0 sample の policy_selected_prob は 0.0
    sp_stats = pass_fam["policy_selected_prob_stats"]
    # count==1 で mean==0.0 (selected_idx=-1 で 0.0 を埋めるため)
    assert sp_stats["count"] == 1
    assert sp_stats["mean"] == pytest.approx(0.0)


def test_audit_selected_prob_uses_combined_softmax():
    """discard / candidate sample の selected_prob が
    [discard_logits, candidate_scores] の combined softmax 上の確率に一致する
    (regression for follow-up #2)。"""
    torch.manual_seed(0)
    model = _build_model()
    # discard sample (selected tile=5、legal=3,5,7)
    discard_sample = _make_discard_sample(
        tile_type=5,
        legal_tiles=(3, 5, 7),
        step_id=0,
        teacher_tt=5,
        teacher_best=(5, 7),
    )
    # candidate sample (3 candidates, selected idx=1)
    candidate_sample = _make_candidate_sample(
        num_cands=3, selected_idx=1, step_id=1
    )
    samples = [discard_sample, candidate_sample]
    # 1 つの batch にまとめて model に渡し、手動で combined softmax を作って
    # 期待 selected probability を出す。
    from mahjong_agent.data.collate import collate_decision_samples

    batch = collate_decision_samples(samples)
    model.eval()
    with torch.no_grad():
        fwd = model(
            batch.observation.float(),
            discard_mask=batch.discard_mask.float(),
        )
        cscore = model.score_candidates(
            batch.observation.float(), batch.candidate_features.float()
        )
        cmax = int(cscore.candidate_scores.size(1))
        cand_mask = batch.candidate_mask.float()
        if cmax > 0:
            masked_cand = cscore.candidate_scores + (1.0 - cand_mask) * -1.0e9
        else:
            masked_cand = cscore.candidate_scores
        combined = torch.cat([fwd.discard_logits, masked_cand], dim=-1)
        probs = torch.softmax(combined, dim=-1)
    expected_discard_sel_prob = float(probs[0, 5].item())
    expected_candidate_sel_prob = float(probs[1, 34 + 1].item())

    summary = audit_decision_samples(samples, model=model)
    # selected_prob_stats は count=2 の combined softmax probability を集約。
    # 個別値は per_sample 経由で family_audit から取れる。
    nd = summary.family_audit[ActionFamily.NORMAL_DISCARD.value]
    pass_fam = summary.family_audit[ActionFamily.PASS.value]
    # 各 family の policy_selected_prob_stats.mean は 1 sample しかないので
    # その sample の selected_prob と一致する。
    assert nd["policy_selected_prob_stats"]["mean"] == pytest.approx(
        expected_discard_sel_prob, abs=1e-6
    )
    assert pass_fam["policy_selected_prob_stats"]["mean"] == pytest.approx(
        expected_candidate_sel_prob, abs=1e-6
    )
    # top-level selected_prob_stats も同じく combined 確率の集約。
    top = summary.policy["selected_prob_stats"]
    assert top["count"] == 2
    assert top["mean"] == pytest.approx(
        (expected_discard_sel_prob + expected_candidate_sel_prob) / 2.0,
        abs=1e-6,
    )


def test_audit_teacher_best_mass_is_combined_probability():
    """teacher_best_mass が combined softmax 上の確率質量に一致する。"""
    torch.manual_seed(0)
    model = _build_model()
    # teacher_best=(3,5) なので、selected=5 + teacher_best_mass = P(3) + P(5)
    discard_sample = _make_discard_sample(
        tile_type=5,
        legal_tiles=(3, 5, 7),
        step_id=0,
        teacher_tt=5,
        teacher_best=(3, 5),
    )
    samples = [discard_sample]
    from mahjong_agent.data.collate import collate_decision_samples

    batch = collate_decision_samples(samples)
    model.eval()
    with torch.no_grad():
        fwd = model(
            batch.observation.float(),
            discard_mask=batch.discard_mask.float(),
        )
        cscore = model.score_candidates(
            batch.observation.float(), batch.candidate_features.float()
        )
        cmax = int(cscore.candidate_scores.size(1))
        cand_mask = batch.candidate_mask.float()
        if cmax > 0:
            masked_cand = cscore.candidate_scores + (1.0 - cand_mask) * -1.0e9
        else:
            masked_cand = cscore.candidate_scores
        combined = torch.cat([fwd.discard_logits, masked_cand], dim=-1)
        probs = torch.softmax(combined, dim=-1)
    expected_best_mass = float((probs[0, 3] + probs[0, 5]).item())
    expected_teacher_sel = float(probs[0, 5].item())

    summary = audit_decision_samples(samples, model=model)
    tbm_stats = summary.policy["teacher_best_mass_stats"]
    tsp_stats = summary.policy["teacher_selected_prob_stats"]
    assert tbm_stats["count"] == 1
    assert tbm_stats["mean"] == pytest.approx(expected_best_mass, abs=1e-6)
    assert tsp_stats["count"] == 1
    assert tsp_stats["mean"] == pytest.approx(expected_teacher_sel, abs=1e-6)


def test_audit_max_model_batches_truncates():
    torch.manual_seed(0)
    model = _build_model()
    samples = [
        _make_discard_sample(tile_type=5, step_id=i) for i in range(8)
    ]
    cfg = ShardAuditConfig(batch_size=2, max_model_batches=2)
    summary = audit_decision_samples(samples, model=model, config=cfg)
    assert summary.policy["samples_evaluated"] == 4


# ----------------------------------------------------------------------
# JSON / Markdown
# ----------------------------------------------------------------------


def test_summary_to_json_is_serializable():
    samples = [
        _make_discard_sample(tile_type=5, step_id=0, teacher_tt=5),
        _make_discard_sample(tile_type=7, step_id=1),
    ]
    summary = audit_decision_samples(samples)
    s = summary_to_json(summary)
    parsed = json.loads(s)
    assert parsed["num_samples"] == 2
    assert "decision_family_counts" in parsed
    # Counter / numpy が混入していないことの確認 (json.dumps が通れば OK)


def test_summary_to_markdown_returns_str():
    samples = [_make_discard_sample(tile_type=5, step_id=0)]
    summary = audit_decision_samples(samples)
    md = summary_to_markdown(summary)
    assert isinstance(md, str)
    assert "# Shard Audit Summary" in md
    assert "num_samples" in md


# ----------------------------------------------------------------------
# hidden info / local docs guards
# ----------------------------------------------------------------------


def test_audit_signature_does_not_take_hidden_inputs():
    sig = inspect.signature(audit_decision_samples)
    forbidden = {"env", "hands", "wall", "state", "mjai_log"}
    assert forbidden.isdisjoint(set(sig.parameters.keys()))


def test_diagnostics_source_does_not_reference_hidden_state():
    import mahjong_agent.diagnostics as pkg

    pkg_dir = Path(pkg.__file__).parent
    forbidden = (
        "env.hands",
        "env.wall",
        "env.state",
        "obs.hands[",
        "full_state",
        "private_hand",
        "mjai_log",
        "PROJECT_RULE.md",
        "ISSUE_BOARD.md",
        "ISSUE-",
        "AGENTS.md",
        "CLAUDE.md",
        "majong-rl",
    )
    for f in pkg_dir.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, (
                f"diagnostics source {f.name} mentions {tok!r}"
            )


def test_markdown_does_not_reference_local_docs():
    summary = audit_decision_samples(
        [_make_discard_sample(tile_type=5, step_id=0)]
    )
    md = summary_to_markdown(summary)
    for tok in (
        "PROJECT_RULE.md",
        "ISSUE_BOARD.md",
        "AGENTS.md",
        "CLAUDE.md",
    ):
        assert tok not in md
