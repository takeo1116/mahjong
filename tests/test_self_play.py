"""Tests for ``SelfPlayRunner`` and supporting helpers.

unit + small end-to-end smoke against PyPI ``riichienv``.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from mahjong_agent.actions.types import (
    ActionFamily,
    ActionKey,
    LegalActionSet,
    ModelAction,
)
from mahjong_agent.agents import RandomAgent, RuleBasedBaselineAgent
from mahjong_agent.data import (
    SCHEMA_VERSION,
    collate_decision_samples,
    read_decision_shard,
    write_decision_shard,
)
from mahjong_agent.evaluation import (
    EpisodeResult,
    RoundTracker,
    SeatAgents,
    SelfPlayConfig,
    SelfPlayRunner,
    aggregate_metrics,
    make_initial_sample,
)

# ----------------------------------------------------------------------
# SeatAgents
# ----------------------------------------------------------------------


def test_seat_agents_homogeneous_assigns_all_seats():
    agent = RandomAgent(seed=0)
    sa = SeatAgents.homogeneous(agent, actor_type="random", num_players=4)
    assert sa.player_ids == [0, 1, 2, 3]
    for pid in range(4):
        assert sa.agents[pid] is agent
        assert sa.actor_types[pid] == "random"


def test_seat_agents_from_pairs_supports_mixed():
    r = RandomAgent(seed=1)
    b = RuleBasedBaselineAgent()
    sa = SeatAgents.from_pairs(
        {0: (r, "random"), 1: (b, "rule_based"), 2: (r, "random"), 3: (b, "rule_based")}
    )
    assert sa.player_ids == [0, 1, 2, 3]
    assert sa.actor_types[1] == "rule_based"
    assert sa.agents[3] is b


def test_seat_agents_mismatched_keys_raise():
    with pytest.raises(ValueError):
        SeatAgents(agents={0: RandomAgent()}, actor_types={1: "random"})


# ----------------------------------------------------------------------
# RoundTracker unit tests
# ----------------------------------------------------------------------


def test_round_tracker_last_discarder_tracking():
    t = RoundTracker(num_players=4)
    assert t.last_discarder is None
    t.note_discard_applied(2)
    assert t.last_discarder == 2
    t.clear_last_discarder()
    assert t.last_discarder is None


def test_round_tracker_consume_mjai_for_hora_backfills():
    """単純な mjai_log を流して round-end backfill が走ることを確認する。

    encoder / env を介さない pure unit test。
    """
    t = RoundTracker(num_players=4)
    # 適当な observation feat / cand feat / discard mask の dummy sample を
    # round 0 に 1 つだけ突っ込む
    s = make_initial_sample(
        episode_id="ep",
        round_idx=0,
        step_id=0,
        player_id=1,
        decision_family=ActionFamily.NORMAL_DISCARD.value,
        actor_type="random",
        observation_feat=np.zeros(363, dtype=np.float32),
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=np.zeros((0, 12), dtype=np.float32),
        selected_discard_tile_type=5,
        selected_candidate_index=-1,
    )
    t.append_pending(s)
    # mjai_log: start_game / start_kyoku / hora / end_kyoku
    log = [
        {"type": "start_game"},
        {"type": "start_kyoku"},
        {
            "type": "hora",
            "actor": 1,
            "target": 0,
            "tsumo": False,
            "deltas": [-8000, 8000, 0, 0],
        },
        {"type": "end_kyoku"},
    ]
    t.consume_mjai_events(log)
    # backfill が走り samples に move された
    assert len(t.samples) == 1
    assert len(t.pending_samples) == 0
    out = t.samples[0]
    assert out.score_delta == 8000
    assert out.reward == pytest.approx(8000 * 1e-4)
    assert out.round_over is True
    # winner pid 1 で no open meld → WIN_MENZEN (= 0)
    assert out.terminal_class == 0  # win_menzen


def test_round_tracker_consume_mjai_for_ryukyoku_tenpai():
    t = RoundTracker(num_players=4)
    # realistic flow: まず start_game / start_kyoku を処理して
    # round 開始時点まで進める。
    t.consume_mjai_events([{"type": "start_game"}, {"type": "start_kyoku"}])
    # round 内で player 2 が tenpai だったことを snapshot
    t.last_tenpai[2] = True
    # round 内に sample を 1 つ追加
    s = make_initial_sample(
        episode_id="ep",
        round_idx=0,
        step_id=0,
        player_id=2,
        decision_family=ActionFamily.NORMAL_DISCARD.value,
        actor_type="random",
        observation_feat=np.zeros(363, dtype=np.float32),
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=np.zeros((0, 12), dtype=np.float32),
        selected_discard_tile_type=0,
        selected_candidate_index=-1,
    )
    t.append_pending(s)
    # 続けて ryukyoku / end_kyoku event を流す
    t.consume_mjai_events(
        [
            {"type": "start_game"},
            {"type": "start_kyoku"},
            {
                "type": "ryukyoku",
                "reason": "exhaustive_draw",
                "deltas": [-1000, -1000, 3000, -1000],
            },
            {"type": "end_kyoku"},
        ]
    )
    out = t.samples[0]
    assert out.score_delta == 3000
    assert out.reward == pytest.approx(3000 * 1e-4)
    # winner=None, draw=True, tenpai=True → DRAW_TENPAI (= 2)
    assert out.terminal_class == 2


def test_round_tracker_round_idx_increments():
    t = RoundTracker(num_players=4)
    assert t.round_idx == 0
    log = [
        {"type": "start_game"},
        {"type": "start_kyoku"},
        {
            "type": "ryukyoku",
            "reason": "exhaustive_draw",
            "deltas": [0, 0, 0, 0],
        },
        {"type": "end_kyoku"},
        {"type": "start_kyoku"},
        {
            "type": "ryukyoku",
            "reason": "exhaustive_draw",
            "deltas": [0, 0, 0, 0],
        },
        {"type": "end_kyoku"},
    ]
    t.consume_mjai_events(log)
    assert t.round_idx == 2


# ----------------------------------------------------------------------
# Real RiichiEnv smoke
# ----------------------------------------------------------------------


def _build_random_seat(num_players: int = 4, seed: int = 0) -> SeatAgents:
    return SeatAgents.homogeneous(
        RandomAgent(seed=seed), actor_type="random", num_players=num_players
    )


def _build_rule_based_seat(num_players: int = 4) -> SeatAgents:
    return SeatAgents.homogeneous(
        RuleBasedBaselineAgent(), actor_type="rule_based", num_players=num_players
    )


def _build_mixed_seat() -> SeatAgents:
    return SeatAgents.from_pairs(
        {
            0: (RandomAgent(seed=10), "random"),
            1: (RuleBasedBaselineAgent(), "rule_based"),
            2: (RandomAgent(seed=11), "random"),
            3: (RuleBasedBaselineAgent(), "rule_based"),
        }
    )


def test_run_episode_random_4player_smoke():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    res = runner.run_episode(_build_random_seat(seed=0), seed=0)
    assert isinstance(res, EpisodeResult)
    assert res.crash_context is None
    assert res.num_rounds >= 1
    assert res.num_steps > 0
    assert len(res.samples) > 0
    assert all(len(res.final_scores) == 4 for _ in [0])
    assert sorted(res.final_ranks) == [1, 2, 3, 4]


def test_run_episode_rule_based_4player_smoke():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    res = runner.run_episode(_build_rule_based_seat(), seed=1)
    assert res.crash_context is None
    assert res.num_rounds >= 1
    assert len(res.samples) > 0


def test_run_episode_mixed_agents_smoke():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    res = runner.run_episode(_build_mixed_seat(), seed=2)
    assert res.crash_context is None
    # 4 player の actor_type が 2 種類混ざっている
    assert set(res.actor_types.values()) == {"random", "rule_based"}
    # samples の actor_type も 2 種類含む
    families = {s.actor_type for s in res.samples}
    assert {"random", "rule_based"} <= families


def test_run_episode_seed_smoke_structural():
    """同じ seed で 2 回回しても crash せず、構造的に整合した EpisodeResult が
    返ることを smoke で確認する (YON_TONPUSEN partial-determinism版)。

    Note
    ----
    PyPI ``riichienv 0.4.8`` の ``env.reset(wall=...)`` は最初の round の wall
    のみ deterministic に固定でき、2 つ目以降の round の wall は engine 内部の
    randomness で引かれる。そのため YON_TONPUSEN (= multi-round) では
    final_scores の strict equality は担保できない。本テストは
    「seed/wall 経路が crash せず動く」「wall_seed/wall_digest が再現可能」
    「ranks が legal permutation」「sample 生成が完走」を確認する。

    Strict determinism (full equality) は ``test_strict_determinism_yon_ikkyoku``
    側で ``YON_IKKYOKU`` (= single round) を使って担保する。
    """
    runner1 = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    runner2 = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    res1 = runner1.run_episode(_build_random_seat(seed=7), seed=7)
    res2 = runner2.run_episode(_build_random_seat(seed=7), seed=7)
    for res in (res1, res2):
        assert res.crash_context is None
        assert len(res.final_scores) == 4
        assert sorted(res.final_ranks) == [1, 2, 3, 4]
        assert res.num_rounds >= 1
        assert len(res.samples) > 0
    # wall_seed / wall_digest は同 seed・同 config で run 間一致する
    assert res1.wall_seed == res2.wall_seed
    assert res1.wall_digest == res2.wall_digest
    assert res1.reset_seed == res2.reset_seed


# ----------------------------------------------------------------------
# Strict determinism (single-round game type) and wall reproduction
# ----------------------------------------------------------------------


def _strict_config() -> SelfPlayConfig:
    """Strict determinism を担保できる single-round (YON_IKKYOKU) config。"""
    import riichienv

    return SelfPlayConfig(
        game_type=riichienv.GameType.YON_IKKYOKU,
        max_steps_per_game=4000,
        deterministic_wall=True,
    )


def _episode_signature(res: EpisodeResult) -> dict:
    """deterministic 比較用 dict (numpy / 関数を含まない form)。"""
    return {
        "wall_seed": res.wall_seed,
        "wall_digest": res.wall_digest,
        "reset_seed": res.reset_seed,
        "final_scores": list(res.final_scores),
        "final_ranks": list(res.final_ranks),
        "num_rounds": int(res.num_rounds),
        "num_steps": int(res.num_steps),
        "num_samples": len(res.samples),
        "round_summaries": [
            (
                rs.get("round_idx"),
                rs.get("is_draw"),
                rs.get("winner"),
                rs.get("deal_in"),
                rs.get("is_tsumo"),
                tuple(rs.get("deltas", [])),
                rs.get("ryukyoku_reason"),
            )
            for rs in res.round_summaries
        ],
        "decision_family_seq": [s.decision_family for s in res.samples],
        "actor_type_seq": [s.actor_type for s in res.samples],
        "player_id_seq": [s.player_id for s in res.samples],
        "selected_tile_seq": [
            (s.selected_discard_tile_type, s.selected_candidate_index)
            for s in res.samples
        ],
    }


def test_strict_determinism_yon_ikkyoku_random():
    """``YON_IKKYOKU`` + ``deterministic_wall=True`` で 2 回 run_episode した
    結果が完全一致することを assert する (strict determinism)。"""
    cfg = _strict_config()
    runner1 = SelfPlayRunner(config=cfg)
    runner2 = SelfPlayRunner(config=cfg)
    sa1 = SeatAgents.homogeneous(
        RandomAgent(seed=7), actor_type="random", num_players=4
    )
    sa2 = SeatAgents.homogeneous(
        RandomAgent(seed=7), actor_type="random", num_players=4
    )
    res1 = runner1.run_episode(sa1, seed=7)
    res2 = runner2.run_episode(sa2, seed=7)
    assert res1.crash_context is None
    assert res2.crash_context is None
    sig1 = _episode_signature(res1)
    sig2 = _episode_signature(res2)
    assert sig1 == sig2, f"signatures differ: sig1={sig1}\nsig2={sig2}"
    # observation feature も完全一致 (= 同じ wall / 同じ deal / 同じ history)
    obs1 = [s.observation for s in res1.samples]
    obs2 = [s.observation for s in res2.samples]
    assert len(obs1) == len(obs2)
    for a, b in zip(obs1, obs2, strict=True):
        np.testing.assert_array_equal(a, b)


def test_strict_determinism_yon_ikkyoku_rule_based():
    """rule-based baseline でも strict determinism を担保する。"""
    cfg = _strict_config()
    runner1 = SelfPlayRunner(config=cfg)
    runner2 = SelfPlayRunner(config=cfg)
    sa1 = SeatAgents.homogeneous(
        RuleBasedBaselineAgent(seed=0), actor_type="rule_based", num_players=4
    )
    sa2 = SeatAgents.homogeneous(
        RuleBasedBaselineAgent(seed=0), actor_type="rule_based", num_players=4
    )
    res1 = runner1.run_episode(sa1, seed=11)
    res2 = runner2.run_episode(sa2, seed=11)
    assert res1.crash_context is None
    assert res2.crash_context is None
    assert _episode_signature(res1) == _episode_signature(res2)


def test_strict_determinism_run_games_multi_seed():
    """``run_games(seeds=[...])`` を 2 回呼んでも全 episode 完全一致。"""
    cfg = _strict_config()
    seeds = [0, 1, 2, 5, 8]

    def _go():
        runner = SelfPlayRunner(config=cfg)
        sa = SeatAgents.homogeneous(
            RandomAgent(seed=0), actor_type="random", num_players=4
        )
        return runner.run_games(sa, seeds=list(seeds))

    a = _go()
    b = _go()
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert _episode_signature(x) == _episode_signature(y)


def test_wall_seed_offset_changes_wall_only():
    """``wall_seed_offset`` を変えると wall_seed / wall_digest だけが変わり、
    同じ ``run_episode(seed=...)`` でも別 wall が生成される。"""
    import riichienv

    cfg0 = SelfPlayConfig(
        game_type=riichienv.GameType.YON_IKKYOKU,
        deterministic_wall=True,
        wall_seed_offset=0,
    )
    cfg100 = SelfPlayConfig(
        game_type=riichienv.GameType.YON_IKKYOKU,
        deterministic_wall=True,
        wall_seed_offset=100,
    )
    sa = SeatAgents.homogeneous(
        RandomAgent(seed=0), actor_type="random", num_players=4
    )
    r0 = SelfPlayRunner(config=cfg0).run_episode(sa, seed=3)
    r1 = SelfPlayRunner(config=cfg100).run_episode(
        SeatAgents.homogeneous(
            RandomAgent(seed=0), actor_type="random", num_players=4
        ),
        seed=3,
    )
    assert r0.wall_seed == 3
    assert r1.wall_seed == 103
    assert r0.wall_digest != r1.wall_digest
    assert r0.reset_seed == r1.reset_seed == 3


def test_deterministic_wall_false_does_not_record_wall_info():
    """``deterministic_wall=False`` のときは wall_seed / wall_digest が None。"""
    cfg = SelfPlayConfig(
        max_steps_per_game=8000,
        deterministic_wall=False,
    )
    runner = SelfPlayRunner(config=cfg)
    sa = SeatAgents.homogeneous(
        RandomAgent(seed=0), actor_type="random", num_players=4
    )
    res = runner.run_episode(sa, seed=4)
    assert res.wall_seed is None
    assert res.wall_digest is None
    # reset_seed は seed と一致 (= env.reset(seed=...) に渡したから)
    assert res.reset_seed == 4


def test_crash_context_contains_wall_reproduction_info():
    """crash 時の ``crash_context`` に wall_seed / wall_digest / reset_seed が
    含まれることを確認する。"""

    class _BadAgent:
        def select_action(self, legal_set, *, rng=None, observation=None):
            raise RuntimeError("boom")

    cfg = _strict_config()
    runner = SelfPlayRunner(config=cfg)
    sa = SeatAgents.homogeneous(_BadAgent(), actor_type="bad", num_players=4)
    res = runner.run_episode(sa, seed=17)
    assert res.crash_context is not None
    ctx = res.crash_context
    assert ctx["wall_seed"] == 17
    assert isinstance(ctx["wall_digest"], str) and len(ctx["wall_digest"]) > 0
    assert ctx["reset_seed"] == 17
    # JSON serializable
    json.dumps(ctx)


def test_metrics_contains_wall_seeds_and_digests():
    """``aggregate_metrics`` の出力に wall_seeds / wall_digests が入る。"""
    cfg = _strict_config()
    runner = SelfPlayRunner(config=cfg)
    sa = SeatAgents.homogeneous(
        RandomAgent(seed=0), actor_type="random", num_players=4
    )
    results = runner.run_games(sa, seeds=[0, 1, 2])
    m = aggregate_metrics(results)
    assert m["wall_seeds"] == [0, 1, 2]
    assert len(m["wall_digests"]) == 3
    assert all(isinstance(d, str) and len(d) > 0 for d in m["wall_digests"])
    # JSON serializable
    json.dumps(m)


def test_decision_sample_does_not_carry_wall_tiles():
    """``DecisionSample.metadata`` に wall list そのものが入っていない。

    再現用には ``wall_seed`` / ``wall_digest`` のみが使われる方針。
    """
    cfg = _strict_config()
    runner = SelfPlayRunner(config=cfg)
    sa = SeatAgents.homogeneous(
        RandomAgent(seed=0), actor_type="random", num_players=4
    )
    res = runner.run_episode(sa, seed=2)
    for s in res.samples:
        for key, val in s.metadata.items():
            # 長さ 136 の list が紛れ込んでいないことを sanity check
            if isinstance(val, list):
                assert len(val) < 50, (
                    f"metadata[{key!r}] has length {len(val)} — wall list?"
                )


def test_run_games_multiple_seeds():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    results = runner.run_games(
        _build_random_seat(seed=0), seeds=[0, 1, 2]
    )
    assert len(results) == 3
    for r in results:
        assert r.crash_context is None
        assert r.num_rounds >= 1


def test_run_games_base_seed_default():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    results = runner.run_games(
        _build_random_seat(seed=0), base_seed=100, num_games=2
    )
    assert len(results) == 2
    assert [r.seed for r in results] == [100, 101]


# ----------------------------------------------------------------------
# DecisionSample integration
# ----------------------------------------------------------------------


def test_episode_samples_are_decision_samples():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    res = runner.run_episode(_build_random_seat(seed=5), seed=5)
    assert all(s.schema_version == SCHEMA_VERSION for s in res.samples)
    # 各 sample が observation / discard_mask / cand feature を持つ
    for s in res.samples[:10]:
        assert s.observation.shape == (363,)
        assert s.discard_mask.shape == (34,)
        assert s.candidate_features.ndim == 2
        assert s.candidate_features.shape[1] in (0, 86)
    # round_over=True が各 round の per-player 終端のみ立っている
    # 同じ (round_id, player_id) に対して round_over=True は最大 1 個
    for (rid, pid) in {(s.round_id, s.player_id) for s in res.samples}:
        flags = [s.round_over for s in res.samples if s.round_id == rid and s.player_id == pid]
        assert flags.count(True) <= 1


def test_episode_samples_can_be_collated():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    res = runner.run_episode(_build_random_seat(seed=3), seed=3)
    assert len(res.samples) > 0
    batch = collate_decision_samples(res.samples[:32])
    assert batch.batch_size == min(32, len(res.samples))


def test_episode_samples_roundtrip_through_shard(tmp_path: Path):
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    res = runner.run_episode(_build_random_seat(seed=4), seed=4)
    assert len(res.samples) > 0
    samples = res.samples[:48]
    out = tmp_path / "shard.npz"
    write_decision_shard(out, samples)
    loaded = read_decision_shard(out)
    assert len(loaded) == len(samples)
    for orig, got in zip(samples, loaded, strict=True):
        assert orig.episode_id == got.episode_id
        assert orig.round_id == got.round_id
        assert orig.player_id == got.player_id
        assert orig.decision_family == got.decision_family
        assert orig.actor_type == got.actor_type
        np.testing.assert_allclose(orig.observation, got.observation)


def test_episode_terminated_flag_set_on_final_per_player():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    res = runner.run_episode(_build_random_seat(seed=6), seed=6)
    assert res.crash_context is None
    term = [s for s in res.samples if s.terminated]
    assert len(term) == 4  # 4 player 分の最終 sample に 1 つずつ
    assert {s.player_id for s in term} == {0, 1, 2, 3}


def test_episode_reward_is_score_delta_scaled():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    res = runner.run_episode(_build_random_seat(seed=8), seed=8)
    # reward == score_delta * 1e-4 for each sample
    for s in res.samples:
        assert s.reward == pytest.approx(float(s.score_delta) * 1e-4)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def test_aggregate_metrics_is_json_serializable():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    results = runner.run_games(
        _build_random_seat(seed=0), seeds=[0, 1]
    )
    m = aggregate_metrics(results)
    s = json.dumps(m)
    parsed = json.loads(s)
    assert parsed["num_games"] == 2
    assert parsed["num_rounds"] >= 2
    # schema keys
    expected_keys = {
        "num_games", "num_rounds", "num_steps", "num_samples",
        "crash_count", "seeds", "final_scores", "final_ranks",
        "rank_counts", "avg_rank_by_actor_type", "win_count",
        "tsumo_count", "ron_count", "decision_family_counts",
        "actor_type_counts", "crashes",
    }
    assert expected_keys <= parsed.keys()


def test_aggregate_metrics_counts_actor_types():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    results = [runner.run_episode(_build_mixed_seat(), seed=42)]
    m = aggregate_metrics(results)
    assert m["actor_type_counts"]["random"] > 0
    assert m["actor_type_counts"]["rule_based"] > 0
    # rank_counts は 2 種類の actor_type を分けて持つ
    assert "random" in m["rank_counts"]
    assert "rule_based" in m["rank_counts"]


# ----------------------------------------------------------------------
# Crash context
# ----------------------------------------------------------------------


class _ExplodingAgent:
    """select_action で必ず例外を投げる pathological agent。"""

    def select_action(self, legal_set, *, rng=None, observation=None):
        raise RuntimeError("boom!")


def test_run_episode_records_crash_context_on_agent_exception():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    sa = SeatAgents.homogeneous(_ExplodingAgent(), actor_type="bad", num_players=4)
    res = runner.run_episode(sa, seed=0)
    assert res.crash_context is not None
    ctx = res.crash_context
    assert ctx["seed"] == 0
    assert ctx["error_type"] == "RuntimeError"
    assert "boom" in ctx["error_message"]
    # crash context は JSON serializable
    json.dumps(ctx)


def test_run_episode_max_steps_records_crash_context():
    # max_steps を 1 にして即 abort させる
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=1))
    sa = _build_random_seat(seed=0)
    res = runner.run_episode(sa, seed=0)
    assert res.crash_context is not None
    assert res.crash_context["reason"] == "max_steps_per_game_exceeded"


def test_aggregate_metrics_counts_crashes():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    sa_bad = SeatAgents.homogeneous(
        _ExplodingAgent(), actor_type="bad", num_players=4
    )
    sa_good = _build_random_seat(seed=0)
    results = [
        runner.run_episode(sa_bad, seed=0),
        runner.run_episode(sa_good, seed=1),
    ]
    m = aggregate_metrics(results)
    assert m["crash_count"] == 1
    assert len(m["crashes"]) == 1


# ----------------------------------------------------------------------
# Hidden info guard / source guard
# ----------------------------------------------------------------------


def test_runner_signature_does_not_take_hidden_state_params():
    """SelfPlayRunner.run_episode / run_games が hidden state を受け取らない。"""
    for fn in (SelfPlayRunner.run_episode, SelfPlayRunner.run_games):
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())
        forbidden = {"hands", "wall", "state", "full_state", "private_hand"}
        assert forbidden.isdisjoint(params), (
            f"{fn.__qualname__} signature includes hidden-info parameter"
        )


def test_evaluation_source_does_not_reference_hidden_state():
    """evaluation/ 内 source に env.hands / .wall / env.state 等が無い。"""
    import mahjong_agent.evaluation as eval_pkg

    pkg_dir = Path(eval_pkg.__file__).parent
    py_files = list(pkg_dir.rglob("*.py"))
    assert py_files
    forbidden_tokens = (
        "env.hands",
        "env.wall",
        "env.state",
        "full_state",
        "private_hand",
    )
    for f in py_files:
        text = f.read_text(encoding="utf-8")
        for tok in forbidden_tokens:
            assert tok not in text, (
                f"evaluation source {f.name} references {tok!r}"
            )


def test_evaluation_source_does_not_reference_local_docs():
    """evaluation/ 内 source に local docs / 旧 repo 名への参照が無い。"""
    import mahjong_agent.evaluation as eval_pkg

    pkg_dir = Path(eval_pkg.__file__).parent
    py_files = list(pkg_dir.rglob("*.py"))
    assert py_files
    forbidden = (
        "PROJECT_RULE.md",
        "ISSUE_BOARD.md",
        "ISSUE-",
        "AGENTS.md",
        "CLAUDE.md",
        "majong-rl",
    )
    for f in py_files:
        text = f.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, (
                f"evaluation source {f.name} mentions {tok!r}"
            )


# ----------------------------------------------------------------------
# last_discarder tracking against real env (smoke)
# ----------------------------------------------------------------------


def test_response_phase_last_discarder_tracking_smoke():
    """response phase で agent に渡される LegalActionSet の Chi/Pon の
    target_rel_seat が、tracker の last_discarder と整合することを確認する。

    実 env を回し、response phase で agent が呼ばれる際の legal_set を傍受
    する spy agent を仕込む。
    """

    class _SpyAgent:
        def __init__(self):
            self.captured: list[tuple[int, int | None, list[ActionKey]]] = []
            self._inner = RuleBasedBaselineAgent()

        def select_action(self, legal_set: LegalActionSet, *, rng=None, observation=None):
            # response phase の判定: candidates に CHI/PON/DAIMINKAN/RON が
            # 出ている (RIICHI_DISCARD は出ないか tile_type が None)
            if any(
                c.family in (ActionFamily.CHI, ActionFamily.PON,
                             ActionFamily.DAIMINKAN, ActionFamily.RON)
                for c in legal_set.candidates
            ):
                # 各 call candidate の target_rel_seat を記録する
                for c in legal_set.candidates:
                    if c.family in (
                        ActionFamily.CHI, ActionFamily.PON,
                        ActionFamily.DAIMINKAN, ActionFamily.RON
                    ):
                        self.captured.append(
                            (legal_set.decision_player,
                             c.key.target_rel_seat,
                             [x.key for x in legal_set.candidates])
                        )
                        break
            return self._inner.select_action(legal_set, rng=rng, observation=observation)

    spies = {pid: _SpyAgent() for pid in range(4)}
    sa = SeatAgents.from_pairs(
        {pid: (spies[pid], "rule_based") for pid in range(4)}
    )
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    # response phase が起きやすい seed を探索
    found = False
    for seed in range(20):
        for sp in spies.values():
            sp.captured.clear()
        res = runner.run_episode(sa, seed=seed)
        assert res.crash_context is None
        if any(sp.captured for sp in spies.values()):
            found = True
            break
    assert found, "no response phase observed in 20 seeds (test setup issue)"
    # captured の target_rel_seat が None でない (=正しく追跡されている)
    for sp in spies.values():
        for decision_pid, rel_seat, _keys in sp.captured:
            assert rel_seat is not None, (
                f"target_rel_seat=None at decision_player={decision_pid}; "
                f"last_discarder tracking failed"
            )
            assert 0 <= int(rel_seat) < 4


# ----------------------------------------------------------------------
# duck-typed agent
# ----------------------------------------------------------------------


class _ConstantPassAgent:
    """常に Pass を選ぶ (Pass 無ければ normal_discard 最小 tile_type を選ぶ) duck-typed agent。"""

    def select_action(self, legal_set, *, rng=None, observation=None):
        from mahjong_agent.agents.base import AgentDecision

        for c in legal_set.candidates:
            if c.family == ActionFamily.PASS:
                return AgentDecision(action=c, rationale="const_pass")
        if legal_set.normal_discard:
            tt = min(legal_set.normal_discard.keys())
            return AgentDecision(
                action=legal_set.normal_discard[tt], rationale="const_min_discard"
            )
        # fallback: first candidate
        c0 = legal_set.candidates[0]
        return AgentDecision(action=c0, rationale="const_fallback")


def test_duck_typed_agent_works_in_loop():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    sa = SeatAgents.homogeneous(
        _ConstantPassAgent(), actor_type="duck", num_players=4
    )
    res = runner.run_episode(sa, seed=0)
    assert res.crash_context is None
    assert len(res.samples) > 0
    # actor_type label が反映されている
    assert all(s.actor_type == "duck" for s in res.samples)


# ----------------------------------------------------------------------
# sample candidate_index / discard tile_type consistency
# ----------------------------------------------------------------------


def test_sample_selection_indices_consistent_with_family():
    runner = SelfPlayRunner(config=SelfPlayConfig(max_steps_per_game=8000))
    res = runner.run_episode(_build_random_seat(seed=9), seed=9)
    for s in res.samples:
        if s.decision_family == ActionFamily.NORMAL_DISCARD.value:
            assert 0 <= s.selected_discard_tile_type < 34
            assert s.selected_candidate_index == -1
        else:
            assert s.selected_discard_tile_type == -1
            # candidate index は 0 以上 (candidate が無ければそもそも選べない)
            assert s.selected_candidate_index >= 0


# ----------------------------------------------------------------------
# unused fixture / helper hidden via _ModelAction wrap
# ----------------------------------------------------------------------


def _dummy_legal_set(actor=0):
    return LegalActionSet(
        decision_player=actor,
        normal_discard={
            0: ModelAction(
                key=ActionKey(family=ActionFamily.NORMAL_DISCARD, tile_type=0),
                actor=actor,
                _raw_actions=(),
            )
        },
    )


def test_make_initial_sample_defaults_are_clean():
    feat = np.zeros(363, dtype=np.float32)
    s = make_initial_sample(
        episode_id="x",
        round_idx=0,
        step_id=0,
        player_id=0,
        decision_family=ActionFamily.NORMAL_DISCARD.value,
        actor_type="random",
        observation_feat=feat,
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=np.zeros((0, 12), dtype=np.float32),
        selected_discard_tile_type=0,
        selected_candidate_index=-1,
    )
    assert s.schema_version == SCHEMA_VERSION
    assert s.old_log_prob == 0.0
    assert s.value == 0.0
    assert s.reward == 0.0
    assert s.terminated is False
    assert s.round_over is False
    assert s.terminal_class == -1
    assert s.yaku_loss_mask == 0.0
    assert s.han == -1
    assert s.fu == -1
    assert s.score_delta == 0
    assert s.teacher_available is False
