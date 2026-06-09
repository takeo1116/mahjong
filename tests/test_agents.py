"""Tests for ``RandomAgent`` and ``RuleBasedBaselineAgent``.

unit + small smoke. real RiichiEnv against a short random play tests are
included to verify that agent decisions can be resolved back to raw
``riichienv.Action`` and fed into ``env.step``.
"""
from __future__ import annotations

import inspect
import random
from pathlib import Path

import pytest

from mahjong_agent.actions.types import (
    ActionFamily,
    ActionKey,
    LegalActionSet,
    ModelAction,
)
from mahjong_agent.agents import (
    AgentDecision,
    RandomAgent,
    RuleBasedBaselineAgent,
)

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _normal_discard(tile_type: int, actor: int = 0) -> ModelAction:
    key = ActionKey(family=ActionFamily.NORMAL_DISCARD, tile_type=tile_type)
    return ModelAction(key=key, actor=actor, _raw_actions=())


def _candidate(
    family: ActionFamily,
    tile_type: int | None = None,
    consume: tuple[int, ...] = (),
    target_rel_seat: int | None = None,
    actor: int = 0,
) -> ModelAction:
    key = ActionKey(
        family=family,
        tile_type=tile_type,
        consume_tile_types=consume,
        target_rel_seat=target_rel_seat,
    )
    return ModelAction(key=key, actor=actor, _raw_actions=())


def _make_set(
    normal_tts: list[int] | None = None,
    candidates: list[ModelAction] | None = None,
    actor: int = 0,
) -> LegalActionSet:
    nd = {tt: _normal_discard(tt, actor=actor) for tt in (normal_tts or [])}
    return LegalActionSet(
        decision_player=actor,
        normal_discard=nd,
        candidates=tuple(candidates or []),
    )


# ----------------------------------------------------------------------
# AgentDecision
# ----------------------------------------------------------------------


def test_agent_decision_carries_action_and_rationale():
    ma = _normal_discard(5)
    dec = AgentDecision(action=ma, rationale="random")
    assert dec.action is ma
    assert dec.family == ActionFamily.NORMAL_DISCARD
    assert dec.tile_type == 5
    assert dec.rationale == "random"
    # AgentDecision.raw_actions() は ModelAction の raw を中継する
    assert dec.raw_actions() == ()


# ----------------------------------------------------------------------
# RandomAgent
# ----------------------------------------------------------------------


def test_random_agent_picks_legal_action_only_discard_case():
    agent = RandomAgent(seed=0)
    lset = _make_set(normal_tts=[0, 5, 17, 33])
    for _ in range(50):
        dec = agent.select_action(lset)
        assert isinstance(dec, AgentDecision)
        assert dec.family == ActionFamily.NORMAL_DISCARD
        assert dec.tile_type in {0, 5, 17, 33}


def test_random_agent_picks_legal_action_only_candidate_case():
    agent = RandomAgent(seed=1)
    cands = [
        _candidate(ActionFamily.TSUMO),
        _candidate(ActionFamily.PASS),
        _candidate(ActionFamily.RIICHI_DISCARD, tile_type=3),
    ]
    lset = _make_set(candidates=cands)
    chosen_families: set[ActionFamily] = set()
    for _ in range(100):
        dec = agent.select_action(lset)
        # 必ず candidates 由来 (normal_discard は空)
        assert dec.action in cands
        chosen_families.add(dec.family)
    # 100 trial で 3 family すべて出るはず (3^-100 で同じものしか出ない確率)
    assert chosen_families == {
        ActionFamily.TSUMO,
        ActionFamily.PASS,
        ActionFamily.RIICHI_DISCARD,
    }


def test_random_agent_mixed_normal_and_candidate_case():
    """normal_discard と candidates が混在する場合、両方が起こりうる。"""
    agent = RandomAgent(seed=2)
    cands = [_candidate(ActionFamily.RIICHI_DISCARD, tile_type=3)]
    lset = _make_set(normal_tts=[1, 2, 3], candidates=cands)
    families: set[ActionFamily] = set()
    for _ in range(200):
        families.add(agent.select_action(lset).family)
    assert ActionFamily.NORMAL_DISCARD in families
    assert ActionFamily.RIICHI_DISCARD in families


def test_random_agent_is_deterministic_with_seed():
    lset = _make_set(
        normal_tts=[0, 1, 2, 3, 4],
        candidates=[
            _candidate(ActionFamily.PASS),
            _candidate(ActionFamily.RIICHI_DISCARD, tile_type=2),
        ],
    )
    a1 = RandomAgent(seed=12345)
    a2 = RandomAgent(seed=12345)
    seq1 = [a1.select_action(lset).action.key for _ in range(30)]
    seq2 = [a2.select_action(lset).action.key for _ in range(30)]
    assert seq1 == seq2


def test_random_agent_per_call_rng_override():
    lset = _make_set(normal_tts=[0, 1, 2])
    agent = RandomAgent(seed=999)
    rng = random.Random(42)
    seq = [agent.select_action(lset, rng=rng).action.key for _ in range(20)]
    rng2 = random.Random(42)
    expected = [
        agent.select_action(lset, rng=rng2).action.key for _ in range(20)
    ]
    assert seq == expected


def test_random_agent_empty_legal_set_raises():
    agent = RandomAgent(seed=0)
    lset = _make_set()
    with pytest.raises(ValueError):
        agent.select_action(lset)


# ----------------------------------------------------------------------
# RuleBasedBaselineAgent
# ----------------------------------------------------------------------


def test_rule_based_prefers_tsumo_over_discard():
    agent = RuleBasedBaselineAgent()
    cands = [_candidate(ActionFamily.TSUMO)]
    lset = _make_set(normal_tts=[0, 1, 2, 3], candidates=cands)
    dec = agent.select_action(lset)
    assert dec.family == ActionFamily.TSUMO
    assert dec.rationale.startswith("win")


def test_rule_based_prefers_ron_over_pass():
    agent = RuleBasedBaselineAgent()
    cands = [
        _candidate(ActionFamily.PASS),
        _candidate(ActionFamily.RON, tile_type=4, target_rel_seat=3),
    ]
    lset = _make_set(candidates=cands)
    dec = agent.select_action(lset)
    assert dec.family == ActionFamily.RON
    assert dec.rationale.startswith("win")


def test_rule_based_picks_kyushu_when_available():
    agent = RuleBasedBaselineAgent()
    cands = [
        _candidate(ActionFamily.PASS),
        _candidate(ActionFamily.KYUSHU_KYUHAI),
    ]
    lset = _make_set(normal_tts=[0, 5, 9, 27], candidates=cands)
    dec = agent.select_action(lset)
    assert dec.family == ActionFamily.KYUSHU_KYUHAI


def test_rule_based_normal_discard_picks_highest_tile_type():
    """最も大きい tile_type (字牌寄り) を選ぶ deterministic 仕様。"""
    agent = RuleBasedBaselineAgent()
    lset = _make_set(normal_tts=[0, 5, 17, 31])
    dec = agent.select_action(lset)
    assert dec.family == ActionFamily.NORMAL_DISCARD
    assert dec.tile_type == 31


def test_rule_based_prefers_riichi_discard_by_default():
    """prefer_riichi=True (default) で立直できる局面は RIICHI_DISCARD を優先。"""
    agent = RuleBasedBaselineAgent()  # default prefer_riichi=True
    cands = [_candidate(ActionFamily.RIICHI_DISCARD, tile_type=3)]
    lset = _make_set(normal_tts=[1, 2, 3], candidates=cands)
    dec = agent.select_action(lset)
    assert dec.family == ActionFamily.RIICHI_DISCARD
    assert dec.tile_type == 3


def test_rule_based_prefer_riichi_off_keeps_normal_discard():
    """prefer_riichi=False で従来通り通常打牌を優先 (自動立直しない)。"""
    agent = RuleBasedBaselineAgent(prefer_riichi=False)
    cands = [_candidate(ActionFamily.RIICHI_DISCARD, tile_type=3)]
    lset = _make_set(normal_tts=[1, 2, 3], candidates=cands)
    dec = agent.select_action(lset)
    assert dec.family == ActionFamily.NORMAL_DISCARD


def test_rule_based_riichi_discard_deterministic_tile_choice():
    """複数 RIICHI_DISCARD candidate (observation 無し) で deterministic に
    tile_type 昇順先頭を選ぶ。"""
    agent = RuleBasedBaselineAgent()
    cands = [
        _candidate(ActionFamily.RIICHI_DISCARD, tile_type=8),
        _candidate(ActionFamily.RIICHI_DISCARD, tile_type=3),
    ]
    lset = _make_set(normal_tts=[3, 8], candidates=cands)
    decs = [agent.select_action(lset) for _ in range(5)]
    for dec in decs:
        assert dec.family == ActionFamily.RIICHI_DISCARD
        assert dec.tile_type == 3  # tile_type 昇順の先頭、deterministic


def test_rule_based_riichi_discard_fallback_when_no_normal_discard():
    """通常打牌が無く RIICHI_DISCARD だけ残る例外ケース (prefer_riichi に
    関わらず立直打牌を選ぶ)。"""
    for prefer in (True, False):
        agent = RuleBasedBaselineAgent(prefer_riichi=prefer)
        cands = [
            _candidate(ActionFamily.RIICHI_DISCARD, tile_type=8),
            _candidate(ActionFamily.RIICHI_DISCARD, tile_type=3),
        ]
        lset = _make_set(candidates=cands)
        dec = agent.select_action(lset)
        assert dec.family == ActionFamily.RIICHI_DISCARD
        assert dec.tile_type == 3  # tile_type 昇順の先頭


class _RiichiFakeObs:
    """riichi discard tile choice 用の薄い observation stub。"""

    def __init__(self, hand: list[int], player_id: int = 0):
        self.hand = hand
        self.player_id = player_id
        self.melds = [[], [], [], []]


def test_rule_based_riichi_discard_uses_shanten_best_set_with_obs():
    """observation/hand_counts が使えるとき、riichi discard tile は
    shanten 最小 + ukeire 最大の best set 内から選ばれ teacher 情報が入る。"""
    import numpy as np

    from mahjong_agent.baseline.discard_select import find_best_discard

    # tenpai 形: 123m 456m 789m 22p + 5p (14 枚)。5p を切れば 22p 雀頭 +
    # 123/456/789m 三面子 = 22p 単騎 tenpai。2p を 1 枚切っても tenpai 形が
    # 変わる。riichi candidate は手牌にある tile_type のうち複数用意する。
    hand = (
        [0, 1, 2]      # 1m2m3m (tile_type 0,1,2)
        + [12, 16, 20]  # 4m5m6m -> tile_type 3,4,5? いや tile_id//4
        + [24, 28, 32]  # 7m8m9m
        + [40, 41]      # 2p (tile_type 10) x2
        + [52]          # 5p (tile_type 13)
    )
    obs = _RiichiFakeObs(hand=hand)
    # riichi candidate を tile_type 13 (5p) と 10 (2p) で用意
    cands = [
        _candidate(ActionFamily.RIICHI_DISCARD, tile_type=10),
        _candidate(ActionFamily.RIICHI_DISCARD, tile_type=13),
    ]
    lset = _make_set(normal_tts=[10, 13], candidates=cands)
    agent = RuleBasedBaselineAgent()  # prefer_riichi=True
    dec = agent.select_action(lset, observation=obs)
    assert dec.family == ActionFamily.RIICHI_DISCARD
    assert "teacher_best_mask" in dec.extras
    assert "teacher_discard_tile_type" in dec.extras
    assert "teacher_shanten" in dec.extras
    assert "teacher_ukeire" in dec.extras
    # 選んだ tile は riichi candidate の tile_type のいずれか
    assert dec.tile_type in {10, 13}
    # teacher_discard_tile_type は best_mask 内 (find_best_discard と整合)
    from mahjong_agent.baseline.call_policy import extract_hand_counts

    hc = extract_hand_counts(obs)
    legal_mask = np.zeros(34, dtype=np.float32)
    legal_mask[10] = 1.0
    legal_mask[13] = 1.0
    expected = find_best_discard(hc, legal_mask, meld_count=0)
    assert dec.tile_type == expected.best_tile_type


def test_rule_based_passes_on_response():
    agent = RuleBasedBaselineAgent()
    cands = [
        _candidate(
            ActionFamily.CHI,
            tile_type=5,
            consume=(3, 4),
            target_rel_seat=3,
        ),
        _candidate(
            ActionFamily.PON, tile_type=5, consume=(5, 5), target_rel_seat=2
        ),
        _candidate(ActionFamily.PASS),
    ]
    lset = _make_set(candidates=cands)
    dec = agent.select_action(lset)
    assert dec.family == ActionFamily.PASS


def test_rule_based_call_fallback_when_no_pass_no_normal():
    """Pass も normal_discard も無い病的ケースで crash しないこと。"""
    agent = RuleBasedBaselineAgent()
    cands = [
        _candidate(
            ActionFamily.CHI,
            tile_type=5,
            consume=(3, 4),
            target_rel_seat=3,
        )
    ]
    lset = _make_set(candidates=cands)
    dec = agent.select_action(lset)
    assert dec.family == ActionFamily.CHI


def test_rule_based_kita_chosen_when_offered():
    agent = RuleBasedBaselineAgent()
    cands = [_candidate(ActionFamily.KITA, tile_type=30)]
    lset = _make_set(candidates=cands)
    dec = agent.select_action(lset)
    assert dec.family == ActionFamily.KITA


def test_rule_based_empty_legal_set_raises():
    agent = RuleBasedBaselineAgent()
    with pytest.raises(ValueError):
        agent.select_action(_make_set())


def test_rule_based_decision_is_deterministic():
    """同じ legal_set には常に同じ decision を返す。"""
    agent = RuleBasedBaselineAgent()
    cands = [
        _candidate(ActionFamily.PASS),
        _candidate(ActionFamily.RIICHI_DISCARD, tile_type=2),
    ]
    lset = _make_set(normal_tts=[0, 7, 33], candidates=cands)
    decs = {agent.select_action(lset).action.key for _ in range(10)}
    assert len(decs) == 1


# ----------------------------------------------------------------------
# hidden info guard / API surface
# ----------------------------------------------------------------------


def test_agent_select_action_signature_does_not_take_env():
    """agent.select_action は env / hands / wall / state を受け取らない。"""
    for cls in (RandomAgent, RuleBasedBaselineAgent):
        sig = inspect.signature(cls.select_action)
        param_names = set(sig.parameters.keys())
        forbidden = {"env", "hands", "wall", "state", "full_state"}
        assert forbidden.isdisjoint(param_names), (
            f"{cls.__name__}.select_action signature includes hidden-info "
            f"parameter: {param_names & forbidden}"
        )


def test_agent_module_source_does_not_reference_hidden_state():
    """agent source が env.hands / wall / state を参照していない。"""
    import mahjong_agent.agents as agents_pkg

    pkg_dir = Path(agents_pkg.__file__).parent
    py_files = list(pkg_dir.rglob("*.py"))
    assert py_files
    forbidden_tokens = (
        "env.hands",
        ".wall",
        "env.state",
        "full_state",
        "private_hand",
    )
    for f in py_files:
        text = f.read_text(encoding="utf-8")
        for tok in forbidden_tokens:
            assert tok not in text, (
                f"agents source {f.name} references forbidden token {tok!r}"
            )


def test_agent_module_source_does_not_reference_local_docs():
    """package source が local docs / 旧 repo 名へ言及していない。"""
    import mahjong_agent.agents as agents_pkg

    pkg_dir = Path(agents_pkg.__file__).parent
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
                f"agents source {f.name} mentions {tok!r}"
            )


# ----------------------------------------------------------------------
# Real RiichiEnv smoke
# ----------------------------------------------------------------------


def _play_one_game(agent_factory, *, max_seeds: int = 3, max_steps: int = 4000):
    """1 episode を ``agent_factory()`` で作った agent 4 人で進める helper。

    各 seed で reset し、全 player が同種の agent で動く。step 数の上限と
    seed 上限を超えても終わらなければ assertion failure とする (本来は
    short 4P 東風が数百 step 程度で終わる)。

    Returns
    -------
    (env, seed_used, steps_taken, num_decisions)
    """
    import riichienv

    from mahjong_agent.actions.convert import legal_actions_to_model_set
    from mahjong_agent.actions.resolver import resolve

    for seed in range(max_seeds):
        env = riichienv.RiichiEnv(riichienv.GameType.YON_TONPUSEN)
        env.reset(seed=seed)
        agents = {pid: agent_factory(seed + pid) for pid in range(env.num_players)}
        steps = 0
        n_decisions = 0
        while not env.is_done and steps < max_steps:
            steps += 1
            if env.current_claims:
                # 鳴き応答 phase: 各 claimant に対して agent decision を作る
                step_actions: dict[int, object] = {}
                for cp in list(env.current_claims.keys()):
                    obs = env.get_observation(cp)
                    la = list(obs.legal_actions())
                    if not la:
                        continue
                    # response phase では last_discarder = current_player
                    lset = legal_actions_to_model_set(
                        la,
                        actor=cp,
                        num_players=env.num_players,
                        last_discarder=env.current_player,
                        env_for_riichi=None,
                    )
                    dec = agents[cp].select_action(lset)
                    raw_seq = resolve(lset, dec.action)
                    assert len(raw_seq) == 1, (
                        "response phase action は 1-step のはず"
                    )
                    step_actions[cp] = raw_seq[0]
                    n_decisions += 1
                if not step_actions:
                    break
                env.step(step_actions)
                continue
            cp = env.current_player
            obs = env.get_observation(cp)
            la = list(obs.legal_actions())
            if not la:
                break
            lset = legal_actions_to_model_set(
                la,
                actor=cp,
                num_players=env.num_players,
                env_for_riichi=env,
            )
            dec = agents[cp].select_action(lset)
            raw_seq = resolve(lset, dec.action)
            n_decisions += 1
            # RiichiDiscard は 2-step だが、env.step 自体は 1 step ずつ。
            for raw_act in raw_seq:
                env.step({cp: raw_act})
                if env.is_done:
                    break
                # 次の raw act 適用前に再度同 player の合法手を再評価
                # (本来は env API の WaitAct で current_player が変わる場合
                # があるが、RiichiDiscard では Riichi -> 自家 Discard の流れ
                # で current_player は変わらない想定)。
        if env.is_done:
            return env, seed, steps, n_decisions
    raise AssertionError(
        f"agent smoke game did not terminate within {max_steps} steps for "
        f"seeds 0..{max_seeds - 1}"
    )


def test_random_agent_real_env_short_game_smoke():
    env, seed, steps, n_decisions = _play_one_game(
        lambda s: RandomAgent(seed=s)
    )
    assert env.is_done
    assert steps > 0
    assert n_decisions > 0


def test_rule_based_agent_real_env_short_game_smoke():
    env, seed, steps, n_decisions = _play_one_game(
        lambda s: RuleBasedBaselineAgent(seed=s)
    )
    assert env.is_done
    assert steps > 0
    assert n_decisions > 0


def test_real_env_reset_random_agent_resolver_smoke():
    """reset 直後の legal_actions から random agent の選択 -> resolver -> raw step。"""
    import riichienv

    from mahjong_agent.actions.convert import legal_actions_to_model_set
    from mahjong_agent.actions.resolver import resolve

    env = riichienv.RiichiEnv(riichienv.GameType.YON_TONPUSEN)
    env.reset(seed=7)
    cp = env.current_player
    la = list(env.get_observation(cp).legal_actions())
    lset = legal_actions_to_model_set(
        la, actor=cp, num_players=env.num_players, env_for_riichi=env
    )
    agent = RandomAgent(seed=7)
    dec = agent.select_action(lset)
    raw_seq = resolve(lset, dec.action)
    assert len(raw_seq) >= 1
    # ちゃんと env.step が走る
    env.step({cp: raw_seq[0]})


def test_real_env_reset_rule_based_agent_resolver_smoke():
    """reset 直後の legal_actions から rule_based agent の選択 -> resolver -> raw step。"""
    import riichienv

    from mahjong_agent.actions.convert import legal_actions_to_model_set
    from mahjong_agent.actions.resolver import resolve

    env = riichienv.RiichiEnv(riichienv.GameType.YON_TONPUSEN)
    env.reset(seed=11)
    cp = env.current_player
    la = list(env.get_observation(cp).legal_actions())
    lset = legal_actions_to_model_set(
        la, actor=cp, num_players=env.num_players, env_for_riichi=env
    )
    agent = RuleBasedBaselineAgent()
    dec = agent.select_action(lset)
    raw_seq = resolve(lset, dec.action)
    assert len(raw_seq) >= 1
    env.step({cp: raw_seq[0]})
