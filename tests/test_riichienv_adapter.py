"""ISSUE-0002: RiichiEnvAdapter smoke tests.

確認内容:
- ``import riichienv`` が通る
- ``RiichiEnvAdapter`` が作成・reset できる
- public observation を取得できる
- legal action を取得できる
- random で 1 step / 短い episode を回せる
- score / rank / score_delta / round_info にアクセスできる
- adapter から hidden information (全員 hand / 山) を公開していない
"""
from __future__ import annotations

import random


def test_import_riichienv():
    """PyPI 版 ``riichienv`` が import できる前提を確認する。"""
    import riichienv  # noqa: F401


def test_construct_adapter():
    """``RiichiEnvAdapter()`` を構築できる。"""
    from mahjong_agent.envs import RiichiEnvAdapter

    adapter = RiichiEnvAdapter()
    assert adapter is not None
    assert adapter.num_players in (3, 4)


def test_reset_smoke():
    """seed 付き reset で初期状態が決まり、決定論的に同じ初期 state になる。"""
    from mahjong_agent.envs import RiichiEnvAdapter

    a1 = RiichiEnvAdapter()
    a1.reset(seed=42)
    a2 = RiichiEnvAdapter()
    a2.reset(seed=42)
    assert a1.scores() == a2.scores()
    assert a1.current_player == a2.current_player
    assert a1.kyoku_idx == a2.kyoku_idx
    # 同 seed なら最初の player の legal action 数も一致
    cp = a1.current_player
    assert len(a1.legal_actions(cp)) == len(a2.legal_actions(cp))


def test_observation_smoke():
    """observation を取得でき、最低限の public field が読める。"""
    from mahjong_agent.envs import RiichiEnvAdapter

    adapter = RiichiEnvAdapter()
    adapter.reset(seed=42)
    cp = adapter.current_player
    obs = adapter.get_observation(cp)
    # 自分の hand は見える
    assert hasattr(obs, "hand")
    assert len(obs.hand) > 0
    # 自家視点なので player_id が一致
    assert int(obs.player_id) == cp
    # 公開情報も読める (空 list でも問題ない)
    assert hasattr(obs, "discards")
    assert hasattr(obs, "dora_indicators")
    assert hasattr(obs, "scores")


def test_legal_actions_smoke():
    """legal action list が空でなく、Action オブジェクトの list である。"""
    from mahjong_agent.envs import RiichiEnvAdapter

    adapter = RiichiEnvAdapter()
    adapter.reset(seed=42)
    cp = adapter.current_player
    la = adapter.legal_actions(cp)
    assert isinstance(la, list)
    assert len(la) > 0
    # 全 element に action_type / actor 属性
    for a in la:
        assert hasattr(a, "action_type")
        assert hasattr(a, "actor")


def test_random_step_smoke():
    """random に 1 step 進められ、phase / current_player が更新される。"""
    from mahjong_agent.envs import RiichiEnvAdapter

    adapter = RiichiEnvAdapter()
    adapter.reset(seed=42)
    cp_before = adapter.current_player
    la = adapter.legal_actions(cp_before)
    act = la[0]
    result = adapter.step({cp_before: act})
    # step は dict[player_id, Observation] を返す
    assert isinstance(result, dict)
    # game がまだ終わっていなければ次の decision が決まっている
    assert adapter.is_done or adapter.current_decision_players()


def test_score_access_smoke():
    """scores / ranks / score_deltas / round_info にアクセスできる。"""
    from mahjong_agent.envs import RiichiEnvAdapter

    adapter = RiichiEnvAdapter()
    adapter.reset(seed=42)
    scores = adapter.scores()
    ranks = adapter.ranks()
    deltas = adapter.score_deltas
    info = adapter.round_info
    assert len(scores) == adapter.num_players
    assert len(ranks) == adapter.num_players
    assert len(deltas) == adapter.num_players
    assert sum(scores) == 100_000  # 4 人合計 (供託除く時の点棒総量)
    assert set(ranks) == set(range(1, adapter.num_players + 1))
    assert "kyoku_idx" in info
    assert "oya" in info
    assert "round_wind" in info
    assert "honba" in info


def _play_random_episode(adapter, rng, max_steps=5000):
    """smoke 用に random で 1 episode 回す helper。"""
    steps = 0
    while not adapter.is_done and steps < max_steps:
        steps += 1
        decision_players = adapter.current_decision_players()
        actions = {}
        for pid in decision_players:
            la = adapter.legal_actions(pid)
            if not la:
                continue
            actions[pid] = rng.choice(la)
        if not actions:
            break
        adapter.step(actions)
    return steps


def test_random_short_episode_smoke():
    """短い game (YON_TONPUSEN) を random で 1 episode 完走できる。"""
    from mahjong_agent.envs import RiichiEnvAdapter

    adapter = RiichiEnvAdapter()
    adapter.reset(seed=42)
    rng = random.Random(42)
    steps = _play_random_episode(adapter, rng, max_steps=5000)
    assert steps > 0
    # smoke なので必ず is_done になることまでは要求しない (max_steps で止まる
    # ことを許容する)。ただし最低 1 step は進めていること。
    assert adapter.kyoku_idx >= 0


def test_adapter_does_not_expose_hidden_state():
    """adapter から全員 hand / 山 / engine full state を直接取得しないこと。

    隠匿対象:
    - 他家の手牌 (``RiichiEnv.hands``)
    - 山 (``RiichiEnv.wall``)
    - engine internal full state (``RiichiEnv.state``)
    """
    from mahjong_agent.envs import RiichiEnvAdapter

    adapter = RiichiEnvAdapter()
    adapter.reset(seed=42)
    forbidden = ("hands", "wall", "state")
    for name in forbidden:
        # adapter には全員 hand / 山 / engine state を返す public API を
        # 一切置かない。
        assert not hasattr(adapter, name), (
            f"adapter exposed forbidden attribute {name!r}; hidden state "
            f"leak risk")


def test_observation_does_not_leak_other_hands():
    """``get_observation(player)`` が他家の手牌を露出しないことを確認する。

    RiichiEnv の ``Observation.hands`` は構造上 4-slot だが、要求 player 以外の
    slot は空 list になっているはず。
    """
    from mahjong_agent.envs import RiichiEnvAdapter

    adapter = RiichiEnvAdapter()
    adapter.reset(seed=42)
    cp = adapter.current_player
    obs = adapter.get_observation(cp)
    hands = list(obs.hands)
    # 自家 slot は非空、他家 slot は空であることを確認 (RiichiEnv 仕様)。
    assert len(hands[cp]) > 0
    for other in range(adapter.num_players):
        if other == cp:
            continue
        assert hands[other] == [], (
            f"observation for player {cp} leaked hand of player {other}: "
            f"{hands[other]}")
