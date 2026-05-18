"""Tests for model-facing action abstraction and RiichiEnv resolver.

確認内容:
- 通常打牌の tile_type extraction
- candidate kind mapping (Tsumo / Ron / Chi / Pon / Daiminkan / Ankan /
  Kakan / KyushuKyuhai / Pass)
- RiichiDiscard candidate の列挙 (env.clone() で Riichi 後 discard を見る)
- 赤牌 / 通常牌 同 tile_type の場合に通常牌優先 resolver
- legal action list を並べ替えても semantic key が同じ candidate を選べる
  (= index 非依存)
- resolver が legal raw action を返す
- 実 RiichiEnv reset 後の legal_actions を変換する smoke
"""
from __future__ import annotations

import random
from typing import Any

import pytest

# ----------------------------------------------------------------------
# Fake Action fixture: riichienv.Action は Rust binding なので任意値を
# 作りにくい。tests では Python の dummy を用意し、変換器に渡す。
# 変換器は ``action_type`` / ``tile`` / ``consume_tiles`` / ``actor`` を
# 参照するだけなので、dummy で十分。
# ----------------------------------------------------------------------


class _FakeAction:
    """riichienv.Action 互換の最小 dummy。tests でのみ使う。"""

    def __init__(
        self,
        action_type: Any,
        tile: int | None = None,
        consume_tiles: list[int] | None = None,
        actor: int | None = 0,
    ):
        self.action_type = action_type
        self.tile = tile
        self.consume_tiles = list(consume_tiles or [])
        self.actor = actor

    def __repr__(self) -> str:
        return (
            f"_FakeAction({self.action_type}, tile={self.tile}, "
            f"consume_tiles={self.consume_tiles}, actor={self.actor})"
        )


def _at():
    """riichienv.ActionType への shortcut。"""
    import riichienv

    return riichienv.ActionType


# ----------------------------------------------------------------------
# normal discard
# ----------------------------------------------------------------------


def test_normal_discard_tile_type_extraction():
    """tile_id をすべて tile_type にまとめ、tile_type ごとに 1 ModelAction にする。"""
    from mahjong_agent.actions import legal_actions_to_model_set

    AT = _at()
    # 1m / 2m / 5m (normal+red) / 9p の discard
    raw = [
        _FakeAction(AT.DISCARD, tile=0, actor=0),   # 1m
        _FakeAction(AT.DISCARD, tile=4, actor=0),   # 2m
        _FakeAction(AT.DISCARD, tile=16, actor=0),  # 5m red
        _FakeAction(AT.DISCARD, tile=17, actor=0),  # 5m normal
        _FakeAction(AT.DISCARD, tile=68, actor=0),  # 9p (tile_type 17)
    ]
    lset = legal_actions_to_model_set(raw, actor=0)
    tt_list = lset.normal_discard_tile_types()
    assert tt_list == [0, 1, 4, 17]
    assert lset.can_normal_discard


def test_red_tile_priority_resolves_to_normal():
    """同 tile_type に red と normal の discard があれば normal が選ばれる。"""
    from mahjong_agent.actions import (
        is_red_tile_id,
        legal_actions_to_model_set,
        resolve_normal_discard,
    )

    AT = _at()
    raw = [
        _FakeAction(AT.DISCARD, tile=16, actor=0),  # 5m red
        _FakeAction(AT.DISCARD, tile=18, actor=0),  # 5m normal (other copy)
        _FakeAction(AT.DISCARD, tile=19, actor=0),  # 5m normal (other copy)
    ]
    lset = legal_actions_to_model_set(raw, actor=0)
    # tile_type 4 のみ
    assert list(lset.normal_discard.keys()) == [4]
    resolved = resolve_normal_discard(lset, tile_type=4)
    assert len(resolved) == 1
    chosen_tile = resolved[0].tile
    # 通常牌が選ばれる (red が選ばれない)
    assert not is_red_tile_id(chosen_tile)


def test_only_red_tile_falls_back_to_red():
    """red しか legal でないとき (normal が手にない場合) は red を使う。"""
    from mahjong_agent.actions import (
        is_red_tile_id,
        legal_actions_to_model_set,
        resolve_normal_discard,
    )

    AT = _at()
    raw = [
        _FakeAction(AT.DISCARD, tile=16, actor=0),  # 5m red のみ
    ]
    lset = legal_actions_to_model_set(raw, actor=0)
    assert list(lset.normal_discard.keys()) == [4]
    resolved = resolve_normal_discard(lset, tile_type=4)
    assert is_red_tile_id(resolved[0].tile)


# ----------------------------------------------------------------------
# candidate kind mapping
# ----------------------------------------------------------------------


def test_candidate_kind_mapping():
    """主要 ActionType -> ActionFamily の mapping をすべて確認する。"""
    from mahjong_agent.actions import ActionFamily, legal_actions_to_model_set

    AT = _at()
    raw = [
        _FakeAction(AT.TSUMO, tile=None, actor=0),
        _FakeAction(AT.RON, tile=64, actor=0),
        _FakeAction(AT.CHI, tile=66, consume_tiles=[59, 62], actor=0),
        _FakeAction(AT.PON, tile=81, consume_tiles=[80, 82], actor=0),
        _FakeAction(AT.DAIMINKAN, tile=39, consume_tiles=[36, 37, 38], actor=0),
        _FakeAction(AT.ANKAN, tile=36, consume_tiles=[36, 37, 38, 39], actor=0),
        _FakeAction(AT.KAKAN, tile=20, consume_tiles=[20, 21, 22], actor=0),
        _FakeAction(AT.KYUSHU_KYUHAI, tile=None, actor=0),
        _FakeAction(AT.PASS, tile=None, actor=0),
    ]
    lset = legal_actions_to_model_set(raw, actor=0)
    families = {c.family for c in lset.candidates}
    assert families == {
        ActionFamily.TSUMO,
        ActionFamily.RON,
        ActionFamily.CHI,
        ActionFamily.PON,
        ActionFamily.DAIMINKAN,
        ActionFamily.ANKAN,
        ActionFamily.KAKAN,
        ActionFamily.KYUSHU_KYUHAI,
        ActionFamily.PASS,
    }
    # tile_id ベースではなく tile_type ベース
    chi = next(c for c in lset.candidates if c.family == ActionFamily.CHI)
    assert chi.key.tile_type == 66 // 4
    assert chi.key.consume_tile_types == tuple(sorted([59 // 4, 62 // 4]))


def test_chi_with_multiple_consume_patterns_have_distinct_keys():
    """同 tile_type の Chi でも consume が違うなら別 candidate になる。"""
    from mahjong_agent.actions import ActionFamily, legal_actions_to_model_set

    AT = _at()
    # 5m (id 16/17/18/19) を Chi 2 通り: (4m-6m) and (3m-4m)
    # 4m=12-15, 3m=8-11, 6m=20-23
    raw = [
        _FakeAction(AT.CHI, tile=17, consume_tiles=[15, 20], actor=0),  # 4m-6m
        _FakeAction(AT.CHI, tile=17, consume_tiles=[8, 12], actor=0),    # 3m-4m
    ]
    lset = legal_actions_to_model_set(raw, actor=0)
    chi_keys = [c.key for c in lset.candidates if c.family == ActionFamily.CHI]
    assert len(chi_keys) == 2
    assert chi_keys[0] != chi_keys[1]
    assert {tuple(k.consume_tile_types) for k in chi_keys} == {
        tuple(sorted([15 // 4, 20 // 4])),
        tuple(sorted([8 // 4, 12 // 4])),
    }


# ----------------------------------------------------------------------
# semantic key stability across reordering
# ----------------------------------------------------------------------


def test_semantic_key_stable_under_legal_action_reordering():
    """legal action list を並べ替えても、resolver は同じ key で同じ raw_action を取れる。"""
    from mahjong_agent.actions import (
        ActionFamily,
        ActionKey,
        legal_actions_to_model_set,
        resolve_candidate,
    )

    AT = _at()
    raw_a = [
        _FakeAction(AT.PASS, tile=None, actor=0),
        _FakeAction(AT.CHI, tile=66, consume_tiles=[59, 62], actor=0),
        _FakeAction(AT.PON, tile=81, consume_tiles=[80, 82], actor=0),
    ]
    raw_b = list(reversed(raw_a))
    lset_a = legal_actions_to_model_set(raw_a, actor=0)
    lset_b = legal_actions_to_model_set(raw_b, actor=0)

    chi_key = ActionKey(
        family=ActionFamily.CHI,
        tile_type=66 // 4,
        consume_tile_types=tuple(sorted([59 // 4, 62 // 4])),
        target_rel_seat=None,
    )
    raw_a_resolved = resolve_candidate(lset_a, chi_key)
    raw_b_resolved = resolve_candidate(lset_b, chi_key)
    # 同 semantic key -> 同 raw_action.tile (Chi の主牌 id)
    assert raw_a_resolved[0].tile == 66
    assert raw_b_resolved[0].tile == 66
    # candidates tuple そのものも family/tile_type/consume の sort key で
    # 安定なので順序が一致する
    assert [c.key for c in lset_a.candidates] == [c.key for c in lset_b.candidates]


def test_resolver_returns_legal_raw_action():
    """resolve(model_action) が returns する raw_actions が元の legal set に
    含まれていることを確認する。"""
    from mahjong_agent.actions import legal_actions_to_model_set, resolve

    AT = _at()
    raw = [
        _FakeAction(AT.DISCARD, tile=0, actor=0),
        _FakeAction(AT.PASS, tile=None, actor=0),
    ]
    lset = legal_actions_to_model_set(raw, actor=0)
    # normal discard
    nd_action = lset.normal_discard[0]
    resolved = resolve(lset, nd_action)
    assert resolved == nd_action.raw_actions()
    assert resolved[0] in raw
    # Pass
    pass_cand = next(c for c in lset.candidates)
    resolved2 = resolve(lset, pass_cand)
    assert resolved2[0] in raw


def test_resolve_missing_key_raises():
    """legal set に無い semantic key は KeyError。"""
    from mahjong_agent.actions import (
        ActionFamily,
        ActionKey,
        legal_actions_to_model_set,
        resolve_candidate,
        resolve_normal_discard,
    )

    AT = _at()
    raw = [_FakeAction(AT.DISCARD, tile=0, actor=0)]
    lset = legal_actions_to_model_set(raw, actor=0)
    with pytest.raises(KeyError):
        resolve_normal_discard(lset, tile_type=33)
    with pytest.raises(KeyError):
        resolve_candidate(lset, ActionKey(family=ActionFamily.TSUMO))


# ----------------------------------------------------------------------
# target_rel_seat
# ----------------------------------------------------------------------


def test_target_rel_seat_for_chi_and_ron():
    """Chi / Pon / Daiminkan / Ron は last_discarder への rel_seat を持つ。"""
    from mahjong_agent.actions import ActionFamily, legal_actions_to_model_set

    AT = _at()
    raw = [
        _FakeAction(AT.CHI, tile=66, consume_tiles=[59, 62], actor=1),
        _FakeAction(AT.PASS, tile=None, actor=1),
    ]
    # actor=1, last_discarder=0 (kamicha = rel_seat 3)
    lset = legal_actions_to_model_set(
        raw, actor=1, num_players=4, last_discarder=0
    )
    chi = next(c for c in lset.candidates if c.family == ActionFamily.CHI)
    assert chi.key.target_rel_seat == (0 - 1) % 4  # = 3 (kamicha)
    # Pass は target_rel_seat=None
    pass_c = next(c for c in lset.candidates if c.family == ActionFamily.PASS)
    assert pass_c.key.target_rel_seat is None


# ----------------------------------------------------------------------
# Riichi discard
# ----------------------------------------------------------------------


def _find_riichi_state(max_seeds=300):
    """Riichi が legal な状態の env / observation を探す helper。"""
    import riichienv

    for seed in range(max_seeds):
        env = riichienv.RiichiEnv(riichienv.GameType.YON_TONPUSEN)
        env.reset(seed=seed)
        rng = random.Random(seed)
        steps = 0
        while not env.is_done and steps < 1500:
            steps += 1
            if env.current_claims:
                acts = {}
                for cp in env.current_claims:
                    la = env.get_observation(cp).legal_actions()
                    if la:
                        acts[cp] = rng.choice(la)
                if not acts:
                    break
                env.step(acts)
                continue
            cp = env.current_player
            obs = env.get_observation(cp)
            la = obs.legal_actions()
            if not la:
                break
            riichi_acts = [
                a for a in la if a.action_type == riichienv.ActionType.RIICHI
            ]
            if riichi_acts:
                return env, cp, la
            env.step({cp: rng.choice(la)})
    return None, None, None


def test_riichi_discard_candidate_smoke():
    """Riichi 可能な実 env で RiichiDiscard candidate が tile_type 別に列挙される。"""
    from mahjong_agent.actions import ActionFamily, legal_actions_to_model_set

    env, cp, la = _find_riichi_state()
    if env is None:
        pytest.skip("Riichi-legal state not found in 300 seeds")
    lset = legal_actions_to_model_set(
        la, actor=cp, env_for_riichi=env
    )
    riichi_cands = [
        c for c in lset.candidates if c.family == ActionFamily.RIICHI_DISCARD
    ]
    assert len(riichi_cands) >= 1
    # 各 RiichiDiscard candidate は 2-step raw (Riichi, Discard)
    for c in riichi_cands:
        raws = c.raw_actions()
        assert len(raws) == 2
        import riichienv as r

        assert raws[0].action_type == r.ActionType.RIICHI
        assert raws[1].action_type == r.ActionType.DISCARD
        # tile_type 一致
        assert c.key.tile_type == raws[1].tile // 4


def test_riichi_candidate_not_generated_without_env():
    """env_for_riichi=None なら RiichiDiscard candidate は生成しない。"""
    from mahjong_agent.actions import ActionFamily, legal_actions_to_model_set

    AT = _at()
    raw = [
        _FakeAction(AT.RIICHI, tile=None, actor=0),
        _FakeAction(AT.DISCARD, tile=0, actor=0),
    ]
    lset = legal_actions_to_model_set(raw, actor=0, env_for_riichi=None)
    riichi_cands = [
        c for c in lset.candidates if c.family == ActionFamily.RIICHI_DISCARD
    ]
    assert riichi_cands == []


# ----------------------------------------------------------------------
# end-to-end smoke against real RiichiEnv
# ----------------------------------------------------------------------


def test_real_env_reset_legal_actions_convert_smoke():
    """実 RiichiEnv reset 後の legal_actions を model set に変換できる。

    初期 turn の player は通常 14 種の discard を持つ。"""
    import riichienv

    from mahjong_agent.actions import legal_actions_to_model_set

    env = riichienv.RiichiEnv(riichienv.GameType.YON_TONPUSEN)
    env.reset(seed=42)
    cp = env.current_player
    la = list(env.get_observation(cp).legal_actions())
    lset = legal_actions_to_model_set(
        la, actor=cp, num_players=env.num_players, env_for_riichi=env
    )
    # 起家 14 牌 → 全 tile_type 数 (重複あり) で 1..14。
    assert lset.can_normal_discard
    assert 0 < len(lset.normal_discard) <= 14
    # 通常初手で副露/和了 candidates は無いはず
    from mahjong_agent.actions import ActionFamily

    non_discard_families = {
        c.family
        for c in lset.candidates
        if c.family != ActionFamily.RIICHI_DISCARD
    }
    assert non_discard_families <= {ActionFamily.PASS}
