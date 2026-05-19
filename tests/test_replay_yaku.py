"""Tests for ``mahjong_agent.evaluation.replay_yaku``.

- ``collect_round_yaku_records`` の smoke (実 YON_TONPUSEN log で seed scan)。
- ``RoundTracker.finalize_with_round_yaku_records`` の synthetic unit test。
- ``SelfPlayRunner`` 経由で mid-game winner sample に yaku target が入ること。
- replay 失敗時に runner が crash せず fallback すること。
- hidden info を sample / encoder に流していないこと (source guard)。
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import riichienv

from mahjong_agent.actions.types import ActionFamily
from mahjong_agent.agents import RandomAgent
from mahjong_agent.evaluation import (
    RoundTracker,
    RoundYakuRecord,
    SeatAgents,
    SelfPlayConfig,
    SelfPlayRunner,
    collect_round_yaku_records,
    make_initial_sample,
)
from mahjong_agent.targets.yaku import NUM_YAKU

_ENCODER_OBS_DIM = 4


# ----------------------------------------------------------------------
# RoundYakuRecord / collect_round_yaku_records — smoke / synth
# ----------------------------------------------------------------------


def test_collect_round_yaku_records_returns_records_on_hora_log():
    """実 env を直接駆動して mjai_log を作り、collect_round_yaku_records が
    1 件以上 RoundYakuRecord を返すこと。"""
    import random

    from mahjong_agent.actions.convert import legal_actions_to_model_set
    from mahjong_agent.actions.resolver import resolve

    def _drive_until_hora(max_seeds: int = 300):
        for seed in range(max_seeds):
            rng = random.Random(seed)
            env = riichienv.RiichiEnv(riichienv.GameType.YON_TONPUSEN)
            env.reset(seed=seed)
            last_discarder: int | None = None
            for _ in range(4000):
                if env.is_done:
                    break
                if env.current_claims:
                    claimants = sorted(int(p) for p in env.current_claims.keys())
                    atp: dict[int, object] = {}
                    fams: dict[int, ActionFamily] = {}
                    for cp in claimants:
                        obs = env.get_observation(cp)
                        legal_raw = list(obs.legal_actions())
                        las = legal_actions_to_model_set(
                            legal_raw,
                            actor=cp,
                            num_players=4,
                            last_discarder=last_discarder,
                            env_for_riichi=None,
                        )
                        actions = list(las.normal_discard.values()) + list(
                            las.candidates
                        )
                        if not actions:
                            continue
                        ma = rng.choice(actions)
                        raw_seq = resolve(las, ma)
                        if len(raw_seq) != 1:
                            continue
                        atp[cp] = raw_seq[0]
                        fams[cp] = ma.family
                    if atp:
                        env.step(atp)
                    else:
                        try:
                            env.step({})
                        except Exception:
                            break
                    for f in fams.values():
                        if f in (
                            ActionFamily.CHI,
                            ActionFamily.PON,
                            ActionFamily.DAIMINKAN,
                        ):
                            last_discarder = None
                else:
                    cp = int(env.current_player)
                    obs = env.get_observation(cp)
                    legal_raw = list(obs.legal_actions())
                    las = legal_actions_to_model_set(
                        legal_raw,
                        actor=cp,
                        num_players=4,
                        last_discarder=last_discarder,
                        env_for_riichi=env,
                    )
                    actions = list(las.normal_discard.values()) + list(
                        las.candidates
                    )
                    if not actions:
                        break
                    ma = rng.choice(actions)
                    raw_seq = resolve(las, ma)
                    for raw_act in raw_seq:
                        env.step({cp: raw_act})
                        if env.is_done:
                            break
                    if ma.family in (
                        ActionFamily.NORMAL_DISCARD,
                        ActionFamily.RIICHI_DISCARD,
                    ):
                        last_discarder = cp
            if any(ev.get("type") == "hora" for ev in env.mjai_log):
                return env, seed
        return None, -1

    env, seed = _drive_until_hora()
    if env is None:
        pytest.skip("no hora found in seed scan; non-deterministic riichienv wall")
    records = collect_round_yaku_records(list(env.mjai_log))
    assert isinstance(records, tuple)
    assert len(records) >= 1
    for r in records:
        assert isinstance(r, RoundYakuRecord)
        assert 0 <= int(r.winner_seat) < 4
        assert r.han >= 0
        assert r.fu >= 0
        # yaku_ids は riichienv の yaku id 集合 (1..49 で hardcode vocab 内)
        for yid in r.yaku_ids:
            assert 1 <= int(yid) <= 100  # 緩めの sanity (vocab 範囲は別 test)


def test_round_yaku_record_is_frozen_dataclass():
    import dataclasses

    rec = RoundYakuRecord(
        round_index=1,
        winner_seat=2,
        yaku_ids=(2, 14),
        han=2,
        fu=30,
        yakuman=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.han = 5  # type: ignore[misc]


# ----------------------------------------------------------------------
# RoundTracker.finalize_with_round_yaku_records — synthetic
# ----------------------------------------------------------------------


def _make_dummy_sample(*, round_idx: int, player_id: int) -> object:
    return make_initial_sample(
        episode_id="ep",
        round_idx=round_idx,
        step_id=0,
        player_id=player_id,
        decision_family=ActionFamily.NORMAL_DISCARD.value,
        actor_type="random",
        observation_feat=np.zeros(_ENCODER_OBS_DIM, dtype=np.float32),
        discard_mask=np.zeros(34, dtype=np.float32),
        candidate_features=np.zeros((0, 4), dtype=np.float32),
        selected_discard_tile_type=5,
        selected_candidate_index=-1,
    )


def test_finalize_with_round_yaku_records_backfills_mid_round_winner():
    """同 round の winner sample に yaku_target / loss_mask / han / fu が入ること。"""
    t = RoundTracker(num_players=4)
    s = _make_dummy_sample(round_idx=2, player_id=1)
    t.samples.append(s)
    rec = RoundYakuRecord(
        round_index=2,
        winner_seat=1,
        yaku_ids=(2, 14, 30),  # Riichi / Pinfu / Ippatsu の id
        han=3,
        fu=40,
        yakuman=False,
    )
    applied = t.finalize_with_round_yaku_records((rec,))
    assert applied == 1
    out = t.samples[0]
    assert out.yaku_loss_mask == pytest.approx(1.0)
    assert out.han == 3
    assert out.fu == 40
    nonzero = int((out.yaku_target > 0).sum())
    assert nonzero == 3
    assert out.yaku_target.shape == (NUM_YAKU,)


def test_finalize_with_round_yaku_records_handles_double_ron():
    """同 round 複数 winner (= 同 round_index の 2 record) が両方反映されること。"""
    t = RoundTracker(num_players=4)
    s1 = _make_dummy_sample(round_idx=3, player_id=0)
    s2 = _make_dummy_sample(round_idx=3, player_id=2)
    t.samples.extend([s1, s2])
    recs = (
        RoundYakuRecord(
            round_index=3,
            winner_seat=0,
            yaku_ids=(7,),  # Yakuhai Haku
            han=1,
            fu=30,
            yakuman=False,
        ),
        RoundYakuRecord(
            round_index=3,
            winner_seat=2,
            yaku_ids=(12,),  # Tanyao
            han=1,
            fu=30,
            yakuman=False,
        ),
    )
    applied = t.finalize_with_round_yaku_records(recs)
    assert applied == 2
    assert t.samples[0].yaku_loss_mask == pytest.approx(1.0)
    assert t.samples[1].yaku_loss_mask == pytest.approx(1.0)
    # それぞれ別の yaku が立っている
    assert int(t.samples[0].yaku_target[0]) == 0  # idx 0 = Menzen Tsumo, not in record
    nonzero_0 = int((t.samples[0].yaku_target > 0).sum())
    nonzero_2 = int((t.samples[1].yaku_target > 0).sum())
    assert nonzero_0 == 1
    assert nonzero_2 == 1


def test_finalize_with_round_yaku_records_skips_missing_samples():
    """winner sample が無い round / pid の record は silent にスキップ。"""
    t = RoundTracker(num_players=4)
    s = _make_dummy_sample(round_idx=0, player_id=1)
    t.samples.append(s)
    recs = (
        RoundYakuRecord(
            round_index=99,  # 該当 round 無し
            winner_seat=3,
            yaku_ids=(2,),
            han=1,
            fu=30,
            yakuman=False,
        ),
    )
    applied = t.finalize_with_round_yaku_records(recs)
    assert applied == 0
    # 元 sample は変更されない
    assert float(t.samples[0].yaku_loss_mask) == 0.0


def test_finalize_with_round_yaku_records_empty_no_op():
    t = RoundTracker(num_players=4)
    s = _make_dummy_sample(round_idx=0, player_id=0)
    t.samples.append(s)
    applied = t.finalize_with_round_yaku_records(())
    assert applied == 0
    assert float(t.samples[0].yaku_loss_mask) == 0.0


# ----------------------------------------------------------------------
# SelfPlayRunner 経由の mid-game backfill regression
# ----------------------------------------------------------------------


def _find_episode_with_mid_game_hora():
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_TONPUSEN,
            max_steps_per_game=8000,
        )
    )
    for s in range(0, 200):
        sa = SeatAgents.homogeneous(
            RandomAgent(seed=s), actor_type="random", num_players=4
        )
        res = runner.run_episode(sa, seed=s)
        if res.crash_context is not None:
            continue
        winners = [smp for smp in res.samples if smp.yaku_loss_mask > 0]
        if not winners:
            continue
        winner_rounds = {smp.round_id for smp in winners}
        # mid-game: at least one winner round is NOT the last
        if any(r < res.num_rounds - 1 for r in winner_rounds):
            return res, s
    return None, -1


def test_self_play_backfills_mid_round_yaku_via_replay():
    """SelfPlayRunner 経由で、final round 以外の winner sample に yaku target が
    入ること (= ISSUE-0014 の主目的)。"""
    res, seed = _find_episode_with_mid_game_hora()
    if res is None:
        pytest.skip(
            "no mid-game hora found in seed scan (random play rarely wins early)"
        )
    winners = [smp for smp in res.samples if smp.yaku_loss_mask > 0]
    assert winners, "expected at least one winner sample"
    mid_game_winners = [smp for smp in winners if smp.round_id < res.num_rounds - 1]
    assert mid_game_winners, (
        f"seed={seed}: winners exist but all in final round "
        f"(rounds={[smp.round_id for smp in winners]}, num_rounds={res.num_rounds})"
    )
    for smp in mid_game_winners:
        assert smp.yaku_loss_mask == pytest.approx(1.0)
        assert smp.han >= 0
        assert smp.fu >= 0
        nonzero = int((smp.yaku_target > 0).sum())
        # 和了している以上、最低 1 yaku は立つ (例: 立直・断幺九・役牌)
        assert nonzero >= 1


def test_self_play_runner_falls_back_when_replay_collector_fails(monkeypatch):
    """``collect_round_yaku_records`` が例外を上げても runner は crash せず、
    既存 ``finalize_game_with_win_results`` fallback で完走すること。"""
    import mahjong_agent.evaluation.runner as runner_mod

    def _boom(_mjai_log):
        raise RuntimeError("synthetic collector failure")

    monkeypatch.setattr(runner_mod, "collect_round_yaku_records", _boom)
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    sa = SeatAgents.homogeneous(
        RandomAgent(seed=0), actor_type="random", num_players=4
    )
    res = runner.run_episode(sa, seed=0)
    # game 自体は走り切る
    assert res.crash_context is None
    assert res.num_rounds >= 1
    # samples は生成される (yaku target は collector 失敗で empty fallback)
    assert len(res.samples) > 0


# ----------------------------------------------------------------------
# Hidden info guards (mjai_log / tehais / hands を model 経路に流さない)
# ----------------------------------------------------------------------


def test_decision_sample_metadata_excludes_hidden_log_payloads():
    """sample.metadata に mjai_log / tehais / hands / wall が混入していないこと。"""
    runner = SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )
    sa = SeatAgents.homogeneous(
        RandomAgent(seed=0), actor_type="random", num_players=4
    )
    res = runner.run_episode(sa, seed=0)
    assert res.crash_context is None
    forbidden_keys = {"mjai_log", "tehais", "hands", "wall", "state", "win_results"}
    for smp in res.samples[:50]:
        md = dict(smp.metadata or {})
        assert forbidden_keys.isdisjoint(set(md.keys())), (
            f"sample.metadata contains hidden-info keys: "
            f"{forbidden_keys & set(md.keys())}"
        )


def test_replay_yaku_source_does_not_leak_into_encoder():
    """``replay_yaku.py`` が ``encoder`` / ``model`` を import しないこと
    (= post-game label extraction 専用境界の static guard)。"""
    import mahjong_agent.evaluation.replay_yaku as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    forbidden_imports = (
        "from mahjong_agent.encoders",
        "from mahjong_agent.models",
        "import mahjong_agent.encoders",
        "import mahjong_agent.models",
    )
    for tok in forbidden_imports:
        assert tok not in src, f"replay_yaku.py imports {tok!r}"


def test_collect_round_yaku_records_signature_only_takes_mjai_log():
    """API surface guard: signature が ``mjai_log`` 引数のみで、env / encoder などを
    受け取らない (= hidden state を引き回さない境界の証跡)。"""
    sig = inspect.signature(collect_round_yaku_records)
    params = list(sig.parameters.keys())
    assert params == ["mjai_log"]
