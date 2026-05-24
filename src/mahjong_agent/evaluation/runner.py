"""Self-play / evaluation episode runner.

このモジュールは:

1. ``RiichiEnvAdapter`` (= PyPI ``riichienv``) を初期化して 1 game を進める。
2. ``Phase.WaitAct`` / ``Phase.WaitResponse`` で agent decision を作る。
3. ``last_discarder`` を loop 側で正しく追跡する (response phase で
   ``LegalActionSet`` の ``target_rel_seat`` を正しく出すために必要)。
4. ``PublicObservationEncoder`` で observation / candidate / discard mask を
   encode し、``DecisionSample`` を生成する。
5. ``RoundTracker`` を介して mjai_log から round-end を検出、round-level
   facts を pending samples へ backfill する。
6. game 終了時に terminated flag と final round の yaku/han/fu を backfill。
7. crash 時には seed/step/player/phase/legal action summary を ``crash_context``
   として ``EpisodeResult`` に残し、game をスキップする。

public-only 方針:

- engine 内部の hidden state (全 player の手牌 / 山 / engine state) を一切
  参照しない。
- observation は public-only encoder の出力を sample に保存する。
- reward は ``round_score_delta * 1e-4`` (round-level)。
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import riichienv

from mahjong_agent.actions.convert import legal_actions_to_model_set
from mahjong_agent.actions.resolver import resolve
from mahjong_agent.actions.types import ActionFamily, LegalActionSet, ModelAction
from mahjong_agent.data.types import DecisionSample
from mahjong_agent.encoders.public_observation import PublicObservationEncoder
from mahjong_agent.evaluation.replay_yaku import collect_round_yaku_records
from mahjong_agent.evaluation.round_tracker import (
    RoundTracker,
    make_initial_sample,
)
from mahjong_agent.evaluation.seat_agents import SeatAgents

_DEFAULT_GAME_TYPE = riichienv.GameType.YON_TONPUSEN
_WALL_SIZE: int = 136  # 4 人麻雀の総牌数 (0..135)


def _generate_wall(wall_seed: int) -> list[int]:
    """``wall_seed`` から決定論的に shuffle した 136 牌の wall を返す。

    内部で ``random.Random(wall_seed)`` を新規作成して使うので、global rng
    状態に依存しない。
    """
    rng = random.Random(int(wall_seed))
    wall = list(range(_WALL_SIZE))
    rng.shuffle(wall)
    return wall


def _wall_digest(wall: list[int]) -> str:
    """wall 内容の reproduction digest (SHA-256 hex, 先頭 16 文字)。

    wall 自体は sample / metadata に **保存しない** こと。reproduction には
    ``wall_seed`` を保存し、必要に応じて再生成する。digest は同 seed-同 wall
    の整合性チェックや experiment report の照合用。
    """
    h = hashlib.sha256()
    for t in wall:
        h.update(int(t).to_bytes(2, "little", signed=False))
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class SelfPlayConfig:
    """1 episode を回すための config。

    Attributes
    ----------
    game_type:
        ``riichienv.GameType``。省略時は ``YON_TONPUSEN`` (4 人東風戦)。
    num_players:
        4 (v1 では 4 人麻雀のみ対応)。
    max_steps_per_game:
        無限 loop 防止のための step 上限。超えたら ``crash_context`` を残して
        episode を打ち切る。
    collect_samples:
        ``True`` (default) で ``DecisionSample`` を生成する。``False`` なら
        sample 生成をスキップし、metrics だけ集める軽量モード。
    deterministic_wall:
        ``True`` (default) で wall を Python 側で deterministic に生成し、
        ``env.reset(wall=...)`` に渡す。``False`` だと wall は engine 任せに
        なり、reproduction 強度が落ちる。詳細は note 参照。
    wall_seed_offset:
        wall 生成用 seed = ``run_episode(seed=...)`` + ``wall_seed_offset``。
        agent 側 seed と wall 側 seed を分離したい場合に使う (default 0)。

    Note
    ----
    PyPI ``riichienv 0.4.8`` の ``env.reset(wall=...)`` は **最初の round の
    壁牌のみ** を指定する。2 つ目以降の round の wall は engine 内部の
    randomness で引かれるため、``YON_TONPUSEN`` / ``YON_HANCHAN`` のような
    multi-round game では、wall 固定でも round 2 以降の actions/scores が
    run 間で divergence する。完全な hand-level 再現性が必要なときは
    ``game_type=YON_IKKYOKU`` (= single round) を使うか、engine 側に
    multi-round wall API が入るのを待つ必要がある。

    ``deterministic_wall=False`` を選んだ場合、wall 固定すら無くなるため、
    全 round で run 間 divergence が起こる前提となる (= reproduction は
    seed のみではなく ``wall_digest`` も使えなくなる)。
    """

    game_type: Any = _DEFAULT_GAME_TYPE
    num_players: int = 4
    max_steps_per_game: int = 4000
    collect_samples: bool = True
    deterministic_wall: bool = True
    wall_seed_offset: int = 0


@dataclass
class EpisodeResult:
    """1 game (= 1 episode) の結果。

    Attributes
    ----------
    seed:
        この game に使った seed (= ``run_episode`` の ``seed`` 引数)。
    episode_id:
        sample の ``episode_id`` と一致する文字列識別子。
    num_rounds:
        進んだ round 数 (kyoku 数)。最終 round は含む。
    num_steps:
        env.step 呼び出し回数。
    samples:
        生成された ``DecisionSample`` の flat list。``config.collect_samples=False``
        なら空 list。
    final_scores:
        ``num_players`` 個の int list。
    final_ranks:
        ``num_players`` 個の int list (1 が 1 位)。
    actor_types:
        ``player_id`` -> actor_type label の dict。metrics 集計用。
    round_summaries:
        各 round の {round_idx, is_draw, winner, deal_in, deltas, ...} の list。
    wall_seed:
        wall 生成に使った seed (= ``seed + config.wall_seed_offset``)。
        ``deterministic_wall=False`` のときは ``None``。
    wall_digest:
        生成した wall の SHA-256 hex (先頭 16 文字)。同 seed-同 wall の
        整合性チェック / experiment report 照合用。``deterministic_wall=False``
        のときは ``None``。wall list 自体は保存しない (hidden info 漏洩防止)。
    reset_seed:
        ``env.reset(seed=...)`` に渡した seed。``run_episode(seed=...)`` と
        同値だが、命名上の明示。``deterministic_wall=False`` でも非 ``None``。
    crash_context:
        crash した場合のみ非 None。``seed`` / ``wall_seed`` / ``wall_digest``
        / ``reset_seed`` / ``step_id`` / ``phase`` / ``error_type`` 等を含む。
    """

    seed: int
    episode_id: str
    num_rounds: int
    num_steps: int
    samples: list[DecisionSample]
    final_scores: list[int]
    final_ranks: list[int]
    actor_types: dict[int, str]
    round_summaries: list[dict[str, Any]] = field(default_factory=list)
    wall_seed: int | None = None
    wall_digest: str | None = None
    reset_seed: int | None = None
    crash_context: dict[str, Any] | None = None


class SelfPlayRunner:
    """4 席に agent を割り当てて 1 game を進める runner (v1)。

    Parameters
    ----------
    encoder:
        ``PublicObservationEncoder`` 相当の encoder。``encode_observation`` /
        ``discard_legal_mask`` / ``encode_candidates`` を持つ duck-typed
        object。
    config:
        ``SelfPlayConfig``。
    """

    def __init__(
        self,
        encoder: PublicObservationEncoder | None = None,
        config: SelfPlayConfig | None = None,
    ) -> None:
        self.config = config or SelfPlayConfig()
        if encoder is None:
            encoder = PublicObservationEncoder(num_players=self.config.num_players)
        self.encoder = encoder

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run_episode(
        self,
        seat_agents: SeatAgents,
        *,
        seed: int = 0,
        episode_id: str | None = None,
        rng: random.Random | None = None,
    ) -> EpisodeResult:
        """1 game を seed 付きで完走させ、``EpisodeResult`` を返す。

        crash 時は例外を呑み込み、``crash_context`` をセットして部分結果を返す
        (上位 ``run_games`` が複数 game を回せるようにするため)。

        ``config.deterministic_wall=True`` (default) のときは、
        ``wall_seed = seed + config.wall_seed_offset`` から決定論的に shuffle
        した 136 牌の wall を Python 側で生成し、``env.reset(wall=...)`` に
        渡す。これにより最初の round の壁牌が完全に再現可能になる
        (multi-round 含む engine 全体の strict determinism は engine 側制約。
        ``SelfPlayConfig`` の docstring 参照)。
        """
        episode_id = episode_id or f"ep_{int(seed)}"
        rng = rng or random.Random(seed)
        # wall 生成 (deterministic_wall=True のみ)
        wall_seed: int | None
        wall_digest: str | None
        wall: list[int] | None
        if self.config.deterministic_wall:
            wall_seed = int(seed) + int(self.config.wall_seed_offset)
            wall = _generate_wall(wall_seed)
            wall_digest = _wall_digest(wall)
        else:
            wall_seed = None
            wall_digest = None
            wall = None
        reset_seed = int(seed)
        env = riichienv.RiichiEnv(self.config.game_type)
        if wall is not None:
            env.reset(wall=list(wall), seed=reset_seed)
        else:
            env.reset(seed=reset_seed)
        # wall list は env に渡したら local 変数を解放する (sample / metadata
        # に載せないため。tile id を model-facing data に混ぜない方針に沿う)。
        wall = None
        tracker = RoundTracker(num_players=self.config.num_players)
        step_id_counter = 0
        env_step_count = 0
        crash_ctx: dict[str, Any] | None = None
        try:
            while not env.is_done:
                if env_step_count >= self.config.max_steps_per_game:
                    crash_ctx = {
                        "seed": int(seed),
                        "episode_id": episode_id,
                        "step_id": int(step_id_counter),
                        "env_step_count": int(env_step_count),
                        "reason": "max_steps_per_game_exceeded",
                        "phase": str(env.phase),
                        "wall_seed": wall_seed,
                        "wall_digest": wall_digest,
                        "reset_seed": reset_seed,
                    }
                    break
                if env.current_claims:
                    step_id_counter, env_step_count = self._handle_response_phase(
                        env=env,
                        seat_agents=seat_agents,
                        tracker=tracker,
                        episode_id=episode_id,
                        step_id_counter=step_id_counter,
                        env_step_count=env_step_count,
                        rng=rng,
                    )
                else:
                    step_id_counter, env_step_count = self._handle_act_phase(
                        env=env,
                        seat_agents=seat_agents,
                        tracker=tracker,
                        episode_id=episode_id,
                        step_id_counter=step_id_counter,
                        env_step_count=env_step_count,
                        rng=rng,
                    )
        except Exception as exc:  # noqa: BLE001
            crash_ctx = {
                "seed": int(seed),
                "episode_id": episode_id,
                "step_id": int(step_id_counter),
                "env_step_count": int(env_step_count),
                "reason": "exception",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "phase": str(getattr(env, "phase", "unknown")),
                "wall_seed": wall_seed,
                "wall_digest": wall_digest,
                "reset_seed": reset_seed,
            }

        # finalize: 最後の mjai_log walk (game の終了時 end_kyoku も拾う)
        tracker.consume_mjai_events(list(env.mjai_log))
        # game 終了時の terminated flag と all-round yaku/han/fu
        tracker.mark_terminated()
        # all-round yaku backfill: ``riichienv.MjaiReplay`` 経由で全 round の
        # winner WinResultContext を取り出し、mid-game も含めて winner sample
        # に yaku/han/fu を書き戻す。
        replay_records: tuple[Any, ...] = ()
        replay_failed = False
        if not crash_ctx:
            try:
                replay_records = collect_round_yaku_records(list(env.mjai_log))
            except Exception:  # noqa: BLE001
                # mjai_log の partial / Replay 構築失敗時は silent に fallback。
                replay_failed = True
                replay_records = ()
        if replay_records:
            tracker.finalize_with_round_yaku_records(replay_records)
        else:
            # replay 失敗 / records 空のときは旧 path (final round の
            # env.win_results) を fallback として使う。
            try:
                wr = dict(env.win_results) if not crash_ctx else {}
            except Exception:  # noqa: BLE001
                wr = {}
            if wr:
                tracker.finalize_game_with_win_results(wr)
        # replay_failed は明示利用しないが、診断時の breakpoint 用に保持する
        # (現状 EpisodeResult schema を増やさないため metadata には載せない)。
        del replay_failed
        # scores / ranks
        try:
            final_scores = [int(s) for s in env.scores()]
            final_ranks = [int(r) for r in env.ranks()]
        except Exception:  # noqa: BLE001
            final_scores = [0] * self.config.num_players
            final_ranks = [0] * self.config.num_players

        return EpisodeResult(
            seed=int(seed),
            episode_id=episode_id,
            num_rounds=int(tracker.round_idx),
            num_steps=int(env_step_count),
            samples=list(tracker.samples) if self.config.collect_samples else [],
            final_scores=final_scores,
            final_ranks=final_ranks,
            actor_types=dict(seat_agents.actor_types),
            round_summaries=list(tracker.round_summaries),
            wall_seed=wall_seed,
            wall_digest=wall_digest,
            reset_seed=reset_seed,
            crash_context=crash_ctx,
        )

    def run_games(
        self,
        seat_agents: SeatAgents,
        *,
        seeds: list[int] | None = None,
        base_seed: int = 0,
        num_games: int = 1,
    ) -> list[EpisodeResult]:
        """複数 game を回す convenience。

        ``seeds`` が指定されていればそれを使う。そうでなければ
        ``base_seed, base_seed+1, ..., base_seed+num_games-1`` を使う。
        """
        if seeds is None:
            seeds = [int(base_seed) + i for i in range(int(num_games))]
        out: list[EpisodeResult] = []
        for s in seeds:
            out.append(
                self.run_episode(
                    seat_agents, seed=int(s), episode_id=f"ep_{int(s)}"
                )
            )
        return out

    # ------------------------------------------------------------------
    # phase handlers
    # ------------------------------------------------------------------

    def _handle_act_phase(
        self,
        *,
        env: Any,
        seat_agents: SeatAgents,
        tracker: RoundTracker,
        episode_id: str,
        step_id_counter: int,
        env_step_count: int,
        rng: random.Random,
    ) -> tuple[int, int]:
        cp = int(env.current_player)
        obs = env.get_observation(cp)
        tracker.snapshot_observation(cp, obs)
        legal_raw = list(obs.legal_actions())
        legal_set = legal_actions_to_model_set(
            legal_raw,
            actor=cp,
            num_players=int(env.num_players),
            last_discarder=None,
            env_for_riichi=env,
        )
        # legal action 全くなしのケースは fail-fast (上位の try/except が拾う)
        if not legal_set.normal_discard and not legal_set.candidates:
            raise RuntimeError(
                f"act phase: no legal actions for player {cp} "
                f"(legal_raw_count={len(legal_raw)})"
            )

        agent = seat_agents.agents[cp]
        actor_type = seat_agents.actor_types[cp]
        decision = agent.select_action(legal_set, rng=rng, observation=obs)

        if self.config.collect_samples:
            sample = self._build_sample(
                episode_id=episode_id,
                round_idx=tracker.round_idx,
                step_id=step_id_counter,
                player_id=cp,
                actor_type=actor_type,
                obs=obs,
                legal_set=legal_set,
                decision_action=decision.action,
                rationale=decision.rationale,
                extras=dict(decision.extras),
            )
            tracker.append_pending(sample)
        step_id_counter += 1

        raw_seq = resolve(legal_set, decision.action)
        family = decision.action.family
        for raw_act in raw_seq:
            env.step({cp: raw_act})
            env_step_count += 1
            tracker.consume_mjai_events(list(env.mjai_log))
            if env.is_done:
                break
        # discard tracking
        if family in (ActionFamily.NORMAL_DISCARD, ActionFamily.RIICHI_DISCARD):
            tracker.note_discard_applied(cp)
        return step_id_counter, env_step_count

    def _handle_response_phase(
        self,
        *,
        env: Any,
        seat_agents: SeatAgents,
        tracker: RoundTracker,
        episode_id: str,
        step_id_counter: int,
        env_step_count: int,
        rng: random.Random,
    ) -> tuple[int, int]:
        last_discarder = tracker.last_discarder
        # response phase の last_discarder が None なら、env.current_player を
        # fallback として使う (initial 立直直後など、tracker が discard を見て
        # いない pathological ケース)。
        if last_discarder is None:
            last_discarder = int(env.current_player)

        # claim iteration order を player_id 昇順で安定化する。
        # PyPI riichienv の ``env.current_claims`` は HashMap で、Rust 側 hash
        # iteration order が run 間で揺れる挙動を確認済み。strict deterministic
        # に近づけるため loop 側で sorted iteration に統一する。
        claimants = sorted(int(p) for p in env.current_claims.keys())
        actions_to_apply: dict[int, Any] = {}
        applied_families: dict[int, ActionFamily] = {}
        for cp in claimants:
            cp = int(cp)
            obs = env.get_observation(cp)
            tracker.snapshot_observation(cp, obs)
            legal_raw = list(obs.legal_actions())
            legal_set = legal_actions_to_model_set(
                legal_raw,
                actor=cp,
                num_players=int(env.num_players),
                last_discarder=last_discarder,
                env_for_riichi=None,
            )
            if not legal_set.normal_discard and not legal_set.candidates:
                continue  # 何もできない claimant は skip (env.step 側で OK)
            agent = seat_agents.agents[cp]
            actor_type = seat_agents.actor_types[cp]
            decision = agent.select_action(legal_set, rng=rng, observation=obs)
            if self.config.collect_samples:
                sample = self._build_sample(
                    episode_id=episode_id,
                    round_idx=tracker.round_idx,
                    step_id=step_id_counter,
                    player_id=cp,
                    actor_type=actor_type,
                    obs=obs,
                    legal_set=legal_set,
                    decision_action=decision.action,
                    rationale=decision.rationale,
                    extras=dict(decision.extras),
                )
                tracker.append_pending(sample)
            step_id_counter += 1
            raw_seq = resolve(legal_set, decision.action)
            if len(raw_seq) != 1:
                raise RuntimeError(
                    f"response phase action must be single-step, got "
                    f"{len(raw_seq)} for player {cp} family "
                    f"{decision.action.family.value}"
                )
            actions_to_apply[cp] = raw_seq[0]
            applied_families[cp] = decision.action.family

        if actions_to_apply:
            env.step(actions_to_apply)
            env_step_count += 1
            tracker.consume_mjai_events(list(env.mjai_log))
            # 副露 (Chi/Pon/Daiminkan) が成立すると discard chain が切れる。
            # 副露成立 player は env.current_player に変わるはずなので、それを
            # 検出する代わりに、副露 family が含まれていたら conservative に
            # last_discarder をクリアする。
            for fam in applied_families.values():
                if fam in (
                    ActionFamily.CHI,
                    ActionFamily.PON,
                    ActionFamily.DAIMINKAN,
                ):
                    tracker.clear_last_discarder()
                    break
        else:
            # 全 claimant が legal action 無しの病的 case。env を進めるため
            # 空 dict を渡す試行を 1 回だけ行う (riichienv が許せば前進する)。
            try:
                env.step({})
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "response phase: no claimant produced an action"
                ) from exc
            env_step_count += 1
            tracker.consume_mjai_events(list(env.mjai_log))
        return step_id_counter, env_step_count

    # ------------------------------------------------------------------
    # sample construction
    # ------------------------------------------------------------------

    def _build_sample(
        self,
        *,
        episode_id: str,
        round_idx: int,
        step_id: int,
        player_id: int,
        actor_type: str,
        obs: Any,
        legal_set: LegalActionSet,
        decision_action: ModelAction,
        rationale: str,
        extras: dict[str, Any] | None = None,
    ) -> DecisionSample:
        # ``ModelPolicyAgent`` 等が ``extras["observation_feat"]`` に encoder
        # 出力済み feature (= numpy float32, shape=(observation_dim,)) を
        # 載せている場合、再 encode をスキップする。shape mismatch は
        # fail-fast (= 差し替え可能な silent corruption を防ぐ)。
        observation_feat = self._observation_feat_from_extras_or_encode(
            obs, extras
        )
        discard_mask = self.encoder.discard_legal_mask(legal_set)
        cand_feat = self.encoder.encode_candidates(legal_set)
        family = decision_action.family
        if family == ActionFamily.NORMAL_DISCARD:
            sel_tt = int(decision_action.tile_type)
            sel_idx = -1
        else:
            sel_tt = -1
            sel_idx = _find_candidate_index(legal_set, decision_action)
        metadata: dict[str, Any] = {
            "rationale": str(rationale),
        }
        # AgentDecision.extras から teacher / on-policy 情報を取り出す
        extras = extras or {}
        teacher_best_mask = extras.get("teacher_best_mask")
        teacher_tt = int(
            extras.get("teacher_discard_tile_type", -1)
        )
        teacher_ci = int(extras.get("teacher_candidate_index", -1))
        teacher_available = bool(
            (teacher_best_mask is not None and teacher_best_mask.any())
            or teacher_tt >= 0
            or teacher_ci >= 0
        )
        old_log_prob = float(extras.get("log_prob", 0.0))
        value = float(extras.get("value", 0.0))
        # 数値化されない補助 metric は metadata に追記する
        for k in ("teacher_shanten", "teacher_ukeire", "call_score"):
            if k in extras:
                metadata[k] = float(extras[k])
        # PPO eligibility 用フラグ: deterministic shortcut (TSUMO/RON/KYUSHU
        # 等) はここで True が入る。``compute_returns_and_advantages`` は
        # この flag を見て eligible=False に倒す。
        if bool(extras.get("ppo_exclude", False)):
            metadata["ppo_exclude"] = True
        # post-riichi discard 検出: NORMAL_DISCARD かつ player が既に riichi
        # 宣言済みのとき、env 側で行動が強制 (tsumogiri only) されており、
        # 学習対象として除外したい場合がある。trainer 側 opt-in flag で
        # 除外できるよう metadata に sticky flag を入れる (実 sample が
        # post-riichi かどうかは観測時の obs.riichi_declared から決まる)。
        if family == ActionFamily.NORMAL_DISCARD:
            riichi_declared = getattr(obs, "riichi_declared", None)
            if (
                riichi_declared is not None
                and 0 <= int(player_id) < len(riichi_declared)
                and bool(riichi_declared[int(player_id)])
            ):
                metadata["is_post_riichi_discard"] = True
        return make_initial_sample(
            episode_id=episode_id,
            round_idx=round_idx,
            step_id=step_id,
            player_id=player_id,
            decision_family=family.value,
            actor_type=actor_type,
            observation_feat=observation_feat,
            discard_mask=discard_mask,
            candidate_features=cand_feat,
            selected_discard_tile_type=sel_tt,
            selected_candidate_index=sel_idx,
            metadata=metadata,
            teacher_discard_tile_type=teacher_tt,
            teacher_candidate_index=teacher_ci,
            teacher_best_mask=teacher_best_mask,
            teacher_available=teacher_available,
            old_log_prob=old_log_prob,
            value=value,
        )

    def _observation_feat_from_extras_or_encode(
        self,
        obs: Any,
        extras: dict[str, Any] | None,
    ) -> np.ndarray:
        """``extras["observation_feat"]`` があれば再利用、無ければ encode する。

        validation
        ----------
        - shape は ``(self.encoder.metadata().observation_dim,)`` でないと
          ``ValueError`` で fail-fast。
        - dtype は ``np.float32`` に正規化する (= 別 dtype が来ても accept する
          がコピー 1 回挟まる)。
        - 任意の追加 hidden info が混入しないように、numpy array 以外は
          encoder fallback に倒す (= dict / object / None など扱わない)。
        """
        if extras is not None:
            cached = extras.get("observation_feat")
            if cached is not None:
                if not isinstance(cached, np.ndarray):
                    raise TypeError(
                        f"extras['observation_feat'] must be np.ndarray, "
                        f"got {type(cached).__name__}"
                    )
                expected_dim = int(self.encoder.metadata().observation_dim)
                feat = np.asarray(cached, dtype=np.float32).reshape(-1)
                if feat.shape != (expected_dim,):
                    raise ValueError(
                        f"extras['observation_feat'] shape "
                        f"{cached.shape if hasattr(cached, 'shape') else 'unknown'} "
                        f"mismatches encoder.observation_dim={expected_dim}"
                    )
                return feat
        return self.encoder.encode_observation(obs)


def _find_candidate_index(
    legal_set: LegalActionSet, chosen: ModelAction
) -> int:
    for i, c in enumerate(legal_set.candidates):
        if c.key == chosen.key:
            return i
    raise KeyError(
        f"selected candidate {chosen.key!r} not found in legal_set.candidates"
    )


# ---------------------------------------------------------------------------
# Mapping convenience
# ---------------------------------------------------------------------------


def make_seat_agents_from_mapping(
    mapping: Mapping[int, tuple[Any, str]],
) -> SeatAgents:
    """``{player_id: (agent, actor_type)}`` から ``SeatAgents`` を作る短縮表記。

    ``SeatAgents.from_pairs`` の re-export。runner.py 単体での使い勝手のため。
    """
    return SeatAgents.from_pairs(dict(mapping))


__all__ = [
    "SelfPlayConfig",
    "EpisodeResult",
    "SelfPlayRunner",
    "make_seat_agents_from_mapping",
]
