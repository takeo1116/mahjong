"""Per-round state tracker for self-play loop.

主な責務:

- `last_discarder` の追跡 (response phase の ``last_discarder`` 解決に使う)。
- 1 round 内で生成された ``DecisionSample`` の pending buffer 管理。
- ``env.mjai_log`` を walk して round-end event (``hora`` / ``ryukyoku`` /
  ``end_kyoku`` / ``start_kyoku``) を検出し、round 終了時に pending samples
  へ score_delta / reward / terminal_class / yaku_target を backfill する。
- 各 player の最新 ``riichienv.Observation`` を keep (winner の menzen 判定 /
  流局時 tenpai 判定に使う)。

設計メモ:

- PyPI 版 ``riichienv`` は round 境界を ``step()`` 内で auto-advance してしまう
  ため、``env.win_results`` / ``env.score_deltas`` を mid-game に直接読めない。
  そこで ``env.mjai_log`` を walk して round-end events を取得し、そこから
  per-player score delta を取り出す方針を採る。
- ``hora`` event は actor / target / tsumo / deltas を持つが、yaku / han / fu は
  含まれない。通常は game 完走後に ``MjaiReplay`` で全 round の yaku/han/fu を
  再計算し、``finalize_with_round_yaku_records()`` で winner samples に backfill
  する。``finalize_game_with_win_results()`` は replay 失敗時の fallback として残す。
- 流局の途中流局 (kyushu_kyuhai 等) は v1 では全 player を
  ``OTHER_NON_DEALIN`` に倒す。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mahjong_agent.data.types import SCHEMA_VERSION, DecisionSample
from mahjong_agent.evaluation.replay_yaku import RoundYakuRecord
from mahjong_agent.targets.terminal import (
    RoundOutcome,
    terminal_class_index,
)
from mahjong_agent.targets.yaku import NUM_YAKU, extract_yaku_target

# round-level reward scale: ``score_delta * 0.0001``。Stage02 で raw point
# scale は学習を歪めることが確認されており、初期方針として 1e-4 を採用する。
_ROUND_REWARD_SCALE: float = 1e-4


@dataclass
class _RoundEndInfo:
    """1 round の終局 facts (mjai_log から抽出)。"""

    deltas: list[int]
    winner: int | None = None  # hora actor (= 和了 player)
    deal_in: int | None = None  # ron の場合の放銃 player (tsumo では None)
    is_tsumo: bool = False
    is_draw: bool = False
    ryukyoku_reason: str | None = None
    # 流局 tenpai の player set (mjai_log の ryukyoku.tenpais がある場合)。
    # 取れない実装では空 set のままにしておく (後段で is_tenpai_at_draw を
    # 各 player の cached observation から拾う方針)。
    tenpai_pids: set[int] = field(default_factory=set)


@dataclass
class RoundTracker:
    """1 episode 内の round-by-round state を管理する mutable tracker。

    Parameters
    ----------
    num_players:
        4 (4 人麻雀) を想定。3 人麻雀は v1 では未対応。

    Notes
    -----
    `RoundTracker` instance は 1 episode に対して 1 つ作る。round が
    進むごとに内部 state を更新し、最終的に ``samples`` から flat な
    ``DecisionSample`` list を取り出す。
    """

    num_players: int = 4
    round_idx: int = 0
    last_log_idx: int = 0
    last_discarder: int | None = None
    pending_samples: list[DecisionSample] = field(default_factory=list)
    samples: list[DecisionSample] = field(default_factory=list)
    # per-player last observation snapshot (winner menzen / draw tenpai 判定用)
    last_open_meld: dict[int, bool] = field(default_factory=dict)
    last_tenpai: dict[int, bool] = field(default_factory=dict)
    # per-player の最新 sample (game 終了時 terminated flag 立て用)
    last_sample_index_per_player: dict[int, int] = field(default_factory=dict)
    # round-level fact summary (metrics 用)
    round_summaries: list[dict[str, Any]] = field(default_factory=list)
    # pending round end event (まだ end_kyoku を見ていない hora/ryukyoku)
    _pending_round_end: _RoundEndInfo | None = None

    # ------------------------------------------------------------------
    # observation snapshot (encoder 入力前に呼ぶ)
    # ------------------------------------------------------------------

    def snapshot_observation(self, player_id: int, obs: Any) -> None:
        """player の最新 observation から menzen / tenpai を覚える。

        hidden info には触れず、public な ``obs.melds[player_id]`` と
        ``obs.is_tenpai`` のみを読む。
        """
        pid = int(player_id)
        # menzen 判定: 自分の melds に open meld があるか
        own_melds = obs.melds[pid] if obs.melds else []
        has_open = False
        for meld in own_melds:
            if _meld_is_open(meld):
                has_open = True
                break
        self.last_open_meld[pid] = has_open
        self.last_tenpai[pid] = bool(obs.is_tenpai)

    # ------------------------------------------------------------------
    # sample buffering
    # ------------------------------------------------------------------

    def append_pending(self, sample: DecisionSample) -> None:
        """1 つの sample を pending buffer に追加する。"""
        self.pending_samples.append(sample)

    # ------------------------------------------------------------------
    # discard tracking
    # ------------------------------------------------------------------

    def note_discard_applied(self, player_id: int) -> None:
        """``NORMAL_DISCARD`` / ``RIICHI_DISCARD`` を env に適用した直後に呼ぶ。

        次に発生する response phase で ``last_discarder=player_id`` として
        使われる。
        """
        self.last_discarder = int(player_id)

    def clear_last_discarder(self) -> None:
        """副露 (Chi/Pon/Daiminkan) が成立すると discard chain が切れる。"""
        self.last_discarder = None

    # ------------------------------------------------------------------
    # mjai_log walking
    # ------------------------------------------------------------------

    def consume_mjai_events(self, mjai_log: list) -> None:
        """``env.mjai_log`` を ``last_log_idx`` から末尾まで walk する。

        新規 ``hora`` / ``ryukyoku`` を見つけたら ``_pending_round_end`` に
        積み、``end_kyoku`` が出てきた時点で round backfill を実行する。
        次の ``start_kyoku`` で round_idx をインクリメントし、buffer を
        新規 round 用に reset する。
        """
        log_len = len(mjai_log)
        while self.last_log_idx < log_len:
            ev = mjai_log[self.last_log_idx]
            self.last_log_idx += 1
            if not isinstance(ev, dict):
                continue
            t = ev.get("type", "")
            if t == "hora":
                self._record_hora_event(ev)
            elif t == "ryukyoku":
                self._record_ryukyoku_event(ev)
            elif t == "end_kyoku":
                self._on_end_kyoku()
            elif t == "start_kyoku":
                # 新しい round の開始: last_discarder / observation snapshot を reset。
                # round_idx 自体は end_kyoku 側で既にインクリメント済み (= 最初の
                # start_kyoku は round_idx=0 のまま開始する)。
                self.last_discarder = None
                self.last_open_meld.clear()
                self.last_tenpai.clear()

    def _record_hora_event(self, ev: dict) -> None:
        actor = int(ev.get("actor", -1))
        target = int(ev.get("target", actor))
        is_tsumo = bool(ev.get("tsumo", False))
        deltas = [int(x) for x in ev.get("deltas", [0] * self.num_players)]
        if self._pending_round_end is None:
            self._pending_round_end = _RoundEndInfo(deltas=list(deltas))
        else:
            # ダブロン等で複数 hora event がある場合、deltas を加算する。
            info = self._pending_round_end
            n = max(len(info.deltas), len(deltas))
            merged = [0] * n
            for i in range(n):
                merged[i] = (
                    (info.deltas[i] if i < len(info.deltas) else 0)
                    + (deltas[i] if i < len(deltas) else 0)
                )
            info.deltas = merged
        info = self._pending_round_end
        # winner / deal_in は最後の event を採用する conservative v1。
        # ダブロンで複数 winner / 複数 deal-in が居る場合、terminal_class は
        # 最後の event 視点に倒れる (= 他の winner は OTHER_NON_DEALIN に倒す)。
        info.winner = actor
        info.deal_in = None if is_tsumo else (target if target != actor else None)
        info.is_tsumo = is_tsumo
        info.is_draw = False

    def _record_ryukyoku_event(self, ev: dict) -> None:
        deltas = [int(x) for x in ev.get("deltas", [0] * self.num_players)]
        reason = ev.get("reason", None)
        tenpais = ev.get("tenpais", None)
        info = _RoundEndInfo(deltas=deltas, is_draw=True, ryukyoku_reason=reason)
        if isinstance(tenpais, (list, tuple)):
            info.tenpai_pids = {
                pid for pid, flag in enumerate(tenpais) if bool(flag)
            }
        if self._pending_round_end is None:
            self._pending_round_end = info
        else:
            # hora が先に来て同 step で ryukyoku が来るパターンは想定外。
            # 一応 conservative に ryukyoku を優先する。
            self._pending_round_end = info

    def _on_end_kyoku(self) -> None:
        info = self._pending_round_end
        # round summary を記録 (info が無くても empty round_summary を残す)
        summary = {
            "round_idx": self.round_idx,
            "is_draw": bool(info.is_draw) if info else False,
            "ryukyoku_reason": info.ryukyoku_reason if info else None,
            "winner": info.winner if info else None,
            "deal_in": info.deal_in if info else None,
            "is_tsumo": bool(info.is_tsumo) if info else False,
            "deltas": list(info.deltas) if info else [0] * self.num_players,
        }
        self.round_summaries.append(summary)
        # backfill pending samples
        if info is not None:
            self._backfill_round_end(info)
        else:
            # 終局情報が取れなかった場合 (途中流局など) は conservative に
            # deltas=0 で flush だけする。
            self._backfill_round_end(
                _RoundEndInfo(deltas=[0] * self.num_players, is_draw=True)
            )
        # buffer flush
        self._flush_pending_samples()
        self._pending_round_end = None
        self.round_idx += 1
        # next round 用に snapshot を reset (start_kyoku で player の手牌が
        # 配り直されるので、古い is_tenpai / open_meld を引きずらない)。
        self.last_open_meld.clear()
        self.last_tenpai.clear()
        self.last_discarder = None

    # ------------------------------------------------------------------
    # backfill / flush
    # ------------------------------------------------------------------

    def _backfill_round_end(self, info: _RoundEndInfo) -> None:
        """pending samples (= 今 round 内の全 sample) に round-level fact を書き込む。

        - score_delta: ``info.deltas[player_id]``
        - reward: ``score_delta * 1e-4``
        - terminal_class: per-player ``RoundOutcome`` から導出
        - round_over: 該当 player の **最後** の sample のみ True
        - yaku_target / yaku_loss_mask: v1 では mid-game hora では空 (final
          round で ``finalize_game_with_win_results`` が別途上書きする)

        Note: ``pending_samples`` は in-place 置き換え (``DecisionSample`` は
        mutable dataclass)。
        """
        # 各 player の最終 sample index を pending 内で記録
        last_sample_idx_per_pid: dict[int, int] = {}
        for i, s in enumerate(self.pending_samples):
            last_sample_idx_per_pid[int(s.player_id)] = i
        for i, s in enumerate(self.pending_samples):
            pid = int(s.player_id)
            delta = int(info.deltas[pid]) if pid < len(info.deltas) else 0
            s.score_delta = delta
            s.reward = float(delta) * _ROUND_REWARD_SCALE
            s.round_over = i == last_sample_idx_per_pid.get(pid, -1)
            s.terminal_class = self._compute_terminal_class(pid, info)
            # mid-game では yaku target は空 (final round で上書き)
            s.yaku_target = np.zeros(NUM_YAKU, dtype=np.float32)
            s.yaku_loss_mask = 0.0
            s.han = -1
            s.fu = -1

    def _compute_terminal_class(
        self, player_id: int, info: _RoundEndInfo
    ) -> int:
        """1 player 視点で terminal class index を返す (0..4)。"""
        is_winner = info.winner is not None and info.winner == player_id
        won_menzen = False
        if is_winner:
            won_menzen = not self.last_open_meld.get(player_id, False)
        is_deal_in = (
            info.deal_in is not None and info.deal_in == player_id
        )
        is_tenpai_at_draw = False
        if info.is_draw:
            if info.tenpai_pids:
                is_tenpai_at_draw = player_id in info.tenpai_pids
            else:
                # mjai_log に tenpais が出ない場合は cached observation の
                # is_tenpai を使う。
                is_tenpai_at_draw = bool(self.last_tenpai.get(player_id, False))
        outcome = RoundOutcome(
            is_winner=is_winner,
            won_menzen=won_menzen,
            is_deal_in_payer=is_deal_in,
            is_tenpai_at_draw=is_tenpai_at_draw,
            is_draw=bool(info.is_draw),
        )
        return terminal_class_index(outcome)

    def _flush_pending_samples(self) -> None:
        """pending を flat samples list に移し、per-player 最新 index を更新する。"""
        for s in self.pending_samples:
            self.samples.append(s)
            self.last_sample_index_per_player[int(s.player_id)] = (
                len(self.samples) - 1
            )
        self.pending_samples.clear()

    # ------------------------------------------------------------------
    # finalize / game-end backfill
    # ------------------------------------------------------------------

    def finalize_game_with_win_results(self, win_results: dict[int, Any]) -> None:
        """game 終了時 (env.is_done) に env.win_results からの yaku/han/fu を
        最後の winner 用 samples に書き込む。

        Notes
        -----
        - PyPI 版 ``riichienv`` は mid-game の round 終了で ``env.win_results``
          を clear するが、最終 round の終了直後 (= env.is_done になった瞬間)
          には ``env.win_results`` が残るため、final round の winner 用に
          yaku / han / fu / yaku_loss_mask を遅延 backfill できる。
        - 最終 round 以外の winner samples では mid-game の制約により
          yaku target は空 (loss mask 0)。完了報告に明記。
        """
        for pid, wr in win_results.items():
            pid = int(pid)
            yaku_ids = getattr(wr, "yaku", None) or []
            han = int(getattr(wr, "han", -1))
            fu = int(getattr(wr, "fu", -1))
            target, mask = extract_yaku_target(
                list(yaku_ids), is_winner=True, allow_unknown=True
            )
            # samples を後ろから走査し、最終 round (= 同 pid の round_over=True
            # を見た直後の round) の samples にだけ書き込む。
            seen_round_over = False
            for s in reversed(self.samples):
                if int(s.player_id) != pid:
                    continue
                if seen_round_over and s.round_over:
                    # さらに前の round に入ったので終了。
                    break
                s.yaku_target = target.copy()
                s.yaku_loss_mask = float(mask)
                s.han = han
                s.fu = fu
                if s.round_over:
                    seen_round_over = True

    def finalize_with_round_yaku_records(
        self, records: Sequence[RoundYakuRecord]
    ) -> int:
        """post-game に再計算した ``RoundYakuRecord`` を winner samples に backfill する。

        Parameters
        ----------
        records:
            ``mahjong_agent.evaluation.replay_yaku.collect_round_yaku_records``
            の戻り値。``round_index`` は ``MjaiReplay.take_kyokus()`` の順、
            ``winner_seat`` は和了した player の seat id。

        Returns
        -------
        int:
            backfill が反映された **distinct な (round_idx, player_id) ペア** の数。
            diagnostics / regression 用。

        Notes
        -----
        - mid-game / final round の区別なく、全 hora round の winner sample
          に ``yaku_target`` / ``yaku_loss_mask=1.0`` / ``han`` / ``fu`` を
          書き込む。``finalize_game_with_win_results`` は fallback として残す。
        - 同 round 複数 winner (double-ron 等) は record が複数返るため、
          各 winner ごとに backfill する。
        - 該当 winner sample が見つからない (= その round / seat には sample が
          存在しない config の場合など) ときは crash せずスキップする。
        - score_delta / reward / terminal_class / round_over は本関数では
          変更しない (round-end backfill 経路で確定済み)。
        """
        if not records:
            return 0
        # (round_idx -> player_id -> sample indices) の index を 1 回作る。
        per_round_player_indices: dict[int, dict[int, list[int]]] = {}
        for i, s in enumerate(self.samples):
            per_round_player_indices.setdefault(
                int(getattr(s, "round_id", 0)), {}
            ).setdefault(int(s.player_id), []).append(i)

        applied_pairs: set[tuple[int, int]] = set()
        for rec in records:
            pid = int(rec.winner_seat)
            ridx = int(rec.round_index)
            round_map = per_round_player_indices.get(ridx)
            if not round_map:
                continue
            sample_indices = round_map.get(pid)
            if not sample_indices:
                continue
            target, mask = extract_yaku_target(
                list(rec.yaku_ids), is_winner=True, allow_unknown=True
            )
            han = int(rec.han)
            fu = int(rec.fu)
            for idx in sample_indices:
                s = self.samples[idx]
                s.yaku_target = target.copy()
                s.yaku_loss_mask = float(mask)
                s.han = han
                s.fu = fu
            applied_pairs.add((ridx, pid))
        return len(applied_pairs)

    def mark_terminated(self) -> None:
        """game 終了時に、各 player の最後の sample に ``terminated=True`` を立てる。"""
        for idx in self.last_sample_index_per_player.values():
            if 0 <= idx < len(self.samples):
                self.samples[idx].terminated = True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _meld_is_open(meld: Any) -> bool:
    """Ankan 以外を「公開副露」と判定する。

    ``riichienv.Meld.opened`` 属性を優先、それが無ければ ``meld_type`` の
    最後の segment (``"Ankan"``) で判定する。これは encoder と同じロジック
    だが、private helper の再利用にこだわらず重複させる方針 (encoder は
    feature encode 用、tracker は terminal class 用と責務が異なる)。
    """
    opened = getattr(meld, "opened", None)
    if opened is not None:
        return bool(opened)
    mtype = getattr(meld, "meld_type", None)
    if mtype is None:
        return False
    return str(mtype).rsplit(".", 1)[-1] != "Ankan"


_DISCARD_MASK_DIM_T: int = 34


def make_initial_sample(
    *,
    episode_id: str,
    round_idx: int,
    step_id: int,
    player_id: int,
    decision_family: str,
    actor_type: str,
    observation_feat: np.ndarray,
    discard_mask: np.ndarray,
    candidate_features: np.ndarray,
    selected_discard_tile_type: int,
    selected_candidate_index: int,
    metadata: dict[str, Any] | None = None,
    teacher_discard_tile_type: int = -1,
    teacher_candidate_index: int = -1,
    teacher_best_mask: np.ndarray | None = None,
    teacher_available: bool = False,
    old_log_prob: float = 0.0,
    value: float = 0.0,
) -> DecisionSample:
    """まだ round/game 終了が分からない段階の DecisionSample を組み立てる。

    score_delta / reward / round_over / terminated / terminal_class /
    yaku_target / yaku_loss_mask / han / fu は ``RoundTracker`` が後段で
    backfill する。

    ``teacher_*`` / ``old_log_prob`` / ``value`` は agent が出力した
    ``AgentDecision.extras`` から caller が転記する想定。未提供時は
    default (``-1`` / 0.0 / False / 全 0 mask)。
    """
    if teacher_best_mask is None:
        tbm = np.zeros(_DISCARD_MASK_DIM_T, dtype=np.float32)
    else:
        tbm = np.asarray(teacher_best_mask, dtype=np.float32).reshape(-1)
        if tbm.size != _DISCARD_MASK_DIM_T:
            raise ValueError(
                f"teacher_best_mask must be length {_DISCARD_MASK_DIM_T}, "
                f"got {tbm.size}"
            )
    return DecisionSample(
        schema_version=SCHEMA_VERSION,
        episode_id=str(episode_id),
        round_id=int(round_idx),
        step_id=int(step_id),
        player_id=int(player_id),
        decision_family=str(decision_family),
        actor_type=str(actor_type),
        observation=np.asarray(observation_feat, dtype=np.float32),
        discard_mask=np.asarray(discard_mask, dtype=np.float32),
        candidate_features=np.asarray(candidate_features, dtype=np.float32),
        selected_discard_tile_type=int(selected_discard_tile_type),
        selected_candidate_index=int(selected_candidate_index),
        old_log_prob=float(old_log_prob),
        value=float(value),
        reward=0.0,
        terminated=False,
        round_over=False,
        terminal_class=-1,
        yaku_target=np.zeros(NUM_YAKU, dtype=np.float32),
        yaku_loss_mask=0.0,
        han=-1,
        fu=-1,
        score_delta=0,
        teacher_discard_tile_type=int(teacher_discard_tile_type),
        teacher_candidate_index=int(teacher_candidate_index),
        teacher_best_mask=tbm,
        teacher_available=bool(teacher_available),
        metadata=dict(metadata or {}),
    )


__all__ = [
    "RoundTracker",
    "make_initial_sample",
]
