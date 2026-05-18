"""Thin smoke adapter around ``riichienv.RiichiEnv``.

このモジュールは Stage03 の最初の adapter として、PyPI 版 ``riichienv`` の
最小 API を確認できる薄い wrapper を提供する。本格的な action abstraction /
encoder / model 入力整形は後続レイヤで行う。

公開する情報は public-only。具体的には:

- ``get_observation(player_id)`` が返す ``riichienv.Observation`` は、その
  player から見える情報のみを保持する (自分の hand / 公開副露 / 河 / 場況)。
- adapter からは ``RiichiEnv.hands`` (全 player の hand) や ``RiichiEnv.wall``
  (山) などの hidden な engine 内部 state を露出しない。
"""
from __future__ import annotations

from typing import Any

import riichienv

# 既定 game type: 4 人東風戦 (smoke / unit test での待ち時間を抑えるため)。
# 本番学習向け game type は後続 issue で config から渡す想定。
_DEFAULT_GAME_TYPE = riichienv.GameType.YON_TONPUSEN


class RiichiEnvAdapter:
    """``riichienv.RiichiEnv`` の最小 wrapper。

    Stage03 smoke 用。後続 issue で action abstraction / encoder / agent loop
    などの上位レイヤを別途実装する。ここでは public observation と最低限の
    progression API だけを公開する。

    Parameters
    ----------
    game_type:
        ``riichienv.GameType`` enum。省略時は ``YON_TONPUSEN`` を使う。
    """

    def __init__(self, game_type: Any | None = None) -> None:
        if game_type is None:
            game_type = _DEFAULT_GAME_TYPE
        self._game_type = game_type
        self._env = riichienv.RiichiEnv(game_type)

    # ------------------------------------------------------------------
    # progression
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """新規 game を開始する。

        Parameters
        ----------
        seed:
            ``riichienv.RiichiEnv.reset`` に渡す seed。省略時は engine 既定。
        """
        if seed is None:
            self._env.reset()
        else:
            self._env.reset(seed=int(seed))

    def step(self, actions: dict[int, Any]) -> dict[int, Any]:
        """``{player_id: riichienv.Action, ...}`` の形で 1 step 進める。

        ``riichienv.RiichiEnv.step`` をそのまま呼ぶ。返り値は engine が返す
        ``{player_id: Observation}`` 形式の dict。
        """
        if not isinstance(actions, dict):
            raise TypeError(
                f"actions must be a dict[int, Action], got {type(actions).__name__}")
        return self._env.step(actions)

    # ------------------------------------------------------------------
    # state inspection (public-only)
    # ------------------------------------------------------------------

    @property
    def phase(self) -> Any:
        """``Phase.WaitAct`` (自家行動待ち) または ``Phase.WaitResponse`` (鳴き応答待ち)。"""
        return self._env.phase

    @property
    def current_player(self) -> int:
        """``Phase.WaitAct`` のときの行動 player_id。"""
        return int(self._env.current_player)

    @property
    def current_claims(self) -> dict[int, Any]:
        """``Phase.WaitResponse`` のときに応答が必要な player ごとの claim dict。

        空 dict のときは応答待ちではない。
        """
        return dict(self._env.current_claims)

    @property
    def is_done(self) -> bool:
        """エピソード (game 全体) の終了 flag。"""
        return bool(self._env.is_done)

    def get_observation(self, player_id: int) -> Any:
        """指定 player の public observation を返す。

        返される ``riichienv.Observation`` は、その player から見える情報のみを
        含む (自分の hand / 公開副露 / 河 / 場況など)。他家手牌や山などの
        hidden state は含まれない。
        """
        return self._env.get_observation(int(player_id))

    def legal_actions(self, player_id: int) -> list[Any]:
        """指定 player の合法 action list を返す。"""
        obs = self.get_observation(player_id)
        return list(obs.legal_actions())

    def current_decision_players(self) -> list[int]:
        """現在 decision を求めている player の list を返す。

        - ``WaitAct``: ``[current_player]``
        - ``WaitResponse``: claims の key list
        """
        if self._env.current_claims:
            return [int(p) for p in self._env.current_claims.keys()]
        return [self.current_player]

    # ------------------------------------------------------------------
    # score / round info
    # ------------------------------------------------------------------

    def scores(self) -> list[int]:
        """4 人 (3 人麻雀のときは 3 人) の現在 score list。"""
        return [int(s) for s in self._env.scores()]

    def ranks(self) -> list[int]:
        """4 人 (3 人麻雀のときは 3 人) の現在順位 list (1 が 1 位)。"""
        return [int(r) for r in self._env.ranks()]

    @property
    def score_deltas(self) -> list[int]:
        """直近 step の score delta。"""
        return [int(d) for d in self._env.score_deltas]

    @property
    def kyoku_idx(self) -> int:
        """game 内 round index (0 始まり)。"""
        return int(self._env.kyoku_idx)

    @property
    def round_info(self) -> dict[str, int]:
        """場風 / 親 / 本場 / 供託 / 局 index を一括で返す。"""
        return {
            "kyoku_idx": int(self._env.kyoku_idx),
            "oya": int(self._env.oya),
            "round_wind": int(self._env.round_wind),
            "honba": int(self._env.honba),
            "riichi_sticks": int(self._env.riichi_sticks),
        }

    @property
    def num_players(self) -> int:
        return int(self._env.num_players)

    @property
    def win_results(self) -> dict[int, Any]:
        """直近 round の和了結果 dict。"""
        return dict(self._env.win_results)


__all__ = ["RiichiEnvAdapter"]
