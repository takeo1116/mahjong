"""Post-game yaku/han/fu recovery via ``riichienv.MjaiReplay``.

PyPI ``riichienv 0.4.8`` の ``env.win_results`` は round 境界で engine 側が
``_initialize_next_round`` を即時実行して clear するため、mid-game の round で
得点情報を Python 側から直接 polling することはできない (game 終了時の
final round のみ残る)。本モジュールは game 完走後に ``env.mjai_log`` を
``riichienv.MjaiReplay`` に流し直し、``WinResultContext`` から
``(seat, yaku ids, han, fu, yakuman)`` を per-round に取り出す。

Hidden info 境界:

- ``mjai_log`` には engine 出力としての ``start_kyoku.tehais`` (全員の配牌) が
  含まれるが、本モジュールは復元した **公開情報** (= 和了時点で卓上に開示される
  情報) のみを ``RoundYakuRecord`` として返す。生 mjai_log を sample
  / encoder input に流す経路は提供しない。
- 呼び出し元は ``mjai_log`` 自体を ``DecisionSample.metadata`` 等に保存しない
  運用を継続する責務がある (本モジュールは label extraction 専用)。
"""
from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import riichienv


@dataclass(frozen=True)
class RoundYakuRecord:
    """1 winner 分の post-game yaku label。

    ``round_index`` は ``mjai_log`` 中で何番目の kyoku か (= 0 始まり、
    ryukyoku round もカウントに含めて並び順は ``MjaiReplay.take_kyokus()``
    と同じ)。同一 round で多人和了がある場合は同 ``round_index`` の record
    が複数返る。
    """

    round_index: int
    winner_seat: int
    yaku_ids: tuple[int, ...]
    han: int
    fu: int
    yakuman: bool
    agari_tile: int | None = None
    round_wind: int | None = None
    oya: int | None = None
    honba: int | None = None


def collect_round_yaku_records(
    mjai_log: Sequence[Mapping[str, Any]],
) -> tuple[RoundYakuRecord, ...]:
    """``env.mjai_log`` を ``MjaiReplay`` に通し、winner 毎の record を返す。

    - 入力は ``riichienv`` が出力する dict event の列。
    - ``riichienv.MjaiReplay.from_jsonl`` がファイル path のみを受け付けるため、
      一時 jsonl ファイル経由で渡し、関数を抜ける前に削除する。
    - ryukyoku round / hora の無い round は単に出力に含まれない。
    - ``MjaiReplay`` の構築または iteration が失敗した場合は例外を伝播する。
      呼び出し元 (runner) が fallback して既存 ``env.win_results`` ベースの
      backfill に倒す責務を負う。
    """
    with tempfile.TemporaryDirectory(prefix="mjai_replay_") as tmp_dir:
        path = Path(tmp_dir) / "log.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ev in mjai_log:
                f.write(
                    json.dumps(dict(ev), ensure_ascii=False, separators=(",", ":"))
                )
                f.write("\n")
        replay = riichienv.MjaiReplay.from_jsonl(str(path))
        records: list[RoundYakuRecord] = []
        for round_index, kyoku in enumerate(replay.take_kyokus()):
            kyoku_chang = _safe_int_attr(kyoku, "chang")
            kyoku_ju = _safe_int_attr(kyoku, "ju")
            kyoku_ben = _safe_int_attr(kyoku, "ben")
            ctxs = list(kyoku.take_win_result_contexts())
            for ctx in ctxs:
                wr = ctx.actual
                yaku_ids = tuple(int(y) for y in (getattr(wr, "yaku", []) or []))
                han = int(getattr(wr, "han", -1))
                fu = int(getattr(wr, "fu", -1))
                yakuman = bool(getattr(wr, "yakuman", False))
                seat = int(getattr(ctx, "seat", -1))
                agari_tile_attr = getattr(ctx, "agari_tile", None)
                agari_tile = (
                    int(agari_tile_attr) if agari_tile_attr is not None else None
                )
                records.append(
                    RoundYakuRecord(
                        round_index=int(round_index),
                        winner_seat=seat,
                        yaku_ids=yaku_ids,
                        han=han,
                        fu=fu,
                        yakuman=yakuman,
                        agari_tile=agari_tile,
                        round_wind=kyoku_chang,
                        oya=kyoku_ju,
                        honba=kyoku_ben,
                    )
                )
        return tuple(records)


def _safe_int_attr(obj: Any, name: str) -> int | None:
    """``getattr`` で int に変換可能ならその値、そうでなければ ``None``。"""
    v = getattr(obj, name, None)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


__all__ = ["RoundYakuRecord", "collect_round_yaku_records"]
