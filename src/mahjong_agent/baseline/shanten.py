"""Shanten 計算 (Python 実装)。

標準形 (4 面子 1 雀頭) + 七対子 + 国士の minimum を返す。副露がある
(`meld_count > 0`) ときは標準形のみ。実装は recursive な mentsu 抽出 +
不完全面子 (対子・ターツ) の貪欲カウントで、典型 14 牌手で 1 ms 以下。

入力 ``counts`` は 34 種 tile_type の枚数 (= ``tile_id // 4`` で集計した数)。
hidden info は使わず、agent / teacher が自分の手牌から計算する用途のみ。
"""
from __future__ import annotations

from collections.abc import Sequence

# 么九牌 (1m, 9m, 1p, 9p, 1s, 9s, 東, 南, 西, 北, 白, 發, 中)
_TERMINALS_AND_HONORS: tuple[int, ...] = (
    0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33,
)


def compute_shanten(
    counts: Sequence[int],
    meld_count: int = 0,
) -> int:
    """34 種 tile_type 枚数からシャンテン数を返す。

    Parameters
    ----------
    counts:
        長さ 34 の整数列。``tile_id // 4`` で集計した手牌枚数。
        副露牌は **含めない** (面子として除外して ``meld_count`` で渡す)。
    meld_count:
        副露済み面子数 (0..4)。``>0`` のとき七対子 / 国士をスキップする。

    Returns
    -------
    int:
        ``-1`` で和了、``0`` でテンパイ、``1`` で 1 向聴、... ``8`` まで。
    """
    if len(counts) != 34:
        raise ValueError(
            f"shanten counts must be length 34, got {len(counts)}"
        )
    c = [int(x) for x in counts]
    meld_count = int(meld_count)
    if meld_count > 0:
        return _regular_shanten(c, meld_count)
    return min(
        _regular_shanten(c, 0),
        _chiitoitsu_shanten(c),
        _kokushi_shanten(c),
    )


def _kokushi_shanten(counts: list[int]) -> int:
    """国士無双の向聴数。"""
    kinds = 0
    has_pair = False
    for t in _TERMINALS_AND_HONORS:
        if counts[t] > 0:
            kinds += 1
            if counts[t] >= 2:
                has_pair = True
    return 13 - kinds - (1 if has_pair else 0)


def _chiitoitsu_shanten(counts: list[int]) -> int:
    """七対子の向聴数。"""
    pairs = 0
    kinds = 0
    for c in counts:
        if c >= 1:
            kinds += 1
        if c >= 2:
            pairs += 1
    base = 6 - pairs
    # 7 種未満だと 7 種を満たすまでに必ず牌種を増やす必要がある。
    if kinds < 7:
        base += 7 - kinds
    return base


def _regular_shanten(counts: list[int], meld_count: int = 0) -> int:
    """標準形 (4 面子 1 雀頭) の向聴数。

    雀頭の取り方を分岐し、面子分解を探索する。``meld_count`` は副露で既に
    確定済みの面子数として加える。
    """
    best = [8 - 2 * meld_count]
    c = list(counts)

    # 雀頭なし
    _remove_groups(c, 0, meld_count, 0, best)

    # 各牌種を雀頭として取って探索
    for t in range(34):
        if c[t] >= 2:
            c[t] -= 2
            _remove_groups(c, 0, meld_count, 1, best)
            c[t] += 2

    return best[0]


def _remove_groups(
    counts: list[int],
    pos: int,
    mentsu: int,
    jantai: int,
    best: list[int],
) -> None:
    """完成面子 (刻子・順子) を抽出し、残りから不完全面子を数える。

    再帰中は ``counts`` を破壊的に書き換え、戻る前に元に戻す。
    """
    # 4 面子に達したら不完全面子は要らない
    if mentsu >= 4:
        shanten = 8 - 2 * 4 - jantai
        if shanten < best[0]:
            best[0] = shanten
        return

    # 残牌がある位置を探す
    idx = pos
    while idx < 34 and counts[idx] == 0:
        idx += 1

    if idx >= 34:
        # 全位置処理済み → 残り牌から不完全面子をカウント
        partial = _count_partial(counts)
        max_partial = 4 - mentsu
        if partial > max_partial:
            partial = max_partial
        shanten = 8 - 2 * mentsu - partial - jantai
        if shanten < best[0]:
            best[0] = shanten
        return

    # 枝刈り: 残りの最大改善でも best 以下にならないなら打ち切り
    remaining_tiles = 0
    for i in range(idx, 34):
        remaining_tiles += counts[i]
    max_more_mentsu = remaining_tiles // 3
    max_total_mentsu = min(4, mentsu + max_more_mentsu)
    lower_bound = (
        8 - 2 * max_total_mentsu - (4 - max_total_mentsu) - jantai
    )
    if lower_bound >= best[0]:
        return

    # 刻子
    if counts[idx] >= 3:
        counts[idx] -= 3
        _remove_groups(counts, idx, mentsu + 1, jantai, best)
        counts[idx] += 3

    # 順子 (数牌のみ)
    suit = idx // 9
    rel = idx % 9
    if suit < 3 and rel <= 6:
        base = suit * 9
        if counts[base + rel + 1] > 0 and counts[base + rel + 2] > 0:
            counts[idx] -= 1
            counts[base + rel + 1] -= 1
            counts[base + rel + 2] -= 1
            _remove_groups(counts, idx, mentsu + 1, jantai, best)
            counts[idx] += 1
            counts[base + rel + 1] += 1
            counts[base + rel + 2] += 1

    # この位置で面子を取らずに次へ
    _remove_groups(counts, idx + 1, mentsu, jantai, best)


def _count_partial(counts: list[int]) -> int:
    """残り牌から不完全面子 (対子・ターツ) を貪欲に数える。

    数牌は順次 (両面 / 嵌張) + 対子を貪欲に取る。字牌は対子のみ。
    """
    c = list(counts)
    partial = 0

    for suit in range(3):
        base = suit * 9
        # 対子
        for i in range(9):
            t = base + i
            if c[t] >= 2:
                c[t] -= 2
                partial += 1
        # 両面 / 連続
        for i in range(8):
            t = base + i
            if c[t] > 0 and c[t + 1] > 0:
                c[t] -= 1
                c[t + 1] -= 1
                partial += 1
        # 嵌張
        for i in range(7):
            t = base + i
            if c[t] > 0 and c[t + 2] > 0:
                c[t] -= 1
                c[t + 2] -= 1
                partial += 1

    # 字牌の対子
    for t in range(27, 34):
        if c[t] >= 2:
            partial += 1

    return partial


__all__ = ["compute_shanten"]
