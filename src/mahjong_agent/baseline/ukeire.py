"""Ukeire (受け入れ枚数) 計算。

手牌 counts と現在 shanten に対し、各種牌を 1 枚追加して shanten が下がる
種類について、山に残っている枚数 (= 4 - 既見 counts) を合計する。

簡易版: 山に「ある」かどうかは捨て牌・副露の counts まで引いた現実値で
計算するのが望ましいが、本 helper は手牌側の counts のみを引数に取り、
``4 - counts[t]`` を残数として扱う。捨て牌・副露 counts を引いた強化版が
必要なら別 helper として上層で作る。
"""
from __future__ import annotations

from collections.abc import Sequence

from mahjong_agent.baseline.shanten import compute_shanten


def count_acceptance(
    counts: Sequence[int],
    shanten: int,
    meld_count: int = 0,
    *,
    seen_counts: Sequence[int] | None = None,
) -> int:
    """``counts`` の手牌に対する受け入れ枚数を返す。

    Parameters
    ----------
    counts:
        34 種 tile_type 枚数 (= ``tile_id // 4`` で集計した手牌)。
    shanten:
        現在の shanten 数 (``compute_shanten(counts, meld_count)`` の値)。
    meld_count:
        副露済み面子数。
    seen_counts:
        捨て牌 + 副露 + dora 表示 + 自分の他位置に出た牌で **既に公開** されて
        いる枚数。``None`` のときは ``counts`` のみを既見扱いする (= 山には
        ``4 - counts[t]`` 枚残っている前提)。
        Stage03 では public-only 範囲で全 player の discards / melds /
        dora indicators / 自手 を加算した seen_counts を上位から渡せる。

    Returns
    -------
    int:
        各 tile_type のうち「ツモれば shanten が下がる」もので、残り枚数の
        合計。
    """
    c = [int(x) for x in counts]
    if len(c) != 34:
        raise ValueError(f"counts must be length 34, got {len(c)}")
    if seen_counts is not None and len(seen_counts) != 34:
        raise ValueError(
            f"seen_counts must be length 34, got {len(seen_counts)}"
        )
    total = 0
    for t in range(34):
        # 自分の手牌に既に 4 枚あるとそれ以上ツモれない (山に無い)。
        if c[t] >= 4:
            continue
        remaining = 4 - (
            int(seen_counts[t]) if seen_counts is not None else c[t]
        )
        if remaining <= 0:
            continue
        c[t] += 1
        try:
            new_sh = compute_shanten(c, meld_count)
        finally:
            c[t] -= 1
        if new_sh < shanten:
            total += remaining
    return total


__all__ = ["count_acceptance"]
