"""Per-(episode, round, player) sample weighting helper.

Stage02 で「同じ player の同じ round に多数 sample が積まれて loss が
偏る」のを抑えるために導入された weighting。Stage03 でも opt-in で
使えるようにする。

仕様:

- ``(episode_id, round_id, player_id)`` を key として、同一 key の sample
  の weight 合計が 1.0 になるよう normalize する (= 出現 count の逆数)。
- **trainer では minibatch-local に使う**: PPO (``_iter_minibatches``) /
  imitation (``compute_imitation_loss``) ともに、collate された 1 minibatch
  内の sample 集合に対してこの関数を呼ぶ。trajectory 全体や全 sample list
  に対しては呼ばない (= 2 trainer で flag の意味を揃えるため)。
- default は equal weight (= 1.0 / N) なので、weighting を OFF にしたい
  trainer 側は本関数を呼ばずに済む。
- 重み合計が 0 になる key (空 batch) は無視する。
- 戻り値は ``(N,) float32`` numpy。trainer 側で torch tensor 化する。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_per_player_round_weights(
    episode_ids: Sequence[str],
    round_ids: Sequence[int],
    player_ids: Sequence[int],
) -> np.ndarray:
    """同じ (episode, round, player) に属する sample の重み合計が 1.0 になる
    weight 配列を返す。

    Parameters
    ----------
    episode_ids:
        ``len == N`` の string sequence。
    round_ids:
        ``len == N`` の int sequence。
    player_ids:
        ``len == N`` の int sequence。

    Returns
    -------
    np.ndarray:
        ``(N,) float32``。各 sample i について ``1.0 / count[key_i]``。
        N=0 のときは空 array。
    """
    n = len(episode_ids)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if len(round_ids) != n or len(player_ids) != n:
        raise ValueError(
            f"per-player-round weighting: length mismatch "
            f"(episode={n}, round={len(round_ids)}, player={len(player_ids)})"
        )
    keys = [
        (str(episode_ids[i]), int(round_ids[i]), int(player_ids[i]))
        for i in range(n)
    ]
    counts: dict[tuple[str, int, int], int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    out = np.empty(n, dtype=np.float32)
    for i, k in enumerate(keys):
        out[i] = 1.0 / float(counts[k])
    return out


__all__ = ["compute_per_player_round_weights"]
