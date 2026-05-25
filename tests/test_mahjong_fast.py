"""Tests for the optional C++ fast path (``mahjong_agent._mahjong_fast``).

- C++ ``compute_shanten`` / ``find_best_discard`` / ``analyze_discards`` が
  純 Python fallback と代表ケースで一致すること。
- C++ import unavailable を monkeypatch した fallback でも encoder が動くこと。
- encoder dim 440 維持。
- fast wrapper が hidden state を受け取らない (signature guard)。

C++ extension が未ビルドの環境では equivalence test を skip する
(fallback 経路自体は別 test でカバー)。
"""
from __future__ import annotations

import inspect
import random

import numpy as np
import pytest

from mahjong_agent.baseline import _fast
from mahjong_agent.baseline.discard_select import find_best_discard
from mahjong_agent.baseline.shanten import compute_shanten, compute_shanten_python

_FAST = _fast.FAST_AVAILABLE
_skip_no_fast = pytest.mark.skipif(
    not _FAST, reason="C++ _mahjong_fast extension not built in this env"
)


def _random_hand(rng: random.Random, n: int = 14) -> list[int]:
    counts = [0] * 34
    placed = 0
    while placed < n:
        t = rng.randrange(34)
        if counts[t] < 4:
            counts[t] += 1
            placed += 1
    return counts


# ----------------------------------------------------------------------
# C++ vs Python equivalence
# ----------------------------------------------------------------------


@_skip_no_fast
def test_cpp_compute_shanten_matches_python():
    rng = random.Random(0)
    for _ in range(3000):
        counts = _random_hand(rng, rng.choice([13, 14]))
        assert _fast.compute_shanten(counts, 0) == compute_shanten_python(
            counts, 0
        )


@_skip_no_fast
def test_cpp_compute_shanten_matches_python_open_hand():
    rng = random.Random(7)
    for _ in range(2000):
        meld_count = rng.randint(1, 4)
        n = 13 - 3 * meld_count + rng.choice([0, 1])
        if n < 1:
            continue
        counts = _random_hand(rng, n)
        assert _fast.compute_shanten(counts, meld_count) == (
            compute_shanten_python(counts, meld_count)
        )


@_skip_no_fast
def test_cpp_find_best_discard_matches_python():
    rng = random.Random(1)
    for _ in range(1500):
        counts = _random_hand(rng, 14)
        legal = [1 if counts[t] > 0 else 0 for t in range(34)]
        cpp = _fast.find_best_discard(counts, legal, 0)
        # Python fallback を強制
        prev = _fast.FAST_AVAILABLE
        _fast.FAST_AVAILABLE = False
        try:
            py = find_best_discard(
                counts, [float(x) for x in legal], meld_count=0
            )
        finally:
            _fast.FAST_AVAILABLE = prev
        assert cpp["best_shanten"] == py.best_shanten
        assert cpp["best_acceptance"] == py.best_acceptance
        assert cpp["best_tile"] == py.best_tile_type
        assert cpp["best_mask"] == [int(x) for x in py.best_mask.tolist()]


@_skip_no_fast
def test_cpp_analyze_discards_matches_python_encoder_loop():
    from mahjong_agent.baseline.ukeire import count_acceptance

    rng = random.Random(2)
    for _ in range(1000):
        counts = _random_hand(rng, 14)
        legal = [1] * 34
        an = _fast.analyze_discards(counts, legal, 0)
        # encoder semantics: tile in hand を decrement → shanten + acceptance
        for t in range(34):
            if counts[t] <= 0:
                continue
            counts[t] -= 1
            sh = compute_shanten_python(counts, 0)
            acc = count_acceptance(counts, sh, 0)
            counts[t] += 1
            assert an["shanten_after"][t] == sh
            assert an["acceptance"][t] == acc


# ----------------------------------------------------------------------
# public compute_shanten dispatches to fast / fallback identically
# ----------------------------------------------------------------------


def test_public_compute_shanten_consistent_with_python():
    """公開 ``compute_shanten`` は fast / fallback いずれでも Python 実装と一致。"""
    rng = random.Random(3)
    for _ in range(1000):
        counts = _random_hand(rng, rng.choice([13, 14]))
        expected = compute_shanten_python(counts, 0)
        assert compute_shanten(counts, 0) == expected


# ----------------------------------------------------------------------
# fallback (monkeypatch FAST_AVAILABLE=False)
# ----------------------------------------------------------------------


def test_compute_shanten_fallback_when_ext_unavailable(monkeypatch):
    monkeypatch.setattr(_fast, "FAST_AVAILABLE", False)
    rng = random.Random(4)
    for _ in range(500):
        counts = _random_hand(rng, 14)
        assert compute_shanten(counts, 0) == compute_shanten_python(counts, 0)


def test_find_best_discard_fallback_when_ext_unavailable(monkeypatch):
    monkeypatch.setattr(_fast, "FAST_AVAILABLE", False)
    rng = random.Random(5)
    counts = _random_hand(rng, 14)
    legal = [1.0 if counts[t] > 0 else 0.0 for t in range(34)]
    res = find_best_discard(counts, legal, meld_count=0)
    assert res.best_mask.shape == (34,)
    assert res.best_shanten < 999


def test_encoder_fallback_matches_fast(monkeypatch):
    """encoder hint が fast path / fallback で完全一致すること。"""
    import riichienv

    from mahjong_agent.encoders.public_observation import (
        PublicObservationEncoder,
    )

    enc = PublicObservationEncoder()
    env = riichienv.RiichiEnv(riichienv.GameType.YON_TONPUSEN)
    env.reset(seed=11)
    obs = env.get_observation(env.current_player)

    # fast path (利用可能なら) で 1 回
    feat_fast = enc.encode_observation(obs)
    # fallback 強制
    monkeypatch.setattr(_fast, "FAST_AVAILABLE", False)
    feat_py = enc.encode_observation(obs)
    assert np.array_equal(feat_fast, feat_py)


# ----------------------------------------------------------------------
# encoder dim / source guard
# ----------------------------------------------------------------------


def test_encoder_dim_still_440():
    from mahjong_agent.encoders.public_observation import (
        PublicObservationEncoder,
    )

    enc = PublicObservationEncoder(enable_hints=True)
    assert enc.metadata().observation_dim == 440
    enc_off = PublicObservationEncoder(enable_hints=False)
    assert enc_off.metadata().observation_dim == 363


def test_fast_wrapper_signatures_take_only_public_counts():
    """fast wrapper は counts / legal_mask / meld_count のみを受け取り、
    env / wall / hands / state を受け取らない (hidden info 境界)。"""
    for fn, expected in (
        (_fast.compute_shanten, {"counts", "meld_count"}),
        (_fast.analyze_discards, {"counts", "legal_mask", "meld_count"}),
        (_fast.find_best_discard, {"counts", "legal_mask", "meld_count"}),
    ):
        params = set(inspect.signature(fn).parameters.keys())
        assert params == expected
        forbidden = {"env", "wall", "hands", "state", "obs", "observation"}
        assert forbidden.isdisjoint(params)


def test_fast_source_does_not_reference_hidden_state():
    from pathlib import Path

    import mahjong_agent.baseline._fast as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    for tok in ("env.hands", "env.wall", "env.state", "full_state", "mjai_log"):
        assert tok not in src
