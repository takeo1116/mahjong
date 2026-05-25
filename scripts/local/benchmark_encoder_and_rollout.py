#!/usr/bin/env python3
"""Local-only perf benchmark for Stage03 encoder + agent rollouts.

ローカル運用 script: production code には触らず、計測のみを行う。pytest や
package import からは参照されない。

実行例:
    cd mahjong
    python3 scripts/local/benchmark_encoder_and_rollout.py
    # JSON dump
    python3 scripts/local/benchmark_encoder_and_rollout.py --json

計測項目:
1. PublicObservationEncoder.encode_observation の hints on/off 速度差。
2. shanten / count_acceptance / per-tile-type ukeire loop の内訳 timing
   (= encoder の hint 経路が支配的な部分)。
3. RandomAgent / RuleBasedBaselineAgent / ModelPolicyAgent それぞれの
   self-play 1 episode (YON_IKKYOKU) の wall-clock。
4. 各 agent の rollout で ``encode_observation`` が何回呼ばれているか
   (= ModelPolicyAgent の二重 encode 検出)。

注意:
- gitignored な path (``scripts/local/``) に置く。``.gitignore`` 上は
  既に ``runs/`` / ``outputs/`` 等の experiment artifact を除外する設定
  だが、``scripts/local/`` 自体は package layout に含まれない (= setuptools
  src layout は ``src/`` のみを対象)。
- production code (``mahjong_agent/**``) を一切変更しない。``encode_observation``
  の呼び出し回数集計のために class method を一時的に monkey-patch するが、
  context manager 終了時に元に戻す。
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from typing import Any

import riichienv  # noqa: F401  (rollout 経路で必要)
import torch

from mahjong_agent.agents import (
    ModelPolicyAgent,
    ModelPolicyConfig,
    RandomAgent,
    RuleBasedBaselineAgent,
)
from mahjong_agent.baseline import _fast
from mahjong_agent.baseline.discard_select import find_best_discard
from mahjong_agent.baseline.shanten import compute_shanten
from mahjong_agent.baseline.ukeire import count_acceptance
from mahjong_agent.encoders import PublicObservationEncoder
from mahjong_agent.evaluation import (
    SeatAgents,
    SelfPlayConfig,
    SelfPlayRunner,
)
from mahjong_agent.models import Stage03Model, Stage03ModelConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def count_encode_calls(encoder_cls=PublicObservationEncoder):
    """``encoder_cls.encode_observation`` の呼び出し回数を一時的に数える。

    context 終了時に元のメソッドへ戻す。production 経路は不変。
    """
    orig = encoder_cls.encode_observation
    counter = {"n": 0}

    def wrapped(self, obs):  # type: ignore[no-redef]
        counter["n"] += 1
        return orig(self, obs)

    encoder_cls.encode_observation = wrapped
    try:
        yield counter
    finally:
        encoder_cls.encode_observation = orig


def _real_obs(seed: int = 0):
    """YON_IKKYOKU の reset 後 observation を 1 件返す (encode bench 用)。"""
    env = riichienv.RiichiEnv(riichienv.GameType.YON_IKKYOKU)
    env.reset(seed=seed)
    return env.get_observation(env.current_player)


# ---------------------------------------------------------------------------
# Encoder benchmark
# ---------------------------------------------------------------------------


def bench_encoder(num_calls: int = 200) -> dict[str, Any]:
    """``encode_observation`` を hints on / off で同 obs に対して回す。"""
    obs = _real_obs(seed=0)
    out: dict[str, Any] = {}

    enc_off = PublicObservationEncoder(enable_hints=False)
    enc_off.encode_observation(obs)  # warmup
    t0 = time.perf_counter()
    for _ in range(num_calls):
        enc_off.encode_observation(obs)
    out["hints_off_ms_per_call"] = (
        (time.perf_counter() - t0) / num_calls * 1000
    )
    out["hints_off_dim"] = enc_off.metadata().observation_dim

    enc_on = PublicObservationEncoder(enable_hints=True)
    enc_on.encode_observation(obs)  # warmup
    t0 = time.perf_counter()
    for _ in range(num_calls):
        enc_on.encode_observation(obs)
    out["hints_on_ms_per_call"] = (
        (time.perf_counter() - t0) / num_calls * 1000
    )
    out["hints_on_dim"] = enc_on.metadata().observation_dim

    out["hints_overhead_x"] = (
        out["hints_on_ms_per_call"] / out["hints_off_ms_per_call"]
        if out["hints_off_ms_per_call"] > 0
        else 0.0
    )
    return out


def bench_hint_components(num_calls: int = 300) -> dict[str, Any]:
    """hint 計算の主要 component 個別 timing。

    encoder 内部の hint 計算は概ね:
        - current shanten: 1 回 compute_shanten
        - per-tile-type loop: 手牌に存在する tile_type ごとに
          compute_shanten + count_acceptance (= +34 shanten)
        - riichi_discard_mask / remaining / tile_presence: 軽微

    本 bench は per-tile-type loop が支配的かを示すためのもの。
    """
    obs = _real_obs(seed=0)
    counts = [0] * 34
    for tid in obs.hand:
        counts[int(tid) // 4] += 1
    distinct = sum(1 for c in counts if c > 0)
    out: dict[str, Any] = {
        "hand_distinct_tile_types": distinct,
        "hand_total_tiles": sum(counts),
    }

    # 1) compute_shanten 単発
    compute_shanten(counts, 0)  # warmup
    t0 = time.perf_counter()
    for _ in range(num_calls):
        compute_shanten(counts, 0)
    out["compute_shanten_us_per_call"] = (
        (time.perf_counter() - t0) / num_calls * 1e6
    )

    # 2) count_acceptance 単発 (内部で 34 回 shanten)
    sh = compute_shanten(counts, 0)
    count_acceptance(counts, sh, 0)  # warmup
    t0 = time.perf_counter()
    for _ in range(num_calls):
        count_acceptance(counts, sh, 0)
    out["count_acceptance_us_per_call"] = (
        (time.perf_counter() - t0) / num_calls * 1e6
    )

    # 3) encoder と同等の per-tile-type loop (shanten_delta + ukeire)
    def per_tile_loop():
        c = list(counts)
        for t in range(34):
            if c[t] <= 0:
                continue
            c[t] -= 1
            sh_a = compute_shanten(c, 0)
            count_acceptance(c, sh_a, 0)
            c[t] += 1

    per_tile_loop()  # warmup
    t0 = time.perf_counter()
    for _ in range(num_calls):
        per_tile_loop()
    out["per_tile_loop_ms_per_call"] = (
        (time.perf_counter() - t0) / num_calls * 1000
    )
    return out


# ---------------------------------------------------------------------------
# Rollout benchmark
# ---------------------------------------------------------------------------


def _runner() -> SelfPlayRunner:
    return SelfPlayRunner(
        config=SelfPlayConfig(
            game_type=riichienv.GameType.YON_IKKYOKU,
            max_steps_per_game=4000,
        )
    )


def bench_rollout(agent_factory, label: str, num_seeds: int = 2) -> dict[str, Any]:
    """1 episode の wall-clock + encode 呼び出し回数を計測する。"""
    runner = _runner()
    wall_clocks: list[float] = []
    encode_counts: list[int] = []
    sample_counts: list[int] = []
    for seed in range(num_seeds):
        sa = agent_factory()
        with count_encode_calls() as counter:
            t0 = time.perf_counter()
            res = runner.run_episode(sa, seed=seed)
            dt = time.perf_counter() - t0
            n_encode = counter["n"]
        wall_clocks.append(dt)
        encode_counts.append(n_encode)
        sample_counts.append(len(res.samples))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    encode_per_sample = [
        ec / ns if ns > 0 else 0.0
        for ec, ns in zip(encode_counts, sample_counts, strict=True)
    ]
    return {
        "label": label,
        "num_seeds": num_seeds,
        "wall_clock_s_each": wall_clocks,
        "wall_clock_s_mean": _mean(wall_clocks),
        "samples_each": sample_counts,
        "samples_mean": _mean(sample_counts),
        "encode_calls_each": encode_counts,
        "encode_calls_mean": _mean(encode_counts),
        "encode_calls_per_sample_each": encode_per_sample,
        "encode_calls_per_sample_mean": _mean(encode_per_sample),
    }


def make_random_sa() -> SeatAgents:
    return SeatAgents.homogeneous(
        RandomAgent(seed=0), actor_type="random", num_players=4,
    )


def make_rule_sa() -> SeatAgents:
    return SeatAgents.homogeneous(
        RuleBasedBaselineAgent(), actor_type="rule_based", num_players=4,
    )


def make_model_sa() -> SeatAgents:
    """ModelPolicyAgent (default config) の 4 席。"""
    enc = PublicObservationEncoder(enable_hints=True)
    cfg = Stage03ModelConfig.from_encoder_metadata(
        enc.metadata(),
        hidden_dim=64,
        trunk_layers=1,
        candidate_hidden_dim=32,
    )
    model = Stage03Model(cfg).eval()
    agent = ModelPolicyAgent(
        model, enc, ModelPolicyConfig(greedy=False, temperature=1.0), seed=0,
    )
    return SeatAgents.homogeneous(agent, actor_type="policy", num_players=4)


# ---------------------------------------------------------------------------
# C++ fast path vs Python fallback comparison (ISSUE-0017)
# ---------------------------------------------------------------------------


@contextmanager
def _force_fast(enabled: bool):
    """``_fast.FAST_AVAILABLE`` を一時的に切り替える (context 終了で復元)。"""
    prev = _fast.FAST_AVAILABLE
    _fast.FAST_AVAILABLE = bool(enabled and prev)
    try:
        yield _fast.FAST_AVAILABLE
    finally:
        _fast.FAST_AVAILABLE = prev


def bench_fast_vs_fallback(
    *, encode_n: int = 200, fbd_n: int = 200, teacher_games: int = 10
) -> dict[str, Any]:
    """fast path enabled / fallback forced を比較する。"""
    out: dict[str, Any] = {"ext_available": bool(_fast.FAST_AVAILABLE)}
    obs = _real_obs(seed=0)
    counts = [0] * 34
    for tid in obs.hand:
        counts[int(tid) // 4] += 1
    legal = [1.0 if counts[t] > 0 else 0.0 for t in range(34)]

    def _time_encode() -> float:
        enc = PublicObservationEncoder(enable_hints=True)
        enc.encode_observation(obs)  # warmup
        t0 = time.perf_counter()
        for _ in range(encode_n):
            enc.encode_observation(obs)
        return (time.perf_counter() - t0) / encode_n * 1000

    def _time_fbd() -> float:
        find_best_discard(counts, legal, meld_count=0)  # warmup
        t0 = time.perf_counter()
        for _ in range(fbd_n):
            find_best_discard(counts, legal, meld_count=0)
        return (time.perf_counter() - t0) / fbd_n * 1000

    def _time_teacher_rollout() -> float:
        runner = SelfPlayRunner(
            config=SelfPlayConfig(
                game_type=riichienv.GameType.YON_TONPUSEN,
                max_steps_per_game=8000,
            )
        )
        t0 = time.perf_counter()
        for seed in range(teacher_games):
            sa = make_rule_sa()
            runner.run_episode(sa, seed=seed)
        return time.perf_counter() - t0

    for label, enabled in (("fast", True), ("fallback", False)):
        with _force_fast(enabled) as active:
            out[f"{label}_active"] = bool(active)
            out[f"{label}_encode_hints_on_ms_per_call"] = _time_encode()
            out[f"{label}_find_best_discard_ms_per_call"] = _time_fbd()
            out[f"{label}_teacher_rollout_{teacher_games}games_s"] = (
                _time_teacher_rollout()
            )

    # speedup ratios
    if out.get("fallback_encode_hints_on_ms_per_call", 0) > 0:
        out["encode_speedup_x"] = (
            out["fallback_encode_hints_on_ms_per_call"]
            / out["fast_encode_hints_on_ms_per_call"]
            if out["fast_encode_hints_on_ms_per_call"] > 0
            else 0.0
        )
    if out.get(f"fallback_teacher_rollout_{teacher_games}games_s", 0) > 0:
        fast_s = out[f"fast_teacher_rollout_{teacher_games}games_s"]
        out["teacher_rollout_speedup_x"] = (
            out[f"fallback_teacher_rollout_{teacher_games}games_s"] / fast_s
            if fast_s > 0
            else 0.0
        )
        out["fast_teacher_rollout_s_per_game"] = fast_s / max(1, teacher_games)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _print_section(title: str, data: dict[str, Any]) -> None:
    print(f"\n[{title}]")
    for k, v in data.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true",
                        help="Also dump aggregated results as JSON.")
    parser.add_argument("--encode-n", type=int, default=200,
                        help="Number of encode calls per scenario.")
    parser.add_argument("--hint-n", type=int, default=300,
                        help="Number of hint-component repetitions.")
    parser.add_argument("--rollout-seeds", type=int, default=2,
                        help="Seeds per agent for rollout benchmark.")
    parser.add_argument("--compare-fast", action="store_true",
                        help="Compare C++ fast path vs Python fallback.")
    parser.add_argument("--teacher-games", type=int, default=10,
                        help="Teacher rollout games for --compare-fast.")
    args = parser.parse_args()

    torch.manual_seed(0)

    if args.compare_fast:
        cmp_bench = bench_fast_vs_fallback(
            encode_n=args.encode_n,
            teacher_games=args.teacher_games,
        )
        _print_section("Fast path vs fallback", cmp_bench)
        if args.json:
            print("\n[JSON]")
            print(json.dumps(cmp_bench, indent=2, ensure_ascii=False))
        return

    encoder_bench = bench_encoder(num_calls=args.encode_n)
    _print_section("Encoder benchmark", encoder_bench)

    hint_bench = bench_hint_components(num_calls=args.hint_n)
    _print_section("Hint component timing", hint_bench)

    rollout_results: list[dict[str, Any]] = []
    for factory, label in (
        (make_random_sa, "RandomAgent"),
        (make_rule_sa, "RuleBasedBaselineAgent"),
        (make_model_sa, "ModelPolicyAgent"),
    ):
        r = bench_rollout(factory, label, num_seeds=args.rollout_seeds)
        rollout_results.append(r)
        _print_section(f"Rollout: {label}", r)

    if args.json:
        out = {
            "encoder_benchmark": encoder_bench,
            "hint_components": hint_bench,
            "rollout_benchmarks": rollout_results,
        }
        print("\n[JSON]")
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
