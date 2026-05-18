"""JSON-serializable metrics aggregator for self-play / evaluation runs.

各 ``EpisodeResult`` から最低限の game-level / sample-level 統計を取り、
1 つの dict として返す。downstream (experiment driver / report) で JSON
保存できることを test で確認する。
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mahjong_agent.evaluation.runner import EpisodeResult


def aggregate_metrics(
    episode_results: Iterable[EpisodeResult],
) -> dict[str, Any]:
    """``EpisodeResult`` list から JSON serializable な summary dict を返す。

    Keys
    ----
    num_games / num_rounds / num_steps / num_samples / crash_count:
        scalar 統計。
    seeds:
        各 game の seed list。
    final_scores / final_ranks:
        per-game の (player_id 順) スコア / 順位の list of list。
    rank_counts:
        ``{actor_type: {"1": int, "2": int, "3": int, "4": int}}`` の dict。
    avg_rank_by_actor_type:
        ``{actor_type: float}``。
    win_count / tsumo_count / ron_count:
        全 game 合計の和了数。
    decision_family_counts:
        ``{family: int}``。
    actor_type_counts:
        ``{actor_type: int}``。
    crashes:
        crash した game の crash_context list。
    """
    episodes = list(episode_results)
    num_games = len(episodes)
    seeds: list[int] = []
    wall_seeds: list[int | None] = []
    wall_digests: list[str | None] = []
    final_scores: list[list[int]] = []
    final_ranks: list[list[int]] = []
    num_rounds = 0
    num_steps = 0
    num_samples = 0
    crash_count = 0
    crashes: list[dict[str, Any]] = []
    win_count = 0
    tsumo_count = 0
    ron_count = 0
    decision_family_counter: Counter[str] = Counter()
    actor_type_counter: Counter[str] = Counter()
    rank_by_actor: dict[str, list[int]] = {}

    for ep in episodes:
        seeds.append(int(ep.seed))
        wall_seeds.append(int(ep.wall_seed) if ep.wall_seed is not None else None)
        wall_digests.append(
            str(ep.wall_digest) if ep.wall_digest is not None else None
        )
        final_scores.append([int(x) for x in ep.final_scores])
        final_ranks.append([int(x) for x in ep.final_ranks])
        num_rounds += int(ep.num_rounds)
        num_steps += int(ep.num_steps)
        num_samples += len(ep.samples)
        if ep.crash_context is not None:
            crash_count += 1
            crashes.append(dict(ep.crash_context))
        # per-sample counters
        for s in ep.samples:
            decision_family_counter[str(s.decision_family)] += 1
            actor_type_counter[str(s.actor_type)] += 1
        # win / tsumo / ron counters from round summaries
        for rs in ep.round_summaries:
            if rs.get("winner") is not None and not rs.get("is_draw", False):
                win_count += 1
                if rs.get("is_tsumo", False):
                    tsumo_count += 1
                else:
                    ron_count += 1
        # rank per actor
        for pid, rank in enumerate(ep.final_ranks):
            actor_type = str(ep.actor_types.get(int(pid), "unknown"))
            rank_by_actor.setdefault(actor_type, []).append(int(rank))

    rank_counts: dict[str, dict[str, int]] = {}
    avg_rank_by_actor_type: dict[str, float] = {}
    for actor_type, ranks in rank_by_actor.items():
        c: Counter[int] = Counter(ranks)
        rank_counts[actor_type] = {
            str(k): int(c.get(k, 0)) for k in sorted(set([1, 2, 3, 4]) | set(c))
        }
        if ranks:
            avg_rank_by_actor_type[actor_type] = float(
                sum(ranks) / len(ranks)
            )
        else:
            avg_rank_by_actor_type[actor_type] = 0.0

    return {
        "num_games": int(num_games),
        "num_rounds": int(num_rounds),
        "num_steps": int(num_steps),
        "num_samples": int(num_samples),
        "crash_count": int(crash_count),
        "seeds": seeds,
        "wall_seeds": wall_seeds,
        "wall_digests": wall_digests,
        "final_scores": final_scores,
        "final_ranks": final_ranks,
        "rank_counts": rank_counts,
        "avg_rank_by_actor_type": avg_rank_by_actor_type,
        "win_count": int(win_count),
        "tsumo_count": int(tsumo_count),
        "ron_count": int(ron_count),
        "decision_family_counts": {
            k: int(v) for k, v in decision_family_counter.items()
        },
        "actor_type_counts": {
            k: int(v) for k, v in actor_type_counter.items()
        },
        "crashes": crashes,
    }


__all__ = ["aggregate_metrics"]
