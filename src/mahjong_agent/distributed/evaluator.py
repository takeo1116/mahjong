"""File-based distributed evaluator + best checkpoint selection.

distributed run の ``checkpoints/`` を監視し、各 checkpoint を ``rule_based``
opponent との mixed-seat 評価にかけて ``metrics/eval.jsonl`` に結果を残し、
best checkpoint を ``best/`` に publish する。

learner の checkpoint publish path (``checkpoints/`` + ``latest.json``) には
**一切書き込まない** (evaluator は read-only で評価し、成果物は ``eval/`` /
``best/`` / ``metrics/eval.jsonl`` にのみ書く)。

評価は既存の self-play スタック (``SelfPlayRunner`` + ``ModelPolicyAgent`` +
``RuleBasedBaselineAgent``) を流用し、policy を 4 席に rotate させて 1 席を
policy、残り 3 席を rule_based にした mixed-seat で avg_rank 等を測る。

hidden info は扱わない (manifest / best.json は dim / version / hash / metric の
メタのみ、sample は public observation feature のみ)。全体設計は
``docs/distributed_actor_learner.md``。
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import riichienv
import torch

from mahjong_agent.agents import (
    ModelPolicyAgent,
    ModelPolicyConfig,
    RuleBasedBaselineAgent,
)
from mahjong_agent.data.types import SCHEMA_VERSION
from mahjong_agent.distributed.manifest import (
    read_json,
    sha256_file,
    write_json_atomic,
)
from mahjong_agent.encoders import PublicObservationEncoder
from mahjong_agent.evaluation import SeatAgents, SelfPlayConfig, SelfPlayRunner
from mahjong_agent.models import Stage03Model, Stage03ModelConfig

# call とみなす decision family (riichi_rate / call_rate proxy 用)。
_CALL_FAMILIES = frozenset({"chi", "pon", "ankan", "kakan", "daiminkan"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_device(name: str | None) -> str:
    if name:
        return str(name)
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalMode:
    name: str
    greedy: bool
    temperature: float


def build_modes(mode_names: list[str], *, low_temp: float) -> list[EvalMode]:
    out: list[EvalMode] = []
    for name in mode_names:
        if name == "greedy":
            out.append(EvalMode("greedy", greedy=True, temperature=1.0))
        elif name == "stochastic":
            out.append(EvalMode("stochastic", greedy=False, temperature=1.0))
        elif name == "low_temp":
            out.append(EvalMode("low_temp", greedy=False, temperature=float(low_temp)))
        else:
            raise ValueError(f"unknown eval mode: {name!r}")
    return out


@dataclass(frozen=True)
class EvaluatorConfig:
    run_root: str
    checkpoint_version: str = "latest"  # "latest" | int | "unevaluated"
    eval_games_per_seat: int = 25
    mode_names: tuple[str, ...] = ("greedy",)
    low_temp: float = 0.2
    opponent: str = "rule_based"
    device: str | None = None
    game_type_name: str = "YON_TONPUSEN"
    seed_start: int = 700_000
    max_steps_per_game: int = 4000
    poll_interval_sec: float = 5.0
    stop_after_checkpoints: int = 1
    max_poll_iterations: int = 0
    best_metric: str = "avg_rank"
    best_lower_is_better: bool = True
    metrics_jsonl: str | None = None
    dry_run: bool = False
    seed: int = 42


# ---------------------------------------------------------------------------
# directory layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RunDirs:
    root: Path
    checkpoints: Path
    eval: Path
    best: Path
    metrics: Path

    @classmethod
    def from_root(cls, run_root: str | Path) -> _RunDirs:
        root = Path(run_root)
        return cls(
            root=root,
            checkpoints=root / "checkpoints",
            eval=root / "eval",
            best=root / "best",
            metrics=root / "metrics",
        )

    def ensure(self) -> None:
        self.eval.mkdir(parents=True, exist_ok=True)
        self.best.mkdir(parents=True, exist_ok=True)
        self.metrics.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# checkpoint discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Target:
    policy_version: int
    checkpoint_path: str  # run-root 相対
    checkpoint_sha256: str
    model_config: dict[str, Any]
    encoder_metadata: dict[str, Any] | None


def _target_from_checkpoint(dirs: _RunDirs, policy_version: int) -> _Target:
    rel = f"checkpoints/policy_{policy_version:06d}.pt"
    ckpt = dirs.root / rel
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu")
    if "model_config" not in payload:
        raise ValueError(f"invalid checkpoint payload: {ckpt}")
    return _Target(
        policy_version=int(policy_version),
        checkpoint_path=rel,
        checkpoint_sha256=sha256_file(ckpt),
        model_config=dict(payload["model_config"]),
        encoder_metadata=payload.get("encoder_metadata"),
    )


def _all_checkpoint_versions(dirs: _RunDirs) -> list[int]:
    out: list[int] = []
    if not dirs.checkpoints.is_dir():
        return out
    for p in dirs.checkpoints.glob("policy_*.pt"):
        stem = p.stem  # policy_000123
        try:
            out.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def discover_targets(dirs: _RunDirs, config: EvaluatorConfig, eval_rows: list[dict[str, Any]]) -> list[_Target]:
    """評価対象 checkpoint を解決する (``latest`` / ``N`` / ``unevaluated``)。"""
    spec = str(config.checkpoint_version)
    if spec == "latest":
        latest = dirs.checkpoints / "latest.json"
        if not latest.is_file():
            raise FileNotFoundError(f"checkpoint registry not found: {latest}")
        pv = int(read_json(latest)["policy_version"])
        return [_target_from_checkpoint(dirs, pv)]
    if spec in ("unevaluated", "all", "all-unevaluated"):
        modes = list(config.mode_names)
        targets: list[_Target] = []
        for pv in _all_checkpoint_versions(dirs):
            # 全 mode が既に評価済みの version は skip。
            if all(
                _already_evaluated(eval_rows, pv, m, config)
                for m in modes
            ):
                continue
            targets.append(_target_from_checkpoint(dirs, pv))
            if (
                int(config.stop_after_checkpoints) > 0
                and len(targets) >= int(config.stop_after_checkpoints)
            ):
                break
        return targets
    # 数値指定
    try:
        pv = int(spec)
    except ValueError as exc:
        raise ValueError(
            f"--checkpoint-version must be latest|int|unevaluated, got {spec!r}"
        ) from exc
    return [_target_from_checkpoint(dirs, pv)]


def _already_evaluated(
    eval_rows: list[dict[str, Any]],
    policy_version: int,
    mode_name: str,
    config: EvaluatorConfig,
) -> bool:
    for r in eval_rows:
        if (
            r.get("event") == "checkpoint_eval"
            and int(r.get("policy_version", -1)) == int(policy_version)
            and str(r.get("mode")) == str(mode_name)
            and str(r.get("opponent")) == str(config.opponent)
            and int(r.get("eval_games_per_seat", -1))
            == int(config.eval_games_per_seat)
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# model / encoder build
# ---------------------------------------------------------------------------


def _build_model_encoder(dirs: _RunDirs, target: _Target, device: str) -> tuple[Stage03Model, PublicObservationEncoder]:
    ckpt = (dirs.root / target.checkpoint_path).resolve()
    payload = torch.load(ckpt, map_location="cpu")
    model = Stage03Model(Stage03ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    model.to(torch.device(device))
    enable_hints = True
    if isinstance(target.encoder_metadata, dict):
        enable_hints = bool(target.encoder_metadata.get("enable_hints", True))
    encoder = PublicObservationEncoder(enable_hints=enable_hints)
    return model, encoder


def _make_opponent(opponent: str, seed: int):
    if opponent == "rule_based":
        return RuleBasedBaselineAgent(prefer_riichi=True, seed=seed)
    raise ValueError(f"unsupported opponent: {opponent!r}")


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate_policy(
    model: Stage03Model,
    encoder: PublicObservationEncoder,
    *,
    mode: EvalMode,
    config: EvaluatorConfig,
    device: str,
) -> dict[str, Any]:
    """policy を 4 席に rotate して mixed-seat 評価し、policy 視点 metrics を返す。"""
    start = time.time()
    runner = SelfPlayRunner(
        encoder=encoder,
        config=SelfPlayConfig(
            game_type=getattr(riichienv.GameType, config.game_type_name),
            max_steps_per_game=int(config.max_steps_per_game),
            collect_samples=True,
            deterministic_wall=True,
        ),
    )
    policy_agent = ModelPolicyAgent(
        model=model,
        encoder=encoder,
        config=ModelPolicyConfig(
            greedy=bool(mode.greedy),
            temperature=float(mode.temperature),
            device=device,
        ),
        seed=int(config.seed),
    )
    games_per_seat = int(config.eval_games_per_seat)
    policy_ranks: list[int] = []
    policy_scores: list[int] = []
    win_rounds = 0
    deal_in_rounds = 0
    total_rounds = 0
    family_counter: Counter[str] = Counter()
    num_games = 0
    crashes = 0
    for policy_seat in range(4):
        pairs: dict[int, tuple[Any, str]] = {}
        for pid in range(4):
            if pid == policy_seat:
                pairs[pid] = (policy_agent, "policy")
            else:
                pairs[pid] = (
                    _make_opponent(
                        config.opponent,
                        int(config.seed) + 100 + pid + policy_seat * 10,
                    ),
                    config.opponent,
                )
        seat_agents = SeatAgents.from_pairs(pairs)
        base = int(config.seed_start) + policy_seat * games_per_seat
        seeds = list(range(base, base + games_per_seat))
        results = runner.run_games(seat_agents, seeds=seeds)
        for ep in results:
            if ep.crash_context is not None:
                crashes += 1
                continue
            num_games += 1
            policy_ranks.append(int(ep.final_ranks[policy_seat]))
            policy_scores.append(int(ep.final_scores[policy_seat]))
            for rs in ep.round_summaries:
                total_rounds += 1
                if rs.get("winner") == policy_seat and not rs.get("is_draw", False):
                    win_rounds += 1
                if rs.get("deal_in") == policy_seat:
                    deal_in_rounds += 1
            for s in ep.samples:
                if str(s.actor_type) == "policy":
                    family_counter[str(s.decision_family)] += 1

    rank_c: Counter[int] = Counter(policy_ranks)
    rank_counts = {str(k): int(rank_c.get(k, 0)) for k in (1, 2, 3, 4)}
    avg_rank = (
        float(sum(policy_ranks) / len(policy_ranks)) if policy_ranks else 0.0
    )
    mean_score = (
        float(sum(policy_scores) / len(policy_scores)) if policy_scores else 0.0
    )
    total_decisions = int(sum(family_counter.values()))
    riichi_rate = (
        float(family_counter.get("riichi_discard", 0)) / total_decisions
        if total_decisions > 0 else 0.0
    )
    call_rate = (
        float(sum(family_counter.get(f, 0) for f in _CALL_FAMILIES))
        / total_decisions
        if total_decisions > 0 else 0.0
    )
    return {
        "avg_rank": avg_rank,
        "rank_counts": rank_counts,
        "mean_score": mean_score,
        "win_rate": float(win_rounds / total_rounds) if total_rounds else 0.0,
        "deal_in_rate": (
            float(deal_in_rounds / total_rounds) if total_rounds else 0.0
        ),
        "riichi_rate": riichi_rate,
        "call_rate": call_rate,
        "num_games": int(num_games),
        "crash_count": int(crashes),
        "elapsed_sec": round(time.time() - start, 3),
    }


# ---------------------------------------------------------------------------
# best selection
# ---------------------------------------------------------------------------


def _read_best(dirs: _RunDirs) -> dict[str, Any] | None:
    best_json = dirs.best / "best.json"
    if not best_json.is_file():
        return None
    return read_json(best_json)


def _is_better(new_value: float, cur_value: float, *, lower_is_better: bool) -> bool:
    # 厳密に良いときのみ更新 (同点は既存 best を維持)。
    return new_value < cur_value if lower_is_better else new_value > cur_value


def _maybe_update_best(
    dirs: _RunDirs,
    *,
    target: _Target,
    mode: EvalMode,
    config: EvaluatorConfig,
    eval_result: dict[str, Any],
) -> dict[str, Any] | None:
    """best metric が改善していれば best.json + checkpoint copy を publish。"""
    metric = str(config.best_metric)
    if metric not in eval_result:
        return None
    value = float(eval_result[metric])
    current = _read_best(dirs)
    if current is not None and not _is_better(
        value,
        float(current.get("metric_value")),
        lower_is_better=bool(config.best_lower_is_better),
    ):
        return None

    import shutil

    src = (dirs.root / target.checkpoint_path).resolve()
    dst = dirs.best / f"policy_{target.policy_version:06d}.pt"
    dirs.best.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    best_entry = {
        "policy_version": int(target.policy_version),
        "checkpoint_path": target.checkpoint_path,
        "checkpoint_sha256": target.checkpoint_sha256,
        "best_checkpoint_copy": f"best/policy_{target.policy_version:06d}.pt",
        "metric": metric,
        "metric_value": value,
        "lower_is_better": bool(config.best_lower_is_better),
        "mode": mode.name,
        "opponent": config.opponent,
        "eval_games_per_seat": int(config.eval_games_per_seat),
        "schema_version": SCHEMA_VERSION,
        "updated_at": _utc_now_iso(),
    }
    write_json_atomic(dirs.best / "best.json", best_entry)
    return best_entry


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _metrics_path(dirs: _RunDirs, config: EvaluatorConfig) -> Path:
    if config.metrics_jsonl:
        return Path(config.metrics_jsonl)
    return dirs.metrics / "eval.jsonl"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _eval_row(
    target: _Target, mode: EvalMode, config: EvaluatorConfig, result: dict[str, Any]
) -> dict[str, Any]:
    return {
        "event": "checkpoint_eval",
        "policy_version": int(target.policy_version),
        "checkpoint_path": target.checkpoint_path,
        "checkpoint_sha256": target.checkpoint_sha256,
        "mode": mode.name,
        "temperature": float(mode.temperature),
        "greedy": bool(mode.greedy),
        "opponent": config.opponent,
        "eval_games_per_seat": int(config.eval_games_per_seat),
        "avg_rank": result["avg_rank"],
        "rank_counts": result["rank_counts"],
        "mean_score": result["mean_score"],
        "win_rate": result.get("win_rate"),
        "deal_in_rate": result.get("deal_in_rate"),
        "riichi_rate": result.get("riichi_rate"),
        "call_rate": result.get("call_rate"),
        "num_games": result.get("num_games"),
        "crash_count": result.get("crash_count"),
        "elapsed_sec": result.get("elapsed_sec"),
        "created_at": _utc_now_iso(),
    }


# ---------------------------------------------------------------------------
# top-level run
# ---------------------------------------------------------------------------


@dataclass
class _RunState:
    rows: list[dict[str, Any]] = field(default_factory=list)
    best_updates: list[dict[str, Any]] = field(default_factory=list)
    evaluated_checkpoints: int = 0
    skipped: int = 0


def run_evaluator(config: EvaluatorConfig) -> dict[str, Any]:
    """checkpoint を評価し eval.jsonl / best.json を更新する。"""
    device = resolve_device(config.device)
    dirs = _RunDirs.from_root(config.run_root)
    dirs.ensure()
    metrics_path = _metrics_path(dirs, config)
    modes = build_modes(list(config.mode_names), low_temp=config.low_temp)

    eval_rows = _read_existing_rows(metrics_path)
    targets = discover_targets(dirs, config, eval_rows)
    state = _RunState()

    for target in targets:
        if (
            int(config.stop_after_checkpoints) > 0
            and state.evaluated_checkpoints >= int(config.stop_after_checkpoints)
        ):
            break
        model = None
        encoder = None
        evaluated_this_ckpt = False
        for mode in modes:
            if _already_evaluated(eval_rows, target.policy_version, mode.name, config):
                state.skipped += 1
                continue
            if config.dry_run:
                state.rows.append(
                    {
                        "event": "eval_dry_run",
                        "policy_version": int(target.policy_version),
                        "mode": mode.name,
                        "opponent": config.opponent,
                        "would_evaluate": True,
                    }
                )
                continue
            if model is None:
                model, encoder = _build_model_encoder(dirs, target, device)
            result = evaluate_policy(
                model, encoder, mode=mode, config=config, device=device
            )
            row = _eval_row(target, mode, config, result)
            _append_jsonl(metrics_path, row)
            eval_rows.append(row)
            state.rows.append(row)
            evaluated_this_ckpt = True
            best = _maybe_update_best(
                dirs, target=target, mode=mode, config=config, eval_result=result
            )
            if best is not None:
                state.best_updates.append(best)
        if evaluated_this_ckpt:
            state.evaluated_checkpoints += 1

    best_now = _read_best(dirs)
    return {
        "ok": True,
        "dry_run": bool(config.dry_run),
        "run_root": str(dirs.root),
        "device": device,
        "evaluated_checkpoints": state.evaluated_checkpoints,
        "evaluated_rows": [r for r in state.rows if r.get("event") == "checkpoint_eval"],
        "dry_run_rows": [r for r in state.rows if r.get("event") == "eval_dry_run"],
        "skipped": state.skipped,
        "best_updates": state.best_updates,
        "best": best_now,
        "metrics_jsonl": str(metrics_path),
    }


def _read_existing_rows(metrics_path: Path) -> list[dict[str, Any]]:
    import json

    if not metrics_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


__all__ = [
    "EvalMode",
    "EvaluatorConfig",
    "build_modes",
    "discover_targets",
    "evaluate_policy",
    "resolve_device",
    "run_evaluator",
]
