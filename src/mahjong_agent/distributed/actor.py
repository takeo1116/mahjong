"""File-based distributed actor worker (CPU self-play shard producer).

actor は学習 loop の中で **rollout shard を作り続ける CPU worker** で、GPU
learner からは独立して動く。1 回の起動で:

1. ``checkpoints/latest.json`` (または pin した policy version) を
   ``CheckpointRegistryEntry`` として読む。
2. ``checkpoint_sha256`` で checkpoint の整合性を検証し、Stage03 model を復元
   する。registry の ``observation_dim`` / ``candidate_dim`` が model / encoder
   と一致することも確認する (mismatch は fail-fast)。
3. ``SelfPlayRunner`` + ``ModelPolicyAgent`` (greedy=False, temperature=1.0 で
   PPO eligible) で homogeneous policy self-play を回し ``DecisionSample`` を作る。
4. ``rollouts/pending/<shard_id>/`` に ``shard.npz`` + ``manifest.json`` を書き
   切ってから、ディレクトリごと ``rollouts/ready/<shard_id>/`` へ atomic rename
   (``os.replace``) する。learner は ready だけを見るので partial write を拾わない。
5. ``metrics/actors/<actor_id>.jsonl`` に shard ごとの 1 行を追記する。

hidden info は持たない: sample に載るのは encoder の public observation feature
のみで、manifest は dim / version / hash / id のメタだけを持つ。

learner 側 (shard 消費 / PPO update / checkpoint publish) はこの module の責務
ではない。全体設計は ``docs/distributed_actor_learner.md`` を参照。
"""
from __future__ import annotations

import os
import shutil
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import riichienv
import torch

from mahjong_agent.agents import ModelPolicyAgent, ModelPolicyConfig
from mahjong_agent.data import (
    read_shard_metadata,
    write_decision_shard,
)
from mahjong_agent.data.types import SCHEMA_VERSION
from mahjong_agent.distributed.manifest import (
    CheckpointRegistryEntry,
    RolloutShardManifest,
    read_json,
    sha256_file,
    write_json_atomic,
)
from mahjong_agent.encoders import PublicObservationEncoder
from mahjong_agent.evaluation import SeatAgents, SelfPlayConfig, SelfPlayRunner
from mahjong_agent.models import Stage03Model, Stage03ModelConfig

# sample.metadata の key に現れたら hidden-info leak とみなす部分文字列。
# 公開 metadata key (rationale / teacher_* / call_score / ppo_exclude /
# is_post_riichi_discard) はこれらを含まない。
_LEAK_KEY_SUBSTRINGS = ("tehai", "hand", "wall", "win_result", "mjai")


def _utc_now_iso() -> str:
    """現在時刻を ``YYYY-MM-DDTHH:MM:SSZ`` (UTC) で返す。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActorConfig:
    """1 回の actor invocation の設定。

    Attributes
    ----------
    run_root:
        distributed run root。``checkpoints/latest.json`` と ``rollouts/`` を含む。
    actor_id:
        manifest / metrics に入れる actor id。
    num_games:
        この invocation で生成する total games。
    chunk_games:
        1 shard あたりの games。
    seed_base:
        episode seed の起点 (chunk は seed_base から連番で割り当てる)。
    policy_version:
        ``"latest"`` で ``latest.json`` を読む。``int`` で
        ``checkpoints/policy_NNNNNN.pt`` を pin する。
    game_type_name:
        ``riichienv.GameType`` の名前 (default ``YON_TONPUSEN``)。
    temperature:
        sampling 温度。PPO eligible に保つため default ``1.0``。
    max_steps_per_game / max_runtime_sec / metrics_jsonl / dry_run:
        補助オプション。
    """

    run_root: str
    actor_id: str
    num_games: int = 100
    chunk_games: int = 25
    seed_base: int = 1_000_000
    policy_version: str = "latest"
    game_type_name: str = "YON_TONPUSEN"
    temperature: float = 1.0
    max_steps_per_game: int = 4000
    max_runtime_sec: float | None = None
    metrics_jsonl: str | None = None
    dry_run: bool = False
    torch_threads: int = 1
    seed: int = 42


@dataclass
class _LoadedPolicy:
    """checkpoint から復元した model + encoder + registry entry。"""

    entry: CheckpointRegistryEntry
    model: Stage03Model
    encoder: PublicObservationEncoder
    checkpoint_path: Path


# ---------------------------------------------------------------------------
# directory layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RunDirs:
    root: Path
    checkpoints: Path
    pending: Path
    ready: Path
    metrics_actors: Path

    @classmethod
    def from_root(cls, run_root: str | Path) -> _RunDirs:
        root = Path(run_root)
        return cls(
            root=root,
            checkpoints=root / "checkpoints",
            pending=root / "rollouts" / "pending",
            ready=root / "rollouts" / "ready",
            metrics_actors=root / "metrics" / "actors",
        )

    def ensure_actor_dirs(self) -> None:
        """actor が書く directory を作る (checkpoint は learner の責務なので作らない)。"""
        self.pending.mkdir(parents=True, exist_ok=True)
        self.ready.mkdir(parents=True, exist_ok=True)
        self.metrics_actors.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# checkpoint registry resolution / load
# ---------------------------------------------------------------------------


def resolve_registry_entry(
    dirs: _RunDirs, policy_version: str
) -> CheckpointRegistryEntry:
    """``latest`` または pin した version から ``CheckpointRegistryEntry`` を得る。

    - ``"latest"``: ``checkpoints/latest.json`` を読む (learner が publish した
      registry をそのまま使う)。
    - ``int``: ``latest.json`` の ``policy_version`` と一致すればそれを使い、
      そうでなければ ``checkpoints/policy_NNNNNN.pt`` を直接読んで entry を
      合成する (sha256 / dim は checkpoint から導出)。
    """
    if str(policy_version) == "latest":
        latest = dirs.checkpoints / "latest.json"
        if not latest.is_file():
            raise FileNotFoundError(
                f"checkpoint registry not found: {latest} "
                "(learner がまだ checkpoint を publish していない可能性)"
            )
        return CheckpointRegistryEntry.from_dict(read_json(latest))

    try:
        version = int(policy_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--policy-version must be 'latest' or int, got {policy_version!r}"
        ) from exc

    latest = dirs.checkpoints / "latest.json"
    if latest.is_file():
        entry = CheckpointRegistryEntry.from_dict(read_json(latest))
        if int(entry.policy_version) == version:
            return entry

    # registry に無い version を pin: checkpoint file から entry を合成する。
    ckpt = dirs.checkpoints / f"policy_{version:06d}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"pinned checkpoint not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu")
    if "model_config" not in payload or "model_state_dict" not in payload:
        raise ValueError(f"invalid checkpoint payload: {ckpt}")
    model_config = dict(payload["model_config"])
    return CheckpointRegistryEntry(
        policy_version=version,
        checkpoint_path=os.path.relpath(ckpt, dirs.root),
        checkpoint_sha256=sha256_file(ckpt),
        created_at=_utc_now_iso(),
        schema_version=SCHEMA_VERSION,
        observation_dim=int(model_config["observation_dim"]),
        candidate_dim=int(model_config["candidate_dim"]),
        model_config=model_config,
        encoder_metadata=payload.get("encoder_metadata"),
    )


def load_policy(dirs: _RunDirs, entry: CheckpointRegistryEntry) -> _LoadedPolicy:
    """registry entry から checkpoint を検証・ロードして model + encoder を作る。

    - ``checkpoint_path`` は run_root 相対として解決する。
    - ``checkpoint_sha256`` を ``sha256_file`` で検証 (mismatch は fail-fast)。
    - model_config の dim と registry の dim、encoder の dim が一致することを確認。
    """
    ckpt = (dirs.root / entry.checkpoint_path).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {ckpt}")
    actual_sha = sha256_file(ckpt)
    if actual_sha != entry.checkpoint_sha256:
        raise ValueError(
            f"checkpoint sha256 mismatch for {ckpt}: "
            f"registry={entry.checkpoint_sha256} actual={actual_sha}"
        )
    payload = torch.load(ckpt, map_location="cpu")
    if "model_config" not in payload or "model_state_dict" not in payload:
        raise ValueError(f"invalid checkpoint payload: {ckpt}")
    model = Stage03Model(Stage03ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    # encoder の hint 設定は registry の encoder_metadata から引く (default on)。
    enable_hints = True
    if isinstance(entry.encoder_metadata, dict):
        enable_hints = bool(entry.encoder_metadata.get("enable_hints", True))
    encoder = PublicObservationEncoder(enable_hints=enable_hints)

    meta = encoder.metadata()
    mcfg = model.config
    # registry / model / encoder の三者で dim 一致を確認する。
    if int(mcfg.observation_dim) != int(entry.observation_dim):
        raise ValueError(
            f"observation_dim mismatch: model={mcfg.observation_dim} "
            f"registry={entry.observation_dim}"
        )
    if int(mcfg.candidate_dim) != int(entry.candidate_dim):
        raise ValueError(
            f"candidate_dim mismatch: model={mcfg.candidate_dim} "
            f"registry={entry.candidate_dim}"
        )
    if int(meta.observation_dim) != int(entry.observation_dim):
        raise ValueError(
            f"encoder observation_dim {meta.observation_dim} != registry "
            f"{entry.observation_dim} (enable_hints={enable_hints} 不一致の可能性)"
        )
    if int(meta.candidate_dim) != int(entry.candidate_dim):
        raise ValueError(
            f"encoder candidate_dim {meta.candidate_dim} != registry "
            f"{entry.candidate_dim}"
        )
    return _LoadedPolicy(
        entry=entry, model=model, encoder=encoder, checkpoint_path=ckpt
    )


# ---------------------------------------------------------------------------
# shard generation / publish
# ---------------------------------------------------------------------------


def _game_type(name: str) -> Any:
    try:
        return getattr(riichienv.GameType, str(name))
    except AttributeError as exc:
        raise ValueError(f"unknown game_type: {name!r}") from exc


def _assert_no_hidden_metadata(samples: list[Any]) -> None:
    for i, s in enumerate(samples):
        for key in s.metadata.keys():
            low = str(key).lower()
            if any(sub in low for sub in _LEAK_KEY_SUBSTRINGS):
                raise RuntimeError(
                    f"sample {i} metadata key {key!r} looks like hidden-info leak"
                )


def _chunk_seed_ranges(
    *, seed_base: int, num_games: int, chunk_games: int
) -> list[tuple[int, int]]:
    """``[(seed_start, seed_end_inclusive), ...]`` を返す (seed は連番)。"""
    chunk_games = max(1, int(chunk_games))
    out: list[tuple[int, int]] = []
    start = int(seed_base)
    remaining = int(num_games)
    while remaining > 0:
        n = min(chunk_games, remaining)
        out.append((start, start + n - 1))
        start += n
        remaining -= n
    return out


@dataclass
class ShardPublishResult:
    """1 chunk の結果 (publish 成功 / crash skip)。"""

    published: bool
    shard_id: str | None
    ready_path: str | None
    num_games: int
    num_samples: int
    seed_start: int
    seed_end: int
    elapsed_sec: float
    crash_count: int
    crashes: list[dict[str, Any]] = field(default_factory=list)


def _unique_ready_dir(ready_root: Path, shard_id: str) -> tuple[str, Path]:
    """``ready/<shard_id>/`` が衝突したら suffix を足して回避する。"""
    candidate = ready_root / shard_id
    if not candidate.exists():
        return shard_id, candidate
    i = 1
    while True:
        alt_id = f"{shard_id}_dup{i}"
        alt = ready_root / alt_id
        if not alt.exists():
            return alt_id, alt
        i += 1


def generate_and_publish_chunk(
    *,
    policy: _LoadedPolicy,
    dirs: _RunDirs,
    config: ActorConfig,
    seed_start: int,
    seed_end: int,
    chunk_index: int,
) -> ShardPublishResult:
    """1 chunk 分の self-play を回し、crash が無ければ ready へ atomic publish する。

    crash があった chunk は publish せず (= partial / 壊れた shard を learner に
    渡さない)、``ShardPublishResult.published=False`` を返す。
    """
    start = time.time()
    seeds = list(range(int(seed_start), int(seed_end) + 1))
    runner = SelfPlayRunner(
        encoder=policy.encoder,
        config=SelfPlayConfig(
            game_type=_game_type(config.game_type_name),
            max_steps_per_game=int(config.max_steps_per_game),
            collect_samples=True,
            deterministic_wall=True,
        ),
    )
    agent = ModelPolicyAgent(
        model=policy.model,
        encoder=policy.encoder,
        config=ModelPolicyConfig(
            greedy=False,
            temperature=float(config.temperature),
            device="cpu",
        ),
        seed=int(config.seed) + 5000 + int(chunk_index),
    )
    results = runner.run_games(
        SeatAgents.homogeneous(agent, actor_type="policy"),
        seeds=seeds,
    )
    crashes = [r.crash_context for r in results if r.crash_context is not None]
    samples = [s for r in results for s in r.samples]
    if crashes:
        # crash があれば publish しない (毒入り shard を learner に渡さない)。
        return ShardPublishResult(
            published=False,
            shard_id=None,
            ready_path=None,
            num_games=len(seeds),
            num_samples=len(samples),
            seed_start=int(seed_start),
            seed_end=int(seed_end),
            elapsed_sec=round(time.time() - start, 3),
            crash_count=len(crashes),
            crashes=[c for c in crashes[:3]],
        )
    if not samples:
        raise RuntimeError(
            f"chunk seeds {seed_start}-{seed_end} produced zero samples"
        )
    _assert_no_hidden_metadata(samples)

    # PPO eligible に取れているか (temperature==1.0) を manifest に記録する。
    old_log_prob_available = float(config.temperature) == 1.0

    shard_id = (
        f"policy_{policy.entry.policy_version:06d}_{config.actor_id}_"
        f"seed_{seed_start}_{seed_end}_{_utc_now_iso().replace(':', '')}_"
        f"c{chunk_index:04d}"
    )
    pending_dir = dirs.pending / shard_id
    if pending_dir.exists():
        shutil.rmtree(pending_dir)
    pending_dir.mkdir(parents=True, exist_ok=True)

    shard_path = pending_dir / "shard.npz"
    write_decision_shard(
        shard_path,
        samples,
        metadata={
            "kind": "actor_rollout",
            "actor_id": config.actor_id,
            "policy_version": int(policy.entry.policy_version),
            "seed_start": int(seed_start),
            "seed_end": int(seed_end),
        },
    )
    # manifest の dim / num_samples は **実 shard の meta** から取り、確実に
    # shard と一致させる。
    shard_meta = read_shard_metadata(shard_path)

    manifest = RolloutShardManifest(
        policy_version=int(policy.entry.policy_version),
        checkpoint_sha256=str(policy.entry.checkpoint_sha256),
        actor_id=str(config.actor_id),
        created_at=_utc_now_iso(),
        num_games=len(seeds),
        num_samples=int(shard_meta["num_samples"]),
        schema_version=int(shard_meta["schema_version"]),
        observation_dim=int(shard_meta["observation_dim"]),
        candidate_dim=int(shard_meta["candidate_dim"]),
        shard_path="shard.npz",
        seed_start=int(seed_start),
        seed_end=int(seed_end),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        old_log_prob_available=bool(old_log_prob_available),
    )
    write_json_atomic(pending_dir / "manifest.json", manifest.to_dict())

    # directory 単位 atomic publish: pending/<id>/ -> ready/<id>/。
    ready_id, ready_dir = _unique_ready_dir(dirs.ready, shard_id)
    os.replace(pending_dir, ready_dir)

    return ShardPublishResult(
        published=True,
        shard_id=ready_id,
        ready_path=str(ready_dir),
        num_games=len(seeds),
        num_samples=int(shard_meta["num_samples"]),
        seed_start=int(seed_start),
        seed_end=int(seed_end),
        elapsed_sec=round(time.time() - start, 3),
        crash_count=0,
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _metrics_path(dirs: _RunDirs, config: ActorConfig) -> Path:
    if config.metrics_jsonl:
        return Path(config.metrics_jsonl)
    return dirs.metrics_actors / f"{config.actor_id}.jsonl"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _shard_metrics_row(
    *, config: ActorConfig, policy: _LoadedPolicy, res: ShardPublishResult
) -> dict[str, Any]:
    elapsed = max(res.elapsed_sec, 1e-9)
    event = "shard_published" if res.published else "shard_skipped_crash"
    return {
        "event": event,
        "actor_id": config.actor_id,
        "policy_version": int(policy.entry.policy_version),
        "shard_id": res.shard_id,
        "num_games": res.num_games,
        "num_samples": res.num_samples,
        "elapsed_sec": res.elapsed_sec,
        "games_per_sec": round(res.num_games / elapsed, 4),
        "samples_per_sec": round(res.num_samples / elapsed, 4),
        "seed_start": res.seed_start,
        "seed_end": res.seed_end,
        "ready_path": res.ready_path,
        "crash_count": res.crash_count,
        "created_at": _utc_now_iso(),
    }


# ---------------------------------------------------------------------------
# top-level run
# ---------------------------------------------------------------------------


def run_actor(config: ActorConfig) -> dict[str, Any]:
    """1 回の actor invocation を実行し summary dict を返す。

    ``dry_run=True`` のときは checkpoint load と directory validation だけ行い、
    self-play / shard publish はしない。
    """
    torch.set_num_threads(max(1, int(config.torch_threads)))
    dirs = _RunDirs.from_root(config.run_root)
    dirs.ensure_actor_dirs()

    entry = resolve_registry_entry(dirs, config.policy_version)
    policy = load_policy(dirs, entry)

    if config.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "actor_id": config.actor_id,
            "policy_version": int(entry.policy_version),
            "checkpoint": str(policy.checkpoint_path),
            "observation_dim": int(entry.observation_dim),
            "candidate_dim": int(entry.candidate_dim),
            "run_root": str(dirs.root),
        }

    metrics_path = _metrics_path(dirs, config)
    ranges = _chunk_seed_ranges(
        seed_base=config.seed_base,
        num_games=config.num_games,
        chunk_games=config.chunk_games,
    )
    start = time.time()
    published: list[dict[str, Any]] = []
    published_count = 0
    crash_chunks = 0
    total_samples = 0
    for chunk_index, (seed_start, seed_end) in enumerate(ranges):
        if (
            config.max_runtime_sec is not None
            and (time.time() - start) >= float(config.max_runtime_sec)
        ):
            break
        res = generate_and_publish_chunk(
            policy=policy,
            dirs=dirs,
            config=config,
            seed_start=seed_start,
            seed_end=seed_end,
            chunk_index=chunk_index,
        )
        row = _shard_metrics_row(config=config, policy=policy, res=res)
        _append_jsonl(metrics_path, row)
        published.append(row)
        if res.published:
            published_count += 1
            total_samples += res.num_samples
        else:
            crash_chunks += 1
    return {
        "ok": True,
        "dry_run": False,
        "actor_id": config.actor_id,
        "policy_version": int(entry.policy_version),
        "checkpoint": str(policy.checkpoint_path),
        "run_root": str(dirs.root),
        "num_chunks": len(ranges),
        "published_shards": published_count,
        "crash_chunks": crash_chunks,
        "total_samples": total_samples,
        "elapsed_sec": round(time.time() - start, 3),
        "metrics_jsonl": str(metrics_path),
        "shards": published,
    }


__all__ = [
    "ActorConfig",
    "ShardPublishResult",
    "generate_and_publish_chunk",
    "load_policy",
    "resolve_registry_entry",
    "run_actor",
]
