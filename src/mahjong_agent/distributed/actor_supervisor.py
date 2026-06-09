"""Phase-aware long-running actor supervisor.

CPU サーバごとに 1 コマンドで起動し、`run_config.json` と `control/state.json`
に従って以下を自動実行する常駐 actor:

- `imitation_collect`: rule-based teacher self-play shard を `imitation/ready/` に生成。
- `imitation_train`: 待機。
- `ppo`: latest checkpoint で policy rollout shard を `rollouts/ready/` に生成。
- `draining`: 新 chunk を開始せず終了。
- `stopped` / `failed`: 終了。
- `initializing`: 待機。

chunk 開始前に毎回 state を読み、phase / phase_generation の変化に追従する。
worker process は supervisor が `--workers` 個起動・監視し、異常終了は backoff 付きで
再起動する。各 worker は heartbeat を書き、backpressure（ready 過多）時は生成を止める。

seed は run config の `seed_namespace` + actor_id + worker_id + phase_generation +
chunk_index から SHA-256 で決定論的に導出し、複数 server / worker / phase で衝突
しないようにする（Python 組み込み `hash()` は process 間で不安定なので使わない）。

hidden info は扱わない（manifest は phase/version/hash/id のメタ、sample は public
observation feature のみ）。設計は `docs/distributed_actor_learner.md`。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import riichienv

from mahjong_agent.agents import (
    ModelPolicyAgent,
    ModelPolicyConfig,
    RuleBasedBaselineAgent,
)
from mahjong_agent.data import read_shard_metadata, write_decision_shard
from mahjong_agent.distributed import actor as actor_mod
from mahjong_agent.distributed.manifest import (
    RolloutShardManifest,
    read_json,
    write_json_atomic,
)
from mahjong_agent.distributed.run_state import (
    DistributedRunConfig,
    RunPhase,
    read_run_config,
    read_run_state,
    run_config_path,
    state_path,
)
from mahjong_agent.encoders import PublicObservationEncoder
from mahjong_agent.evaluation import SeatAgents, SelfPlayConfig, SelfPlayRunner

_LEAK_KEY_SUBSTRINGS = ("tehai", "hand", "wall", "win_result", "mjai")
_MAX_STEPS_PER_GAME = 4000
_SEED_SPACE = 1_000_000_000_000  # 10^12: chunk base seed の空間


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# identity / seed derivation
# ---------------------------------------------------------------------------


def worker_id(actor_id: str, worker_index: int) -> str:
    return f"{actor_id}/w{int(worker_index):03d}"


def safe_worker_id(wid: str) -> str:
    return wid.replace("/", "_").replace("\\", "_")


def derive_chunk_base_seed(
    *,
    seed_namespace: str,
    actor_id: str,
    wid: str,
    phase_generation: int,
    chunk_index: int,
) -> int:
    """決定論的に chunk の base seed を導出する（process 非依存）。

    chunk 内の各 game は ``base + game_index`` を使う。namespace / actor / worker /
    generation / chunk が 1 つでも違えば別の base になる。
    """
    key = "|".join(
        [
            str(seed_namespace),
            str(actor_id),
            str(wid),
            str(int(phase_generation)),
            str(int(chunk_index)),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_SPACE


# ---------------------------------------------------------------------------
# persistent per-worker chunk counter
# ---------------------------------------------------------------------------
#
# worker / supervisor 再起動後に chunk_index を 0 へ戻すと、同じ
# (namespace, actor_id, worker_id, phase_generation, chunk_index) から同一 seed
# が再生成され、同じ対局データが別 shard として learner に混入し得る。これを防ぐ
# ため、worker_id ごとの ``next_chunk_index`` を run-root に永続化し、生成開始前に
# index を予約（read → +1 して atomic write）する。
#
# 同一 worker_id を持つ process は同時に 1 つだけ（supervisor が index ごとに 1
# process、再起動時は旧 process は死亡済み）なので single-writer が保たれる。
# crash した chunk の index は欠番になってよい（再利用しないことを優先）。


def _counter_path(dirs: _SupDirs, wid: str) -> Path:
    return dirs.counters / f"{safe_worker_id(wid)}.json"


def read_next_chunk_index(dirs: _SupDirs, wid: str) -> int:
    """worker の次 chunk_index（未保存なら 0）を返す。"""
    path = _counter_path(dirs, wid)
    if not path.is_file():
        return 0
    try:
        return int(read_json(path).get("next_chunk_index", 0))
    except (ValueError, KeyError, TypeError, OSError):
        return 0


def reserve_chunk_index(dirs: _SupDirs, wid: str) -> int:
    """次の chunk_index を予約し、永続 counter を進めて予約値を返す。

    生成開始前に呼ぶ。crash しても index は消費済み（欠番）として扱われ、
    再起動後に同じ index（= 同じ seed）が再生成されない。
    """
    dirs.counters.mkdir(parents=True, exist_ok=True)
    idx = read_next_chunk_index(dirs, wid)
    write_json_atomic(
        _counter_path(dirs, wid),
        {
            "worker_id": wid,
            "next_chunk_index": idx + 1,
            "updated_at": _utc_now_iso(),
        },
    )
    return idx


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupervisorConfig:
    run_root: str
    actor_id: str | None = None
    workers: int = 1
    poll_interval_sec: float | None = None  # None -> run_config から
    max_chunks: int = 0  # 0 = unlimited
    max_runtime_sec: float | None = None
    max_poll_iterations: int = 0  # 0 = unlimited（idle poll の上限。test 用）
    worker_script: str | None = None  # supervisor が worker を spawn する CLI path

    def resolved_actor_id(self) -> str:
        return str(self.actor_id) if self.actor_id else socket.gethostname()


# ---------------------------------------------------------------------------
# directory layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SupDirs:
    root: Path
    checkpoints: Path
    imitation_pending: Path
    imitation_ready: Path
    rollouts_pending: Path
    rollouts_ready: Path
    heartbeats: Path
    counters: Path

    @classmethod
    def from_root(cls, run_root: str | Path) -> _SupDirs:
        root = Path(run_root)
        return cls(
            root=root,
            checkpoints=root / "checkpoints",
            imitation_pending=root / "imitation" / "pending",
            imitation_ready=root / "imitation" / "ready",
            rollouts_pending=root / "rollouts" / "pending",
            rollouts_ready=root / "rollouts" / "ready",
            heartbeats=root / "metrics" / "heartbeats",
            counters=root / "control" / "actor_counters",
        )

    def ensure(self) -> None:
        for p in (
            self.imitation_pending,
            self.imitation_ready,
            self.rollouts_pending,
            self.rollouts_ready,
            self.heartbeats,
            self.counters,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def queues_for(self, phase: RunPhase) -> tuple[Path, Path]:
        if phase == RunPhase.IMITATION_COLLECT:
            return self.imitation_pending, self.imitation_ready
        return self.rollouts_pending, self.rollouts_ready


def _count_ready(ready_root: Path) -> int:
    if not ready_root.is_dir():
        return 0
    return sum(1 for _ in ready_root.rglob("manifest.json"))


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


# ---------------------------------------------------------------------------
# heartbeat
# ---------------------------------------------------------------------------


@dataclass
class _Heartbeat:
    actor_id: str
    wid: str
    chunk_counter: int = 0
    loaded_policy_version: int | None = None
    last_publish_at: str | None = None
    samples_per_sec: float = 0.0
    games_per_sec: float = 0.0
    policy_version_state_mismatch: bool = False

    def write(self, dirs: _SupDirs, *, phase: RunPhase, phase_generation: int, status: str) -> None:
        row = {
            "actor_id": self.actor_id,
            "worker_id": self.wid,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "phase": phase.value,
            "phase_generation": int(phase_generation),
            "loaded_policy_version": self.loaded_policy_version,
            "chunk_counter": int(self.chunk_counter),
            "last_publish_at": self.last_publish_at,
            "last_update_at": _utc_now_iso(),
            "status": status,
            "samples_per_sec": round(float(self.samples_per_sec), 4),
            "games_per_sec": round(float(self.games_per_sec), 4),
            "policy_version_state_mismatch": bool(self.policy_version_state_mismatch),
        }
        path = dirs.heartbeats / f"{safe_worker_id(self.wid)}.json"
        write_json_atomic(path, row)


# ---------------------------------------------------------------------------
# shard generation / publish
# ---------------------------------------------------------------------------


def _publish_shard(
    *,
    pending_root: Path,
    ready_root: Path,
    shard_id: str,
    samples: list[Any],
    manifest_fields: dict[str, Any],
    actor_id: str,
    role: str,
    old_log_prob_available: bool,
    phase: RunPhase,
    phase_generation: int,
    wid: str,
    chunk_index: int,
    seed_start: int,
    seed_end: int,
    num_games: int,
) -> str:
    """samples を pending に書き切ってから ready へ directory atomic rename。"""
    pending_dir = pending_root / shard_id
    if pending_dir.exists():
        shutil.rmtree(pending_dir)
    pending_dir.mkdir(parents=True, exist_ok=True)
    shard_path = pending_dir / "shard.npz"
    write_decision_shard(
        shard_path,
        samples,
        metadata={
            "kind": role,
            "actor_id": actor_id,
            "phase": phase.value,
            "phase_generation": int(phase_generation),
            "worker_id": wid,
            "chunk_index": int(chunk_index),
        },
    )
    shard_meta = read_shard_metadata(shard_path)
    manifest = RolloutShardManifest(
        policy_version=int(manifest_fields["policy_version"]),
        checkpoint_sha256=str(manifest_fields["checkpoint_sha256"]),
        actor_id=str(actor_id),
        created_at=_utc_now_iso(),
        num_games=int(num_games),
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
        role=role,
        phase=phase.value,
        phase_generation=int(phase_generation),
        worker_id=wid,
        chunk_index=int(chunk_index),
    )
    write_json_atomic(pending_dir / "manifest.json", manifest.to_dict())

    ready_dir = ready_root / shard_id
    if ready_dir.exists():
        i = 1
        while (ready_root / f"{shard_id}_dup{i}").exists():
            i += 1
        ready_dir = ready_root / f"{shard_id}_dup{i}"
    os.replace(pending_dir, ready_dir)
    return str(ready_dir)


@dataclass
class _ChunkResult:
    published: bool
    ready_path: str | None
    num_games: int
    num_samples: int
    crash_count: int
    elapsed_sec: float
    policy_version: int | None = None


def _run_self_play(
    *, encoder, seat_agents, run_config: DistributedRunConfig, seeds: list[int]
):
    runner = SelfPlayRunner(
        encoder=encoder,
        config=SelfPlayConfig(
            game_type=_game_type(run_config.ppo.game_type),
            max_steps_per_game=_MAX_STEPS_PER_GAME,
            collect_samples=True,
            deterministic_wall=True,
        ),
    )
    results = runner.run_games(seat_agents, seeds=seeds)
    crashes = [r.crash_context for r in results if r.crash_context is not None]
    samples = [s for r in results for s in r.samples]
    return samples, crashes


def generate_imitation_chunk(
    *,
    dirs: _SupDirs,
    run_config: DistributedRunConfig,
    actor_id: str,
    wid: str,
    phase_generation: int,
    chunk_index: int,
) -> _ChunkResult:
    """rule-based teacher self-play で imitation shard を生成・publish する。"""
    start = time.time()
    chunk_games = int(run_config.imitation.chunk_games)
    base = derive_chunk_base_seed(
        seed_namespace=run_config.seed_namespace,
        actor_id=actor_id,
        wid=wid,
        phase_generation=phase_generation,
        chunk_index=chunk_index,
    )
    seeds = [base + i for i in range(chunk_games)]
    encoder = PublicObservationEncoder(enable_hints=True)
    teacher = RuleBasedBaselineAgent(prefer_riichi=True, seed=base)
    seat_agents = SeatAgents.homogeneous(teacher, actor_type="teacher")
    samples, crashes = _run_self_play(
        encoder=encoder, seat_agents=seat_agents, run_config=run_config, seeds=seeds
    )
    if crashes:
        return _ChunkResult(False, None, len(seeds), len(samples), len(crashes),
                            round(time.time() - start, 3))
    if not samples:
        raise RuntimeError("imitation chunk produced zero samples")
    _assert_no_hidden_metadata(samples)
    shard_id = (
        f"imit_g{phase_generation:04d}_{safe_worker_id(wid)}_"
        f"c{chunk_index:06d}_{_utc_now_iso().replace(':', '')}"
    )
    ready_path = _publish_shard(
        pending_root=dirs.imitation_pending,
        ready_root=dirs.imitation_ready,
        shard_id=shard_id,
        samples=samples,
        manifest_fields={"policy_version": -1, "checkpoint_sha256": ""},
        actor_id=actor_id,
        role="imitation_teacher",
        old_log_prob_available=False,
        phase=RunPhase.IMITATION_COLLECT,
        phase_generation=phase_generation,
        wid=wid,
        chunk_index=chunk_index,
        seed_start=seeds[0],
        seed_end=seeds[-1],
        num_games=len(seeds),
    )
    return _ChunkResult(True, ready_path, len(seeds), len(samples), 0,
                        round(time.time() - start, 3), policy_version=-1)


def generate_ppo_chunk(
    *,
    dirs: _SupDirs,
    run_config: DistributedRunConfig,
    actor_id: str,
    wid: str,
    phase_generation: int,
    chunk_index: int,
    policy,
) -> _ChunkResult:
    """latest policy で homogeneous policy rollout shard を生成・publish する。"""
    start = time.time()
    chunk_games = int(run_config.ppo.actor_chunk_games)
    base = derive_chunk_base_seed(
        seed_namespace=run_config.seed_namespace,
        actor_id=actor_id,
        wid=wid,
        phase_generation=phase_generation,
        chunk_index=chunk_index,
    )
    seeds = [base + i for i in range(chunk_games)]
    agent = ModelPolicyAgent(
        model=policy.model,
        encoder=policy.encoder,
        config=ModelPolicyConfig(
            greedy=False,
            temperature=float(run_config.ppo.temperature),
            device="cpu",
        ),
        seed=base,
    )
    seat_agents = SeatAgents.homogeneous(agent, actor_type="policy")
    samples, crashes = _run_self_play(
        encoder=policy.encoder, seat_agents=seat_agents,
        run_config=run_config, seeds=seeds,
    )
    pv = int(policy.entry.policy_version)
    if crashes:
        return _ChunkResult(False, None, len(seeds), len(samples), len(crashes),
                            round(time.time() - start, 3), policy_version=pv)
    if not samples:
        raise RuntimeError("ppo chunk produced zero samples")
    _assert_no_hidden_metadata(samples)
    shard_id = (
        f"ppo_p{pv:06d}_g{phase_generation:04d}_{safe_worker_id(wid)}_"
        f"c{chunk_index:06d}_{_utc_now_iso().replace(':', '')}"
    )
    ready_path = _publish_shard(
        pending_root=dirs.rollouts_pending,
        ready_root=dirs.rollouts_ready,
        shard_id=shard_id,
        samples=samples,
        manifest_fields={
            "policy_version": pv,
            "checkpoint_sha256": str(policy.entry.checkpoint_sha256),
        },
        actor_id=actor_id,
        role="actor_rollout",
        old_log_prob_available=(float(run_config.ppo.temperature) == 1.0),
        phase=RunPhase.PPO,
        phase_generation=phase_generation,
        wid=wid,
        chunk_index=chunk_index,
        seed_start=seeds[0],
        seed_end=seeds[-1],
        num_games=len(seeds),
    )
    return _ChunkResult(True, ready_path, len(seeds), len(samples), 0,
                        round(time.time() - start, 3), policy_version=pv)


# ---------------------------------------------------------------------------
# worker loop
# ---------------------------------------------------------------------------


_TERMINAL_PHASES = frozenset({RunPhase.STOPPED, RunPhase.FAILED})


def _load_latest_policy(dirs: _SupDirs, hb: _Heartbeat, state_policy_version):
    """latest registry を読み policy をロードする。

    state.policy_version と registry の policy_version が食い違う場合、
    **disk 上に実在する latest registry を採用**（安全側）し、heartbeat に
    mismatch flag を立てる。
    """
    actor_dirs = actor_mod._RunDirs.from_root(dirs.root)
    entry = actor_mod.resolve_registry_entry(actor_dirs, "latest")
    hb.policy_version_state_mismatch = (
        state_policy_version is not None
        and int(state_policy_version) != int(entry.policy_version)
    )
    policy = actor_mod.load_policy(actor_dirs, entry)
    hb.loaded_policy_version = int(entry.policy_version)
    return policy


def run_worker_loop(config: SupervisorConfig, worker_index: int) -> dict[str, Any]:
    """1 worker の常駐ループ。state に従って生成 / 待機 / 終了する。"""
    dirs = _SupDirs.from_root(config.run_root)
    dirs.ensure()
    run_config = read_run_config(run_config_path(config.run_root))
    actor_id = config.resolved_actor_id()
    wid = worker_id(actor_id, worker_index)
    poll = float(
        config.poll_interval_sec
        if config.poll_interval_sec is not None
        else run_config.operations.actor_poll_interval_sec
    )
    hb = _Heartbeat(actor_id=actor_id, wid=wid)

    start = time.time()
    generated = 0  # この invocation で生成した chunk 数（max_chunks 用）
    idle_polls = 0
    published: list[dict[str, Any]] = []
    cached_policy = None
    cached_key: tuple[int, int] | None = None  # (policy_version, phase_generation)
    exit_reason = "loop_end"

    while True:
        if (
            config.max_runtime_sec is not None
            and (time.time() - start) >= float(config.max_runtime_sec)
        ):
            exit_reason = "max_runtime"
            break
        state = read_run_state(state_path(config.run_root))
        phase = state.phase
        gen = int(state.phase_generation)

        if phase in _TERMINAL_PHASES:
            hb.write(dirs, phase=phase, phase_generation=gen, status=phase.value)
            exit_reason = phase.value
            break
        if phase == RunPhase.DRAINING:
            hb.write(dirs, phase=phase, phase_generation=gen, status="draining")
            exit_reason = "draining"
            break
        if phase in (RunPhase.INITIALIZING, RunPhase.IMITATION_TRAIN):
            hb.write(dirs, phase=phase, phase_generation=gen, status="waiting")
            idle_polls += 1
            if config.max_poll_iterations and idle_polls >= config.max_poll_iterations:
                exit_reason = "max_poll_iterations"
                break
            time.sleep(poll)
            continue

        # generating phases: imitation_collect / ppo
        _, ready_root = dirs.queues_for(phase)
        if _count_ready(ready_root) >= int(run_config.operations.max_ready_shards):
            hb.write(dirs, phase=phase, phase_generation=gen, status="backpressure")
            idle_polls += 1
            if config.max_poll_iterations and idle_polls >= config.max_poll_iterations:
                exit_reason = "max_poll_iterations"
                break
            time.sleep(poll)
            continue

        if config.max_chunks and generated >= int(config.max_chunks):
            exit_reason = "max_chunks"
            break

        # 永続 counter から chunk_index を予約（生成開始前）。crash しても欠番に
        # なるだけで再利用しない＝restart 後に同じ seed を再生成しない。
        chunk_index = reserve_chunk_index(dirs, wid)
        hb.write(dirs, phase=phase, phase_generation=gen, status="generating")
        if phase == RunPhase.IMITATION_COLLECT:
            res = generate_imitation_chunk(
                dirs=dirs, run_config=run_config, actor_id=actor_id, wid=wid,
                phase_generation=gen, chunk_index=chunk_index,
            )
        else:  # ppo
            key = (gen, int(state.policy_version) if state.policy_version is not None else -1)
            if cached_policy is None or cached_key != key:
                cached_policy = _load_latest_policy(dirs, hb, state.policy_version)
                cached_key = key
            res = generate_ppo_chunk(
                dirs=dirs, run_config=run_config, actor_id=actor_id, wid=wid,
                phase_generation=gen, chunk_index=chunk_index, policy=cached_policy,
            )

        generated += 1
        hb.chunk_counter = read_next_chunk_index(dirs, wid)
        idle_polls = 0
        if res.published:
            elapsed = max(res.elapsed_sec, 1e-9)
            hb.last_publish_at = _utc_now_iso()
            hb.games_per_sec = res.num_games / elapsed
            hb.samples_per_sec = res.num_samples / elapsed
            published.append(
                {
                    "phase": phase.value,
                    "phase_generation": gen,
                    "ready_path": res.ready_path,
                    "num_samples": res.num_samples,
                    "policy_version": res.policy_version,
                }
            )
            status = "generating"
        else:
            status = "crash_skipped"
        hb.write(dirs, phase=phase, phase_generation=gen, status=status)

    return {
        "ok": True,
        "actor_id": actor_id,
        "worker_id": wid,
        "exit_reason": exit_reason,
        "chunks_published": len(published),
        "published": published,
        "elapsed_sec": round(time.time() - start, 3),
    }


# ---------------------------------------------------------------------------
# supervisor (multi-worker process management)
# ---------------------------------------------------------------------------


def compute_backoff(consecutive_crashes: int, *, base: float = 1.0, cap: float = 30.0) -> float:
    """crash loop を避ける exponential backoff（秒）。"""
    if consecutive_crashes <= 0:
        return 0.0
    return float(min(cap, base * (2 ** (consecutive_crashes - 1))))


def is_terminal_state(run_root: str | Path) -> bool:
    try:
        state = read_run_state(state_path(run_root))
    except (FileNotFoundError, KeyError, ValueError):
        return False
    return state.phase in _TERMINAL_PHASES


@dataclass
class _WorkerProc:
    index: int
    proc: Any
    status: str = "running"  # running | waiting_for_restart | done
    consecutive_crashes: int = 0
    restarts: int = 0
    next_restart_at: float = 0.0


def _supervise_step(
    workers: dict[int, _WorkerProc],
    *,
    terminal: bool,
    now: float,
    spawn,
) -> bool:
    """1 監視ステップ。worker 状態を更新し、ループ継続可否を返す。

    - 正常終了 (rc==0) / terminal state での終了 → ``done``。
    - 異常終了 (rc!=0, 非 terminal) → ``waiting_for_restart`` にし backoff 時刻を
      セット。backoff 到達後に ``spawn`` で再起動して ``running`` へ。
    - **backoff 待機中の worker が居る限り supervisor は終了しない**
      （``alive==0`` だけでは止めない）。terminal state か **全 worker done** の
      ときだけ ``False``（停止）を返す。
    """
    for wp in workers.values():
        if wp.status == "done":
            continue
        rc = wp.proc.poll()
        if rc is None:
            wp.status = "running"
            continue
        # process が終了している
        if rc == 0 or terminal:
            wp.status = "done"
            continue
        # 異常終了
        if wp.status != "waiting_for_restart":
            # crash を初めて観測したタイミングで backoff を決める
            wp.consecutive_crashes += 1
            wp.next_restart_at = now + compute_backoff(wp.consecutive_crashes)
            wp.status = "waiting_for_restart"
        if now >= wp.next_restart_at:
            wp.proc = spawn(wp.index)
            wp.restarts += 1
            wp.status = "running"

    if terminal:
        return False
    if all(wp.status == "done" for wp in workers.values()):
        return False
    return True


class Supervisor:
    """`--workers` 個の worker subprocess を起動・監視し、crash を backoff 再起動する。"""

    def __init__(self, config: SupervisorConfig):
        if not config.worker_script:
            raise ValueError("Supervisor requires config.worker_script to spawn workers")
        self.config = config
        self.dirs = _SupDirs.from_root(config.run_root)

    def _spawn(self, index: int):
        import subprocess
        import sys

        cmd = [
            sys.executable,
            str(self.config.worker_script),
            "--run-root", str(self.config.run_root),
            "--workers", str(self.config.workers),
            "--_worker-index", str(index),
        ]
        if self.config.actor_id:
            cmd += ["--actor-id", str(self.config.actor_id)]
        if self.config.max_chunks:
            cmd += ["--max-chunks", str(self.config.max_chunks)]
        if self.config.max_runtime_sec is not None:
            cmd += ["--max-runtime-sec", str(self.config.max_runtime_sec)]
        if self.config.poll_interval_sec is not None:
            cmd += ["--poll-interval-sec", str(self.config.poll_interval_sec)]
        log_dir = self.dirs.root / "metrics" / "supervisor_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = (log_dir / f"worker_{index:03d}.log").open("a", encoding="utf-8")
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    def run(self) -> dict[str, Any]:
        self.dirs.ensure()
        start = time.time()
        workers = {i: _WorkerProc(index=i, proc=self._spawn(i))
                   for i in range(int(self.config.workers))}
        while True:
            if (
                self.config.max_runtime_sec is not None
                and (time.time() - start) >= float(self.config.max_runtime_sec)
            ):
                break
            terminal = is_terminal_state(self.config.run_root)
            keep_going = _supervise_step(
                workers, terminal=terminal, now=time.time(), spawn=self._spawn
            )
            if not keep_going:
                break
            time.sleep(0.2)

        # 残存 worker を終了
        for wp in workers.values():
            if wp.status != "done" and wp.proc.poll() is None:
                wp.proc.terminate()
        return {
            "ok": True,
            "run_root": str(self.dirs.root),
            "workers": int(self.config.workers),
            "total_restarts": int(sum(wp.restarts for wp in workers.values())),
            "elapsed_sec": round(time.time() - start, 3),
        }


__all__ = [
    "SupervisorConfig",
    "Supervisor",
    "compute_backoff",
    "derive_chunk_base_seed",
    "generate_imitation_chunk",
    "generate_ppo_chunk",
    "is_terminal_state",
    "read_next_chunk_index",
    "reserve_chunk_index",
    "run_worker_loop",
    "safe_worker_id",
    "worker_id",
]
