"""Decision shard writer / reader.

Shard format (v2):

1 つの ``.npz`` (np.savez_compressed) ファイルに以下を格納する。

schema_version
--------------
- v1 (初版)
- v2: per-sample ``teacher_best_mask`` (34-dim float32) を追加。
  tie-aware imitation loss の soft target に使う。v1 shard は読めない
  (fail-fast)。

- shard-level metadata は JSON encode して 0-d object array
  ``"_shard_meta"`` に保存する:
  ``{"schema_version": int, "observation_dim": int, "candidate_dim": int,
    "num_samples": int, "user_metadata": dict | None}``
- 固定 shape per-sample arrays は ``(N, ...)`` 形式でそのまま numpy array
  として保存する (例: ``observation``, ``discard_mask``,
  ``selected_discard_tile_type`` 等)。
- 可変長 ``candidate_features`` は flat-pack して保存する:
  ``"candidate_features_flat": (sum_C, candidate_dim) float32``,
  ``"candidate_counts": (N,) int64``。
- string 列 (``decision_family`` / ``actor_type`` / ``episode_id``) は
  numpy ``dtype=object`` array で保存。
- per-sample free-form ``metadata`` dict は JSON 文字列化して長さ N の
  object array ``"metadata_json"`` に保存。

reader は ``_shard_meta["schema_version"]`` を verify する。
mismatch / 欠如 / 不正 JSON は ``ValueError`` で fail-fast。
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from mahjong_agent.data.types import (
    SCHEMA_VERSION,
    DecisionSample,
)
from mahjong_agent.targets.yaku import NUM_YAKU

_DISCARD_MASK_DIM = 34
_SHARD_META_KEY = "_shard_meta"


def _normalize_candidate_features(
    feats: np.ndarray | None, candidate_dim: int
) -> np.ndarray:
    """``DecisionSample.candidate_features`` を ``(C, candidate_dim)`` float32 に整える。

    ``C=0`` のときは ``(0, candidate_dim)`` を返す。dim mismatch は ValueError。
    """
    if feats is None:
        return np.zeros((0, candidate_dim), dtype=np.float32)
    arr = np.asarray(feats, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size == 0:
            return np.zeros((0, candidate_dim), dtype=np.float32)
        # 1D は (1, candidate_dim) と解釈する
        if arr.size != candidate_dim:
            raise ValueError(
                f"candidate_features 1D size {arr.size} mismatches "
                f"candidate_dim {candidate_dim}"
            )
        return arr.reshape(1, candidate_dim)
    if arr.ndim != 2:
        raise ValueError(
            f"candidate_features must be 2D (C, candidate_dim), got shape "
            f"{arr.shape}"
        )
    if arr.shape[0] > 0 and arr.shape[1] != candidate_dim:
        raise ValueError(
            f"candidate_features last dim {arr.shape[1]} mismatches "
            f"candidate_dim {candidate_dim}"
        )
    if arr.shape[0] == 0:
        # (0, ?) を統一形 (0, candidate_dim) にする
        return np.zeros((0, candidate_dim), dtype=np.float32)
    return arr


def _infer_dims(samples: list[DecisionSample]) -> tuple[int, int]:
    """sample list から observation_dim / candidate_dim を推定する。

    最初の non-empty な sample を採用する。両方が空のときは 0 を返す。
    """
    obs_dim = 0
    cand_dim = 0
    for s in samples:
        if s.observation is not None and s.observation.size > 0:
            obs_dim = int(np.asarray(s.observation).reshape(-1).shape[0])
            break
    for s in samples:
        if s.candidate_features is not None:
            arr = np.asarray(s.candidate_features)
            # ``(C, K)`` で K が確定していれば C=0 でも cand_dim を引ける。
            if arr.ndim == 2 and arr.shape[1] > 0:
                cand_dim = int(arr.shape[1])
                break
            if arr.ndim == 1 and arr.size > 0:
                cand_dim = int(arr.size)
                break
    return obs_dim, cand_dim


def write_decision_shard(
    path: Path | str,
    samples: list[DecisionSample],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """sample list を 1 つの ``.npz`` shard に書き出す。

    Parameters
    ----------
    path:
        出力ファイルパス。``.npz`` 拡張子を推奨。
    samples:
        書き出す ``DecisionSample`` の list。空 list でも可。
    metadata:
        shard-level free-form metadata (JSON serializable)。

    Returns
    -------
    Path: 実際に書き出した path。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(samples)

    # schema_version は全 sample で揃っていることを assert する。
    for i, s in enumerate(samples):
        if int(s.schema_version) != SCHEMA_VERSION:
            raise ValueError(
                f"sample {i} has schema_version={s.schema_version}; "
                f"writer expects {SCHEMA_VERSION}"
            )

    obs_dim, cand_dim = _infer_dims(samples)

    # observation を (N, obs_dim) float32 に統一
    if n == 0:
        observation = np.zeros((0, obs_dim), dtype=np.float32)
    else:
        observation = np.stack(
            [np.asarray(s.observation, dtype=np.float32).reshape(obs_dim)
             if obs_dim > 0
             else np.zeros(0, dtype=np.float32)
             for s in samples]
        ).astype(np.float32, copy=False)

    # discard_mask (N, 34) float32
    discard_mask = (
        np.stack(
            [np.asarray(s.discard_mask, dtype=np.float32).reshape(
                _DISCARD_MASK_DIM)
             for s in samples]
        ).astype(np.float32, copy=False)
        if n > 0
        else np.zeros((0, _DISCARD_MASK_DIM), dtype=np.float32)
    )

    # candidate_features を flat-pack
    cand_feats_list = [
        _normalize_candidate_features(s.candidate_features, cand_dim)
        for s in samples
    ]
    candidate_counts = np.asarray(
        [arr.shape[0] for arr in cand_feats_list], dtype=np.int64
    )
    if cand_feats_list and any(arr.shape[0] > 0 for arr in cand_feats_list):
        candidate_features_flat = np.concatenate(
            cand_feats_list, axis=0
        ).astype(np.float32, copy=False)
    else:
        candidate_features_flat = np.zeros(
            (0, cand_dim), dtype=np.float32
        )

    # scalar int / float / bool arrays
    def _int_arr(getter):
        return np.asarray([int(getter(s)) for s in samples], dtype=np.int64)

    def _float_arr(getter):
        return np.asarray(
            [float(getter(s)) for s in samples], dtype=np.float32
        )

    def _bool_arr(getter):
        return np.asarray([bool(getter(s)) for s in samples], dtype=np.bool_)

    round_id = _int_arr(lambda s: s.round_id)
    step_id = _int_arr(lambda s: s.step_id)
    player_id = _int_arr(lambda s: s.player_id)
    selected_discard_tile_type = _int_arr(
        lambda s: s.selected_discard_tile_type
    )
    selected_candidate_index = _int_arr(lambda s: s.selected_candidate_index)
    old_log_prob = _float_arr(lambda s: s.old_log_prob)
    value = _float_arr(lambda s: s.value)
    reward = _float_arr(lambda s: s.reward)
    terminated = _bool_arr(lambda s: s.terminated)
    round_over = _bool_arr(lambda s: s.round_over)
    terminal_class = _int_arr(lambda s: s.terminal_class)
    yaku_target = (
        np.stack([
            np.asarray(s.yaku_target, dtype=np.float32).reshape(NUM_YAKU)
            for s in samples
        ]).astype(np.float32, copy=False)
        if n > 0
        else np.zeros((0, NUM_YAKU), dtype=np.float32)
    )
    yaku_loss_mask = _float_arr(lambda s: s.yaku_loss_mask)
    han = _int_arr(lambda s: s.han)
    fu = _int_arr(lambda s: s.fu)
    score_delta = _int_arr(lambda s: s.score_delta)
    teacher_discard_tile_type = _int_arr(
        lambda s: s.teacher_discard_tile_type
    )
    teacher_candidate_index = _int_arr(lambda s: s.teacher_candidate_index)
    teacher_available = _bool_arr(lambda s: s.teacher_available)
    teacher_best_mask = (
        np.stack([
            np.asarray(s.teacher_best_mask, dtype=np.float32).reshape(
                _DISCARD_MASK_DIM
            )
            for s in samples
        ]).astype(np.float32, copy=False)
        if n > 0
        else np.zeros((0, _DISCARD_MASK_DIM), dtype=np.float32)
    )

    # string columns -> dtype=object arrays
    decision_family = np.array(
        [str(s.decision_family) for s in samples], dtype=object
    )
    actor_type = np.array(
        [str(s.actor_type) for s in samples], dtype=object
    )
    episode_id = np.array(
        [str(s.episode_id) for s in samples], dtype=object
    )

    # free-form metadata -> per-sample JSON string array
    metadata_json = np.array(
        [json.dumps(s.metadata or {}, ensure_ascii=False) for s in samples],
        dtype=object,
    )

    # shard-level metadata (JSON-encoded 0-d object array)
    shard_meta = {
        "schema_version": SCHEMA_VERSION,
        "observation_dim": int(obs_dim),
        "candidate_dim": int(cand_dim),
        "num_samples": int(n),
        "user_metadata": metadata or {},
    }
    shard_meta_arr = np.array(
        json.dumps(shard_meta, ensure_ascii=False), dtype=object
    )

    # ファイルに書き出し
    payload = {
        _SHARD_META_KEY: shard_meta_arr,
        "observation": observation,
        "discard_mask": discard_mask,
        "candidate_features_flat": candidate_features_flat,
        "candidate_counts": candidate_counts,
        "round_id": round_id,
        "step_id": step_id,
        "player_id": player_id,
        "selected_discard_tile_type": selected_discard_tile_type,
        "selected_candidate_index": selected_candidate_index,
        "old_log_prob": old_log_prob,
        "value": value,
        "reward": reward,
        "terminated": terminated,
        "round_over": round_over,
        "terminal_class": terminal_class,
        "yaku_target": yaku_target,
        "yaku_loss_mask": yaku_loss_mask,
        "han": han,
        "fu": fu,
        "score_delta": score_delta,
        "teacher_discard_tile_type": teacher_discard_tile_type,
        "teacher_candidate_index": teacher_candidate_index,
        "teacher_best_mask": teacher_best_mask,
        "teacher_available": teacher_available,
        "decision_family": decision_family,
        "actor_type": actor_type,
        "episode_id": episode_id,
        "metadata_json": metadata_json,
    }
    # np.savez_compressed は object dtype を pickle で保存するため、
    # allow_pickle が必要なので reader 側で明示する。
    # In-memory buffer を経由してファイルに書く (atomic 性を意図して)。
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    path.write_bytes(buf.getvalue())
    return path


def _parse_shard_meta(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if _SHARD_META_KEY not in npz.files:
        raise ValueError(
            "shard meta missing: expected '_shard_meta' entry in shard file"
        )
    raw = npz[_SHARD_META_KEY]
    # 0-d object array. .item() で str を取り出す。
    try:
        meta_str = raw.item() if raw.shape == () else raw[0]
    except (AttributeError, IndexError) as e:
        raise ValueError(f"shard meta entry malformed: {e}") from e
    try:
        meta = json.loads(meta_str)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"shard meta is not valid JSON: {e}") from e
    if "schema_version" not in meta:
        raise ValueError("shard meta missing 'schema_version'")
    if int(meta["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"shard schema_version={meta['schema_version']} does not match "
            f"reader SCHEMA_VERSION={SCHEMA_VERSION}"
        )
    return meta


def read_decision_shard(path: Path | str) -> list[DecisionSample]:
    """``write_decision_shard`` で書き出した shard を ``DecisionSample`` list に復元する。

    schema_version mismatch / shard meta 欠如 / 不正 JSON は
    ``ValueError`` で fail-fast。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"shard file not found: {path}")
    with np.load(path, allow_pickle=True) as npz:
        meta = _parse_shard_meta(npz)
        n = int(meta["num_samples"])
        cand_dim = int(meta["candidate_dim"])

        observation = npz["observation"]
        discard_mask = npz["discard_mask"]
        candidate_features_flat = npz["candidate_features_flat"]
        candidate_counts = npz["candidate_counts"]
        round_id = npz["round_id"]
        step_id = npz["step_id"]
        player_id = npz["player_id"]
        selected_discard_tile_type = npz["selected_discard_tile_type"]
        selected_candidate_index = npz["selected_candidate_index"]
        old_log_prob = npz["old_log_prob"]
        value = npz["value"]
        reward = npz["reward"]
        terminated = npz["terminated"]
        round_over = npz["round_over"]
        terminal_class = npz["terminal_class"]
        yaku_target = npz["yaku_target"]
        yaku_loss_mask = npz["yaku_loss_mask"]
        han = npz["han"]
        fu = npz["fu"]
        score_delta = npz["score_delta"]
        teacher_discard_tile_type = npz["teacher_discard_tile_type"]
        teacher_candidate_index = npz["teacher_candidate_index"]
        teacher_best_mask = npz["teacher_best_mask"]
        teacher_available = npz["teacher_available"]
        decision_family = npz["decision_family"]
        actor_type = npz["actor_type"]
        episode_id = npz["episode_id"]
        metadata_json = npz["metadata_json"]

    samples: list[DecisionSample] = []
    cursor = 0
    for i in range(n):
        c = int(candidate_counts[i])
        cand_feats = candidate_features_flat[cursor:cursor + c].astype(
            np.float32, copy=False
        )
        if cand_feats.shape[0] == 0:
            cand_feats = np.zeros((0, cand_dim), dtype=np.float32)
        cursor += c
        meta_dict = json.loads(str(metadata_json[i])) if metadata_json[i] else {}
        samples.append(
            DecisionSample(
                schema_version=SCHEMA_VERSION,
                episode_id=str(episode_id[i]),
                round_id=int(round_id[i]),
                step_id=int(step_id[i]),
                player_id=int(player_id[i]),
                decision_family=str(decision_family[i]),
                actor_type=str(actor_type[i]),
                observation=np.asarray(observation[i], dtype=np.float32),
                discard_mask=np.asarray(discard_mask[i], dtype=np.float32),
                candidate_features=cand_feats,
                selected_discard_tile_type=int(selected_discard_tile_type[i]),
                selected_candidate_index=int(selected_candidate_index[i]),
                old_log_prob=float(old_log_prob[i]),
                value=float(value[i]),
                reward=float(reward[i]),
                terminated=bool(terminated[i]),
                round_over=bool(round_over[i]),
                terminal_class=int(terminal_class[i]),
                yaku_target=np.asarray(yaku_target[i], dtype=np.float32),
                yaku_loss_mask=float(yaku_loss_mask[i]),
                han=int(han[i]),
                fu=int(fu[i]),
                score_delta=int(score_delta[i]),
                teacher_discard_tile_type=int(teacher_discard_tile_type[i]),
                teacher_candidate_index=int(teacher_candidate_index[i]),
                teacher_best_mask=np.asarray(
                    teacher_best_mask[i], dtype=np.float32
                ),
                teacher_available=bool(teacher_available[i]),
                metadata=meta_dict,
            )
        )
    return samples


def read_shard_metadata(path: Path | str) -> dict[str, Any]:
    """shard の shard-level metadata だけを読む (samples は読まない)。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"shard file not found: {path}")
    with np.load(path, allow_pickle=True) as npz:
        return _parse_shard_meta(npz)


__all__ = [
    "write_decision_shard",
    "read_decision_shard",
    "read_shard_metadata",
]
