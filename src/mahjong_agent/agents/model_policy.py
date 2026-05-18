"""Model-policy agent (PPO 用 on-policy rollout actor)。

``Stage03Model`` + ``PublicObservationEncoder`` を組み合わせ、normal discard
head と candidate head の両方で action を sample する agent。``log_prob`` と
``value`` を ``AgentDecision.extras`` に乗せ、``SelfPlayRunner`` 経由で
``DecisionSample.old_log_prob`` / ``DecisionSample.value`` に保存される。

特徴
----
- hidden info を一切使わない。``encoder.encode_observation`` の出力 (public
  observation feature) を model forward に渡すだけ。
- ``encoder`` は ``PublicObservationEncoder`` を内部で持つ (rollout 時に
  agent 自身が encode する)。observation_dim / candidate_dim は model の
  config と一致する想定。
- 通常打牌は ``model.forward`` の masked discard_logits から、副露/特殊は
  ``model.score_candidates`` の masked scores から sample する。
- greedy / sampling / temperature を config 切り替え。
- legal action のみ選ぶ (= discard_mask / candidate_mask で illegal を
  ``-1e9`` に倒した後で softmax → sample)。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from mahjong_agent.actions.types import ActionFamily, LegalActionSet, ModelAction
from mahjong_agent.agents.base import AgentDecision
from mahjong_agent.encoders.public_observation import PublicObservationEncoder
from mahjong_agent.models.stage03_model import Stage03Model

_LARGE_NEG: float = -1.0e9


@dataclass(frozen=True)
class ModelPolicyConfig:
    """``ModelPolicyAgent`` の挙動設定。

    Attributes
    ----------
    greedy:
        ``True`` で argmax (deterministic)、``False`` で softmax sampling。
        PPO eligible な on-policy sample を集めるには ``False`` 固定が必須
        (argmax は確率分布と乖離するため ratio が一貫しない)。
    temperature:
        sampling 時の logit 温度 (>0)。``temperature=1.0`` で素の softmax。
        ``greedy=True`` のときは無視。**PPO eligible** に保ちたい場合は
        ``1.0`` 固定 (PPO 側 ``compute_ppo_loss`` は temperature scaling を
        適用せず raw combined logits 上で ``new_log_prob`` を再計算するため、
        ``temperature != 1.0`` で取った old_log_prob とは整合しない)。
    device:
        ``"cpu"`` / ``"cuda"`` 等。model の重みが置かれた device と揃える
        必要がある。default ``"cpu"``。

    PPO eligibility note
    --------------------
    本 agent の rollout sample が PPO ratio 計算に乗るのは
    ``greedy=False`` かつ ``temperature == 1.0`` のときに限る。それ以外の
    設定 (greedy argmax / temperature ≠ 1.0) では、``select_action`` が
    ``AgentDecision.extras["ppo_exclude"] = True`` を立て、
    ``SelfPlayRunner`` 経由で ``DecisionSample.metadata["ppo_exclude"]``
    に永続化される。``compute_returns_and_advantages`` がこの flag を見て
    eligible から外すため、不整合な ratio が学習に混入しない。
    """

    greedy: bool = False
    temperature: float = 1.0
    device: str = "cpu"


class ModelPolicyAgent:
    """``Stage03Model`` から action を sample する on-policy agent。

    Parameters
    ----------
    model:
        ``Stage03Model``。``forward`` / ``score_candidates`` / ``value`` を
        持つ multi-head model。``eval()`` mode で forward する想定だが、
        caller 側で必要に応じて train/eval を切り替えてよい。
    encoder:
        ``PublicObservationEncoder``。``riichienv.Observation`` を model の
        入力 feature に変換する。
    config:
        ``ModelPolicyConfig``。
    seed:
        ``random.Random`` の seed。``select_action(rng=...)`` で上書き可能。

    Notes
    -----
    - win action (``TSUMO`` / ``RON``) は **常に取る** (model の score を
      問わない安全側 default)。teacher / behavioral cloning とは独立。
    - ``KYUSHU_KYUHAI`` も常に取る。
    - 残りの normal_discard / その他 candidate の中から model score で
      sample する。両方が存在する場合は **両者を 1 つの選択集合に
      まとめて softmax sample** する (= legal action の union の中から
      確率付きで 1 つ選ぶ)。
    - 上記まとめ sampling のため、選択結果の ``log_prob`` は
      "両 head の logit に同じ scale 化を施したうえでの softmax" として
      計算する。具体的には、normal_discard logits (34-dim, masked) と
      candidate scores (Cmax-dim, masked) を **1 本のベクトル** に concat
      し、その上で log_softmax を取り、選んだ index の log_prob を返す。
      これにより PPO ratio が一貫した分布から評価できる。
    """

    def __init__(
        self,
        model: Stage03Model,
        encoder: PublicObservationEncoder,
        config: ModelPolicyConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self._model = model
        self._encoder = encoder
        self._config = config or ModelPolicyConfig()
        self._rng = random.Random(seed)

    @property
    def model(self) -> Stage03Model:
        return self._model

    @property
    def encoder(self) -> PublicObservationEncoder:
        return self._encoder

    @property
    def config(self) -> ModelPolicyConfig:
        return self._config

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------

    def select_action(
        self,
        legal_set: LegalActionSet,
        *,
        rng: random.Random | None = None,
        observation: Any | None = None,
    ) -> AgentDecision:
        """1 つの decision を選び、``AgentDecision`` を返す。

        ``AgentDecision.extras`` に以下を入れる:
        - ``log_prob`` (float): 選択 action の log π(a|s)。
        - ``value`` (float): value head の出力 scalar。
        - ``family`` (str): ``ActionFamily.value``。
        - ``combined_idx`` (int): combined logits 上で選ばれた index
          (= ``tile_type`` または ``34 + candidate_index``)。
        - ``ppo_exclude`` (bool, optional): 後段 PPO の ratio 計算から
          除外する必要があるとき True。次のいずれかの場合に立つ:
          (1) Win-action shortcut (TSUMO / RON / KYUSHU_KYUHAI) を
          deterministic に取った、
          (2) ``config.greedy=True`` (argmax で sampling 分布と乖離)、
          (3) ``config.temperature != 1.0`` (PPO 側は unscaled combined
          log_softmax で ``new_log_prob`` を再計算するため整合しない)。

        Win-action shortcut: TSUMO / RON / KYUSHU_KYUHAI は **model を
        forward せず常に取る** (= safe default)。この場合は ``log_prob=0``
        (= log 1; これらは optional ではなく取らない方が損な action だから
        ratio を介して学習させない) を入れる。
        """
        observation = self._require_observation(observation, legal_set)
        device = torch.device(self._config.device)
        rng = rng if rng is not None else self._rng

        cand_by_family = _group_candidates(legal_set)

        # Encode observation **once** per decision and reuse for both the
        # shortcut path (value forward) and the combined sampling path
        # (forward + score_candidates). ``observation_feat`` を
        # ``AgentDecision.extras`` に乗せ、``SelfPlayRunner._build_sample``
        # は sample 保存時に再 encode せずに済む (= double encode 解消)。
        obs_feat_np = self._encoder.encode_observation(observation).astype(
            "float32", copy=False
        )
        obs_feat_t = torch.from_numpy(obs_feat_np).to(device).float().unsqueeze(0)

        # Always-take families: TSUMO / RON / KYUSHU_KYUHAI
        # これらは deterministic に取るため、log_prob を learner に渡しても
        # ratio の意味が一貫しない。``ppo_exclude=True`` を立てて
        # ``compute_returns_and_advantages`` から PPO 対象外にする。
        for fam in (
            ActionFamily.TSUMO,
            ActionFamily.RON,
            ActionFamily.KYUSHU_KYUHAI,
        ):
            if fam in cand_by_family:
                chosen = cand_by_family[fam][0]
                # value を取りに forward する (PPO value target を埋めるため)。
                # encode は再利用、forward だけ走らせる。
                with torch.no_grad():
                    shortcut_fwd = self._model(obs_feat_t, discard_mask=None)
                value = float(shortcut_fwd.value.squeeze(0).item())
                return AgentDecision(
                    action=chosen,
                    rationale=f"win_shortcut:{fam.value}",
                    extras={
                        "log_prob": 0.0,
                        "value": float(value),
                        "family": fam.value,
                        "ppo_exclude": True,
                        "observation_feat": obs_feat_np,
                    },
                )

        # Build combined action space: discard + candidates。obs encode は
        # 既に上で済んでいるので再利用。
        d_mask = torch.from_numpy(
            self._encoder.discard_legal_mask(legal_set)
        ).to(device).float().unsqueeze(0)
        cand_feat_np = self._encoder.encode_candidates(legal_set)
        cand_count = int(cand_feat_np.shape[0])
        cand_feat = torch.from_numpy(cand_feat_np).to(device).float().unsqueeze(0)

        with torch.no_grad():
            fwd = self._model(obs_feat_t, discard_mask=d_mask)
            cand_out = self._model.score_candidates(obs_feat_t, cand_feat)
        discard_logits = fwd.discard_logits.squeeze(0)  # (34,)
        candidate_scores = (
            cand_out.candidate_scores.squeeze(0)
            if cand_count > 0
            else torch.zeros(0, device=device)
        )  # (C,)
        value_scalar = float(fwd.value.squeeze(0).item())

        # Legal-action union: discard_legal は d_mask=1.0、candidate は実在で
        # OK (encode_candidates が返した順そのまま legal)。
        discard_legal = d_mask.squeeze(0) > 0.5  # (34,)

        # Concat into one logits vector of length 34 + cand_count
        combined = torch.cat([discard_logits, candidate_scores], dim=0)
        # Apply mask: 34-dim discard 部分は discard_legal 外を large neg、
        # candidate 部分は全て legal (= mask しない)。
        combined_mask = torch.cat(
            [
                discard_legal.float(),
                torch.ones(cand_count, device=device, dtype=combined.dtype),
            ],
            dim=0,
        )
        masked_logits = combined + (1.0 - combined_mask) * _LARGE_NEG

        # legal action が 0 件: fail-fast
        if (masked_logits > _LARGE_NEG * 0.5).sum().item() == 0:
            raise ValueError(
                f"ModelPolicyAgent: no legal action for player "
                f"{legal_set.decision_player}"
            )

        # temperature
        temperature = max(float(self._config.temperature), 1e-6)
        scaled_logits = masked_logits / temperature

        log_probs = F.log_softmax(scaled_logits, dim=-1)
        if self._config.greedy:
            idx = int(torch.argmax(scaled_logits).item())
        else:
            probs = torch.exp(log_probs)
            # sampling は CPU 上で行って rng を使う (deterministic 化のため)。
            probs_cpu = probs.detach().cpu().numpy()
            # numpy で sampling. rng を seed と共に使う。
            r = rng.random()
            cum = 0.0
            idx = int(probs_cpu.shape[0]) - 1
            for i, p in enumerate(probs_cpu):
                cum += float(p)
                if r <= cum:
                    idx = i
                    break

        chosen_log_prob = float(log_probs[idx].item())

        if idx < 34:
            # normal_discard
            if not bool(discard_legal[idx].item()):
                raise RuntimeError(
                    f"ModelPolicyAgent: sampled illegal discard idx={idx} "
                    f"(discard_legal_mask doesn't match)"
                )
            tile_type = int(idx)
            chosen = legal_set.normal_discard[tile_type]
            family = ActionFamily.NORMAL_DISCARD
        else:
            cand_idx = idx - 34
            if cand_idx < 0 or cand_idx >= cand_count:
                raise RuntimeError(
                    f"ModelPolicyAgent: sampled candidate idx out of range: "
                    f"{cand_idx} / {cand_count}"
                )
            chosen = legal_set.candidates[cand_idx]
            family = chosen.family

        # PPO eligibility: greedy / temperature scaling は old_log_prob と
        # PPO 側 unscaled combined log_softmax との整合を壊すため、ratio
        # 計算から外す。``ppo_exclude=True`` を立てておけば
        # ``compute_returns_and_advantages`` が eligible から除外する。
        # ``log_prob`` は diagnostics 用に残す (= 学習信号には使わない)。
        ppo_exclude = bool(self._config.greedy) or (
            float(self._config.temperature) != 1.0
        )
        extras: dict[str, Any] = {
            "log_prob": chosen_log_prob,
            "value": value_scalar,
            "family": family.value,
            "combined_idx": int(idx),
            "observation_feat": obs_feat_np,
        }
        if ppo_exclude:
            extras["ppo_exclude"] = True
        return AgentDecision(
            action=chosen,
            rationale=f"model:{family.value}",
            extras=extras,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_observation(observation: Any, legal_set: LegalActionSet) -> Any:
        if observation is None:
            raise ValueError(
                "ModelPolicyAgent requires `observation` to forward the model "
                f"(player {legal_set.decision_player})"
            )
        return observation


def _group_candidates(
    legal_set: LegalActionSet,
) -> dict[ActionFamily, list[ModelAction]]:
    out: dict[ActionFamily, list[ModelAction]] = {}
    for c in legal_set.candidates:
        out.setdefault(c.family, []).append(c)
    return out


__all__ = ["ModelPolicyAgent", "ModelPolicyConfig"]
