"""Metadata for the public observation encoder."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EncoderMetadata:
    """Shape / layout metadata for ``PublicObservationEncoder`` v1.

    Attributes
    ----------
    observation_dim:
        Fixed-size observation feature dim (= ``encode_observation()``
        output length).
    candidate_dim:
        Per-candidate feature dim (= 2nd axis of ``encode_candidates()``).
    discard_mask_dim:
        Discard legal mask dim. Always 34 in 4-player mode.
    feature_ranges:
        ``{name: (start, end)}`` slice ranges into the observation feature
        vector. Half-open intervals.
    candidate_feature_ranges:
        Same as above, but for the candidate feature vector.
    """

    observation_dim: int
    candidate_dim: int
    discard_mask_dim: int = 34
    feature_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    candidate_feature_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)


__all__ = ["EncoderMetadata"]
