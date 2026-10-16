"""Typed options for the ranking label model (string values stay accepted everywhere)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum


class TrainMethod(str, Enum):
    """How per-LF expected distances are estimated (``*_opt``: theta by exact Mallows moment inversion,
    otherwise ``theta = 1 / E[d]``).

    ``triplet``/``median_triplet``/``lstsq`` use the additive identity ``E d_ab = e_a + e_b``, which
    underestimates the distance of near-random LFs (Kendall distances saturate). ``sign_triplet`` uses the
    per-pair identity ``E[z_a z_b] = s_a s_b`` (exact under conditional independence) and is unaffected by
    masking; see :func:`uws.label_model.ranking.sign_expected_distances`."""

    TRIPLET = "triplet"
    TRIPLET_OPT = "triplet_opt"
    MEDIAN_TRIPLET = "median_triplet"
    MEDIAN_TRIPLET_OPT = "median_triplet_opt"
    LSTSQ = "lstsq"
    LSTSQ_OPT = "lstsq_opt"
    SIGN_TRIPLET = "sign_triplet"
    SIGN_TRIPLET_OPT = "sign_triplet_opt"

    @property
    def estimator(self) -> str:
        return self.value[: -len("_opt")] if self.value.endswith("_opt") else self.value

    @property
    def exact_theta(self) -> bool:
        return self.value.endswith("_opt")


class InferenceRule(str, Enum):
    KEMENY = "kemeny"
    WEIGHTED_KEMENY = "weighted_kemeny"
    KEMENY_MEDOID = "kemeny_medoid"
    WEIGHTED_KEMENY_MEDOID = "weighted_kemeny_medoid"
    PAIRWISE_MAJORITY = "pairwise_majority"
    WEIGHTED_PAIRWISE_MAJORITY = "weighted_pairwise_majority"
    POSITION_ESTIMATION = "position_estimation"
    WEIGHTED_POSITION_ESTIMATION = "weighted_position_estimation"

    @property
    def weighted(self) -> bool:
        return self.value.startswith("weighted_")


@dataclass
class RankingConfig:
    train_method: TrainMethod = TrainMethod.MEDIAN_TRIPLET_OPT
    inference_rule: InferenceRule = InferenceRule.WEIGHTED_KEMENY

    def __post_init__(self):
        self.train_method = TrainMethod(self.train_method)
        self.inference_rule = InferenceRule(self.inference_rule)

    @classmethod
    def from_dict(cls, conf: dict) -> RankingConfig:
        """Build from a (YAML) dict, ignoring unrelated keys."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in conf.items() if k in names})
