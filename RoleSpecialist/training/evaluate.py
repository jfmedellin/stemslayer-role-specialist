"""Scoring a checkpoint with the measurements admission will use on it.

Training loss says how close the waveforms are. It says nothing about whether
a result is publishable: whether a silent lead lane still reconstructs, whether
the roles leak into each other, whether a lane that is not absent is audible
rather than a noise-floor placeholder.

Those are the questions the admission contract asks, and they are answered here
with the vendored functions rather than with anything defined locally. A second
definition would let a checkpoint look ready here and be refused there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from RoleSpecialist.vendor.role_metrics import (
    RoleThresholds,
    cross_role_energy_ratio_db,
    is_absent,
    is_audible,
    signal_to_residual_db,
)


@dataclass(frozen=True)
class ExampleScore:
    """What one predicted decomposition is worth against the gates."""

    reconstruction_db: float
    lead_leakage_db: float
    rhythm_leakage_db: float
    lead_absent_expected: bool
    lead_absent_predicted: bool
    lead_audible: bool
    rhythm_audible: bool

    def passes(self, thresholds: RoleThresholds) -> bool:
        """Report whether this result would be publishable.

        Absence is decided by reconstruction, never by lane energy alone. A
        predicted silent lead whose pair still reconstructs the family is a
        correct result; the same silence with the energy lost is a failure.
        """
        if self.reconstruction_db < thresholds.reconstruction_minimum_db:
            return False
        if max(self.lead_leakage_db, self.rhythm_leakage_db) > thresholds.leakage_maximum_db:
            return False
        if not self.rhythm_audible:
            return False
        if self.lead_absent_expected:
            return self.lead_absent_predicted
        return self.lead_audible and not self.lead_absent_predicted


def score_example(
    mixture: np.ndarray,
    predicted_lead: np.ndarray,
    predicted_rhythm: np.ndarray,
    thresholds: RoleThresholds,
    *,
    lead_absent_expected: bool,
) -> ExampleScore:
    """Measure one predicted decomposition against the family it came from."""
    return ExampleScore(
        reconstruction_db=signal_to_residual_db(mixture, predicted_lead + predicted_rhythm),
        lead_leakage_db=cross_role_energy_ratio_db(predicted_lead, predicted_rhythm),
        rhythm_leakage_db=cross_role_energy_ratio_db(predicted_rhythm, predicted_lead),
        lead_absent_expected=lead_absent_expected,
        lead_absent_predicted=is_absent(predicted_lead, thresholds),
        lead_audible=is_audible(predicted_lead, thresholds),
        rhythm_audible=is_audible(predicted_rhythm, thresholds),
    )


def summarise(scores, thresholds: RoleThresholds) -> dict:
    """Return the report a training run should print instead of loss alone."""
    scores = list(scores)
    if not scores:
        return {"examples": 0, "publishable": 0.0}
    finite = [s.reconstruction_db for s in scores if np.isfinite(s.reconstruction_db)]
    return {
        "examples": len(scores),
        "publishable": sum(1 for s in scores if s.passes(thresholds)) / len(scores),
        "median_reconstruction_db": float(np.median(finite)) if finite else float("inf"),
        "worst_leakage_db": max(
            max(s.lead_leakage_db, s.rhythm_leakage_db) for s in scores
        ),
        "absence_recalled": _absence_recall(scores),
    }


def _absence_recall(scores) -> float:
    """Return how often a genuinely absent lead was published as silence."""
    expected = [s for s in scores if s.lead_absent_expected]
    if not expected:
        return float("nan")
    return sum(1 for s in expected if s.lead_absent_predicted) / len(expected)
