"""Deciding what a model's output actually publishes as.

A network does not emit digital silence, and asking it to was a mistake this
project made and measured. Training a term that chased the contract's -80 dBFS
absence floor moved a silent lane from -23 dBFS to -51 dBFS and cost nine
decibels of reconstruction, taking a run from 75% publishable to nothing.

The contract never asked for that. It says absence is decided by
reconstruction, never by lane energy alone: a silent lead lane is valid
whenever `lead + rhythm` still reproduces the guitar family. That is a decision
about what to publish, made here, from a model's output — not a shape the model
has to learn to produce.

Measured on this corpus, forcing a lane to true silence when the lead is
genuinely absent leaves reconstruction unchanged at 33.3 dB, and doing it when
a lead is present collapses reconstruction from 31.0 dB to 2.9 dB. The test
distinguishes the two cases by a wide margin, which is what makes it safe to
act on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from RoleSpecialist.vendor.role_metrics import (
    RoleThresholds,
    is_audible,
    signal_to_residual_db,
)


@dataclass(frozen=True)
class Publication:
    """What a decomposition publishes, and why."""

    lead: np.ndarray
    rhythm: np.ndarray
    lead_absent: bool
    reason: str
    reconstruction_db: float


def resolve_absence(
    mixture: np.ndarray,
    lead: np.ndarray,
    rhythm: np.ndarray,
    thresholds: RoleThresholds,
) -> Publication:
    """Return the decomposition to publish, with absence decided by reconstruction.

    A lane becomes a candidate for absence only when it is too quiet to be
    audible, and is published as silence only when removing it costs no energy.
    Level alone would call a badly separated lane absent; reconstruction alone
    would do the same to a lane the model simply failed to fill.
    """
    if is_audible(lead, thresholds):
        return Publication(
            lead=lead,
            rhythm=rhythm,
            lead_absent=False,
            reason="",
            reconstruction_db=signal_to_residual_db(mixture, lead + rhythm),
        )

    silence = np.zeros_like(lead)
    without_lead = signal_to_residual_db(mixture, silence + rhythm)
    if without_lead >= thresholds.reconstruction_minimum_db:
        return Publication(
            lead=silence,
            rhythm=rhythm,
            lead_absent=True,
            reason="no lead guitar in this track",
            reconstruction_db=without_lead,
        )

    # Too quiet to hear and still carrying energy the pair needs: the model
    # failed to fill the lane rather than finding it empty. Publishing silence
    # here would lose that energy, which the contract calls a failure.
    return Publication(
        lead=lead,
        rhythm=rhythm,
        lead_absent=False,
        reason="",
        reconstruction_db=signal_to_residual_db(mixture, lead + rhythm),
    )
