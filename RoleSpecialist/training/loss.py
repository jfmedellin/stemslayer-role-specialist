"""The training objective, and why an L1 term alone cannot reach the gate.

Two hundred epochs took reconstruction from 4.4 dB to 32.2 dB and left absence
recall at exactly zero. On tracks whose true lead is digital silence the model
settled at about -23 dBFS while the contract counts a lane as absent at or
below -80 dBFS, and it had stopped approaching.

That is not undertraining, it is a mismatch of units. L1 measures an average
distance, so pushing the last of a lane from -23 dBFS to -80 dBFS buys almost
nothing: in linear amplitude those are both indistinguishable from zero. The
contract asks an absolute question in decibels, and the loss was answering a
relative one in amplitude.

The term below asks the contract's question. It applies only to lanes whose
target is genuinely silent, because a penalty that rewarded quiet everywhere
would teach the model that emitting nothing is always safe, and Stemslayer's
contract is explicit that a silent lane which loses energy is a failure.
"""

from __future__ import annotations

SILENCE_FLOOR = 1e-12
# Aim below the gate rather than at it. A model trained to land exactly on the
# threshold fails it on the first track that is slightly harder.
DEFAULT_MARGIN_DB = 10.0
# Loss contributed per decibel of excess. At the ~57 dB gap this run started
# with, 0.001 per dB is about 0.057, which is the same order as the L1 term it
# has to share gradient with rather than overwhelm.
DEFAULT_WEIGHT_PER_DB = 0.001


def silent_lane_mask(targets):
    """Return which (example, source) targets are genuinely digital silence.

    The corpus renders an absent lead as exact zeros, so this is a fact about
    the label rather than a threshold judgement.
    """
    import torch

    return torch.amax(torch.abs(targets), dim=(-2, -1)) == 0


def lane_level_db(estimate):
    """Return each lane's RMS level in dB.

    RMS rather than peak: the gate measures peak, but peak carries gradient at
    one sample while RMS carries it at every one. A lane driven to a low RMS
    arrives at a low peak, and it gets there from a signal that is actually
    learnable.
    """
    import torch

    power = torch.mean(estimate**2, dim=(-2, -1))
    return 10.0 * torch.log10(power + SILENCE_FLOOR)


def silence_penalty(
    estimate,
    targets,
    *,
    floor_dbfs: float,
    margin_db: float = DEFAULT_MARGIN_DB,
    weight_per_db: float = DEFAULT_WEIGHT_PER_DB,
):
    """Return the cost of energy in a lane that should hold none.

    Zero once a silent lane is quiet enough, and linear in decibels above that,
    so the push is the same strength at -20 dBFS as at -60. That constancy is
    the point: it is exactly what an amplitude-domain term stops providing.
    """
    import torch

    mask = silent_lane_mask(targets)
    if not bool(torch.any(mask)):
        return torch.zeros((), device=estimate.device, dtype=estimate.dtype)
    excess = torch.relu(lane_level_db(estimate) - (floor_dbfs - margin_db))
    return weight_per_db * (excess * mask).sum() / mask.sum()


def separation_loss(
    estimate,
    targets,
    *,
    floor_dbfs: float,
    margin_db: float = DEFAULT_MARGIN_DB,
    weight_per_db: float = DEFAULT_WEIGHT_PER_DB,
):
    """Return the total loss and its two parts, so a run can report both."""
    import torch

    reconstruction = torch.nn.functional.l1_loss(estimate, targets)
    silence = silence_penalty(
        estimate,
        targets,
        floor_dbfs=floor_dbfs,
        margin_db=margin_db,
        weight_per_db=weight_per_db,
    )
    return reconstruction + silence, float(reconstruction.item()), float(silence.item())
