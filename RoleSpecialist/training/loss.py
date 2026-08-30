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

The term below asks a decibel question, and asks a smaller one than the first
attempt did. Aiming it at the -80 dBFS absence floor cost nine decibels of
reconstruction and took a run from 75% publishable to nothing, because a shared
network cannot push one lane thirty decibels further down without biasing the
same lane everywhere else.

It does not need to. Absence is decided by reconstruction at publication time,
and a lane only has to be quiet enough to *become a candidate* for that
decision: below the audibility minimum. That is forty decibels less work, and
it is the whole job. Aiming at the floor was aiming at a classification
threshold as though it were a target the network had to reach.

The term applies only to lanes whose target is genuinely silent, because a
penalty that rewarded quiet everywhere would teach the model that emitting
nothing is always safe, and Stemslayer's contract is explicit that a silent
lane which loses energy is a failure.
"""

from __future__ import annotations

SILENCE_FLOOR = 1e-12
# Aim below the threshold rather than at it. A model trained to land exactly on
# it fails on the first track that is slightly harder.
DEFAULT_MARGIN_DB = 10.0
# The share of samples the level estimate averages. Measured at 0.01% it sits
# about 3 dB below the true peak, which the margin above absorbs.
LOUDEST_FRACTION = 1e-4
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


def lane_level_db(estimate, fraction: float = LOUDEST_FRACTION):
    """Return each lane's level in dB, measured the way the gate measures it.

    This began as RMS, on the reasoning that peak carries gradient at a single
    sample while RMS carries it everywhere, and that a lane driven to a low RMS
    arrives at a low peak. The second half of that is false, and a run proved
    it: the term reported itself satisfied from epoch 49 onward while the level
    the gate actually reads drifted from -26 dBFS to -13 dBFS.

    The residue in a lane that should be silent is impulsive, so RMS averages
    the transients away and peak does not. Measured on that run's checkpoint:

        RMS                 -55.3 dB      30.4 dB below the peak
        mean of loudest 1%  -41.6 dB      16.7 dB below
        mean of loudest 0.1% -32.3 dB      7.4 dB below
        mean of loudest 0.01% -27.8 dB     2.9 dB below
        peak                -24.9 dBFS    what the gate reads

    The mean of the loudest fraction tracks the peak within a few decibels and
    still spreads gradient across tens of samples rather than one. Optimising
    it moves the number that decides publication, which RMS did not.
    """
    import torch

    flat = torch.abs(estimate).flatten(start_dim=-2)
    count = max(1, int(fraction * flat.shape[-1]))
    loudest = torch.topk(flat, count, dim=-1).values.mean(dim=-1)
    return 20.0 * torch.log10(loudest + SILENCE_FLOOR)


def silence_penalty(
    estimate,
    targets,
    *,
    candidate_dbfs: float,
    margin_db: float = DEFAULT_MARGIN_DB,
    weight_per_db: float = DEFAULT_WEIGHT_PER_DB,
):
    """Return the cost of energy in a lane that should hold none.

    `candidate_dbfs` is the level a lane must fall below to be considered for
    absence at publication time, not the level it must eventually reach. The
    penalty is zero once a silent lane is that quiet, and linear in decibels
    above it, so the push is the same strength at -20 dBFS as at -45. That
    constancy is the point: it is exactly what an amplitude-domain term stops
    providing.
    """
    import torch

    mask = silent_lane_mask(targets)
    if not bool(torch.any(mask)):
        return torch.zeros((), device=estimate.device, dtype=estimate.dtype)
    excess = torch.relu(lane_level_db(estimate) - (candidate_dbfs - margin_db))
    return weight_per_db * (excess * mask).sum() / mask.sum()


def separation_loss(
    estimate,
    targets,
    *,
    candidate_dbfs: float,
    margin_db: float = DEFAULT_MARGIN_DB,
    weight_per_db: float = DEFAULT_WEIGHT_PER_DB,
):
    """Return the total loss and its two parts, so a run can report both."""
    import torch

    reconstruction = torch.nn.functional.l1_loss(estimate, targets)
    silence = silence_penalty(
        estimate,
        targets,
        candidate_dbfs=candidate_dbfs,
        margin_db=margin_db,
        weight_per_db=weight_per_db,
    )
    return reconstruction + silence, float(reconstruction.item()), float(silence.item())
