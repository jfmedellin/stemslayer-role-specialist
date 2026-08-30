"""A two-output HTDemucs, built from the reference implementation.

The admission contract names an HTDemucs-family checkpoint applied to the
guitar family that `htdemucs_6s` already isolates. Reimplementing that
architecture would be a second place for it to be subtly wrong, so this
constructs the published one with two sources instead of four.

Source order is fixed and meaningful. The contract requires stable roles rather
than a permutation, so output zero is always lead and output one is always
rhythm, in training, in evaluation, and in whatever runs at inference.
"""

from __future__ import annotations

SOURCES = ("lead_guitar", "rhythm_guitar")


def build_model(*, sample_rate: int = 44_100, channels: int = 2, depth: int = 4, width: int = 32):
    """Return an untrained two-source HTDemucs.

    `depth` and `width` are smaller than the published four-source model. The
    corpus is synthetic and modest, and a network with far more capacity than
    the data supports memorises it instead of learning the role.
    """
    from demucs.htdemucs import HTDemucs

    return HTDemucs(
        sources=list(SOURCES),
        audio_channels=channels,
        channels=width,
        depth=depth,
        samplerate=sample_rate,
        segment=8,
    )


def source_index(name: str) -> int:
    """Return the fixed output position of a role."""
    return SOURCES.index(name)
