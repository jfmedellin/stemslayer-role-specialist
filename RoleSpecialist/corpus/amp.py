"""Deterministic amplifier and cabinet simulation.

Metal guitar is defined as much by the amplifier as by the string, so a corpus
of clean plucks would not resemble the material the specialist has to separate.
This is a coarse model, not a convolution of a captured cabinet: an impulse
response captured from real hardware carries the same rights question as a
sample library, and the point of synthesising is to avoid that question.

Nothing here is random. The same input always produces the same output, so a
regenerated corpus is identical to the one a checkpoint was trained on.
"""

from __future__ import annotations

import numpy as np


def _one_pole_lowpass(signal: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    coefficient = float(np.exp(-2.0 * np.pi * cutoff_hz / sample_rate))
    output = np.empty_like(signal, dtype=np.float64)
    previous = 0.0
    for index, sample in enumerate(signal.astype(np.float64)):
        previous = (1.0 - coefficient) * sample + coefficient * previous
        output[index] = previous
    return output


def amplify(
    signal: np.ndarray,
    sample_rate: int,
    *,
    drive: float = 18.0,
    presence: float = 0.35,
    cabinet_hz: float = 5_000.0,
    body_hz: float = 95.0,
) -> np.ndarray:
    """Return the signal driven through a saturating stage and a cabinet.

    `drive` is pre-gain into the saturation, which is what separates a rhythm
    tone from a clean one. `presence` mixes back the high end the cabinet
    removes, so a lead line still cuts through a dense mix.
    """
    if drive <= 0.0:
        raise ValueError("Drive must be positive.")
    driven = np.tanh(np.asarray(signal, dtype=np.float64) * drive)
    body = _one_pole_lowpass(driven, cabinet_hz, sample_rate)
    # What the cabinet removed, added back in a controlled amount.
    top = driven - body
    voiced = body + presence * top
    # A guitar cabinet has no useful output below the low string, and leaving
    # it in makes mixtures sum into rumble that no real recording contains.
    voiced = voiced - _one_pole_lowpass(voiced, body_hz, sample_rate)
    peak = float(np.max(np.abs(voiced)))
    if peak > 0.0:
        voiced = voiced / peak
    return voiced.astype(np.float32)
