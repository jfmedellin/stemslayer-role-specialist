"""A plucked string modelled from scratch, so every sample is ours to license.

A recorded DI library or a sampled virtual instrument would sound better, but
neither states whether its licence permits training redistributable weights,
and the admission contract rejects material whose rights are unclear whatever
the measured quality. Synthesis has no such ambiguity: nothing here is derived
from a recording.

The model is Karplus-Strong. A period of noise is filtered as it recirculates,
which is a crude digital waveguide and behaves like a struck string: an
inharmonic attack that settles into a decaying harmonic series.
"""

from __future__ import annotations

import numpy as np

MIN_PERIOD = 2


def pluck(
    frequency_hz: float,
    duration_seconds: float,
    sample_rate: int,
    *,
    seed: int,
    decay: float = 0.996,
    brightness: float = 0.5,
) -> np.ndarray:
    """Return one mono plucked note.

    The seed fixes the excitation noise, so a corpus built from the same seeds
    is reproducible sample for sample. A corpus that cannot be regenerated
    cannot be audited, and admission asks where the training data came from.
    """
    if frequency_hz <= 0.0:
        raise ValueError("A plucked note needs a positive frequency.")
    if duration_seconds <= 0.0:
        raise ValueError("A plucked note needs a positive duration.")
    if not 0.0 < decay <= 1.0:
        raise ValueError("Decay must fall in (0, 1].")
    if not 0.0 <= brightness <= 1.0:
        raise ValueError("Brightness must fall in [0, 1].")

    period = max(MIN_PERIOD, int(round(sample_rate / frequency_hz)))
    frames = int(round(duration_seconds * sample_rate))
    buffer = np.random.default_rng(seed).uniform(-1.0, 1.0, period)
    output = np.empty(frames, dtype=np.float64)

    # The recirculating filter is a running average of adjacent samples, so it
    # cannot be vectorised: each output feeds the next.
    index = 0
    for frame in range(frames):
        current = buffer[index]
        output[frame] = current
        following = buffer[(index + 1) % period]
        buffer[index] = decay * (brightness * current + (1.0 - brightness) * following)
        index = (index + 1) % period
    return output.astype(np.float32)


def envelope(frames: int, sample_rate: int, *, attack_ms: float = 4.0, release_ms: float = 25.0) -> np.ndarray:
    """Return a gain ramp that keeps note edges from clicking."""
    ramp = np.ones(frames, dtype=np.float32)
    attack = min(frames // 2, int(sample_rate * attack_ms / 1000.0))
    release = min(frames - attack, int(sample_rate * release_ms / 1000.0))
    if attack > 0:
        ramp[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    if release > 0:
        ramp[frames - release :] = np.linspace(1.0, 0.0, release, dtype=np.float32)
    return ramp
