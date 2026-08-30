"""Rhythm and lead parts built from disjoint material.

The contract requires stable roles, not a permutation: a model must always put
foreground melodic material in the lead lane and accompaniment in the rhythm
lane. A corpus can only teach that if no single source ever carries both, so
these two functions never share a note event.
"""

from __future__ import annotations

import numpy as np

from RoleSpecialist.corpus.strings import envelope, pluck

# E standard, dropped to the register metal rhythm guitar actually occupies.
LOW_E_HZ = 82.41
SEMITONE = 2.0 ** (1.0 / 12.0)

# A minor pentatonic degrees, the vocabulary most metal lead lines are built on.
PENTATONIC_SEMITONES = (0, 3, 5, 7, 10, 12, 15, 17)


def note_hz(root_hz: float, semitones: int) -> float:
    return root_hz * (SEMITONE**semitones)


def rhythm_part(
    duration_seconds: float,
    sample_rate: int,
    *,
    seed: int,
    tempo_bpm: float = 160.0,
    root_hz: float = LOW_E_HZ,
) -> np.ndarray:
    """Return a driving low-register riff of repeated power chords.

    A power chord is the root and its fifth, which is why rhythm guitar sits in
    a narrow band and stacks without turning to mud.
    """
    eighth = 60.0 / tempo_bpm / 2.0
    frames_per_note = max(1, int(round(eighth * sample_rate)))
    total = int(round(duration_seconds * sample_rate))
    part = np.zeros(total, dtype=np.float64)
    shape = envelope(frames_per_note, sample_rate)

    pattern = (0, 0, 3, 0, 0, 5, 3, 0)
    for step, offset in enumerate(np.resize(pattern, total // frames_per_note + 1)):
        start = step * frames_per_note
        if start >= total:
            break
        span = min(frames_per_note, total - start)
        root = pluck(note_hz(root_hz, int(offset)), eighth, sample_rate, seed=seed + step)
        fifth = pluck(note_hz(root_hz, int(offset) + 7), eighth, sample_rate, seed=seed + 977 + step)
        chord = (root[:span] + 0.8 * fifth[:span]) * shape[:span]
        part[start : start + span] += chord
    return part.astype(np.float32)


def lead_part(
    duration_seconds: float,
    sample_rate: int,
    *,
    seed: int,
    tempo_bpm: float = 160.0,
    root_hz: float = LOW_E_HZ * 2.0,
) -> np.ndarray:
    """Return a single-note melodic line an octave above the riff.

    One note at a time, in a higher register, is what makes a lead line read as
    foreground. It shares no note event with the rhythm part.
    """
    sixteenth = 60.0 / tempo_bpm / 4.0
    frames_per_note = max(1, int(round(sixteenth * sample_rate)))
    total = int(round(duration_seconds * sample_rate))
    part = np.zeros(total, dtype=np.float64)
    shape = envelope(frames_per_note, sample_rate)
    rng = np.random.default_rng(seed)

    steps = total // frames_per_note + 1
    degrees = rng.integers(0, len(PENTATONIC_SEMITONES), size=steps)
    for step, degree in enumerate(degrees):
        start = step * frames_per_note
        if start >= total:
            break
        span = min(frames_per_note, total - start)
        semitones = PENTATONIC_SEMITONES[int(degree)]
        note = pluck(note_hz(root_hz, semitones), sixteenth, sample_rate, seed=seed + 5_003 + step)
        part[start : start + span] += note[:span] * shape[:span]
    return part.astype(np.float32)
