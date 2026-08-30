"""Rhythm and lead parts built from disjoint material.

The contract requires stable roles, not a permutation: a model must always put
foreground melodic material in the lead lane and accompaniment in the rhythm
lane. A corpus can only teach that if no single source ever carries both, so
these two functions never share a note event.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from RoleSpecialist.corpus.strings import envelope, pluck

# E standard, dropped to the register metal rhythm guitar actually occupies.
LOW_E_HZ = 82.41
SEMITONE = 2.0 ** (1.0 / 12.0)

# A minor pentatonic degrees, the vocabulary most metal lead lines are built on.
PENTATONIC_SEMITONES = (0, 3, 5, 7, 10, 12, 15, 17)


def note_hz(root_hz: float, semitones: int) -> float:
    return root_hz * (SEMITONE**semitones)


# A palm-muted note is a damped string: it loses energy fast and loses its top
# faster. These are what separate a chugging riff from a ringing chord, and a
# corpus without them does not sound like metal to anything, model or ear.
MUTED_DECAY = 0.930
OPEN_DECAY = 0.996
MUTED_BRIGHTNESS = 0.80
OPEN_BRIGHTNESS = 0.50
MUTED_LENGTH = 0.40


def rhythm_part(
    duration_seconds: float,
    sample_rate: int,
    *,
    seed: int,
    riff: Sequence[int],
    tempo_bpm: float = 160.0,
    root_hz: float = LOW_E_HZ,
    palm_mute_rate: float = 0.5,
) -> np.ndarray:
    """Return a driving low-register riff of repeated power chords.

    A power chord is the root and its fifth, which is why rhythm guitar sits in
    a narrow band and stacks without turning to mud.

    The riff, tempo and root are arguments rather than constants because a
    corpus that fixes them renders one example many times over.
    """
    if not len(riff):
        raise ValueError("A rhythm part needs at least one note in its riff.")
    if not 0.0 <= palm_mute_rate <= 1.0:
        raise ValueError("The palm-mute rate must fall in [0, 1].")
    eighth = 60.0 / tempo_bpm / 2.0
    frames_per_note = max(1, int(round(eighth * sample_rate)))
    total = int(round(duration_seconds * sample_rate))
    part = np.zeros(total, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for step, offset in enumerate(np.resize(riff, total // frames_per_note + 1)):
        start = step * frames_per_note
        if start >= total:
            break
        span = min(frames_per_note, total - start)
        muted = bool(rng.random() < palm_mute_rate)
        decay = MUTED_DECAY if muted else OPEN_DECAY
        brightness = MUTED_BRIGHTNESS if muted else OPEN_BRIGHTNESS
        # A muted note stops well before the next one, which is what makes a
        # riff read as separate hits rather than one sustained wall.
        length = eighth * (MUTED_LENGTH if muted else 1.0)
        root = pluck(
            note_hz(root_hz, int(offset)), length, sample_rate,
            seed=seed + step, decay=decay, brightness=brightness,
        )
        fifth = pluck(
            note_hz(root_hz, int(offset) + 7), length, sample_rate,
            seed=seed + 977 + step, decay=decay, brightness=brightness,
        )
        voiced = max(1, min(span, root.shape[0], fifth.shape[0]))
        shape = envelope(voiced, sample_rate, release_ms=12.0 if muted else 25.0)
        chord = (root[:voiced] + 0.8 * fifth[:voiced]) * shape[:voiced]
        part[start : start + voiced] += chord
    return part.astype(np.float32)


def lead_part(
    duration_seconds: float,
    sample_rate: int,
    *,
    seed: int,
    scale: Sequence[int] = PENTATONIC_SEMITONES,
    tempo_bpm: float = 160.0,
    root_hz: float = LOW_E_HZ * 2.0,
    notes_per_beat: int = 4,
    sustain_rate: float = 0.25,
) -> np.ndarray:
    """Return a single-note melodic line above the riff.

    One note at a time, in a higher register, is what makes a lead line read as
    foreground. It shares no note event with the rhythm part.

    The scale and note density vary because a corpus built on one vocabulary
    teaches the model that vocabulary rather than the role it belongs to.
    """
    if not len(scale):
        raise ValueError("A lead part needs at least one degree in its scale.")
    if notes_per_beat <= 0:
        raise ValueError("A lead part needs a positive note density.")
    step_seconds = 60.0 / tempo_bpm / notes_per_beat
    frames_per_note = max(1, int(round(step_seconds * sample_rate)))
    total = int(round(duration_seconds * sample_rate))
    part = np.zeros(total, dtype=np.float64)
    rng = np.random.default_rng(seed)

    steps = total // frames_per_note + 1
    degrees = rng.integers(0, len(scale), size=steps)
    step = 0
    while step < steps:
        start = step * frames_per_note
        if start >= total:
            break
        # A lead line is not a metronome. Held notes are what make it sing
        # rather than chatter, and a corpus of even sixteenths teaches the
        # model that density alone marks the role.
        held = 4 if rng.random() < sustain_rate else 1
        semitones = scale[int(degrees[step])]
        note = pluck(
            note_hz(root_hz, semitones), step_seconds * held, sample_rate,
            seed=seed + 5_003 + step, decay=0.9985,
        )
        # Take the span from the note that was generated. Multiplying a rounded
        # per-note frame count by the hold length rounds differently, and the
        # two disagree by a sample or two.
        span = min(note.shape[0], total - start)
        shape = envelope(span, sample_rate, release_ms=40.0 if held > 1 else 25.0)
        part[start : start + span] += note[:span] * shape[:span]
        step += held
    return part.astype(np.float32)
