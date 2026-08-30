"""Every musical choice one example makes, and how a split draws them.

A corpus rendered from one riff at one tempo in one key is a thousand copies of
the same example however many seeds it uses. The excitation noise differs, so
the waveforms decorrelate and the corpus looks varied to a correlation check
while teaching the model exactly one riff.

Splitting matters as much as varying. Holding out only the seed puts the same
riff on both sides of the split, and validation then measures memorisation and
calls it accuracy. The pools below are disjoint by construction: a riff or a
scale that trains is never one that validates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRAIN = "train"
VALIDATION = "validation"

# Semitone offsets from the root, one entry per eighth note. Metal rhythm
# playing is mostly the open root with figures moving off it, which is why the
# root repeats inside every pattern.
TRAIN_RIFFS = (
    (0, 0, 3, 0, 0, 5, 3, 0),
    (0, 0, 0, 5, 0, 0, 3, 5),
    (0, 3, 0, 3, 5, 3, 0, 0),
    (0, 0, 7, 5, 0, 0, 3, 2),
    (0, 5, 0, 5, 0, 3, 0, 2),
    (0, 0, 2, 3, 0, 0, 7, 5),
)
VALIDATION_RIFFS = (
    (0, 0, 5, 3, 0, 2, 0, 0),
    (0, 7, 0, 3, 0, 5, 2, 0),
)

# Lead vocabularies. Held out the same way: a scale the model trained on is not
# one it is validated on.
TRAIN_SCALES = (
    (0, 3, 5, 7, 10, 12, 15, 17),  # minor pentatonic
    (0, 2, 3, 5, 7, 8, 10, 12),  # natural minor
    (0, 1, 5, 7, 8, 12, 13, 17),  # phrygian dominant fragment
)
VALIDATION_SCALES = (
    (0, 2, 3, 5, 7, 10, 12, 14),  # dorian fragment
    (0, 3, 5, 6, 7, 10, 12, 15),  # blues
)

# Guitars are not all tuned to E, and a corpus in one key teaches pitch rather
# than role. These span E standard down to drop-A territory.
ROOT_HZ_RANGE = (55.0, 98.0)
TEMPO_BPM_RANGE = (95.0, 210.0)
RHYTHM_DRIVE_RANGE = (14.0, 30.0)
LEAD_DRIVE_RANGE = (9.0, 18.0)
LEAD_REGISTER_CHOICES = (2.0, 3.0, 4.0)
LEAD_DENSITY_CHOICES = (2, 4)

# Metal arrangements are full of passages with no lead at all, and a corpus
# without them teaches the model that a silent lead lane is always wrong.
LEAD_ABSENT_RATE = 0.25
# The deterministic Metal Stereo profile already reads stereo position. A
# specialist that learned position instead of role would be worthless, so a
# quarter of the corpus places the rhythm where position cannot help.
CENTRED_RHYTHM_RATE = 0.25


@dataclass(frozen=True)
class Arrangement:
    """The complete musical description of one rendered example."""

    split: str
    index: int
    seed: int
    tempo_bpm: float
    root_hz: float
    riff: tuple[int, ...]
    scale: tuple[int, ...]
    rhythm_drive: float
    lead_drive: float
    lead_register: float
    lead_density: int
    with_lead: bool
    centred_rhythm: bool

    @property
    def identity(self) -> tuple:
        """Return what makes this example musically distinct.

        The seed is excluded on purpose: two examples that differ only by
        excitation noise are the same example, and treating them as distinct is
        how a corpus fools itself into looking varied.
        """
        return (
            self.riff,
            self.scale,
            round(self.tempo_bpm, 2),
            round(self.root_hz, 2),
            self.lead_register,
            self.lead_density,
            self.with_lead,
            self.centred_rhythm,
        )


def _pools(split: str) -> tuple[tuple, tuple]:
    if split == TRAIN:
        return TRAIN_RIFFS, TRAIN_SCALES
    if split == VALIDATION:
        return VALIDATION_RIFFS, VALIDATION_SCALES
    raise ValueError(f"Unknown split {split!r}; expected {TRAIN!r} or {VALIDATION!r}.")


def sample(split: str, index: int) -> Arrangement:
    """Return the arrangement at one position of a split, deterministically.

    The same split and index always produce the same arrangement, so a corpus
    is regenerated rather than stored, and a checkpoint's training data can be
    reproduced from its manifest alone.
    """
    if index < 0:
        raise ValueError("An arrangement index cannot be negative.")
    riffs, scales = _pools(split)
    # Separate streams per split, so adding examples to one never shifts the
    # other's draws and invalidates a comparison.
    rng = np.random.default_rng(abs(hash((split, index))) % (2**63))

    return Arrangement(
        split=split,
        index=index,
        seed=int(rng.integers(1, 2**31 - 1)),
        tempo_bpm=float(rng.uniform(*TEMPO_BPM_RANGE)),
        root_hz=float(rng.uniform(*ROOT_HZ_RANGE)),
        riff=riffs[int(rng.integers(len(riffs)))],
        scale=scales[int(rng.integers(len(scales)))],
        rhythm_drive=float(rng.uniform(*RHYTHM_DRIVE_RANGE)),
        lead_drive=float(rng.uniform(*LEAD_DRIVE_RANGE)),
        lead_register=float(LEAD_REGISTER_CHOICES[int(rng.integers(len(LEAD_REGISTER_CHOICES)))]),
        lead_density=int(LEAD_DENSITY_CHOICES[int(rng.integers(len(LEAD_DENSITY_CHOICES)))]),
        with_lead=bool(rng.random() >= LEAD_ABSENT_RATE),
        centred_rhythm=bool(rng.random() < CENTRED_RHYTHM_RATE),
    )
