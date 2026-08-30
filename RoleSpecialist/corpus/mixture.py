"""One training example whose role labels are exact by construction.

Nothing here annotates audio. The lead and rhythm stems are the sources that
were rendered, so their labels cannot be wrong, and the guitar family the
specialist will receive is their sum. That is the whole reason to synthesise a
corpus rather than label a real one.

The stereo placement is the convention the deterministic Metal Stereo profile
already exploits: rhythm guitars doubled and panned wide, lead centred. The
specialist must not learn to rely on it, so the corpus has to be able to
violate it on purpose, which `centred_rhythm` exists to do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from RoleSpecialist.corpus.amp import amplify
from RoleSpecialist.corpus.performance import lead_part, rhythm_part

SAMPLE_RATE = 44_100


def _stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([left, right], axis=1).astype(np.float32)


def _centred(mono: np.ndarray, gain: float) -> np.ndarray:
    scaled = (mono * gain).astype(np.float32)
    return _stereo(scaled, scaled)


@dataclass(frozen=True)
class LabelledMixture:
    """A rendered example and the two sources it was rendered from."""

    lead: np.ndarray
    rhythm: np.ndarray
    sample_rate: int
    lead_is_absent: bool

    @property
    def guitar_family(self) -> np.ndarray:
        """Return what the isolation stage hands the specialist."""
        return (self.lead + self.rhythm).astype(np.float32)

    @property
    def frame_count(self) -> int:
        return int(self.lead.shape[0])


def render(
    duration_seconds: float = 4.0,
    *,
    seed: int,
    sample_rate: int = SAMPLE_RATE,
    with_lead: bool = True,
    centred_rhythm: bool = False,
    rhythm_gain: float = 0.55,
    lead_gain: float = 0.70,
) -> LabelledMixture:
    """Render one example.

    `with_lead=False` produces a track whose lead lane is real silence. Metal
    arrangements are full of them, and the contract treats such a lane as a
    valid result rather than a failure, so the corpus has to contain them or
    the specialist will never learn to publish one.
    """
    rhythm_mono = amplify(
        rhythm_part(duration_seconds, sample_rate, seed=seed), sample_rate, drive=22.0
    )
    if centred_rhythm:
        rhythm = _centred(rhythm_mono, rhythm_gain)
    else:
        # Two takes, not one take copied: a doubled rhythm guitar is two
        # performances, and duplicating one channel would make the pair
        # perfectly correlated and trivially separable by position alone.
        second = amplify(
            rhythm_part(duration_seconds, sample_rate, seed=seed + 131), sample_rate, drive=22.0
        )
        rhythm = _stereo(rhythm_mono * rhythm_gain, second[: rhythm_mono.shape[0]] * rhythm_gain)

    if with_lead:
        lead_mono = amplify(
            lead_part(duration_seconds, sample_rate, seed=seed), sample_rate, drive=14.0, presence=0.5
        )
        lead = _centred(lead_mono, lead_gain)
    else:
        lead = np.zeros_like(rhythm)

    return LabelledMixture(
        lead=lead, rhythm=rhythm, sample_rate=sample_rate, lead_is_absent=not with_lead
    )
