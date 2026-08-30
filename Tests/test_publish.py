"""Absence is decided here, by reconstruction, exactly as the contract says.

A network does not emit digital silence, and training one to try cost this
project a run. What it can do is leave a lane quiet enough to be considered,
and publication decides the rest.
"""

import unittest

import numpy as np

from RoleSpecialist.publish import resolve_absence
from RoleSpecialist.vendor.role_metrics import RoleThresholds

THRESHOLDS = RoleThresholds(
    calibrated=False,
    reconstruction_minimum_db=30.0,
    leakage_maximum_db=-12.0,
    audibility_minimum_dbfs=-40.0,
    absence_at_or_below_dbfs=-80.0,
)
FRAMES = 4_096


def tone(frequency: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(FRAMES, dtype=np.float32) / 44_100.0
    mono = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    return np.stack([mono, mono], axis=1)


class AbsenceDecisionTests(unittest.TestCase):
    def test_a_quiet_lane_whose_removal_costs_nothing_publishes_as_silence(self):
        rhythm = tone(110.0)
        residue = tone(880.0, amplitude=10 ** (-55.0 / 20.0))

        published = resolve_absence(rhythm + residue, residue, rhythm, THRESHOLDS)

        self.assertTrue(published.lead_absent)
        self.assertEqual(0.0, float(np.max(np.abs(published.lead))))
        self.assertTrue(published.reason)

    def test_an_audible_lane_is_never_removed_however_well_it_reconstructs(self):
        """Level makes a lane a candidate; it never makes it absent."""
        lead, rhythm = tone(880.0), tone(110.0)

        published = resolve_absence(lead + rhythm, lead, rhythm, THRESHOLDS)

        self.assertFalse(published.lead_absent)
        self.assertTrue(np.array_equal(lead, published.lead))

    def test_removing_a_real_lead_would_lose_energy_so_it_is_kept(self):
        """This is the guard that makes the decision safe to act on.

        Without it, any lane quiet enough to be inaudible would be discarded,
        and the contract calls a silent lane that loses energy a failure.
        """
        rhythm = tone(110.0)
        # A mixture whose lead the model under-produced: inaudible, and still
        # carrying energy the pair needs to reconstruct.
        mixture = rhythm * 2.0

        published = resolve_absence(mixture, tone(880.0, 10 ** (-50.0 / 20.0)), rhythm, THRESHOLDS)

        self.assertFalse(published.lead_absent)
        self.assertLess(published.reconstruction_db, THRESHOLDS.reconstruction_minimum_db)

    def test_the_published_rhythm_is_never_altered(self):
        rhythm = tone(110.0)
        residue = tone(880.0, amplitude=10 ** (-55.0 / 20.0))

        published = resolve_absence(rhythm + residue, residue, rhythm, THRESHOLDS)

        self.assertTrue(np.array_equal(rhythm, published.rhythm))

    def test_the_reported_reconstruction_is_the_one_that_was_published(self):
        rhythm = tone(110.0)
        residue = tone(880.0, amplitude=10 ** (-55.0 / 20.0))

        published = resolve_absence(rhythm + residue, residue, rhythm, THRESHOLDS)

        self.assertGreaterEqual(published.reconstruction_db, THRESHOLDS.reconstruction_minimum_db)


if __name__ == "__main__":
    unittest.main()
