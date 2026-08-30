"""Scoring has to answer the contract's questions, not the loss function's.

These use hand-built signals rather than a model, because what is under test is
the judgement, not the network.
"""

import unittest

import numpy as np

from RoleSpecialist.training.evaluate import score_example, summarise
from RoleSpecialist.vendor.role_metrics import RoleThresholds

THRESHOLDS = RoleThresholds(
    calibrated=False,
    reconstruction_minimum_db=30.0,
    leakage_maximum_db=-12.0,
    audibility_minimum_dbfs=-40.0,
    absence_at_or_below_dbfs=-80.0,
    calibration_blocker="targets copied from the admission contract",
)
FRAMES = 4_096


def tone(frequency: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(FRAMES, dtype=np.float32) / 44_100.0
    mono = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    return np.stack([mono, mono], axis=1)


class ScoringTests(unittest.TestCase):
    def test_a_clean_decomposition_is_publishable(self):
        lead, rhythm = tone(880.0), tone(110.0)

        score = score_example(lead + rhythm, lead, rhythm, THRESHOLDS, lead_absent_expected=False)

        self.assertTrue(score.passes(THRESHOLDS))

    def test_losing_energy_fails_however_clean_the_lanes_look(self):
        lead, rhythm = tone(880.0), tone(110.0)

        score = score_example(
            lead + rhythm, lead, rhythm * 0.5, THRESHOLDS, lead_absent_expected=False
        )

        self.assertFalse(score.passes(THRESHOLDS))

    def test_a_silent_lead_that_still_reconstructs_is_a_correct_result(self):
        """Absence is decided by reconstruction, never by lane energy alone."""
        rhythm = tone(110.0)
        silence = np.zeros_like(rhythm)

        score = score_example(rhythm, silence, rhythm, THRESHOLDS, lead_absent_expected=True)

        self.assertTrue(score.lead_absent_predicted)
        self.assertTrue(score.passes(THRESHOLDS))

    def test_a_silent_lead_nobody_asked_for_fails(self):
        lead, rhythm = tone(880.0), tone(110.0)
        silence = np.zeros_like(lead)

        score = score_example(
            lead + rhythm, silence, lead + rhythm, THRESHOLDS, lead_absent_expected=False
        )

        self.assertFalse(score.passes(THRESHOLDS))

    def test_a_noise_floor_placeholder_is_not_an_absent_lane(self):
        """The band between the absence floor and the audibility minimum is
        exactly where a model hides a lane it could not produce."""
        rhythm = tone(110.0)
        placeholder = tone(880.0, amplitude=10 ** (-60.0 / 20.0))

        score = score_example(
            rhythm + placeholder, placeholder, rhythm, THRESHOLDS, lead_absent_expected=False
        )

        self.assertFalse(score.lead_audible)
        self.assertFalse(score.lead_absent_predicted)
        self.assertFalse(score.passes(THRESHOLDS))

    def test_duplicated_roles_fail_on_leakage(self):
        rhythm = tone(110.0)

        score = score_example(
            rhythm * 2, rhythm, rhythm, THRESHOLDS, lead_absent_expected=False
        )

        self.assertFalse(score.passes(THRESHOLDS))


class SummaryTests(unittest.TestCase):
    def test_the_summary_reports_publishability_rather_than_loss(self):
        lead, rhythm = tone(880.0), tone(110.0)
        good = score_example(lead + rhythm, lead, rhythm, THRESHOLDS, lead_absent_expected=False)
        bad = score_example(lead + rhythm, lead, rhythm * 0.5, THRESHOLDS, lead_absent_expected=False)

        report = summarise([good, bad], THRESHOLDS)

        self.assertEqual(2, report["examples"])
        self.assertEqual(0.5, report["publishable"])

    def test_absence_recall_is_reported_only_when_absence_was_expected(self):
        lead, rhythm = tone(880.0), tone(110.0)
        present = score_example(lead + rhythm, lead, rhythm, THRESHOLDS, lead_absent_expected=False)

        self.assertTrue(np.isnan(summarise([present], THRESHOLDS)["absence_recalled"]))

    def test_an_empty_run_reports_nothing_publishable(self):
        self.assertEqual(0, summarise([], THRESHOLDS)["examples"])


if __name__ == "__main__":
    unittest.main()
