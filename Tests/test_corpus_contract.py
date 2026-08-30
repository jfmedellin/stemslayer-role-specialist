"""What the synthetic corpus has to be true of before anything trains on it.

These measure the rendered ground truth with the same functions admission
uses. If our own labels cannot satisfy the gates a checkpoint will be held to,
no amount of training will fix it, and we would find out only after spending
the compute.
"""

import unittest

import numpy as np

from RoleSpecialist.corpus.mixture import render
from RoleSpecialist.vendor.role_metrics import (
    RoleThresholds,
    cross_role_energy_ratio_db,
    is_absent,
    is_audible,
    peak_dbfs,
    signal_to_residual_db,
)

# Short renders: the string model recirculates one sample at a time, so a
# four-second example is slow enough to make a suite unpleasant.
DURATION = 0.75

TARGETS = RoleThresholds(
    calibrated=False,
    reconstruction_minimum_db=30.0,
    leakage_maximum_db=-12.0,
    audibility_minimum_dbfs=-40.0,
    absence_at_or_below_dbfs=-80.0,
    calibration_blocker="targets copied from the admission contract",
)


class CorpusContractTests(unittest.TestCase):
    def setUp(self):
        self.example = render(DURATION, seed=7)

    def test_both_roles_carry_meaningful_signal(self):
        for role, samples in (("lead", self.example.lead), ("rhythm", self.example.rhythm)):
            with self.subTest(role=role):
                self.assertTrue(
                    is_audible(samples, TARGETS),
                    f"{role} peaks at {peak_dbfs(samples):.1f} dBFS, which is a placeholder",
                )
                self.assertFalse(is_absent(samples, TARGETS))

    def test_the_roles_do_not_carry_each_other(self):
        """Leakage the corpus contains is leakage the specialist cannot beat.

        The gate limits how much of one role's energy appears in the other's
        lane. Ground truth that already breaches it would teach the model that
        breaching it is correct.
        """
        for lane_name, lane, other in (
            ("lead", self.example.lead, self.example.rhythm),
            ("rhythm", self.example.rhythm, self.example.lead),
        ):
            with self.subTest(lane=lane_name):
                leakage = cross_role_energy_ratio_db(lane, other)
                self.assertLessEqual(
                    leakage,
                    TARGETS.leakage_maximum_db,
                    f"{lane_name} already carries {leakage:.1f} dB of the other role",
                )

    def test_the_corpus_clears_the_leakage_gate_with_room_to_spare(self):
        """Clearing the gate is not by itself evidence of clean labels.

        Measured against this corpus, a lead lane carrying 30% of the rhythm
        still lands at about -12.4 dB and passes. The contract records that
        limit as an uncalibrated target, and this is the first evidence that
        it is permissive. Ours sits far below it; this pins how far, so a
        change to the synthesis that quietly muddied the labels would show up
        as a collapsing margin rather than as a still-passing gate.
        """
        leakage = cross_role_energy_ratio_db(self.example.lead, self.example.rhythm)

        self.assertLess(leakage, -30.0, f"the corpus only clears the gate by {leakage:.1f} dB")

    def test_a_track_with_no_lead_publishes_real_silence_and_still_reconstructs(self):
        example = render(DURATION, seed=11, with_lead=False)

        self.assertTrue(example.lead_is_absent)
        self.assertTrue(is_absent(example.lead, TARGETS))
        self.assertEqual(-np.inf, peak_dbfs(example.lead))
        self.assertTrue(is_audible(example.rhythm, TARGETS))
        # Absence is decided by reconstruction, never by the silent lane alone.
        self.assertEqual(
            np.inf, signal_to_residual_db(example.guitar_family, example.lead + example.rhythm)
        )

    def test_the_family_the_specialist_receives_is_exactly_its_two_sources(self):
        self.assertEqual(
            np.inf,
            signal_to_residual_db(
                self.example.guitar_family, self.example.lead + self.example.rhythm
            ),
        )

    def test_the_doubled_rhythm_is_two_takes_rather_than_one_copied(self):
        """A copied channel would be separable by stereo position alone.

        That is what the deterministic profile already does, and a specialist
        trained on it would learn position instead of role.
        """
        left, right = self.example.rhythm[:, 0], self.example.rhythm[:, 1]

        self.assertFalse(np.allclose(left, right))
        self.assertLess(abs(float(np.corrcoef(left, right)[0, 1])), 0.5)

    def test_a_centred_rhythm_render_breaks_the_position_convention_on_purpose(self):
        example = render(DURATION, seed=13, centred_rhythm=True)
        left, right = example.rhythm[:, 0], example.rhythm[:, 1]

        self.assertTrue(np.array_equal(left, right))
        self.assertTrue(is_audible(example.rhythm, TARGETS))

    def test_the_rhythm_breathes_between_hits_rather_than_ringing_throughout(self):
        """Palm muting is what a chugging riff is made of.

        A rhythm part that never stops sounding is a wall of sustain, and a
        model trained on it will not recognise the gaps a real riff has.
        """
        rhythm = self.example.rhythm

        sounding = float(np.mean(np.abs(rhythm) > 1e-4))

        self.assertLess(sounding, 0.95, "the riff never stops sounding")
        self.assertGreater(sounding, 0.20, "the riff barely sounds at all")

    def test_the_audio_contract_matches_what_the_specialist_must_emit(self):
        self.assertEqual(44_100, self.example.sample_rate)
        self.assertEqual(2, self.example.lead.shape[1])
        self.assertEqual(self.example.lead.shape, self.example.rhythm.shape)
        self.assertEqual(np.float32, self.example.lead.dtype)


class ReproducibilityTests(unittest.TestCase):
    def test_the_same_seed_renders_the_same_bytes(self):
        """A corpus that cannot be regenerated cannot be audited, and
        admission asks where the training data came from."""
        first = render(DURATION, seed=3)
        second = render(DURATION, seed=3)

        self.assertTrue(np.array_equal(first.lead, second.lead))
        self.assertTrue(np.array_equal(first.rhythm, second.rhythm))

    def test_different_seeds_render_different_material(self):
        first = render(DURATION, seed=3)
        other = render(DURATION, seed=4)

        self.assertFalse(np.array_equal(first.lead, other.lead))


if __name__ == "__main__":
    unittest.main()
