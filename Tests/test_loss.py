"""The silence term has to push only where silence is correct.

The danger it introduces is the mirror of the problem it solves: a penalty on
quiet lanes that applied everywhere would teach the model that emitting nothing
is always safe, and Stemslayer's contract calls a silent lane that loses energy
a failure.
"""

import unittest

try:
    import torch

    from RoleSpecialist.training.loss import (
        DEFAULT_MARGIN_DB,
        lane_level_db,
        separation_loss,
        silence_penalty,
        silent_lane_mask,
    )

    TORCH_ERROR = None
except Exception as error:  # pragma: no cover - exercised only without PyTorch
    TORCH_ERROR = error

CANDIDATE = -40.0
FRAMES = 4_096


class SilencePenaltyTests(unittest.TestCase):
    def setUp(self):
        if TORCH_ERROR is not None:
            raise unittest.SkipTest(f"PyTorch unavailable: {TORCH_ERROR}")
        torch.manual_seed(0)

    def lanes(self, lead_amplitude: float, rhythm_amplitude: float = 0.5):
        """Return one batch of (lead, rhythm) at chosen levels."""
        lead = lead_amplitude * torch.ones(1, 1, 2, FRAMES)
        rhythm = rhythm_amplitude * torch.ones(1, 1, 2, FRAMES)
        return torch.cat([lead, rhythm], dim=1)

    def test_a_lane_whose_target_has_content_is_never_penalised(self):
        """Otherwise the model learns that emitting nothing always pays."""
        targets = self.lanes(0.5)
        estimate = self.lanes(1e-9)

        self.assertEqual(0.0, float(silence_penalty(estimate, targets, candidate_dbfs=CANDIDATE)))

    def test_energy_in_a_lane_that_should_be_silent_costs_something(self):
        targets = self.lanes(0.0)
        loud = self.lanes(0.5)

        self.assertGreater(float(silence_penalty(loud, targets, candidate_dbfs=CANDIDATE)), 0.0)

    def test_the_penalty_stops_once_the_lane_is_quiet_enough(self):
        targets = self.lanes(0.0)
        below_the_floor = self.lanes(10 ** ((CANDIDATE - DEFAULT_MARGIN_DB - 20.0) / 20.0))

        self.assertEqual(
            0.0, float(silence_penalty(below_the_floor, targets, candidate_dbfs=CANDIDATE))
        )

    def test_the_push_does_not_fade_as_the_lane_gets_quieter(self):
        """This is the whole reason the term exists.

        An amplitude-domain term stops providing gradient long before the gate
        is reached, which is how a run sat at -23 dBFS for a hundred epochs
        while its loss kept falling.
        """
        targets = self.lanes(0.0)
        # Both levels sit above the candidate threshold, so both are still
        # being pushed. Straddling it would measure the cutoff instead.
        loud = self.lanes(10 ** (-20.0 / 20.0))
        quieter = self.lanes(10 ** (-30.0 / 20.0))

        gap = float(silence_penalty(loud, targets, candidate_dbfs=CANDIDATE)) - float(
            silence_penalty(quieter, targets, candidate_dbfs=CANDIDATE)
        )

        # Ten decibels of difference must still cost ten decibels' worth.
        self.assertAlmostEqual(10.0, gap / 0.001, places=1)

    def test_only_the_silent_lane_of_a_mixed_example_is_counted(self):
        targets = torch.cat(
            [torch.zeros(1, 1, 2, FRAMES), 0.5 * torch.ones(1, 1, 2, FRAMES)], dim=1
        )
        estimate = 0.5 * torch.ones(1, 2, 2, FRAMES)

        both_silent = silence_penalty(estimate, torch.zeros_like(targets), candidate_dbfs=CANDIDATE)
        one_silent = silence_penalty(estimate, targets, candidate_dbfs=CANDIDATE)

        self.assertAlmostEqual(float(both_silent), float(one_silent), places=5)
        self.assertEqual(1, int(silent_lane_mask(targets).sum()))


class LevelTests(unittest.TestCase):
    def setUp(self):
        if TORCH_ERROR is not None:
            raise unittest.SkipTest(f"PyTorch unavailable: {TORCH_ERROR}")

    def test_a_full_scale_lane_reads_as_zero_dbfs(self):
        self.assertAlmostEqual(0.0, float(lane_level_db(torch.ones(1, 1, 2, FRAMES))[0, 0]), places=4)

    def test_digital_silence_reads_far_below_the_floor(self):
        self.assertLess(float(lane_level_db(torch.zeros(1, 1, 2, FRAMES))[0, 0]), CANDIDATE)


class SeparationLossTests(unittest.TestCase):
    def setUp(self):
        if TORCH_ERROR is not None:
            raise unittest.SkipTest(f"PyTorch unavailable: {TORCH_ERROR}")

    def test_the_loss_reports_both_parts_so_a_run_can_watch_them_separately(self):
        targets = torch.cat(
            [torch.zeros(1, 1, 2, FRAMES), 0.5 * torch.ones(1, 1, 2, FRAMES)], dim=1
        )
        estimate = 0.4 * torch.ones(1, 2, 2, FRAMES)

        total, reconstruction, silence = separation_loss(estimate, targets, candidate_dbfs=CANDIDATE)

        self.assertGreater(reconstruction, 0.0)
        self.assertGreater(silence, 0.0)
        self.assertAlmostEqual(reconstruction + silence, float(total), places=6)

    def test_a_perfect_estimate_costs_nothing_on_either_term(self):
        targets = torch.cat(
            [torch.zeros(1, 1, 2, FRAMES), 0.5 * torch.ones(1, 1, 2, FRAMES)], dim=1
        )

        total, reconstruction, silence = separation_loss(targets, targets, candidate_dbfs=CANDIDATE)

        self.assertEqual(0.0, reconstruction)
        self.assertEqual(0.0, silence)
        self.assertEqual(0.0, float(total))

    def test_the_silent_term_carries_gradient_back_to_the_estimate(self):
        targets = torch.zeros(1, 2, 2, FRAMES)
        estimate = (0.1 * torch.ones(1, 2, 2, FRAMES)).requires_grad_(True)

        silence_penalty(estimate, targets, candidate_dbfs=CANDIDATE).backward()

        self.assertIsNotNone(estimate.grad)
        self.assertGreater(float(estimate.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
