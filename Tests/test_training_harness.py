"""The harness has to be able to learn before anything it reports means much.

These need PyTorch and take a minute on a GPU, so they skip when it is absent
rather than failing a machine that was never meant to train.
"""

import tempfile
import unittest
from pathlib import Path

from RoleSpecialist.corpus.arrangement import TRAIN
from RoleSpecialist.training.dataset import RenderedCorpus
from Tools.build_corpus import build

try:
    import torch

    from RoleSpecialist.training.evaluate import summarise
    from RoleSpecialist.training.model import SOURCES, build_model
    from RoleSpecialist.training.train import evaluate, overfit

    TORCH_ERROR = None
except Exception as error:  # pragma: no cover - exercised only without PyTorch
    TORCH_ERROR = error


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class ModelTests(unittest.TestCase):
    def setUp(self):
        if TORCH_ERROR is not None:
            raise unittest.SkipTest(f"PyTorch unavailable: {TORCH_ERROR}")

    def test_the_model_emits_the_two_roles_in_a_fixed_order(self):
        """The contract requires stable roles, not a permutation. Lane identity
        has to mean the same thing on every run and every track."""
        model = build_model()

        self.assertEqual(["lead_guitar", "rhythm_guitar"], list(model.sources))
        self.assertEqual(0, SOURCES.index("lead_guitar"))

    def test_it_returns_one_estimate_per_role_shaped_like_the_input(self):
        model = build_model()
        frames = 44_100 * 2
        mixture = torch.zeros(1, 2, frames)

        with torch.no_grad():
            predicted = model(mixture)

        self.assertEqual((1, len(SOURCES), 2, frames), tuple(predicted.shape))


class OverfitTests(unittest.TestCase):
    """The cheapest evidence that data, model, loss and optimiser are wired
    to each other rather than merely present."""

    @classmethod
    def setUpClass(cls):
        if TORCH_ERROR is not None:
            raise unittest.SkipTest(f"PyTorch unavailable: {TORCH_ERROR}")
        cls.root = Path(tempfile.mkdtemp())
        build(TRAIN, 2, 6.0, cls.root)
        cls.corpus = RenderedCorpus(cls.root / TRAIN)

    def test_the_harness_can_drive_a_fixed_batch_toward_zero(self):
        history = overfit(self.corpus, steps=200, size=2, device=device())

        self.assertLess(
            history[-1],
            history[0] / 10.0,
            f"loss only moved from {history[0]:.4f} to {history[-1]:.4f}; "
            "the harness cannot learn two examples and nothing it reports later will mean anything",
        )

    def test_evaluation_reports_the_contract_gates_rather_than_loss(self):
        from RoleSpecialist.vendor.role_metrics import RoleThresholds

        thresholds = RoleThresholds(
            calibrated=False,
            reconstruction_minimum_db=30.0,
            leakage_maximum_db=-12.0,
            audibility_minimum_dbfs=-40.0,
            absence_at_or_below_dbfs=-80.0,
        )
        report = evaluate(build_model().to(device()), self.corpus, thresholds, limit=2, device=device())

        self.assertEqual(2, report["examples"])
        for key in ("publishable", "median_reconstruction_db", "worst_leakage_db", "absence_recalled"):
            self.assertIn(key, report)

    def test_an_untrained_model_is_not_publishable(self):
        """If noise passed the gates, the gates would not be measuring anything."""
        from RoleSpecialist.vendor.role_metrics import RoleThresholds

        thresholds = RoleThresholds(
            calibrated=False,
            reconstruction_minimum_db=30.0,
            leakage_maximum_db=-12.0,
            audibility_minimum_dbfs=-40.0,
            absence_at_or_below_dbfs=-80.0,
        )

        report = evaluate(build_model().to(device()), self.corpus, thresholds, limit=2, device=device())

        self.assertEqual(0.0, report["publishable"])


if __name__ == "__main__":
    unittest.main()
