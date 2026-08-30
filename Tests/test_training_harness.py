"""The harness has to be able to learn before anything it reports means much.

These need PyTorch and take a minute on a GPU, so they skip when it is absent
rather than failing a machine that was never meant to train.
"""

import tempfile
import unittest
from pathlib import Path

from RoleSpecialist.corpus.arrangement import TRAIN, VALIDATION
from RoleSpecialist.training.dataset import RenderedCorpus
from Tools.build_corpus import build

try:
    import torch

    from RoleSpecialist.training.evaluate import summarise
    from RoleSpecialist.training.model import SOURCES, build_model
    from RoleSpecialist.training.train import Epoch, evaluate, fit, overfit, selection_key

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
        """Run the real defaults, and assert a bar that was measured.

        Shortening this to be quick is how the check stops meaning anything:
        200 steps reduce the loss about ninefold and 400 reduce it about
        eightyfold, so a tenfold bar is a failure at one step count and a
        formality at the other. It runs at the defaults for that reason.
        """
        history = overfit(self.corpus, size=2, device=device())

        self.assertLess(
            history[-1],
            history[0] / 10.0,
            f"loss only moved from {history[0]:.4f} to {history[-1]:.4f}; "
            "the harness cannot learn two examples and nothing it reports later will mean anything",
        )
        self.assertLess(history[-1], history[len(history) // 2], "the loss stopped improving")

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


class SelectionTests(unittest.TestCase):
    """Selecting the checkpoint is not the same question as ranking the loss."""

    def setUp(self):
        if TORCH_ERROR is not None:
            raise unittest.SkipTest(f"PyTorch unavailable: {TORCH_ERROR}")

    def epoch(self, index: int, publishable: float, reconstruction: float):
        return Epoch(
            index=index,
            train_loss=0.1,
            seconds=1.0,
            report={"publishable": publishable, "median_reconstruction_db": reconstruction},
        )

    def test_a_run_where_nothing_publishes_still_keeps_its_best_model(self):
        """This is the whole early life of the project.

        Ranking on publishability alone leaves every epoch tied at zero, and
        the first one wins by default. A twenty-epoch run then saves the model
        from before it learned anything and calls it the best.
        """
        history = [
            self.epoch(0, 0.0, 4.6),
            self.epoch(10, 0.0, 17.5),
            self.epoch(19, 0.0, 20.1),
        ]

        self.assertEqual(19, max(history, key=selection_key).index)

    def test_publishability_still_outranks_reconstruction(self):
        history = [self.epoch(0, 0.0, 40.0), self.epoch(1, 0.25, 31.0)]

        self.assertEqual(1, max(history, key=selection_key).index)

    def test_a_degenerate_infinite_reconstruction_does_not_win_by_default(self):
        """A model that emits the mixture as one lane reconstructs perfectly
        and separates nothing."""
        history = [self.epoch(0, 0.0, float("inf")), self.epoch(1, 0.0, 25.0)]

        self.assertEqual(1, max(history, key=selection_key).index)


class FitTests(unittest.TestCase):
    """The loop has to select on publishability, not on the lowest loss."""

    @classmethod
    def setUpClass(cls):
        if TORCH_ERROR is not None:
            raise unittest.SkipTest(f"PyTorch unavailable: {TORCH_ERROR}")
        cls.root = Path(tempfile.mkdtemp())
        build(TRAIN, 3, 6.0, cls.root)
        build(VALIDATION, 2, 6.0, cls.root)
        cls.train_corpus = RenderedCorpus(cls.root / TRAIN)
        cls.validation_corpus = RenderedCorpus(cls.root / VALIDATION)

    def thresholds(self):
        from RoleSpecialist.vendor.role_metrics import RoleThresholds

        return RoleThresholds(
            calibrated=False,
            reconstruction_minimum_db=30.0,
            leakage_maximum_db=-12.0,
            audibility_minimum_dbfs=-40.0,
            absence_at_or_below_dbfs=-80.0,
        )

    def test_every_epoch_is_reported_against_the_gates(self):
        seen = []
        history = fit(
            self.train_corpus,
            self.validation_corpus,
            self.thresholds(),
            epochs=2,
            batch_size=1,
            steps_per_epoch=2,
            device=device(),
            on_epoch=seen.append,
        )

        self.assertEqual(2, len(history))
        self.assertEqual([0, 1], [epoch.index for epoch in seen])
        for epoch in history:
            self.assertIn("publishable", epoch.report)
            self.assertGreater(epoch.train_loss, 0.0)

    def test_the_saved_checkpoint_records_the_role_order_it_was_trained_in(self):
        """A checkpoint whose lane order is unknown is a permutation model, and
        the contract rejects those however well they score."""
        checkpoint = self.root / "checkpoint.pt"

        fit(
            self.train_corpus,
            self.validation_corpus,
            self.thresholds(),
            epochs=1,
            batch_size=1,
            steps_per_epoch=2,
            device=device(),
            checkpoint=checkpoint,
        )

        saved = torch.load(checkpoint, weights_only=False)
        self.assertEqual(list(SOURCES), saved["sources"])
        self.assertIn("epoch", saved)
        self.assertEqual(44_100, saved["sample_rate"])
        self.assertIn("report", saved)


if __name__ == "__main__":
    unittest.main()
