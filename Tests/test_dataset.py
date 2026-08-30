"""A window has to be three views of one moment, not three moments."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from Tools.build_corpus import build
from RoleSpecialist.corpus.arrangement import TRAIN
from RoleSpecialist.training.dataset import RenderedCorpus

SECONDS = 1.0


class RenderedCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        build(TRAIN, 4, SECONDS, cls.root)
        cls.corpus = RenderedCorpus(cls.root / TRAIN)

    def test_it_reads_back_every_example_it_rendered(self):
        self.assertEqual(4, len(self.corpus))
        self.assertEqual(44_100, self.corpus.sample_rate)

    def test_a_crop_takes_the_same_window_from_the_mixture_and_both_labels(self):
        """Cropping the labels independently teaches the model to invent."""
        frames = 4_096
        mixture, lead, rhythm = self.corpus.crop(0, frames, offset=1_000)

        self.assertEqual((frames, 2), mixture.shape)
        self.assertEqual(mixture.shape, lead.shape)
        self.assertEqual(mixture.shape, rhythm.shape)
        self.assertTrue(np.allclose(mixture, lead + rhythm, atol=1e-6))

    def test_a_window_longer_than_the_example_is_refused(self):
        with self.assertRaises(ValueError):
            self.corpus.crop(0, int(SECONDS * 44_100) + 1)

    def test_random_offsets_stay_inside_the_example(self):
        rng = np.random.default_rng(0)
        frames = 8_192
        for _ in range(20):
            mixture, _lead, _rhythm = self.corpus.crop(0, frames, rng=rng)
            self.assertEqual(frames, mixture.shape[0])

    def test_a_missing_manifest_says_how_to_build_one(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError) as raised:
                RenderedCorpus(empty)

        self.assertIn("build_corpus", str(raised.exception))

    def test_the_manifest_records_which_examples_have_no_lead(self):
        flags = [example.with_lead for example in self.corpus.examples]

        self.assertEqual(4, len(flags))
        self.assertTrue(all(isinstance(flag, bool) for flag in flags))


if __name__ == "__main__":
    unittest.main()
