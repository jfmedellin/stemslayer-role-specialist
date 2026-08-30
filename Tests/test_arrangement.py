"""What a corpus has to vary, and what the two splits must never share.

The failure these exist to prevent is a corpus that looks varied and is not.
Rendering one riff a thousand times with different excitation noise produces a
thousand decorrelated waveforms and one example.
"""

import unittest

from RoleSpecialist.corpus.arrangement import (
    TRAIN,
    TRAIN_RIFFS,
    TRAIN_SCALES,
    VALIDATION,
    VALIDATION_RIFFS,
    VALIDATION_SCALES,
    sample,
)

POPULATION = 200


class SplitTests(unittest.TestCase):
    def test_the_two_splits_share_no_riff_and_no_scale(self):
        """Held-out seeds are not a split.

        If a riff trains and also validates, the validation score measures how
        well the model memorised it.
        """
        self.assertEqual(set(), set(TRAIN_RIFFS) & set(VALIDATION_RIFFS))
        self.assertEqual(set(), set(TRAIN_SCALES) & set(VALIDATION_SCALES))

    def test_no_drawn_riff_or_scale_reaches_both_splits(self):
        """Compare the memorisable part, not the whole arrangement.

        Tempo and key are continuous, so two draws never collide on a full
        identity even when both splits pull from the same riff pool. Comparing
        identities therefore passes while the split leaks. What a model can
        memorise across the split is the riff and the scale, so those are what
        must not appear on both sides.
        """
        train = {(a.riff, a.scale) for a in (sample(TRAIN, i) for i in range(POPULATION))}
        validation = {
            (a.riff, a.scale) for a in (sample(VALIDATION, i) for i in range(POPULATION))
        }

        self.assertEqual(set(), train & validation)
        self.assertEqual(set(), {r for r, _ in train} & {r for r, _ in validation})

    def test_a_split_only_draws_from_its_own_pools(self):
        for split, riffs, scales in (
            (TRAIN, TRAIN_RIFFS, TRAIN_SCALES),
            (VALIDATION, VALIDATION_RIFFS, VALIDATION_SCALES),
        ):
            for index in range(50):
                drawn = sample(split, index)
                with self.subTest(split=split, index=index):
                    self.assertIn(drawn.riff, riffs)
                    self.assertIn(drawn.scale, scales)

    def test_an_unknown_split_is_refused(self):
        with self.assertRaises(ValueError):
            sample("test", 0)


class VariationTests(unittest.TestCase):
    def setUp(self):
        self.drawn = [sample(TRAIN, index) for index in range(POPULATION)]

    def test_the_corpus_is_not_one_example_rendered_many_times(self):
        identities = {arrangement.identity for arrangement in self.drawn}

        self.assertGreater(
            len(identities),
            POPULATION * 0.9,
            f"{POPULATION} draws collapsed into {len(identities)} distinct arrangements",
        )

    def test_every_riff_and_every_scale_is_actually_reached(self):
        self.assertEqual(set(TRAIN_RIFFS), {a.riff for a in self.drawn})
        self.assertEqual(set(TRAIN_SCALES), {a.scale for a in self.drawn})

    def test_tempo_and_key_span_their_ranges_rather_than_clustering(self):
        tempos = [a.tempo_bpm for a in self.drawn]
        roots = [a.root_hz for a in self.drawn]

        self.assertGreater(max(tempos) - min(tempos), 90.0)
        self.assertGreater(max(roots) - min(roots), 35.0)

    def test_the_corpus_contains_tracks_with_no_lead(self):
        """A corpus of nothing but lead-bearing tracks teaches the model that a
        silent lead lane is always an error."""
        without = [a for a in self.drawn if not a.with_lead]

        self.assertGreater(len(without), POPULATION * 0.10)
        self.assertLess(len(without), POPULATION * 0.45)

    def test_the_corpus_contains_rhythm_that_position_cannot_find(self):
        """Stemslayer already splits by stereo position. A specialist that
        learned position instead of role would add nothing."""
        centred = [a for a in self.drawn if a.centred_rhythm]

        self.assertGreater(len(centred), POPULATION * 0.10)
        self.assertLess(len(centred), POPULATION * 0.45)

    def test_excitation_noise_alone_does_not_count_as_a_distinct_example(self):
        """Two renders differing only by seed are the same example.

        The identity deliberately excludes the seed, because a corpus that
        counts noise as variation is how one riff passes for a thousand.
        """
        first = sample(TRAIN, 0)
        same_music = type(first)(**{**first.__dict__, "seed": first.seed + 1})

        self.assertEqual(first.identity, same_music.identity)
        self.assertNotEqual(first.seed, same_music.seed)


class DeterminismTests(unittest.TestCase):
    def test_the_same_position_always_draws_the_same_arrangement(self):
        """A corpus is regenerated from its manifest, never stored, so a
        checkpoint's training data has to be reproducible from an index."""
        self.assertEqual(sample(TRAIN, 42), sample(TRAIN, 42))

    def test_neighbouring_positions_draw_different_arrangements(self):
        self.assertNotEqual(sample(TRAIN, 42).identity, sample(TRAIN, 43).identity)


if __name__ == "__main__":
    unittest.main()
