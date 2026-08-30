"""The pilot has to be auditable, not just audible."""

import json
import tempfile
import unittest
from pathlib import Path

import soundfile as sf

from Tools.render_pilot import CASES, SUBTYPE, main


class RenderPilotTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.assertEqual(0, main(["--seconds", "0.5", "--out", str(self.root)]))

    def test_every_case_writes_the_family_and_both_labels(self):
        for case in CASES:
            with self.subTest(case=case.name):
                folder = self.root / case.name
                for name in ("guitar_family.wav", "lead_guitar.wav", "rhythm_guitar.wav"):
                    self.assertTrue((folder / name).exists(), f"{case.name}/{name} is missing")

    def test_the_corpus_is_written_without_a_quantisation_floor(self):
        """Training truth at 16 bits would carry a floor the sources never had."""
        info = sf.info(self.root / CASES[0].name / "guitar_family.wav")

        self.assertEqual(SUBTYPE, info.subtype)
        self.assertEqual(2, info.channels)
        self.assertEqual(44_100, info.samplerate)

    def test_each_example_records_the_seed_that_regenerates_it(self):
        for case in CASES:
            with self.subTest(case=case.name):
                record = json.loads((self.root / case.name / "example.json").read_text())
                self.assertEqual(case.seed, record["seed"])
                self.assertEqual(case.with_lead, record["with_lead"])
                self.assertIn("reconstruction_db", record["measurements"])

    def test_the_rhythm_only_case_writes_a_genuinely_silent_lead(self):
        record = json.loads((self.root / "02-rhythm-only" / "example.json").read_text())

        self.assertEqual("-inf", record["measurements"]["lead_peak_dbfs"])
        self.assertNotEqual("-inf", record["measurements"]["rhythm_peak_dbfs"])

    def test_the_manifest_lists_every_case(self):
        manifest = json.loads((self.root / "pilot.json").read_text())

        self.assertEqual([case.name for case in CASES], [e["name"] for e in manifest["examples"]])


if __name__ == "__main__":
    unittest.main()
