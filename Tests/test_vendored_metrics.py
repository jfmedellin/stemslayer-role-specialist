"""The shared definitions must be a copy, never a rewrite.

Admission decides whether a checkpoint is publishable using these exact
functions. If this repository edited its copy to be more forgiving, a
checkpoint could train and evaluate clean here and still be rejected there,
or worse, be admitted on measurements the app never agreed to.
"""

import hashlib
import json
import unittest
from pathlib import Path

VENDOR = Path(__file__).resolve().parent.parent / "RoleSpecialist" / "vendor"


class VendoredMetricsTests(unittest.TestCase):
    def setUp(self):
        self.provenance = json.loads((VENDOR / "PROVENANCE.json").read_text(encoding="utf-8"))

    def test_every_vendored_file_matches_its_recorded_digest(self):
        for name, entry in self.provenance["files"].items():
            with self.subTest(file=name):
                digest = hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()
                self.assertEqual(
                    entry["sha256"],
                    digest,
                    f"{name} no longer matches the copy taken from "
                    f"{self.provenance['source_tag']}; refresh it instead of editing it",
                )

    def test_the_provenance_names_where_the_copy_came_from(self):
        self.assertTrue(self.provenance["source_repository"].startswith("https://"))
        self.assertRegex(self.provenance["source_tag"], r"^v\d+\.\d+\.\d+$")
        for entry in self.provenance["files"].values():
            self.assertTrue(entry["origin_path"])

    def test_the_thresholds_are_still_uncalibrated(self):
        """Training against calibrated-looking targets that nobody measured
        would produce evidence the admission harness must reject anyway."""
        payload = json.loads((VENDOR / "thresholds.json").read_text(encoding="utf-8"))

        self.assertFalse(payload["calibrated"])
        self.assertTrue(payload["calibration_blocker"])


if __name__ == "__main__":
    unittest.main()
