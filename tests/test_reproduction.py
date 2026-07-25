from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from reproduce_headline import compute_headline_metrics


class HeadlineReproductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = compute_headline_metrics(
            data_dir=ROOT / "data",
            model_dir=ROOT / "models",
            device="cpu",
        )

    def assertMetricAlmostEqual(
        self, key: str, expected: float, places: int = 2
    ) -> None:
        self.assertAlmostEqual(self.metrics[key], expected, places=places)

    def test_primary_test_set_errors_match_the_manuscript(self) -> None:
        self.assertMetricAlmostEqual("n_profile_median_rel_l2_pct", 3.19)
        self.assertMetricAlmostEqual("m_profile_median_rel_l2_pct", 9.45)
        self.assertMetricAlmostEqual("sigma_t_profile_median_rel_l2_pct", 3.40)

    def test_secondary_peak_and_cracking_checks_match(self) -> None:
        self.assertMetricAlmostEqual("sigma_t_peak_median_rel_error_pct", 3.71)
        self.assertMetricAlmostEqual(
            "cracking_extent_median_abs_error_percentage_points", 1.00
        )


if __name__ == "__main__":
    unittest.main()
