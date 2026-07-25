from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from reproduce_statistics import compute_statistics, validate_analysis_tree, write_source_figures


class AnalysisReleaseTest(unittest.TestCase):
    def test_all_released_analysis_files_have_safe_schema(self):
        self.assertEqual(validate_analysis_tree(ROOT / "data"), [])

    def test_baseline_and_ablation_values_round_to_paper_values(self):
        result = compute_statistics(ROOT / "data")
        self.assertAlmostEqual(result["baseline"]["N_median_pct"], 8.41, places=2)
        self.assertAlmostEqual(result["baseline"]["M_median_pct"], 53.35, places=2)
        self.assertAlmostEqual(result["baseline"]["M_wilcoxon_p"], 3.62396240234375e-05)
        expected = {
            "primary_k32": (9.7, 0.5, 3.2, 0.2, 12.6, 1.4),
            "parameters_pca_k32": (13.6, 0.7, 4.1, 0.5, 19.3, 2.1),
            "parameters_pca_k16": (14.0, 1.1, 4.3, 0.2, 20.0, 1.9),
            "pca_only_k32": (21.6, 0.6, 7.7, 0.9, 50.4, 6.4),
        }
        for name, want in expected.items():
            got = result["ablations"][name]
            observed = tuple(round(got[k], 1) for k in (
                "slope_mean_pct", "slope_std_pct", "N_mean_pct",
                "N_std_pct", "M_mean_pct", "M_std_pct"))
            self.assertEqual(observed, want)

    def test_alternative_split_ranges_round_to_paper_values(self):
        ranges = compute_statistics(ROOT / "data")["alternative_splits"]["ranges"]
        self.assertEqual(round(ranges["M_min_pct"], 1), 6.6)
        self.assertEqual(round(ranges["M_max_pct"], 1), 9.7)
        self.assertEqual(round(ranges["N_min_pct"], 1), 2.0)
        self.assertEqual(round(ranges["N_max_pct"], 1), 2.6)
        self.assertEqual(round(ranges["sigma_profile_min_pct"], 1), 2.5)
        self.assertEqual(round(ranges["sigma_profile_max_pct"], 1), 3.5)
        self.assertEqual(round(ranges["sigma_peak_max_pct"], 1), 2.8)
        self.assertEqual(round(ranges["cracking_extent_max_pp"], 1), 1.0)

    def test_four_source_figures_are_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_figures(ROOT / "data", Path(tmp))
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))
