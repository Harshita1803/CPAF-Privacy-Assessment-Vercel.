import tempfile
import unittest
from pathlib import Path
import json

import pandas as pd

from cpaf.experiments import pareto_frontier, regenerate_sweep_outputs, run_privacy_sweep


class SweepTests(unittest.TestCase):
    def test_pareto_frontier_dominance(self):
        frame = pd.DataFrame({
            "retained_records_pct": [100, 90, 95, 80],
            "worst_categorical_tv": [.8, .4, .9, .5],
            "config": ["A", "B", "C", "D"],
        })
        frontier = pareto_frontier(frame)
        self.assertEqual(set(frontier["config"]), {"A", "B"})

    def test_pareto_excludes_nonfinite_and_out_of_domain_objectives(self):
        frame = pd.DataFrame({
            "retained_records_pct": [90.0, float("inf"), 101.0, 80.0],
            "worst_categorical_tv": [.2, .1, .1, float("nan")],
            "config": ["valid", "infinite", "out-of-range", "missing"],
        })
        self.assertEqual(pareto_frontier(frame)["config"].tolist(), ["valid"])

    def test_sweep_rejects_fractional_integer_parameters(self):
        with self.assertRaisesRegex(ValueError, "k and l"):
            run_privacy_sweep(n=100, seeds=[7], k_values=[2.5], l_values=[2],
                              t_values=[.2], hierarchy_levels=[1])
        with self.assertRaisesRegex(ValueError, "seeds"):
            run_privacy_sweep(n=100, seeds=[7.5], k_values=[2], l_values=[2],
                              t_values=[.2], hierarchy_levels=[1])
        with self.assertRaisesRegex(ValueError, "n and every sweep grid"):
            run_privacy_sweep(n="not-a-number", seeds=[7], k_values=[2], l_values=[2],
                              t_values=[.2], hierarchy_levels=[1])
        with self.assertRaisesRegex(ValueError, "t values"):
            run_privacy_sweep(n=100, seeds=[7], k_values=[2], l_values=[2],
                              t_values=[float("nan")], hierarchy_levels=[1])

    def test_small_reproducible_sweep_exports_and_empty_release(self):
        workspace_test_tmp = Path(__file__).parent / "_sweep_tmp"
        workspace_test_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_test_tmp) as tmp:
            result, frontier = run_privacy_sweep(
                n=150, seeds=[7], k_values=[1, 500], l_values=[2],
                t_values=[.2], hierarchy_levels=[2], out_dir=tmp)
            self.assertEqual(len(result), 2)
            empty = result[result.k == 500].iloc[0]
            self.assertEqual(empty.suppressed_records, 150)
            self.assertTrue(pd.isna(empty.worst_categorical_tv))
            self.assertTrue((result.formal_privacy_guarantee == False).all())
            for name in ["privacy_sweep.csv", "privacy_sweep.json",
                         "privacy_sweep_manifest.json", "pareto_frontier.csv",
                         "pareto_frontier.json", "pareto_frontier.png"]:
                self.assertTrue((Path(tmp) / name).exists(), name)
            self.assertGreaterEqual(len(frontier), 1)
            manifest = json.loads((Path(tmp) / "privacy_sweep_manifest.json").read_text())
            self.assertEqual(manifest["output_dir"], str(Path(tmp).resolve()))
            self.assertIn("synthetic", manifest["dataset_provenance"])
            self.assertEqual(manifest["result_rows"], len(result))
            self.assertEqual(manifest["pareto_excluded_invalid_objective_rows"], 1)
            rebuilt = regenerate_sweep_outputs(tmp)
            self.assertEqual(set(rebuilt["seed"]), {7})
            self.assertTrue((Path(tmp) / "pareto_frontier.png").exists())


if __name__ == "__main__":
    unittest.main()
