import tempfile
import unittest
from pathlib import Path
import json

import pandas as pd

from cpaf.evaluation import run_repeated_utility


class RepeatedEvaluationTests(unittest.TestCase):
    def test_repeated_metrics_fairness_and_exports(self):
        workspace_tmp = Path(__file__).parent / "_eval_tmp"
        workspace_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_tmp) as tmp:
            runs, summary = run_repeated_utility(n=500, seed=31, repeats=2, k=2, output_dir=tmp)
            self.assertEqual(set(runs.representation), {"Raw", "Generalized", "Generalized+k=2"})
            self.assertTrue(runs.roc_auc.between(0, 1).all())
            self.assertTrue({"roc_auc_mean", "roc_auc_std", "pr_auc_mean", "brier_mean"}.issubset(summary))
            fairness = pd.read_csv(Path(tmp) / "synthetic_age_fairness.csv")
            self.assertTrue({"positive_n", "negative_n", "small_subgroup_warning"}.issubset(fairness.columns))
            self.assertTrue(fairness.positive_n.ge(0).all() and fairness.negative_n.ge(0).all())
            for name in ["repeated_utility_runs.csv", "repeated_utility_summary.csv",
                         "synthetic_age_fairness.csv", "synthetic_age_fairness_gaps.csv",
                         "suppression_distribution_drift.csv", "repeated_utility_manifest.json"]:
                self.assertTrue((Path(tmp) / name).exists(), name)
            manifest = json.loads((Path(tmp) / "repeated_utility_manifest.json").read_text())
            self.assertEqual(manifest["label_seed"], 32)
            self.assertIn("synthetic", manifest["dataset_provenance"])
            self.assertEqual(Path(manifest["output_dir"]).resolve(), Path(tmp).resolve())


if __name__ == "__main__":
    unittest.main()
