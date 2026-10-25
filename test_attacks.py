import tempfile
import unittest
from pathlib import Path
import json

import pandas as pd
from cpaf.attacks import run_shadow_mia, synthetic_disclosure_proxies


class AttackTests(unittest.TestCase):
    def test_shadow_attack_partitions_and_outputs(self):
        local = Path(__file__).parent / "_attack_tmp"
        local.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as tmp:
            results, proxies = run_shadow_mia(n=3000, seed=19, k=2, output_dir=tmp)
            self.assertEqual(set(results.representation), {"Raw", "Generalized", "Generalized+k=2"})
            self.assertTrue(results.partition_sets_pairwise_disjoint.all())
            self.assertTrue(results.attack_test_auc.between(0, 1).all())
            self.assertTrue(results.attack_auc_bootstrap_95ci_low.le(results.attack_test_auc).all())
            self.assertTrue(results.attack_auc_bootstrap_95ci_high.ge(results.attack_test_auc).all())
            self.assertTrue((results.attack_auc_bootstrap_replicates == 1000).all())
            self.assertTrue((results.roc_auc_chance_baseline == .5).all())
            self.assertTrue(proxies["synthetic_only"])
            self.assertIn("nearest_neighbor_distance_median", proxies)
            for name in ["shadow_mia_results.csv", "shadow_mia_results.json",
                         "shadow_mia_manifest.json", "synthetic_disclosure_proxies.json"]:
                self.assertTrue((Path(tmp) / name).exists(), name)
            manifest = json.loads((Path(tmp) / "shadow_mia_manifest.json").read_text())
            self.assertEqual(manifest["label_seed"], 20)
            self.assertEqual(Path(manifest["output_dir"]).resolve(), Path(tmp).resolve())

    def test_empty_disclosure_proxy_is_defined(self):
        empty = pd.DataFrame(columns=["record_id", "age", "region", "city",
                                      "merchant_type", "amount_inr", "hour"])
        self.assertEqual(synthetic_disclosure_proxies(empty)["rows"], 0)


if __name__ == "__main__":
    unittest.main()
