import tempfile
import unittest
from pathlib import Path

from cpaf.reproducibility import (create_unique_run_dir, require_fresh_output_dir,
                                  require_outputs_absent)


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.workspace_tmp = Path(__file__).parent / "_repro_tmp"
        self.workspace_tmp.mkdir(exist_ok=True)

    def test_fresh_output_guard_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory(dir=self.workspace_tmp) as tmp:
            root = Path(tmp)
            old = root / "old.csv"
            old.write_text("historical", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                require_fresh_output_dir(root)
            self.assertEqual(old.read_text(encoding="utf-8"), "historical")

    def test_unique_run_directory_keeps_repeated_outputs_separate(self):
        with tempfile.TemporaryDirectory(dir=self.workspace_tmp) as tmp:
            parent = Path(tmp) / "runs"
            first = create_unique_run_dir(parent, "sweep")
            marker = first / "result.csv"
            marker.write_text("first", encoding="utf-8")
            second = create_unique_run_dir(parent, "sweep")
            self.assertNotEqual(first, second)
            self.assertEqual(marker.read_text(encoding="utf-8"), "first")
            self.assertTrue(second.is_dir())

    def test_writer_guard_protects_matching_artifact(self):
        with tempfile.TemporaryDirectory(dir=self.workspace_tmp) as tmp:
            artifact = Path(tmp) / "privacy_sweep.csv"
            artifact.write_text("old result", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                require_outputs_absent(tmp, ("privacy_sweep.csv", "other.json"))
            self.assertEqual(artifact.read_text(encoding="utf-8"), "old result")


if __name__ == "__main__":
    unittest.main()
