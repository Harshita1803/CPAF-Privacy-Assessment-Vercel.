import unittest
import pandas as pd

from cpaf.core import generate_synthetic, validate_dataset, generalize, suppress_k, privacy_diagnostics, microaggregate


class CoreTests(unittest.TestCase):
    def test_reproducible_valid_data(self):
        a, b = generate_synthetic(200, 11), generate_synthetic(200, 11)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(validate_dataset(a)["rows"], 200)

    def test_invalid_generation(self):
        for n, seed in [(0, 1), (1, -1)]:
            with self.assertRaises(ValueError):
                generate_synthetic(n, seed)

    def test_k_suppression_is_validated(self):
        frame = pd.DataFrame({"q": ["a", "a", "b"], "risk_category": ["x", "y", "z"]})
        out, d = suppress_k(frame, ["q"], 2)
        self.assertEqual(len(out), 2)
        self.assertTrue(d["k_satisfied"])
        self.assertEqual(d["suppressed_records"], 1)

    def test_l_and_t_diagnostics(self):
        frame = pd.DataFrame({"q": ["a"] * 4, "s": ["x", "x", "y", "y"]})
        d = privacy_diagnostics(frame, ["q"], "s", .5)
        self.assertEqual(d["distinct_l"], 2)
        self.assertEqual(d["worst_categorical_tv"], 0)
        self.assertTrue(d["l_pass"])
        self.assertTrue(d["t_pass"])
        self.assertTrue(d["diagnostic_only"])

    def test_missing_quasi_value_is_an_explicit_group(self):
        frame = pd.DataFrame({"q": [None, None, "a"], "s": ["x", "y", "x"]})
        d = privacy_diagnostics(frame, ["q"], "s", .5, l_threshold=2)
        self.assertEqual(d["groups"], 2)
        self.assertEqual(d["distinct_l"], 1)
        self.assertFalse(d["l_pass"])

    def test_missing_sensitive_value_is_rejected(self):
        frame = pd.DataFrame({"q": ["a", "a"], "s": ["x", None]})
        with self.assertRaisesRegex(ValueError, "sensitive values must not be missing"):
            privacy_diagnostics(frame, ["q"], "s")

    def test_empty_release_and_small_k_suppression(self):
        frame = pd.DataFrame({"q": ["a", "b"], "s": ["x", "y"]})
        released, diag = suppress_k(frame, ["q"], 3)
        self.assertTrue(released.empty)
        self.assertEqual(diag["suppressed_records"], 2)
        self.assertTrue(diag["k_satisfied"])
        empty_diag = privacy_diagnostics(released, ["q"], "s")
        self.assertEqual(empty_diag["groups"], 0)
        self.assertFalse(empty_diag["l_pass"])

    def test_microaggregation_partial_block(self):
        frame = pd.DataFrame({"x": [4., 1., 3., 2., 10.]})
        out, d = microaggregate(frame, ["x"], 2)
        self.assertEqual(len(out), len(frame))
        self.assertIn("x", d["rmse"])

    def test_microaggregation_integer_age_preserves_fractional_mean(self):
        frame = pd.DataFrame({"age": [18, 19, 19, 19, 19], "integer_measure": [1, 2, 3, 4, 5]})
        out, diagnostics = microaggregate(frame, ["age", "integer_measure"], group_size=5.0)
        self.assertTrue(pd.api.types.is_float_dtype(out["age"].dtype))
        self.assertTrue(pd.api.types.is_float_dtype(out["integer_measure"].dtype))
        self.assertTrue((out["age"] == 18.8).all())
        self.assertTrue((out["integer_measure"] == 3.0).all())
        self.assertEqual(diagnostics["group_size"], 5)
        self.assertAlmostEqual(diagnostics["rmse"]["age"], 0.4)
        self.assertEqual(frame["age"].dtype, "int64")  # input is not mutated

    def test_microaggregation_rejects_nonfinite_values_and_invalid_group_sizes(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite values"):
                microaggregate(pd.DataFrame({"x": [1.0, value]}), ["x"], 2)
        frame = pd.DataFrame({"x": [1, 2, 3]})
        for group_size in (True, 1, 0, -2, 2.5, float("nan"), float("inf"), "2"):
            with self.subTest(group_size=group_size), self.assertRaises(ValueError):
                microaggregate(frame, ["x"], group_size)

    def test_microaggregation_rejects_empty_missing_and_invalid_group_size(self):
        with self.assertRaisesRegex(ValueError, "finite values"):
            microaggregate(pd.DataFrame({"x": []}), ["x"], 2)
        with self.assertRaisesRegex(ValueError, "finite values"):
            microaggregate(pd.DataFrame({"x": [1.0, float("nan")]}), ["x"], 2)
        with self.assertRaises(ValueError):
            microaggregate(pd.DataFrame({"x": [1.0, 2.0]}), ["x"], 2.5)

    def test_schema_rejects_duplicate_identifier(self):
        frame = generate_synthetic(10)
        frame.loc[1, "record_id"] = frame.loc[0, "record_id"]
        with self.assertRaises(ValueError):
            validate_dataset(frame)


if __name__ == "__main__":
    unittest.main()
