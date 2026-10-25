import hashlib
import unittest
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from cpaf.evidence import REQUIRED_EVIDENCE_COLUMNS, read_evidence_csv
from cpaf_advanced import (APPS, DIMENSIONS, compute_app_metrics,
                           format_evidence_for_display, validate_evidence_rows)


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "cpaf_outputs" / "evidence_reviews" / "evidence_register_reviewed.csv"
TEMPLATE_PATH = ROOT / "cpaf_outputs" / "evidence_register_template.csv"
TEMPLATE_SHA256_AT_REVIEW_START = "1ab16c7f5cddf467100d9015528e621c8be4586a07ebdd342cc603b44981932b"
EXPECTED_UNSCORED = {
    ("Paytm", "Anonymisation & Re-identification"),
    ("Paytm", "DPDP Alignment"),
    ("Google Pay", "DPDP Alignment"),
    ("PhonePe", "DPDP Alignment"),
}


class ResearchedEvidenceRegisterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = read_evidence_csv(REVIEW_PATH)

    def test_all_21_provider_dimension_pairs_exist_once(self):
        expected = {(app, dimension) for app in APPS for dimension in DIMENSIONS}
        pairs = list(zip(self.frame.app, self.frame.dimension))
        self.assertEqual(len(pairs), 21)
        self.assertEqual(len(set(pairs)), 21)
        self.assertEqual(set(pairs), expected)
        self.assertTrue(set(REQUIRED_EVIDENCE_COLUMNS).issubset(self.frame.columns))

    def test_source_urls_and_scored_values_are_valid(self):
        self.assertTrue(self.frame.source_url.notna().all())
        for raw_url in self.frame.source_url.astype(str):
            url = urlparse(raw_url.strip())
            self.assertIn(url.scheme, {"http", "https"})
            self.assertTrue(url.netloc)

        scores = pd.to_numeric(self.frame.raw_score, errors="coerce")
        strengths = pd.to_numeric(self.frame.evidence_strength, errors="coerce")
        scored = scores.notna()
        self.assertEqual(int(scored.sum()), 17)
        self.assertTrue(scores[scored].between(1, 5).all())
        self.assertTrue(strengths[scored].between(0, 1).all())
        self.assertTrue(self.frame.loc[scored, "claim"].astype(str).str.strip().ne("").all())
        self.assertTrue(self.frame.loc[scored, "review_status"].eq("reviewed").all())
        for column in ("source_title", "source_url", "access_date", "reviewer", "score_rationale"):
            self.assertTrue(self.frame.loc[scored, column].notna().all(), column)
            self.assertTrue(self.frame.loc[scored, column].astype(str).str.strip().ne("").all(), column)
        eligible, _ = validate_evidence_rows(self.frame)
        self.assertEqual(int(eligible.sum()), int(scored.sum()))
        scores_by_provider = compute_app_metrics(self.frame)
        self.assertTrue(scores_by_provider["Weighted CPI"].notna().any())

    def test_unsupported_rows_remain_unscored_and_are_explained(self):
        scores = pd.to_numeric(self.frame.raw_score, errors="coerce")
        unscored = self.frame.loc[~scores.notna()]
        actual = set(zip(unscored.app, unscored.dimension))
        self.assertEqual(actual, EXPECTED_UNSCORED)
        self.assertEqual(len(unscored), 4)
        self.assertTrue(unscored.notes.astype(str).str.contains("Unsupported", case=False).all())
        self.assertTrue(unscored.score_rationale.astype(str).str.contains("No raw score").all())
        self.assertTrue(unscored.review_status.eq("unreviewed").all())
        self.assertTrue(unscored.reviewer.fillna("").astype(str).str.strip().eq("").all())

    def test_missing_fields_display_blank_and_saved_file_reloads(self):
        loaded_again = read_evidence_csv(REVIEW_PATH)
        pd.testing.assert_frame_equal(self.frame, loaded_again)
        shown = format_evidence_for_display(loaded_again)
        unscored = pd.to_numeric(loaded_again.raw_score, errors="coerce").isna()
        self.assertTrue(shown.loc[unscored, "raw_score"].eq("").all())
        self.assertTrue(shown.loc[unscored, "claim"].eq("").all())
        self.assertTrue(shown.loc[unscored, "reviewer"].eq("").all())
        self.assertNotEqual(REVIEW_PATH.resolve(), TEMPLATE_PATH.resolve())
        template = read_evidence_csv(TEMPLATE_PATH)
        self.assertEqual(len(template), 21)
        self.assertEqual(list(self.frame.columns), list(template.columns))
        template_hash = hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()
        self.assertEqual(template_hash, TEMPLATE_SHA256_AT_REVIEW_START)


if __name__ == "__main__":
    unittest.main()
