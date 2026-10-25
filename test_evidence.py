import unittest
from io import StringIO
import pandas as pd

from cpaf_advanced import (APPS, DIMENSIONS, compute_app_metrics,
                           compute_weight_sensitivity, format_evidence_for_display)


class EvidenceTests(unittest.TestCase):
    def test_missing_evidence_is_not_scored_as_good_or_bad(self):
        frame = pd.DataFrame(columns=["app", "dimension", "raw_score", "evidence_strength"])
        scores = compute_app_metrics(frame)
        self.assertEqual(len(scores), len(APPS))
        self.assertTrue(scores["Weighted CPI"].isna().all())
        self.assertTrue((scores["Evidence coverage (%)"] == 0).all())

    def test_app_scores_require_reviewed_traceable_evidence(self):
        row = {"app": APPS[0], "dimension": DIMENSIONS[0], "raw_score": 5,
               "evidence_strength": 1, "review_status": "unreviewed",
               "source_url": "", "access_date": "", "claim": "",
               "reviewer": "", "score_rationale": ""}
        unreviewed = compute_app_metrics(pd.DataFrame([row]))
        self.assertTrue(unreviewed["Weighted CPI"].isna().all())
        row.update({"review_status": "reviewed", "source_url": "https://example.test/policy",
                    "access_date": "2026-10-24", "claim": "Policy states a retention period",
                    "reviewer": "reviewer-1", "score_rationale": "Specific text and date checked"})
        reviewed = compute_app_metrics(pd.DataFrame([row]))
        score = reviewed.loc[reviewed.App == APPS[0], "Weighted CPI"].iloc[0]
        self.assertEqual(score, 5.0)

    def test_weight_sensitivity_respects_observed_scores(self):
        row = {"App": APPS[0], **{d: None for d in DIMENSIONS}}
        row[DIMENSIONS[0]] = 4.0
        result = compute_weight_sensitivity(pd.DataFrame([row]))
        self.assertEqual(result.loc[0, "equal_weighted_cpi"], 4.0)

    def test_missing_csv_text_values_display_as_blank_without_fabrication(self):
        loaded = pd.read_csv(StringIO(
            "app,dimension,claim,source_title,source_url,access_date,review_status,raw_score\n"
            "Paytm,Data Minimisation,,,,,unreviewed,\n"
        ))
        self.assertTrue(pd.isna(loaded.loc[0, "claim"]))
        shown = format_evidence_for_display(loaded)
        for column in ("claim", "source_title", "source_url", "access_date"):
            self.assertEqual(shown.loc[0, column], "")
        self.assertEqual(shown.loc[0, "review_status"], "unreviewed")
        self.assertEqual(shown.loc[0, "raw_score"], "")
        self.assertTrue(pd.isna(loaded.loc[0, "claim"]))  # display formatting is non-mutating

    def test_missing_text_columns_are_optional_for_display_formatter(self):
        result = format_evidence_for_display(pd.DataFrame({"app": [APPS[0]]}))
        self.assertEqual(result.loc[0, "app"], APPS[0])


if __name__ == "__main__":
    unittest.main()
