import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cpaf.evidence import (REQUIRED_EVIDENCE_COLUMNS, available_evidence_sources,
                           evidence_frames_equal, load_evidence_source,
                           read_evidence_csv, save_evidence_csv)
from cpaf_advanced import (APPS, DIMENSIONS, compute_app_metrics,
                           format_evidence_for_display, validate_evidence_rows)


def evidence_frame():
    row = {column: "" for column in REQUIRED_EVIDENCE_COLUMNS}
    row.update({
        "app": APPS[0], "dimension": DIMENSIONS[0], "claim": "A reviewed policy statement",
        "source_title": "Policy", "source_url": "https://example.test/policy",
        "access_date": "2026-10-24", "source_type": "policy",
        "evidence_note": "Relevant passage reviewed", "evidence_strength": 0.8,
        "reviewer": "reviewer-1", "review_status": "reviewed",
        "score_rationale": "Rationale recorded", "raw_score": 4, "notes": "initial",
        "future_extension": "retain me",
    })
    return pd.DataFrame([row])


class EvidencePersistenceTests(unittest.TestCase):
    def temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)

    def test_save_to_working_copy_preserves_template_and_unknown_columns(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            template = root / "evidence_register_template.csv"
            template.write_text("original template bytes", encoding="utf-8")
            edited = evidence_frame()
            edited.loc[0, "claim"] = "Updated reviewed claim"
            destination = root / "evidence_reviews" / "evidence_register_reviewed.csv"
            result = save_evidence_csv(edited, destination)
            self.assertEqual(result, destination)
            self.assertEqual(template.read_text(encoding="utf-8"), "original template bytes")
            loaded = read_evidence_csv(destination)
            self.assertEqual(loaded.loc[0, "claim"], "Updated reviewed claim")
            self.assertEqual(loaded.loc[0, "future_extension"], "retain me")

    def test_existing_destination_requires_confirmation_and_is_backed_up(self):
        with self.temporary_directory() as directory:
            destination = Path(directory) / "evidence_register_reviewed.csv"
            destination.write_text(evidence_frame().to_csv(index=False), encoding="utf-8")
            before = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                save_evidence_csv(evidence_frame(), destination)
            self.assertEqual(destination.read_bytes(), before)
            changed = evidence_frame()
            changed.loc[0, "claim"] = "replacement"
            save_evidence_csv(changed, destination, confirm_overwrite=True)
            backups = list(destination.parent.glob("evidence_register_reviewed.backup-*.csv"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), before)
            self.assertEqual(read_evidence_csv(destination).loc[0, "claim"], "replacement")

    def test_missing_malformed_duplicate_and_incomplete_csv_are_reported(self):
        with self.temporary_directory() as directory:
            missing = Path(directory) / "missing.csv"
            with self.assertRaises(FileNotFoundError):
                read_evidence_csv(missing)
            malformed = Path(directory) / "malformed.csv"
            header = ",".join(REQUIRED_EVIDENCE_COLUMNS)
            malformed.write_text(f"{header}\n" + ",".join(["extra"] * (len(REQUIRED_EVIDENCE_COLUMNS) + 1)) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_evidence_csv(malformed)
            duplicate = Path(directory) / "duplicate.csv"
            duplicate.write_text("app,app\nPaytm,PhonePe\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_evidence_csv(duplicate)
            incomplete = Path(directory) / "incomplete.csv"
            incomplete.write_text("app,dimension\nPaytm,Data Minimisation\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                read_evidence_csv(incomplete)

    def test_missing_fields_remain_blank_and_unscored(self):
        frame = evidence_frame()
        frame.loc[0, "claim"] = ""
        frame.loc[0, "source_url"] = ""
        frame.loc[0, "raw_score"] = None
        eligible, issues = validate_evidence_rows(frame)
        self.assertFalse(eligible.iloc[0])
        self.assertIn("claim", issues[0])
        self.assertIn("valid HTTP(S) source_url", issues[0])
        scores = compute_app_metrics(frame)
        self.assertTrue(scores["Weighted CPI"].isna().all())

    def test_invalid_score_type_is_reported_as_ineligible(self):
        frame = evidence_frame()
        frame["raw_score"] = frame["raw_score"].astype(object)
        frame.loc[0, "raw_score"] = "five"
        eligible, issues = validate_evidence_rows(frame)
        self.assertFalse(eligible.iloc[0])
        self.assertIn("raw_score (1–5)", issues[0])

    def test_missing_strength_is_not_eligible_or_silently_scored(self):
        frame = evidence_frame()
        frame["evidence_strength"] = None
        eligible, issues = validate_evidence_rows(frame)
        self.assertFalse(eligible.iloc[0])
        self.assertIn("evidence_strength (0–1)", issues[0])
        self.assertTrue(compute_app_metrics(frame)["Weighted CPI"].isna().all())

    def test_display_blanks_missing_numeric_values_without_filling_scores(self):
        frame = evidence_frame()
        frame["evidence_strength"] = None
        frame["raw_score"] = None
        display = format_evidence_for_display(frame)
        self.assertEqual(display.loc[0, "evidence_strength"], "")
        self.assertEqual(display.loc[0, "raw_score"], "")
        self.assertTrue(pd.isna(frame.loc[0, "evidence_strength"]))
        self.assertTrue(pd.isna(frame.loc[0, "raw_score"]))

    def test_null_to_blank_editor_round_trip_is_not_a_dirty_edit(self):
        original = evidence_frame()
        original["raw_score"] = None
        editor = format_evidence_for_display(original)
        editor.loc[0, "raw_score"] = ""
        self.assertTrue(evidence_frames_equal(editor, original))

    def test_reviewed_csv_is_selected_loaded_and_source_path_is_truthful(self):
        root = Path(__file__).resolve().parents[1]
        template = root / "cpaf_outputs" / "evidence_register_template.csv"
        review = root / "cpaf_outputs" / "evidence_reviews" / "evidence_register_reviewed.csv"
        sources = available_evidence_sources(template, review)
        self.assertEqual(next(iter(sources)), "Saved review (working copy)")
        loaded, label, source_path = load_evidence_source(sources, next(iter(sources)))
        self.assertEqual(label, "Saved review (working copy)")
        self.assertEqual(Path(source_path), review.resolve())
        self.assertEqual(len(loaded), 21)
        self.assertEqual(int(pd.to_numeric(loaded.raw_score, errors="coerce").notna().sum()), 17)
        self.assertIn("evidence_strength", loaded.columns)
        self.assertIn("raw_score", loaded.columns)

    def test_unknown_columns_survive_saved_source_loading(self):
        with self.temporary_directory() as directory:
            path = Path(directory) / "review.csv"
            frame = evidence_frame()
            frame["unknown_future_column"] = "keep"
            frame.to_csv(path, index=False)
            sources = available_evidence_sources(path, path.with_name("missing.csv"))
            loaded, _, actual_path = load_evidence_source(sources, "Original template")
            self.assertEqual(loaded.loc[0, "unknown_future_column"], "keep")
            self.assertEqual(Path(actual_path), path.resolve())

    def test_missing_destination_template_is_protected_and_save_failure_is_not_success(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            template_destination = root / "evidence_register_template.csv"
            with self.assertRaisesRegex(ValueError, "original evidence template"):
                save_evidence_csv(evidence_frame(), template_destination)
            file_parent = root / "not-a-directory"
            file_parent.write_text("block", encoding="utf-8")
            with self.assertRaises(OSError):
                save_evidence_csv(evidence_frame(), file_parent / "review.csv")
            self.assertFalse((file_parent / "review.csv").exists())


if __name__ == "__main__":
    unittest.main()
