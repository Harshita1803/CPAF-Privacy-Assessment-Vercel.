"""Safe loading and persistence helpers for the human-reviewed evidence register."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd


REQUIRED_EVIDENCE_COLUMNS = (
    "app", "dimension", "claim", "source_title", "source_url", "access_date",
    "source_type", "evidence_note", "evidence_strength", "reviewer",
    "review_status", "score_rationale", "raw_score", "notes",
)


def available_evidence_sources(template_path, review_path):
    """Return existing saved sources, preferring the working review copy."""
    sources = {}
    review = Path(review_path)
    template = Path(template_path)
    if review.is_file():
        sources["Saved review (working copy)"] = review
    if template.is_file():
        sources["Original template"] = template
    return sources


def load_evidence_source(sources, label):
    """Load a selected saved source and return frame plus truthful source metadata."""
    if label not in sources:
        raise ValueError(f"Evidence source is not available: {label}")
    path = Path(sources[label]).resolve()
    return read_evidence_csv(path), label, str(path)


def evidence_frames_equal(left, right):
    """Compare editor frames while treating null and blank display cells equally."""
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        return False
    if not left.columns.equals(right.columns) or left.shape != right.shape:
        return False
    left_clean = left.astype(object).where(left.notna(), "").reset_index(drop=True)
    right_clean = right.astype(object).where(right.notna(), "").reset_index(drop=True)
    return left_clean.equals(right_clean)


def _read_text(source) -> str:
    if isinstance(source, (str, os.PathLike)):
        return Path(source).read_text(encoding="utf-8-sig")
    if hasattr(source, "getvalue"):
        content = source.getvalue()
    elif hasattr(source, "read"):
        content = source.read()
    else:
        content = source
    if isinstance(content, bytes):
        return content.decode("utf-8-sig")
    if not isinstance(content, str):
        raise TypeError("CSV source must be a path, text, bytes, or readable file object")
    return content.lstrip("\ufeff")


def read_evidence_csv(source) -> pd.DataFrame:
    """Read and validate the evidence schema while retaining unknown CSV columns."""
    text = _read_text(source)
    try:
        reader = csv.reader(StringIO(text), strict=True)
        header = next(reader)
    except StopIteration as exc:
        raise ValueError("Evidence CSV is empty and has no header row") from exc
    except csv.Error as exc:
        raise ValueError(f"Evidence CSV header is malformed: {exc}") from exc
    if len(header) != len(set(header)):
        duplicates = sorted({name for name in header if header.count(name) > 1})
        raise ValueError(f"Evidence CSV contains duplicate column names: {', '.join(duplicates)}")
    missing = [name for name in REQUIRED_EVIDENCE_COLUMNS if name not in header]
    if missing:
        raise ValueError(f"Evidence CSV is missing required columns: {', '.join(missing)}")
    try:
        for line_number, row in enumerate(reader, start=2):
            if row and len(row) != len(header):
                raise ValueError(
                    f"Evidence CSV row {line_number} has {len(row)} fields; expected {len(header)}"
                )
    except csv.Error as exc:
        raise ValueError(f"Evidence CSV is malformed: {exc}") from exc
    try:
        frame = pd.read_csv(StringIO(text))
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Evidence CSV is malformed: {exc}") from exc
    if list(frame.columns) != header:
        raise ValueError("Evidence CSV columns could not be parsed consistently")
    return frame


def save_evidence_csv(frame: pd.DataFrame, destination, *, confirm_overwrite=False) -> Path:
    """Atomically save to a working file; confirmed replacements get a timestamped backup."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Evidence to save must be a pandas DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError("Evidence contains duplicate column names")
    missing = [name for name in REQUIRED_EVIDENCE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Evidence is missing required columns: {', '.join(missing)}")

    target = Path(destination)
    template_name = "evidence_register_template.csv"
    if target.name.lower() == template_name:
        raise ValueError("Refusing to save over the original evidence template")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not target.is_file():
        raise IsADirectoryError(f"Save destination is not a file: {target}")
    if target.exists() and not confirm_overwrite:
        raise FileExistsError(f"Destination already exists; confirm replacement to save: {target}")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".tmp",
            prefix=f".{target.name}.", dir=parent, delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())

        if target.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = target.with_name(f"{target.stem}.backup-{stamp}{target.suffix}")
            shutil.copy2(target, backup)
        os.replace(temp_path, target)
        temp_path = None
        return target
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
