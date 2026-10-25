"""Repeated-split predictive utility and synthetic subgroup performance evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import wasserstein_distance
from .reproducibility import require_outputs_absent

from .core import generate_synthetic, generalize, suppress_k
QI = ["age_band", "location", "merchant_type", "amount_band", "time_band"]


def _tv(reference: pd.Series, candidate: pd.Series) -> float:
    a, b = reference.value_counts(normalize=True), candidate.value_counts(normalize=True)
    labels = a.index.union(b.index)
    return float((a.reindex(labels, fill_value=0) - b.reindex(labels, fill_value=0)).abs().sum() / 2)


def _distribution_drift(reference: pd.DataFrame, selected: pd.DataFrame) -> dict:
    """Separate, interpretable frequency drift and range-normalized numeric MAE."""
    result = {}
    for col in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[col]):
            denom = float(reference[col].max() - reference[col].min())
            distance = wasserstein_distance(reference[col].to_numpy(), selected[col].to_numpy()) if len(selected) else np.nan
            result[f"{col}_normalized_wasserstein"] = distance / denom if denom else 0.0
        else:
            result[f"{col}_tv"] = _tv(reference[col], selected[col]) if len(selected) else np.nan
    return result


def _fairness_by_age(age: pd.Series, y: np.ndarray, p: np.ndarray, representation: str,
                     repeat: int) -> tuple[list[dict], dict]:
    bands = pd.cut(age, bins=[17, 29, 49, 100], labels=["18-29", "30-49", "50+"])
    rows = []
    for band in ["18-29", "30-49", "50+"]:
        mask = (bands.astype(str).to_numpy() == band)
        if not mask.any():
            continue
        pred = (p[mask] >= .5).astype(int)
        positives, negatives = int((y[mask] == 1).sum()), int((y[mask] == 0).sum())
        rows.append({"representation": representation, "repeat": repeat, "age_group": band,
                     "n": int(mask.sum()), "positive_rate": float(y[mask].mean()),
                     "positive_n": positives, "negative_n": negatives,
                     "small_subgroup_warning": bool(mask.sum() < 30 or positives < 10 or negatives < 10),
                     "selection_rate": float(pred.mean()),
                     "tpr": float(((pred == 1) & (y[mask] == 1)).sum() / positives) if positives else np.nan,
                     "fpr": float(((pred == 1) & (y[mask] == 0)).sum() / negatives) if negatives else np.nan})
    summary = {"representation": representation, "repeat": repeat,
               "small_subgroup_warning": any(r["small_subgroup_warning"] for r in rows),
               "age_tpr_gap": (max(r["tpr"] for r in rows if np.isfinite(r["tpr"])) -
                               min(r["tpr"] for r in rows if np.isfinite(r["tpr"])))
               if any(np.isfinite(r["tpr"]) for r in rows) else np.nan,
               "age_fpr_gap": (max(r["fpr"] for r in rows if np.isfinite(r["fpr"])) -
                               min(r["fpr"] for r in rows if np.isfinite(r["fpr"])))
               if any(np.isfinite(r["fpr"]) for r in rows) else np.nan}
    return rows, summary


def run_repeated_utility(
    n: int = 10_000, seed: int = 42, repeats: int = 3, k: int = 5,
    output_dir: str | Path = "cpaf_outputs",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare raw/generalized/k-suppressed logistic utility on repeated stratified splits.

    The same held-out indices are scored for each representation within each repeat.
    Preprocessing remains inside each sklearn pipeline and is fit only on that repeat's
    training records. Suppression affects training records; the held-out test set remains
    the same generalized population.
    """
    if n < 100 or repeats < 2 or k < 1:
        raise ValueError("n must be >=100, repeats >=2, and k >=1")
    output_files = ("repeated_utility_runs.csv", "repeated_utility_summary.csv",
                    "synthetic_age_fairness.csv", "synthetic_age_fairness_gaps.csv",
                    "suppression_distribution_drift.csv", "repeated_utility_manifest.json")
    require_outputs_absent(output_dir, output_files)
    # Import lazily because the backward-compatible ML module imports the CLI.
    from cpaf_ml import add_synthetic_label, cols_of, make_model, raw_view
    raw = generate_synthetic(n, seed)
    y = add_synthetic_label(raw, seed + 1)
    if len(np.unique(y)) != 2:
        raise ValueError("synthetic target must contain both classes")
    splitter = StratifiedShuffleSplit(n_splits=repeats, test_size=.25, random_state=seed)
    utilities, fairness, fairness_summary, drift_rows, skipped = [], [], [], [], []

    for repeat, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(n), y), start=1):
        raw_train, raw_test = raw.iloc[train_idx].reset_index(drop=True), raw.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y[train_idx], y[test_idx]
        gen_train, gen_test = generalize(raw_train), generalize(raw_test)
        keep = (gen_train.groupby(QI, dropna=False)[QI[0]].transform("size") >= k).to_numpy()
        views = {
            "Raw": (raw_view(raw_train), raw_view(raw_test), np.ones(len(raw_train), dtype=bool)),
            "Generalized": (gen_train[QI], gen_test[QI], np.ones(len(raw_train), dtype=bool)),
            f"Generalized+k={k}": (gen_train[QI], gen_test[QI], keep),
        }
        for name, (xtrain, xtest, mask) in views.items():
            if int(mask.sum()) < 20 or len(np.unique(y_train[mask])) < 2:
                skipped.append({"representation": name, "repeat": repeat,
                                "training_rows": int(mask.sum()),
                                "reason": "fewer than 20 rows or one training label class"})
                continue
            cats, nums = cols_of(xtrain)
            model = make_model("Logistic Regression", cats, nums, C=1.0)
            model.fit(xtrain.loc[mask], y_train[mask])
            prob = model.predict_proba(xtest)[:, 1]
            pred = (prob >= .5).astype(int)
            cm = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
            utilities.append({
                "representation": name, "repeat": repeat, "seed": seed, "n": n,
                "train_rows": int(mask.sum()), "test_rows": int(len(y_test)),
                "positive_rate_test": float(y_test.mean()),
                "roc_auc": float(roc_auc_score(y_test, prob)),
                "pr_auc": float(average_precision_score(y_test, prob)),
                "brier": float(brier_score_loss(y_test, prob)),
                "precision_at_0_5": float(precision_score(y_test, pred, zero_division=0)),
                "recall_at_0_5": float(recall_score(y_test, pred, zero_division=0)),
                "f1_at_0_5": float(f1_score(y_test, pred, zero_division=0)),
                "confusion_matrix_labels_0_1": cm,
            })
            fr, fs = _fairness_by_age(raw_test.age.reset_index(drop=True), y_test, prob, name, repeat)
            fairness.extend(fr)
            fairness_summary.append(fs)
            if name.endswith(f"+k={k}"):
                selected_raw = raw_train.iloc[np.flatnonzero(mask)].reset_index(drop=True)
                drift_rows.append({"representation": name, "repeat": repeat,
                                   "selected_rows": int(mask.sum()), "input_rows": len(raw_train),
                                   **_distribution_drift(raw_train.drop(columns=["record_id"]),
                                                         selected_raw.drop(columns=["record_id"]))})

    utility = pd.DataFrame(utilities)
    summary = utility.groupby("representation", as_index=False).agg(
        runs=("repeat", "count"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        brier_mean=("brier", "mean"), brier_std=("brier", "std"),
        f1_mean=("f1_at_0_5", "mean"), f1_std=("f1_at_0_5", "std"),
    )
    fair = pd.DataFrame(fairness)
    fair_summary = pd.DataFrame(fairness_summary)
    drift = pd.DataFrame(drift_rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    utility.to_csv(out / "repeated_utility_runs.csv", index=False)
    summary.to_csv(out / "repeated_utility_summary.csv", index=False)
    fair.to_csv(out / "synthetic_age_fairness.csv", index=False)
    fair_summary.to_csv(out / "synthetic_age_fairness_gaps.csv", index=False)
    drift.to_csv(out / "suppression_distribution_drift.csv", index=False)
    packages = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    manifest = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "seed": seed, "label_seed": seed + 1,
                "dataset_size": n,
                "repeats": repeats, "split": "StratifiedShuffleSplit, 75/25",
                "k": k, "target_provenance": "artificial noisy label generated from synthetic age/amount/hour/merchant",
                "models": ["LogisticRegression"], "metrics": ["ROC-AUC", "PR-AUC", "Brier", "precision", "recall", "F1", "confusion matrix"],
                "fairness": "synthetic age-band TPR/FPR/selection-rate; descriptive only",
                "dataset_provenance": "generated synthetic transactions with artificial noisy label; no external/provider records",
                "output_dir": str(out.resolve()),
                "output_files": list(output_files),
                "skipped_configurations": skipped,
                "packages": packages,
                "status": "completed_with_skips" if skipped else "completed"}
    (out / "repeated_utility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return utility, summary
