"""Synthetic disclosure-risk proxies and disjoint shadow-model membership inference."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .reproducibility import require_outputs_absent

from .core import generate_synthetic, generalize

QI = ["age_band", "location", "merchant_type", "amount_band", "time_band"]


def _features(raw: pd.DataFrame, representation: str) -> pd.DataFrame:
    if representation == "Raw":
        from cpaf_ml import raw_view
        return raw_view(raw).reset_index(drop=True)
    return generalize(raw).loc[:, QI].reset_index(drop=True)


def _attack_features(model, x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    p = np.clip(model.predict_proba(x)[:, 1], 1e-8, 1 - 1e-8)
    confidence = np.where(y == 1, p, 1 - p)
    loss = -np.log(np.clip(confidence, 1e-8, 1))
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return np.column_stack([confidence, loss, entropy])


def synthetic_disclosure_proxies(raw: pd.DataFrame, seed: int = 42,
                                 neighbor_sample_size: int = 2_000) -> dict:
    """Aggregate uniqueness, duplicate and approximate-neighbor diagnostics on synthetic rows."""
    if raw.empty:
        return {"rows": 0, "exact_duplicate_rows": 0, "rare_qi_rows_le_2": 0,
                "nearest_neighbor_sample_rows": 0, "nearest_distance_median": None}
    qids = ["age", "region", "city", "merchant_type", "amount_inr"]
    sizes = raw.groupby(qids, dropna=False).size()
    duplicates = int(raw.drop(columns=["record_id"], errors="ignore").duplicated().sum())
    profile_sizes = np.asarray(raw.set_index(qids).index.map(sizes), dtype=int)
    rare_count = int(np.count_nonzero(profile_sizes <= 2))
    rng = np.random.default_rng(seed)
    sample = raw if len(raw) <= neighbor_sample_size else raw.iloc[
        np.sort(rng.choice(len(raw), neighbor_sample_size, replace=False))]
    numeric = np.column_stack([
        sample.age.to_numpy() / 61.0,
        np.log(sample.amount_inr.to_numpy()) / np.log(150_000 / 20),
        sample.hour.to_numpy() / 23.0,
    ])
    categorical = pd.get_dummies(sample[["region", "city", "merchant_type"]], dtype=float).to_numpy()
    matrix = np.column_stack([numeric, categorical])
    if len(sample) >= 2:
        model = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="euclidean", n_jobs=1)
        model.fit(matrix)
        distances, _ = model.kneighbors(matrix)
        nn = distances[:, 1]
    else:
        nn = np.array([], dtype=float)
    return {
        "synthetic_only": True, "rows": int(len(raw)),
        "exact_duplicate_rows_excluding_synthetic_id": duplicates,
        "exact_duplicate_rate_pct": 100 * duplicates / len(raw),
        "rare_exact_qi_rows_at_group_size_le_2": rare_count,
        "rare_exact_qi_rate_pct": 100 * rare_count / len(raw),
        "qi_definition": qids,
        "nearest_neighbor_method": "Euclidean on range-normalized age/log-amount/hour and one-hot region/city/merchant",
        "nearest_neighbor_sample_rows": int(len(sample)),
        "nearest_neighbor_distance_min": float(np.min(nn)) if len(nn) else None,
        "nearest_neighbor_distance_p05": float(np.quantile(nn, .05)) if len(nn) else None,
        "nearest_neighbor_distance_median": float(np.median(nn)) if len(nn) else None,
        "interpretation": "synthetic uniqueness/similarity proxies; not re-identification evidence",
    }


def run_shadow_mia(
    n: int = 8_000, seed: int = 42, k: int = 5,
    output_dir: str | Path = "cpaf_outputs",
) -> tuple[pd.DataFrame, dict]:
    """Train target/shadow models on disjoint synthetic partitions and test a shadow attack.

    Target train (40%), shadow train (20%), shadow nonmember calibration (20%), and
    target nonmember evaluation (20%) are pairwise disjoint. A logistic attack learns
    from shadow membership outcomes and is tested once on balanced, unseen target
    member/nonmember examples. This measures one confidence/loss attack only.
    """
    if n < 500 or k < 1:
        raise ValueError("n must be >=500 and k >=1")
    output_files = ("shadow_mia_results.csv", "shadow_mia_results.json",
                    "shadow_mia_manifest.json", "synthetic_disclosure_proxies.json")
    require_outputs_absent(output_dir, output_files)
    from cpaf_ml import add_synthetic_label, cols_of, make_model

    raw = generate_synthetic(n, seed)
    y = add_synthetic_label(raw, seed + 1)
    all_idx = np.arange(n)
    target_train, remain = train_test_split(
        all_idx, test_size=.60, random_state=seed, stratify=y)
    shadow_train, final = train_test_split(
        remain, test_size=2/3, random_state=seed + 1, stratify=y[remain])
    shadow_nonmember, target_nonmember = train_test_split(
        final, test_size=.5, random_state=seed + 2, stratify=y[final])
    partitions = [set(v.tolist()) for v in
                  [target_train, shadow_train, shadow_nonmember, target_nonmember]]
    disjoint = all(not partitions[i].intersection(partitions[j])
                   for i in range(len(partitions)) for j in range(i + 1, len(partitions)))
    if not disjoint:
        raise RuntimeError("internal split error: attack partitions overlap")

    rows, skipped_representations = [], []
    rng = np.random.default_rng(seed + 3)
    for representation in ["Raw", "Generalized", f"Generalized+k={k}"]:
        x_target = _features(raw.iloc[target_train], representation.split("+k=")[0])
        x_shadow = _features(raw.iloc[shadow_train], representation.split("+k=")[0])
        x_shadow_non = _features(raw.iloc[shadow_nonmember], representation.split("+k=")[0])
        x_target_non = _features(raw.iloc[target_nonmember], representation.split("+k=")[0])
        yt, ys = y[target_train], y[shadow_train]
        if representation.startswith("Generalized+k="):
            target_keep = (x_target.groupby(QI, dropna=False)[QI[0]].transform("size") >= k).to_numpy()
            shadow_keep = (x_shadow.groupby(QI, dropna=False)[QI[0]].transform("size") >= k).to_numpy()
        else:
            target_keep = np.ones(len(x_target), dtype=bool)
            shadow_keep = np.ones(len(x_shadow), dtype=bool)
        if target_keep.sum() < 30 or shadow_keep.sum() < 30:
            skipped_representations.append({
                "representation": representation,
                "target_training_rows": int(target_keep.sum()),
                "shadow_training_rows": int(shadow_keep.sum()),
                "reason": "fewer than 30 retained target or shadow training rows",
            })
            continue
        cat, num = cols_of(x_target)
        target_model = make_model("Logistic Regression", cat, num)
        target_model.fit(x_target.loc[target_keep], yt[target_keep])
        shadow_cat, shadow_num = cols_of(x_shadow)
        shadow_model = make_model("Logistic Regression", shadow_cat, shadow_num)
        shadow_model.fit(x_shadow.loc[shadow_keep], ys[shadow_keep])

        shadow_member_x = x_shadow.loc[shadow_keep]
        shadow_member_y = ys[shadow_keep]
        shadow_member_features = _attack_features(shadow_model, shadow_member_x, shadow_member_y)
        shadow_non_features = _attack_features(
            shadow_model, x_shadow_non, y[shadow_nonmember])
        calibration_n = min(len(shadow_member_features), len(shadow_non_features))
        cm = rng.choice(len(shadow_member_features), calibration_n, replace=False)
        cn = rng.choice(len(shadow_non_features), calibration_n, replace=False)
        attack_train_x = np.vstack([shadow_member_features[cm], shadow_non_features[cn]])
        attack_train_y = np.r_[np.ones(calibration_n), np.zeros(calibration_n)]
        attack = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
        attack.fit(attack_train_x, attack_train_y)

        target_member_x = x_target.loc[target_keep]
        target_member_y = yt[target_keep]
        target_member_features = _attack_features(target_model, target_member_x, target_member_y)
        target_non_features = _attack_features(target_model, x_target_non, y[target_nonmember])
        eval_n = min(len(target_member_features), len(target_non_features))
        tm = rng.choice(len(target_member_features), eval_n, replace=False)
        tn = rng.choice(len(target_non_features), eval_n, replace=False)
        attack_test_x = np.vstack([target_member_features[tm], target_non_features[tn]])
        attack_test_y = np.r_[np.ones(eval_n), np.zeros(eval_n)]
        attack_score = attack.predict_proba(attack_test_x)[:, 1]
        attack_pred = (attack_score >= .5).astype(int)
        # Stratified bootstrap preserves the balanced member/nonmember design.
        # This quantifies finite attack-test uncertainty; it is not a privacy bound.
        boot_rng = np.random.default_rng(seed + 10_000 + len(rows))
        boot_aucs = []
        for _ in range(1000):
            member_draw = boot_rng.integers(0, eval_n, size=eval_n)
            nonmember_draw = boot_rng.integers(eval_n, 2 * eval_n, size=eval_n)
            draw = np.r_[member_draw, nonmember_draw]
            boot_aucs.append(roc_auc_score(attack_test_y[draw], attack_score[draw]))
        auc_ci_low, auc_ci_high = np.quantile(boot_aucs, [.025, .975])
        target_prob = target_model.predict_proba(x_target_non)[:, 1]
        target_utility = roc_auc_score(y[target_nonmember], target_prob)
        rows.append({
            "representation": representation, "seed": seed, "k": k,
            "target_train_rows_before_suppression": len(target_train),
            "target_train_rows_after_suppression": int(target_keep.sum()),
            "shadow_train_rows_after_suppression": int(shadow_keep.sum()),
            "shadow_attack_train_rows": int(2 * calibration_n),
            "attack_test_members": int(eval_n), "attack_test_nonmembers": int(eval_n),
            "attack_test_auc": float(roc_auc_score(attack_test_y, attack_score)),
            "attack_auc_bootstrap_95ci_low": float(auc_ci_low),
            "attack_auc_bootstrap_95ci_high": float(auc_ci_high),
            "attack_auc_bootstrap_replicates": len(boot_aucs),
            "attack_test_precision_at_0_5": float(precision_score(attack_test_y, attack_pred, zero_division=0)),
            "attack_test_recall_at_0_5": float(recall_score(attack_test_y, attack_pred, zero_division=0)),
            "attack_test_f1_at_0_5": float(f1_score(attack_test_y, attack_pred, zero_division=0)),
            "balanced_random_accuracy_baseline": .5, "roc_auc_chance_baseline": .5,
            "target_model_nonmember_roc_auc": float(target_utility),
            "partition_sets_pairwise_disjoint": disjoint,
            "attack_scope": "shadow-trained confidence, loss and entropy attack",
            "formal_privacy_guarantee": False,
        })
    result = pd.DataFrame(rows)
    proxies = synthetic_disclosure_proxies(raw, seed=seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "shadow_mia_results.csv", index=False)
    result.to_json(out / "shadow_mia_results.json", orient="records", indent=2)
    (out / "synthetic_disclosure_proxies.json").write_text(json.dumps(proxies, indent=2), encoding="utf-8")
    packages = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    manifest = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "seed": seed, "label_seed": seed + 1,
                "dataset_size": n,
                "target_and_shadow_partitions_disjoint": disjoint, "k": k,
                "partition_proportions": {"target_train": .4, "shadow_train": .2,
                                          "shadow_nonmember_calibration": .2,
                                          "target_nonmember_evaluation": .2},
                "attack": "shadow-trained logistic classifier on confidence/loss/entropy",
                "limitations": ["one attack family", "synthetic population only",
                                "attack failure is not proof of privacy"],
                "dataset_provenance": "generated synthetic transactions with artificial noisy label; no external/provider records",
                "output_dir": str(out.resolve()),
                "output_files": list(output_files),
                "skipped_representations": skipped_representations,
                "packages": packages,
                "status": "completed_with_skips" if skipped_representations else "completed",
                "result_rows": len(result)}
    (out / "shadow_mia_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result, proxies
