"""Validated synthetic data and privacy transformations used by CLI and dashboard."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

SCHEMA = ["record_id", "age", "region", "city", "merchant_type", "amount_inr", "hour", "risk_category"]


def generate_synthetic(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible artificial records. Risk category is a synthetic sensitive value."""
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    regions = np.array(["North", "South", "East", "West", "Central"])
    merchants = np.array(["Grocery", "Utilities", "Food", "Travel", "Retail", "Education"])
    age = rng.integers(18, 80, int(n))
    region = rng.choice(regions, n, p=[.22, .24, .17, .21, .16])
    city = rng.choice([f"City_{i:02d}" for i in range(1, 31)], n)
    merchant = rng.choice(merchants, n, p=[.25, .15, .20, .10, .22, .08])
    amount = np.round(np.clip(rng.lognormal(6.1, .85, n), 20, 150_000), 2)
    hour = rng.integers(0, 24, n)
    risk = np.where((amount > 25_000) | ((hour < 5) & (merchant == "Travel")), "High",
                    np.where(amount > 8_000, "Medium", "Low"))
    frame = pd.DataFrame({"record_id": [f"SYN{i:07d}" for i in range(1, n + 1)],
                          "age": age, "region": region, "city": city,
                          "merchant_type": merchant, "amount_inr": amount,
                          "hour": hour, "risk_category": risk})
    validate_dataset(frame)
    return frame


def validate_dataset(df: pd.DataFrame) -> dict:
    """Raise ValueError for schema/quality failures and return useful quality summaries."""
    missing_cols = sorted(set(SCHEMA) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if df.empty:
        raise ValueError("Dataset must contain at least one row")
    if df[SCHEMA].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns")
    if df.record_id.duplicated().any():
        raise ValueError("record_id values must be unique")
    if not df.age.between(18, 79).all() or not df.hour.between(0, 23).all():
        raise ValueError("age or hour outside documented range")
    if not df.amount_inr.between(20, 150_000).all():
        raise ValueError("amount_inr outside documented range")
    return {"rows": int(len(df)), "missing_cells": int(df[SCHEMA].isna().sum().sum()),
            "duplicate_ids": int(df.record_id.duplicated().sum()),
            "class_balance": {str(k): int(v) for k, v in df.risk_category.value_counts().items()},
            "synthetic_only": True}


def generalize(df: pd.DataFrame, age_level: int = 1, amount_level: int = 1,
               location_level: int = 1, time_level: int = 1) -> pd.DataFrame:
    """Apply fixed hierarchy levels: 0 keeps exact granularity; 1 applies documented bands."""
    for name, level in {"age_level": age_level, "amount_level": amount_level,
                        "location_level": location_level, "time_level": time_level}.items():
        if level not in (0, 1, 2):
            raise ValueError(f"{name} must be 0, 1, or 2")
    x = df.copy()
    if age_level == 0:
        x["age_band"] = x.age.astype(str)
    else:
        widths = {1: 10, 2: 20}[age_level]
        edges = list(range(18, 81, widths)) + [100]
        x["age_band"] = pd.cut(x.age, bins=edges, right=False, include_lowest=True).astype(str)
    if amount_level == 0:
        x["amount_band"] = x.amount_inr.map(lambda v: f"{v:.2f}")
    else:
        bins = [0, 500, 2_000, 5_000, 15_000, 50_000, np.inf] if amount_level == 1 else [0, 2_000, 15_000, np.inf]
        x["amount_band"] = pd.cut(x.amount_inr, bins=bins, include_lowest=True).astype(str)
    if location_level == 0:
        x["location"] = x.city
    elif location_level == 1:
        x["location"] = x.region
    else:
        x["location"] = "India"
    if time_level == 0:
        x["time_band"] = x.hour.astype(str)
    else:
        bins = [-1, 5, 11, 17, 23] if time_level == 1 else [-1, 11, 23]
        labels = ["Night", "Morning", "Afternoon", "Evening"] if time_level == 1 else ["Off-peak", "Daytime"]
        x["time_band"] = pd.cut(x.hour, bins=bins, labels=labels).astype(str)
    return x[["age_band", "location", "merchant_type", "amount_band", "time_band", "risk_category"]]


def suppress_k(df: pd.DataFrame, quasi: Sequence[str], k: int) -> tuple[pd.DataFrame, dict]:
    """Suppress full equivalence classes below k and return release plus diagnostics."""
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    if not quasi or not set(quasi).issubset(df.columns):
        raise ValueError("quasi identifiers must be non-empty columns in the dataset")
    sizes = df.groupby(list(quasi), dropna=False).size()
    counts = df.groupby(list(quasi), dropna=False)[quasi[0]].transform("size")
    released = df.loc[counts >= k].copy().reset_index(drop=True)
    released_sizes = released.groupby(list(quasi), dropna=False).size()
    diag = {"k": int(k), "input_records": int(len(df)), "suppressed_records": int(len(df)-len(released)),
            "retained_records_pct": 100.0 * len(released) / len(df) if len(df) else 0.0,
            "input_groups": int(len(sizes)), "suppressed_groups": int((sizes < k).sum()),
            "minimum_released_group_size": int(released_sizes.min()) if len(released_sizes) else 0,
            "k_satisfied": bool(len(released_sizes) == 0 or released_sizes.min() >= k)}
    return released, diag


def privacy_diagnostics(df: pd.DataFrame, quasi: Sequence[str], sensitive: str = "risk_category",
                        categorical_t: float = .2, l_threshold: int = 2) -> dict:
    """Report distinct-l and categorical-TV diagnostics for the supplied release.

    Missing quasi-identifiers form an explicit equivalence-class value via dropna=False.
    Missing sensitive values are rejected: they must not disappear from the global or
    group distributions and make a release look more diverse than it is.
    """
    if not quasi or len(set(quasi)) != len(quasi):
        raise ValueError("quasi identifiers must be a non-empty unique column list")
    if sensitive not in df or not set(quasi).issubset(df.columns):
        raise ValueError("sensitive attribute and quasi identifiers must exist")
    if df[sensitive].isna().any():
        raise ValueError("sensitive values must not be missing; resolve or explicitly encode missingness")
    if not 0 <= categorical_t <= 1:
        raise ValueError("categorical_t must be in [0,1]")
    if isinstance(l_threshold, bool) or int(l_threshold) != l_threshold or l_threshold < 1:
        raise ValueError("l_threshold must be a positive integer")
    groups = list(df.groupby(list(quasi), dropna=False, sort=False))
    if not groups:
        return {"groups": 0, "min_group_size": 0, "distinct_l": 0, "l_satisfying_pct": 0.0,
                "l_threshold": l_threshold, "worst_categorical_tv": None,
                "t_satisfying_pct": 0.0, "t_threshold": categorical_t,
                "t_metric": "categorical total variation", "diagnostic_only": True,
                "l_pass": False, "t_pass": False}
    overall = df[sensitive].value_counts(normalize=True)
    lvals, tvs = [], []
    for _, grp in groups:
        lvals.append(int(grp[sensitive].nunique()))
        local = grp[sensitive].value_counts(normalize=True)
        idx = overall.index.union(local.index)
        tvs.append(float((overall.reindex(idx, fill_value=0) - local.reindex(idx, fill_value=0)).abs().sum() / 2))
    l_pct = 100.0 * np.mean(np.array(lvals) >= l_threshold)
    t_pct = 100.0 * np.mean(np.array(tvs) <= categorical_t)
    return {"groups": len(groups), "min_group_size": min(len(g) for _, g in groups),
            "distinct_l": min(lvals), "l_satisfying_pct": l_pct,
            "l_threshold": l_threshold, "worst_categorical_tv": max(tvs), "t_threshold": categorical_t,
            "t_satisfying_pct": t_pct,
            "t_metric": "categorical total variation",
            "distance": "group vs released-data global distribution",
            "diagnostic_only": True, "l_pass": bool(l_pct == 100.0),
            "t_pass": bool(t_pct == 100.0)}


def microaggregate(df: pd.DataFrame, columns: Sequence[str], group_size: int = 5) -> tuple[pd.DataFrame, dict]:
    """Sort each numeric column and replace consecutive blocks with their mean (univariate baseline)."""
    try:
        valid_group_size = (not isinstance(group_size, (bool, np.bool_)) and
                            np.isscalar(group_size) and np.isfinite(group_size) and
                            int(group_size) == group_size and group_size >= 2)
    except (TypeError, ValueError, OverflowError):
        valid_group_size = False
    if (not valid_group_size or not columns or len(set(columns)) != len(columns) or
            not set(columns).issubset(df.columns)):
        raise ValueError("group_size must be >=2 and columns must exist")
    if any(not pd.api.types.is_numeric_dtype(df[c]) for c in columns):
        raise ValueError("microaggregation supports numerical columns only")
    try:
        values = df[list(columns)].to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("microaggregation requires non-empty columns with finite values") from exc
    if df.empty or not np.isfinite(values).all():
        raise ValueError("microaggregation requires non-empty columns with finite values")
    # Accept values such as 5.0 after validation, and pass an integer to range().
    group_size = int(group_size)
    x = df.copy()
    rmse = {}
    for col in columns:
        # Group means can be fractional even when the input column is integer-valued
        # (for example, ages 18, 19, 19, 19, 19 produce 18.8). Preserve that exact
        # mean rather than truncating or rounding it back to the source dtype.
        x[col] = x[col].astype(float)
        order = x[col].sort_values(kind="mergesort").index.to_numpy()
        before = x[col].copy()
        for start in range(0, len(order), group_size):
            block = order[start:start + group_size]
            x.loc[block, col] = float(before.loc[block].mean())
        rmse[col] = float(np.sqrt(np.mean((before.to_numpy() - x[col].to_numpy()) ** 2)))
    return x, {"method": "independent sorted consecutive blocks; final incomplete block retained",
               "group_size": group_size, "rmse": rmse}
