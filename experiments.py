"""Reproducible privacy parameter sweeps and empirical two-objective Pareto analysis."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import platform
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import generate_synthetic, generalize, privacy_diagnostics, suppress_k, validate_dataset
from .reproducibility import require_outputs_absent

DEFAULT_QI = ["age_band", "location", "merchant_type", "amount_band", "time_band"]


def _tv(left: pd.Series, right: pd.Series) -> float:
    """Total variation distance between finite categorical distributions."""
    lp, rp = left.value_counts(normalize=True), right.value_counts(normalize=True)
    labels = lp.index.union(rp.index)
    return float((lp.reindex(labels, fill_value=0) - rp.reindex(labels, fill_value=0)).abs().sum() / 2)


def _valid_pareto_objectives(frame: pd.DataFrame) -> pd.Series:
    values = frame[["retained_records_pct", "worst_categorical_tv"]].to_numpy(dtype=float)
    return pd.Series(
        np.isfinite(values).all(axis=1) &
        frame["retained_records_pct"].between(0, 100).to_numpy() &
        frame["worst_categorical_tv"].between(0, 1).to_numpy(),
        index=frame.index,
    )


def pareto_frontier(results: pd.DataFrame) -> pd.DataFrame:
    """Return configs not dominated on retention (maximize) and worst categorical TV (minimize).

    These are empirical objectives over this generated release, not formal privacy or
    globally optimal configurations. Exact ties remain on the frontier.
    """
    required = {"retained_records_pct", "worst_categorical_tv"}
    if not required.issubset(results.columns):
        raise ValueError(f"results missing columns: {sorted(required - set(results.columns))}")
    frame = results.dropna(subset=list(required)).copy()
    frame = frame.loc[_valid_pareto_objectives(frame)].copy()
    keep = np.ones(len(frame), dtype=bool)
    vals = frame[["retained_records_pct", "worst_categorical_tv"]].to_numpy(float)
    for i, (ret_i, tv_i) in enumerate(vals):
        dominates = ((vals[:, 0] >= ret_i) & (vals[:, 1] <= tv_i) &
                     ((vals[:, 0] > ret_i) | (vals[:, 1] < tv_i)))
        dominates[i] = False
        if dominates.any():
            keep[i] = False
    return frame.loc[keep].sort_values(
        ["retained_records_pct", "worst_categorical_tv"],
        ascending=[False, True]).reset_index(drop=True)


def run_privacy_sweep(
    n: int = 5_000,
    seeds: Iterable[int] = (42,),
    k_values: Iterable[int] = (1, 5, 10),
    l_values: Iterable[int] = (2, 3),
    t_values: Iterable[float] = (.1, .2),
    hierarchy_levels: Iterable[int] = (1, 2),
    out_dir: str | Path = "cpaf_outputs",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a complete, deterministic k/l/t and common hierarchy-level sweep.

    l and t values are diagnostic thresholds: they do not alter the released table.
    k controls whole-class suppression. Utility is reported as retention and separate
    pre/post categorical frequency shifts; no incomparable metrics are combined.
    """
    seeds, ks, ls, ts, levels = map(list, (seeds, k_values, l_values, t_values, hierarchy_levels))
    def is_integer(value):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            return False
        try:
            return bool(np.isfinite(value) and int(value) == value)
        except (TypeError, ValueError, OverflowError):
            return False

    def is_threshold(value):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            return False
        try:
            return bool(np.isfinite(value) and 0 <= float(value) <= 1)
        except (TypeError, ValueError, OverflowError):
            return False

    if not is_integer(n) or n < 1 or not seeds or not ks or not ls or not ts or not levels:
        raise ValueError("n and every sweep grid must be non-empty and n positive")
    if any(not is_integer(v) or int(v) < 0 for v in seeds):
        raise ValueError("seeds must be non-negative integers")
    if any(not is_integer(v) or int(v) < 1 for v in ks) or any(not is_integer(v) or int(v) < 1 for v in ls):
        raise ValueError("k and l values must be positive integers")
    if any(not is_threshold(v) for v in ts):
        raise ValueError("t values must be in [0,1]")
    if any(not is_integer(v) or v not in (0, 1, 2) for v in levels):
        raise ValueError("hierarchy levels must be 0, 1, or 2")

    out = Path(out_dir)
    output_files = ("privacy_sweep.csv", "privacy_sweep.json", "privacy_sweep_manifest.json",
                    "pareto_frontier.csv", "pareto_frontier.json", "pareto_frontier.png")
    require_outputs_absent(out, output_files)
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for seed in seeds:
        data_started = time.perf_counter()
        raw = generate_synthetic(n, int(seed))
        data_generation_elapsed = time.perf_counter() - data_started
        quality = validate_dataset(raw)
        for level in levels:
            generalization_started = time.perf_counter()
            generalized = generalize(raw, level, level, level, level)
            generalization_elapsed = time.perf_counter() - generalization_started
            for k in ks:
                start = time.perf_counter()
                released, kdiag = suppress_k(generalized, DEFAULT_QI, int(k))
                suppression_elapsed = time.perf_counter() - start
                # If all rows are suppressed, report absent distances as NaN, not zero.
                if len(released):
                    risk_tv = _tv(generalized["risk_category"], released["risk_category"])
                    merchant_tv = _tv(generalized["merchant_type"], released["merchant_type"])
                else:
                    risk_tv = merchant_tv = np.nan
                for lval in ls:
                    for tval in ts:
                        diagnostic_started = time.perf_counter()
                        diag = privacy_diagnostics(released, DEFAULT_QI, categorical_t=float(tval),
                                                   l_threshold=int(lval))
                        diagnostic_elapsed = time.perf_counter() - diagnostic_started
                        elapsed = (data_generation_elapsed + generalization_elapsed +
                                   suppression_elapsed + diagnostic_elapsed)
                        records.append({
                            "seed": int(seed), "dataset_size": int(n),
                            "age_amount_location_time_level": int(level),
                            "k": int(k), "l_threshold": int(lval), "t_threshold": float(tval),
                            "input_groups": kdiag["input_groups"],
                            "groups_suppressed": kdiag["suppressed_groups"],
                            "released_groups": diag["groups"],
                            "suppressed_records": kdiag["suppressed_records"],
                            "retained_records_pct": round(kdiag["retained_records_pct"], 6),
                            "suppression_loss_pct": round(100-kdiag["retained_records_pct"], 6),
                            "k_satisfied": kdiag["k_satisfied"],
                            "minimum_distinct_sensitive": diag["distinct_l"],
                            "l_satisfying_pct": diag["l_satisfying_pct"],
                            "worst_categorical_tv": diag["worst_categorical_tv"],
                            "t_satisfying_pct": diag["t_satisfying_pct"],
                            "risk_category_frequency_tv_drift": risk_tv,
                            "merchant_frequency_tv_drift": merchant_tv,
                            "dataset_generation_seconds": data_generation_elapsed,
                            "generalization_seconds": generalization_elapsed,
                            "suppression_seconds": suppression_elapsed,
                            "diagnostics_seconds": diagnostic_elapsed,
                            "end_to_end_configuration_seconds": elapsed,
                            "dataset_quality": quality,
                            "formal_privacy_guarantee": False,
                        })
    result = pd.DataFrame(records)
    frontier = pareto_frontier(result)
    result.to_csv(out / "privacy_sweep.csv", index=False)
    result.to_json(out / "privacy_sweep.json", orient="records", indent=2)
    frontier.to_csv(out / "pareto_frontier.csv", index=False)
    frontier.to_json(out / "pareto_frontier.json", orient="records", indent=2)

    packages = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "random_seeds": [int(s) for s in seeds],
        "dataset_size_per_seed": int(n), "method": "generalization + whole-class k suppression",
        "parameter_grid": {"k": ks, "l_diagnostic_thresholds": ls,
                           "categorical_tv_t_thresholds": ts, "hierarchy_levels": levels},
        "objective_definition": {
            "maximize": ["retained_records_pct"],
            "minimize": ["worst_categorical_tv"],
            "pareto_scope": "empirical configurations evaluated in this sweep only",
        },
        "pareto_excluded_invalid_objective_rows": int((~_valid_pareto_objectives(result)).sum()),
        "utility_metrics": ["retained_records_pct", "suppression_loss_pct",
                            "risk_category_frequency_tv_drift", "merchant_frequency_tv_drift"],
        "privacy_metrics": ["k_satisfied", "l_satisfying_pct", "worst_categorical_tv",
                            "t_satisfying_pct"],
        "l_t_values_are_diagnostic_thresholds": True,
        "formal_privacy_guarantee": False,
        "dataset_provenance": "generated synthetic transactions; no external/provider records",
        "output_dir": str(out.resolve()),
        "output_files": list(output_files),
        "packages": packages, "status": "completed", "result_rows": int(len(result)),
    }
    (out / "privacy_sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_pareto(result, frontier, out / "pareto_frontier.png")
    return result, frontier


def plot_pareto(results: pd.DataFrame, frontier: pd.DataFrame, path: str | Path) -> None:
    """Regenerate a retention-vs-categorical-TV scatter from saved sweep tables."""
    fig, ax = plt.subplots(figsize=(8, 5))
    finite = results.dropna(subset=["worst_categorical_tv"])
    ax.scatter(finite["worst_categorical_tv"], finite["retained_records_pct"],
               alpha=.35, s=28, label="Evaluated configurations")
    if len(frontier):
        ax.scatter(frontier["worst_categorical_tv"], frontier["retained_records_pct"],
                   color="#d1495b", s=52, label="Pareto frontier")
    ax.set_xlabel("Worst group categorical TV distance (lower is closer to global)")
    ax.set_ylabel("Records retained (%) (higher is more utility)")
    ax.set_title("Empirical privacy-proxy / retention frontier")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def regenerate_sweep_outputs(output_dir: str | Path = "cpaf_outputs") -> pd.DataFrame:
    """Recompute the frontier and chart from an already saved privacy_sweep.csv."""
    out = Path(output_dir)
    source = out / "privacy_sweep.csv"
    if not source.exists():
        raise FileNotFoundError(f"saved sweep not found: {source}")
    results = pd.read_csv(source)
    frontier = pareto_frontier(results)
    frontier.to_csv(out / "pareto_frontier.csv", index=False)
    frontier.to_json(out / "pareto_frontier.json", orient="records", indent=2)
    plot_pareto(results, frontier, out / "pareto_frontier.png")
    return frontier

