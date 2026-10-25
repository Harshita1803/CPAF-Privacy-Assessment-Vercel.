"""Unverified diffprivlib logistic-regression POC; run only in its optional venv.

This creates a public synthetic dataset, fits one diffprivlib estimator with an
explicit public feature norm and secure SystemRandom, and writes only the fitted
coefficients/intercept plus an accountant manifest. It does not certify a project
guarantee: exact adjacency and library-version theorem assumptions still need review.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np


def run(output_dir: str | Path, seed: int = 2026, epsilon: float = 1.0,
        data_norm: float = 2.5, n: int = 256) -> dict:
    if n < 20 or epsilon <= 0 or data_norm <= 0:
        raise ValueError("n must be >=20; epsilon and data_norm must be positive")

    try:
        from diffprivlib.accountant import BudgetAccountant
        from diffprivlib.models import LogisticRegression
        from diffprivlib.utils import check_random_state
    except ImportError as exc:
        raise RuntimeError(
            "Optional dependency missing. Follow docs/DP_ENVIRONMENT_SETUP.md "
            "in an isolated Python 3.12 virtual environment."
        ) from exc

    # Public, reproducible demonstration data; no real or user-provided rows.
    data_rng = np.random.default_rng(seed)
    X = data_rng.uniform(-1.0, 1.0, size=(n, 5))
    y = ((X[:, 0] + 0.35 * X[:, 1] - 0.15 * X[:, 2]) > 0).astype(int)

    # Row-wise clipping to a public norm; the configured bound is public and
    # comfortably exceeds the maximum norm for these five bounded features.
    row_norms = np.linalg.norm(X, axis=1)
    scale = np.minimum(1.0, data_norm / np.maximum(row_norms, 1e-12))
    X = X * scale[:, None]
    if np.unique(y).size != 2:
        raise RuntimeError("Synthetic smoke data unexpectedly contains one class")

    accountant = BudgetAccountant(epsilon=epsilon, delta=0.0)
    secure_rng = check_random_state(None, secure=True)
    model = LogisticRegression(
        epsilon=epsilon,
        data_norm=data_norm,
        random_state=secure_rng,
        accountant=accountant,
        fit_intercept=True,
        warm_start=False,
        n_jobs=1,
        max_iter=100,
    )
    model.fit(X, y)

    spent = accountant.total()
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ["diffprivlib", "numpy", "scipy", "scikit-learn", "joblib"]
        },
        "mechanism": "diffprivlib LogisticRegression objective perturbation via Vector mechanism",
        "protected_unit": "one (feature vector, binary label) row supplied to fit",
        "adjacency_definition": "EXPERIMENTAL_UNVERIFIED: verify library theorem's neighboring-dataset relation before claiming",
        "dataset": "public synthetic demonstration only; generated in script",
        "dataset_size": n,
        "dataset_seed": seed,
        "data_norm_bound": data_norm,
        "epsilon_configured": epsilon,
        "delta_configured": 0.0,
        "accountant": "diffprivlib BudgetAccountant",
        "accountant_spent_epsilon": float(spent[0]),
        "accountant_spent_delta": float(spent[1]),
        "randomness": "diffprivlib check_random_state(None, secure=True), passed as SystemRandom",
        "optimizer": "L-BFGS-B objective perturbation; max_iter=100; actual iteration count not released",
        "released_model_parameters": {
            "coef": np.asarray(model.coef_).tolist(),
            "intercept": np.asarray(model.intercept_).tolist(),
        },
        "verification_status": "EXPERIMENTAL_UNVERIFIED",
        "formal_guarantee_claimed": False,
        "scope": "single estimator parameter release only; no metrics, attacks, tuning, preprocessing statistics, or repeated fit included",
        "unaccounted_operations": "This script's status requires independent review of the installed package theorem, exact adjacency, secure RNG propagation, and implementation/version before any VERIFIED_BOUND claim.",
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diffprivlib_smoke_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="cpaf_outputs/dp_library_poc")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--data-norm", type=float, default=2.5)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    result = run(args.output_dir, args.seed, args.epsilon, args.data_norm, args.n)
    print(json.dumps({k: v for k, v in result.items() if k != "released_model_parameters"}, indent=2))
    print(f"Result written to {Path(args.output_dir) / 'diffprivlib_smoke_result.json'}")


if __name__ == "__main__":
    main()
