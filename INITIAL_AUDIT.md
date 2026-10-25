# Initial codebase audit

Audit date: 2026-10-24. Scope: both Python scripts, requirements, README, evidence templates, existing output files, and project directory layout.

## Inventory and execution paths

- `cpaf_advanced.py` is the CLI (`--n`, `--k`, `--seed`, `--template-only`, `--no-ml`, `--ml-only`). It generates synthetic transactions, generalizes age/amount/time, suppresses groups below k, computes a small set of metrics, and optionally invokes `cpaf_ml.py`.
- `cpaf_ml.py` generates a separate artificial binary target, trains logistic regression, random forest and histogram gradient boosting, reports a k sweep, confidence MIA and custom DP-SGD. It writes CSV/JSON/PNG files under `cpaf_outputs/`.
- A duplicate `cpaf_ml.py` exists at the workspace root. The nested project copy is the one imported by the project CLI.
- Requirements list pandas, NumPy, matplotlib, python-pptx, python-docx, reportlab, scikit-learn and SciPy without upper bounds or a tested Python version. There is no test suite, dashboard, package structure, or documentation beyond the README.
- Existing `cpaf_outputs/` contains previously generated synthetic CSVs, metric tables, ML results, JSON and charts. These outputs are preserved during implementation. Word/PDF/presentation files and ZIP are user-provided artifacts and were not modified.

## Correctness and research-validity findings

1. The generator validates neither n nor seed, schema, ranges, duplicate IDs, nor missing values. The privacy script has `risk_category`, but ML generates a separate target, so this target provenance is not manifest-tracked.
2. `generalise()` uses hard-coded hierarchies and removes `record_id`, city, exact amount and hour. `enforce_k_anonymity()` retains equivalence classes of size at least k; it is suppression, not a transformation to make undersized groups valid. Metric computation on the returned release can verify k for the chosen QIs, but the `k_anonymity` metric is just minimum class size and does not report violating groups or sizes.
3. `privacy_metrics()` calculates distinct sensitive values and mean categorical total variation. It does not calculate l-diversity compliance, categorical t-closeness compliance/worst case, numerical distribution distances, or statistical utility against aligned reference data. The existing diversity metric is not a guarantee.
4. The ML label in `cpaf_ml.py` is derived from amount, hour, merchant and age. `risk_category` is excluded from the generalized model view, which avoids one direct leakage path, but generalization of predictors while testing on raw-view test records mixes representation and train/test distribution. Only one train/test split is used; there is no validation split, F1/precision/recall/confusion matrix, fairness audit or interval estimate.
5. MIA compares train confidence on records used to fit the attacked model with test confidence. That confidence attack is a baseline, but member/non-member selection can fail for small n and does not isolate attack calibration. Output labels imply broader attack coverage than is implemented.
6. DP-SGD clips per-example gradients and uses Poisson inclusion sampling with Gaussian noise on the summed gradient. The custom integer-order subsampled-Gaussian RDP accountant has no tests or parameter validation; `delta` is implicitly set to 1/n of a small attacked subset, and the model is trained on a deliberately sampled member subset rather than the stated full training set. Therefore prior epsilon values are experimental and must not be presented as validated formal guarantees. A formal guarantee is withheld pending independent accountant validation.
7. App scoring correctly leaves an empty template unscored and discounts weak evidence toward neutral, but lacks required claim/source excerpts/reviewer/status fields, weight sensitivity analysis, and strict validation. Missing dimensions are excluded from the weighted average and coverage is separate; this is more defensible than imputing an adverse/favorable score.
8. No test files or runtime validation were present. At audit time, Windows `py.exe` and `python.exe` resolve only to inaccessible Microsoft Store aliases in this execution environment, so Python version/dependency compatibility and executable tests cannot be confirmed here.

## Existing data and integrity

The template contains empty scores, appropriately avoiding fabricated app findings. Existing experiment outputs are retained as historical outputs; they are not silently regenerated as part of the audit. The repository has no real customer transaction dataset in its inspected file inventory. All newly generated example records remain synthetic.

## Backup

Before source edits, copies of the root `cpaf_ml.py` and both nested Python scripts were saved under `backups/20261024-170430/`.

## Implementation scope and unresolved items

The upgrade adds a validated reusable privacy core, explicit k/l/t diagnostics, numeric microaggregation, an auditable evidence template and scoring sensitivity utilities, experiment manifests/exports, test coverage, methodology/threat-model documentation, and an optional Streamlit entry point. Formal DP accounting, advanced learned synthetic data, rigorous MIA calibration, and app-specific evidence research remain out of scope unless independently validated; they are described as limitations rather than represented as complete.
