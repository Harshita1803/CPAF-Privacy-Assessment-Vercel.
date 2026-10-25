# Implementation status

Updated 2026-10-24 after final validation. Status describes implemented scope, not the aspirational full framework.

| Requested area | Status | Implemented scope and boundary |
|---|---|---|
| 1. Privacy analysis | PARTIAL | Distinct-l and categorical-TV t diagnostics; explicit missing-value handling; edge-case tests; k/l/t/hierarchy sweeps; retention, group suppression, frequency drift, manifests and Pareto exports. Sweep rejects fractional integer settings; Pareto excludes non-finite/out-of-domain objectives. No l-diversity/t-closeness enforcement transform; t is categorical only. |
| 2. Fairness and utility | PARTIAL | Repeated stratified logistic regression across raw, generalized and generalized+k training; ROC-AUC, PR-AUC, Brier, precision, recall, F1, confusion matrices and mean/std; synthetic age-band TPR/FPR/selection rates, subgroup denominators/warnings, and distribution drift. Small subgroup warnings are emitted; inherited multi-model comparison remains single-split. |
| 3. Privacy attacks | PARTIAL | Shadow-model membership attack with disjoint target/shadow/calibration/evaluation partitions; balanced attack test, AUC, seeded stratified-bootstrap interval, precision/recall/F1 and chance baseline. Synthetic duplicate, rare-QI and nearest-neighbor proxies are exported. One attack family/seed; interval covers test-sampling uncertainty only. |
| 4. Pareto and optimization | PARTIAL | Finite reproducible k/l/t/hierarchy grid; CSV/JSON/manifest; Pareto analysis maximizing retention and minimizing worst categorical-TV distance; plot regeneration from saved CSV. No global optimum claim, caching or broad multi-seed search. |
| 5. Differential privacy audit | PARTIAL | Per-example clipping; OS-backed unseeded mechanism randomness; fixed public delta; RDP step/run composition calculator; parameter, analytic-case, clipping and mock-noise tests; explicit `EXPERIMENTAL_UNVERIFIED` metadata. diffprivlib 0.6.6 pinned setup and smoke script prepared but not installed/run because DNS access is denied; Opacus remains incompatible with the installed secure RNG stack. No formal guarantee is established. |
| 6. Dashboard | PARTIAL | Ten sections in source for evidence, synthetic data, transforms, diagnostics, utility/fairness, attacks, DP audit, sweeps/Pareto, downloads and limitations. Evidence supports schema-validated selection/upload, in-session edit, separate persistent working-copy save, explicit overwrite confirmation with timestamped backup, guarded reload, and shared eligibility checks for status counts and scoring. Dashboard writes unique timestamped experiment folders. Streamlit 1.66.1 imports; local launch was attempted but loopback health was denied by sandbox socket policy and AppTest did not return rendered sections. UI interactions remain unverified. |
| 7. Reproducibility and research quality | PARTIAL | Checked-in JSON config, CLI overrides, deterministic synthetic seeds, per-run manifests with package/runtime versions, synthetic provenance and artifact paths, and fresh-output CLI guard. DP mechanism randomness remains unseeded by design. Repeated split standard deviations are descriptive because test folds can overlap. See `docs/RESEARCH_READINESS.md`. |
| 8. Final validation | PARTIAL | Current offline reruns completed small privacy, repeated ML, shadow attack, and sweep experiments; saved-CSV plot regeneration matched the one-row frontier. Latest suite: 32 tests and compileall passed. Dashboard local health/UI verification was blocked by sandbox socket policy; see `docs/RESULTS.md` and `docs/FACULTY_DEMO_GUIDE.md`. |

## Implemented CLI commands

- --sweep and --sweep-k, --sweep-l, --sweep-t, --sweep-levels, --sweep-seeds, --sweep-n
- --plot-sweep-only rebuilds Pareto outputs from saved privacy_sweep.csv
- --repeated-ml and --repeats
- --attack-audit
- --output-dir isolates each run from prior results
- --config configs/default_experiment.json loads JSON defaults; CLI flags override config values

## Not implemented

Learned synthetic data, l-diversity enforcement, numerical t-closeness, full fairness assessment, repeated comparison of every model family, maintained-library DP integration and proof, global optimizer, caching, consolidated generated PDF report, and independently reviewed app evidence.
