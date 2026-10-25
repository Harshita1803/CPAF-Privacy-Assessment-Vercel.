# Final independent research-quality audit

Audit date: 2026-10-24. Scope: inspect current source, unit tests, configuration, saved experiment artifacts and documentation; reproduce small privacy, ML, membership-inference and parameter-sweep runs; run tests and compilation; check dashboard startup/health where the environment permits.

## Methodology

Inspected `cpaf/core.py`, `cpaf/evaluation.py`, `cpaf/attacks.py`, `cpaf/experiments.py`, `cpaf_ml.py`, the CLI/dashboard, tests, `configs/default_experiment.json`, run manifests and CSV/JSON/PNG outputs. Compared the 1,000-row k=5 CSV diagnostics with the grouping and suppression source. Re-ran the feature tests, compiled project modules, and ran fresh experiments into `cpaf_outputs/audit_verified/` so prior runs and deliverables remain intact. The existing reports, slide deck and backup directory were not edited.

## Findings and fixes

### Privacy transformations

- The 1,000-row, k=5 result retaining 11 records is reproducible and is an expected consequence of whole-class suppression over five generalized QIs (`age_band`, `location`, `merchant_type`, `amount_band`, `time_band`). The release contains 2 groups, minimum group size 5, and measured k=5 is satisfied for the released rows. The suppression diagnostics count groups on the input and the minimum on the release; empty releases are explicitly treated as vacuously k-satisfying, with zero groups and zero rows.
- l-diversity and categorical t-closeness are diagnostics, not enforced transforms. In the 11-row release, both groups contain only one sensitive value: l=2 passes 0% of groups. t=0.2 passes 100% because each group matches the released-data global sensitive distribution (TV=0); this is not a contradiction and does not establish l-diversity.
- Sweep frequency drift compares the same categorical variable pre/post suppression using total variation over each table's observed category frequencies. Group t diagnostics use the released-data global distribution as the reference, group-local normalized frequencies as the comparison, and TV's one-half L1 definition. Numeric drift uses one-dimensional Wasserstein distance divided by the raw reference range; empty selected data produces NaN, not a fabricated zero. The denominator is zero only for constant reference columns, which are explicitly reported as zero normalized distance.
- Edge tests cover empty releases, missing QIs as explicit groups, and rejection of missing sensitive values. No defect was demonstrated in suppression for the investigated run.

### ML validity and fairness

- Repeated utility splits use `StratifiedShuffleSplit` with the same held-out indices across representations. The synthetic target is generated from age, amount, hour and merchant plus independent random noise; `risk_category` is excluded from predictors. This is not target leakage in the split, but it is an intentionally constructed target: utility measures recovery of that synthetic rule, not a naturally observed outcome.
- Preprocessing is inside sklearn pipelines and fit on each training partition. Suppression determines training rows only; the common held-out population remains the test population. Record IDs are excluded. No duplicate-row train/test contamination was identified in the inspected pipeline; synthetic generation also uses continuous-valued transaction amounts, making exact feature duplicates uncommon, but it does not constitute a general deduplication proof.
- Per-repeat test predictions and labels are used for ROC-AUC, average precision (PR-AUC), Brier, thresholded metrics and confusion matrices. The smoke run used n=500 and two repeats; its ROC-AUC means (std) were Raw 0.7196 (0.0113), Generalized 0.7722 (0.0011), and Generalized+k=2 0.7513 (0.1484). These are small synthetic execution checks, not submission-grade estimates. F1 was 0 in all three means at threshold 0.5, indicating the low-sample/threshold combination did not produce positive predictions; ROC/PR metrics should not be read alone as operational performance.
- Fairness rows include group n, positive and negative denominators and a warning when n<30 or either outcome denominator is <10. All 18 age-band rows in the smoke run were flagged. Age groups and gaps describe only this synthetic generator; no real-population age fairness claim is supported.
- Distribution drift uses the same raw training records as the pre-suppression reference and the retained subset as candidate. Numeric Wasserstein values are normalized by the reference column range; categorical shifts use TV. The metrics quantify selection change, not causation or general-purpose utility.

### Membership inference and disclosure proxies

- Target train, shadow train, shadow nonmember calibration, and target nonmember evaluation partitions are constructed from disjoint index sets and asserted pairwise disjoint. The shadow attack is trained on shadow-model confidence, true-label loss and entropy; member=1 and nonmember=0. It is evaluated on balanced, unseen target-model member/nonmember records. Attack output is the attack classifier probability of membership, so ROC-AUC direction is correctly oriented.
- Added a deterministic stratified bootstrap 95% interval (1,000 resamples, independently resampling member and nonmember test records) around attack-test ROC-AUC. For n=3,000, seed 19, k=2, AUCs (95% intervals) were Raw 0.5050 [0.4723, 0.5389], Generalized 0.5067 [0.4755, 0.5402], and Generalized+k=2 0.5295 [0.4995, 0.5635], against AUC chance 0.5. The intervals include 0.5. This is one shadow-trained confidence/loss/entropy attack on synthetic data and does not establish privacy or rule out other attacks.
- Duplicate, exact-QI rarity and mixed-feature nearest-neighbor measures are synthetic diagnostics, not record linkage. Exact amount in the rare-QI definition is high-cardinality, so its rarity rate is structurally inflated.

### Pareto sweep

- The finite 24-row sweep evaluates retention (maximize) and worst categorical-TV distance (minimize); strict dominance uses both objectives and excludes exact ties from domination. NaN objective rows (e.g., fully suppressed releases with undefined group TV) are dropped explicitly from the frontier. In the recorded run all 24 configurations had finite objectives; 16 were non-dominated.
- The frontier and 51,264-byte PNG were regenerated from the saved `privacy_sweep.csv`; regenerated frontier row count was 16. Eight of 24 rows had no released groups, so their worst-TV objective was NaN and they were explicitly excluded from the frontier; the remaining 16 finite-objective rows were all non-dominated. This is only a grid-scoped empirical frontier, not a global optimizer. l and t sweep values alter diagnostic thresholds, not the underlying release.

### Differential privacy

- Source inspection confirms per-example logistic-loss gradients are L2-clipped at C, Gaussian noise with standard deviation sigma*C is added to the clipped sum, Bernoulli inclusion uses q, and the update divides by q*n. The code's integer-order subsampled-Gaussian RDP expression and composition/conversion are custom. It uses orders 2–128, T-fold composition and delta=1/n in the experiment.
- For add/remove adjacency, a clipped sum has nominal sensitivity C; replace-one can require 2C. The implementation's denominator depends on the supplied row count and therefore is not justified for variable-size neighboring datasets. Data-dependent selection before DP training and pipeline/preprocessing assumptions also prevent an end-to-end guarantee. The accountant was not independently verified against a maintained implementation. Therefore epsilon is experimental/unverified and no formal epsilon guarantee is claimed. No mathematical correctness change to the accountant was justified by this audit; existing tests check domain, monotonic composition/noise behavior, and fitting validation, not proof of the accountant.

## Fixes made

- Added age-subgroup warning and positive/negative denominators to fairness CSVs and summary rows.
- Added deterministic stratified-bootstrap AUC intervals to shadow membership-inference output and a regression assertion that intervals bracket the point estimate.
- Added a test assertion for fairness denominator/warning columns. The changes do not reinterpret synthetic results as real-world evidence or as formal privacy guarantees.

## Actual commands executed and outcomes

Commands were run from the project root with bundled Python 3.12.14 at `C:\Users\harsh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`:

```powershell
$py = 'C:\Users\harsh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -m unittest discover -s tests -v
& $py -m compileall -q cpaf cpaf_advanced.py cpaf_ml.py app.py tests
& $py cpaf_advanced.py --n 1000 --k 5 --l 2 --t 0.2 --no-ml --output-dir cpaf_outputs\audit_verified\privacy
& $py cpaf_advanced.py --repeated-ml --n 500 --repeats 2 --k 2 --seed 31 --output-dir cpaf_outputs\audit_verified\ml
& $py cpaf_advanced.py --attack-audit --n 3000 --k 2 --seed 19 --output-dir cpaf_outputs\audit_verified\attack
& $py cpaf_advanced.py --sweep --sweep-n 600 --sweep-k 1 5 20 --sweep-l 2 3 --sweep-t 0.1 0.2 --sweep-levels 1 2 --seed 123 --output-dir cpaf_outputs\audit_verified\sweep
& $py cpaf_advanced.py --plot-sweep-only --output-dir cpaf_outputs\audit_verified\sweep
```

All 20 unittest cases passed after changes. `compileall` exited 0. Privacy, repeated ML, attack, 24-row sweep, and saved-CSV plot regeneration all completed. Pareto count matched the saved frontier CSV (16). Dashboard startup was attempted with Streamlit 1.66.1, but the health request failed with a Windows socket permission error (`An attempt was made to access a socket in a way forbidden by its access permissions.`); dashboard availability is therefore not claimed in this audit. An earlier stored RESULT entry records a prior successful health check, but this audit did not reproduce it.

## Remaining blockers and limitations

- No formal DP guarantee: custom accountant, dataset-size normalization/adjacency, selection, and preprocessing assumptions need independent review and a justified end-to-end mechanism.
- The generator and target are artificial. Fairness and model metrics are not real payment-app evidence; age-band comparisons are descriptive and often underpowered.
- The MIA covers one attack family and seed. The bootstrap interval captures test-sample uncertainty only, not attack-family or seed uncertainty.
- l-diversity and t-closeness are diagnostics only; t is categorical TV, not numerical t-closeness. k holds only for the exact declared QIs and released table.
- Dashboard HTTP health could not be verified in the current restricted socket environment; page-by-page interaction was not audited.
- Evidence scoring is separate from synthetic experiments and remains unscored absent reviewed, traceable source evidence.

## Final assessment

The project is suitable as a carefully qualified 20-mark academic prototype if submitted with the stated synthetic-data scope, reproducibility artifacts, and limitations. It is not suitable for claims of formal differential privacy, empirical privacy protection for real payment-app users, or real-population fairness. The extreme 1.1% retention at n=1,000 should be discussed as a sparsity/utility outcome, not hidden or generalized away. The submission should present measured results as demonstrations and clearly mark the custom DP accounting and the single-seed attack as incomplete research components.

## Differential privacy follow-up audit

Subsequent DP audit work is recorded in `docs/DP_AUDIT.md`, `docs/DP_METHODOLOGY.md`, and `docs/DP_THREAT_MODEL.md`. DP sampling and Gaussian noise had been tied to a public reproducibility seed. The code now uses OS-backed `secrets.SystemRandom`, rejects a DP seed, uses fixed public delta=1e-5 per candidate, exposes `EXPERIMENTAL_UNVERIFIED`/`NOT_APPLICABLE` result status, and records mechanism/accountant/adjacency/scope/unaccounted-operation metadata. MIA sample counts now cap at available disjoint pools and the manifest reports requested and actual counts.

The latest DP command completed: `cpaf_advanced.py --ml-only --n 3000 --k 2 --seed 19 --output-dir cpaf_outputs/dp_audit_verified`. Five candidate rows were saved, each marked experimental with formal guarantee false. The MIA sample size was 900 members and 900 nonmembers. The custom accountant, floating-point Gaussian implementation, upstream stratified selection, size-dependent normalization, model comparison and private-derived metrics remain blockers; no formal DP guarantee is claimed. The latest suite has 25 passing tests and compileall exits 0. Earlier n=3,000 and n=5,000 attempts failed before the attack pool was capped and are documented in `docs/RESULTS.md`.
