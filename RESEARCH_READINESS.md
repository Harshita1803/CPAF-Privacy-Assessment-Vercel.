# CPAF 2.0 research readiness audit

Audit date: 2026-10-24. This report describes the source and artifacts inspected in this checkout and the final small reruns under `cpaf_outputs/readiness_20261024_release/`. It does not rely on a prior completion summary as validation.

## Research question and primary target

**Bounded question:** On this project's generated transaction-like table, how do fixed generalization and whole-equivalence-class k-suppression affect retained-record rates and categorical privacy diagnostics, and how does logistic-regression predictive utility change on a held-out synthetic target?

**Recommended primary predictive target:** ROC-AUC for the synthetic binary label on stratified held-out records, comparing the same logistic-regression pipeline trained on raw, generalized, and generalized+k-suppressed training representations. Report per-repeat values and mean/standard deviation. This target is already implemented by `cpaf.evaluation.run_repeated_utility`; it narrows the runnable experiment for reporting and does not replace the separate provider-evidence question. The label is generated probabilistically from age, amount, hour, merchant type, and random noise; most of those fields also enter the model. This is an intentionally controlled, partly circular synthetic task, not evidence of real-world prediction quality.

## Data and scope

- **Synthetic experiment track:** all rows are generated in code (`cpaf.core.generate_synthetic`); the classification label is generated in `cpaf_ml.add_synthetic_label`. No external or provider/customer dataset is used.
- **Provider evidence track:** Paytm, Google Pay, and PhonePe have seven scoring dimensions: Data Minimisation; Consent & Transparency; Third-Party Sharing; Security Controls; Anonymisation & Re-identification; Data Lifecycle & User Rights; DPDP Alignment. A score row is included only when marked `reviewed` and it has an HTTP(S) source URL, access date, claim, reviewer, and score rationale. Synthetic experiments do not score providers.
- **Primary baselines:** raw logistic regression; generalized logistic regression; generalized plus k-suppressed logistic regression, all evaluated on common held-out rows within each repeat. The legacy single-split model-family comparison is secondary.

## Defects corrected in this audit

- The evidence scorer previously accepted numeric app scores without a reviewed status or traceable provenance. Scoring now requires `reviewed`, an HTTP(S) URL, access date, claim, reviewer, and rationale; a regression test checks both rejected and accepted cases. This gate checks metadata completeness, not source truth.
- Sweep input validation previously truncated fractional integer-like settings through `int(...)`; it now rejects fractional, nonfinite, and malformed integer settings. Pareto filtering now excludes nonfinite and out-of-domain retention/TV objectives, and the manifest counts those excluded rows.
- Microaggregation could accept empty/nonfinite numeric values and propagate NaN distortion; it now rejects empty, missing/nonfinite inputs and invalid group sizes, with regression tests.
- Repeated CLI runs could overwrite existing result folders. New CLI runs and experiment writers now refuse matching existing artifacts; dashboard experiment actions create unique timestamped folders. Plot-only regeneration remains an explicit derived-output rewrite from the saved sweep CSV.
- The synthetic target's feature overlap and repeated-split overlap were insufficiently prominent in prior guidance; they are stated as limitations here and in `docs/LIMITATIONS.md`.

## Method inventory and evidence boundaries

| Method | Input → output and implementation | Assumption / test coverage | Limitation |
|---|---|---|---|
| Synthetic generator | Seed, n → synthetic table; `cpaf/core.py:generate_synthetic` | Fixed schema, artificial distributions, reproducibility/schema tests | No population validity or provider provenance |
| Generalization | Raw attributes → fixed age/location/amount/time hierarchy; `cpaf/core.py:generalize` | Levels 0–2; shared hierarchy for chosen columns | Not a privacy guarantee; fixed hierarchy can retain identifying combinations |
| k-anonymity suppression | Generalized QIs and k → full small classes removed; `cpaf/core.py:suppress_k` | Group size on exact declared QIs; tests include empty release and min-size checks | Retention can collapse; property applies only to the released rows and selected QIs |
| l-diversity | Released groups and sensitive attribute → minimum distinct value and group pass rate; `cpaf/core.py:privacy_diagnostics` | Missing QI is explicit; missing sensitive value rejected; edge tests | Diagnostic only; distinct-count l does not prevent skew or semantic similarity; no enforcing transform |
| t diagnostic | Released groups → categorical total variation to released-data global distribution; same function | Denominator is released global distribution; empty and missing values explicit | Categorical TV proxy, not full/general t-closeness and not an enforcement transform |
| Microaggregation | Numeric columns → per-column sorted block means; `cpaf/core.py:microaggregate` | Finite nonempty numeric inputs; tests cover partial block and invalid/missing values | Univariate blocks can destroy correlations; not part of k release path |
| Repeated ML utility | Synthetic raw rows and generated label → per-repeat ROC-AUC/PR-AUC/Brier/F1 etc.; `cpaf/evaluation.py` | Stratified repeated 75/25 splits; preprocessing pipeline fits on training data; raw/generalized/suppressed train views share test rows | Test splits overlap across repeats; standard deviation is descriptive, not an independent-sample CI; target is generated from model features |
| Fairness | Held-out synthetic labels/predictions + age band → selection, TPR/FPR and gaps; `cpaf/evaluation.py` | Explicit positive/negative denominators and small-subgroup warnings; tests check exported counts | Small subgroup rates unstable; age groups and target are synthetic, no real-population inference |
| Membership inference | Disjoint target/shadow/calibration/attack-evaluation partitions → attack AUC and test-sampling bootstrap; `cpaf/attacks.py` | Pairwise disjoint splits, balanced member/nonmember test, .5 baseline and score direction from trained attack classifier | One attack family, one seed; bootstrap omits training/attack-design uncertainty; AUC below .5 is not privacy evidence |
| Disclosure proxies | Synthetic rows → exact duplicate/rare-QI/nearest-neighbor summaries; `cpaf/attacks.py` | Declared QIs and scaling are recorded | Synthetic uniqueness/similarity only; not actual re-identification risk |
| Sweep / Pareto | Finite synthetic parameter grid → CSV/JSON and retained-vs-worst-TV frontier; `cpaf/experiments.py` | Maximize retention, minimize worst categorical TV; invalid/nonfinite objective rows excluded; ties retained | Grid-specific empirical frontier, no global optimum and no formal privacy axis |
| Custom DP-SGD | Encoded fit rows → noisy logistic weights and custom diagnostic accountant; `cpaf_ml.py` | Per-example clipping, Bernoulli sampling, Gaussian noise, fixed steps; tests cover parameter guards, analytic cases and test RNG | Custom accountant and finite-precision randomness unverified; caller operations/metrics/releases outside any proof; status stays `EXPERIMENTAL_UNVERIFIED` |
| Evidence score | Reviewer-entered source-backed rows → evidence-adjusted, weighted scores; `cpaf_advanced.py` | Requires `review_status=reviewed`, HTTP(S) URL, access date, claim, reviewer, rationale; tests prevent scoring unreviewed entries | Metadata checks cannot verify that a source claim is true; human scoring judgment/weights remain subjective; current templates are unscored |

## Reproduced in this audit

Commands used the bundled Python 3.12.14 interpreter and fresh output paths:

```powershell
$py = 'C:\Users\harsh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py cpaf_advanced.py --n 300 --k 2 --l 2 --t 0.2 --no-ml --seed 6 --output-dir cpaf_outputs\readiness_20261024_release\privacy
& $py cpaf_advanced.py --repeated-ml --n 500 --repeats 2 --k 2 --seed 31 --output-dir cpaf_outputs\readiness_20261024_release\ml
& $py cpaf_advanced.py --attack-audit --n 1000 --k 2 --seed 19 --output-dir cpaf_outputs\readiness_20261024_release\attack
& $py cpaf_advanced.py --sweep --sweep-n 150 --sweep-k 1 5 --sweep-l 2 --sweep-t 0.2 --sweep-levels 1 --seed 7 --output-dir cpaf_outputs\readiness_20261024_release\sweep
& $py cpaf_advanced.py --plot-sweep-only --output-dir cpaf_outputs\readiness_20261024_release\sweep
```

All five commands exited 0. Privacy retained 55/300 (18.33%) rows, with minimum released group size 2; l=2 passed 0% of groups, while categorical t=0.2 passed 100% with worst TV=0. This is a diagnostic for that generated release only. The two-repeat ML run produced ROC-AUC mean (sample SD): raw 0.7196 (0.0113), generalized 0.7722 (0.0011), generalized+k=2 0.7513 (0.1484); the k-suppressed result is highly variable at this small n. Attack AUCs were 0.4951, 0.4763, and 0.4939 with test-bootstrap 95% intervals [0.4338, 0.5486], [0.4224, 0.5325], and [0.4031, 0.5801]; all four partitions were disjoint. The two-row sweep had one non-dominated row; saved-CSV plot regeneration returned one frontier row and wrote a 51,565-byte PNG.

Regression validation: 32 tests passed; `compileall` exited 0. A repeat of the n=100 privacy CLI run into the same folder was rejected with exit 2, and SHA-256 of its saved manifest was unchanged. Offline package metadata reports diffprivlib absent and Streamlit 1.66.1 installed. The app was **not launched** because this task explicitly prohibited network access; therefore no current dashboard runtime/interaction claim is made. No DP result file was generated by the repeated-ML rerun.

## Reproducibility checklist

- [ ] Use the same documented interpreter and pinned project dependencies; do not install optional DP packages into the project runtime.
- [ ] Use a new, empty output directory for every experiment. The CLI now refuses directories containing prior artifacts. Dashboard actions create timestamped run folders.
- [ ] Record the full command/config file, n, seeds (including label seed derived as experiment seed + 1), k/l/t/hierarchy/repeats, package versions, timestamp, and output folder in the run manifest.
- [ ] Confirm manifests identify generated data as synthetic and list expected artifacts; retain CSV/JSON raw outputs separately from generated summaries/plots.
- [ ] Recompute summary values from per-run CSVs; regenerate Pareto plots from `privacy_sweep.csv` and compare frontier row counts.
- [ ] Record failed, skipped, or unavailable experiments explicitly. No network attempt or diffprivlib run is required for offline reproduction.
- [ ] Keep privacy diagnostics, attack outcomes, and mathematical privacy guarantees separate in all reporting.

## Limitations and publication work

The synthetic label's feature overlap makes predictive utility a controlled generator-recovery exercise. Overlapping repeated test folds weaken interpretation of across-repeat SD as uncertainty. Fairness is synthetic and small groups are warned, not repaired. MIA estimates only one attack and one seed. k/l/t diagnostics cannot establish DP or protect unlisted QIs. The custom DP accountant/mechanism remains experimental; diffprivlib is not installed or integrated, and no formal epsilon guarantee is justified. Provider evidence remains empty until reviewed sources are populated. Raw experiment CSVs, summary CSV/JSON files and plots are still colocated in each run folder rather than physically separated into raw/derived subfolders; they are identified individually in manifests. Skipped settings are recorded, but a process failure before a writer creates its final manifest may be represented only by the terminal error, not a durable failure manifest.

The README heading and Streamlit page title now use **CPAF 2.0**, consistent with the source modules and project artifacts. Existing reports and presentation files were not rewritten. The research scope remains the two-track evidence/synthetic framework described above.

Before publication: define the intended population and external evidence protocol; document independent validation of app evidence and inter-rater reliability; obtain a relevant external dataset or restrict claims to synthetic software validation; consider an independent non-overlapping evaluation design and multiple seeds; and have a DP specialist review any future claimed mechanism and release boundary.

## Dashboard follow-up

The previous paragraph records the earlier readiness audit. In the later faculty-demo check, Streamlit 1.66.1 was available and a local server launch was attempted, but the sandbox denied the loopback health request. In-process AppTest did not return rendered elements. The current session therefore has no verified landing-page or section interaction; details and the exact socket error are recorded in `docs/RESULTS.md`.
