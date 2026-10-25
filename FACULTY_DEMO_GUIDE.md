# CPAF 2.0 — Faculty Demonstration Guide

## Two-minute introduction

CPAF 2.0 is an educational research prototype with two separate tracks. The first is a structured review register for public privacy disclosures from Paytm, Google Pay, and PhonePe. The second is a synthetic-data lab for experimenting with generalization, k-anonymity-style suppression, diagnostic l-diversity and categorical t-closeness, model utility, synthetic subgroup metrics, membership-inference attacks, and parameter sweeps.

The tracks are intentionally kept separate: generated data are not payment-app customer data, and the synthetic results do not measure any provider's real practices. Provider scores stay blank until a reviewer enters and checks traceable evidence. The custom DP-SGD code is a teaching experiment marked `EXPERIMENTAL_UNVERIFIED`; it makes no formal differential-privacy claim. This demo presents reproducible software behavior and its limits, not proof of privacy for real financial records.

## Problem statement and objective

Transaction-like datasets can expose patterns through detailed attributes. This project demonstrates how fixed generalization and whole-equivalence-class suppression change record retention and group diagnostics, and how the same transformations affect prediction on a controlled artificial target.

The runnable primary evaluation target is synthetic-label ROC-AUC across raw, generalized, and generalized+k-suppressed training data, evaluated on common stratified holdout splits. The synthetic label is probabilistically generated from age, amount, transaction hour, merchant type, and noise; the models see most of those same predictors. This makes the evaluation partly circular and useful as a controlled code exercise, but unsuitable as evidence of real-world fraud prediction.

## Architecture and data flow

1. `cpaf.core` creates deterministic synthetic transaction-like rows and implements validation, fixed generalization, whole-group k suppression, l/t diagnostics, and numeric microaggregation.
2. `cpaf.evaluation` generates the separate artificial prediction label, makes repeated stratified splits, fits preprocessing only on each training split, and exports utility/fairness CSVs and manifests.
3. `cpaf.attacks` creates disjoint target/shadow/calibration/evaluation partitions, trains one shadow attack, and exports attack and synthetic disclosure-proxy results.
4. `cpaf.experiments` evaluates finite parameter grids, filters Pareto objectives, and writes CSV/JSON/PNG outputs. The plot can be rebuilt from the saved sweep CSV.
5. `cpaf_ml.py` contains the legacy model comparison and experimental custom DP-SGD/accounting path.
6. `cpaf_advanced.py` is the CLI and provider evidence scorer; `app.py` calls shared package methods and presents the two tracks in separate dashboard sections.

Experiment outputs are stored under `cpaf_outputs/<run-folder>/`. Use a fresh folder for each CLI run. Dashboard utility, attack, and sweep buttons create timestamped run folders. Manifests record configuration, seeds, software versions, data provenance, status, and artifact paths.

## Provider evidence dimensions

The evidence register covers these seven dimensions:

1. Data Minimisation
2. Consent & Transparency
3. Third-Party Sharing
4. Security Controls
5. Anonymisation & Re-identification
6. Data Lifecycle & User Rights
7. DPDP Alignment

Rows contribute to a provider score only when marked `reviewed` and supplied with an HTTP(S) source URL, access date, claim, reviewer, score rationale and valid score. This is a metadata completeness gate, not a guarantee that the interpretation is correct. The current working register has 17 AI-assisted documentary scores and four explicitly unscored rows. Treat all scores as provisional assessments of public statements, not technical audits or DPDP compliance findings; the Paytm anonymisation row and all DPDP-alignment rows remain unscored. See `docs/EVIDENCE_REVIEW_20261024.md` for sources and caveats.

## Synthetic data and transformations

The generator samples age 18–79, region, city, merchant type, amount, and hour from fixed artificial distributions. `risk_category` is a synthetic sensitive label derived from amount/hour/merchant rules. The model prediction label is generated separately with a probabilistic rule. There is no external or payment-app dataset.

- **Generalization:** replace exact age, location, amount, and hour with fixed hierarchy bands.
- **k suppression:** remove every full quasi-identifier group with fewer than k records. Check minimum group size on the actual released table and the stated QIs; report retention with it.
- **l-diversity diagnostic:** count distinct sensitive values in each released group and report the minimum and proportion passing a selected l. It does not enforce l-diversity.
- **t-closeness diagnostic:** compute categorical total-variation distance from each released group to the released-table global distribution. It is not a numerical/continuous t-closeness implementation and does not enforce t.
- **Microaggregation:** independently sort selected numeric columns into consecutive blocks and replace each block with its mean. It may distort relationships between columns and is a simple baseline.

k/l/t describe only a particular synthetic release and selected quasi-identifiers. None implies differential privacy.

## How to interpret reported metrics

- **Retention / suppression loss:** fraction of generated records remaining after suppression; higher retention is not itself stronger privacy.
- **ROC-AUC:** ranking discrimination for the generated label; 0.5 is chance ranking. It does not indicate real-world model quality here.
- **PR-AUC:** precision-recall area; interpret alongside the synthetic positive rate.
- **Brier score:** mean squared error of predicted probabilities; lower is better calibration/accuracy for this generated target.
- **Fairness rates/gaps:** age-band selection rate, TPR, and FPR with subgroup counts. Small denominators are flagged. They describe only generated age bands and the artificial label.
- **Membership attack AUC:** attack classifier's ability to separate member from nonmember examples; 0.5 is chance for this attack. The bootstrap interval covers attack-test sampling only, not model/attack design uncertainty. Attack failure is not proof of privacy.
- **Pareto frontier:** finite-grid configurations that are not dominated when maximizing retention and minimizing worst categorical-TV distance. It is not a global optimum and its TV objective is only a diagnostic proxy.
- **DP epsilon fields:** custom accountant diagnostics are unverified. No formal epsilon guarantee is established for the model or pipeline.

## PowerShell setup and run commands

Run from the project root in the existing environment; these commands do not install packages. Choose a timestamped output folder so previous results are preserved.

```powershell
cd C:\Users\harsh\Downloads\CPAF_Advanced_Project_v2\CPAF_Advanced_Project
$py = 'C:\Users\harsh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$demoRoot = Join-Path 'cpaf_outputs' ("faculty_demo_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))

& $py --version
& $py -m streamlit --version
& $py -m unittest discover -s tests -q
& $py -m compileall -q cpaf cpaf_advanced.py cpaf_ml.py app.py tests scripts

& $py cpaf_advanced.py --n 300 --k 2 --l 2 --t 0.2 --no-ml --seed 6 --output-dir (Join-Path $demoRoot 'privacy')
& $py cpaf_advanced.py --repeated-ml --n 500 --repeats 2 --k 2 --seed 31 --output-dir (Join-Path $demoRoot 'ml')
& $py cpaf_advanced.py --attack-audit --n 1000 --k 2 --seed 19 --output-dir (Join-Path $demoRoot 'attack')
& $py cpaf_advanced.py --sweep --sweep-n 150 --sweep-k 1 5 --sweep-l 2 --sweep-t 0.2 --sweep-levels 1 --seed 7 --output-dir (Join-Path $demoRoot 'sweep')
& $py cpaf_advanced.py --plot-sweep-only --output-dir (Join-Path $demoRoot 'sweep')

& $py -m streamlit run app.py
```

The Streamlit command normally prints a local address such as `http://localhost:8501`; open it in a browser on the same computer. In the audit sandbox, loopback socket access was denied, so the app could not be verified through a live browser here. Do not install or access external services during this demonstration.

## Five-minute demonstration sequence

1. **0:00–0:40 — Scope:** show Overview and explain that provider evidence and synthetic experiments are separate.
2. **0:40–1:20 — Evidence register:** show the seven dimensions, source links, scored documentary findings, and four unscored gaps. Explain that the original template remains blank and untouched. If demonstrating edits, save to the separate reviewed working copy; replacing it requires confirmation and creates a timestamped backup.
3. **1:20–2:10 — Synthetic transformation:** generate a small dataset, generalize, apply k suppression, and display retention and the minimum released group size.
4. **2:10–2:50 — Diagnostics and utility:** show l/t as diagnostics, then explain the repeated synthetic-label utility comparison and small subgroup warnings.
5. **2:50–3:40 — Attacks:** explain the disjoint shadow-model partitions, chance baseline, and uncertainty interval; emphasize that one attack does not establish privacy.
6. **3:40–4:20 — Sweep:** show the saved configurations and Pareto frontier, noting that the result is grid-scoped and includes retention/TV proxies.
7. **4:20–5:00 — DP and limitations:** show `EXPERIMENTAL_UNVERIFIED`, name the unverified accountant and missing end-to-end proof, then summarize future work.

## Likely viva questions

1. **What is the project trying to answer?** It demonstrates how generalization and k suppression change retention and selected privacy diagnostics, and how these changes affect utility on a controlled synthetic label.
2. **Did you use Paytm, Google Pay, or PhonePe customer data?** No. The experiment data are generated by the project. Provider scoring requires a separate reviewed public-evidence record.
3. **Why keep evidence review separate from experiments?** Synthetic measurements cannot establish real provider practices; mixing them would make the evidence claim misleading.
4. **What does k=5 mean?** Each released group, defined by the selected quasi-identifiers, has at least five records after small whole groups are removed.
5. **Does k-anonymity guarantee privacy?** No. It depends on selected QIs and does not prevent all attribute inference or protect against auxiliary information.
6. **Is l-diversity enforced?** No. The current code reports distinct sensitive-value diagnostics; it does not transform data to guarantee an l threshold.
7. **What does the t metric measure?** Categorical total variation between a group’s sensitive-value distribution and the global distribution in the released table.
8. **Is this full t-closeness?** No. It is a categorical diagnostic, not a general numerical-distance implementation or enforcement procedure.
9. **Why can suppression retain very few rows?** Fine-grained quasi-identifiers create many small groups; every group below k is removed as a whole.
10. **What is the prediction label?** A generated binary label sampled from a probability function using age, amount, hour, merchant type, and random noise.
11. **Is the ML evaluation independent of the label rule?** Not fully. The models use overlapping predictors, so it is a partly circular synthetic rule-recovery exercise.
12. **How do you avoid preprocessing leakage?** In the repeated utility runner, preprocessing is inside the scikit-learn pipeline and fit only on each training split; the same test rows are used across representations within a repeat.
13. **Are repeated test splits independent?** No. Stratified shuffle splits may overlap. The reported standard deviation is descriptive, not an independent-sample confidence interval.
14. **What is membership-inference AUC?** It measures how well the trained attack classifier ranks member vs. nonmember examples for the evaluated target model and data split.
15. **Does a near-chance attack prove privacy?** No. It only indicates limited performance for this attack setup, seed, sample, and representation.
16. **What does the fairness analysis establish?** Only descriptive age-band rates for the synthetic label. Small group denominators are flagged; no real-population fairness conclusion follows.
17. **What is the Pareto plot showing?** Non-dominated configurations in the evaluated finite grid, maximizing retention and minimizing worst categorical-TV distance.
18. **Does the custom DP-SGD result have a valid epsilon guarantee?** No. It is explicitly `EXPERIMENTAL_UNVERIFIED`; the custom accountant, finite-precision mechanism, and full pipeline boundary have not been independently validated.
19. **Why use an experiment seed if DP noise is not seeded?** The seed makes synthetic data and ordinary experiment splits repeatable; DP coins intentionally use OS-backed randomness and are not bit-for-bit reproducible.
20. **What would be needed before publication?** A clearly scoped research question, stronger external validation or narrower synthetic claims, repeated independent evaluation, reviewed provider evidence, and independent review of any claimed DP mechanism and release boundary.

## Known limitations and future work

- No external transaction/customer dataset or independently audited provider evidence is included. The current AI-assisted documentary scores still require human source and rationale review.
- The synthetic label and predictors overlap, so ML results are not real-world utility estimates.
- Fairness results are synthetic and may have small subgroup denominators.
- MIA covers one attack family and one seed; intervals omit training and attack-design uncertainty.
- l-diversity/t-closeness are diagnostics only; t is categorical TV. Microaggregation is independent per column.
- The custom DP accountant and implementation remain `EXPERIMENTAL_UNVERIFIED`; no formal guarantee is established. diffprivlib migration is not complete.
- Dashboard page interactions could not be verified in the current sandbox because local socket access was denied. Verify the local address manually on the demonstration computer.
- Raw outputs, summaries and plots are colocated in run folders; process failures before final manifest creation are not always recorded durably.

Future work: independent evaluation splits/multiple seeds, richer and externally validated synthetic generators, reviewer agreement and source-version tracking for provider evidence, more threat-model-specific attacks, clearer raw/derived artifact partitioning, and independently audited DP-library integration.
