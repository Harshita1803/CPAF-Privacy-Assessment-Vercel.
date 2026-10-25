# Differential privacy implementation audit

Audit date: 2026-10-24. Scope: trace `cpaf_ml.run_ml_lab` through feature preparation, partitioning, `DPLogReg.fit`, epsilon reporting, MIA/utility exports, and the dashboard. The existing formal-guarantee claim was false/disabled; this audit does not reinstate one.

## Execution path and data boundary

1. `make_synthetic_transactions`/the CPAF generator produces synthetic records and a synthetic label. For privacy analysis, a row and all its features/label are the protected unit; the project does not use payment-app records.
2. `run_ml_lab` derives the prediction target from artificial transaction attributes and noise. This is private-data-dependent if those rows are treated as a private dataset.
3. `train_test_split(..., stratify=y)` partitions rows using labels. Member selection and nonmember sampling use a seeded NumPy generator. These are outside the DP mechanism; stratification is label-dependent. The selected member subset is passed into the DP model.
4. `fixed_encode` maps age/hour/amount and fixed category vocabularies into numeric features. Bounds/category list are source constants and therefore public, but category handling and row selection must be treated as caller operations; the guarantee, if any, would be relative to arrays passed to `fit`, not automatically to original raw data.
5. `DPLogReg.fit` initializes weights to zeros, then performs a fixed number of steps. For each selected record, logistic per-example gradient is `(sigmoid(x·w)-y)x`; its row L2 norm is clipped to C. Independent Bernoulli inclusion is intended at rate q. Gaussian noise with standard deviation sigma*C is added to the clipped gradient sum. The result is divided by q*n, where n is the supplied array length, then used in a deterministic update. The normalization is postprocessing of the noisy sum only when n, q, and all update parameters are public/fixed.
6. The model is evaluated on selected training members, selected nonmembers, and the held-out test split. These reported ROC-AUC, PR-AUC, and membership attack metrics depend on the data/model and are not themselves DP releases. The CLI compares several noise multipliers and publishes all rows; selection by the reported metrics is not accounted for.
7. The custom integer-order RDP code composes T step values and converts each order using `RDP(alpha)+log(1/delta)/(alpha-1)`, then minimizes over orders 2–128. Each candidate now uses a fixed configured delta=1e-5, independent of the selected member count. The denominator n remains the fit-array length, whose publicness and behavior under add/remove neighbors are not established.

## Operation classification

| Operation | Data dependence | Covered by sampled Gaussian mechanism? | Audit finding |
|---|---|---|---|
| Fixed encoding bounds and category vocabularies | Public constants | Not a private query | No learned statistics in `fixed_encode` |
| Synthetic target generation | Per-row attributes and random draw | Not before `fit` | Synthetic only; part of caller input construction |
| Stratified train/test split | Labels and row arrangement | No | Upstream partition map is not analyzed |
| Member/nonmember row selection | Seeded random indices | No | Seed is in experiment metadata; not DP selection |
| Zero initialization, fixed steps/lr | Public constants | Postprocessing/control | No data-driven early stopping in this DP class |
| Per-example gradient calculation and L2 clipping | Private rows and current model | Yes, conditionally | Clipped per row, not per batch |
| Bernoulli inclusion and Gaussian noise | Randomized mechanism | Intended | Previously driven by `np.random.default_rng(seed)`; that public seed made the run deterministic/known and invalidated a formal DP claim. Changed to `secrets.SystemRandom`; still not an end-to-end proof. |
| Divide by q*n and optimizer update | n/q and noisy sum | Yes only if n/q/public schedule fixed | n is supplied-array length and publicness is assumed |
| Custom RDP accountant | Public parameters | Accounting only | Formula is a custom integer-order implementation; not independently validated against a maintained accountant |
| Train/test metrics, MIA, sigma comparison | Private-derived outputs | No | Outside claimed guarantee; do not publish as DP-protected statistics |

## Adjacency and guarantee boundary

The accountant is intended for **add/remove-one record adjacency** for each supplied fit array: D and D' differ by one example. This is not replace-one adjacency. A clipped per-example gradient has norm at most C, so the clipped sum has add/remove sensitivity at most C; replacement may change one bounded vector to another and can have sensitivity 2C. With independent Poisson/Bernoulli inclusion, the accountant must match that sampling model and adjacency.

No end-to-end claim is justified for the raw-input-to-CSV CLI pipeline. The private, label-dependent upstream partition/selection, potentially data-dependent n/delta, metric releases, multi-sigma comparison, and custom accountant validation sit outside an established proof. In addition, the code uses floating-point Gaussian sampling; it has no finite-precision implementation proof. The mechanism is therefore labeled `EXPERIMENTAL_UNVERIFIED`, not `VERIFIED_BOUND`.

## Specific correction

The DP model previously accepted a reproducibility seed and used NumPy's seeded generator for both Bernoulli selection and Gaussian noise; the same public seed reproduced all DP randomness. This contradicts the intended randomized mechanism. The implementation now uses OS-backed `secrets.SystemRandom`, rejects a caller-provided DP seed, and records that source. Other experiment randomness remains seeded. The privacy delta is fixed public 1e-5 rather than 1/n. This improves randomization and parameter handling but does not validate the custom accountant or bring upstream operations and result metrics under DP.

The current calculator may return `epsilon_experimental_unverified` for diagnostic continuity. Consumers must not treat it as a valid epsilon guarantee. `formal_guarantee_claimed` remains false and each row carries `verification_status=EXPERIMENTAL_UNVERIFIED`, assumptions, scope, and unaccounted operations.

## Accountant review and references

For integer alpha, the code's binomial log-sum is the standard Poisson-sampled Gaussian RDP expression under the stated sampling and adjacency assumptions; the q=1 case reduces to alpha/(2 sigma^2). Composition adds RDP over steps/runs; the conversion is a valid conservative form under the RDP-to-(epsilon,delta) relation. This mathematical correspondence is not an independent implementation audit and does not establish that all runtime assumptions hold. Tests check q=1 analytically, alpha=2 against its closed form, domain validation, monotonicity, repeated-step/run composition, clipping, mock noise, and status propagation; they are not a formal proof of implementation correctness.

- Yuqing Zhu and Yu-Xiang Wang, [Poisson Subsampled Rényi Differential Privacy](https://proceedings.mlr.press/v97/zhu19c.html), ICML 2019.
- Opacus documents that cryptographically strong randomness is needed for a strong DP guarantee and provides `secure_mode` for its supported implementation: [Opacus FAQ](https://opacus.ai/docs/faq), [PrivacyEngine API](https://opacus.ai/api/_modules/opacus/privacy_engine.html). Opacus, PyTorch, Google dp-accounting, TensorFlow Privacy, and diffprivlib were not installed in the supported runtime; no unverified network installation was attempted.

## Status

`EXPERIMENTAL_UNVERIFIED`. The measured epsilon is diagnostic only. To upgrade this status would require a reviewed maintained accountant/library, a public-size contract or fixed public denominator, audited end-to-end partitioning and preprocessing, finite-precision mechanism review, and a release boundary excluding or privatizing every reported metric and model-selection output. The runtime initially lacked Opacus, Google dp-accounting, TensorFlow Privacy, PyTorch, and diffprivlib. A later migration investigation attempted to download diffprivlib 0.6.6 but package-index DNS failed; see `docs/DP_LIBRARY_MIGRATION_REPORT.md`.

## Research-readiness recheck

The offline recheck found no change to the DP mechanism or accountant and made no DP claim upgrade. diffprivlib remains absent; PyPI access was denied, so installation was stopped. The small repeated-ML and shadow-attack experiments are synthetic evaluation outputs and do not enter a DP guarantee. See `docs/RESEARCH_READINESS.md` for commands and current artifacts.
