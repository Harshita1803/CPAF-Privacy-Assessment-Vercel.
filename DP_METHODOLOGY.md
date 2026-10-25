# Differential privacy methodology

## Current implementation status

The project implements a teaching DP-SGD-style logistic model, not a validated end-to-end DP training pipeline. Every result row is marked `EXPERIMENTAL_UNVERIFIED`; no `VERIFIED_BOUND` is reported. A reported numeric `epsilon_experimental_unverified` is an unsupported diagnostic estimate, not a privacy guarantee.

An established-library replacement was evaluated but not integrated: PyTorch/Opacus is absent and Opacus secure mode depends on a CSPRNG package whose published Windows support does not match this Python 3.12 runtime; diffprivlib is more compatible on paper but could not be downloaded because package-index DNS was blocked. Do not infer a library guarantee from that compatibility review. See `docs/DP_LIBRARY_MIGRATION_REPORT.md`.

## Intended mechanism

For each step t, current weights w_t are fixed by prior outputs. Per-record logistic gradients are

`g_i(w_t) = (sigmoid(x_i · w_t) - y_i) x_i`.

Each gradient is L2-clipped: `gbar_i = g_i * min(1, C / ||g_i||_2)`. Independent Bernoulli inclusion at q forms `S_t = sum_{i sampled} gbar_i`; Gaussian noise with standard deviation `sigma*C` is added to S_t. The update divides by `q*n` and applies a fixed learning rate schedule. There are T fixed steps, zero public initialization, no data-dependent stopping. The intended protected unit is one row (features plus label) in the arrays passed to fit, under add/remove adjacency. It is not automatically the raw CPAF dataset because upstream stratified partitioning and selection are not covered. Delta is fixed public 1e-5 per candidate, not derived from the member sample size; the denominator n still depends on the fit-array length.

If the sampling theorem's assumptions hold, an integer-order sampled-Gaussian RDP expression is composed over T steps, then converted to approximate DP for a chosen delta. For q=1, one step has Gaussian RDP `alpha/(2*sigma^2)` when sensitivity is C and standard deviation is sigma*C. Repeated adaptive steps compose additively in RDP. The custom implementation has not been independently validated and finite-precision Gaussian sampling has not been proved; hence the code does not claim that bound as verified.

## Randomness and reproducibility

DP batch selection and noise use OS-backed `secrets.SystemRandom`, not the public experiment seed. Supplying a seed to `DPLogReg` is rejected. Consequently, the DP model is intentionally not bit-for-bit reproducible. Synthetic data, splits outside the mechanism, and other experiments remain seeded. The manifest's experiment seed must not be mistaken for the DP noise seed. The finite-precision Gaussian sampler has not been proved to implement the ideal continuous mechanism.

## Scope and exclusions

The result protects neither upstream label generation/stratification nor data-dependent parameter selection, utility/fairness/MIA metrics, exported raw or transformed datasets, evidence scoring, or multiple released models. If several independently trained private models were released and each had a valid guarantee, their privacy losses would compose; the current custom pipeline does not account for this release set. Metrics should be treated as experimental outputs without a DP guarantee.

## Concept distinctions

Differential privacy is a mathematical bound on how output distributions change for neighboring datasets. Adding random noise is only a mechanism ingredient; sensitivity, distribution, randomness, composition, and all data-dependent steps matter. Membership-inference attacks are empirical probes and attack failure does not prove DP. k-anonymity, l-diversity, and t-closeness concern properties of particular released tables and do not imply DP. Utility on synthetic data measures consistency with an artificial generator/label rule, not empirical privacy of real payment-app records.

The current project status remains `EXPERIMENTAL_UNVERIFIED`. The offline research-readiness rerun did not execute or integrate diffprivlib and did not change this status. No epsilon value in the project is a formal bound; see `docs/DP_LIBRARY_MIGRATION_REPORT.md` and `docs/RESEARCH_READINESS.md`.
