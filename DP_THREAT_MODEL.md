# Differential privacy threat model

## Protected unit and adjacency

The intended record-level unit is one row of model features and its label. The local accountant's intended relation is add/remove-one adjacency for the exact arrays passed to `DPLogReg.fit`. Replace-one adjacency is not equivalent: for a clipped sum the sensitivity can rise from C to 2C. The model fit currently assumes the fit-array size n, sample rate q, clipping norm, number of steps, and delta choice are public/fixed.

## Adversary and release

An adversary may see saved CSV/JSON/plots and any released model or reported metric. The current CLI emits metrics derived from training/test records and compares multiple noise settings. These outputs are outside the experimental DP mechanism and are not covered by a guarantee. A downstream user's model selection based on these scores would also be unaccounted.

## Boundary

Public constants include feature bounds, category vocabulary, initialization, optimization schedule, and declared hyperparameters. The generated label, stratified train/test split, member selection and reported metrics depend on rows/labels and are outside a proved mechanism boundary. The DP class clips each per-example gradient, uses Bernoulli sampling and adds Gaussian noise to the clipped sum, then normalizes and updates. However, the custom accountant has not been independently verified; data-size assumptions and finite-precision randomness also remain unaudited. The current status therefore remains `EXPERIMENTAL_UNVERIFIED`.

## Interpretations not supported

No result establishes privacy for actual Paytm, Google Pay, PhonePe users, or real financial records. No MIA result, k-anonymity result, l/t diagnostic, or addition of noise alone establishes a formal DP guarantee. The current status is `EXPERIMENTAL_UNVERIFIED`; no `VERIFIED_BOUND` is emitted. A third-party library was evaluated but could not be installed and validated in this runtime; see `docs/DP_LIBRARY_MIGRATION_REPORT.md` and `docs/DP_AUDIT.md`.

The 2026-10-24 offline research-readiness audit did not alter this boundary. No dependency installation was attempted; a bounded PyPI DNS check was denied and the library POC remains unrun. Synthetic attack/utility reruns do not validate the DP mechanism or provide evidence about provider customers. See `docs/RESEARCH_READINESS.md`.
