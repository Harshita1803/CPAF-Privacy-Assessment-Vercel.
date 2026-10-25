# Methodology

## Synthetic reference

NumPy's seeded generator samples age uniformly from integer ages 18–79; region from fixed probabilities North .22, South .24, East .17, West .21, Central .16; city uniformly from 30 artificial labels; merchant category from Grocery .25, Utilities .15, Food .20, Travel .10, Retail .22, Education .08; amount from a lognormal distribution (mean parameter 6.1, sigma .85) clipped to INR 20–150,000; hour uniformly from 0–23. `risk_category` is synthetic: High above INR 25,000 or for Travel before 05:00, Medium above INR 8,000, else Low. It is not a real risk label. Validation checks schema, nonempty data, missing cells, duplicate IDs and ranges.

## Privacy transformations

Generalization uses fixed age, amount, location and hour hierarchies. Level 0 retains the original value as a string, level 1 applies decade age bands, six amount bands, region, and four time bands; level 2 coarsens age/amount/time and replaces location with India. The output omits the synthetic row ID, city, exact amount, age and hour. K suppression retains an entire equivalence class iff its size is at least k. K compliance is verified on the released table and only for the declared QIs.

Distinct l-diversity for a group means it has at least l distinct sensitive values. The CSV reports minimum group distinct count and fraction passing the default l=2 threshold. Categorical t-closeness diagnostic is total variation, `0.5 * sum_v |p_group(v)-p_global(v)|`, where the global distribution is computed over the released data. The reported worst distance and passing-group fraction are diagnostics; they are not numerical t-closeness and say nothing about other QIs.

Microaggregation independently stable-sorts each selected numeric column, partitions consecutive values into blocks of requested size, and replaces each block by its mean; the final incomplete block is retained. Aggregated columns use floating-point dtype so fractional means (for example, a mean age of 18.8) are retained exactly rather than rounded or truncated. The input frame is not mutated. This simple univariate baseline does not preserve multivariate clusters. RMSE is computed against each original column.

## Repeated utility, distributions and fairness

The inherited three-model run uses one stratified train/test split. A separate repeated runner compares logistic regression over repeated stratified 75/25 splits using identical held-out indices for raw, generalized and k-suppressed training. It reports mean and standard deviation for ROC-AUC, PR-AUC, Brier and F1, plus per-run precision, recall and confusion matrices. Preprocessing is fit inside each training split.

Age-band selection rate, TPR and FPR are descriptive subgroup metrics for the artificial target. Suppression distribution drift reports categorical TV and range-normalized one-dimensional Wasserstein distance between raw training rows and the retained subset. These can reflect the synthetic label-generation rule and do not support real-world fairness claims.

## Utility and scoring

The inherited ML target is independently sampled from an artificial logistic probability using age, amount, hour, merchant and random noise. The models exclude `risk_category`. The inherited test split is stratified once; output metrics ROC-AUC, PR-AUC and Brier are narrow and do not constitute repeated-seed uncertainty estimates. App evidence scoring is `adjusted = 3 + (raw - 3)*strength`; dimension CPI is a weighted mean of available dimensions. Coverage is observed dimensions / 7; confidence is coverage times mean strength. Missing dimensions remain missing and weights are renormalized over available dimensions. Results are provisional pending human evidence review.

## Parameter optimization

The finite sweep evaluates hierarchy levels and k values. l and t values change diagnostic thresholds only; they do not alter the released table. The sweep reports suppression retention, group counts, l/t coverage, frequency drift and runtime. Pareto points maximize retention and minimize worst group categorical TV. This is scoped to the searched grid and synthetic generator; it is not a global optimum or formal privacy/utility trade-off.

## DP experiment status

Per-example L2 clipping at C bounds each clipped gradient contribution. Add/remove sensitivity of the clipped sum is at most C; replace-one sensitivity can be 2C. Each step uses independent Bernoulli inclusion with rate q and Gaussian noise of standard deviation sigma*C on the clipped sum, for T steps. The update divides by q*n where n comes from the supplied training array. The experimental accountant uses integer Renyi orders and a delta conversion; the reported DP-SGD run sets delta=1/n.

No formal claim is made: the custom accountant and finite-precision Gaussian mechanism have not been independently validated; data-dependent row selection precedes the DP example; normalization and adjacency behavior under variable-size neighboring datasets have not been proven; and utility/attack metrics are private-derived outputs outside the mechanism. DP randomness now comes from OS-backed `SystemRandom`, and delta is fixed public 1e-5 per candidate. CSV uses `EXPERIMENTAL_UNVERIFIED` and `formal_guarantee_claimed=false`. See `docs/DP_AUDIT.md`.

DP-SGD uses Poisson inclusion, per-example gradient clipping, and Gaussian noise on the clipped summed gradient. A custom integer-order RDP accountant is retained for teaching, with parameter guards, but has not been independently checked against a maintained DP accountant. Epsilon is explicitly marked unverified and no formal guarantee is claimed. Sampling and adjacency assumptions, delta selection and preprocessing require independent review before any formal claim.
