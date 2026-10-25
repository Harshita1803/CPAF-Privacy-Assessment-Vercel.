# Architecture

`cpaf/core.py` is the reusable layer: artificial data generation and validation, fixed generalization hierarchies, whole-equivalence-class k suppression, distinct-l and categorical-TV t diagnostics, and numeric microaggregation. `cpaf/experiments.py` runs configured sweeps, exports manifests/tables and calculates/regenerates the finite empirical Pareto set. `cpaf/evaluation.py` performs repeated stratified logistic utility and synthetic age-band fairness summaries. `cpaf/attacks.py` runs the disjoint shadow-model attack and synthetic disclosure proxies.

`cpaf_ml.py` runs the inherited three-family predictive comparison, k sweep, legacy confidence MIA and custom DP-SGD example. Its epsilon output is explicitly unverified. The ten-section `app.py` dashboard calls shared modules and separates evidence review from synthetic results. `cpaf_outputs/` stores run-scoped CSV/JSON/PNG artifacts and manifests. Evidence templates are separate reviewer input. Reports/presentations are not experiment outputs and remain untouched.

Data flow: synthetic generator → schema validation → generalization → k suppression → diagnostics and ML experiment → CSV/JSON/PNG. The app evidence flow is reviewer-entered public source evidence → evidence-adjusted dimension scores → weighted CPI and coverage/confidence. No data or score flows between those tracks.
