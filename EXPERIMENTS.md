# Experiments and interpretation

The checked-in configs/default_experiment.json supplies reproducible defaults. CLI flags override JSON values; run manifests record the config path and effective settings.

Run the Windows commands in the README from this directory. The default CLI experiment is `python cpaf_advanced.py --n 50000 --k 5 --no-ml`; `--ml-only` selects the inherited ML suite; omit `--no-ml` to run both. `--l` and `--t` set diagnostic thresholds. Use `--output-dir` to keep runs separate.

Always choose a new, empty output directory. The CLI refuses to write a new experiment into a directory containing results. Dashboard experiments use timestamped run directories. Output manifests record synthetic provenance, configuration, seeds, versions, timestamp, and artifact paths. The fixed experiment seed is used to derive the synthetic-label seed as `seed + 1`; DP mechanism randomness is intentionally unseeded and remains experimental.

Run a parameter grid with `--sweep --sweep-n 600 --sweep-k 1 5 20 --sweep-l 2 3 --sweep-t 0.1 0.2 --sweep-levels 1 2 --seed 123`. The l/t settings are diagnostic thresholds only. `--plot-sweep-only --output-dir <same-run-folder>` rebuilds the frontier files and chart from the saved privacy_sweep.csv.

Use `--repeated-ml --n 5000 --repeats 3 --k 5 --seed 42` for repeated stratified logistic utility plus synthetic age-band gaps. Use `--attack-audit --n 3000 --k 2 --seed 19` for the shadow-model experiment and synthetic disclosure proxies. `--ml-only` runs inherited model comparison and experimental DP-SGD outputs.

`streamlit run app.py` launches the dashboard after installing requirements-dashboard.txt. `python -m unittest discover -s tests -v` runs automated checks.

Outputs are CSV/JSON/PNG under cpaf_outputs or the supplied output directory. Compare retention and group diagnostics together: suppression can achieve the selected k threshold by dropping complete small groups. l/t results are observations of the selected released data. Pareto analysis maximizes retained rows and minimizes worst categorical-TV distance within the supplied finite grid. Model utility and fairness refer to an artificial target. A near-chance attack AUC is not proof of privacy. Epsilon is experimental/unverified. Do not combine these unlike metrics into one arbitrary score.

For a compact reproducibility checklist and current audited run commands, see `docs/RESEARCH_READINESS.md`.
