# Optional diffprivlib proof-of-concept setup

This guide is for a separate environment; it does not alter CPAF's `requirements.txt`, the current Python installation, or existing outputs. The tested-at-runtime status is **not complete**: this host could not resolve PyPI. The pins below are a proposed compatible set based on the current environment versions and diffprivlib's published requirements, but have not been installed together here.

## Compatibility basis

- Recommended interpreter: CPython 3.12 x64 on Windows, matching the project's supported bundled interpreter. diffprivlib 0.6.6 declares Python >=3.8 and publishes a platform-independent wheel.
- The current runtime has NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, and setuptools 84.1.0. diffprivlib 0.6.6 declares minimums NumPy>=2.0, SciPy>=1.13, scikit-learn>=1.4, joblib>=0.16, setuptools>=49.
- The proposed pins are in `requirements-dp-library.txt`. They match this runtime's available versions and satisfy the declared minima. Because diffprivlib was not installed, dependency resolution and runtime compatibility remain to be verified on the target machine.
- diffprivlib documents its logistic model as regularized logistic regression using objective perturbation and the Vector mechanism, with `epsilon`, `data_norm`, and a `BudgetAccountant`. The data norm must be explicitly chosen independently of private records; omitting it causes the implementation to infer it from fit data and raise a privacy warning. The smoke script sets `data_norm` explicitly.
- The script requests `check_random_state(None, secure=True)` and passes the returned `SystemRandom` instance into the estimator. This is based on diffprivlib's documented secure RNG helper, but propagation through this exact estimator must be checked when the script is run.

## Create and install into an isolated virtual environment

Run in PowerShell from the CPAF project root on a machine with working package access:

```powershell
cd C:\Users\harsh\Downloads\CPAF_Advanced_Project_v2\CPAF_Advanced_Project
py -3.12 -m venv .venv-dp-library
.\.venv-dp-library\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dp-library.txt
New-Item -ItemType Directory -Force -Path cpaf_outputs\dp_library_poc | Out-Null
python -m pip freeze --all | Set-Content cpaf_outputs\dp_library_poc\environment-freeze.txt
python scripts\dp_diffprivlib_smoke.py --output-dir cpaf_outputs\dp_library_poc
```

If script execution is blocked by PowerShell policy, use the venv interpreter directly without activation:

```powershell
.\.venv-dp-library\Scripts\python.exe -m pip install -r requirements-dp-library.txt
.\.venv-dp-library\Scripts\python.exe scripts\dp_diffprivlib_smoke.py --output-dir cpaf_outputs\dp_library_poc
```

The smoke script creates its output directory and writes `diffprivlib_smoke_result.json`; it does not overwrite the existing custom-DP audit directory. Its fixed public synthetic generator is seeded for dataset reproducibility. The model's private randomization uses the library's secure RNG path and is intentionally not seeded. The JSON includes the installed versions, configured epsilon, actual accountant spend, public norm, adjacency placeholder, scope, and explicit `EXPERIMENTAL_UNVERIFIED` status.

## What the smoke script does and does not verify

It fits one binary diffprivlib logistic-regression estimator on 256 generated rows with fixed public feature bounds, explicit `data_norm`, no data-derived preprocessing, no held-out metrics, no tuning, and no second model. It writes only the fitted coefficients/intercept and a manifest. It reports the library accountant's spend for that estimator. It omits the optimizer's observed iteration count because it may depend on the data and is not part of the declared released output.

It does **not** itself establish an accepted project guarantee. Before using `VERIFIED_BOUND`, a reviewer must confirm the installed version's theorem and exact adjacency relation, verify secure RNG propagation into the Vector mechanism, inspect the library's norm and label assumptions, confirm accountant spend matches the single estimator call, and inspect the exact saved output. The current script deliberately emits `EXPERIMENTAL_UNVERIFIED` and `formal_guarantee_claimed=false`; it does not convert a library's documented model guarantee into an independent CPAF verification.

## Network-blocked case

The current audit environment returned access denied when resolving `pypi.org`. Do not retry repeatedly. Run the steps above on a compatible machine with package access, then preserve `environment-freeze.txt`, the smoke JSON, console output, and test results. Until that run and review occur, the library integration remains not implemented and no `VERIFIED_BOUND` is claimed.

## Algorithm distinction

This diffprivlib estimator is not the existing custom DP-SGD. The custom path computes per-example logistic gradients, clips each at C, Poisson-subsamples, adds Gaussian noise to a summed gradient, applies T optimizer updates, and estimates epsilon with a custom sampled-Gaussian RDP accountant. diffprivlib LogisticRegression instead perturbs the regularized empirical objective with its Vector mechanism and optimizes that perturbed objective; its documented estimator spend is epsilon-DP, delta=0. It does not use the custom path's q, sigma, clipping norm, number of DP-SGD steps, or RDP accounting. Do not compare the two epsilons as if they measured the same mechanism, or carry over any custom pipeline metric/claim to the library model.

## Official documentation checked

- [diffprivlib 0.6.6 on PyPI](https://pypi.org/project/diffprivlib/)
- [diffprivlib LogisticRegression API](https://diffprivlib.readthedocs.io/en/latest/modules/models.html)
- [diffprivlib secure RNG helper](https://diffprivlib.readthedocs.io/en/latest/modules/utils.html)
- [diffprivlib BudgetAccountant](https://diffprivlib.readthedocs.io/en/latest/_modules/diffprivlib/accountant.html)
