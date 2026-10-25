# Differential privacy library migration report

Audit date: 2026-10-24. Decision: **migration not completed; status remains `EXPERIMENTAL_UNVERIFIED`**. No third-party package or previous results were overwritten.

## Summary

The current environment has Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0 and setuptools 84.1.0. It has no PyTorch, Opacus, Google `dp-accounting`, TensorFlow Privacy, or diffprivlib installed. This turn's one network probe, `Resolve-DnsName pypi.org`, returned `Access is denied`; I did not retry or attempt a package download. A previous turn's actual pip download attempt failed with `NameResolutionError`. No installed environment or dependency files were changed.

I evaluated Opacus and IBM diffprivlib against the current environment and architecture:

| Library | Compatibility evidence | Blocking issue / decision |
|---|---|---|
| Opacus 1.6.0 | Published Python wheel and PyPI metadata indicate Python support; current release notes require `torch>=2.6`. | PyTorch is absent. Opacus `secure_mode=True` requires `torchcsprng`; its published Windows wheels are for older CPython (3.7–3.9), and the project runs Python 3.12. Thus I cannot establish a supported secure Opacus stack in this environment, and it would require replacing the NumPy logistic implementation with a PyTorch model/data-loader path. Not selected. |
| IBM diffprivlib 0.6.6 | Its Python 3 wheel supports Python 3.12. Its current package requirements (NumPy>=2, scikit-learn>=1.4, SciPy>=1.13) are met by the installed versions. Its documented LogisticRegression uses the Vector mechanism to perturb the objective and exposes `epsilon`, `data_norm`, and an accountant. | It is a different objective-perturbation method, not DP-SGD. The guarantee requires a data-independent public norm bound and secure randomness. The model API documents a `random_state`; passing a secure `SystemRandom` appears possible from the utility API, but this exact code path, accounting spend, package compatibility, and end-to-end use were not installed or executed. Therefore no output from it can be reported and no status upgrade is justified. |

I added an isolated optional requirements file, a smoke script, and setup instructions. These artifacts are a handoff for a network-enabled Python 3.12 environment, not a completed integration. The script intentionally labels itself `EXPERIMENTAL_UNVERIFIED` and does not report a project-verified bound.

The diffprivlib documentation warns that inferring `data_norm` from the fit data leaks information. It also documents secure RNG creation through `check_random_state(seed=None, secure=True)`. Its LogisticRegression implementation uses a Vector mechanism and records `(epsilon, 0)` in a `BudgetAccountant`. Those are promising library behaviors, not verification of this project integration. The library docs should be checked again at implementation time because package versions and interfaces can change.

## Why the previous custom path remains insufficient

The prior custom DP-SGD path used a home-grown sampled-Gaussian RDP accountant and a public experiment seed for mechanism randomness. The latest remediation moved batch sampling/noise to OS-backed `secrets.SystemRandom`, rejects a DP seed, fixes delta to public `1e-5` per candidate, adds a bounded MIA sample, and annotates results with `EXPERIMENTAL_UNVERIFIED`. These corrections remove the deterministic-noise flaw and improve reporting; they do not validate the accountant or prove the float implementation.

The complete CLI also performs private-label stratification, chooses member/nonmember subsets, evaluates candidates using data-dependent metrics, and emits several sigma results. The size-dependent training denominator and reported outputs are outside a proven guarantee boundary. A model-level claim for only the fit input would not certify the raw-input-to-CSV experiment.

## Intended narrow claim if a library migration is completed

No claim is made now. A future library demo should isolate a single binary logistic model trained on one fixed synthetic dataset, where one row (features plus label) is the protected unit. It must state the library's adjacency definition and theorem precisely, provide a public feature norm bound before fitting, use library-supported secure randomness, and release only that single fitted model. Synthetic generation/configuration may be treated as public only when the demonstration explicitly declares the generated dataset public; otherwise the generated records remain the protected data. Evaluation metrics, attack results, raw/transformed datasets, and model selection must be kept outside the release boundary or separately privatized and composed.

For such a demo, `epsilon` and `delta` must come from the library's actual accountant/output for the exact fit and declared mechanism. If the library guarantee is pure epsilon-DP, report delta=0; do not reuse the existing custom DP-SGD epsilon. Repeated fits/releases must share and spend a composition budget or be explicitly treated as separate unaccounted runs. No exact adjacency or numerical privacy bound is asserted in this report because no library fit was performed.

## Commands and actual outcomes

Environment inspection:

```powershell
$py = 'C:\Users\harsh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py --version
& $py -m pip show opacus torch dp-accounting tensorflow-privacy diffprivlib
```

Observed: Python 3.12.14; all five packages were reported not found.

Dependency availability attempt:

```powershell
& $py -m pip download --no-deps diffprivlib==0.6.6 --dest (Join-Path $env:TEMP 'cpaf_dp_library_wheels')
```

Observed: retries failed with `NameResolutionError` for `pypi.org`; pip ended with “No matching distribution found” because the index could not be reached. This does not establish that the package itself is unavailable or incompatible.

The existing full custom DP experiment remains available under `cpaf_outputs/dp_audit_verified/`; it was not overwritten by this migration probe. No library-based training run or library accountant comparison was performed. The new `scripts/dp_diffprivlib_smoke.py` has not been run because diffprivlib is absent. It defaults to creating outputs under `cpaf_outputs/dp_library_poc/` on the user's network-enabled environment. A prior-turn direct custom-model smoke fit is not evidence for the library and remains unverified.

After the dashboard notice and documentation updates, the current final validation was rerun with the bundled Python 3.12.14 interpreter:

```powershell
& $py -m unittest discover -s tests -q
& $py -m compileall -q cpaf cpaf_advanced.py cpaf_ml.py app.py tests scripts
```

Observed output: `Ran 25 tests ... OK`; compileall exited 0 (`COMPILEALL_EXIT=0`). The dashboard page itself was not interactively launched in this restricted socket environment. The library smoke script was compile-checked only, not executed, because diffprivlib is absent.

## Exact scope and status

- Candidate prepared for external smoke testing: diffprivlib==0.6.6 with a pinned Python 3.12 requirements file; installation and estimator execution remain unverified.
- Formal guarantee: none for the CPAF pipeline, and none newly established for the DP-SGD mechanism.
- Current status: `EXPERIMENTAL_UNVERIFIED` for existing custom results; `NOT_APPLICABLE` where no DP result exists.
- No `VERIFIED_BOUND` is emitted.
- Remaining work: install/test the pinned candidate in a Python 3.12 environment with working package access; audit its public-bound and secure-randomness path; capture actual accountant output and package versions; test repeated-release accounting; keep non-DP metrics outside the claim; and obtain independent review of the exact theorem-to-code mapping. Until then, the POC is not complete and must not be marked `VERIFIED_BOUND`.

## Environment handoff artifacts

- Setup instructions and exact PowerShell commands: `docs/DP_ENVIRONMENT_SETUP.md`.
- Candidate-only pins: `requirements-dp-library.txt`.
- Isolated single-fit smoke script: `scripts/dp_diffprivlib_smoke.py`.
- These are preparation artifacts only. No library output directory/result was generated on this host, and the existing custom DP results remain unchanged and `EXPERIMENTAL_UNVERIFIED`.

## Follow-up verification attempt (2026-10-24 18:46 +05:30)

Commands and observed outcomes from the latest request:

```powershell
git status --short
# fatal: not a git repository (or any of the parent directories): .git

py -3.12 --version
# WindowsApps py.exe could not start in this session (ResourceUnavailable).

Resolve-DnsName pypi.org -DnsOnly -QuickTimeout
# pypi.org : Access is denied.

$py = 'C:\Users\harsh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py --version
# Python 3.12.14
& $py -m pip show diffprivlib
# WARNING: Package(s) not found: diffprivlib
Test-Path .venv-dp-library\Scripts\python.exe
# False; the isolated environment does not exist.
```

The network probe failed, so installation was stopped without retrying. The normal bundled interpreter was used only for read-only inspection and regression checks; no dependency was installed there. Regression commands and results:

```powershell
& $py -m unittest discover -s tests -q
# Ran 25 tests in 7.825s; OK; exit 0
& $py -m compileall -q cpaf cpaf_advanced.py cpaf_ml.py app.py tests scripts
# exit 0
```

The POC was not executed. No library version, model output, accountant spend, freeze file, or verified epsilon was produced. Migration remains blocked before installation; smoke test, independent guarantee review, and full integration are all incomplete. No Git status/diff could be obtained because this project root is not a Git repository.

## Primary references checked

- [Opacus on PyPI](https://pypi.org/project/opacus/) and [Opacus PrivacyEngine API](https://opacus.ai/api/_modules/opacus/privacy_engine.html): release/version and `secure_mode` requirements.
- [Opacus FAQ on secure RNG](https://opacus.ai/docs/faq) and [torchcsprng supported platforms/versions](https://github.com/meta-pytorch/csprng): secure-mode dependency and compatibility.
- [diffprivlib on PyPI](https://pypi.org/project/diffprivlib/) and [LogisticRegression documentation](https://diffprivlib.readthedocs.io/en/latest/modules/models.html): version metadata, model mechanism, explicit `data_norm`, epsilon, and accountant interface.
- [diffprivlib secure RNG utility](https://diffprivlib.readthedocs.io/en/latest/modules/utils.html): `check_random_state(..., secure=True)` behavior.
