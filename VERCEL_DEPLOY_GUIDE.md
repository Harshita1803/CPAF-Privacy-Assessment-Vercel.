# CPAF 2.0 — Vercel Edition

This is a separate Vercel-compatible interface. It does not replace the original Streamlit dashboard.

## Deploy from GitHub without VS Code

1. Keep `Harshita1803/CPAF-Privacy-Assessment` and its Streamlit deployment unchanged.
2. On GitHub, create a **new repository** named `CPAF-Privacy-Assessment-Vercel`.
3. Open the new repository, choose **Add file → Upload files**, and upload the contents of this ZIP (upload the files and folders inside `CPAF-Vercel-Version`, not the outer folder itself).
4. Commit the files to the new repository's `main` branch.
5. In Vercel, create a new project and import `Harshita1803/CPAF-Privacy-Assessment-Vercel`.
6. Use the project root, leave the build command empty/default, and deploy. `vercel.json` routes `/api/*` to the FastAPI function and serves `index.html` as the UI.
7. After deployment, open the website and run a small experiment (e.g. 500 records) first.

## What this edition exposes

- Synthetic record generation, dataset validation, generalisation and k-suppression.
- k/l/t-style privacy diagnostic output and a preview of generated records.
- Evidence entry and provisional evidence-weighted summary for Paytm, Google Pay and PhonePe.
- JSON and CSV downloads.

## What it does not yet expose as web controls

The original research code and docs are preserved in the repository, including model-training comparisons, repeated utility/fairness experiments, membership-inference audits, DP-SGD and parameter sweeps. These computationally heavier workflows are **not** wired into this first Vercel UI. Continue using the existing Streamlit deployment for the complete original dashboard.

## Research integrity

All generated transaction-like records are synthetic. The evidence score is only as reliable as reviewer-entered, documented public evidence. It is not an independent technical audit or a legal-compliance conclusion. Generalisation, suppression, and k/l/t-style diagnostics are not formal differential privacy. Do not describe the unverified DP accountant in the preserved research code as a formal privacy guarantee.
