"""Navigable dashboard that calls the same functions as the CPAF CLI."""
from pathlib import Path

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit("Dashboard requires: python -m pip install -r requirements-dashboard.txt") from exc

import pandas as pd
from cpaf.attacks import run_shadow_mia
from cpaf.core import (generate_synthetic, generalize, microaggregate,
                       privacy_diagnostics, suppress_k, validate_dataset)
from cpaf.evaluation import run_repeated_utility
from cpaf.experiments import run_privacy_sweep
from cpaf.reproducibility import create_unique_run_dir

ROOT = Path(__file__).parent
OUT = ROOT / "cpaf_outputs" / "dashboard"
st.set_page_config(page_title="CPAF 2.0 Research Dashboard", layout="wide")
st.title("CPAF 2.0 · Comparative Privacy Research")
st.caption("Public evidence and synthetic experiments are separate. Synthetic results do not establish actual provider practices.")
sections = [
    "Overview and scope", "Payment-app evidence scoring", "Synthetic data generation",
    "Privacy transformations", "k / l / t diagnostics", "Utility and fairness",
    "Membership and disclosure attacks", "Differential privacy audit",
    "Parameter sweep and Pareto", "Downloads and limitations",
]
section = st.sidebar.radio("Research section", sections)


def saved_df(filename):
    path = ROOT / "cpaf_outputs" / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Saved result could not be read ({path.name}): {exc}")
        return None


if section == "Overview and scope":
    st.markdown("""### Two independent research tracks
- **Evidence review:** reviewer-entered public disclosures for Paytm, Google Pay and PhonePe. Missing evidence is not scored as good or bad.
- **Synthetic experiment:** reproducible artificial rows for privacy, utility, fairness and attack diagnostics.

Generalization, suppression, l-diversity checks and t-closeness checks are empirical techniques. They do not provide differential privacy. The DP-SGD accountant here is unverified, so its epsilon outputs are not formal guarantees.""")
    st.write("CLI and dashboard share the same cpaf functions. Results appear only after an experiment completes.")

elif section == "Payment-app evidence scoring":
    from cpaf.evidence import (REQUIRED_EVIDENCE_COLUMNS, available_evidence_sources,
                               evidence_frames_equal, load_evidence_source,
                               read_evidence_csv, save_evidence_csv)
    from cpaf_advanced import (compute_app_metrics, compute_weight_sensitivity,
                               format_evidence_for_display, validate_evidence_rows)

    template_path = ROOT / "cpaf_outputs" / "evidence_register_template.csv"
    review_path = ROOT / "cpaf_outputs" / "evidence_reviews" / "evidence_register_reviewed.csv"
    sources = available_evidence_sources(template_path, review_path)

    if "evidence_loaded" not in st.session_state:
        try:
            if sources:
                initial_label = next(iter(sources))
                (st.session_state.evidence_loaded,
                 st.session_state.evidence_source_label,
                 st.session_state.evidence_source_path) = load_evidence_source(sources, initial_label)
            else:
                st.session_state.evidence_loaded = pd.DataFrame(columns=REQUIRED_EVIDENCE_COLUMNS)
                st.session_state.evidence_source_label = "No CSV loaded"
                st.session_state.evidence_source_path = "(none)"
            st.session_state.evidence_revision = 0
        except (OSError, ValueError, TypeError) as exc:
            st.error(f"Could not load the evidence template: {exc}")
            st.stop()

    st.markdown("Review each row using a specific claim and traceable source. Set **review_status** to `reviewed` only after checking the source; then enter the access date, reviewer, rationale, evidence strength, and a 1–5 score. Completing fields does not itself verify that a claim is true.")
    st.caption("Seven dimensions remain fixed: Data Minimisation; Consent & Transparency; Third-Party Sharing; Security Controls; Anonymisation & Re-identification; Data Lifecycle & User Rights; DPDP Alignment. Scores are evidence-based reviewer assessments, not verified provider technical audits.")
    if not template_path.exists():
        st.warning("The original template is missing. No evidence rows were created; select an existing working CSV or upload a valid evidence CSV.")
    save_notice = st.session_state.pop("evidence_save_notice", None)
    if save_notice:
        st.success(save_notice)
    selected_source = st.selectbox("Choose a saved CSV to load", list(sources),
                                   index=0 if sources else None, disabled=not sources)
    uploaded = st.file_uploader("Or choose another evidence CSV", type=["csv"], key="evidence_upload")

    current_label = st.session_state.evidence_source_label
    current_path = st.session_state.evidence_source_path
    original = format_evidence_for_display(st.session_state.evidence_loaded)
    editor_key = f"evidence_editor_{st.session_state.evidence_revision}"
    edited = st.data_editor(format_evidence_for_display(original), use_container_width=True,
                            num_rows="fixed", key=editor_key)
    dirty = not evidence_frames_equal(edited, original)
    st.write(f"**Currently editing:** {current_label} — `{current_path}`")
    st.info("Unsaved changes are present." if dirty else "No unsaved changes in this editing session.")

    eligible, row_issues = validate_evidence_rows(edited)
    reviewed_count = int(edited.get("review_status", pd.Series(dtype=str)).astype(str).str.strip().str.lower().eq("reviewed").sum())
    incomplete_count = sum(bool(items) for items in row_issues)
    st.write(f"Rows: **{len(edited)} total** · **{reviewed_count} marked reviewed** · **{incomplete_count} incomplete** · **{int(eligible.sum())} score-eligible**")
    if incomplete_count:
        missing_rows = [{"row": i + 1, "app": edited.iloc[i].get("app", ""),
                         "dimension": edited.iloc[i].get("dimension", ""),
                         "missing requirements": ", ".join(row_issues[i])}
                        for i in range(len(row_issues)) if row_issues[i]]
        st.caption("Incomplete rows and unmet requirements (nothing is filled automatically):")
        st.dataframe(pd.DataFrame(missing_rows), use_container_width=True, hide_index=True)
    if "evidence_strength" in edited.columns:
        strengths = pd.to_numeric(edited["evidence_strength"], errors="coerce")
        strength_values_present = edited["evidence_strength"].notna() & edited["evidence_strength"].astype(str).str.strip().ne("")
        invalid_strength = strength_values_present & (strengths.isna() | ~strengths.between(0, 1))
        if invalid_strength.any():
            st.warning(f"{int(invalid_strength.sum())} row(s) have evidence_strength values that are not numeric from 0 to 1. Review these values; they are not automatically corrected.")

    pending = st.session_state.get("evidence_pending_load")
    if pending:
        st.warning("Loading another file will discard the current unsaved edits.")
        c1, c2 = st.columns(2)
        if c1.button("Discard edits and load selected CSV", type="primary"):
            try:
                if isinstance(pending["source"], (str, Path)):
                    pending_frame, _, _ = load_evidence_source(
                        {pending["label"]: pending["source"]}, pending["label"])
                else:
                    pending_frame = read_evidence_csv(pending["source"])
                st.session_state.evidence_loaded = pending_frame
                st.session_state.evidence_source_label = pending["label"]
                st.session_state.evidence_source_path = pending["path"]
                st.session_state.evidence_revision += 1
                st.session_state.pop("evidence_pending_load", None)
                st.rerun()
            except (OSError, ValueError, TypeError) as exc:
                st.error(f"Could not load selected evidence: {exc}")
        if c2.button("Keep editing"):
            st.session_state.pop("evidence_pending_load", None)
            st.rerun()
    else:
        c1, c2, c3 = st.columns(3)
        if c1.button("Load selected CSV"):
            candidate = sources[selected_source]
            if dirty:
                st.session_state.evidence_pending_load = {"source": candidate, "path": str(candidate), "label": selected_source}
                st.rerun()
            else:
                try:
                    (st.session_state.evidence_loaded,
                     st.session_state.evidence_source_label,
                     st.session_state.evidence_source_path) = load_evidence_source(
                        sources, selected_source)
                    st.session_state.evidence_revision += 1
                    st.rerun()
                except (OSError, ValueError, TypeError) as exc:
                    st.error(f"Could not load selected evidence: {exc}")
        if c2.button("Load uploaded CSV", disabled=uploaded is None):
            try:
                upload_bytes = uploaded.getvalue()
                upload_name = Path(uploaded.name).name
                if dirty:
                    st.session_state.evidence_pending_load = {"source": upload_bytes, "path": f"uploaded: {upload_name}", "label": "Uploaded CSV"}
                    st.rerun()
                else:
                    st.session_state.evidence_loaded = read_evidence_csv(upload_bytes)
                    st.session_state.evidence_source_label = "Uploaded CSV"
                    st.session_state.evidence_source_path = f"uploaded: {upload_name}"
                    st.session_state.evidence_revision += 1
                    st.rerun()
            except (OSError, ValueError, TypeError) as exc:
                st.error(f"Uploaded evidence CSV could not be loaded: {exc}")
        if c3.button("Reload saved evidence", disabled=not review_path.exists()):
            if dirty:
                st.session_state.evidence_pending_load = {"source": review_path, "path": str(review_path), "label": "Saved review (working copy)"}
                st.rerun()
            else:
                try:
                    (st.session_state.evidence_loaded,
                     st.session_state.evidence_source_label,
                     st.session_state.evidence_source_path) = load_evidence_source(
                        sources, "Saved review (working copy)")
                    st.session_state.evidence_revision += 1
                    st.rerun()
                except (OSError, ValueError, TypeError) as exc:
                    st.error(f"Saved evidence could not be reloaded: {exc}")

    st.divider()
    overwrite_confirmed = st.checkbox(
        "Confirm replacing the existing saved review (a timestamped backup will be kept)",
        disabled=not review_path.exists(),
        key=f"evidence_confirm_overwrite_{st.session_state.evidence_revision}")
    if st.button("Save reviewed evidence", type="primary"):
        try:
            existed = review_path.exists()
            saved_path = save_evidence_csv(edited, review_path, confirm_overwrite=overwrite_confirmed)
            st.session_state.evidence_loaded = edited.copy()
            st.session_state.evidence_source_label = "Saved review (working copy)"
            st.session_state.evidence_source_path = str(saved_path)
            st.session_state.evidence_revision += 1
            st.session_state.evidence_save_notice = f"Evidence review saved to {saved_path}" + ("; the previous saved review was backed up." if existed else ".")
            st.rerun()
        except (OSError, ValueError, TypeError) as exc:
            st.error(f"Evidence review was not saved: {exc}")

    if st.button("Compute provisional evidence scores"):
        if not eligible.any():
            st.info("No rows meet every review, source, access-date, claim, reviewer, rationale, and valid-score requirement. No provider score is calculated.")
        else:
            scores = compute_app_metrics(edited)
            st.dataframe(scores, use_container_width=True)
            st.dataframe(compute_weight_sensitivity(scores), use_container_width=True)
    st.download_button("Download current evidence CSV", edited.to_csv(index=False),
                       "evidence_register.csv", "text/csv")

elif section == "Synthetic data generation":
    st.caption("Synthetic-only records generated by this project; no payment-provider or customer data are used.")
    n = st.number_input("Records", min_value=100, max_value=250_000, value=5_000, step=1_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    if st.button("Generate and validate synthetic data"):
        try:
            data = generate_synthetic(int(n), int(seed))
            st.session_state["data"] = data
            st.json(validate_dataset(data))
            st.dataframe(data.head(20), use_container_width=True)
            st.download_button("Download generated CSV", data.to_csv(index=False),
                               "synthetic_transactions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

elif section == "Privacy transformations":
    st.caption("Transformations and metrics on the in-session synthetic dataset only.")
    data = st.session_state.get("data")
    if data is None:
        st.warning("Generate synthetic data first.")
    else:
        k = st.number_input("k", min_value=1, value=5)
        level = st.select_slider("Common hierarchy level", options=[0, 1, 2], value=1)
        if st.button("Generalize, suppress, and measure microaggregation distortion"):
            try:
                generalized = generalize(data, level, level, level, level)
                qis = ["age_band", "location", "merchant_type", "amount_band", "time_band"]
                released, kdiag = suppress_k(generalized, qis, int(k))
                _, distortion = microaggregate(data, ["age", "amount_inr"], group_size=5)
                st.session_state["generalized"] = generalized
                st.session_state["released"] = released
                st.json({"suppression": kdiag, "microaggregation": distortion})
                st.dataframe(released.head(20), use_container_width=True)
                st.download_button("Download released CSV", released.to_csv(index=False),
                                   "generalized_k_release.csv", "text/csv")
            except Exception as exc:
                st.error(f"Transformation failed: {exc}")

elif section == "k / l / t diagnostics":
    st.caption("Diagnostics apply to the selected released synthetic table and listed quasi-identifiers only.")
    released = st.session_state.get("released")
    if released is None:
        st.warning("Run the transformation section first.")
    else:
        l = st.number_input("Distinct l threshold", min_value=1, value=2)
        t = st.slider("Categorical total-variation t threshold", 0.0, 1.0, .2)
        qis = ["age_band", "location", "merchant_type", "amount_band", "time_band"]
        try:
            st.json(privacy_diagnostics(released, qis, categorical_t=t, l_threshold=int(l)))
            st.caption("Missing QI values form a group. Missing sensitive values are rejected. These are diagnostics, not guarantees.")
        except Exception as exc:
            st.error(f"Diagnostic failed: {exc}")

elif section == "Utility and fairness":
    st.caption("Utility and age-group metrics evaluate the artificial target on generated synthetic records; they are not real-world model estimates.")
    n = st.number_input("Synthetic records", min_value=500, max_value=100_000, value=5_000, step=500)
    repeats = st.number_input("Stratified repeats", min_value=2, max_value=10, value=3)
    seed = st.number_input("Evaluation seed", min_value=0, value=42)
    k = st.number_input("k for training suppression", min_value=1, value=5)
    if st.button("Run repeated utility and age-group audit"):
        try:
            with st.spinner("Running repeated stratified splits..."):
                run_dir = create_unique_run_dir(OUT / "utility", "run")
                _, summary = run_repeated_utility(int(n), int(seed), int(repeats), int(k), run_dir)
            st.dataframe(summary, use_container_width=True)
            st.dataframe(pd.read_csv(run_dir / "synthetic_age_fairness_gaps.csv"), use_container_width=True)
            st.caption(f"Saved this run to {run_dir.relative_to(ROOT)}")
            st.caption("Age-band gaps describe only this artificial label and sample.")
        except Exception as exc:
            st.error(f"Utility evaluation failed: {exc}")

elif section == "Membership and disclosure attacks":
    st.caption("Attack and uniqueness results are empirical synthetic-data probes, not proof of privacy or evidence about payment-app users.")
    n = st.number_input("Synthetic records", min_value=500, max_value=30_000, value=3_000, step=500)
    seed = st.number_input("Attack seed", min_value=0, value=19)
    k = st.number_input("k for generalized training release", min_value=1, value=2)
    if st.button("Run shadow-model attack and disclosure proxies"):
        try:
            with st.spinner("Training target, shadow, and attack models..."):
                run_dir = create_unique_run_dir(OUT / "attacks", "run")
                results, proxies = run_shadow_mia(int(n), int(seed), int(k), run_dir)
            st.dataframe(results, use_container_width=True)
            st.json(proxies)
            st.caption(f"Saved this run to {run_dir.relative_to(ROOT)}")
            st.caption("AUC near 0.5 is chance-level performance for this attack only; it does not prove privacy.")
        except Exception as exc:
            st.error(f"Attack experiment failed: {exc}")

elif section == "Differential privacy audit":
    st.markdown("""### EXPERIMENTAL_UNVERIFIED — no formal guarantee claimed
Per-example L2 clipping at C bounds each clipped gradient contribution. Each step uses independent Bernoulli inclusion at rate q and Gaussian noise with standard deviation sigma·C on the clipped sum; the update divides by q·n. Sampling and noise use OS-backed SystemRandom and are not seeded from the experiment seed. The custom accountant has not been independently validated. The CLI selects member rows using a stratified split and publishes metrics outside any proved DP boundary. Epsilon is an unverified diagnostic, not a formal bound.""")
    st.info("Trusted-library migration is not active. Opacus secure mode is blocked by the current Windows/Python dependency combination; diffprivlib is compatible on paper but could not be installed because package-index DNS was unavailable. No library-backed result or VERIFIED_BOUND is reported. See docs/DP_LIBRARY_MIGRATION_REPORT.md.")
    dp = saved_df("dp_audit_verified/ml_dp_results.csv")
    if dp is None:
        dp = saved_df("final_ml_dp_audit/ml_dp_results.csv")
    if dp is None:
        dp = saved_df("ml_dp_results.csv")
    if dp is not None:
        st.dataframe(dp, use_container_width=True)
        st.caption("Status shown in each row: EXPERIMENTAL_UNVERIFIED. No valid epsilon bound is asserted.")
    else:
        st.info("NOT_APPLICABLE — no saved DP mechanism result is available.")
    st.warning("The result CSV records mechanism metadata and unaccounted operations. No formal DP guarantee is claimed for the training pipeline or its reported metrics.")

elif section == "Parameter sweep and Pareto":
    n = st.number_input("Rows per seed", min_value=100, max_value=20_000, value=1_000, step=500)
    seed = st.number_input("Sweep seed", min_value=0, value=123)
    if st.button("Run default k/l/t and hierarchy sweep"):
        try:
            with st.spinner("Evaluating parameter grid..."):
                run_dir = create_unique_run_dir(OUT / "sweep", "run")
                result, frontier = run_privacy_sweep(
                    int(n), [int(seed)], [1, 5, 10], [2, 3], [.1, .2], [1, 2],
                    run_dir)
            st.write(f"{len(result)} configurations; {len(frontier)} non-dominated configurations.")
            st.dataframe(frontier, use_container_width=True)
            st.image(str(run_dir / "pareto_frontier.png"))
            st.caption(f"Saved this run to {run_dir.relative_to(ROOT)}")
            st.caption("Pareto objectives: maximize retention and minimize worst categorical-TV distance within this grid only.")
        except Exception as exc:
            st.error(f"Sweep failed: {exc}")

else:
    st.markdown("### Saved result files")
    base = ROOT / "cpaf_outputs"
    if base.exists():
        files = sorted(p for p in base.rglob("*") if p.is_file())
        st.dataframe([{"file": str(p.relative_to(base)), "bytes": p.stat().st_size} for p in files],
                     use_container_width=True)
        if files:
            selected = st.selectbox("Choose a result to download", files,
                                    format_func=lambda p: str(p.relative_to(base)))
            st.download_button("Download selected file", selected.read_bytes(), selected.name)
    else:
        st.info("No experiment results have been saved.")
    st.markdown("""### Limitations
- All experimental rows and the classification label are synthetic.
- Empirical k/l/t, MIA, uniqueness and nearest-neighbor checks do not certify privacy.
- Fairness figures reflect the artificial label-generation rule.
- DP accountant and implementation are unverified; no formal epsilon guarantee is claimed.
- App scores require traceable reviewed evidence and remain provisional.""")
