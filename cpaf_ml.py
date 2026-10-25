"""
CPAF 2.0 — Machine-Learning Privacy Lab (model training extension)

Everything here runs on the SYNTHETIC dataset from cpaf_advanced.py. Nothing is
measured on Paytm, Google Pay or PhonePe, and no result may be presented as such.

Experiments
  1. Utility of privacy transforms: train 3 model families on raw / generalised /
     k-anonymous data and compare predictive performance on one common test set.
  2. k-sweep: how model quality and retained records change as k grows.
  3. Membership-inference attack (MIA): can an attacker tell whether a record was
     in the training set? Compared across data representations and regularisation.
  4. Differential privacy: DP-SGD logistic regression (implemented from scratch in
     NumPy) with an RDP privacy accountant -> epsilon vs. utility vs. MIA leakage.
"""
from pathlib import Path
from datetime import datetime, timezone
import platform
import json
import secrets
from importlib.metadata import PackageNotFoundError, version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp, gammaln, expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cpaf_advanced import make_synthetic_transactions, generalise, OUT
from cpaf.reproducibility import require_outputs_absent

QUASI = ["age_band", "location", "merchant_type", "amount_band", "time_band"]
REGIONS = ["North", "South", "East", "West", "Central"]
MERCHANTS = ["Grocery", "Utilities", "Food", "Travel", "Retail", "Education"]
CITIES = [f"City_{i:02d}" for i in range(1, 31)]

# Chart colours: categorical slots 1-3 of the reference palette (light surface).
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


# ----------------------------------------------------------------------------
# Synthetic target label + feature views
# ----------------------------------------------------------------------------
def add_synthetic_label(raw, seed=7):
    """Noisy synthetic 'suspicious activity' flag (~8-10% positive).
    It depends on amount, night-time, merchant and age PLUS noise, so models cannot
    reach perfect accuracy and overfitting is observable. Purely artificial."""
    rng = np.random.default_rng(seed)
    z = (-3.4 + 0.9 * (np.log(raw["amount_inr"]) - 6.1) / 0.85
         + 1.2 * (raw["hour"] < 5) + 0.8 * (raw["merchant_type"] == "Travel")
         + 0.5 * (raw["age"] > 65) + rng.normal(0, 0.8, len(raw)))
    y = (rng.random(len(raw)) < expit(z)).astype(int)
    return np.asarray(y)


def raw_view(raw):
    """Fine-grained features, including a city identifier (high-cardinality)."""
    return pd.DataFrame({"age": raw["age"], "hour": raw["hour"],
                         "log_amount": np.log(raw["amount_inr"]),
                         "region": raw["region"], "merchant_type": raw["merchant_type"],
                         "city": raw["city"]})


def gen_view(raw):
    """Generalised features only (risk_category is excluded: it is derived from the
    raw amount/hour and would leak the label's drivers)."""
    return generalise(raw)[QUASI]


def k_mask(df, k):
    """Boolean mask of rows whose quasi-identifier group has >= k members."""
    return (df.groupby(QUASI, dropna=False)[QUASI[0]].transform("size") >= k).to_numpy()


def make_model(name, cat_cols, num_cols, **kw):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols)] if num_cols else
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])
    if name == "Logistic Regression":
        est = LogisticRegression(max_iter=2000, C=kw.get("C", 1.0))
    elif name == "Random Forest":
        est = RandomForestClassifier(n_estimators=kw.get("n_estimators", 150),
                                     min_samples_leaf=kw.get("min_samples_leaf", 5),
                                     n_jobs=-1, random_state=0)
    else:
        est = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=0)
    return make_pipeline(pre, est)


def cols_of(view):
    num = [c for c in ["age", "hour", "log_amount"] if c in view.columns]
    return [c for c in view.columns if c not in num], num


def scores(y, p):
    return {"ROC_AUC": round(roc_auc_score(y, p), 4),
            "PR_AUC": round(average_precision_score(y, p), 4),
            "Brier": round(brier_score_loss(y, p), 4)}


# ----------------------------------------------------------------------------
# Differential privacy: DP-SGD logistic regression + RDP accountant
# ----------------------------------------------------------------------------
def fixed_encode(raw):
    """Data-independent encoding (fixed public bounds), so no statistics are learned
    from the private data before DP-SGD starts."""
    lo, hi = np.log(20), np.log(150000)
    cols = [raw["age"].to_numpy() / 100.0, raw["hour"].to_numpy() / 23.0,
            (np.log(raw["amount_inr"].to_numpy()) - lo) / (hi - lo)]
    for name, cats in [("region", REGIONS), ("merchant_type", MERCHANTS), ("city", CITIES)]:
        cols += [(raw[name].to_numpy() == c).astype(float) for c in cats]
    cols.append(np.ones(len(raw)))  # bias
    return np.column_stack(cols)


def rdp_sampled_gaussian(q, sigma, alpha):
    """Custom integer-order subsampled-Gaussian RDP calculation; unverified."""
    if isinstance(q, (bool, np.bool_)) or not np.isfinite(q) or not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    if isinstance(sigma, (bool, np.bool_)) or sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    if isinstance(alpha, bool) or int(alpha) != alpha or alpha < 2:
        raise ValueError("alpha must be an integer >= 2")
    if q == 1:
        return float(alpha / (2 * sigma ** 2))
    k = np.arange(alpha + 1)
    log_binom = gammaln(alpha + 1) - gammaln(k + 1) - gammaln(alpha - k + 1)
    terms = log_binom + (alpha - k) * np.log1p(-q) + k * np.log(q) + (k * k - k) / (2 * sigma ** 2)
    return float(logsumexp(terms) / (alpha - 1))


def clip_per_example_gradients(gradients, clip_norm):
    """L2-clip each row independently; returns a copy for auditable tests."""
    gradients = np.asarray(gradients, dtype=float)
    if gradients.ndim != 2 or not np.isfinite(gradients).all():
        raise ValueError("gradients must be a finite 2D array")
    if isinstance(clip_norm, (bool, np.bool_)) or not np.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    norms = np.linalg.norm(gradients, axis=1)
    factors = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
    return gradients * factors[:, None]


def dp_epsilon(q, sigma, steps, delta, num_runs=1):
    """Unverified experimental RDP composition/conversion estimate, not a certified bound."""
    if isinstance(delta, (bool, np.bool_)) or not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if isinstance(steps, bool) or int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(num_runs, bool) or int(num_runs) != num_runs or num_runs < 1:
        raise ValueError("num_runs must be a positive integer")
    orders = np.arange(2, 129)
    eps = [steps * num_runs * rdp_sampled_gaussian(q, sigma, int(a)) + np.log(1 / delta) / (a - 1) for a in orders]
    return float(min(eps))


class DPLogReg:
    """Experimental DP-SGD-style logistic training; accounting/pipeline stay unverified.

    Sampling and Gaussian noise use OS-backed SystemRandom. A public seed must not
    determine DP coins. `seed` remains in the signature for compatibility but a
    non-None value is rejected rather than silently making the mechanism repeatable.
    """
    def __init__(self, sigma, clip=1.0, q=0.01, steps=800, lr=2.0, seed=None, delta=1e-5):
        if isinstance(sigma, (bool, np.bool_)) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive; sigma=0 is not DP")
        if isinstance(clip, (bool, np.bool_)) or not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be finite and positive")
        if isinstance(q, (bool, np.bool_)) or not np.isfinite(q) or not 0 < q <= 1:
            raise ValueError("q must be in (0, 1]")
        if isinstance(steps, bool) or int(steps) != steps or steps < 1:
            raise ValueError("steps must be a positive integer")
        if isinstance(lr, (bool, np.bool_)) or not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        if isinstance(delta, (bool, np.bool_)) or not np.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("delta must be finite and in (0, 1)")
        if seed is not None:
            raise ValueError("DP randomness is OS-backed and cannot be initialized from a reproducibility seed")
        self.sigma, self.clip, self.q, self.steps, self.lr, self.delta = sigma, clip, q, steps, lr, delta
        self.accounting_status = "experimental; custom accountant not independently validated"
        self.verification_status = "EXPERIMENTAL_UNVERIFIED"
        self.adjacency_assumption = "nominal add/remove-one for clipped sum; end-to-end size adjacency unverified"
        self.sampling_assumption = "independent Poisson (Bernoulli) inclusion with rate q per step"
        self.randomness_source = "secrets.SystemRandom (OS-backed; not seeded from experiment seed)"

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D feature matrix")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y must be a 1D vector matching X rows")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values")
        if not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("y must contain binary labels 0 and 1")
        rng = secrets.SystemRandom()
        n, d = X.shape
        w = np.zeros(d)
        for t in range(self.steps):
            m = np.fromiter((rng.random() < self.q for _ in range(n)), dtype=bool, count=n)
            g = np.zeros(d)
            if m.any():
                Xb, yb = X[m], y[m]
                gi = (expit(Xb @ w) - yb)[:, None] * Xb
                gi = clip_per_example_gradients(gi, self.clip)
                g = gi.sum(0)
            noise = np.fromiter((rng.gauss(0.0, self.sigma * self.clip) for _ in range(d)),
                                dtype=float, count=d)
            g = g + noise
            w -= self.lr / (1 + t / 300) * g / (self.q * n)
        self.w = w
        self.epsilon_unverified = dp_epsilon(self.q, self.sigma, self.steps, self.delta)
        return self

    def predict_proba1(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.w) or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D feature matrix with the fitted feature count")
        return expit(X @ self.w)


def dp_verification_fields(model):
    """Canonical result status; this project currently has no verified DP path."""
    status = "NOT_APPLICABLE" if model is None else "EXPERIMENTAL_UNVERIFIED"
    return {"verification_status": status, "formal_guarantee_claimed": False}


# ----------------------------------------------------------------------------
# Membership inference
# ----------------------------------------------------------------------------
def mia_auc(p_members, y_members, p_non, y_non):
    """Confidence attack: score = probability assigned to the TRUE label.
    AUC 0.5 is chance-level performance; this does not rule out other attacks."""
    cm = np.where(y_members == 1, p_members, 1 - p_members)
    cn = np.where(y_non == 1, p_non, 1 - p_non)
    return float(roc_auc_score(np.r_[np.ones(len(cm)), np.zeros(len(cn))], np.r_[cm, cn]))


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def style(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------
def run_ml_lab(n=50000, k=5, seed=42, mia_m=2000, output_dir=None):
    global OUT
    if output_dir is not None:
        OUT = Path(output_dir)
    output_files = ("ml_utility_results.csv", "ml_k_sweep.csv", "ml_dp_results.csv",
                    "ml_mia_results.csv", "ml_summary.json", "ml_utility.png",
                    "ml_k_sweep.png", "ml_mia.png", "ml_dp_tradeoff.png",
                    "ml_experiment_manifest.json")
    require_outputs_absent(OUT, output_files)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = make_synthetic_transactions(n, seed)
    y = add_synthetic_label(raw, seed + 1)
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.3, stratify=y, random_state=seed)
    raw_tr, raw_te = raw.iloc[idx_tr].reset_index(drop=True), raw.iloc[idx_te].reset_index(drop=True)
    y_tr, y_te = y[idx_tr], y[idx_te]
    print(f"[ML] positives: {y.mean()*100:.1f}%  train={len(idx_tr)}  test={len(idx_te)}")

    views = {"Raw": (raw_view(raw_tr), raw_view(raw_te), np.ones(len(raw_tr), bool)),
             "Generalised": (gen_view(raw_tr), gen_view(raw_te), np.ones(len(raw_tr), bool))}
    gtr = gen_view(raw_tr)
    views[f"Generalised + k-anon (k={k})"] = (gtr, gen_view(raw_te), k_mask(gtr, k))

    # --- Experiment 1: utility across representations and model families --------------
    rows = []
    for vname, (Xtr, Xte, keep) in views.items():
        cat, num = cols_of(Xtr)
        for mname in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
            mdl = make_model(mname, cat, num).fit(Xtr[keep], y_tr[keep])
            rows.append({"representation": vname, "model": mname, "train_rows": int(keep.sum()),
                         **scores(y_te, mdl.predict_proba(Xte)[:, 1])})
    util = pd.DataFrame(rows); util.to_csv(OUT / "ml_utility_results.csv", index=False)
    print("\n[ML] Experiment 1 — utility (common held-out test set)"); print(util.to_string(index=False))

    # --- Experiment 2: k-sweep --------------------------------------------------------
    rows = []
    skipped_k_values = []
    Xte_g = gen_view(raw_te); cat, num = cols_of(gtr)
    for kk in [1, 2, 5, 10, 15, 20, 30]:
        keep = k_mask(gtr, kk)
        if keep.sum() < 500:   # too little data left to train a meaningful model
            skipped_k_values.append({"k": kk, "retained_training_rows": int(keep.sum()),
                                     "reason": "fewer than 500 rows retained"})
            continue
        mdl = make_model("Gradient Boosting", cat, num).fit(gtr[keep], y_tr[keep])
        rows.append({"k": kk, "records_retained_pct": round(100 * keep.mean(), 2),
                     **scores(y_te, mdl.predict_proba(Xte_g)[:, 1])})
    ksw = pd.DataFrame(rows); ksw.to_csv(OUT / "ml_k_sweep.csv", index=False)
    print("\n[ML] Experiment 2 — k sweep (Gradient Boosting on generalised data)"); print(ksw.to_string(index=False))

    # --- Experiment 3: membership inference ------------------------------------------
    rng = np.random.default_rng(seed)
    keep_k = views[f"Generalised + k-anon (k={k})"][2]
    mem_pool = np.flatnonzero(keep_k)
    requested_mia_m = int(mia_m)
    mia_m = min(requested_mia_m, len(mem_pool), len(idx_te))
    if mia_m < 1:
        raise ValueError("MIA requires at least one retained member and one held-out nonmember")
    mem = rng.choice(mem_pool, mia_m, replace=False)
    non = rng.choice(len(idx_te), mia_m, replace=False)
    raw_m, raw_n = raw_tr.iloc[mem], raw_te.iloc[non]
    ym, yn = y_tr[mem], y_te[non]
    # held-out test reference for utility of each attacked model
    mia_rows = []

    def attack(label, model_fn, Xm, Xn, Xtest):
        mdl = model_fn(Xm)
        pm, pn = mdl(Xm), mdl(Xn)
        mia_rows.append({"configuration": label,
                         "train_ROC_AUC": round(roc_auc_score(ym, pm), 4),
                         "test_ROC_AUC": round(roc_auc_score(y_te, mdl(Xtest)), 4),
                         "MIA_attack_AUC": round(mia_auc(pm, ym, pn, yn), 4)})

    def sk(name, **kw):
        def f(X):
            cat_, num_ = cols_of(X)
            m = make_model(name, cat_, num_, **kw).fit(X, ym)
            return lambda Z: m.predict_proba(Z)[:, 1]
        return f

    Rm, Rn, Rt = raw_view(raw_m), raw_view(raw_n), raw_view(raw_te)
    Gm, Gn, Gt = gen_view(raw_m), gen_view(raw_n), gen_view(raw_te)
    attack("Raw, RF overfit (leaf=1)", sk("Random Forest", min_samples_leaf=1), Rm, Rn, Rt)
    attack("Raw, RF regularised (leaf=50)", sk("Random Forest", min_samples_leaf=50), Rm, Rn, Rt)
    attack("Generalised + k-anon, RF overfit (leaf=1)", sk("Random Forest", min_samples_leaf=1), Gm, Gn, Gt)
    attack("Generalised + k-anon, RF regularised (leaf=50)", sk("Random Forest", min_samples_leaf=50), Gm, Gn, Gt)

    # DP-SGD models on the same members/non-members
    Xm_dp, Xn_dp, Xt_dp = fixed_encode(raw_m), fixed_encode(raw_n), fixed_encode(raw_te)
    dp_rows = []
    for sigma in [0.7, 1.0, 1.5, 3.0, 6.0]:
        q = 0.05  # larger sampling rate: the member set is small (2,000 records)
        m = DPLogReg(sigma, q=q, steps=400, lr=30.0, delta=1e-5).fit(Xm_dp, ym)
        pm, pn, pt = m.predict_proba1(Xm_dp), m.predict_proba1(Xn_dp), m.predict_proba1(Xt_dp)
        eps = m.epsilon_unverified
        dp_rows.append({"noise_multiplier": sigma,
                        "epsilon_experimental_unverified": round(eps, 2),
                        **dp_verification_fields(m),
                        "mechanism_name": "custom Poisson-sampled clipped Gaussian logistic training",
                        "protected_unit": "one row of the member arrays passed to DPLogReg.fit",
                        "adjacency_definition": "nominal add/remove-one for fit input; upstream raw-data mapping unverified",
                        "delta_scope": "fixed public delta=1e-5 for each candidate model; composition across released candidates is not applied",
                        "accountant": "custom integer-order sampled-Gaussian RDP; not independently validated",
                        "accountant_version": "project-custom-unverified-v1",
                        "randomness_source": m.randomness_source,
                        "guarantee_scope": "no formal guarantee claimed; model and reported metrics not certified",
                        "unaccounted_operations": "stratified private-data partitioning, MIA member selection, model comparison, private-data metrics, and custom accountant validation",
                        "delta": f"{m.delta:.1e}", "sample_rate": q,
                        "steps": m.steps, "clip_norm": m.clip,
                        "adjacency_assumption": m.adjacency_assumption,
                        "sampling_assumption": m.sampling_assumption,
                        "accounting_status": m.accounting_status,
                        "test_ROC_AUC": round(roc_auc_score(y_te, pt), 4),
                        "test_PR_AUC": round(average_precision_score(y_te, pt), 4),
                        "MIA_attack_AUC": round(mia_auc(pm, ym, pn, yn), 4)})
    dp = pd.DataFrame(dp_rows); dp.to_csv(OUT / "ml_dp_results.csv", index=False)
    for r in dp_rows:
        if r["noise_multiplier"] in (1.0,):
            mia_rows.append({"configuration": f"DP-SGD logistic reg. (sigma=1.0, unverified eps={r['epsilon_experimental_unverified']})",
                             "train_ROC_AUC": np.nan, "test_ROC_AUC": r["test_ROC_AUC"],
                             "MIA_attack_AUC": r["MIA_attack_AUC"]})
    mia = pd.DataFrame(mia_rows); mia.to_csv(OUT / "ml_mia_results.csv", index=False)
    print("\n[ML] Experiment 3 — membership inference (AUC 0.5 = chance baseline)"); print(mia.to_string(index=False))
    print("\n[ML] Experiment 4 — custom DP-SGD/accountant outputs (EXPERIMENTAL_UNVERIFIED; no formal guarantee)"); print(dp.to_string(index=False))

    plot_ml(util, ksw, mia, dp, k)
    summary = {"synthetic_only": True, "n_records": n, "k": k, "positive_rate_pct": round(100 * float(y.mean()), 2),
               "best_utility_row": util.sort_values("ROC_AUC", ascending=False).iloc[0].to_dict()}
    pd.Series(summary).to_json(OUT / "ml_summary.json", indent=2)
    packages={}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]:
        try:
            packages[package]=version(package)
        except PackageNotFoundError:
            packages[package]="not-installed"
    manifest={"timestamp_utc":datetime.now(timezone.utc).isoformat(), "python":platform.python_version(),
              "seed":seed, "label_seed":seed + 1, "dataset_size":n,
              "method":"raw/generalized/k-suppressed + ML/MIA/experimental-DP-SGD",
              "parameters":{"k":k,"mia_sample_size":mia_m,"train_fraction":.7,
                            "dp_noise_multipliers":[.7,1.0,1.5,3.0,6.0],
                            "dp_delta":1e-5,
                            "mia_member_and_nonmember_rows_each":mia_m,
                            "mia_requested_rows_each":requested_mia_m},
              "model_configuration":["LogisticRegression", "RandomForestClassifier", "HistGradientBoostingClassifier"],
              "metric_definitions":{"utility":["ROC_AUC", "PR_AUC", "Brier"],
                                    "membership_attack":"confidence-based ROC-AUC; 0.5 is chance"},
              "dp_formal_guarantee_claimed":False,
              "dp_verification_status":"EXPERIMENTAL_UNVERIFIED",
              "dp_randomness_source":"OS-backed SystemRandom; DP coins are not seeded from experiment seed",
              "dp_mechanism":"custom Poisson-sampled clipped Gaussian logistic training",
              "dp_protected_unit":"one row of the member arrays passed to DPLogReg.fit",
              "dp_adjacency":"nominal add/remove-one for fit input; upstream raw-data mapping unverified",
              "dp_accountant":"project-custom-unverified-v1",
              "dp_guarantee_scope":"none claimed; metrics and pipeline are outside a proved boundary",
              "dataset_provenance":"generated synthetic transactions with artificial noisy label; no external/provider records",
              "output_dir":str(OUT.resolve()),
              "output_files":list(output_files),
              "skipped_k_values":skipped_k_values,
              "packages":packages,
              "status":"completed_with_skips" if skipped_k_values else "completed"}
    (OUT / "ml_experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return util, ksw, mia, dp


def plot_ml(util, ksw, mia, dp, k):
    # 1: utility grouped bars (one hue per representation, fixed order)
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor="#fcfcfb")
    reps = list(util["representation"].unique()); models = list(util["model"].unique())
    w = .26
    for i, (r, c) in enumerate(zip(reps, [C_BLUE, C_ORANGE, C_AQUA])):
        v = [util[(util.representation == r) & (util.model == m)]["ROC_AUC"].iloc[0] for m in models]
        xs = np.arange(len(models)) + (i - 1) * (w + .02)
        ax.bar(xs, v, w, color=c, label=r)
        for x, val in zip(xs, v):
            ax.text(x, val + .004, f"{val:.3f}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models); ax.set_ylim(.5, max(util.ROC_AUC) + .07)
    style(ax, "Model utility on held-out synthetic test set", "ROC-AUC")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3)
    fig.tight_layout(); fig.savefig(OUT / "ml_utility.png", dpi=160); plt.close(fig)

    # 2: k-sweep as two single-axis panels (no dual axis)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), facecolor="#fcfcfb")
    xs = np.arange(len(ksw)); lab = [str(v) for v in ksw["k"]]
    axs[0].plot(xs, ksw["records_retained_pct"], color=C_BLUE, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=2)
    axs[1].plot(xs, ksw["ROC_AUC"], color=C_ORANGE, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=2)
    for a, t, yl in [(axs[0], "Records retained after k-suppression", "% of training records"),
                     (axs[1], "Model quality (Gradient Boosting)", "ROC-AUC")]:
        a.set_xticks(xs); a.set_xticklabels(lab); a.set_xlabel("k", color=MUTED); style(a, t, yl)
    fig.tight_layout(); fig.savefig(OUT / "ml_k_sweep.png", dpi=160); plt.close(fig)

    # 3: MIA bars with 0.5 reference
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor="#fcfcfb")
    labels = [c.replace(", ", ",\n", 1) for c in mia["configuration"]]
    ax.barh(range(len(mia)), mia["MIA_attack_AUC"] - .5, left=.5, color=C_BLUE, height=.55)
    for i, v in enumerate(mia["MIA_attack_AUC"]):
        ax.text(max(v, .5) + .005, i, f"{v:.3f}", va="center", fontsize=9, color=MUTED)
    ax.axvline(.5, color=INK, lw=1)
    ax.set_yticks(range(len(mia))); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(.45, max(mia.MIA_attack_AUC.max() + .06, .6))
    ax.set_title("Membership-inference attack AUC (0.5 = chance baseline)", loc="left", fontsize=11, color=INK)
    ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    fig.tight_layout(); fig.savefig(OUT / "ml_mia.png", dpi=160); plt.close(fig)

    # 4: DP trade-off, two single-axis panels sharing categorical x (noise level)
    d = dp.copy(); d["lab"] = [f"unverified ε≈{e:g}" for e in d["epsilon_experimental_unverified"]]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), facecolor="#fcfcfb")
    xs = np.arange(len(d))
    axs[0].plot(xs, d["test_ROC_AUC"], color=C_BLUE, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=2)
    axs[1].plot(xs, d["MIA_attack_AUC"], color=C_ORANGE, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=2)
    axs[1].axhline(.5, color=INK, lw=1, ls=":"); axs[1].set_ylim(.45, .75)  # same scale as the MIA chart
    for a, t, yl in [(axs[0], "Utility vs unverified accountant output", "Test ROC-AUC"),
                     (axs[1], "Attack success vs unverified accountant output", "MIA AUC (0.5 = chance)")]:
        a.set_xticks(xs); a.set_xticklabels(d["lab"], fontsize=8); style(a, t, yl)
        a.set_xlabel("higher noise (experimental estimate only)", color=MUTED, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "ml_dp_tradeoff.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    run_ml_lab()
