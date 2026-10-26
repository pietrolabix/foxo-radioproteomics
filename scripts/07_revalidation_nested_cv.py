#!/usr/bin/env python3
"""
Robustness re-analysis of the Table 1 models (EHJCI-S-26-02344) before the
Circulation Research Research Letter.

Data loading is taken verbatim from generate_figures.py (Parts 1-2), so the
cohort, pathway sets and raw/residualized targets are identical to the paper.

Checks per model:
  A. Original estimate: single 5-fold split (random_state=42), n_comp chosen
     as the best CV Q2 on the same folds (as in the manuscript).
  B. Repeated 5-fold CV (N_REPEATS different splits), n_comp chosen in an
     inner CV on the training folds only (nested). Median Q2, IQR, share > 0.
  C. Repeated 5-fold CV with a fixed single PLS component (no tuning).
  D. Permutation p-value for the nested statistic of B (whole pipeline
     re-run on every permutation).
  E. Spearman rho of the target with log voxel count (raw models only).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

N_REPEATS = 50
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N_PERM_REPEATS = 5
MAX_COMP = 4
SEED = 2026

# --- Parts 1-2 of generate_figures.py, unchanged ---
src = open("scripts/05_figures_full_manuscript.py", encoding="utf-8").read()
src = src.split("# ... existing code (Parts 1-2 stay the same) ...")[0]
ns = {"__name__": "pipeline_data"}
exec(compile(src, "05_figures_full_manuscript.py[1-2]", "exec"), ns)

pathway_names = ns["pathway_names"]
curated = ns["curated_pathways"]
df = ns["df_analysis"]
features = ns["radiomic_features"]
y_raw, y_resid = ns["y_raw_matrix"], ns["y_resid_matrix"]
vol_log = ns["vol_log"]

TABLE1 = [
    ("foxo", "firstorder_Energy", "raw", 0.481),
    ("foxo", "firstorder_TotalEnergy", "raw", 0.413),
    ("foxo", "firstorder_RobustMeanAbsoluteDeviation", "raw", 0.345),
    ("foxo", "firstorder_InterquartileRange", "raw", 0.342),
    ("relaxin", "firstorder_Energy", "raw", 0.320),
    ("efferocytosis", "firstorder_Energy", "resid", 0.514),
    ("nod_like", "glrlm_RunLengthNonUniformity", "resid", 0.389),
    ("efferocytosis", "firstorder_RootMeanSquared", "resid", 0.385),
    ("toll_like", "glrlm_RunLengthNonUniformity", "resid", 0.372),
    ("efferocytosis", "glcm_ClusterTendency", "resid", 0.304),
]


def pathway_X(term):
    name = [p for p in pathway_names if term in p.lower()][0]
    prots = curated[name]["proteins"]
    X = df[prots].apply(ns["robust_numeric"]).fillna(df[prots].median()).values
    return name, X


def cv_predict(X, y, splits, n_comp):
    pred = np.zeros_like(y)
    for tr, te in splits:
        sc = StandardScaler().fit(X[tr])
        my = y[tr].mean()
        pls = PLSRegression(n_components=n_comp, scale=False, max_iter=500)
        pls.fit(sc.transform(X[tr]), y[tr] - my)
        pred[te] = pls.predict(sc.transform(X[te])).ravel() + my
    return pred


def q2(y, pred):
    return 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def best_ncomp_inner(X, y, seed):
    splits = list(KFold(5, shuffle=True, random_state=seed).split(X))
    scores = [q2(y, cv_predict(X, y, splits, nc)) for nc in range(1, MAX_COMP + 1)]
    return int(np.argmax(scores)) + 1


def nested_q2(X, y, seed):
    pred = np.zeros_like(y)
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(X)):
        nc = best_ncomp_inner(X[tr], y[tr], seed + 1000 + k)
        sc = StandardScaler().fit(X[tr])
        my = y[tr].mean()
        pls = PLSRegression(n_components=nc, scale=False, max_iter=500)
        pls.fit(sc.transform(X[tr]), y[tr] - my)
        pred[te] = pls.predict(sc.transform(X[te])).ravel() + my
    return q2(y, pred)


def original_q2(X, y):
    splits = list(KFold(5, shuffle=True, random_state=42).split(X))
    return max(q2(y, cv_predict(X, y, splits, nc)) for nc in range(1, MAX_COMP + 1))


rng = np.random.default_rng(SEED)
rows = []
for term, feat, branch, q2_paper in TABLE1:
    pw, X = pathway_X(term)
    y = (y_raw if branch == "raw" else y_resid)[features.index(feat)].astype(float)

    q_orig = original_q2(X, y)
    q_nested = np.array([nested_q2(X, y, SEED + r) for r in range(N_REPEATS)])
    q_fixed1 = np.array([
        q2(y, cv_predict(X, y, list(KFold(5, shuffle=True, random_state=SEED + r).split(X)), 1))
        for r in range(N_REPEATS)
    ])

    stat_obs = np.median(q_nested[:N_PERM_REPEATS])
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        yp = rng.permutation(y)
        null[b] = np.median([nested_q2(X, yp, SEED + r) for r in range(N_PERM_REPEATS)])
    p_perm = (1 + np.sum(null >= stat_obs)) / (1 + N_PERM)

    rho_vol = spearmanr(vol_log, (y_raw[features.index(feat)]))[0]
    rows.append({
        "pathway": pw, "feature": feat, "branch": branch, "n_proteins": X.shape[1],
        "Q2_paper": q2_paper, "Q2_original_split": round(q_orig, 3),
        "Q2_nested_median": round(np.median(q_nested), 3),
        "Q2_nested_IQR": f"{np.percentile(q_nested, 25):.3f}..{np.percentile(q_nested, 75):.3f}",
        "share_nested_pos": round(np.mean(q_nested > 0), 2),
        "Q2_1comp_median": round(np.median(q_fixed1), 3),
        "p_perm_nested": round(p_perm, 4),
        "rho_rawfeature_vs_logvoxels": round(rho_vol, 2),
    })
    print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_csv("results/reanalysis_letter_results.csv", index=False)
print(out.to_string(index=False))
