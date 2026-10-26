#!/usr/bin/env python3
"""
GPU (MPS/CUDA) version of the nested permutation test from reanalysis_letter.py.

The statistic is the same: median Q2 over N_STAT_REPEATS repeated 5-fold CVs,
with the number of PLS components chosen by an inner 5-fold CV on the training
folds only. Each permutation re-runs the whole nested pipeline; permutations
are processed in batches with a vectorized NIPALS-free PLS1 (as in
run_gpu_pls_cv from generate_figures.py).
"""
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
N_STAT_REPEATS = 5
MAX_COMP = 4
SEED = 2026
BATCH = 2000

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float32

# --- Parts 1-2 of generate_figures.py, unchanged ---
src = open("scripts/05_figures_full_manuscript.py", encoding="utf-8").read()
src = src.split("# ... existing code (Parts 1-2 stay the same) ...")[0]
ns = {"__name__": "pipeline_data"}
exec(compile(src, "05_figures_full_manuscript.py[1-2]", "exec"), ns)
pathway_names, curated, df = ns["pathway_names"], ns["curated_pathways"], ns["df_analysis"]
features, y_raw, y_resid = ns["radiomic_features"], ns["y_raw_matrix"], ns["y_resid_matrix"]

TABLE1 = [
    ("foxo", "firstorder_Energy", "raw"),
    ("foxo", "firstorder_TotalEnergy", "raw"),
    ("foxo", "firstorder_RobustMeanAbsoluteDeviation", "raw"),
    ("foxo", "firstorder_InterquartileRange", "raw"),
    ("relaxin", "firstorder_Energy", "raw"),
    ("efferocytosis", "firstorder_Energy", "resid"),
    ("nod_like", "glrlm_RunLengthNonUniformity", "resid"),
    ("efferocytosis", "firstorder_RootMeanSquared", "resid"),
    ("toll_like", "glrlm_RunLengthNonUniformity", "resid"),
    ("efferocytosis", "glcm_ClusterTendency", "resid"),
]


def pls_predict_all_comps(X_tr, y_tr, X_te, max_comp=MAX_COMP):
    """X_tr (n_tr, p) shared, y_tr (B, n_tr) batched; returns (max_comp, B, n_te)."""
    mean_X = X_tr.mean(0, keepdim=True)
    std_X = X_tr.std(0, unbiased=False, keepdim=True)
    std_X = torch.where(std_X == 0, torch.ones_like(std_X), std_X)
    Xs = ((X_tr - mean_X) / std_X).expand(y_tr.shape[0], -1, -1).clone()   # (B, n, p)
    Xt = (X_te - mean_X) / std_X                                           # (n_te, p)
    my = y_tr.mean(1, keepdim=True)
    yk = (y_tr - my).unsqueeze(-1)                                         # (B, n, 1)
    W, P, Q, preds = [], [], [], []
    for _ in range(max_comp):
        w = Xs.transpose(1, 2) @ yk
        w = w / torch.clamp(w.norm(dim=1, keepdim=True), min=1e-12)
        t = Xs @ w
        tt = torch.clamp(t.transpose(1, 2) @ t, min=1e-12)
        p_load = Xs.transpose(1, 2) @ t / tt
        q_load = yk.transpose(1, 2) @ t / tt
        Xs = Xs - t @ p_load.transpose(1, 2)
        yk = yk - t * q_load
        W.append(w); P.append(p_load); Q.append(q_load)
        Wm, Pm, Qm = torch.cat(W, 2), torch.cat(P, 2), torch.cat(Q, 1)
        beta = Wm @ torch.linalg.pinv(Pm.transpose(1, 2) @ Wm) @ Qm        # (B, p, 1)
        preds.append((Xt.unsqueeze(0) @ beta).squeeze(-1) + my)           # (B, n_te)
    return torch.stack(preds)


def cv_pred(X, Y, splits):
    """X (n, p), Y (B, n) -> predictions (max_comp, B, n)."""
    out = torch.zeros((MAX_COMP,) + Y.shape, device=device, dtype=DT)
    for tr, te in splits:
        out[:, :, te] = pls_predict_all_comps(X[tr], Y[:, tr], X[te])
    return out


def q2_rows(Y, pred):
    ss_tot = ((Y - Y.mean(-1, keepdim=True)) ** 2).sum(-1)
    return 1 - ((Y - pred) ** 2).sum(-1) / ss_tot


def nested_q2(X, Y, seed):
    """Batched nested CV Q2 for each row of Y (B, n)."""
    n = Y.shape[1]
    pred = torch.zeros_like(Y)
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(np.arange(n))):
        inner = list(KFold(5, shuffle=True, random_state=seed + 1000 + k).split(tr))
        inner_pred = cv_pred(X[tr], Y[:, tr], inner)
        best = q2_rows(Y[:, tr].unsqueeze(0), inner_pred).argmax(0)       # (B,)
        outer = pls_predict_all_comps(X[tr], Y[:, tr], X[te])              # (C, B, n_te)
        pred[:, te] = outer.gather(0, best.view(1, -1, 1).expand(1, -1, len(te))).squeeze(0)
    return q2_rows(Y, pred)


def statistic(X, Y):
    return torch.stack([nested_q2(X, Y, SEED + r) for r in range(N_STAT_REPEATS)]).median(0).values


g = torch.Generator().manual_seed(SEED)
rows = []
for term, feat, branch in TABLE1:
    pw = [p for p in pathway_names if term in p.lower()][0]
    prots = curated[pw]["proteins"]
    X = torch.tensor(df[prots].apply(ns["robust_numeric"]).fillna(df[prots].median()).values,
                     dtype=DT, device=device)
    y = torch.tensor((y_raw if branch == "raw" else y_resid)[features.index(feat)],
                     dtype=DT, device=device)
    obs = statistic(X, y.unsqueeze(0)).item()
    null = []
    for b in range(0, N_PERM, BATCH):
        nb = min(BATCH, N_PERM - b)
        idx = torch.stack([torch.randperm(len(y), generator=g) for _ in range(nb)]).to(device)
        null.append(statistic(X, y[idx]).cpu())
    null = torch.cat(null).numpy()
    p = (1 + np.sum(null >= obs)) / (1 + N_PERM)
    rows.append({"pathway": pw, "feature": feat, "branch": branch,
                 "stat_nested_median_Q2": round(obs, 3), "p_perm_nested": p})
    print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_csv("results/reanalysis_letter_perm_gpu.csv", index=False)
print(out.to_string(index=False))
