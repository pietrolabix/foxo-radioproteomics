#!/usr/bin/env python3
"""
Additional analyses for the Circulation Research letter.

A. Pathway specificity: nested Q2 of each robust model versus random protein sets
   of the same size drawn from all Olink proteins.
B. Acquisition sensitivity: targets additionally residualized on physical voxel
   volume (TotalEnergy / Energy, mm3) and physical mask volume (mL), which capture
   reconstruction / scanner differences.
C. Myocardial tissue: LV apical-core MS proteome (dgc_ijms loaders) versus the
   radiomic targets and versus plasma levels of the same proteins.
D. Clinical: cross-validated plasma efferocytosis score versus clinical variables.

Run from the repository root; needs the dgc_ijms loaders (data.py) on DGC_DIR.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import KFold

N_RANDOM = int(os.environ.get("N_RANDOM", 1000))
N_REP_RANDOM = 5
N_REP = 50

sys.argv = [sys.argv[0], "0"]
_src = open("scripts/07_revalidation_nested_cv.py", encoding="utf-8").read().split("rng = np.random.default_rng(SEED)")[0]
exec(_src)

sys.path.insert(0, os.environ.get("DGC_DIR", os.path.expanduser("~/dgc_ijms")))
import data as dgc  # noqa: E402

fi = features.index
ids = [dgc._norm_id(i) for i in df.index.astype(str)]
all_prots = [c for c in df.columns if "_OID" in str(c)]
X_all = df[all_prots].apply(ns["robust_numeric"])
X_all = X_all.fillna(X_all.median()).values

ROBUST = [
    ("efferocytosis", "firstorder_Energy", "resid", "Efferocytosis -> Energy (resid)"),
    ("foxo", "firstorder_TotalEnergy", "raw", "FoxO -> Total Energy (raw)"),
    ("relaxin", "firstorder_Energy", "raw", "Relaxin -> Energy (raw)"),
    ("nod_like", "glrlm_RunLengthNonUniformity", "resid", "NOD-like -> RLNU (resid)"),
    ("toll_like", "glrlm_RunLengthNonUniformity", "resid", "Toll-like -> RLNU (resid)"),
]


def target(feat, branch):
    return (y_raw if branch == "raw" else y_resid)[fi(feat)].astype(float)


def median_nested(X, y, n_rep):
    return float(np.median([nested_q2(X, y, SEED + r) for r in range(n_rep)]))


import torch  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def _pls_all_comps(Xtr, ytr, Xte, max_comp=MAX_COMP):
    """Batched PLS1. Xtr (B, n, p), ytr (B, n), Xte (B, m, p) -> predictions (C, B, m)."""
    mx = Xtr.mean(1, keepdim=True)
    sx = Xtr.std(1, unbiased=False, keepdim=True)
    sx = torch.where(sx == 0, torch.ones_like(sx), sx)
    Xk, Xt = (Xtr - mx) / sx, (Xte - mx) / sx
    my = ytr.mean(1, keepdim=True)
    yk = (ytr - my).unsqueeze(-1)
    W, P, Q, preds = [], [], [], []
    for _ in range(max_comp):
        w = Xk.transpose(1, 2) @ yk
        w = w / torch.clamp(w.norm(dim=1, keepdim=True), min=1e-12)
        t = Xk @ w
        tt = torch.clamp(t.transpose(1, 2) @ t, min=1e-12)
        pl = Xk.transpose(1, 2) @ t / tt
        ql = yk.transpose(1, 2) @ t / tt
        Xk = Xk - t @ pl.transpose(1, 2)
        yk = yk - t * ql
        W.append(w); P.append(pl); Q.append(ql)
        Wm, Pm, Qm = torch.cat(W, 2), torch.cat(P, 2), torch.cat(Q, 1)
        beta = Wm @ torch.linalg.pinv(Pm.transpose(1, 2) @ Wm) @ Qm
        preds.append((Xt @ beta).squeeze(-1) + my)
    return torch.stack(preds)


def _q2(Y, pred):
    return 1 - ((Y - pred) ** 2).sum(-1) / ((Y - Y.mean(-1, keepdim=True)) ** 2).sum(-1)


def nested_q2_batched_X(Xb, y, seed):
    """Nested CV Q2 for a batch of predictor sets Xb (B, n, p) and one target y (n,)."""
    B, n, _ = Xb.shape
    Y = y.expand(B, -1)
    pred = torch.zeros_like(Y)
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(np.arange(n))):
        inner = list(KFold(5, shuffle=True, random_state=seed + 1000 + k).split(tr))
        ip = torch.zeros((MAX_COMP, B, len(tr)), device=DEV)
        for itr, ite in inner:
            ip[:, :, ite] = _pls_all_comps(Xb[:, tr[itr]], Y[:, tr[itr]], Xb[:, tr[ite]])
        best = _q2(Y[:, tr].unsqueeze(0), ip).argmax(0)
        op = _pls_all_comps(Xb[:, tr], Y[:, tr], Xb[:, te])
        pred[:, te] = op.gather(0, best.view(1, -1, 1).expand(1, -1, len(te))).squeeze(0)
    return _q2(Y, pred)


def random_null(size, y_np, n_sets, batch=250):
    Xt_all = torch.tensor(X_all, dtype=torch.float32, device=DEV)
    y = torch.tensor(y_np, dtype=torch.float32, device=DEV)
    g = np.random.default_rng(SEED + size)
    vals = []
    for b in range(0, n_sets, batch):
        nb = min(batch, n_sets - b)
        idx = torch.tensor(np.stack([g.choice(X_all.shape[1], size, replace=False) for _ in range(nb)]), device=DEV)
        Xb = Xt_all[:, idx].permute(1, 0, 2)  # (nb, n, size)
        reps = torch.stack([nested_q2_batched_X(Xb, y, SEED + r) for r in range(N_REP_RANDOM)])
        vals.append(reps.median(0).values.cpu().numpy())
    return np.concatenate(vals)


out = {}

# ---------------- A. Pathway specificity ----------------
print("\n=== A. Pathway specificity vs random protein sets ===", flush=True)
rows = []
for term, feat, branch, label in ROBUST[:3]:
        pw, X = pathway_X(term)
        obs = median_nested(X, target(feat, branch), N_REP_RANDOM)
        null = random_null(X.shape[1], target(feat, branch), N_RANDOM)
        rows.append({"model": label, "set_size": X.shape[1], "nested_Q2": round(obs, 3),
                     "random_median_Q2": round(float(np.median(null)), 3),
                     "random_95th": round(float(np.percentile(null, 95)), 3),
                     "percentile_vs_random": round(100 * float(np.mean(null < obs)), 1),
                     "p_random": round((1 + np.sum(null >= obs)) / (1 + N_RANDOM), 4)})
        print(rows[-1], flush=True)
out["A_specificity"] = pd.DataFrame(rows)

# ---------------- B. Acquisition sensitivity ----------------
print("\n=== B. Adjustment for voxel volume and physical mask volume ===", flush=True)
R = dgc.radiomics_myocardium()
cov = R.reindex(ids)[["voxel_vol_mm3", "mask_vol_ml"]]
print(f"covariates available for {cov.notna().all(axis=1).sum()}/{len(ids)} patients; "
      f"voxel volume range {cov.voxel_vol_mm3.min():.3f}-{cov.voxel_vol_mm3.max():.3f} mm3", flush=True)
Z = np.c_[np.ones(len(ids)), np.log(cov.voxel_vol_mm3.values), np.log(cov.mask_vol_ml.values)]
ok = np.isfinite(Z).all(axis=1)
rows = []
for term, feat, branch, label in ROBUST:
    pw, X = pathway_X(term)
    y = target(feat, branch)[ok]
    Zk = Z[ok]
    y_adj = y - Zk @ np.linalg.lstsq(Zk, y, rcond=None)[0]
    q = np.array([nested_q2(X[ok], y_adj, SEED + r) for r in range(N_REP)])
    rows.append({"model": label, "n": int(ok.sum()), "nested_Q2_adjusted": round(float(np.median(q)), 3),
                 "IQR": f"{np.percentile(q, 25):.2f} to {np.percentile(q, 75):.2f}",
                 "share_pos": round(float(np.mean(q > 0)), 2)})
    print(rows[-1], flush=True)
out["B_acquisition"] = pd.DataFrame(rows)

# ---------------- C. Myocardial tissue ----------------
print("\n=== C. Myocardial tissue proteome ===", flush=True)
H = dgc.heart_proteome()
P = dgc.olink()
pw_eff, _ = pathway_X("efferocytosis")
eff_genes = [c.split("_OID")[0] for c in curated[pw_eff]["proteins"]]
focus = ["MERTK", "MAPK12", "PECAM1", "ADGRB1", "HAVCR1", "ADAM9"]
paired = [p for p in ids if p in H.columns]
print(f"patients with radiomics and tissue proteome: {len(paired)}", flush=True)
pos = [ids.index(p) for p in paired]
y_eff = target("firstorder_Energy", "resid")[pos]
y_te = target("firstorder_TotalEnergy", "raw")[pos]
rows = []
for g in sorted(set(eff_genes + focus + ["EGFR", "PTEN", "FOXO3", "PRKAB1", "COL4A1", "MMP9"])):
    if g not in H.index:
        continue
    t = H.loc[g, paired].astype(float).values
    r1, p1 = spearmanr(t, y_eff, nan_policy="omit")
    r2, p2 = spearmanr(t, y_te, nan_policy="omit")
    pl = P.reindex(paired)[g].values if g in P.columns else np.full(len(paired), np.nan)
    m = np.isfinite(pl) & np.isfinite(t)
    r3, p3 = spearmanr(t[m], pl[m]) if m.sum() >= 6 else (np.nan, np.nan)
    rows.append({"gene": g, "in_efferocytosis_set": g in eff_genes, "n": int(np.isfinite(t).sum()),
                 "rho_tissue_vs_residEnergy": round(r1, 2), "p1": round(p1, 3),
                 "rho_tissue_vs_rawTotalEnergy": round(r2, 2), "p2": round(p2, 3),
                 "rho_tissue_vs_plasma": round(r3, 2) if np.isfinite(r3) else np.nan,
                 "p3": round(p3, 3) if np.isfinite(p3) else np.nan})
C = pd.DataFrame(rows)
print(C.to_string(index=False), flush=True)
eff_in_tissue = [g for g in eff_genes if g in H.index]
Ht = H.loc[eff_in_tissue, paired].astype(float)
score = ((Ht.T - Ht.T.mean()) / Ht.T.std()).mean(axis=1).values
r, p = spearmanr(score, y_eff, nan_policy="omit")
print(f"tissue efferocytosis score ({len(eff_in_tissue)} genes) vs resid Energy: rho={r:.2f}, p={p:.3f}", flush=True)
out["C_tissue"] = C

# ---------------- D. Clinical ----------------
print("\n=== D. Clinical variables vs cross-validated efferocytosis score ===", flush=True)
_, X_eff = pathway_X("efferocytosis")
y = target("firstorder_Energy", "resid")
preds = []
for r in range(N_REP):
    pred = np.zeros_like(y)
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED + r).split(X_eff)):
        nc = best_ncomp_inner(X_eff[tr], y[tr], SEED + 1000 + k)
        pred[te] = cv_predict(X_eff, y, [(tr, te)], nc)[te]
    preds.append(pred)
score_cv = np.mean(preds, axis=0)
clin = dgc.clinical().reindex(ids)
rows = []
for c in clin.columns:
    v = pd.to_numeric(clin[c], errors="coerce").values
    m = np.isfinite(v)
    if m.sum() < 20 or np.nanstd(v) == 0:
        continue
    for name, s in [("cv_efferocytosis_score", score_cv), ("resid_Energy", y)]:
        r, p = spearmanr(s[m], v[m])
        rows.append({"variable": c, "against": name, "n": int(m.sum()), "rho": round(r, 2), "p": p})
D = pd.DataFrame(rows)
for name in D["against"].unique():
    sel = D["against"] == name
    D.loc[sel, "q_fdr"] = multipletests(D.loc[sel, "p"], method="fdr_bh")[1]
D = D.sort_values("p")
print(D.head(20).to_string(index=False), flush=True)
out["D_clinical"] = D

with pd.ExcelWriter("results/extra_analyses_letter.xlsx") as w:
    for k, v in out.items():
        v.to_excel(w, sheet_name=k, index=False)
print("\n[+] Saved extra_analyses_letter.xlsx")
