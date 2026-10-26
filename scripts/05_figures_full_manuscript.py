#!/usr/bin/env python3
"""
EHJ-CVI Figure Generation Pipeline
====================================
Produces 4 publication-quality figures from the radioproteomic analysis.

Figure 1: Screening Landscape (Q² distribution, raw + residualized)
Figure 2: Forest Plot of Significant Models (Table 1 data + bootstrap CIs)
Figure 3: VIP Profiles for Key Pathway-Radiomic Models
Figure 4: Clinical Integration (ICM vs DCM + latent score correlations)

Requirements: same data files as the analysis pipeline.
Also reads output CSVs: Radiomics_RAW_Bootstrap_Stability.csv,
                         Radiomics_FoxO_LinLog_LiJi_Candidates.csv
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from bs4 import BeautifulSoup
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = os.environ.get("FOXO_DATA_DIR", "data")
PATH_CLINICAL = os.path.join(DATA_DIR, "bdanych-11.xlsx")
PATH_KEGG = os.path.join(DATA_DIR, "HealthyvsLVAD_KEGGenrich.html")
PATH_OLINK = os.path.join(DATA_DIR, "pre_quant.csv")
PATH_PROBKI = os.path.join(DATA_DIR, "lvad_serum2_scalony_POPRAWIONY_probki.csv")

# Output CSVs from your existing pipeline (optional — used for forest plot)
PATH_RAW_CSV = "Radiomics_RAW_Bootstrap_Stability.csv"
PATH_RESID_CSV = "Radiomics_FoxO_LinLog_LiJi_Candidates.csv"

MAX_PLS_COMPONENTS = 4
N_SPLITS_CV = 5
Q2_THRESHOLD = 0.30
FIGURE_DPI = 600

# Journal style: sans-serif, Arial
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Color palette (colorblind-friendly)
C_RAW = "#2166AC"       # blue — raw/macroscopic
C_RESID = "#B2182B"     # red — residualized/microstructural
C_FOXO = "#4393C3"
C_RELAXIN = "#92C5DE"
C_EFFEROCYTOSIS = "#D6604D"
C_NOD = "#F4A582"
C_TLR = "#FDDBC7"
C_NEUTRAL = "#878787"
C_THRESHOLD = "#333333"
C_VIP_HIGH = "#B2182B"
C_VIP_LOW = "#2166AC"

def robust_numeric(series):
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".").str.strip()
    return pd.to_numeric(series, errors="coerce")


# =============================================================================
# PART 1: DATA LOADING (reused from your pipeline)
# =============================================================================
print("[*] Loading datasets...", flush=True)

# --- KEGG ---
with open(PATH_KEGG, "r", encoding="utf-8", errors="ignore") as f:
    soup = BeautifulSoup(f, "html.parser")
rows_html = soup.find("table").find_all("tr")[1:]
pathways_raw, pathway_meta = {}, {}
for row in rows_html:
    cols = [td.text.strip() for td in row.find_all("td")]
    if len(cols) >= 8:
        kegg_id, p_name = cols[0].strip(), cols[1].strip()
        safe_name = p_name.replace(" ", "_").replace("/", "_").replace("-", "_")[:30]
        pathways_raw[safe_name] = cols[7].split()
        pathway_meta[safe_name] = {"KEGG_ID": kegg_id, "Raw_Name": p_name}

def is_cardiovascular_metabolic(kegg_id, pathway_name, n_proteins):
    if n_proteins < 5:
        return False, "Too Few Proteins"
    p_name = str(pathway_name).strip().lower()
    k_id = str(kegg_id).strip().lower()
    blacklist = [
        "cancer", "carcinoma", "melanoma", "glioma", "leukemia", "sarcoma",
        "infection", "infectious", "viral", "virus", "bacterial", "parasite",
        "malaria", "leishmaniasis", "amoebiasis", "pertussis", "tuberculosis",
        "chagas", "toxoplasmosis", "hepatitis", "measles", "influenza", "herpes",
        "addiction", "cocaine", "morphine", "nicotine", "alcoholism",
        "systemic lupus", "rheumatoid", "asthma", "allograft"
    ]
    if any(bad in p_name for bad in blacklist):
        return False, "Blacklisted"
    if k_id.startswith("hsa00"):
        return True, "Metabolism"
    elif k_id.startswith("hsa04"):
        return True, "Signaling / Systems"
    elif k_id.startswith("hsa054"):
        return True, "Cardiovascular Disease"
    return False, "Other"

# --- Clinical + Radiomics ---
df_clin = pd.read_excel(PATH_CLINICAL, sheet_name="baza myocardium", header=[0, 1]).iloc[:, :95]
flat_cols = [
    f"{str(l0).strip()}_{str(l1).strip()}" if pd.notna(l0) and pd.notna(l1)
    else str(l1).strip() if pd.isna(l0) else str(l0).strip()
    for l0, l1 in df_clin.columns
]
df_clin.columns = flat_cols
df_clin = df_clin.dropna(how="all").reset_index(drop=True).set_index(df_clin.columns[0])
df_clin.index.name = "Patient"
df_clin = df_clin.dropna(axis=1, how="all")

radiomic_prefixes = ("firstorder_", "glszm_", "ngtdm_", "glcm_", "glrlm_", "shape_", "gldm_")
radiomic_features = [c for c in df_clin.columns if str(c).startswith(radiomic_prefixes)]
for feat in radiomic_features:
    df_clin[feat] = robust_numeric(df_clin[feat])
df_rad = df_clin.dropna(subset=radiomic_features, how="all")

vec_energy = df_rad["firstorder_Energy"].values
vec_rms = df_rad["firstorder_RootMeanSquared"].values
df_rad["reconstructed_voxel_count"] = np.round(vec_energy / (vec_rms ** 2))

# --- Olink ---
df_olink = pd.read_csv(PATH_OLINK, sep=";")
df_olink["Feature"] = df_olink["Assay"].astype(str) + "_" + df_olink["OlinkID"].astype(str)
df_olink_wide = df_olink.pivot_table(index="SampleID", columns="Feature", values="NPX", aggfunc="mean")
df_olink_wide.index.name = "Patient"

df_analysis = df_rad[radiomic_features + ["reconstructed_voxel_count"]].merge(
    df_olink_wide, left_index=True, right_index=True, how="inner"
)
n_samples = len(df_analysis)
print(f"[*] Matched Cohort: N = {n_samples} patients")

protein_cols = [c for c in df_analysis.columns if "_OID" in str(c)]
curated_pathways = {}
for p_name, genes in pathways_raw.items():
    matched = list(dict.fromkeys(
        [c for gene in genes for c in protein_cols if str(c).startswith(f"{gene}_OID")]
    ))
    kegg_id = pathway_meta[p_name]["KEGG_ID"]
    raw_name = pathway_meta[p_name]["Raw_Name"]
    keep, cat_label = is_cardiovascular_metabolic(kegg_id, raw_name, len(matched))
    if keep:
        curated_pathways[p_name] = {
            "proteins": matched, "KEGG_ID": kegg_id,
            "Category": cat_label, "Raw_Name": raw_name
        }
pathway_names = list(curated_pathways.keys())
n_rad = len(radiomic_features)
n_paths = len(pathway_names)
print(f"[*] Features: {n_rad} radiomic x {n_paths} pathways = {n_rad * n_paths} models")


# =============================================================================
# PART 2: PREPARE RAW & RESIDUALIZED TARGETS
# =============================================================================
print("[*] Preparing raw and residualized target matrices...", flush=True)

vol_vec = df_analysis["reconstructed_voxel_count"].values
vol_log = np.log(np.maximum(vol_vec, 1.0))
vol_log_std = StandardScaler().fit_transform(vol_log.reshape(-1, 1)).flatten()

y_raw_matrix = np.zeros((n_rad, n_samples), dtype=np.float32)
y_resid_matrix = np.zeros((n_rad, n_samples), dtype=np.float32)
vol_raw_corr = {}

for idx, feat in enumerate(radiomic_features):
    vals = df_analysis[feat].fillna(df_analysis[feat].median()).values
    vol_raw_corr[feat] = float(np.corrcoef(vol_vec, vals)[0, 1])
    y_std = StandardScaler().fit_transform(vals.reshape(-1, 1)).flatten()
    y_raw_matrix[idx, :] = y_std
    slope, intercept = np.polyfit(vol_log_std, y_std, 1)
    y_resid_matrix[idx, :] = y_std - (intercept + slope * vol_log_std)


# ... existing code (Parts 1-2 stay the same) ...

# =============================================================================
# PART 3: GPU PLS SCREENING (reusing your vectorized engine)
# =============================================================================
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[*] Accelerator: {device}", flush=True)

def run_gpu_pls_cv(X_tensor, y_tensor, splits, max_comp=MAX_PLS_COMPONENTS):
    B, N, p = X_tensor.shape
    y_mean_global = y_tensor.mean(dim=1, keepdim=True)
    ss_tot = torch.sum((y_tensor - y_mean_global) ** 2, dim=1).squeeze(-1)
    best_q2 = torch.full((B,), -float('inf'), device=device, dtype=torch.float32)
    y_preds_comps = torch.zeros((max_comp, B, N, 1), device=device, dtype=torch.float32)
    for train_idx, test_idx in splits:
        X_tr, y_tr = X_tensor[:, train_idx, :], y_tensor[:, train_idx, :]
        X_te = X_tensor[:, test_idx, :]
        mean_X = X_tr.mean(dim=1, keepdim=True)
        std_X = X_tr.std(dim=1, keepdim=True, unbiased=False)
        std_X = torch.where(std_X == 0, torch.ones_like(std_X), std_X)
        X_tr_sc = (X_tr - mean_X) / std_X
        X_te_sc = (X_te - mean_X) / std_X
        mean_y = y_tr.mean(dim=1, keepdim=True)
        y_tr_sc = y_tr - mean_y
        X_k, y_k = X_tr_sc.clone(), y_tr_sc.clone()
        W_list, P_list, q_list = [], [], []
        for comp in range(max_comp):
            w = torch.bmm(X_k.transpose(1, 2), y_k)
            norm_w = torch.norm(w, dim=1, keepdim=True)
            norm_w = torch.where(norm_w == 0, torch.ones_like(norm_w), norm_w)
            w = w / norm_w
            t = torch.bmm(X_k, w)
            tt = torch.bmm(t.transpose(1, 2), t)
            tt = torch.where(tt == 0, torch.ones_like(tt), tt)
            p_load = torch.bmm(X_k.transpose(1, 2), t) / tt
            q_load = torch.bmm(y_k.transpose(1, 2), t) / tt
            X_k = X_k - torch.bmm(t, p_load.transpose(1, 2))
            y_k = y_k - t * q_load
            W_list.append(w)
            P_list.append(p_load)
            q_list.append(q_load)
        W_mat = torch.cat(W_list, dim=2)
        P_mat = torch.cat(P_list, dim=2)
        q_vec = torch.cat(q_list, dim=1)
        PtW = torch.bmm(P_mat.transpose(1, 2), W_mat)
        PtW_inv = torch.linalg.pinv(PtW)
        beta = torch.bmm(torch.bmm(W_mat, PtW_inv), q_vec)
        y_preds_comps[comp, :, test_idx, :] = torch.bmm(X_te_sc, beta) + mean_y
    safe_ss_tot = torch.where(ss_tot == 0, torch.ones_like(ss_tot) * 1e-12, ss_tot)
    for comp in range(max_comp):
        ss_res = torch.sum((y_tensor - y_preds_comps[comp]) ** 2, dim=1).squeeze(-1)
        best_q2 = torch.maximum(best_q2, 1.0 - (ss_res / safe_ss_tot))
    return best_q2

# Tensors on GPU
Y_raw_t = torch.tensor(y_raw_matrix, dtype=torch.float32, device=device).unsqueeze(-1)
Y_resid_t = torch.tensor(y_resid_matrix, dtype=torch.float32, device=device).unsqueeze(-1)

kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
cv_splits = list(kf.split(np.arange(n_samples)))

q2_raw_matrix = np.full((n_paths, n_rad), np.nan)
q2_resid_matrix = np.full((n_paths, n_rad), np.nan)
pathway_X_tensors = []
pathway_X_matrices = {}

for p_idx, p_name in enumerate(tqdm(pathway_names, desc="[GPU Screening]")):
    prots = curated_pathways[p_name]["proteins"]
    X_raw = df_analysis[prots].apply(robust_numeric).fillna(df_analysis[prots].median()).values
    X_t = torch.tensor(X_raw, dtype=torch.float32, device=device).unsqueeze(0)
    pathway_X_tensors.append(X_t)
    pathway_X_matrices[p_name] = X_raw 

    X_batch = X_t.expand(n_rad, -1, -1)  # (n_rad, N, p) — all features in one shot

    q2_raw = run_gpu_pls_cv(X_batch, Y_raw_t, cv_splits).cpu().numpy()
    q2_resid = run_gpu_pls_cv(X_batch, Y_resid_t, cv_splits).cpu().numpy()

    q2_raw_matrix[p_idx, :] = q2_raw
    q2_resid_matrix[p_idx, :] = q2_resid

print(f"[*] Done. Raw hits (Q²≥0.30): {np.sum(q2_raw_matrix >= Q2_THRESHOLD)}")
print(f"[*] Done. Resid hits (Q²≥0.30): {np.sum(q2_resid_matrix >= Q2_THRESHOLD)}")

# ... rest of the script (Parts 4-6 + all figures) stays the same ...

# =============================================================================
# PART 4: VIP COMPUTATION FOR KEY MODELS
# =============================================================================
def compute_vip(pls_model, feature_names):
    """VIP scores from a fitted sklearn PLSRegression."""
    W = pls_model.x_weights_
    T = pls_model.x_scores_
    Q = pls_model.y_loadings_
    p, h = W.shape
    ss = np.array([(Q[0, a] ** 2) * np.dot(T[:, a], T[:, a]) for a in range(h)])
    ss_total = np.sum(ss)
    vip_scores = np.sqrt(p * np.sum(ss[np.newaxis, :] * (W ** 2), axis=1) / max(ss_total, 1e-12))
    return pd.Series(vip_scores, index=feature_names).sort_values(ascending=False)

def fit_best_pls_and_vip(X_raw, y_vec, feature_names, cv_splits, max_comp=MAX_PLS_COMPONENTS):
    """Find best n_comp via CV, refit on full data, return VIP."""
    best_q2, best_nc = -np.inf, 1
    n = len(y_vec)
    for nc in range(1, min(max_comp + 1, min(X_raw.shape[1], X_raw.shape[0]) + 1)):
        y_pred = np.zeros(n)
        for tr, te in cv_splits:
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_raw[tr])
            X_te_s = sc.transform(X_raw[te])
            my = y_vec[tr].mean()
            try:
                pls = PLSRegression(n_components=nc, scale=False, max_iter=500)
                pls.fit(X_tr_s, y_vec[tr] - my)
                y_pred[te] = pls.predict(X_te_s).flatten() + my
            except Exception:
                y_pred[te] = my
        ss_res = np.sum((y_vec - y_pred) ** 2)
        ss_tot = np.sum((y_vec - y_vec.mean()) ** 2)
        q2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        if q2 > best_q2:
            best_q2, best_nc = q2, nc
    # Refit on full data
    sc_full = StandardScaler()
    X_full_s = sc_full.fit_transform(X_raw)
    my_full = y_vec.mean()
    pls_full = PLSRegression(n_components=best_nc, scale=False, max_iter=500)
    pls_full.fit(X_full_s, y_vec - my_full)
    vip = compute_vip(pls_full, feature_names)
    return best_q2, best_nc, vip

# Define key models for VIP extraction
key_models = [
    # (pathway_search_term, radiomic_feature, target_matrix, label)
    ("foxo", "firstorder_Energy", "raw", "FoxO → Energy (Raw)"),
    ("foxo", "firstorder_TotalEnergy", "raw", "FoxO → Total Energy (Raw)"),
    ("relaxin", "firstorder_Energy", "raw", "Relaxin → Energy (Raw)"),
    ("efferocytosis", "firstorder_Energy", "resid", "Efferocytosis → Energy (Resid)"),
    ("efferocytosis", "firstorder_RootMeanSquared", "resid", "Efferocytosis → RMS (Resid)"),
    ("efferocytosis", "glcm_ClusterTendency", "resid", "Efferocytosis → Cluster Tend. (Resid)"),
    ("nod_like", "glrlm_RunLengthNonUniformity", "resid", "NOD-like → RLNU (Resid)"),
    ("toll_like", "glrlm_RunLengthNonUniformity", "resid", "Toll-like → RLNU (Resid)"),
]

print("[*] Computing VIP scores for key models...", flush=True)
vip_results = {}
for pw_search, rad_feat, target_type, label in key_models:
    # Find pathway
    pw_match = [p for p in pathway_names if pw_search in p.lower()]
    if not pw_match:
        print(f"  [!] Pathway '{pw_search}' not found, skipping")
        continue
    pw_name = pw_match[0]
    # Find radiomic feature
    feat_match = [f for f in radiomic_features if f.lower() == rad_feat.lower()]
    if not feat_match:
        print(f"  [!] Feature '{rad_feat}' not found, skipping")
        continue
    feat_name = feat_match[0]
    r_idx = radiomic_features.index(feat_name)
    # Get data
    X_p = pathway_X_matrices[pw_name]
    y_vec = y_raw_matrix[r_idx] if target_type == "raw" else y_resid_matrix[r_idx]
    # Clean protein names for display
    prots = curated_pathways[pw_name]["proteins"]
    clean_names = [p.split("_OID")[0] for p in prots]
    q2, nc, vip = fit_best_pls_and_vip(X_p, y_vec, clean_names, cv_splits)
    vip_results[label] = {"q2": q2, "n_comp": nc, "vip": vip, "pathway": pw_name}
    print(f"  [+] {label}: Q²={q2:.3f}, n_comp={nc}, top VIP={vip.index[0]} ({vip.iloc[0]:.2f})")


# =============================================================================
# PART 5: LOAD CANDIDATE CSVs (for forest plot CIs)
# =============================================================================
df_raw_cands = None
df_resid_cands = None
if os.path.exists(PATH_RAW_CSV):
    df_raw_cands = pd.read_csv(PATH_RAW_CSV)
    print(f"[*] Loaded raw candidates: {len(df_raw_cands)} rows")
if os.path.exists(PATH_RESID_CSV):
    df_resid_cands = pd.read_csv(PATH_RESID_CSV)
    print(f"[*] Loaded residualized candidates: {len(df_resid_cands)} rows")


# =============================================================================
# PART 6: LOAD CLINICAL DATA (for Figure 4)
# =============================================================================
df_probki = None
if os.path.exists(PATH_PROBKI):
    df_probki = pd.read_csv(PATH_PROBKI, sep=None, engine="python")
    df_probki = df_probki.set_index(df_probki.columns[0])
    df_probki.index = df_probki.index.astype(str).str.strip()

# =============================================================================
# HELPER: short feature labels for display
# =============================================================================
def short_radiomic_name(feat):
    """Convert 'firstorder_Energy' → 'Energy (1st)' etc."""
    mapping = {
        "firstorder_Energy": "Energy",
        "firstorder_TotalEnergy": "Total Energy",
        "firstorder_RobustMeanAbsoluteDeviation": "RMAD",
        "firstorder_InterquartileRange": "IQR",
        "firstorder_RootMeanSquared": "RMS",
        "glcm_ClusterTendency": "Cluster Tendency\n(GLCM)",
        "glrlm_RunLengthNonUniformity": "Run Length Non-Unif.\n(GLRLM)",
    }
    return mapping.get(feat, feat.split("_", 1)[-1][:20])

def short_pathway_name(pw):
    """Convert safe pathway name to readable."""
    meta = pathway_meta.get(pw, {})
    raw = meta.get("Raw_Name", pw)
    # Shorten common names
    raw = raw.replace(" signaling pathway", "").replace(" - Homo sapiens (human)", "")
    raw = raw.replace("signaling", "signal.").replace("pathway", "")
    return raw.strip()

def radiomic_class(feat):
    """Get feature class from prefix."""
    if feat.startswith("firstorder_"):
        return "First-Order"
    elif feat.startswith("glcm_"):
        return "GLCM"
    elif feat.startswith("glrlm_"):
        return "GLRLM"
    elif feat.startswith("glszm_"):
        return "GLSZM"
    elif feat.startswith("gldm_"):
        return "GLDM"
    elif feat.startswith("ngtdm_"):
        return "NGTDM"
    elif feat.startswith("shape_"):
        return "Shape"
    return "Other"


# #############################################################################
#                           FIGURE 1
#             SCREENING LANDSCAPE (Dual Histogram + Top Hits)
# #############################################################################
print("\n[*] Generating Figure 1: Screening Landscape...", flush=True)

fig1, axes1 = plt.subplots(1, 2, figsize=(17.5 / 2.54, 7.0 / 2.54), sharey=False)

for ax, q2_mat, color, title_str, pipeline_label in [
    (axes1[0], q2_raw_matrix, C_RAW, "Raw (Macroscopic)", "raw"),
    (axes1[1], q2_resid_matrix, C_RESID, "Volume-Residualized (Microstructural)", "resid"),
]:
    q2_flat = q2_mat.flatten()
    q2_valid = q2_flat[~np.isnan(q2_flat)]
    n_sig = np.sum(q2_valid >= Q2_THRESHOLD)

    ax.hist(q2_valid, bins=80, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(Q2_THRESHOLD, color=C_THRESHOLD, linestyle="--", linewidth=1.0, zorder=5)
    ax.text(Q2_THRESHOLD + 0.02, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 10,
            f"Q² ≥ {Q2_THRESHOLD}\nn = {n_sig}",
            fontsize=6.5, color=C_THRESHOLD, fontweight="bold", va="top")

    # Annotate top 3 hits
    top_indices = np.argsort(q2_valid)[-3:][::-1]
    for rank, flat_idx in enumerate(top_indices):
        # Convert flat index back to pathway/feature
        orig_idx = np.where(~np.isnan(q2_flat))[0][flat_idx]
        p_i, r_i = divmod(orig_idx, n_rad)
        if p_i < len(pathway_names) and r_i < len(radiomic_features):
            pw_short = short_pathway_name(pathway_names[p_i])
            feat_short = short_radiomic_name(radiomic_features[r_i])
            q2_val = q2_valid[flat_idx]
            if rank == 0:
                ax.annotate(
                    f"{pw_short}\n→ {feat_short}\nQ²={q2_val:.2f}",
                    xy=(q2_val, 1), fontsize=5.5,
                    xytext=(q2_val - 0.15, ax.get_ylim()[1] * 0.65 if ax.get_ylim()[1] > 0 else 8),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                    color=color, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8, lw=0.5)
                )

    ax.set_xlabel("Cross-Validated Q²", fontweight="bold")
    ax.set_title(title_str, fontweight="bold", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes1[0].set_ylabel("Number of Models", fontweight="bold")

# Panel labels
axes1[0].text(-0.15, 1.05, "A", transform=axes1[0].transAxes, fontsize=12, fontweight="bold")
axes1[1].text(-0.15, 1.05, "B", transform=axes1[1].transAxes, fontsize=12, fontweight="bold")

fig1.suptitle(
    f"Radioproteomic Screening Landscape ({n_paths} Pathways × {n_rad} Features = "
    f"{n_paths * n_rad:,} Models, N = {n_samples})",
    fontsize=9, fontweight="bold", y=1.02
)
fig1.tight_layout()
fig1.savefig("Figure1_Screening_Landscape.pdf", dpi=FIGURE_DPI)
fig1.savefig("Figure1_Screening_Landscape.png", dpi=FIGURE_DPI)
print("  [+] Saved Figure1_Screening_Landscape.pdf/png")


# #############################################################################
#                           FIGURE 2
#                 FOREST PLOT OF SIGNIFICANT MODELS
# #############################################################################
print("[*] Generating Figure 2: Forest Plot...", flush=True)

# Build Table 1 data from CSV outputs + screening results
table1_data = []

# Define expected models (from the paper Table 1)
expected_models = [
    # (pathway_search, feature, pipeline, paper_q2)
    ("foxo", "firstorder_Energy", "raw", None),
    ("foxo", "firstorder_TotalEnergy", "raw", None),
    ("foxo", "firstorder_RobustMeanAbsoluteDeviation", "raw", None),
    ("foxo", "firstorder_InterquartileRange", "raw", None),
    ("relaxin", "firstorder_Energy", "raw", None),
    ("efferocytosis", "firstorder_Energy", "resid", None),
    ("nod_like", "glrlm_RunLengthNonUniformity", "resid", None),
    ("efferocytosis", "firstorder_RootMeanSquared", "resid", None),
    ("toll_like", "glrlm_RunLengthNonUniformity", "resid", None),
    ("efferocytosis", "glcm_ClusterTendency", "resid", None),
]

for pw_search, rad_feat, pipeline, _ in expected_models:
    # Find pathway and feature
    pw_match = [p for p in pathway_names if pw_search in p.lower()]
    feat_match = [f for f in radiomic_features if f.lower() == rad_feat.lower()]
    if not pw_match or not feat_match:
        continue
    pw_name, feat_name = pw_match[0], feat_match[0]
    p_idx = pathway_names.index(pw_name)
    r_idx = radiomic_features.index(feat_name)

    q2_mat = q2_raw_matrix if pipeline == "raw" else q2_resid_matrix
    q2_obs = float(q2_mat[p_idx, r_idx])

    # Try to get bootstrap CIs from CSV
    ci_low, ci_high, p_adj = np.nan, np.nan, np.nan
    cand_df = df_raw_cands if pipeline == "raw" else df_resid_cands
    if cand_df is not None:
        mask = cand_df["Radiomic_Feature"].str.lower() == feat_name.lower()
        mask &= cand_df["Pathway"].str.lower().str.contains(pw_search)
        matched_rows = cand_df[mask]
        if len(matched_rows) > 0:
            row = matched_rows.iloc[0]
            ci_low = row.get("Boot_95CI_Low", np.nan)
            ci_high = row.get("Boot_95CI_High", np.nan)
            q2_col = "Observed_Q2" if "Observed_Q2" in row.index else "Observed_RAW_Q2"
            q2_obs = float(row[q2_col])
            p_adj = row.get("p_Li_Ji_adj", np.nan)

    # Color mapping
    if "foxo" in pw_search:
        color = C_FOXO
    elif "relaxin" in pw_search:
        color = C_RELAXIN
    elif "efferocytosis" in pw_search:
        color = C_EFFEROCYTOSIS
    elif "nod" in pw_search:
        color = C_NOD
    elif "toll" in pw_search:
        color = C_TLR
    else:
        color = C_NEUTRAL

    # Significance marker
    sig_marker = ""
    if not np.isnan(p_adj):
        if p_adj < 0.01:
            sig_marker = " **"
        elif p_adj < 0.05:
            sig_marker = " *"

    table1_data.append({
        "label": f"{short_pathway_name(pw_name)} → {short_radiomic_name(feat_name)}{sig_marker}",
        "q2": q2_obs,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pipeline": pipeline,
        "color": color,
        "pw_name": short_pathway_name(pw_name),
    })

# Sort: raw on top, then resid; within each, by Q2 descending
raw_models = [m for m in table1_data if m["pipeline"] == "raw"]
resid_models = [m for m in table1_data if m["pipeline"] == "resid"]
raw_models.sort(key=lambda x: x["q2"], reverse=True)
resid_models.sort(key=lambda x: x["q2"], reverse=True)
ordered_models = raw_models + resid_models

fig2, ax2 = plt.subplots(figsize=(12.0 / 2.54, 10.0 / 2.54))

y_positions = list(range(len(ordered_models)))[::-1]
section_gap_idx = len(raw_models)  # where the gap goes

# Adjust y positions to add gap between sections
y_pos_adjusted = []
for i, pos in enumerate(y_positions):
    if i >= section_gap_idx:
        y_pos_adjusted.append(pos - 0.8)
    else:
        y_pos_adjusted.append(pos)

for i, model in enumerate(ordered_models):
    yp = y_pos_adjusted[i]
    q2 = model["q2"]
    ci_l = model["ci_low"]
    ci_h = model["ci_high"]
    color = model["color"]

    # Point
    ax2.scatter(q2, yp, color=color, s=50, zorder=5, edgecolors="black", linewidth=0.5)
    # CI line
    if not np.isnan(ci_l) and not np.isnan(ci_h):
        ax2.plot([ci_l, ci_h], [yp, yp], color=color, linewidth=1.5, alpha=0.7, zorder=4)
        # CI caps
        cap = 0.15
        ax2.plot([ci_l, ci_l], [yp - cap, yp + cap], color=color, linewidth=1.0, zorder=4)
        ax2.plot([ci_h, ci_h], [yp - cap, yp + cap], color=color, linewidth=1.0, zorder=4)

# Labels
ax2.set_yticks(y_pos_adjusted)
ax2.set_yticklabels([m["label"] for m in ordered_models], fontsize=6.5)

# Threshold line
ax2.axvline(Q2_THRESHOLD, color=C_THRESHOLD, linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
ax2.axvline(0, color="grey", linestyle="-", linewidth=0.5, alpha=0.4, zorder=1)

# Section labels
mid_raw = np.mean(y_pos_adjusted[:section_gap_idx]) if section_gap_idx > 0 else 0
mid_resid = np.mean(y_pos_adjusted[section_gap_idx:]) if section_gap_idx < len(y_pos_adjusted) else 0
xlim = ax2.get_xlim()

ax2.text(xlim[0] - 0.02, y_pos_adjusted[0] + 0.8,
         "Raw (Macroscopic)", fontsize=7, fontweight="bold", color=C_RAW, va="bottom")
if section_gap_idx < len(y_pos_adjusted):
    ax2.text(xlim[0] - 0.02, y_pos_adjusted[section_gap_idx] + 0.8,
             "Volume-Residualized (Microstructural)", fontsize=7, fontweight="bold",
             color=C_RESID, va="bottom")

ax2.set_xlabel("Cross-Validated Q² (Bootstrap 95% CI)", fontweight="bold")
ax2.set_title(f"Significant Radioproteomic Couplings (N = {n_samples})", fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_FOXO, markersize=6, label='FoxO signaling'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_RELAXIN, markersize=6, label='Relaxin signaling'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_EFFEROCYTOSIS, markersize=6, label='Efferocytosis'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_NOD, markersize=6, label='NOD-like receptor'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_TLR, markersize=6, label='Toll-like receptor'),
    Line2D([0], [0], color=C_THRESHOLD, linestyle='--', linewidth=0.8, label=f'Q² = {Q2_THRESHOLD}'),
]
ax2.legend(handles=legend_elements, loc="lower right", fontsize=6, framealpha=0.9)

fig2.tight_layout()
fig2.savefig("Figure2_Forest_Plot.pdf", dpi=FIGURE_DPI)
fig2.savefig("Figure2_Forest_Plot.png", dpi=FIGURE_DPI)
print("  [+] Saved Figure2_Forest_Plot.pdf/png")


# #############################################################################
#                           FIGURE 3
#              VIP PROFILES FOR KEY MODELS (Lollipop Plots)
# #############################################################################
print("[*] Generating Figure 3: VIP Profiles...", flush=True)

# Select 3 key models for panels
vip_panels = [
    ("FoxO → Energy (Raw)", C_FOXO),
    ("Efferocytosis → Energy (Resid)", C_EFFEROCYTOSIS),
    ("NOD-like → RLNU (Resid)", C_NOD),
]
vip_panels_available = [(label, color) for label, color in vip_panels if label in vip_results]
n_panels = len(vip_panels_available)

if n_panels > 0:
    fig3, axes3 = plt.subplots(1, n_panels, figsize=(17.5 / 2.54, 8.0 / 2.54), sharey=False)
    if n_panels == 1:
        axes3 = [axes3]

    panel_letters = "ABCDEF"
    for idx, (label, base_color) in enumerate(vip_panels_available):
        ax = axes3[idx]
        vip_data = vip_results[label]["vip"]
        top_n = min(15, len(vip_data))
        vip_top = vip_data.head(top_n).iloc[::-1]  # reverse for bottom-to-top

        # Color gradient by VIP magnitude
        colors = []
        for v in vip_top.values:
            if v >= 1.5:
                colors.append(C_VIP_HIGH)
            elif v >= 1.0:
                colors.append(base_color)
            else:
                colors.append(C_NEUTRAL)

        y_pos = range(len(vip_top))
        ax.barh(y_pos, vip_top.values, height=0.6, color=colors, edgecolor="white", linewidth=0.3)

        # VIP = 1.0 reference line
        ax.axvline(1.0, color=C_THRESHOLD, linestyle="--", linewidth=0.7, alpha=0.6)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(vip_top.index, fontsize=6)
        ax.set_xlabel("VIP Score", fontweight="bold")
        ax.set_title(label, fontsize=7.5, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Value labels on bars
        for j, (val, yp) in enumerate(zip(vip_top.values, y_pos)):
            if val > 0.3:
                ax.text(val + 0.02, yp, f"{val:.2f}", va="center", fontsize=5.5, color="#333333")

        # Panel label
        ax.text(-0.25, 1.05, panel_letters[idx], transform=ax.transAxes,
                fontsize=12, fontweight="bold")

    fig3.suptitle("Variable Importance in Projection (VIP) — Top Protein Drivers",
                  fontsize=9, fontweight="bold", y=1.02)
    fig3.tight_layout()
    fig3.savefig("Figure3_VIP_Profiles.pdf", dpi=FIGURE_DPI)
    fig3.savefig("Figure3_VIP_Profiles.png", dpi=FIGURE_DPI)
    print("  [+] Saved Figure3_VIP_Profiles.pdf/png")
else:
    print("  [!] No VIP data available for Figure 3")


# #############################################################################
#                           FIGURE 4
#               CLINICAL INTEGRATION (ICM vs DCM + PLS2)
# #############################################################################
print("[*] Generating Figure 4: Clinical Integration...", flush=True)

# Fit PLS2: FoxO → 4 radiomic targets (same as B.py)
foxo_pw = [p for p in pathway_names if "foxo" in p.lower()]
if foxo_pw:
    foxo_name = foxo_pw[0]
    foxo_prots = curated_pathways[foxo_name]["proteins"]
    foxo_clean_names = [p.split("_OID")[0] for p in foxo_prots]

    target_feats = ["firstorder_Energy", "firstorder_TotalEnergy",
                    "firstorder_RobustMeanAbsoluteDeviation", "firstorder_InterquartileRange"]
    target_cols = []
    for tf in target_feats:
        match = [f for f in radiomic_features if f.lower() == tf.lower()]
        if match:
            target_cols.append(match[0])

    imp = SimpleImputer(strategy="median")
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    X_foxo = sc_x.fit_transform(imp.fit_transform(
        df_analysis[foxo_prots].apply(robust_numeric).values
    ))
    Y_rad = sc_y.fit_transform(imp.fit_transform(
        df_analysis[target_cols].apply(robust_numeric).values
    ))

    pls2 = PLSRegression(n_components=1, scale=False)
    pls2.fit(X_foxo, Y_rad)
    foxo_latent = pls2.x_scores_[:, 0]
    rad_latent = pls2.y_scores_[:, 0]

    if np.corrcoef(foxo_latent, rad_latent)[0, 1] < 0:
        foxo_latent = -foxo_latent

    # Try to find etiology info
    etiology_col = None
    for col in df_clin.columns:
        if any(k in str(col).lower() for k in ["etiol", "icm", "dcm", "rozpoznanie"]):
            etiology_col = col
            break

    # Also check merged clinical data
    df_clin_full = None
    if df_probki is not None:
        df_clin_full = df_probki.merge(
            df_clin, left_index=True, right_index=True, how="outer"
        )
        if etiology_col is None:
            for col in df_clin_full.columns:
                if any(k in str(col).lower() for k in ["etiol", "icm", "dcm", "rozpoznanie"]):
                    etiology_col = col
                    break

    has_etiology = False
    icm_mask = np.zeros(n_samples, dtype=bool)
    dcm_mask = np.zeros(n_samples, dtype=bool)
    patients = list(df_analysis.index)

    if etiology_col is not None and df_clin_full is not None:
        for i, pt in enumerate(patients):
            if pt in df_clin_full.index:
                val = str(df_clin_full.loc[pt, etiology_col]).upper()
                if "ICM" in val or "KORONAR" in val or val == "1":
                    icm_mask[i] = True
                elif "DCM" in val or "DILAT" in val or val == "0":
                    dcm_mask[i] = True
        has_etiology = icm_mask.sum() > 3 and dcm_mask.sum() > 3

    fig4, axes4 = plt.subplots(1, 2, figsize=(17.5 / 2.54, 7.5 / 2.54))

    # Panel A: Scatter of FoxO latent vs Radiomic composite
    ax = axes4[0]
    if has_etiology:
        ax.scatter(foxo_latent[icm_mask], rad_latent[icm_mask],
                   c=C_RAW, s=30, alpha=0.8, edgecolors="black", linewidth=0.4,
                   label=f"ICM (n={icm_mask.sum()})", zorder=5)
        ax.scatter(foxo_latent[dcm_mask], rad_latent[dcm_mask],
                   c=C_RESID, s=30, alpha=0.8, edgecolors="black", linewidth=0.4,
                   label=f"DCM (n={dcm_mask.sum()})", zorder=5)
        other = ~icm_mask & ~dcm_mask
        if other.sum() > 0:
            ax.scatter(foxo_latent[other], rad_latent[other],
                       c=C_NEUTRAL, s=20, alpha=0.5, edgecolors="black", linewidth=0.3,
                       label=f"Other (n={other.sum()})", zorder=4)
        ax.legend(fontsize=6, loc="upper left")
    else:
        ax.scatter(foxo_latent, rad_latent, c=C_FOXO, s=30, alpha=0.8,
                   edgecolors="black", linewidth=0.4, zorder=5)

    # Fit line
    z = np.polyfit(foxo_latent, rad_latent, 1)
    x_line = np.linspace(foxo_latent.min(), foxo_latent.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="black", linewidth=1.0,
            linestyle="-", alpha=0.5, zorder=3)
    rho_lr, p_lr = spearmanr(foxo_latent, rad_latent)
    ax.text(0.95, 0.05, f"ρ = {rho_lr:.2f}\np = {p_lr:.1e}",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xlabel("FoxO Proteomic Latent Score (PLS2)", fontweight="bold")
    ax.set_ylabel("Radiomic Composite Score (PLS2)", fontweight="bold")
    ax.set_title("FoxO–Radiomic Latent Coupling", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: ICM vs DCM boxplots
    ax = axes4[1]
    if has_etiology:
        box_data = [foxo_latent[icm_mask], foxo_latent[dcm_mask]]
        bp = ax.boxplot(box_data, positions=[1, 2], widths=0.5, patch_artist=True,
                        showfliers=True, flierprops=dict(marker='o', markersize=3))
        bp["boxes"][0].set_facecolor(C_RAW)
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(C_RESID)
        bp["boxes"][1].set_alpha(0.6)
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color("black")
                line.set_linewidth(0.8)

        # Overlay individual points
        for pos, data, color in [(1, box_data[0], C_RAW), (2, box_data[1], C_RESID)]:
            jitter = np.random.normal(0, 0.04, size=len(data))
            ax.scatter(np.full_like(data, pos) + jitter, data,
                       c=color, s=15, alpha=0.6, edgecolors="black", linewidth=0.3, zorder=5)

        stat, pval = mannwhitneyu(box_data[0], box_data[1], alternative="two-sided")
        ax.text(1.5, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] != 0 else 2,
                f"p = {pval:.3f}", ha="center", fontsize=7, fontweight="bold")

        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"ICM\n(n={icm_mask.sum()})", f"DCM\n(n={dcm_mask.sum()})"])
        ax.set_ylabel("FoxO Proteomic Latent Score", fontweight="bold")
        ax.set_title("Etiology Independence", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Etiology data\nnot available", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="grey")
        ax.set_title("Etiology Independence", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel labels
    axes4[0].text(-0.15, 1.05, "A", transform=axes4[0].transAxes, fontsize=12, fontweight="bold")
    axes4[1].text(-0.15, 1.05, "B", transform=axes4[1].transAxes, fontsize=12, fontweight="bold")

    fig4.tight_layout()
    fig4.savefig("Figure4_Clinical_Integration.pdf", dpi=FIGURE_DPI)
    fig4.savefig("Figure4_Clinical_Integration.png", dpi=FIGURE_DPI)
    print("  [+] Saved Figure4_Clinical_Integration.pdf/png")
else:
    print("  [!] FoxO pathway not found — skipping Figure 4")


# #############################################################################
#                     SAVE SCREENING MATRICES FOR REFERENCE
# #############################################################################
# Save full Q² matrices as CSV for reproducibility
df_q2_raw = pd.DataFrame(q2_raw_matrix, index=pathway_names, columns=radiomic_features)
df_q2_resid = pd.DataFrame(q2_resid_matrix, index=pathway_names, columns=radiomic_features)
df_q2_raw.to_csv("screening_q2_raw_full.csv")
df_q2_resid.to_csv("screening_q2_residualized_full.csv")
print(f"\n[*] Saved full screening matrices to CSV")


# #############################################################################
#                         SUMMARY
# #############################################################################
print("\n" + "=" * 70)
print("FIGURE GENERATION COMPLETE")
print("=" * 70)
print(f"  Figure 1: Screening Landscape     → Figure1_Screening_Landscape.pdf/png")
print(f"  Figure 2: Forest Plot (Table 1)   → Figure2_Forest_Plot.pdf/png")
print(f"  Figure 3: VIP Profiles            → Figure3_VIP_Profiles.pdf/png")
print(f"  Figure 4: Clinical Integration    → Figure4_Clinical_Integration.pdf/png")
print(f"  Screening matrices                → screening_q2_*.csv")
print("=" * 70)
print("\nNOTE: For the Graphical Abstract and Study Design Figure (Figure 1 in")
print("the manuscript), use PowerPoint, BioRender, or Adobe Illustrator.")
print("The journal may redraw your graphics at no cost upon acceptance.")
