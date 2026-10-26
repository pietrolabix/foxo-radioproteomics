import os
import warnings
import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG & ACCELERATOR
# ==============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[*] Active Compute Accelerator: {device}", flush=True)

DATA_DIR = os.environ.get("FOXO_DATA_DIR", "data")
PATH_CLINICAL = os.path.join(DATA_DIR, "bdanych-11.xlsx")
PATH_KEGG = os.path.join(DATA_DIR, "HealthyvsLVAD_KEGGenrich.html")
PATH_OLINK = os.path.join(DATA_DIR, "pre_quant.csv")
OUTPUT_SUMMARY_CSV = "Radiomics_FoxO_LinLog_LiJi_Candidates.csv"

N_PERM_CANDIDATES = 100000
N_BOOTSTRAP = 5000
MAX_PLS_COMPONENTS = 4
N_SPLITS_CV = 5
CHUNK_SIZE_B = 500
Q2_THRESHOLD = 0.30

def robust_numeric(series):
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".").str.strip()
    return pd.to_numeric(series, errors="coerce")

# ==============================================================================
# 1. LI & JI (2005) EIGENVALUE DECOMPOSITION
# ==============================================================================
def calculate_li_ji_meff(corr_matrix: np.ndarray) -> float:
    corr_clean = np.nan_to_num(corr_matrix, nan=0.0)
    eigenvals = np.linalg.eigvalsh(corr_clean)
    eigenvals = np.maximum(eigenvals, 0.0)
    return float(np.sum((eigenvals >= 1.0).astype(float) + (eigenvals - np.floor(eigenvals))))

# ==============================================================================
# 2. KEGG ONTOLOGY
# ==============================================================================
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

print("[*] Parsing KEGG ontology...", flush=True)
with open(PATH_KEGG, "r", encoding="utf-8", errors="ignore") as f:
    soup = BeautifulSoup(f, "html.parser")
    rows = soup.find("table").find_all("tr")[1:]
    pathways_raw, pathway_meta = {}, {}
    for row in rows:
        cols = [td.text.strip() for td in row.find_all("td")]
        if len(cols) >= 8:
            kegg_id, p_name = cols[0].strip(), cols[1].strip()
            safe_name = p_name.replace(" ", "_").replace("/", "_").replace("-", "_")[:30]
            pathways_raw[safe_name] = cols[7].split()
            pathway_meta[safe_name] = {"KEGG_ID": kegg_id, "Raw_Name": p_name}

# ==============================================================================
# 3. DATASETS & VOLUME RECONSTRUCTION
# ==============================================================================
print("[*] Loading Datasets...", flush=True)
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
    matched = list(dict.fromkeys([c for gene in genes for c in protein_cols if str(c).startswith(f"{gene}_OID")]))
    kegg_id, raw_name = pathway_meta[p_name]["KEGG_ID"], pathway_meta[p_name]["Raw_Name"]
    keep, cat_label = is_cardiovascular_metabolic(kegg_id, raw_name, len(matched))
    if keep:
        curated_pathways[p_name] = {"proteins": matched, "KEGG_ID": kegg_id, "Category": cat_label, "Raw_Name": raw_name}

pathway_names = list(curated_pathways.keys())
n_rad, n_paths = len(radiomic_features), len(pathway_names)

# ==============================================================================
# 4. STRICT LIN-LOG RESIDUALS & LI-JI M_EFF CALCULATION
# ==============================================================================
print("\n" + "=" * 80)
print("[STEP 1] LIN-LOG RESIDUALIZATION & LI-JI M_EFF RESOLUTION")
print("=" * 80)

vol_vec = df_analysis["reconstructed_voxel_count"].values
vol_log = np.log(np.maximum(vol_vec, 1.0))
vol_log_std = StandardScaler().fit_transform(vol_log.reshape(-1, 1)).flatten()

residuals_lin_log = np.zeros((n_rad, n_samples), dtype=np.float32)
vol_raw_corr = {}

for idx, feat in enumerate(radiomic_features):
    y_raw = df_analysis[feat].fillna(df_analysis[feat].median()).values
    vol_raw_corr[feat] = float(np.corrcoef(vol_vec, y_raw)[0, 1])
    y_std = StandardScaler().fit_transform(y_raw.reshape(-1, 1)).flatten()
    s, i = np.polyfit(vol_log_std, y_std, 1)
    residuals_lin_log[idx, :] = y_std - (i + s * vol_log_std)

# Li & Ji dla Radiomiki (Y)
corr_rad = np.corrcoef(residuals_lin_log)
m_eff_rad = calculate_li_ji_meff(corr_rad)

# Li & Ji dla Ścieżek (PCA 1st PC per pathway)
pathway_pcs = []
for p_name in pathway_names:
    prots = curated_pathways[p_name]["proteins"]
    X_mat = df_analysis[prots].apply(robust_numeric).fillna(df_analysis[prots].median()).values
    u, _, _ = np.linalg.svd(StandardScaler().fit_transform(X_mat), full_matrices=False)
    pathway_pcs.append(u[:, 0])

corr_paths = np.corrcoef(np.array(pathway_pcs))
m_eff_paths = calculate_li_ji_meff(corr_paths)

m_eff_total = m_eff_rad * m_eff_paths
alpha_sidak_li_ji = 1.0 - (1.0 - 0.05) ** (1.0 / m_eff_total)

print(f"[*] Radiomic Features: {n_rad}  --> M_eff: {m_eff_rad:.2f}")
print(f"[*] KEGG Pathways:     {n_paths} --> M_eff: {m_eff_paths:.2f}")
print(f"[*] Total Li & Ji M_eff Space: {m_eff_total:.2f} (zamiast nominalnych {n_rad * n_paths})")
print(f"[*] Li & Ji Sidak Alpha Threshold (0.05): {alpha_sidak_li_ji:.6e}")

# ==============================================================================
# 5. VECTORIZED GPU PLS ENGINE
# ==============================================================================
def run_gpu_pls_cv(X_tensor, y_tensor, splits, max_comp=MAX_PLS_COMPONENTS):
    """
    X_tensor: (B, N, p)
    y_tensor: (B, N, 1)
    """
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
            
            W_mat, P_mat = torch.cat(W_list, dim=2), torch.cat(P_list, dim=2)
            q_vec = torch.cat(q_list, dim=1)
            
            PtW = torch.bmm(P_mat.transpose(1, 2), W_mat)
            PtW_inv = torch.linalg.pinv(PtW)
            beta = torch.bmm(torch.bmm(W_mat, PtW_inv), q_vec)
            
            y_preds_comps[comp, :, test_idx, :] = torch.bmm(X_te_sc, beta) + mean_y

    safe_ss_tot = torch.where(ss_tot == 0, torch.ones_like(ss_tot) * 1e-12, ss_tot)
    for comp in range(max_comp):
        y_pred = y_preds_comps[comp]
        ss_res = torch.sum((y_tensor - y_pred) ** 2, dim=1).squeeze(-1)
        q2_comp = 1.0 - (ss_res / safe_ss_tot)
        best_q2 = torch.maximum(best_q2, q2_comp)
        
    return best_q2

# ==============================================================================
# 6. OBSERVED SCREENING
# ==============================================================================
print("=" * 80)
print("[STEP 2] SCREENING OBSERVED Q2 (LIN-LOG)")
print("=" * 80)

kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
cv_splits = list(kf.split(np.arange(n_samples)))

Y_residuals_t = torch.tensor(residuals_lin_log, dtype=torch.float32, device=device).unsqueeze(-1)
pathways_X_tensors = []
observed_records = []

for p_idx, p_name in enumerate(tqdm(pathway_names, desc="[Observed] Pathways")):
    prots = curated_pathways[p_name]["proteins"]
    X_raw = df_analysis[prots].apply(robust_numeric)
    X_imp = X_raw.fillna(X_raw.median()).values
    X_t = torch.tensor(X_imp, dtype=torch.float32, device=device).unsqueeze(0)
    pathways_X_tensors.append(X_t)
    
    X_batch = X_t.expand(n_rad, -1, -1)
    q2_vec = run_gpu_pls_cv(X_batch, Y_residuals_t, cv_splits, max_comp=MAX_PLS_COMPONENTS).cpu().numpy()
    
    for r_idx, feat in enumerate(radiomic_features):
        observed_records.append({
            "Pathway": p_name,
            "Category": curated_pathways[p_name]["Category"],
            "Radiomic_Feature": feat,
            "Observed_Q2": float(q2_vec[r_idx]),
            "Volume_Corr_r": vol_raw_corr[feat],
            "p_idx": p_idx,
            "r_idx": r_idx
        })

df_all = pd.DataFrame(observed_records)
candidates = df_all[df_all["Observed_Q2"] >= Q2_THRESHOLD].sort_values(by="Observed_Q2", ascending=False).reset_index(drop=True)

if len(candidates) == 0:
    print(f"[-] Brak modeli z Q2 >= {Q2_THRESHOLD}. Pobieram Top 5.")
    candidates = df_all.sort_values(by="Observed_Q2", ascending=False).head(5).reset_index(drop=True)

print(f"\n[+] Modele do przetestowania (Q2 >= {Q2_THRESHOLD}): {len(candidates)}\n")

# ==============================================================================
# 7. VECTORIZED CANDIDATE TESTING: PERMUTATIONS + BOOTSTRAP (GPU BATCHED)
# ==============================================================================
print("=" * 80)
print(f"[STEP 3] BATCHED PERMUTATIONS (B={N_PERM_CANDIDATES:,}) & BATCHED BOOTSTRAP (B={N_BOOTSTRAP:,})")
print("=" * 80)

candidate_results = []

for _, cand in tqdm(candidates.iterrows(), total=len(candidates), desc="[Evaluating Candidates]"):
    p_idx, r_idx = int(cand["p_idx"]), int(cand["r_idx"])
    q2_obs = cand["Observed_Q2"]
    
    X_target = pathways_X_tensors[p_idx].squeeze(0)  # (N, p)
    y_target = Y_residuals_t[r_idx]                  # (N, 1)
    
    # --------------------------------------------------------------------------
    # A. VECTORIZED PERMUTATIONS (B = 2,000 in GPU chunks)
    # --------------------------------------------------------------------------
    perm_idx = torch.stack([torch.randperm(n_samples, device=device) for _ in range(N_PERM_CANDIDATES)])
    y_exp = y_target.squeeze(-1).unsqueeze(0).expand(N_PERM_CANDIDATES, -1)
    y_perm_all = torch.gather(y_exp, 1, perm_idx).unsqueeze(-1)  # (B, N, 1)
    X_perm_all = X_target.unsqueeze(0).expand(N_PERM_CANDIDATES, -1, -1)  # (B, N, p)
    
    q2_perms = []
    for b_s in range(0, N_PERM_CANDIDATES, CHUNK_SIZE_B):
        b_e = min(b_s + CHUNK_SIZE_B, N_PERM_CANDIDATES)
        q2_chunk = run_gpu_pls_cv(X_perm_all[b_s:b_e], y_perm_all[b_s:b_e], cv_splits)
        q2_perms.append(q2_chunk.cpu().numpy())
    
    null_q2 = np.concatenate(q2_perms)
    p_perm_raw = float((1.0 + np.sum(null_q2 >= q2_obs)) / (1.0 + N_PERM_CANDIDATES))
    p_li_ji_adj = float(min(1.0, max(0.0, 1.0 - (1.0 - p_perm_raw) ** m_eff_total)))
    
    # --------------------------------------------------------------------------
    # B. VECTORIZED BOOTSTRAP (B = 1,000 in GPU chunks)
    # --------------------------------------------------------------------------
    boot_indices = torch.randint(0, n_samples, (N_BOOTSTRAP, n_samples), device=device)
    X_boot_all = X_target[boot_indices]  # (B, N, p) - czysta wektoryzacja bez pętli
    y_boot_all = y_target[boot_indices]  # (B, N, 1)
    
    q2_boots = []
    for b_s in range(0, N_BOOTSTRAP, CHUNK_SIZE_B):
        b_e = min(b_s + CHUNK_SIZE_B, N_BOOTSTRAP)
        q2_chunk = run_gpu_pls_cv(X_boot_all[b_s:b_e], y_boot_all[b_s:b_e], cv_splits)
        q2_boots.append(q2_chunk.cpu().numpy())
        
    boot_arr = np.concatenate(q2_boots)
    ci_low = float(np.percentile(boot_arr, 2.5))
    ci_high = float(np.percentile(boot_arr, 97.5))
    p_boot_stable = float(np.mean(boot_arr >= Q2_THRESHOLD))
    
    candidate_results.append({
        "Radiomic_Feature": cand["Radiomic_Feature"],
        "Pathway": cand["Pathway"],
        "Category": cand["Category"],
        "Observed_Q2": q2_obs,
        "p_perm_raw": p_perm_raw,
        "p_Li_Ji_adj": p_li_ji_adj,
        "Boot_Median_Q2": float(np.median(boot_arr)),
        "Boot_95CI_Low": ci_low,
        "Boot_95CI_High": ci_high,
        "P(Boot_Q2>=0.3)": p_boot_stable,
        "Volume_Corr_r": cand["Volume_Corr_r"]
    })

df_final = pd.DataFrame(candidate_results).sort_values(by="Observed_Q2", ascending=False).reset_index(drop=True)

# ==============================================================================
# 8. FINAL AUDIT TABLE
# ==============================================================================
print("\n" + "=" * 125)
print("FINAL CANDIDATE AUDIT: TARGETED PERMUTATIONS + LI & JI CORRECTION + BOOTSTRAP STABILITY")
print("=" * 125)
print(f"Total Effective Tests (M_eff): {m_eff_total:.1f} (Rad: {m_eff_rad:.1f} x Path: {m_eff_paths:.1f})")
print(f"Li & Ji Sidak Threshold (alpha=0.05): {alpha_sidak_li_ji:.6e}")
print("-" * 125)
print(f"{'Radiomic Feature':<26} | {'Pathway':<18} | {'Obs Q2':<6} | {'p_perm (raw)':<12} | {'p_Li_Ji (adj)':<13} | {'Boot 95% CI':<15} | {'P(Q2>=0.3)':<9}")
print("-" * 125)

for _, r in df_final.iterrows():
    ci_str = f"[{r['Boot_95CI_Low']:.2f}, {r['Boot_95CI_High']:.2f}]"
    sig = "***" if r["p_Li_Ji_adj"] < 0.001 else "**" if r["p_Li_Ji_adj"] < 0.01 else "*" if r["p_Li_Ji_adj"] < 0.05 else ""
    print(f"{r['Radiomic_Feature'][:24]:<26} | {r['Pathway'][:16]:<18} | {r['Observed_Q2']:<6.3f} | {r['p_perm_raw']:<12.4e} | {r['p_Li_Ji_adj']:<13.4e}{sig:<3} | {ci_str:<15} | {r['P(Boot_Q2>=0.3)']:<9.2f}")

print("=" * 125)
df_final.to_csv(OUTPUT_SUMMARY_CSV, index=False)
print(f"[+] Candidate results exported to '{OUTPUT_SUMMARY_CSV}'\n")
