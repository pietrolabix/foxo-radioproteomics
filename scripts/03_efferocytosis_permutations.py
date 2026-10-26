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
# CONFIG
# ==============================================================================
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"[*] Compute Accelerator: {device}", flush=True)

DATA_DIR = os.environ.get("FOXO_DATA_DIR", "data")
PATH_CLINICAL = os.path.join(DATA_DIR, "bdanych-11.xlsx")
PATH_KEGG = os.path.join(DATA_DIR, "HealthyvsLVAD_KEGGenrich.html")
PATH_OLINK = os.path.join(DATA_DIR, "pre_quant.csv")

TARGET_FEATURE = "firstorder_Energy"
TARGET_PATHWAY_NAME = "Efferocytosis"
M_EFF_TOTAL = 513.0  # Li & Ji z poprzedniego kroku

N_PERMUTATIONS = 1_000_000
CHUNK_SIZE_B = 10000
MAX_PLS_COMPONENTS = 4
N_SPLITS_CV = 5

def robust_numeric(series):
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".").str.strip()
    return pd.to_numeric(series, errors="coerce")

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================
df_clin = pd.read_excel(PATH_CLINICAL, sheet_name="baza myocardium", header=[0, 1]).iloc[:, :95]
flat_cols = [
    f"{str(l0).strip()}_{str(l1).strip()}" if pd.notna(l0) and pd.notna(l1)
    else str(l1).strip() if pd.isna(l0) else str(l0).strip()
    for l0, l1 in df_clin.columns
]
df_clin.columns = flat_cols
df_clin = df_clin.dropna(how="all").reset_index(drop=True).set_index(df_clin.columns[0])
df_clin.index.name = "Patient"

for col in ["firstorder_Energy", "firstorder_RootMeanSquared"]:
    df_clin[col] = robust_numeric(df_clin[col])

df_rad = df_clin.dropna(subset=[TARGET_FEATURE])
vol_vec = np.round(df_rad["firstorder_Energy"].values / (df_rad["firstorder_RootMeanSquared"].values ** 2))
df_rad["reconstructed_voxel_count"] = vol_vec

df_olink = pd.read_csv(PATH_OLINK, sep=";")
df_olink["Feature"] = df_olink["Assay"].astype(str) + "_" + df_olink["OlinkID"].astype(str)
df_olink_wide = df_olink.pivot_table(index="SampleID", columns="Feature", values="NPX", aggfunc="mean")
df_olink_wide.index.name = "Patient"

df_analysis = df_rad[[TARGET_FEATURE, "reconstructed_voxel_count"]].merge(
    df_olink_wide, left_index=True, right_index=True, how="inner"
)
n_samples = len(df_analysis)

# Pobranie genów dla Efferocytosis
with open(PATH_KEGG, "r", encoding="utf-8", errors="ignore") as f:
    soup = BeautifulSoup(f, "html.parser")
    rows = soup.find("table").find_all("tr")[1:]
    target_genes = []
    for row in rows:
        cols = [td.text.strip() for td in row.find_all("td")]
        if len(cols) >= 8 and TARGET_PATHWAY_NAME.lower() in cols[1].strip().lower():
            target_genes = cols[7].split()
            break

protein_cols = [c for c in df_analysis.columns if "_OID" in str(c)]
matched_proteins = list(dict.fromkeys([c for gene in target_genes for c in protein_cols if str(c).startswith(f"{gene}_OID")]))

print(f"[*] Matched Cohort: N = {n_samples}")
print(f"[*] Target Pathway: {TARGET_PATHWAY_NAME} ({len(matched_proteins)} proteins matched)")

# ==============================================================================
# 2. STRICT LIN-LOG RESIDUAL
# ==============================================================================
vol_log = np.log(np.maximum(df_analysis["reconstructed_voxel_count"].values, 1.0))
vol_log_std = StandardScaler().fit_transform(vol_log.reshape(-1, 1)).flatten()

y_raw = df_analysis[TARGET_FEATURE].fillna(df_analysis[TARGET_FEATURE].median()).values
y_std = StandardScaler().fit_transform(y_raw.reshape(-1, 1)).flatten()
s, i = np.polyfit(vol_log_std, y_std, 1)
y_residual = y_std - (i + s * vol_log_std)

X_mat = df_analysis[matched_proteins].apply(robust_numeric).fillna(df_analysis[matched_proteins].median()).values

X_target = torch.tensor(X_mat, dtype=torch.float32, device=device)       # (N, p)
y_target = torch.tensor(y_residual, dtype=torch.float32, device=device).unsqueeze(-1) # (N, 1)

# ==============================================================================
# 3. VECTORIZED GPU PLS ENGINE
# ==============================================================================
kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
cv_splits = list(kf.split(np.arange(n_samples)))

def run_gpu_pls_cv_batch(X_tensor, y_tensor):
    B, N, p = X_tensor.shape
    y_mean_global = y_tensor.mean(dim=1, keepdim=True)
    ss_tot = torch.sum((y_tensor - y_mean_global) ** 2, dim=1).squeeze(-1)
    
    best_q2 = torch.full((B,), -float('inf'), device=device, dtype=torch.float32)
    y_preds = torch.zeros((MAX_PLS_COMPONENTS, B, N, 1), device=device, dtype=torch.float32)

    for train_idx, test_idx in cv_splits:
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
        
        for comp in range(MAX_PLS_COMPONENTS):
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
            
            y_preds[comp, :, test_idx, :] = torch.bmm(X_te_sc, beta) + mean_y

    safe_ss_tot = torch.where(ss_tot == 0, torch.ones_like(ss_tot) * 1e-12, ss_tot)
    for comp in range(MAX_PLS_COMPONENTS):
        ss_res = torch.sum((y_tensor - y_preds[comp]) ** 2, dim=1).squeeze(-1)
        best_q2 = torch.maximum(best_q2, 1.0 - (ss_res / safe_ss_tot))
        
    return best_q2

# Obliczenie obserwowanego Q2
q2_obs = run_gpu_pls_cv_batch(X_target.unsqueeze(0), y_target.unsqueeze(0)).item()
print(f"[*] Observed Q2 for {TARGET_FEATURE} ~ {TARGET_PATHWAY_NAME}: {q2_obs:.4f}")

# ==============================================================================
# 4. RUN 1,000,000 PERMUTATIONS IN BATCHES
# ==============================================================================
print(f"[*] Starting {N_PERMUTATIONS:,} permutations on GPU in batches of {CHUNK_SIZE_B}...", flush=True)

hits = 0
X_chunk = X_target.unsqueeze(0).expand(CHUNK_SIZE_B, -1, -1)
y_raw_tensor = y_target.squeeze(-1)

for b in tqdm(range(0, N_PERMUTATIONS, CHUNK_SIZE_B), desc="[Permuting 1M]"):
    perm_indices = torch.stack([torch.randperm(n_samples, device=device) for _ in range(CHUNK_SIZE_B)])
    y_perm_chunk = torch.gather(y_raw_tensor.expand(CHUNK_SIZE_B, -1), 1, perm_indices).unsqueeze(-1)
    
    q2_perm = run_gpu_pls_cv_batch(X_chunk, y_perm_chunk)
    hits += int(torch.sum(q2_perm >= q2_obs).item())

p_perm_exact = (1.0 + hits) / (1.0 + N_PERMUTATIONS)
p_li_ji_adj = 1.0 - (1.0 - p_perm_exact) ** M_EFF_TOTAL

print("\n" + "=" * 80)
print(f"RESULTS FOR {TARGET_FEATURE} vs {TARGET_PATHWAY_NAME} (B = {N_PERMUTATIONS:,}):")
print("=" * 80)
print(f"Observed Q2:              {q2_obs:.4f}")
print(f"Null Exceedances (hits):  {hits}")
print(f"Exact Permutation p-val:  {p_perm_exact:.6e}")
print(f"Li & Ji Adjusted p-val:   {p_li_ji_adj:.6e}")
print("=" * 80)
