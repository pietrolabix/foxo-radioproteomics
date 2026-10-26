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
OUTPUT_SUMMARY_CSV = "Radiomics_RAW_Bootstrap_Stability.csv"

N_BOOTSTRAP = 5000
MAX_PLS_COMPONENTS = 4
N_SPLITS_CV = 5
CHUNK_SIZE_B = 500
Q2_SCREEN_THRESHOLD = 0.30

def robust_numeric(series):
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".").str.strip()
    return pd.to_numeric(series, errors="coerce")

# ==============================================================================
# 1. KEGG ONTOLOGY
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
# 2. DATASETS & RAW STANDARDIZATION (NO RESIDUALIZATION)
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

# SUROWE MACIERZE Y (Tylko standaryzacja, BEZ odejmowania log-Volume)
y_raw_matrix = np.zeros((n_rad, n_samples), dtype=np.float32)
vol_vec = df_analysis["reconstructed_voxel_count"].values
vol_raw_corr = {}

for idx, feat in enumerate(radiomic_features):
    vals = df_analysis[feat].fillna(df_analysis[feat].median()).values
    vol_raw_corr[feat] = float(np.corrcoef(vol_vec, vals)[0, 1])
    y_raw_matrix[idx, :] = StandardScaler().fit_transform(vals.reshape(-1, 1)).flatten()

Y_raw_tensor = torch.tensor(y_raw_matrix, dtype=torch.float32, device=device).unsqueeze(-1)

# ==============================================================================
# 3. VECTORIZED GPU PLS ENGINE
# ==============================================================================
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
            
            W_mat, P_mat = torch.cat(W_list, dim=2), torch.cat(P_list, dim=2)
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

# ==============================================================================
# 4. OBSERVED SCREENING ON RAW DATA
# ==============================================================================
print("\n" + "=" * 80)
print("[STEP 1] SCREENING OBSERVED Q2 ON RAW (UNRESIDUALIZED) RADIOMICS")
print("=" * 80)

kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
cv_splits = list(kf.split(np.arange(n_samples)))

pathways_X_tensors = []
observed_records = []

for p_idx, p_name in enumerate(tqdm(pathway_names, desc="[Observed RAW] Pathways")):
    prots = curated_pathways[p_name]["proteins"]
    X_raw = df_analysis[prots].apply(robust_numeric)
    X_imp = X_raw.fillna(X_raw.median()).values
    X_t = torch.tensor(X_imp, dtype=torch.float32, device=device).unsqueeze(0)
    pathways_X_tensors.append(X_t)
    
    X_batch = X_t.expand(n_rad, -1, -1)
    q2_vec = run_gpu_pls_cv(X_batch, Y_raw_tensor, cv_splits).cpu().numpy()
    
    for r_idx, feat in enumerate(radiomic_features):
        observed_records.append({
            "Pathway": p_name,
            "Category": curated_pathways[p_name]["Category"],
            "Radiomic_Feature": feat,
            "Observed_RAW_Q2": float(q2_vec[r_idx]),
            "Volume_Corr_r": vol_raw_corr[feat],
            "p_idx": p_idx,
            "r_idx": r_idx
        })

df_all = pd.DataFrame(observed_records)

# Sprawdzamy kluczowe modele z poprzedniego kroku oraz te z RAW Q2 >= 0.30
key_pairs = [
    ("firstorder_Energy", "Efferocytosis"),
    ("firstorder_RootMeanSquared", "Efferocytosis"),
    ("glcm_ClusterTendency", "Efferocytosis"),
    ("glrlm_RunLengthNonUniformity", "Alcoholic_liver_disease"),
    ("glrlm_RunLengthNonUniformity", "NOD_like_receptor_signaling_p"),
    ("glrlm_RunLengthNonUniformity", "Toll_like_receptor_signaling_"),
]

df_high_raw = df_all[df_all["Observed_RAW_Q2"] >= Q2_SCREEN_THRESHOLD]
df_key = df_all[df_all.apply(lambda r: any(r["Radiomic_Feature"] == kf and r["Pathway"].startswith(kp[:15]) for kf, kp in key_pairs), axis=1)]

candidates_to_boot = pd.concat([df_high_raw, df_key]).drop_duplicates().sort_values(by="Observed_RAW_Q2", ascending=False).reset_index(drop=True)
print(f"\n[+] Liczba modeli zakwalifikowanych do Bootstrapu RAW: {len(candidates_to_boot)}\n")

# ==============================================================================
# 5. VECTORIZED GPU BOOTSTRAP (B = 5,000)
# ==============================================================================
print("=" * 80)
print(f"[STEP 2] RUNNING BATCHED BOOTSTRAP (B = {N_BOOTSTRAP:,}) ON RAW DATA")
print("=" * 80)

boot_results = []

for _, cand in tqdm(candidates_to_boot.iterrows(), total=len(candidates_to_boot), desc="[RAW Bootstrap]"):
    p_idx, r_idx = int(cand["p_idx"]), int(cand["r_idx"])
    q2_raw_obs = cand["Observed_RAW_Q2"]
    
    X_target = pathways_X_tensors[p_idx].squeeze(0)  # (N, p)
    y_target = Y_raw_tensor[r_idx]                   # (N, 1)
    
    boot_indices = torch.randint(0, n_samples, (N_BOOTSTRAP, n_samples), device=device)
    X_boot_all = X_target[boot_indices]  # (B, N, p)
    y_boot_all = y_target[boot_indices]  # (B, N, 1)
    
    q2_boots = []
    for b_s in range(0, N_BOOTSTRAP, CHUNK_SIZE_B):
        b_e = min(b_s + CHUNK_SIZE_B, N_BOOTSTRAP)
        q2_chunk = run_gpu_pls_cv(X_boot_all[b_s:b_e], y_boot_all[b_s:b_e], cv_splits)
        q2_boots.append(q2_chunk.cpu().numpy())
        
    boot_arr = np.concatenate(q2_boots)
    ci_low = float(np.percentile(boot_arr, 2.5))
    ci_high = float(np.percentile(boot_arr, 97.5))
    p_boot_stable = float(np.mean(boot_arr >= Q2_SCREEN_THRESHOLD))
    
    boot_results.append({
        "Radiomic_Feature": cand["Radiomic_Feature"],
        "Pathway": cand["Pathway"],
        "Observed_RAW_Q2": q2_raw_obs,
        "Boot_Median_Q2": float(np.median(boot_arr)),
        "Boot_95CI_Low": ci_low,
        "Boot_95CI_High": ci_high,
        "P(Q2>=0.3)": p_boot_stable,
        "Volume_Corr_r": cand["Volume_Corr_r"]
    })

df_raw_final = pd.DataFrame(boot_results).sort_values(by="Observed_RAW_Q2", ascending=False).reset_index(drop=True)

# ==============================================================================
# 6. COMPARISON REPORT
# ==============================================================================
print("\n" + "=" * 120)
print(f"RAW (UNRESIDUALIZED) BOOTSTRAP STABILITY REPORT (B = {N_BOOTSTRAP:,})")
print("=" * 120)
print(f"{'Radiomic Feature':<28} | {'Pathway':<20} | {'RAW Q2':<7} | {'Boot 95% CI':<16} | {'P(Q2>=0.3)':<10} | {'r(Vol)':<6}")
print("-" * 120)

for _, r in df_raw_final.head(25).iterrows():
    ci_str = f"[{r['Boot_95CI_Low']:.2f}, {r['Boot_95CI_High']:.2f}]"
    print(f"{r['Radiomic_Feature'][:26]:<28} | {r['Pathway'][:18]:<20} | {r['Observed_RAW_Q2']:<7.3f} | {ci_str:<16} | {r['P(Q2>=0.3)']:<10.2f} | {r['Volume_Corr_r']:<6.2f}")

print("=" * 120)
df_raw_final.to_csv(OUTPUT_SUMMARY_CSV, index=False)
print(f"[+] Raw stability table exported to '{OUTPUT_SUMMARY_CSV}'\n")
