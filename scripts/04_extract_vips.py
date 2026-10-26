import os
import warnings
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from openpyxl import load_workbook

warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("FOXO_DATA_DIR", "data")
PATH_CLINICAL = os.path.join(DATA_DIR, "bdanych-11.xlsx")
PATH_KEGG = os.path.join(DATA_DIR, "HealthyvsLVAD_KEGGenrich.html")
PATH_OLINK = os.path.join(DATA_DIR, "pre_quant.csv")
OUTPUT_EXCEL = "Supplementary_Table_S1_VIPs.xlsx"

def robust_numeric(series):
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".").str.strip()
    return pd.to_numeric(series, errors="coerce")

def calculate_vip(pls_model, X, y):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, a = w.shape
    ss_y = np.sum(t**2, axis=0) * (q**2).sum(axis=0)
    total_ss_y = np.sum(ss_y)
    vip = np.zeros(p)
    for i in range(p):
        weight_effects = np.sum(ss_y * (w[i, :] / np.linalg.norm(w, axis=0))**2)
        vip[i] = np.sqrt(p * weight_effects / total_ss_y)
    return vip

# 1. Dane
df_clin = pd.read_excel(PATH_CLINICAL, sheet_name="baza myocardium", header=[0, 1]).iloc[:, :95]
flat_cols = [
    f"{str(l0).strip()}_{str(l1).strip()}" if pd.notna(l0) and pd.notna(l1)
    else str(l1).strip() if pd.isna(l0) else str(l0).strip()
    for l0, l1 in df_clin.columns
]
df_clin.columns = flat_cols
df_clin = df_clin.dropna(how="all").reset_index(drop=True).set_index(df_clin.columns[0])
df_clin.index.name = "Patient"

for col in ["firstorder_Energy", "firstorder_RootMeanSquared", "glrlm_RunLengthNonUniformity"]:
    df_clin[col] = robust_numeric(df_clin[col])

df_rad = df_clin.dropna(subset=["glrlm_RunLengthNonUniformity"])
vol_vec = np.round(df_rad["firstorder_Energy"].values / (df_rad["firstorder_RootMeanSquared"].values ** 2))
df_rad["reconstructed_voxel_count"] = vol_vec

df_olink = pd.read_csv(PATH_OLINK, sep=";")
df_olink["Feature"] = df_olink["Assay"].astype(str) + "_" + df_olink["OlinkID"].astype(str)
df_olink_wide = df_olink.pivot_table(index="SampleID", columns="Feature", values="NPX", aggfunc="mean")
df_olink_wide.index.name = "Patient"

df_analysis = df_rad[["glrlm_RunLengthNonUniformity", "reconstructed_voxel_count"]].merge(
    df_olink_wide, left_index=True, right_index=True, how="inner"
)

# 2. Parsowanie KEGG - wyszukiwanie odporne na formatowanie (lowercase, usunięte spacje i myślniki)
with open(PATH_KEGG, "r", encoding="utf-8", errors="ignore") as f:
    soup = BeautifulSoup(f, "html.parser")
    rows = soup.find("table").find_all("tr")[1:]
    pathways = {}
    for row in rows:
        cols = [td.text.strip() for td in row.find_all("td")]
        if len(cols) >= 8:
            p_name = cols[1].strip()
            pathways[p_name] = cols[7].split()

protein_cols = [c for c in df_analysis.columns if "_OID" in str(c)]

def get_pathway_proteins_robust(term):
    t_clean = term.lower().replace("-", "").replace("_", "").replace(" ", "")
    for name, genes in pathways.items():
        n_clean = name.lower().replace("-", "").replace("_", "").replace(" ", "")
        if t_clean in n_clean:
            matched = list(dict.fromkeys([c for g in genes for c in protein_cols if str(c).startswith(f"{g}_OID")]))
            return name, matched
    return None, []

# 3. Modele rezydualne dla GLRLM
vol_log = np.log(np.maximum(df_analysis["reconstructed_voxel_count"].values, 1.0))
vol_log_std = StandardScaler().fit_transform(vol_log.reshape(-1, 1)).flatten()

y_raw = df_analysis["glrlm_RunLengthNonUniformity"].fillna(df_analysis["glrlm_RunLengthNonUniformity"].median()).values
y_std = StandardScaler().fit_transform(y_raw.reshape(-1, 1)).flatten()
s, i = np.polyfit(vol_log_std, y_std, 1)
y_resid = y_std - (i + s * vol_log_std)

MISSING_MODELS = [
    {"term": "nodlike", "sheet": "NOD_like_GLRLM_Resid", "label": "NOD-like receptor signaling"},
    {"term": "tolllike", "sheet": "Toll_like_GLRLM_Resid", "label": "Toll-like receptor signaling"}
]

print("=" * 80)
print("NAPRAWIONE WYCIĄGANIE VIP DLA NOD-LIKE I TOLL-LIKE (GLRLM RESIDUAL):")
print("=" * 80)

for m in MISSING_MODELS:
    p_name, prots = get_pathway_proteins_robust(m["term"])
    if not prots:
        print(f"[-] Błąd: nadal nie znaleziono białek dla {m['term']}")
        continue
    
    print(f"\n[+] Dopasowano szlak: '{p_name}' ({len(prots)} białek)")
    X_mat = df_analysis[prots].apply(robust_numeric).fillna(df_analysis[prots].median()).values
    X_sc = StandardScaler().fit_transform(X_mat)
    
    pls = PLSRegression(n_components=2)
    pls.fit(X_sc, y_resid)
    
    vips = calculate_vip(pls, X_sc, y_resid)
    betas = pls.coef_.flatten()
    
    df_res = pd.DataFrame({
        "Protein": [c.split("_")[0] for c in prots],
        "OlinkID": [c.split("_")[1] for c in prots],
        "VIP": vips,
        "Beta_Direction": betas
    }).sort_values(by="VIP", ascending=False).reset_index(drop=True)
    
    print(f"Top 5 białek dla {m['label']}:")
    print(df_res.head(5)[["Protein", "VIP", "Beta_Direction"]].to_string(index=False))
    
    # Dopisanie arkusza do istniejącego pliku Excel
    if os.path.exists(OUTPUT_EXCEL):
        with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df_res.to_excel(writer, sheet_name=m["sheet"][:31], index=False)

print("\n[+] Zaktualizowano arkusze w pliku: 'Supplementary_Table_S1_VIPs.xlsx'")
