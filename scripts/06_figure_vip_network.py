#!/usr/bin/env python3
"""
Pathway × Radiomic Model → Protein VIP Network
================================================
Left:  10 specific models from Table 1 (pathway × radiomic feature)
Right: 16 key proteins (union of all top-3 drivers)
Edges: EVERY VIP score — including low ones showing specificity.
       Width + color encode VIP magnitude.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib import cm
from bs4 import BeautifulSoup
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
FIGURE_DPI = 600

# =============================================================================
# DATA LOADING (reused from your pipeline)
# =============================================================================
DATA_DIR = os.environ.get("FOXO_DATA_DIR", "data")
PATH_CLINICAL = os.path.join(DATA_DIR, "bdanych-11.xlsx")
PATH_KEGG = os.path.join(DATA_DIR, "HealthyvsLVAD_KEGGenrich.html")
PATH_OLINK = os.path.join(DATA_DIR, "pre_quant.csv")
MAX_PLS_COMPONENTS = 4
N_SPLITS_CV = 5

def robust_numeric(series):
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".").str.strip()
    return pd.to_numeric(series, errors="coerce")

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
    if n_proteins < 5: return False
    p_name = str(pathway_name).strip().lower()
    k_id = str(kegg_id).strip().lower()
    blacklist = ["cancer","carcinoma","melanoma","glioma","leukemia","sarcoma",
                 "infection","infectious","viral","virus","bacterial","parasite",
                 "malaria","leishmaniasis","amoebiasis","pertussis","tuberculosis",
                 "chagas","toxoplasmosis","hepatitis","measles","influenza","herpes",
                 "addiction","cocaine","morphine","nicotine","alcoholism",
                 "systemic lupus","rheumatoid","asthma","allograft"]
    if any(bad in p_name for bad in blacklist): return False
    return k_id.startswith("hsa00") or k_id.startswith("hsa04") or k_id.startswith("hsa054")

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

radiomic_prefixes = ("firstorder_","glszm_","ngtdm_","glcm_","glrlm_","shape_","gldm_")
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
    df_olink_wide, left_index=True, right_index=True, how="inner")
n_samples = len(df_analysis)
protein_cols = [c for c in df_analysis.columns if "_OID" in str(c)]

curated_pathways = {}
for p_name, genes in pathways_raw.items():
    matched = list(dict.fromkeys(
        [c for gene in genes for c in protein_cols if str(c).startswith(f"{gene}_OID")]))
    kegg_id = pathway_meta[p_name]["KEGG_ID"]
    raw_name = pathway_meta[p_name]["Raw_Name"]
    if is_cardiovascular_metabolic(kegg_id, raw_name, len(matched)):
        curated_pathways[p_name] = {"proteins": matched, "Raw_Name": raw_name}

print(f"[*] Cohort: N={n_samples}, Pathways: {len(curated_pathways)}")

# --- Prepare raw + residualized targets ---
vol_vec = df_analysis["reconstructed_voxel_count"].values
vol_log_std = StandardScaler().fit_transform(
    np.log(np.maximum(vol_vec, 1.0)).reshape(-1, 1)).flatten()

def get_target_vector(feat_name, pipeline):
    vals = df_analysis[feat_name].fillna(df_analysis[feat_name].median()).values
    y_std = StandardScaler().fit_transform(vals.reshape(-1, 1)).flatten()
    if pipeline == "resid":
        s, i = np.polyfit(vol_log_std, y_std, 1)
        return y_std - (i + s * vol_log_std)
    return y_std

# =============================================================================
# VIP COMPUTATION
# =============================================================================
def compute_full_vip(X_raw, y_vec, protein_names, max_comp=MAX_PLS_COMPONENTS):
    """Fit PLS with optimal n_comp (via CV), refit on full data, return ALL VIP scores."""
    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
    cv_splits = list(kf.split(np.arange(len(y_vec))))
    n = len(y_vec)
    best_q2, best_nc = -np.inf, 1

    for nc in range(1, min(max_comp + 1, min(X_raw.shape[1], n) + 1)):
        y_pred = np.zeros(n)
        for tr, te in cv_splits:
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_raw[tr])
            Xte = sc.transform(X_raw[te])
            my = y_vec[tr].mean()
            try:
                pls = PLSRegression(n_components=nc, scale=False, max_iter=500)
                pls.fit(Xtr, y_vec[tr] - my)
                y_pred[te] = pls.predict(Xte).flatten() + my
            except:
                y_pred[te] = my
        ss_res = np.sum((y_vec - y_pred)**2)
        ss_tot = np.sum((y_vec - y_vec.mean())**2)
        q2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        if q2 > best_q2:
            best_q2, best_nc = q2, nc

    # Refit on full data
    sc = StandardScaler()
    X_full = sc.fit_transform(X_raw)
    my = y_vec.mean()
    pls = PLSRegression(n_components=best_nc, scale=False, max_iter=500)
    pls.fit(X_full, y_vec - my)

    # VIP formula
    W = pls.x_weights_
    T = pls.x_scores_
    Q = pls.y_loadings_
    p, h = W.shape
    ss = np.array([(Q[0, a]**2) * np.dot(T[:, a], T[:, a]) for a in range(h)])
    ss_total = max(np.sum(ss), 1e-12)
    vip_scores = np.sqrt(p * np.sum(ss[np.newaxis, :] * (W**2), axis=1) / ss_total)

    return dict(zip(protein_names, vip_scores)), best_q2, best_nc


# --- Define the 10 models from Table 1 ---
models_spec = [
    ("foxo",          "firstorder_Energy",                      "raw",   "FoxO → Energy"),
    ("foxo",          "firstorder_TotalEnergy",                 "raw",   "FoxO → Total Energy"),
    ("foxo",          "firstorder_RobustMeanAbsoluteDeviation", "raw",   "FoxO → RMAD"),
    ("foxo",          "firstorder_InterquartileRange",          "raw",   "FoxO → IQR"),
    ("relaxin",       "firstorder_Energy",                      "raw",   "Relaxin → Energy"),
    ("efferocytosis", "firstorder_Energy",                      "resid", "Efferocytosis → Energy"),
    ("efferocytosis", "firstorder_RootMeanSquared",             "resid", "Efferocytosis → RMS"),
    ("efferocytosis", "glcm_ClusterTendency",                   "resid", "Efferocytosis → Clust. Tend."),
    ("nod_like",      "glrlm_RunLengthNonUniformity",           "resid", "NOD-like → RLNU"),
    ("toll_like",     "glrlm_RunLengthNonUniformity",           "resid", "Toll-like → RLNU"),
]

# 16 key proteins from Table 1 top-3 columns
key_proteins_gene = [
    "EGFR", "MAPK12", "PTEN", "FOXO3", "PRKAB1", "PLK2",
    "COL4A1", "MMP9",
    "MERTK", "PECAM1", "ADGRB1", "ADAM9", "HAVCR1",
    "CXCL8", "TLR4", "IL1B",
]

# Compute ALL VIP scores for each model
print("[*] Computing VIP for 10 models...", flush=True)
all_edges = []  # (model_label, protein_gene, vip_score)
model_info = []  # metadata per model

for pw_search, rad_feat, pipeline, label in models_spec:
    pw_match = [p for p in curated_pathways if pw_search in p.lower()]
    feat_match = [f for f in radiomic_features if f.lower() == rad_feat.lower()]
    if not pw_match or not feat_match:
        print(f"  [!] Skipping {label}: not found")
        continue
    pw_name = pw_match[0]
    feat_name = feat_match[0]

    prots = curated_pathways[pw_name]["proteins"]
    gene_names = [p.split("_OID")[0] for p in prots]

    X_raw = df_analysis[prots].apply(robust_numeric).fillna(
        df_analysis[prots].median()).values
    y_vec = get_target_vector(feat_name, pipeline)

    vip_dict, q2, nc = compute_full_vip(X_raw, y_vec, gene_names)
    model_info.append({"label": label, "pipeline": pipeline, "q2": q2, "nc": nc})
    print(f"  [+] {label}: Q²={q2:.3f}, nc={nc}, n_prot={len(prots)}")

    # Extract VIP for each key protein that belongs to this pathway
    for gene in key_proteins_gene:
        if gene in vip_dict:
            all_edges.append((label, gene, vip_dict[gene]))

print(f"[*] Total edges: {len(all_edges)}")

# Show some examples of the contrast the user wants
print("\n[*] Example VIP contrast — EGFR across models:")
for lab, gene, vip in all_edges:
    if gene == "EGFR":
        print(f"    {lab}: VIP = {vip:.3f}")
print("[*] Example VIP contrast — PRKAB1 across FoxO models:")
for lab, gene, vip in all_edges:
    if gene == "PRKAB1":
        print(f"    {lab}: VIP = {vip:.3f}")


# =============================================================================
# FIGURE: Full VIP Network
# =============================================================================
print("\n[*] Drawing network...", flush=True)

# --- Layout ---
model_labels = [m["label"] for m in model_info]
n_models = len(model_labels)

# Left positions (models) — grouped with gap between raw and resid
raw_models = [m for m in model_info if m["pipeline"] == "raw"]
resid_models = [m for m in model_info if m["pipeline"] == "resid"]

model_y = {}
y_cursor = 0.96
for m in raw_models:
    model_y[m["label"]] = y_cursor
    y_cursor -= 0.075
y_cursor -= 0.08  # gap
for m in resid_models:
    model_y[m["label"]] = y_cursor
    y_cursor -= 0.075

model_x = 0.0

# Right positions (proteins) — ordered to minimize crossings
protein_order = [
    "PTEN", "FOXO3", "EGFR", "PRKAB1", "PLK2",
    "COL4A1", "MMP9",
    "MAPK12",
    "MERTK", "PECAM1", "ADGRB1", "ADAM9", "HAVCR1",
    "CXCL8", "TLR4", "IL1B",
]
# Filter to only proteins that actually appear in edges
proteins_in_edges = set(g for _, g, _ in all_edges)
protein_order = [p for p in protein_order if p in proteins_in_edges]

prot_y = {}
n_prot = len(protein_order)
for i, prot in enumerate(protein_order):
    prot_y[prot] = 0.96 - i * (0.92 / max(n_prot - 1, 1))
prot_x = 1.0

# Pathway colors
def get_model_color(label):
    if "FoxO" in label:          return "#4393C3"
    if "Relaxin" in label:       return "#74ADD1"
    if "Efferocytosis" in label: return "#D6604D"
    if "NOD" in label:           return "#F4A582"
    if "Toll" in label:          return "#E8967A"
    return "#888888"

# VIP colormap (diverging: low=steel blue, high=dark red)
vip_all = [v for _, _, v in all_edges]
vip_floor, vip_ceil = 0.0, max(vip_all) + 0.1
cmap = cm.get_cmap("RdYlBu_r")
norm = Normalize(vmin=vip_floor, vmax=vip_ceil)

def vip_to_lw(v, lw_min=0.4, lw_max=4.0):
    t = np.clip((v - vip_floor) / (vip_ceil - vip_floor), 0, 1)
    return lw_min + t * (lw_max - lw_min)

def vip_to_alpha(v, a_min=0.25, a_max=0.9):
    t = np.clip((v - vip_floor) / (vip_ceil - vip_floor), 0, 1)
    return a_min + t * (a_max - a_min)

# Protein display names
prot_display = {
    "EGFR": "EGFR", "MAPK12": "MAPK12 (p38γ)", "PTEN": "PTEN",
    "FOXO3": "FOXO3", "PRKAB1": "PRKAB1 (AMPKβ1)", "PLK2": "PLK2",
    "COL4A1": "COL4A1", "MMP9": "MMP9",
    "MERTK": "MERTK", "PECAM1": "PECAM1 (CD31)",
    "ADGRB1": "ADGRB1 (BAI1)", "ADAM9": "ADAM9",
    "HAVCR1": "HAVCR1 (TIM-1)",
    "CXCL8": "CXCL8 (IL-8)", "TLR4": "TLR4", "IL1B": "IL-1β",
}

# Which proteins are shared across pathways?
prot_to_pathways = {}
for lab, gene, _ in all_edges:
    pw = lab.split(" → ")[0]
    prot_to_pathways.setdefault(gene, set()).add(pw)
shared_proteins = {p for p, pws in prot_to_pathways.items() if len(pws) > 1}

# --- Create figure ---
fig, ax = plt.subplots(figsize=(18.0 / 2.54, 18.0 / 2.54))
ax.set_xlim(-0.42, 1.42)
ax.set_ylim(-0.04, 1.08)
ax.axis("off")

# Section backgrounds
raw_y_vals = [model_y[m["label"]] for m in raw_models]
resid_y_vals = [model_y[m["label"]] for m in resid_models]
if raw_y_vals:
    r_top, r_bot = max(raw_y_vals) + 0.05, min(raw_y_vals) - 0.05
    ax.add_patch(FancyBboxPatch((-0.40, r_bot), 1.80, r_top - r_bot,
        boxstyle="round,pad=0.015", fc="#E8F0FE", ec="none", alpha=0.35, zorder=0))
    ax.text(-0.38, r_top + 0.01, "Raw (Macroscopic)", fontsize=7,
            fontweight="bold", color="#2166AC", va="bottom", fontstyle="italic")
if resid_y_vals:
    r_top, r_bot = max(resid_y_vals) + 0.05, min(resid_y_vals) - 0.05
    ax.add_patch(FancyBboxPatch((-0.40, r_bot), 1.80, r_top - r_bot,
        boxstyle="round,pad=0.015", fc="#FDE8E4", ec="none", alpha=0.35, zorder=0))
    ax.text(-0.38, r_top + 0.01, "Volume-Residualized (Microstructural)", fontsize=7,
            fontweight="bold", color="#B2182B", va="bottom", fontstyle="italic")

# --- Draw edges ---
# Sort edges so low-VIP drawn first (behind), high-VIP on top
edges_sorted = sorted(all_edges, key=lambda e: e[2])

for model_label, gene, vip in edges_sorted:
    if model_label not in model_y or gene not in prot_y:
        continue
    my = model_y[model_label]
    py = prot_y[gene]
    lw = vip_to_lw(vip)
    alpha = vip_to_alpha(vip)
    color = cmap(norm(vip))

    x0, y0 = model_x + 0.18, my
    x1, y1 = prot_x - 0.13, py
    cx0 = x0 + 0.25
    cx1 = x1 - 0.25

    verts = [(x0, y0), (cx0, y0), (cx1, y1), (x1, y1)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
    path = MPath(verts, codes)
    patch = PathPatch(path, facecolor="none", edgecolor=color,
                      linewidth=lw, alpha=alpha, capstyle="round", zorder=1)
    ax.add_patch(patch)

    # VIP label — show on ALL edges but size/weight varies
    t = 0.5
    mid_x = (1-t)**3*x0 + 3*(1-t)**2*t*cx0 + 3*(1-t)*t**2*cx1 + t**3*x1
    mid_y = (1-t)**3*y0 + 3*(1-t)**2*t*y0 + 3*(1-t)*t**2*y1 + t**3*y1

    if vip >= 1.5:
        fs, fw = 5.0, "bold"
    elif vip >= 1.0:
        fs, fw = 4.2, "normal"
    else:
        fs, fw = 3.8, "normal"

    ax.text(mid_x, mid_y, f"{vip:.2f}", fontsize=fs, fontweight=fw,
            ha="center", va="center", color=color, alpha=min(alpha + 0.15, 1.0),
            bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.65),
            zorder=2)

# --- Draw model nodes (left) ---
box_w, box_h = 0.32, 0.045
for m in model_info:
    lab = m["label"]
    y = model_y[lab]
    color = get_model_color(lab)
    q2 = m["q2"]

    box = FancyBboxPatch((model_x - 0.18, y - box_h/2), box_w, box_h,
        boxstyle="round,pad=0.01", fc=color, ec="black", linewidth=0.7,
        alpha=0.85, zorder=3)
    ax.add_patch(box)

    ax.text(model_x - 0.18 + box_w/2, y + 0.005, lab,
            fontsize=5.5, fontweight="bold", ha="center", va="center",
            color="white", zorder=4)
    ax.text(model_x - 0.18 + box_w/2, y - 0.014,
            f"Q²={q2:.2f}", fontsize=4.2, ha="center", va="center",
            color="white", alpha=0.9, zorder=4)

# --- Draw protein nodes (right) ---
pbox_w, pbox_h = 0.22, 0.033
for prot in protein_order:
    y = prot_y[prot]
    is_shared = prot in shared_proteins
    fc = "#FFF3CD" if is_shared else "#F5F5F5"
    ec = "#D4A017" if is_shared else "#999999"
    lw_b = 1.3 if is_shared else 0.6

    box = FancyBboxPatch((prot_x - 0.09, y - pbox_h/2), pbox_w, pbox_h,
        boxstyle="round,pad=0.006", fc=fc, ec=ec, linewidth=lw_b, zorder=3)
    ax.add_patch(box)

    display = prot_display.get(prot, prot)
    ax.text(prot_x - 0.09 + pbox_w/2, y, display,
            fontsize=5.2, fontweight="bold" if is_shared else "normal",
            ha="center", va="center", zorder=4)

    # Colored dots showing which pathways connect
    if is_shared:
        pws = sorted(prot_to_pathways[prot])
        for ci, pw_name in enumerate(pws):
            dummy_label = [lab for lab, g, _ in all_edges if g == prot and lab.startswith(pw_name)][0]
            c = get_model_color(dummy_label)
            dot_x = prot_x - 0.09 + pbox_w + 0.025
            dot_y = y + (ci - (len(pws)-1)/2) * 0.013
            ax.plot(dot_x, dot_y, "o", color=c, markersize=3,
                    markeredgecolor="black", markeredgewidth=0.3, zorder=4)

# --- Colorbar for VIP ---
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.82, 0.08, 0.015, 0.18])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
cbar.set_label("VIP Score", fontsize=6, fontweight="bold")
cbar.ax.tick_params(labelsize=5)

# --- Legend ---
legend_elements = [
    Line2D([0],[0], marker='s', color='w', markerfacecolor="#4393C3", markersize=7, label='FoxO'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor="#74ADD1", markersize=7, label='Relaxin'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor="#D6604D", markersize=7, label='Efferocytosis'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor="#F4A582", markersize=7, label='NOD-like receptor'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor="#E8967A", markersize=7, label='Toll-like receptor'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor="#FFF3CD", markersize=7,
           markeredgecolor="#D4A017", markeredgewidth=1.0, label='Shared protein'),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=5.5,
          framealpha=0.9, edgecolor="grey",
          bbox_to_anchor=(-0.05, -0.02))

# --- Title ---
ax.text(0.45, 1.06,
    "Pathway–Protein VIP Network Across Radiomic Associations",
    fontsize=9, fontweight="bold", ha="center", transform=ax.transAxes)
ax.text(0.45, 1.025,
    "Edge width + color = VIP score",
    fontsize=5.5, ha="center", color="#555555", transform=ax.transAxes)

fig.savefig("Figure_VIP_Network.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
fig.savefig("Figure_VIP_Network.png", dpi=FIGURE_DPI, bbox_inches="tight")
print("[+] Saved Figure_VIP_Network.pdf/.png")
plt.close(fig)
