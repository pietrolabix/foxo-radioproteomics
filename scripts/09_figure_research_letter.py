#!/usr/bin/env python3
"""
Figure for the Circulation Research Research Letter (single display item).

Panel A: pathway-protein VIP network, drawn by generate_figures2.py with the
         numeric labels (edge VIP values, Q2 in model boxes) and the title removed.
Panel B: table of the Table 1 models with screening and re-validated statistics
         (reanalysis_letter_results.csv, reanalysis_letter_perm_gpu.csv).

Run from the repository root:  python scripts/09_figure_research_letter.py
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

NETWORK_BASENAME = "results/panel_network"

# --- Panel A: reuse generate_figures2.py without numeric labels ---
src = open("scripts/06_figure_vip_network.py", encoding="utf-8").read()
src, n_edge = re.subn(r'\n    ax\.text\(mid_x, mid_y, f"\{vip:\.2f\}".*?zorder=2\)\n', "\n", src, flags=re.S)
src, n_q2 = re.subn(r'\n    ax\.text\(model_x - 0\.18 \+ box_w/2, y - 0\.014,.*?zorder=4\)\n', "\n", src, flags=re.S)
src, n_title = re.subn(r'\n# --- Title ---\n.*?transform=ax\.transAxes\)\n.*?transform=ax\.transAxes\)\n', "\n", src, flags=re.S)
src = src.replace('ax.text(model_x - 0.18 + box_w/2, y + 0.005, lab,', 'ax.text(model_x - 0.18 + box_w/2, y, lab,')
src = src.replace('"Figure_VIP_Network.', f'"{NETWORK_BASENAME}.')
src = src.replace('cm.get_cmap("RdYlBu_r")', 'matplotlib.colormaps["RdYlBu_r"]')
assert (n_edge, n_q2, n_title) == (1, 1, 1), (n_edge, n_q2, n_title)
exec(compile(src, "06_figure_vip_network.py[patched]", "exec"), {"__name__": "network"})

# --- Panel B: table ---
nested = pd.read_csv("results/reanalysis_letter_results.csv").drop(columns="p_perm_nested")  # 2-permutation smoke value
perm = pd.read_csv("results/reanalysis_letter_perm_gpu.csv")
tab = nested.merge(perm[["pathway", "feature", "branch", "p_perm_nested"]],
                   on=["pathway", "feature", "branch"])
P_ADJ_LIJI = [0.0417, 0.2187, 0.7204, 0.7204, 1.0, 0.0036, 0.0549, 0.0645, 0.1204, 0.3601]
tab["p_adj"] = P_ADJ_LIJI  # screen-wide Li & Ji adjustment, as in Table 1 of the full manuscript

PATHWAY = {"FoxO_signaling_pathway": "FoxO", "Relaxin_signaling_pathway": "Relaxin",
           "Efferocytosis": "Efferocytosis", "NOD_like_receptor_signaling_pa": "NOD-like receptor",
           "Toll_like_receptor_signaling_p": "Toll-like receptor"}
FEATURE = {"firstorder_Energy": "Energy", "firstorder_TotalEnergy": "Total Energy",
           "firstorder_RobustMeanAbsoluteDeviation": "RMAD", "firstorder_InterquartileRange": "IQR",
           "firstorder_RootMeanSquared": "RMS", "glrlm_RunLengthNonUniformity": "GLRLM RLNU",
           "glcm_ClusterTendency": "GLCM Cluster Tendency"}


def fmt_p(p):
    if p >= 0.9995:
        return ">0.99"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


rows, robust = [], []
for _, r in tab.iterrows():
    lo, hi = r["Q2_nested_IQR"].split("..")
    rows.append([
        "Raw" if r["branch"] == "raw" else "Volume-residualized",
        PATHWAY[r["pathway"]], FEATURE[r["feature"]],
        f"{r['Q2_original_split']:.2f}", fmt_p(r["p_adj"]),
        f"{r['Q2_nested_median']:.2f} ({float(lo):.2f} to {float(hi):.2f})",
        f"{r['share_nested_pos'] * 100:.0f}%", fmt_p(r["p_perm_nested"]),
    ])
    robust.append(r["Q2_nested_median"] >= 0.15 and r["p_perm_nested"] < 0.001)

header = ["Features", "Pathway", "Radiomic feature", "Screening\nQ²", "Screen-wide\nadjusted P",
          "Nested Q², median\n(IQR)", "Repeats\nwith Q²>0", "Permutation\nP"]

# --- Combined figure ---
net = mpimg.imread(f"{NETWORK_BASENAME}.png")
W = 18.0 / 2.54
h_net = W * net.shape[0] / net.shape[1]
h_tab = 2.35
fig = plt.figure(figsize=(W, h_net + h_tab + 0.3))
ax_a = fig.add_axes([0, (h_tab + 0.3) / (h_net + h_tab + 0.3), 1, h_net / (h_net + h_tab + 0.3)])
ax_a.imshow(net, interpolation="lanczos")
ax_a.axis("off")
ax_a.text(0.0, 1.0, "A", transform=ax_a.transAxes, fontsize=12, fontweight="bold", va="top")

ax_b = fig.add_axes([0.0, 0.0, 1.0, h_tab / (h_net + h_tab + 0.3)])
ax_b.axis("off")
ax_b.text(0.0, 1.08, "B", transform=ax_b.transAxes, fontsize=12, fontweight="bold", va="top")
table = ax_b.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
                   colWidths=[0.15, 0.13, 0.15, 0.08, 0.10, 0.17, 0.09, 0.10],
                   bbox=[0.0, 0.0, 1.0, 1.0])
table.auto_set_font_size(False)
table.set_fontsize(5.6)
for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor("#BBBBBB")
    cell.set_linewidth(0.3)
    if i == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#EEEEEE")
        cell.set_height(cell.get_height() * 1.6)
    else:
        branch_raw = rows[i - 1][0] == "Raw"
        cell.set_facecolor("#F4F8FC" if branch_raw else "#FDF5F4")
        if robust[i - 1]:
            cell.set_text_props(fontweight="bold")

fig.savefig("results/Figure1.pdf", dpi=600, bbox_inches="tight")
fig.savefig("results/Figure1.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Figure1.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
print("[+] Saved letter/Figure1.pdf/.png/.tiff")
