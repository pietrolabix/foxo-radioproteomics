# Radioproteomics of the failing myocardium

Analysis code for:

> Łaba P, Mazuruk M, Kacperska M, Paterek A, Załęska-Kocięcka M, Michałowska I, Leszek P, Mączewski M.
> *Machine Learning–Based Integrative Radioproteomics Links Circulating Efferocytosis and FoxO Signaling to Myocardial CT Texture in End-Stage Heart Failure.* (submitted)

The pipeline links CT radiomics of the left ventricular myocardium with plasma proteomics (Olink Explore 5400+) in 42 patients with end-stage heart failure with reduced ejection fraction evaluated for LVAD implantation. Circulating proteins are grouped into KEGG pathways, and each pathway is tested as a predictor of each radiomic feature with partial least squares (PLS) regression.

**This repository contains code only.** Patient-level clinical, imaging and proteomic data are not included and are available from the corresponding author on reasonable request, subject to institutional approval.

## Analysis design

```
CCTA ── TotalSegmentator + manual correction ── PyRadiomics (93 features)
                                                   │
                                   ┌───────────────┴───────────────┐
                              raw features            residualized on log voxel count
                         (size + texture)            (volume-independent texture)
                                   └───────────────┬───────────────┘
Olink Explore 5400+ ── 159 KEGG pathways ── PLS, 5-fold CV screen (14,787 models, Q² ≥ 0.30)
                                                   │
                         permutation tests, Li & Ji screen-wide correction, bootstrap
                                                   │
                     re-validation: 50× repeated nested CV + 10,000 permutations of the
                                    whole nested pipeline (GPU)
```

Two feature branches are analyzed in parallel. The number of voxels in the myocardial mask depends on myocardial size and on reconstruction resolution, and raw first-order Energy scales with it. Residualizing every feature on log voxel count isolates texture that is independent of mask size.

## Scripts

Run from the repository root, in this order.

| Script | Purpose | Main output |
|---|---|---|
| `01_screening_raw_bootstrap.py` | GPU PLS screen of raw features; permutation and bootstrap for candidates | `Radiomics_RAW_Bootstrap_Stability.csv` |
| `02_screening_residualized_liji.py` | Screen of volume-residualized features; Li & Ji effective number of tests; permutations and bootstrap | `Radiomics_FoxO_LinLog_LiJi_Candidates.csv` |
| `03_efferocytosis_permutations.py` | 1,000,000 permutations for efferocytosis → residualized Energy | console |
| `04_extract_vips.py` | Variable importance in projection (VIP) for selected models | `Supplementary_Table_S1_VIPs.xlsx` |
| `05_figures_full_manuscript.py` | Screening landscape, forest plot, VIP profiles, clinical integration | `Figure1–4_*.pdf/png` |
| `06_figure_vip_network.py` | Pathway–protein VIP network | `Figure_VIP_Network.pdf/png` |
| `07_revalidation_nested_cv.py` | Repeated nested cross-validation of the screened models (CPU) | `results/reanalysis_letter_results.csv` |
| `08_revalidation_permutations_gpu.py` | Permutation test of the whole nested pipeline on MPS/CUDA | `results/reanalysis_letter_perm_gpu.csv` |
| `09_figure_research_letter.py` | Combined figure for the research letter (VIP network + results table) | `results/Figure1.pdf/png/tiff` |
| `10_extra_analyses_specificity_acquisition_tissue_clinical.py` | Random protein-set null, adjustment for voxel and myocardial volume, myocardial tissue proteome, clinical correlates | `results/extra_analyses_letter.xlsx` |

Scripts 07–09 reuse the data-loading part of scripts 05 and 06, so the cohort, pathway definitions and targets are identical across analyses.

## Data

Place the input files in `data/` (or point `FOXO_DATA_DIR` to another folder):

| File | Content |
|---|---|
| `bdanych-11.xlsx` | clinical database; sheet `baza myocardium` holds the radiomic features |
| `pre_quant.csv` | Olink NPX values in long format (`SampleID`, `Assay`, `OlinkID`, `NPX`) |
| `HealthyvsLVAD_KEGGenrich.html` | KEGG enrichment report used to define pathway protein sets |
| `lvad_serum2_scalony_POPRAWIONY_probki.csv` | sample metadata (clinical integration, script 05 only) |

## Environment

Python 3.11+.

```bash
pip install -r requirements.txt
```

The GPU scripts use PyTorch and pick Apple MPS, CUDA or CPU automatically.

```bash
FOXO_DATA_DIR=/path/to/data python scripts/02_screening_residualized_liji.py
```

## Methods in brief

- **Segmentation:** TotalSegmentator (`heartchambers_highres`), manual correction in 3D Slicer.
- **Radiomics:** PyRadiomics, 93 features (first order, GLCM, GLRLM, GLSZM, GLDM, NGTDM) at native resolution.
- **Model:** PLS1 with predictors standardized within training folds; median imputation of missing protein values; up to 4–5 latent components.
- **Screening statistic:** cross-validated Q² = 1 − PRESS / TSS.
- **Multiplicity:** effective number of independent tests from the eigenvalues of the pathway and feature correlation matrices (Li & Ji, *Heredity* 2005).
- **Re-validation:** the number of components is chosen by an inner 5-fold CV on training folds only. The outer 5-fold CV is repeated 50 times, and the permutation null re-runs the whole nested pipeline.
