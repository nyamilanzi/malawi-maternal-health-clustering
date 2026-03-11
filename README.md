# Temporal Patterns of Maternal Health Care Utilisation in Malawi
## A Comparative Unsupervised Learning Analysis of the 2015 and 2024 Demographic and Health Surveys

---

## Overview

This repository contains all analysis code, results, and figures for a comparative study of maternal health care utilisation patterns in Malawi, using nationally representative data from the 2015 and 2024 Malawi Demographic and Health Surveys (MDHS). The study applies Gaussian Mixture Model (GMM) clustering to identify distinct utilisation patterns across the antenatal, delivery, and postnatal care continuum, and tracks how these patterns changed over a decade of health system investment.

**Key findings:**
- A four-cluster solution (2015) and three-cluster solution (2024) best described utilisation heterogeneity
- The high-coverage cluster nearly doubled (24.2% → 46.7%), driven by a 31.5 percentage point rise in early ANC initiation
- The minimal utilisation cluster (8.8% in 2015) had no identifiable analogue in 2024
- Sociodemographic gradients narrowed across all subgroups, but adolescent women and the poorest quintile retain persistent gaps

---

## Repository Structure

```
.
├── 00_run_all.py                    # Master script: runs full pipeline in sequence
├── 01_data_harmonization.py         # Harmonises 2015 and 2024 MDHS datasets
├── 02_clustering_by_year.py         # GMM clustering per survey year (K selection)
├── 03_comparative_analysis.py       # Cross-year cluster alignment and temporal comparison
├── 04_subgroup_analysis.py          # Cluster distributions by residence, wealth, age, education
├── 05_predictors_membership.py      # Multinomial logit + Random Forest predictors
├── 06_visualizations.py             # Publication-ready figures (Fig 1–5)
├── build_table1_and_figs.py         # Table 1 (demographics) + Figure 2 cluster profiles
├── generate_scatter.py              # Figure 6: PCA scatter plot
├── requirements.txt                 # Python dependencies
│
├── results/                         # CSV outputs from all scripts
│   ├── harmonised_2015.csv
│   ├── harmonised_2024.csv
│   ├── clusters_2015.csv
│   ├── clusters_2024.csv
│   ├── cluster_profiles_2015.csv
│   ├── cluster_profiles_2024.csv
│   ├── cluster_alignment.csv
│   ├── temporal_comparison.csv
│   ├── subgroup_prevalence_*.csv
│   ├── multinomial_results_*.csv
│   ├── rf_importance_all.csv
│   ├── predictability_summary.csv
│   ├── model_selection_summary.csv
│   └── table1_by_cluster.json
│
├── figures/                         # All generated figures (PNG, 300 DPI)
│   ├── Fig1_flow_diagram.png
│   ├── Fig2_cluster_profiles.png
│   ├── Fig3_temporal_shift.png
│   ├── Fig4_subgroup_distributions.png
│   ├── Fig5_predictors.png
│   ├── Fig6_pca_scatter.png
│   └── 02_*, 03_*, 04_* ...        # Diagnostic/exploratory figures
│
└── manuscript/
    ├── manuscript_comparative_final.md              # Full manuscript (Markdown)
    ├── maternal_health_malawi_comparative_2015_2024.docx  # Submission-ready Word document
    └── create_docx.js               # Node.js script to build the Word document
```

---

## Data

This analysis uses individual recode datasets from the Malawi Demographic and Health Survey:

| Survey | Dataset | N (all women) | N (analysis sample) |
|--------|---------|---------------|---------------------|
| MDHS 2015 | MWIR7AFL.DTA | 24,562 | 13,030 |
| MDHS 2024 | MWIR81FL_compressed.dta | 21,587 | 6,925 |

**The raw DHS datasets are not included in this repository.** They are available for download from the [DHS Program website](https://dhsprogram.com/data/available-datasets.cfm) upon registration (free for researchers).

Place downloaded datasets in the parent `data/` directory before running the pipeline.

---

## Clustering Features (11)

All features are derived from standard DHS variables and harmonised across survey rounds:

| Feature | Type | Description |
|---------|------|-------------|
| `first_anc_month` | Integer (1–9) | Month of first ANC visit in pregnancy |
| `anc_visits` | Count (capped 20) | Total number of ANC visits |
| `early_anc` | Binary | First ANC visit within first trimester |
| `adequate_anc` | Binary | 4 or more ANC visits (WHO minimum) |
| `optimal_anc` | Binary | 8 or more ANC visits (WHO 2016 guideline) |
| `skilled_anc` | Binary | ANC provided by doctor or nurse/midwife |
| `told_complications` | Binary | Woman informed about danger signs at ANC |
| `facility_delivery` | Binary | Delivery at a health facility |
| `skilled_delivery` | Binary | Delivery attended by doctor or nurse/midwife |
| `caesarean` | Binary | Caesarean section delivery |
| `pnc_received` | Binary | Postnatal care received |

Sociodemographic variables (age, residence, wealth quintile, education, parity, region, marital status, perceived distance to facility) are used for **subgroup and predictor analyses only** and are not included in clustering.

---

## Methods Summary

1. **Harmonisation** (`01`): DHS variables extracted, recoded, and harmonised across survey rounds using standard variable names
2. **Clustering** (`02`): GMM independently per year (K = 2–6); optimal K selected by composite score (BIC 60% + silhouette 40%), minimum cluster size 5%
3. **Comparison** (`03`): Hungarian algorithm aligns 2024 clusters to 2015 counterparts; feature-level t-tests and chi-square tests assess temporal change
4. **Subgroups** (`04`): Cramer's V and chi-square tests by residence, wealth, age, education
5. **Predictors** (`05`): Multinomial logistic regression (adjusted ORs) + Random Forest (MDI importance, 5-fold cross-validated F1)
6. **Figures** (`06`, `build_table1_and_figs.py`, `generate_scatter.py`): Publication-quality figures

---

## How to Run

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Place raw DHS data files in `../` (parent of this directory)

```
data/
├── MWIR7AFL.DTA           # MDHS 2015
├── MWIR81FL_compressed.dta  # MDHS 2024
└── comparative_analysis/    # This repository
```

### 3. Run the full pipeline

```bash
python3 00_run_all.py
```

Or run scripts individually in order (01 → 06).

### 4. Generate Word document (optional)

Requires Node.js and the `docx` npm package:

```bash
cd manuscript
npm install docx
node create_docx.js
```

---

## Requirements

**Python ≥ 3.9**

Key packages (see `requirements.txt` for full list):
- `pandas`, `numpy`
- `scikit-learn`
- `scipy`
- `statsmodels`
- `matplotlib`, `seaborn`

**Node.js ≥ 16** (manuscript Word document only)

---

## Results Summary

### Cluster structures

| Cluster | 2015 (n, %) | Description |
|---------|-------------|-------------|
| C-A | 3,149 (24.2%) | Comprehensive ANC + skilled delivery |
| C-B | 7,828 (60.1%) | Late ANC, universal facility delivery |
| C-C | 910 (7.0%) | C-section / high PNC |
| C-D | 1,143 (8.8%) | Minimal utilisation |

| Cluster | 2024 (n, %) | Description |
|---------|-------------|-------------|
| C-1 | 3,232 (46.7%) | High coverage across continuum |
| C-2 | 3,349 (48.4%) | Late ANC, universal facility delivery |
| C-3 | 344 (5.0%) | Moderate ANC, limited delivery |

### Key temporal changes (2015 → 2024)

| Indicator | 2015 | 2024 | Change |
|-----------|------|------|--------|
| Early ANC initiation | 26.0% | 57.5% | **+31.5 pp** |
| Adequate ANC (≥4 visits) | 51.9% | 66.6% | +14.7 pp |
| Skilled birth attendance | 92.0% | 97.1% | +5.0 pp |
| Facility delivery | 94.8% | 97.6% | +2.7 pp |
| Caesarean section | 7.0% | 11.5% | +4.5 pp |
| PNC receipt | 45.1% | 43.2% | −1.9 pp |

---

]

---

## Ethics

Both DHS surveys were conducted under protocols reviewed by the ICF Institutional Review Board and the Malawi National Health Sciences Research Committee. Participation was voluntary and all data are anonymised. Secondary analysis of publicly available de-identified data does not require additional ethics review.

---

## License

Code is released under the [MIT License](LICENSE). Results and figures are released for academic use. Raw DHS data remain subject to the [DHS Program data use agreement](https://dhsprogram.com/data/Terms-of-Use.cfm).

---

*Analysis conducted using Python 3.x and Node.js. Last updated: March 2026.*
