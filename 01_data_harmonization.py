"""
=============================================================================
SCRIPT 01: DATA HARMONIZATION - MDHS 2015 AND 2024
=============================================================================
Load, harmonise, and prepare maternal care utilisation features from the
2015 (MWIR7AFL) and 2024 (MWIR81FL) Malawi Demographic and Health Surveys.

Variables are selected to be comparable across both survey rounds.
Sociodemographic variables are retained for subgroup and predictor analyses.

Output:
  results/harmonised_2015.csv
  results/harmonised_2024.csv
  results/harmonised_pooled.csv
  results/harmonisation_report.txt
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

BASE = Path(__file__).parent
DATA = BASE.parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

log("=" * 70)
log("DATA HARMONISATION: MDHS 2015 AND MDHS 2024")
log("=" * 70)

# ============================================================
# 1. LOAD DATASETS
# ============================================================

log("\n" + "-" * 70)
log("LOADING DATASETS")
log("-" * 70)

# 2015 dataset
path_2015 = DATA / "MWIR7AFL.DTA"
log(f"\nLoading 2015 dataset: {path_2015.name}")
df15 = pd.read_stata(path_2015, convert_categoricals=False)
log(f"  Shape: {df15.shape[0]:,} women × {df15.shape[1]:,} variables")

# 2024 dataset
path_2024 = DATA / "MWIR81FL_compressed.dta"
log(f"\nLoading 2024 dataset: {path_2024.name}")
df24 = pd.read_stata(path_2024, convert_categoricals=False)
log(f"  Shape: {df24.shape[0]:,} women × {df24.shape[1]:,} variables")

# ============================================================
# 2. RESTRICT TO WOMEN WITH A RECENT LIVE BIRTH
# ============================================================

log("\n" + "-" * 70)
log("RESTRICTING TO WOMEN WITH BIRTH IN LAST 5 YEARS (v208 > 0)")
log("-" * 70)

df15_eligible = df15[df15['v208'] > 0].copy()
df24_eligible = df24[df24['v208'] > 0].copy()

log(f"  2015 eligible: {df15_eligible.shape[0]:,} women")
log(f"  2024 eligible: {df24_eligible.shape[0]:,} women")

# ============================================================
# 3. HARMONISED VARIABLE EXTRACTION FUNCTION
# ============================================================

def extract_features(df, year_label):
    """
    Extract harmonised maternal care utilisation features and
    sociodemographic variables from a DHS individual recode dataset.

    All DHS standard variable names are used to ensure comparability.
    Missing value codes (98, 99) are replaced with NaN.
    """
    log(f"\n  --- Extracting features for {year_label} ---")

    features = pd.DataFrame(index=df.index)

    # ----- ANTENATAL CARE -----

    # Timing of first ANC visit (months into pregnancy, 1–9)
    if 'm13_1' in df.columns:
        v = df['m13_1'].copy().astype(float)
        v[v.isin([98, 99])] = np.nan
        v = v.clip(1, 9)
        features['first_anc_month'] = v
        log(f"    ✓ first_anc_month  non-missing={v.notna().sum():,}  mean={v.mean():.2f}")

    # Number of ANC visits
    if 'm14_1' in df.columns:
        v = df['m14_1'].copy().astype(float)
        v[v.isin([98, 99])] = np.nan
        v = v.clip(0, 20)
        features['anc_visits'] = v
        log(f"    ✓ anc_visits       non-missing={v.notna().sum():,}  mean={v.mean():.2f}")

    # Skilled ANC provider (doctor or nurse/midwife)
    for a, b in [('m2a_1', 'm2b_1')]:
        if a in df.columns and b in df.columns:
            features['skilled_anc'] = (
                (df[a] == 1) | (df[b] == 1)
            ).astype(float)
            log(f"    ✓ skilled_anc      pct={(features['skilled_anc'].mean()*100):.1f}%")

    # Derived: early ANC (first visit ≤ 3 months, i.e. first trimester)
    if 'first_anc_month' in features.columns:
        features['early_anc'] = np.where(
            features['first_anc_month'].notna(),
            (features['first_anc_month'] <= 3).astype(float),
            np.nan
        )
        log(f"    ✓ early_anc        pct={(features['early_anc'].mean()*100):.1f}%")

    # Derived: adequate ANC (≥4 visits, WHO minimum)
    if 'anc_visits' in features.columns:
        features['adequate_anc'] = np.where(
            features['anc_visits'].notna(),
            (features['anc_visits'] >= 4).astype(float),
            np.nan
        )
        log(f"    ✓ adequate_anc     pct={(features['adequate_anc'].mean()*100):.1f}%")

    # Derived: optimal ANC (≥8 visits, updated WHO 2016 recommendation)
    if 'anc_visits' in features.columns:
        features['optimal_anc'] = np.where(
            features['anc_visits'].notna(),
            (features['anc_visits'] >= 8).astype(float),
            np.nan
        )
        log(f"    ✓ optimal_anc      pct={(features['optimal_anc'].mean()*100):.1f}%")

    # Informed about danger signs / complications during ANC
    # m57e_1 is specific to MDHS 2024; fall back to m56j_1 or skip
    comp_var = None
    for candidate in ['m57e_1', 'm56j_1', 'm56e_1']:
        if candidate in df.columns:
            comp_var = candidate
            break
    if comp_var:
        v = (df[comp_var] == 1).astype(float)
        features['told_complications'] = v
        log(f"    ✓ told_complications  ({comp_var})  pct={(v.mean()*100):.1f}%")
    else:
        log(f"    ⚠ told_complications: no matching variable in {year_label}; omitted")

    # ----- DELIVERY CARE -----

    # Facility delivery: coding differs by survey round.
    # MDHS 2015: place codes ≥20 = facility (standard DHS 6 coding)
    # MDHS 2024 compressed: codes 2–10 = facility (short-integer recoding)
    if 'm15_1' in df.columns:
        try:
            v = pd.to_numeric(df['m15_1'], errors='coerce')
        except Exception:
            v = df['m15_1'].copy().astype(float)
        v[v.isin([98, 99])] = np.nan
        max_val = v.dropna().max()
        if max_val >= 20:   # Standard DHS 6 place codes
            fac = (v >= 20).astype(float)
        else:               # Short-code scheme (2024 compressed)
            fac = v.isin([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(float)
        features['facility_delivery'] = np.where(v.notna(), fac, np.nan)
        log(f"    ✓ facility_delivery pct={(features['facility_delivery'].mean()*100):.1f}%")

    # Skilled birth attendant (doctor or nurse/midwife)
    for a, b in [('m3a_1', 'm3b_1')]:
        if a in df.columns and b in df.columns:
            features['skilled_delivery'] = (
                (df[a] == 1) | (df[b] == 1)
            ).astype(float)
            log(f"    ✓ skilled_delivery  pct={(features['skilled_delivery'].mean()*100):.1f}%")

    # Caesarean section
    if 'm17_1' in df.columns:
        v = df['m17_1'].copy().astype(float)
        v[v.isin([8, 9])] = np.nan
        features['caesarean'] = np.where(v.notna(), (v == 1).astype(float), np.nan)
        log(f"    ✓ caesarean         pct={(features['caesarean'].mean()*100):.1f}%")

    # ----- POSTNATAL CARE -----

    # PNC timing in hours (m50_1 or similar)
    pnc_found = False
    for var in ['m50_1', 'm51_1', 'm52_1']:
        if var in df.columns:
            v = df[var].copy().astype(float)
            v[v >= 995] = np.nan
            unique_v = v.dropna().unique()
            if len(unique_v) > 0 and v.dropna().max() > 2:
                features['pnc_48h'] = np.where(v.notna(), (v <= 48).astype(float), np.nan)
                log(f"    ✓ pnc_48h          ({var})  pct={(features['pnc_48h'].mean()*100):.1f}%")
                pnc_found = True
                break

    if not pnc_found:
        # Binary PNC receipt
        for var in ['m50_1', 'm70_1', 'm71_1']:
            if var in df.columns:
                v = df[var].copy().astype(float)
                v[v.isin([8, 9])] = np.nan
                features['pnc_received'] = np.where(v.notna(), (v == 1).astype(float), np.nan)
                log(f"    ✓ pnc_received     ({var})  pct={(features['pnc_received'].mean()*100):.1f}%")
                pnc_found = True
                break

    if not pnc_found:
        log(f"    ⚠ No PNC variable found for {year_label}; inserting NaN placeholder")
        features['pnc_received'] = np.nan

    return features


def extract_sociodemographic(df, year_label):
    """
    Extract sociodemographic and background variables for external validation,
    subgroup analyses, and predictor modelling.
    """
    log(f"\n  --- Extracting sociodemographics for {year_label} ---")

    soc = pd.DataFrame(index=df.index)

    # Age (continuous)
    if 'v012' in df.columns:
        soc['age'] = df['v012'].astype(float)

    # Age group (adolescent=15-19, young adult=20-34, older=35-49)
    if 'v012' in df.columns:
        age = df['v012'].astype(float)
        soc['age_group'] = pd.cut(
            age,
            bins=[14, 19, 34, 49],
            labels=['Adolescent (15–19)', 'Young adult (20–34)', 'Older (35–49)']
        )

    # Residence (urban / rural)
    if 'v025' in df.columns:
        # MDHS 2015: 1=Urban, 2=Rural (standard DHS6)
        # MDHS 2024 compressed: 0=Urban, 1=Rural (recoded short integers)
        vals = df['v025'].dropna().unique()
        if set(vals).issubset({0, 1}):
            soc['residence'] = df['v025'].map({0: 'Urban', 1: 'Rural'})
        else:
            soc['residence'] = df['v025'].map({1: 'Urban', 2: 'Rural'})

    # Wealth quintile
    # MDHS 2015: 1=Poorest … 5=Richest  |  MDHS 2024 compressed: 0=Poorest … 4=Richest
    if 'v190' in df.columns:
        vals = df['v190'].dropna().unique()
        if df['v190'].dropna().min() == 0:   # 0-based scheme (2024)
            soc['wealth_quintile'] = df['v190'].map({
                0: 'Poorest', 1: 'Poorer', 2: 'Middle', 3: 'Richer', 4: 'Richest'
            })
        else:                                 # 1-based scheme (2015)
            soc['wealth_quintile'] = df['v190'].map({
                1: 'Poorest', 2: 'Poorer', 3: 'Middle', 4: 'Richer', 5: 'Richest'
            })

    # Education level
    if 'v106' in df.columns:
        soc['education'] = df['v106'].map({
            0: 'No education', 1: 'Primary', 2: 'Secondary', 3: 'Higher'
        })

    # Parity (number of births in last 5 years as proxy, or total children)
    if 'v208' in df.columns:
        soc['births_last5'] = df['v208'].astype(float)
    if 'v201' in df.columns:
        soc['parity'] = df['v201'].astype(float)

    # Region / province
    for v in ['v024', 'v101']:
        if v in df.columns:
            soc['region'] = df[v].astype(float)
            break

    # Marital status
    if 'v501' in df.columns:
        soc['married'] = (df['v501'] == 1).astype(int)

    # Distance to health facility perceived as problem (v467d or similar)
    for v in ['v467d', 'v467b']:
        if v in df.columns:
            soc['distance_problem'] = (df[v] == 1).astype(int)
            break

    log(f"    Sociodemographic variables: {list(soc.columns)}")
    return soc


# ============================================================
# 4. EXTRACT FEATURES FROM BOTH YEARS
# ============================================================

feats15 = extract_features(df15_eligible, '2015')
feats24 = extract_features(df24_eligible, '2024')

soc15 = extract_sociodemographic(df15_eligible, '2015')
soc24 = extract_sociodemographic(df24_eligible, '2024')

# ============================================================
# 5. ALIGN COLUMNS (USE INTERSECTION FOR CLUSTERING)
# ============================================================

log("\n" + "-" * 70)
log("ALIGNING FEATURE SETS")
log("-" * 70)

cols15 = set(feats15.columns)
cols24 = set(feats24.columns)
common_cols = sorted(cols15 & cols24)
only15 = cols15 - cols24
only24 = cols24 - cols15

log(f"  Common clustering features: {common_cols}")
log(f"  Only in 2015: {only15 if only15 else 'none'}")
log(f"  Only in 2024: {only24 if only24 else 'none'}")

# Use common columns for comparative clustering
feats15_common = feats15[common_cols].copy()
feats24_common = feats24[common_cols].copy()

# ============================================================
# 6. HANDLE MISSING DATA (COMPLETE-CASE)
# ============================================================

log("\n" + "-" * 70)
log("MISSING DATA HANDLING (COMPLETE-CASE ANALYSIS)")
log("-" * 70)

for label, feat in [('2015', feats15_common), ('2024', feats24_common)]:
    miss = (feat.isnull().sum() / len(feat) * 100).round(1)
    log(f"\n  {label} missing (%):")
    for col, pct in miss.items():
        log(f"    {col}: {pct}%")

# Drop features with >50% missing in either year
threshold = 50.0
bad_cols = []
for col in common_cols:
    pct15 = feats15_common[col].isnull().mean() * 100
    pct24 = feats24_common[col].isnull().mean() * 100
    if pct15 > threshold or pct24 > threshold:
        bad_cols.append(col)
        log(f"  ⚠ Dropping {col} (>50% missing in at least one year)")

final_cols = [c for c in common_cols if c not in bad_cols]
feats15_final = feats15_common[final_cols].copy()
feats24_final = feats24_common[final_cols].copy()

log(f"\n  Final clustering features ({len(final_cols)}): {final_cols}")

# Complete cases
idx15 = feats15_final.dropna().index
idx24 = feats24_final.dropna().index

feats15_cc = feats15_final.loc[idx15]
feats24_cc = feats24_final.loc[idx24]
soc15_cc = soc15.loc[idx15]
soc24_cc = soc24.loc[idx24]

log(f"\n  2015 complete cases: {len(idx15):,} / {len(feats15_final):,} "
    f"({len(idx15)/len(feats15_final)*100:.1f}%)")
log(f"  2024 complete cases: {len(idx24):,} / {len(feats24_final):,} "
    f"({len(idx24)/len(feats24_final)*100:.1f}%)")

# ============================================================
# 7. BUILD COMBINED DATAFRAMES
# ============================================================

df_out15 = feats15_cc.copy()
df_out15 = df_out15.join(soc15_cc, how='left')
df_out15['survey_year'] = 2015
df_out15.index = range(len(df_out15))

df_out24 = feats24_cc.copy()
df_out24 = df_out24.join(soc24_cc, how='left')
df_out24['survey_year'] = 2024
df_out24.index = range(len(df_out24))

df_pooled = pd.concat([df_out15, df_out24], ignore_index=True)

# ============================================================
# 8. DESCRIPTIVE COMPARISON TABLE
# ============================================================

log("\n" + "-" * 70)
log("DESCRIPTIVE COMPARISON OF CLUSTERING FEATURES BY YEAR")
log("-" * 70)

desc_rows = []
for col in final_cols:
    v15 = df_out15[col]
    v24 = df_out24[col]
    if v15.nunique() <= 2:
        row = {
            'Feature': col,
            'Type': 'Binary',
            '2015 % (n)': f"{v15.mean()*100:.1f}% (n={v15.sum():.0f})",
            '2024 % (n)': f"{v24.mean()*100:.1f}% (n={v24.sum():.0f})"
        }
    else:
        row = {
            'Feature': col,
            'Type': 'Continuous',
            '2015 % (n)': f"mean={v15.mean():.2f}, sd={v15.std():.2f}",
            '2024 % (n)': f"mean={v24.mean():.2f}, sd={v24.std():.2f}"
        }
    desc_rows.append(row)
    log(f"  {col}: 2015 → {row['2015 % (n)']}  |  2024 → {row['2024 % (n)']}")

desc_df = pd.DataFrame(desc_rows)
desc_df.to_csv(RESULTS / "descriptive_comparison.csv", index=False)

# ============================================================
# 9. VISUALISATION: SIDE-BY-SIDE FEATURE COMPARISON
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

colors = {'2015': '#2196F3', '2024': '#F44336'}

for i, col in enumerate(final_cols[:8]):
    ax = axes[i]
    data15 = df_out15[col].dropna()
    data24 = df_out24[col].dropna()

    if data15.nunique() <= 2:
        vals = [data15.mean() * 100, data24.mean() * 100]
        bars = ax.bar(['2015', '2024'], vals,
                      color=[colors['2015'], colors['2024']],
                      width=0.5, edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, f"{val:.1f}%",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(0, 115)
        ax.set_ylabel('Percentage (%)', fontsize=9)
    else:
        ax.violinplot([data15, data24], positions=[1, 2], showmedians=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['2015', '2024'])
        ax.set_ylabel('Value', fontsize=9)

    ax.set_title(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Maternal Care Utilisation Features: MDHS 2015 vs 2024',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES / "01_feature_comparison_2015_vs_2024.png",
            dpi=300, bbox_inches='tight')
plt.close()
log("\n  ✓ Figure saved: 01_feature_comparison_2015_vs_2024.png")

# ============================================================
# 10. SAVE OUTPUTS
# ============================================================

df_out15.to_csv(RESULTS / "harmonised_2015.csv", index=False)
df_out24.to_csv(RESULTS / "harmonised_2024.csv", index=False)
df_pooled.to_csv(RESULTS / "harmonised_pooled.csv", index=False)

with open(RESULTS / "harmonisation_report.txt", 'w') as f:
    f.write('\n'.join(report_lines))

log("\n" + "=" * 70)
log("HARMONISATION COMPLETE")
log(f"  2015 analysis-ready: {len(df_out15):,} women, {len(final_cols)} clustering features")
log(f"  2024 analysis-ready: {len(df_out24):,} women, {len(final_cols)} clustering features")
log(f"  Clustering features: {final_cols}")
log("  Outputs saved to results/")
log("=" * 70)
