"""
=============================================================================
SCRIPT 04: SUBGROUP ANALYSIS
=============================================================================
Examine how cluster distributions differ within demographic subgroups:
  - Residence (urban / rural)
  - Wealth quintile (poorest → richest)
  - Age group (adolescent 15–19 | young adult 20–34 | older 35–49)

For each subgroup × year combination, compute:
  - Cluster prevalences
  - Chi-square test for independence (subgroup × cluster)
  - Cluster profiles within subgroup

Output:
  results/subgroup_*.csv
  figures/04_*.png
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

BASE = Path(__file__).parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("SUBGROUP ANALYSIS")
print("=" * 70)

df15 = pd.read_csv(RESULTS / "clusters_2015.csv")
df24 = pd.read_csv(RESULTS / "clusters_2024.csv")

SOCIO_COLS = ['age', 'age_group', 'residence', 'wealth_quintile', 'education',
              'births_last5', 'parity', 'region', 'married', 'distance_problem',
              'survey_year', 'cluster', 'cluster_label']

feat_cols = [c for c in df15.columns
             if c not in SOCIO_COLS and not c.startswith('prob_cluster')]

# Pool datasets
df_all = pd.concat([df15.assign(year='2015'), df24.assign(year='2024')],
                   ignore_index=True)

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def cluster_prevalence_by_subgroup(df, subgroup_col, year_col='year'):
    """Compute cluster % within each subgroup × year combination."""
    if subgroup_col not in df.columns:
        return None
    df_valid = df[[subgroup_col, year_col, 'cluster']].dropna()
    cross = pd.crosstab(
        index=[df_valid[subgroup_col], df_valid[year_col]],
        columns=df_valid['cluster'],
        normalize='index'
    ) * 100
    cross.index.names = [subgroup_col, 'year']
    return cross.reset_index()


def chi2_by_subgroup_year(df, subgroup_col, year_col='year'):
    """Chi-square test: subgroup × cluster, within each year."""
    results = []
    if subgroup_col not in df.columns:
        return pd.DataFrame()
    for year in sorted(df[year_col].dropna().unique()):
        sub = df[df[year_col] == year][[subgroup_col, 'cluster']].dropna()
        ct = pd.crosstab(sub[subgroup_col], sub['cluster'])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(ct)
        cramers_v = np.sqrt(chi2 / (len(sub) * (min(ct.shape) - 1)))
        results.append({
            'subgroup_var': subgroup_col, 'year': year,
            'chi2': round(chi2, 2), 'df': dof,
            'p_value': round(p, 5),
            'cramers_v': round(cramers_v, 4),
            'n': len(sub)
        })
    return pd.DataFrame(results)


# ============================================================
# 3. DEFINE SUBGROUPS
# ============================================================

SUBGROUPS = {
    'residence': {
        'label': 'Residence',
        'order': ['Urban', 'Rural'],
        'color_palette': 'Set1'
    },
    'wealth_quintile': {
        'label': 'Wealth Quintile',
        'order': ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest'],
        'color_palette': 'RdYlGn'
    },
    'age_group': {
        'label': 'Age Group',
        'order': ['Adolescent (15–19)', 'Young adult (20–34)', 'Older (35–49)'],
        'color_palette': 'Set2'
    },
    'education': {
        'label': 'Education Level',
        'order': ['No education', 'Primary', 'Secondary', 'Higher'],
        'color_palette': 'Blues'
    }
}

all_chi2 = []
all_prevalences = {}

for sg_col, sg_info in SUBGROUPS.items():
    print(f"\n--- Subgroup: {sg_info['label']} ---")
    prev = cluster_prevalence_by_subgroup(df_all, sg_col)
    chi2_res = chi2_by_subgroup_year(df_all, sg_col)

    if prev is not None:
        all_prevalences[sg_col] = prev
        prev.to_csv(RESULTS / f"subgroup_prevalence_{sg_col}.csv", index=False)
        print(prev.to_string(index=False))

    if not chi2_res.empty:
        all_chi2.append(chi2_res)
        print(chi2_res.to_string(index=False))

if all_chi2:
    chi2_all = pd.concat(all_chi2, ignore_index=True)
    chi2_all.to_csv(RESULTS / "subgroup_chi2_tests.csv", index=False)

# ============================================================
# 4. STACKED BAR CHARTS BY SUBGROUP × YEAR
# ============================================================

N_CLUSTERS_15 = df15['cluster'].nunique()
N_CLUSTERS_24 = df24['cluster'].nunique()
N_CLUSTERS = max(N_CLUSTERS_15, N_CLUSTERS_24)
CLUSTER_COLORS = sns.color_palette("Set2", N_CLUSTERS)


def plot_stacked_bars_subgroup(prev_df, sg_col, sg_info, feat_cols,
                               out_path, years=['2015', '2024']):
    """
    Stacked bar chart of cluster prevalences by subgroup × year.
    """
    if prev_df is None or prev_df.empty:
        return

    order = [o for o in sg_info['order']
             if o in prev_df[sg_col].values]
    cluster_cols = [c for c in prev_df.columns
                    if c not in [sg_col, 'year'] and str(c).isdigit() or
                    (isinstance(c, (int, float)) and not pd.isna(c))]
    cluster_cols = sorted([c for c in prev_df.columns
                           if c not in [sg_col, 'year']])

    n_groups = len(order)
    n_years = len(years)
    width = 0.35
    x = np.arange(n_groups)

    fig, axes = plt.subplots(1, n_years, figsize=(n_groups * 2.5 * n_years, 7),
                             sharey=True)
    if n_years == 1:
        axes = [axes]

    for ax, year in zip(axes, years):
        sub = prev_df[prev_df['year'] == str(year)].set_index(sg_col)
        sub = sub.reindex(order)

        bottom = np.zeros(len(order))
        for j, col in enumerate(cluster_cols):
            vals = sub[col].fillna(0).values
            color = CLUSTER_COLORS[j % len(CLUSTER_COLORS)]
            bars = ax.bar(x, vals, bottom=bottom, label=f'Cluster {col}',
                          color=color, edgecolor='white', linewidth=0.8)
            # Label clusters >8% for readability
            for k, (v, b) in enumerate(zip(vals, bottom)):
                if v > 8:
                    ax.text(k, b + v / 2, f'{v:.0f}%',
                            ha='center', va='center', fontsize=8,
                            color='white', fontweight='bold')
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Cluster Prevalence (%)', fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_title(f'MDHS {year}', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title='Cluster', loc='lower center',
               ncol=N_CLUSTERS, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))

    plt.suptitle(f'Cluster Prevalence by {sg_info["label"]}: MDHS 2015 vs 2024',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure: {out_path.name}")


for sg_col, sg_info in SUBGROUPS.items():
    if sg_col in all_prevalences:
        plot_stacked_bars_subgroup(
            all_prevalences[sg_col], sg_col, sg_info, feat_cols,
            FIGURES / f"04_stacked_{sg_col}.png"
        )

# ============================================================
# 5. HEATMAP: SUBGROUP × CLUSTER PREVALENCE (BOTH YEARS)
# ============================================================

def plot_subgroup_heatmap(df_all, sg_col, sg_info, out_path):
    """
    Heatmap: rows = subgroup categories × year, columns = clusters.
    """
    if sg_col not in df_all.columns:
        return

    df_valid = df_all[[sg_col, 'year', 'cluster']].dropna()
    order = [o for o in sg_info['order'] if o in df_valid[sg_col].values]

    rows = []
    for yr in ['2015', '2024']:
        for cat in order:
            sub = df_valid[(df_valid['year'] == yr) & (df_valid[sg_col] == cat)]
            if len(sub) == 0:
                continue
            counts = sub['cluster'].value_counts(normalize=True) * 100
            row = {'subgroup': f"{cat}\n({yr})", 'year': yr, 'category': cat}
            for c in sorted(df_valid['cluster'].unique()):
                row[f'C{int(c)}'] = counts.get(c, 0)
            rows.append(row)

    heat_df = pd.DataFrame(rows).set_index('subgroup')
    c_cols = [c for c in heat_df.columns if c.startswith('C')]
    heat_data = heat_df[c_cols].astype(float)

    # Colour alternating rows for year distinction
    row_colors = [('#BBDEFB' if yr == '2015' else '#FFCDD2')
                  for yr in heat_df['year']]

    fig, ax = plt.subplots(figsize=(max(8, len(c_cols) * 2), len(rows) * 0.55 + 2))
    im = sns.heatmap(heat_data, annot=True, fmt='.0f',
                     cmap='YlOrRd', vmin=0, vmax=80,
                     linewidths=0.5, linecolor='white', ax=ax,
                     cbar_kws={'label': 'Cluster prevalence (%)'})

    ax.set_xlabel('Cluster', fontsize=10)
    ax.set_ylabel(sg_info['label'], fontsize=10)
    ax.set_title(f'Cluster Prevalence by {sg_info["label"]} and Survey Year (%)',
                 fontsize=11, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    # Add year colour bands on left margin
    for i, (row_label, color) in enumerate(zip(heat_df.index, row_colors)):
        ax.add_patch(plt.Rectangle((-0.4, i), 0.3, 1,
                                   color=color, transform=ax.transData,
                                   clip_on=False, zorder=3))

    patch15 = mpatches.Patch(color='#BBDEFB', label='2015')
    patch24 = mpatches.Patch(color='#FFCDD2', label='2024')
    ax.legend(handles=[patch15, patch24], title='Survey year',
              loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure: {out_path.name}")


import matplotlib.patches as mpatches

for sg_col, sg_info in SUBGROUPS.items():
    plot_subgroup_heatmap(
        df_all, sg_col, sg_info,
        FIGURES / f"04_heatmap_{sg_col}.png"
    )

# ============================================================
# 6. EFFECT SIZE SUMMARY — CRAMÉR'S V BY SUBGROUP × YEAR
# ============================================================

if not chi2_all.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    chi2_all['year'] = chi2_all['year'].astype(str)
    chi2_pivot = chi2_all.pivot(index='subgroup_var', columns='year', values='cramers_v')

    x = np.arange(len(chi2_pivot))
    w = 0.35
    bars15 = ax.bar(x - w/2, chi2_pivot.get('2015', [0]*len(x)),
                    w, label='2015', color='#2196F3', edgecolor='white')
    bars24 = ax.bar(x + w/2, chi2_pivot.get('2024', [0]*len(x)),
                    w, label='2024', color='#F44336', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([SUBGROUPS[v]['label'] for v in chi2_pivot.index], fontsize=10)
    ax.set_ylabel("Cramér's V (association strength)", fontsize=10)
    ax.set_title("Association Between Subgroup and Cluster Membership\n"
                 "(Cramér's V: 0=none, 0.1=small, 0.3=medium, 0.5=large)",
                 fontsize=11, fontweight='bold')
    ax.legend(title='Survey year', fontsize=9)
    ax.set_ylim(0, 0.55)
    ax.axhline(0.1, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(0.3, color='grey', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars15, bars24]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES / "04_cramersv_subgroups.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Figure: 04_cramersv_subgroups.png")

# ============================================================
# 7. FEATURE PROFILES WITHIN KEY SUBGROUPS
# ============================================================

def plot_profiles_within_subgroup(df, sg_col, sg_info, feat_cols,
                                   year, out_path):
    """Radar / bar chart of feature means by cluster within each subgroup."""
    if sg_col not in df.columns:
        return

    sub = df[df['year'] == year].copy()
    categories = [c for c in sg_info['order'] if c in sub[sg_col].dropna().values]

    fig, axes = plt.subplots(
        len(categories), len(feat_cols),
        figsize=(len(feat_cols) * 2.2, len(categories) * 2),
        sharex='col'
    )
    if len(categories) == 1:
        axes = [axes]

    pal = sns.color_palette("Set2", sub['cluster'].nunique())

    for i, cat in enumerate(categories):
        cat_sub = sub[sub[sg_col] == cat]
        for j, feat in enumerate(feat_cols):
            ax = axes[i][j] if len(feat_cols) > 1 else axes[i]
            means = cat_sub.groupby('cluster')[feat].mean()
            bars = ax.bar(means.index.astype(str), means.values,
                          color=[pal[k] for k in range(len(means))],
                          edgecolor='white')
            ax.set_ylim(0, 1.1 if means.max() <= 1 else means.max() * 1.2)
            if i == 0:
                ax.set_title(feat.replace('_', '\n'), fontsize=8, fontweight='bold')
            if j == 0:
                ax.set_ylabel(cat, fontsize=8, rotation=0, labelpad=60, va='center')
            ax.tick_params(labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.suptitle(f'Feature Means by Cluster within {sg_info["label"]} — MDHS {year}',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure: {out_path.name}")


# Plot for residence and age group (most policy-relevant)
for sg_col in ['residence', 'age_group']:
    sg_info = SUBGROUPS[sg_col]
    for year in ['2015', '2024']:
        plot_profiles_within_subgroup(
            df_all, sg_col, sg_info, feat_cols, year,
            FIGURES / f"04_profiles_{sg_col}_{year}.png"
        )

print("\n" + "=" * 70)
print("SUBGROUP ANALYSIS COMPLETE")
print("  All outputs saved to results/ and figures/")
print("=" * 70)
