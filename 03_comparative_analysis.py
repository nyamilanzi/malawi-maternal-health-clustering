"""
=============================================================================
SCRIPT 03: COMPARATIVE ANALYSIS — MDHS 2015 vs 2024
=============================================================================
Compare cluster prevalences, profiles, and distributions between survey
years. Assess whether utilisation patterns have converged or diverged.

Methods:
- Cluster profile comparison (feature means ± SD)
- Hungarian algorithm to align clusters across years
- Proportional change in cluster membership
- Chi-square and t-tests for inter-year differences
- Shared-space PCA to visualise temporal shift

Output:
  results/cluster_alignment.csv
  results/temporal_comparison.csv
  figures/03_*.png
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_ind, chi2_contingency, ks_2samp
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
print("COMPARATIVE ANALYSIS: MDHS 2015 vs 2024")
print("=" * 70)

df15 = pd.read_csv(RESULTS / "clusters_2015.csv")
df24 = pd.read_csv(RESULTS / "clusters_2024.csv")
prof15 = pd.read_csv(RESULTS / "cluster_profiles_2015.csv")
prof24 = pd.read_csv(RESULTS / "cluster_profiles_2024.csv")

SOCIO_COLS = ['age', 'age_group', 'residence', 'wealth_quintile', 'education',
              'births_last5', 'parity', 'region', 'married', 'distance_problem',
              'survey_year', 'cluster', 'cluster_label']

feat_cols = [c for c in df15.columns
             if c not in SOCIO_COLS and not c.startswith('prob_cluster')]

print(f"\nClustering features: {feat_cols}")
print(f"2015: K={df15['cluster'].nunique()}, n={len(df15):,}")
print(f"2024: K={df24['cluster'].nunique()}, n={len(df24):,}")

# ============================================================
# 2. CLUSTER ALIGNMENT VIA HUNGARIAN ALGORITHM
# ============================================================

def align_clusters(prof_ref, prof_target, feat_cols):
    """
    Align target clusters to reference clusters to minimise total
    Euclidean distance between cluster centroids in feature space.
    Returns a mapping dict: target_cluster → ref_cluster
    """
    ref_mat = prof_ref.set_index('cluster')[feat_cols].values
    tgt_mat = prof_target.set_index('cluster')[feat_cols].values

    n_ref = ref_mat.shape[0]
    n_tgt = tgt_mat.shape[0]
    n = max(n_ref, n_tgt)

    # Cost matrix (Euclidean distances)
    cost = np.zeros((n, n))
    for i in range(n_ref):
        for j in range(n_tgt):
            cost[i, j] = np.linalg.norm(ref_mat[i] - tgt_mat[j])

    row_ind, col_ind = linear_sum_assignment(cost)

    ref_clusters = prof_ref['cluster'].values
    tgt_clusters = prof_target['cluster'].values

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < len(ref_clusters) and c < len(tgt_clusters):
            mapping[tgt_clusters[c]] = ref_clusters[r]
            print(f"  Target cluster {tgt_clusters[c]} → Ref cluster {ref_clusters[r]} "
                  f"(dist={cost[r,c]:.4f})")
    return mapping


print("\n--- Aligning 2024 clusters to 2015 clusters ---")
alignment_24_to_15 = align_clusters(prof15, prof24, feat_cols)
print(f"\n  Alignment map (2024 → 2015): {alignment_24_to_15}")

df24['cluster_aligned'] = df24['cluster'].map(alignment_24_to_15).fillna(df24['cluster'])

# ============================================================
# 3. CLUSTER PREVALENCE COMPARISON
# ============================================================

print("\n--- Cluster prevalence comparison ---")

prev15 = df15['cluster'].value_counts(normalize=True).sort_index() * 100
prev24 = df24['cluster_aligned'].value_counts(normalize=True).sort_index() * 100

prev_df = pd.DataFrame({
    'cluster_2015': prev15.index,
    'prevalence_2015': prev15.values,
    'prevalence_2024': [prev24.get(c, 0) for c in prev15.index],
    'label_2015': prof15.set_index('cluster').get('label', prev15.index).values
})
prev_df['abs_change'] = prev_df['prevalence_2024'] - prev_df['prevalence_2015']
prev_df['rel_change_pct'] = (prev_df['abs_change'] / prev_df['prevalence_2015'] * 100).round(1)

print("\nCluster prevalence changes (2015 → 2024):")
print(prev_df.to_string(index=False))

# ============================================================
# 4. FEATURE-LEVEL TEMPORAL COMPARISON (BY CLUSTER)
# ============================================================

print("\n--- Feature-level temporal comparison ---")

temporal_rows = []

for feat in feat_cols:
    v15 = df15[feat].dropna()
    v24 = df24[feat].dropna()

    if v15.nunique() <= 2:  # Binary
        p15 = v15.mean() * 100
        p24 = v24.mean() * 100
        # Chi-square
        ct = pd.crosstab(pd.concat([v15, v24], keys=['2015', '2024']).reset_index()['level_0'],
                         pd.concat([v15, v24]).values)
        chi2, pval, _, _ = chi2_contingency(ct)
        temporal_rows.append({
            'feature': feat, 'type': 'binary',
            'value_2015': round(p15, 2), 'value_2024': round(p24, 2),
            'change': round(p24 - p15, 2),
            'test': 'chi2', 'p_value': round(pval, 4)
        })
        print(f"  {feat}: 2015={p15:.1f}%  →  2024={p24:.1f}%  Δ={p24-p15:+.1f}pp  p={pval:.4f}")
    else:  # Continuous
        m15, m24 = v15.mean(), v24.mean()
        s15, s24 = v15.std(), v24.std()
        t_stat, pval = ttest_ind(v15, v24, equal_var=False)
        temporal_rows.append({
            'feature': feat, 'type': 'continuous',
            'value_2015': round(m15, 3), 'value_2024': round(m24, 3),
            'change': round(m24 - m15, 3),
            'test': 'welch_t', 'p_value': round(pval, 4)
        })
        print(f"  {feat}: 2015={m15:.2f}±{s15:.2f}  →  2024={m24:.2f}±{s24:.2f}  "
              f"Δ={m24-m15:+.3f}  p={pval:.4f}")

temporal_df = pd.DataFrame(temporal_rows)
temporal_df.to_csv(RESULTS / "temporal_comparison.csv", index=False)

# ============================================================
# 5. SHARED-SPACE PCA VISUALISATION
# ============================================================

def shared_pca_plot(df15, df24, feat_cols, out_path):
    """
    Project both years into shared PCA space to visualise
    distributional shift between 2015 and 2024.
    """
    X15 = df15[feat_cols].values.astype(float)
    X24 = df24[feat_cols].values.astype(float)
    X_all = np.vstack([X15, X24])

    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)
    X15_s = X_all_s[:len(X15)]
    X24_s = X_all_s[len(X15):]

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_all_s)
    pca15 = pca.transform(X15_s)
    pca24 = pca.transform(X24_s)

    var = pca.explained_variance_ratio_ * 100

    n_clusters_15 = df15['cluster'].nunique()
    n_clusters_24 = df24['cluster'].nunique()
    pal15 = sns.color_palette("Blues_d", n_clusters_15)
    pal24 = sns.color_palette("Reds_d", n_clusters_24)

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, c in enumerate(sorted(df15['cluster'].unique())):
        m = df15['cluster'] == c
        ax.scatter(pca15[m, 0], pca15[m, 1],
                   color=pal15[i], alpha=0.3, s=10, rasterized=True)
        cx, cy = pca15[m, 0].mean(), pca15[m, 1].mean()
        ax.scatter(cx, cy, color=pal15[i], s=180, marker='o',
                   edgecolors='navy', linewidth=1.2, zorder=6)
        ax.annotate(f'2015\nC{c}', (cx, cy), textcoords='offset points',
                    xytext=(8, 4), fontsize=8, color='navy', fontweight='bold')

    for i, c in enumerate(sorted(df24['cluster'].unique())):
        m = df24['cluster'] == c
        ax.scatter(pca24[m, 0], pca24[m, 1],
                   color=pal24[i], alpha=0.3, s=10, rasterized=True)
        cx, cy = pca24[m, 0].mean(), pca24[m, 1].mean()
        ax.scatter(cx, cy, color=pal24[i], s=180, marker='s',
                   edgecolors='darkred', linewidth=1.2, zorder=6)
        ax.annotate(f'2024\nC{c}', (cx, cy), textcoords='offset points',
                    xytext=(8, -12), fontsize=8, color='darkred', fontweight='bold')

    # Draw arrows from 2015 to aligned 2024 centroids if K matches
    if df15['cluster'].nunique() == df24['cluster'].nunique():
        for c15 in sorted(df15['cluster'].unique()):
            m15 = df15['cluster'] == c15
            c15_x, c15_y = pca15[m15, 0].mean(), pca15[m15, 1].mean()
            c24 = next((k for k, v in alignment_24_to_15.items() if v == c15), None)
            if c24 is not None:
                m24 = df24['cluster'] == c24
                c24_x, c24_y = pca24[m24, 0].mean(), pca24[m24, 1].mean()
                ax.annotate('', xy=(c24_x, c24_y), xytext=(c15_x, c15_y),
                            arrowprops=dict(arrowstyle='->', color='grey',
                                           lw=1.5, linestyle='dashed'))

    patch15 = mpatches.Patch(color=pal15[1], label='MDHS 2015 (circles)')
    patch24 = mpatches.Patch(color=pal24[1], label='MDHS 2024 (squares)')
    ax.legend(handles=[patch15, patch24], fontsize=9, loc='upper right')

    ax.set_xlabel(f'PC1 ({var[0]:.1f}% variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var[1]:.1f}% variance)', fontsize=11)
    ax.set_title('Shared PCA Space: Temporal Shift in Utilisation Patterns\n'
                 'MDHS 2015 vs 2024', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Figure: {out_path.name}")


shared_pca_plot(df15, df24, feat_cols, FIGURES / "03_shared_pca_temporal.png")

# ============================================================
# 6. PREVALENCE CHANGE BAR CHART
# ============================================================

def plot_prevalence_change(prev_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Absolute prevalences
    ax = axes[0]
    x = np.arange(len(prev_df))
    w = 0.35
    bars15 = ax.bar(x - w/2, prev_df['prevalence_2015'], w,
                    label='2015', color='#2196F3', edgecolor='white')
    bars24 = ax.bar(x + w/2, prev_df['prevalence_2024'], w,
                    label='2024', color='#F44336', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cluster {int(c)}' for c in prev_df['cluster_2015']],
                       rotation=15)
    ax.set_ylabel('Prevalence (%)', fontsize=11)
    ax.set_title('Cluster Prevalence by Survey Year', fontsize=11, fontweight='bold')
    ax.legend(title='Survey year', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in list(bars15) + list(bars24):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5, f"{bar.get_height():.1f}%",
                ha='center', va='bottom', fontsize=8)

    # Absolute change
    ax2 = axes[1]
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in prev_df['abs_change']]
    bars = ax2.bar(x, prev_df['abs_change'], color=colors, edgecolor='white')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Cluster {int(c)}' for c in prev_df['cluster_2015']],
                        rotation=15)
    ax2.set_ylabel('Absolute Change in Prevalence (pp)', fontsize=11)
    ax2.set_title('Change in Cluster Prevalence (2024 − 2015)', fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for bar in bars:
        va = 'bottom' if bar.get_height() >= 0 else 'top'
        offset = 0.2 if bar.get_height() >= 0 else -0.2
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + offset,
                 f"{bar.get_height():+.1f}", ha='center', va=va, fontsize=9)

    plt.suptitle('Temporal Changes in Maternal Care Utilisation Patterns',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure: {out_path.name}")


plot_prevalence_change(prev_df, FIGURES / "03_prevalence_change.png")

# ============================================================
# 7. FEATURE TEMPORAL CHANGE FOREST PLOT
# ============================================================

def plot_feature_change(temporal_df, out_path):
    df = temporal_df.copy()
    df['sig'] = df['p_value'] < 0.05
    df['label'] = df['feature'].str.replace('_', ' ').str.title()

    # Separate binary and continuous
    bin_df = df[df['type'] == 'binary'].sort_values('change', ascending=True)
    con_df = df[df['type'] == 'continuous'].sort_values('change', ascending=True)

    n_rows = max(len(bin_df), len(con_df))
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n_rows * 0.7 + 2)))

    for ax, sub_df, title, unit in [
        (axes[0], bin_df, 'Binary Indicators', 'percentage points'),
        (axes[1], con_df, 'Continuous Measures', 'units')
    ]:
        if sub_df.empty:
            ax.set_visible(False)
            continue
        colors = ['#4CAF50' if v > 0 else '#F44336' for v in sub_df['change']]
        bars = ax.barh(sub_df['label'], sub_df['change'], color=colors,
                       edgecolor='white', height=0.6)
        ax.axvline(0, color='black', linewidth=0.8)
        for bar, (_, row) in zip(bars, sub_df.iterrows()):
            x = bar.get_width()
            ha = 'left' if x >= 0 else 'right'
            offset = 0.003 * sub_df['change'].abs().max()
            ax.text(x + offset if x >= 0 else x - offset,
                    bar.get_y() + bar.get_height()/2,
                    ('*' if row['sig'] else '') + f" {x:+.2f}",
                    ha=ha, va='center', fontsize=8,
                    color='black')
        ax.set_xlabel(f'Change 2024 − 2015 ({unit})', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=9)

    plt.suptitle('Feature-Level Change: MDHS 2015 → 2024\n(* p<0.05)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure: {out_path.name}")


plot_feature_change(temporal_df, FIGURES / "03_feature_temporal_change.png")

# ============================================================
# 8. SIDE-BY-SIDE PROFILE COMPARISON HEATMAP
# ============================================================

def plot_combined_profiles(prof15, prof24, feat_cols, out_path):
    """Combined heatmap: 2015 clusters | 2024 clusters side by side."""
    data15 = prof15.set_index('cluster')[feat_cols].T
    data24 = prof24.set_index('cluster')[feat_cols].T

    data15.columns = [f'2015\nC{c}' for c in data15.columns]
    data24.columns = [f'2024\nC{c}' for c in data24.columns]

    combined = pd.concat([data15, data24], axis=1)

    fig, ax = plt.subplots(figsize=(max(10, combined.shape[1] * 1.5), 8))
    sns.heatmap(combined.astype(float), annot=True, fmt='.2f',
                cmap='YlOrRd', vmin=0, vmax=1,
                linewidths=0.5, linecolor='white',
                ax=ax, cbar_kws={'label': 'Mean value'})
    ax.set_yticklabels([c.replace('_', ' ').title() for c in combined.index],
                       rotation=0, fontsize=9)
    ax.set_title('Cluster Profiles Comparison: MDHS 2015 vs 2024\n'
                 '(values are means; binary=proportions, continuous=scaled means)',
                 fontsize=11, fontweight='bold')

    # Add vertical line separating 2015 and 2024
    ax.axvline(data15.shape[1], color='black', linewidth=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure: {out_path.name}")


plot_combined_profiles(prof15, prof24, feat_cols,
                       FIGURES / "03_combined_profiles_heatmap.png")

# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

alignment_df = pd.DataFrame([
    {'cluster_2024': k, 'aligned_to_cluster_2015': v}
    for k, v in alignment_24_to_15.items()
])
alignment_df.to_csv(RESULTS / "cluster_alignment.csv", index=False)
prev_df.to_csv(RESULTS / "cluster_prevalence_comparison.csv", index=False)

print("\n" + "=" * 70)
print("COMPARATIVE ANALYSIS COMPLETE")
print("  Outputs: results/temporal_comparison.csv")
print("           results/cluster_alignment.csv")
print("           results/cluster_prevalence_comparison.csv")
print("=" * 70)
