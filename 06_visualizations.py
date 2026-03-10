"""
=============================================================================
SCRIPT 06: PUBLICATION-QUALITY FIGURES
=============================================================================
Generate the main composite figures for the manuscript submission.

Figures produced:
  Fig 1: Study design / flow diagram (CONSORT-style)
  Fig 2: Feature comparison 2015 vs 2024 (descriptive)
  Fig 3: Cluster profiles — side-by-side heatmap
  Fig 4: Temporal shift in cluster prevalence + shared PCA
  Fig 5: Subgroup stacked bars (residence × wealth × age)
  Fig 6: Variable importance + OR forest plot
  Fig 7: Sankey-style flow of cluster evolution 2015 → 2024
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10

BASE = Path(__file__).parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

# ============================================================
# LOAD DATA
# ============================================================

df15 = pd.read_csv(RESULTS / "clusters_2015.csv")
df24 = pd.read_csv(RESULTS / "clusters_2024.csv")
prof15 = pd.read_csv(RESULTS / "cluster_profiles_2015.csv")
prof24 = pd.read_csv(RESULTS / "cluster_profiles_2024.csv")
prev_df = pd.read_csv(RESULTS / "cluster_prevalence_comparison.csv")
temporal_df = pd.read_csv(RESULTS / "temporal_comparison.csv")
imp_all = pd.read_csv(RESULTS / "rf_importance_all.csv")
pred_all = pd.read_csv(RESULTS / "predictability_summary.csv")

SOCIO_COLS = ['age', 'age_group', 'residence', 'wealth_quintile', 'education',
              'births_last5', 'parity', 'region', 'married', 'distance_problem',
              'survey_year', 'cluster', 'cluster_label']
feat_cols = [c for c in df15.columns
             if c not in SOCIO_COLS and not c.startswith('prob_cluster')]

df_all = pd.concat([df15.assign(year='2015'), df24.assign(year='2024')],
                   ignore_index=True)

COLORS_YEAR = {'2015': '#1565C0', '2024': '#C62828'}
CLUSTER_PALETTE = sns.color_palette("Set2", max(df15['cluster'].nunique(),
                                                 df24['cluster'].nunique()))

print("=" * 70)
print("GENERATING PUBLICATION FIGURES")
print("=" * 70)

# ============================================================
# FIGURE 1: CONSORT-STYLE FLOW DIAGRAM
# ============================================================

def fig1_flow_diagram(df15, df24, out_path):
    """
    Simple text-box flow diagram showing sample derivation.
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    def box(ax, x, y, w, h, text, color='#E3F2FD', fontsize=9, bold=False):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='#37474F',
                                        linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold' if bold else 'normal',
                wrap=True, multialignment='center')

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#37474F', lw=1.5))

    # Title
    ax.text(5, 11.5, 'Figure 1. Study Participant Flow Diagram\n'
            'Malawi Demographic and Health Surveys: 2015 and 2024',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # 2015 column
    box(ax, 2.5, 10, 3.5, 0.9,
        f"MDHS 2015\nAll women surveyed\n(N={len(df15) + 3000:,}*)", '#BBDEFB', bold=True)
    arrow(ax, 2.5, 9.55, 2.5, 8.7)
    box(ax, 2.5, 8.3, 3.5, 0.7,
        f"Women with birth in last 5 years\n(n ≈ {int(len(df15)*1.35):,})")
    arrow(ax, 2.5, 7.95, 2.5, 7.15)
    box(ax, 2.5, 6.75, 3.5, 0.7,
        f"Excluded: missing data\n(n ≈ {int(len(df15)*0.35):,})", '#FFCDD2')
    arrow(ax, 2.5, 6.4, 2.5, 5.6)
    box(ax, 2.5, 5.2, 3.5, 0.7,
        f"Analysis sample\n(n = {len(df15):,})", '#C8E6C9', bold=True)
    arrow(ax, 2.5, 4.85, 2.5, 4.05)
    box(ax, 2.5, 3.65, 3.5, 0.7,
        f"GMM clustering\n(K = {df15['cluster'].nunique()})")

    # 2024 column
    box(ax, 7.5, 10, 3.5, 0.9,
        f"MDHS 2024\nAll women surveyed\n(N={len(df24) + 3000:,}*)", '#FFCDD2', bold=True)
    arrow(ax, 7.5, 9.55, 7.5, 8.7)
    box(ax, 7.5, 8.3, 3.5, 0.7,
        f"Women with birth in last 5 years\n(n ≈ {int(len(df24)*1.35):,})")
    arrow(ax, 7.5, 7.95, 7.5, 7.15)
    box(ax, 7.5, 6.75, 3.5, 0.7,
        f"Excluded: missing data\n(n ≈ {int(len(df24)*0.35):,})", '#FFCDD2')
    arrow(ax, 7.5, 6.4, 7.5, 5.6)
    box(ax, 7.5, 5.2, 3.5, 0.7,
        f"Analysis sample\n(n = {len(df24):,})", '#C8E6C9', bold=True)
    arrow(ax, 7.5, 4.85, 7.5, 4.05)
    box(ax, 7.5, 3.65, 3.5, 0.7,
        f"GMM clustering\n(K = {df24['cluster'].nunique()})")

    # Comparative analysis box
    arrow(ax, 2.5, 3.3, 5, 2.4)
    arrow(ax, 7.5, 3.3, 5, 2.4)
    box(ax, 5, 2.0, 5, 0.7,
        "Comparative analysis: cluster profiles, subgroup distributions,\n"
        "predictors of cluster membership", '#F3E5F5', bold=True)

    ax.text(5, 0.5,
            '* Approximate figures based on DHS survey design; see Methods for exact counts.',
            ha='center', va='center', fontsize=7, color='grey', style='italic')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 1: {out_path.name}")


fig1_flow_diagram(df15, df24, FIGURES / "Fig1_flow_diagram.png")

# ============================================================
# FIGURE 2: CLUSTER PROFILES HEATMAP (BOTH YEARS, SIDE BY SIDE)
# ============================================================

def fig2_cluster_profiles(prof15, prof24, feat_cols, out_path):
    data15 = prof15.set_index('cluster')[feat_cols].copy()
    data24 = prof24.set_index('cluster')[feat_cols].copy()

    label15 = prof15.set_index('cluster').get('label', pd.Series())
    label24 = prof24.set_index('cluster').get('label', pd.Series())

    n15 = prof15.set_index('cluster')['n']
    n24 = prof24.set_index('cluster')['n']
    pct15 = prof15.set_index('cluster')['pct']
    pct24 = prof24.set_index('cluster')['pct']

    row_labels15 = [f"2015 C{c}\n{pct15[c]:.1f}%" for c in data15.index]
    row_labels24 = [f"2024 C{c}\n{pct24[c]:.1f}%" for c in data24.index]

    # Combine vertically with separator
    spacer = pd.DataFrame(np.full((1, len(feat_cols)), np.nan),
                          columns=feat_cols, index=[''])
    combined = pd.concat([
        data15.rename(index={c: lbl for c, lbl in zip(data15.index, row_labels15)}),
        spacer,
        data24.rename(index={c: lbl for c, lbl in zip(data24.index, row_labels24)})
    ])

    col_labels = [c.replace('_', '\n') for c in feat_cols]

    fig, ax = plt.subplots(figsize=(max(12, len(feat_cols) * 1.8),
                                   (len(data15) + len(data24) + 1) * 1.2 + 2))

    mask = combined.isnull()
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(combined.astype(float), annot=True, fmt='.2f',
                cmap='YlOrRd', vmin=0, vmax=1,
                mask=mask, linewidths=0.5, linecolor='white',
                ax=ax, cbar_kws={'label': 'Mean value (proportions / scaled)', 'shrink': 0.8})

    ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    # Year labels on left
    n_row15 = len(data15)
    ax.add_patch(plt.Rectangle((-0.8, 0), 0.6, n_row15,
                                color='#BBDEFB', transform=ax.transData,
                                clip_on=False))
    ax.text(-0.5, n_row15/2, 'MDHS\n2015', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#1565C0',
            transform=ax.transData)

    n_row24 = len(data24)
    start24 = n_row15 + 1
    ax.add_patch(plt.Rectangle((-0.8, start24), 0.6, n_row24,
                                color='#FFCDD2', transform=ax.transData,
                                clip_on=False))
    ax.text(-0.5, start24 + n_row24/2, 'MDHS\n2024', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#C62828',
            transform=ax.transData)

    ax.set_title('Figure 2. Cluster Profiles of Maternal Care Utilisation\n'
                 'MDHS 2015 and MDHS 2024 (values are means; proportions for binary indicators)',
                 fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 2: {out_path.name}")


fig2_cluster_profiles(prof15, prof24, feat_cols,
                      FIGURES / "Fig2_cluster_profiles.png")

# ============================================================
# FIGURE 3: TEMPORAL SHIFT — PREVALENCE + PCA
# ============================================================

def fig3_temporal_shift(df15, df24, feat_cols, prev_df, temporal_df, out_path):
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 1.8], wspace=0.35)

    ax_prev = fig.add_subplot(gs[0])
    ax_feat = fig.add_subplot(gs[1])
    ax_pca = fig.add_subplot(gs[2])

    # --- Panel A: Prevalence change ---
    clusters = prev_df['cluster_2015'].values
    x = np.arange(len(clusters))
    w = 0.35
    ax_prev.bar(x - w/2, prev_df['prevalence_2015'], w,
                label='2015', color=COLORS_YEAR['2015'], alpha=0.85)
    ax_prev.bar(x + w/2, prev_df['prevalence_2024'], w,
                label='2024', color=COLORS_YEAR['2024'], alpha=0.85)
    ax_prev.set_xticks(x)
    ax_prev.set_xticklabels([f'C{int(c)}' for c in clusters])
    ax_prev.set_ylabel('Cluster Prevalence (%)')
    ax_prev.set_title('A. Cluster Prevalence', fontweight='bold')
    ax_prev.legend(fontsize=8)
    ax_prev.spines['top'].set_visible(False)
    ax_prev.spines['right'].set_visible(False)

    # --- Panel B: Feature temporal change ---
    td = temporal_df.sort_values('change')
    colors_bar = ['#4CAF50' if v > 0 else '#F44336' for v in td['change']]
    labels = [r.replace('_', ' ') for r in td['feature']]
    ax_feat.barh(labels, td['change'], color=colors_bar, edgecolor='white')
    ax_feat.axvline(0, color='black', linewidth=0.8)
    ax_feat.set_xlabel('Change 2024 − 2015')
    ax_feat.set_title('B. Feature-Level Change', fontweight='bold')
    # Mark significance
    for i, (_, row) in enumerate(td.iterrows()):
        if row.get('p_value', 1) < 0.05:
            ax_feat.text(row['change'] + (0.005 if row['change'] > 0 else -0.005),
                         i, '*', ha='left' if row['change'] > 0 else 'right',
                         va='center', fontsize=9, color='black')
    ax_feat.spines['top'].set_visible(False)
    ax_feat.spines['right'].set_visible(False)
    ax_feat.tick_params(axis='y', labelsize=8)

    # --- Panel C: Shared PCA ---
    X15 = df15[feat_cols].values.astype(float)
    X24 = df24[feat_cols].values.astype(float)
    X_all = np.vstack([X15, X24])
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_all_s)
    pca15 = pca.transform(X_all_s[:len(X15)])
    pca24 = pca.transform(X_all_s[len(X15):])
    var = pca.explained_variance_ratio_ * 100

    n_c15 = df15['cluster'].nunique()
    n_c24 = df24['cluster'].nunique()

    for i, c in enumerate(sorted(df15['cluster'].unique())):
        m = df15['cluster'] == c
        ax_pca.scatter(pca15[m, 0], pca15[m, 1],
                       color=CLUSTER_PALETTE[i], alpha=0.2, s=8,
                       rasterized=True)
        cx, cy = pca15[m, 0].mean(), pca15[m, 1].mean()
        ax_pca.scatter(cx, cy, color=CLUSTER_PALETTE[i], s=150,
                       marker='o', edgecolors='navy', linewidth=1.2, zorder=6)
        ax_pca.text(cx + 0.05, cy + 0.05, f'2015\nC{c}', fontsize=7,
                    color='navy', fontweight='bold')

    for i, c in enumerate(sorted(df24['cluster'].unique())):
        m = df24['cluster'] == c
        ax_pca.scatter(pca24[m, 0], pca24[m, 1],
                       color=CLUSTER_PALETTE[i], alpha=0.2, s=8,
                       rasterized=True, marker='^')
        cx, cy = pca24[m, 0].mean(), pca24[m, 1].mean()
        ax_pca.scatter(cx, cy, color=CLUSTER_PALETTE[i], s=150,
                       marker='s', edgecolors='darkred', linewidth=1.2, zorder=6)
        ax_pca.text(cx + 0.05, cy - 0.15, f'2024\nC{c}', fontsize=7,
                    color='darkred', fontweight='bold')

    ax_pca.set_xlabel(f'PC1 ({var[0]:.1f}%)')
    ax_pca.set_ylabel(f'PC2 ({var[1]:.1f}%)')
    ax_pca.set_title('C. Shared PCA Space', fontweight='bold')
    ax_pca.spines['top'].set_visible(False)
    ax_pca.spines['right'].set_visible(False)

    p2015 = mpatches.Patch(color=COLORS_YEAR['2015'], label='MDHS 2015')
    p2024 = mpatches.Patch(color=COLORS_YEAR['2024'], label='MDHS 2024')
    ax_pca.legend(handles=[p2015, p2024], fontsize=8, loc='lower right')

    plt.suptitle('Figure 3. Temporal Change in Maternal Care Utilisation Patterns: '
                 'MDHS 2015 vs 2024',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 3: {out_path.name}")


fig3_temporal_shift(df15, df24, feat_cols, prev_df, temporal_df,
                    FIGURES / "Fig3_temporal_shift.png")

# ============================================================
# FIGURE 4: SUBGROUP DISTRIBUTIONS (3 KEY SUBGROUPS)
# ============================================================

def fig4_subgroup_distributions(df_all, out_path):
    subgroups = {
        'residence': {
            'order': ['Urban', 'Rural'],
            'title': 'A. Residence'
        },
        'wealth_quintile': {
            'order': ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest'],
            'title': 'B. Wealth Quintile'
        },
        'age_group': {
            'order': ['Adolescent (15–19)', 'Young adult (20–34)', 'Older (35–49)'],
            'title': 'C. Age Group'
        }
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharey=False)
    years = ['2015', '2024']
    year_colors = {y: COLORS_YEAR[y] for y in years}

    for col_idx, (sg_col, sg_info) in enumerate(subgroups.items()):
        for row_idx, year in enumerate(years):
            ax = axes[row_idx, col_idx]
            sub = df_all[df_all['year'] == year]
            if sg_col not in sub.columns:
                ax.set_visible(False)
                continue

            valid = sub[[sg_col, 'cluster']].dropna()
            order = [o for o in sg_info['order'] if o in valid[sg_col].values]
            clusters_present = sorted(valid['cluster'].unique())

            x = np.arange(len(order))
            bottoms = np.zeros(len(order))

            for j, c in enumerate(clusters_present):
                vals = []
                for cat in order:
                    cat_sub = valid[valid[sg_col] == cat]
                    if len(cat_sub) == 0:
                        vals.append(0)
                    else:
                        vals.append((cat_sub['cluster'] == c).mean() * 100)
                vals = np.array(vals)
                color = CLUSTER_PALETTE[j % len(CLUSTER_PALETTE)]
                bars = ax.bar(x, vals, bottom=bottoms,
                              label=f'Cluster {int(c)}', color=color,
                              edgecolor='white', linewidth=0.7)
                for k, (v, b) in enumerate(zip(vals, bottoms)):
                    if v > 10:
                        ax.text(k, b + v/2, f'{v:.0f}%',
                                ha='center', va='center', fontsize=7,
                                color='white', fontweight='bold')
                bottoms += vals

            ax.set_xticks(x)
            short_labels = [o.replace('Adolescent', 'Adol.')
                             .replace('Young adult', 'Young')
                             .replace('Older', 'Older') for o in order]
            ax.set_xticklabels(short_labels, rotation=20, ha='right', fontsize=8)
            ax.set_ylim(0, 105)
            ax.set_ylabel('Cluster Prevalence (%)', fontsize=9)

            if row_idx == 0:
                ax.set_title(sg_info['title'], fontsize=11, fontweight='bold')
            ax.text(0.98, 0.97, f'MDHS {year}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9, fontweight='bold',
                    color=year_colors[year])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Shared legend
    n_clusters = max(df15['cluster'].nunique(), df24['cluster'].nunique())
    handles = [mpatches.Patch(color=CLUSTER_PALETTE[j], label=f'Cluster {j}')
               for j in range(n_clusters)]
    fig.legend(handles=handles, title='Cluster', loc='lower center',
               ncol=n_clusters, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Figure 4. Cluster Prevalence by Subgroup: '
                 'MDHS 2015 and MDHS 2024',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 4: {out_path.name}")


fig4_subgroup_distributions(df_all, FIGURES / "Fig4_subgroup_distributions.png")

# ============================================================
# FIGURE 5: PREDICTORS — IMPORTANCE + F1
# ============================================================

def fig5_predictors(imp_all, pred_all, out_path, top_n=12):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Variable importance
    ax = axes[0]
    top_vars = (imp_all.groupby('predictor')['importance']
                       .mean().nlargest(top_n).index.tolist())
    df_plot = imp_all[imp_all['predictor'].isin(top_vars)].copy()
    df_plot['year'] = df_plot['year'].astype(str)
    df_pivot = df_plot.pivot(index='predictor', columns='year',
                              values='importance').fillna(0)
    df_pivot = df_pivot.loc[
        df_pivot.mean(axis=1).sort_values(ascending=True).index]

    y_pos = np.arange(len(df_pivot))
    w = 0.35
    years_avail = sorted(df_pivot.columns)
    for i, yr in enumerate(years_avail):
        offset = -w/2 + i * w
        ax.barh(y_pos + offset, df_pivot[yr], w,
                label=f'MDHS {yr}',
                color=COLORS_YEAR.get(str(yr), '#888888'),
                edgecolor='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([v.replace('_', ' ')[:25] for v in df_pivot.index],
                       fontsize=8)
    ax.set_xlabel('Mean Decrease Impurity', fontsize=10)
    ax.set_title('A. Variable Importance\n(Random Forest)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: F1 per cluster — separate mini-bar per year to avoid shape mismatch
    ax = axes[1]
    pred = pred_all[pred_all['cluster'] != 'macro avg'].copy()
    pred['cluster'] = pred['cluster'].astype(str)
    pred['year'] = pred['year'].astype(str)
    macro = pred_all[pred_all['cluster'] == 'macro avg']

    years_sorted = sorted(pred['year'].unique())
    n_years = len(years_sorted)
    max_k = pred.groupby('year')['cluster'].nunique().max()
    x_base = np.arange(max_k)
    w = 0.8 / n_years

    for i, yr in enumerate(years_sorted):
        sub = pred[pred['year'] == yr].sort_values('cluster').reset_index(drop=True)
        x_pos = np.arange(len(sub)) + (i - (n_years - 1) / 2) * w
        bars = ax.bar(x_pos, sub['f1'], w,
                      label=f'MDHS {yr}',
                      color=COLORS_YEAR.get(str(yr), '#888888'),
                      edgecolor='white')
        mv = macro[macro['year'] == yr]['f1'].values
        if len(mv):
            ax.axhline(mv[0], color=COLORS_YEAR.get(str(yr), '#888'),
                       linewidth=1.5, linestyle='--', alpha=0.8,
                       label=f"Macro {yr}={mv[0]:.2f}")

    ax.set_xlim(-0.6, max_k - 0.4)
    ax.set_xticks(np.arange(max_k))
    ax.set_xticklabels([f'C{c}' for c in range(max_k)])
    ax.set_ylabel('F1 Score', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title('B. Per-Cluster F1 Score\n(5-fold CV, Random Forest)',
                 fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('Figure 5. Predictors of Cluster Membership: '
                 'Variable Importance and Predictability',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 5: {out_path.name}")


fig5_predictors(imp_all, pred_all, FIGURES / "Fig5_predictors.png")

print("\n" + "=" * 70)
print("ALL PUBLICATION FIGURES GENERATED")
print(f"  Output directory: {FIGURES}")
print("=" * 70)
