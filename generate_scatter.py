"""
Generate PCA scatter plot of cluster assignments for 2015 and 2024.
Produces a publication-quality two-panel figure (Fig 6).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

FEATURES = [
    'first_anc_month','anc_visits','skilled_anc','early_anc',
    'adequate_anc','optimal_anc','told_complications',
    'facility_delivery','skilled_delivery','caesarean','pnc_received'
]

# ─── Cluster labels ────────────────────────────────────────────────────────────
# Correct: c0=910(7%)=C-C, c1=1143(8.8%)=C-D, c2=7828(60.1%)=C-B, c3=3149(24.2%)=C-A
# Correct: c0=3232(46.7%)=C-1, c1=344(5.0%)=C-3, c2=3349(48.4%)=C-2
LABELS_2015 = {
    0: 'C-section / high PNC\n(C-C)',
    1: 'Minimal utilisation\n(C-D)',
    2: 'Late ANC, facility\ndelivery (C-B)',
    3: 'Comprehensive ANC\n& delivery (C-A)',
}
LABELS_2024 = {
    0: 'High coverage\n(C-1)',
    1: 'Moderate ANC, limited\ndelivery (C-3)',
    2: 'Late ANC, facility\ndelivery (C-2)',
}

# Semantic colours: blue=comprehensive, orange=late ANC, green=C-section, red=minimal, purple=moderate
COLORS_2015 = {0: '#3dae2b', 1: '#c5003e', 2: '#e87722', 3: '#1a6faf'}
COLORS_2024 = {0: '#1a6faf', 1: '#9b59b6', 2: '#e87722'}

# ─── Load data ─────────────────────────────────────────────────────────────────
df15 = pd.read_csv('results/clusters_2015.csv').dropna(subset=FEATURES)
df24 = pd.read_csv('results/clusters_2024.csv').dropna(subset=FEATURES)

# ─── Fit shared PCA on pooled standardised features ────────────────────────────
pooled = pd.concat([df15[FEATURES], df24[FEATURES]], ignore_index=True)
scaler = StandardScaler()
pooled_scaled = scaler.fit_transform(pooled)
pca = PCA(n_components=2, random_state=42)
pca.fit(pooled_scaled)

coords_15 = pca.transform(scaler.transform(df15[FEATURES]))
coords_24 = pca.transform(scaler.transform(df24[FEATURES]))

var_exp = pca.explained_variance_ratio_ * 100

# ─── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
fig.patch.set_facecolor('white')

alpha = 0.25
size  = 6

for ax, coords, df, labels, colors, year, k in [
    (axes[0], coords_15, df15, LABELS_2015, COLORS_2015, '2015', 4),
    (axes[1], coords_24, df24, LABELS_2024, COLORS_2024, '2024', 3),
]:
    ax.set_facecolor('#F8F8F8')

    for cid in sorted(df['cluster'].unique()):
        mask = df['cluster'] == cid
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[cid], s=size, alpha=alpha,
                   rasterized=True, label=None)

    # Cluster centroids + labels
    for cid in sorted(df['cluster'].unique()):
        mask = df['cluster'] == cid
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        n = mask.sum()
        pct = n / len(df) * 100
        ax.scatter(cx, cy, c=colors[cid], s=180, marker='*',
                   edgecolors='white', linewidths=0.8, zorder=5)
        label_text = f"{labels[cid]}\n({pct:.1f}%)"
        ax.annotate(label_text, (cx, cy),
                    textcoords='offset points', xytext=(8, 5),
                    fontsize=7.5, fontweight='bold', color=colors[cid],
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75))

    ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% variance)', fontsize=11)
    ax.set_title(f'MDHS {year}  (n = {len(df):,};  K = {k})',
                 fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(labelsize=9)
    ax.spines[['top','right']].set_visible(False)

    # Legend patches
    patches = [mpatches.Patch(color=colors[c], label=labels[c].replace('\n', ' '))
               for c in sorted(df['cluster'].unique())]
    ax.legend(handles=patches, fontsize=8, loc='upper right',
              framealpha=0.9, edgecolor='#CCCCCC', frameon=True,
              title='Cluster', title_fontsize=8.5)

# Overall caption note
fig.suptitle(
    'Figure 6. PCA scatter plot of individual women by cluster membership '
    '(shared feature space; stars = cluster centroids)',
    fontsize=10, style='italic', y=-0.02
)

plt.savefig(f'{OUT}/Fig6_cluster_scatter.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.savefig(f'{OUT}/Fig6_cluster_scatter.pdf', dpi=300, bbox_inches='tight',
            facecolor='white')
print('Scatter figure saved.')
print(f'Variance explained: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%')
