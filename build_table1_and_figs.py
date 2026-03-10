"""
Build revised Table 1 (demographics by cluster), Figure 1 (model selection),
and Figure 2 (cluster profiles) — styled after PMC8533034.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import os, json, warnings
warnings.filterwarnings('ignore')

OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

# ─── Load data ─────────────────────────────────────────────────────────────────
df15 = pd.read_csv('results/clusters_2015.csv')
df24 = pd.read_csv('results/clusters_2024.csv')

FEATURES = ['first_anc_month','anc_visits','skilled_anc','early_anc',
            'adequate_anc','optimal_anc','told_complications',
            'facility_delivery','skilled_delivery','caesarean','pnc_received']

# ─── Remap cluster IDs to match manuscript labels ──────────────────────────────
# 2015: 0=Comprehensive(A), 1=Late/facility(B), 2=C-section(C), 3=Minimal(D)
# 2024: 0=High coverage(1), 1=Late/facility(2), 2=Moderate/limited(3)
# Correct mappings verified against n values in clusters_2015/2024.csv:
# 2015: c0=910 (7.0%)=C-C, c1=1143 (8.8%)=C-D, c2=7828 (60.1%)=C-B, c3=3149 (24.2%)=C-A
# 2024: c0=3232 (46.7%)=C-1, c1=344 (5.0%)=C-3, c2=3349 (48.4%)=C-2
CLUSTER_LABELS_15 = {
    0: "C-section/\nhigh PNC\n(C-C)",
    1: "Minimal\nutilisation\n(C-D)",
    2: "Late ANC,\nfacility delivery\n(C-B)",
    3: "Comprehensive\nANC+delivery\n(C-A)",
}
CLUSTER_LABELS_24 = {
    0: "High\ncoverage\n(C-1)",
    1: "Moderate ANC,\nlimited delivery\n(C-3)",
    2: "Late ANC,\nfacility delivery\n(C-2)",
}
# Semantic colours: blue=comprehensive/high-coverage, orange=late ANC, green=C-section, red=minimal, purple=moderate/limited
COLORS_15 = {0:'#3dae2b', 1:'#c5003e', 2:'#e87722', 3:'#1a6faf'}
COLORS_24 = {0:'#1a6faf', 1:'#9b59b6', 2:'#e87722'}

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1: Demographics by cluster (JSON for docx builder to consume)
# ─────────────────────────────────────────────────────────────────────────────
def cluster_demographics(df, year, cluster_labels, feat_label_map=None):
    rows = []
    clusters = sorted(df['cluster'].unique())
    n_total = len(df)

    # Overall n
    row_n = {"var": "n (%)", "overall": f"{n_total:,}"}
    for c in clusters:
        n_c = (df['cluster']==c).sum()
        row_n[f"c{c}"] = f"{n_c:,} ({n_c/n_total*100:.1f}%)"
    rows.append(row_n)

    # Age (mean ± SD)
    row_age = {"var": "Age, years (mean ± SD)",
               "overall": f"{df['age'].mean():.1f} ± {df['age'].std():.1f}"}
    for c in clusters:
        sub = df[df['cluster']==c]['age']
        row_age[f"c{c}"] = f"{sub.mean():.1f} ± {sub.std():.1f}"
    rows.append(row_age)

    # Age group
    for ag in ['Adolescent (15\u201319)', 'Young adult (20\u201334)', 'Older (35\u201349)']:
        mask = df['age_group']==ag
        n_ag = mask.sum()
        row = {"var": f"  {ag}, n (%)",
               "overall": f"{n_ag:,} ({n_ag/n_total*100:.1f}%)"}
        for c in clusters:
            sub = df[(df['cluster']==c) & mask]
            n_c = (df['cluster']==c).sum()
            n_sub = len(sub)
            row[f"c{c}"] = f"{n_sub:,} ({n_sub/n_c*100:.1f}%)"
        rows.append(row)

    # Residence
    for res in ['Urban','Rural']:
        mask = df['residence']==res
        n_res = mask.sum()
        row = {"var": f"  {res}, n (%)",
               "overall": f"{n_res:,} ({n_res/n_total*100:.1f}%)"}
        for c in clusters:
            n_c = (df['cluster']==c).sum()
            n_sub = ((df['cluster']==c) & mask).sum()
            row[f"c{c}"] = f"{n_sub:,} ({n_sub/n_c*100:.1f}%)"
        rows.append(row)

    # Wealth quintile
    for wq in ['Poorest','Poorer','Middle','Richer','Richest']:
        mask = df['wealth_quintile']==wq
        n_wq = mask.sum()
        row = {"var": f"  {wq}, n (%)",
               "overall": f"{n_wq:,} ({n_wq/n_total*100:.1f}%)"}
        for c in clusters:
            n_c = (df['cluster']==c).sum()
            n_sub = (df[(df['cluster']==c)]['wealth_quintile']==wq).sum()
            row[f"c{c}"] = f"{n_sub:,} ({n_sub/n_c*100:.1f}%)"
        rows.append(row)

    # Education
    for ed in ['No education','Primary','Secondary','Higher']:
        mask = df['education']==ed
        n_ed = mask.sum()
        row = {"var": f"  {ed}, n (%)",
               "overall": f"{n_ed:,} ({n_ed/n_total*100:.1f}%)"}
        for c in clusters:
            n_c = (df['cluster']==c).sum()
            n_sub = (df[df['cluster']==c]['education']==ed).sum()
            row[f"c{c}"] = f"{n_sub:,} ({n_sub/n_c*100:.1f}%)"
        rows.append(row)

    # Parity
    row_par = {"var": "Parity (mean ± SD)",
               "overall": f"{df['parity'].mean():.1f} ± {df['parity'].std():.1f}"}
    for c in clusters:
        sub = df[df['cluster']==c]['parity']
        row_par[f"c{c}"] = f"{sub.mean():.1f} ± {sub.std():.1f}"
    rows.append(row_par)

    # Married
    mask_mar = df['married']==1
    n_mar = mask_mar.sum()
    row_mar = {"var": "Currently married, n (%)",
               "overall": f"{n_mar:,} ({n_mar/n_total*100:.1f}%)"}
    for c in clusters:
        n_c = (df['cluster']==c).sum()
        n_sub = (df[df['cluster']==c]['married']==1).sum()
        row_mar[f"c{c}"] = f"{n_sub:,} ({n_sub/n_c*100:.1f}%)"
    rows.append(row_mar)

    # Distance to facility a problem
    mask_dist = df['distance_problem']==1
    n_dist = mask_dist.sum()
    row_dist = {"var": "Distance to facility a problem, n (%)",
                "overall": f"{n_dist:,} ({n_dist/n_total*100:.1f}%)"}
    for c in clusters:
        n_c = (df['cluster']==c).sum()
        n_sub = (df[df['cluster']==c]['distance_problem']==1).sum()
        row_dist[f"c{c}"] = f"{n_sub:,} ({n_sub/n_c*100:.1f}%)"
    rows.append(row_dist)

    return rows, clusters

rows15, clusters15 = cluster_demographics(df15, 2015, CLUSTER_LABELS_15)
rows24, clusters24 = cluster_demographics(df24, 2024, CLUSTER_LABELS_24)

# Save as JSON for use in docx builder
table1_data = {
    "2015": {"rows": rows15, "clusters": [int(c) for c in clusters15], "labels": {str(k): v for k,v in CLUSTER_LABELS_15.items()}},
    "2024": {"rows": rows24, "clusters": [int(c) for c in clusters24], "labels": {str(k): v for k,v in CLUSTER_LABELS_24.items()}}
}
with open('results/table1_by_cluster.json', 'w') as f:
    json.dump(table1_data, f, indent=2)
print("Table 1 data saved.")

# Print preview
print("\n=== 2015 TABLE 1 PREVIEW ===")
for r in rows15[:5]:
    print(r)
print("\n=== 2024 TABLE 1 PREVIEW ===")
for r in rows24[:5]:
    print(r)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Model selection — BIC + Silhouette vs K (both years)
# ─────────────────────────────────────────────────────────────────────────────
print("\nFitting GMMs for model selection figure...")
fig1, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig1.patch.set_facecolor('white')

year_data = [(df15, 2015, 4, '#1a6faf'), (df24, 2024, 3, '#1a6faf')]

for ax, (df, year, optimal_k, col) in zip(axes, year_data):
    X = df[FEATURES].dropna().values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Ks = range(2, 7)
    bics, silhs, comp_scores = [], [], []

    for k in Ks:
        best_bic = np.inf
        best_model = None
        for seed in range(5):
            try:
                gm = GaussianMixture(n_components=k, covariance_type='full',
                                     n_init=1, random_state=seed, max_iter=300)
                gm.fit(Xs)
                if gm.bic(Xs) < best_bic:
                    best_bic = gm.bic(Xs)
                    best_model = gm
            except:
                pass
        labels_k = best_model.predict(Xs)
        sil = silhouette_score(Xs, labels_k, sample_size=min(3000, len(Xs)), random_state=42)

        bics.append(best_bic)
        silhs.append(sil)
        comp_scores.append(0.6 * (-best_bic/max(abs(best_bic), 1)) + 0.4 * sil)

    # Normalise BIC to 0–1 range for dual-axis
    bics_norm = [(b - min(bics)) / (max(bics) - min(bics) + 1e-9) for b in bics]
    silhs_norm = [(s - min(silhs)) / (max(silhs) - min(silhs) + 1e-9) for s in silhs]

    ax2 = ax.twinx()
    lns1 = ax.plot(list(Ks), bics_norm, 'o-', color='#1a6faf', lw=2.5, ms=8,
                   markerfacecolor='white', markeredgewidth=2, label='Normalised BIC (lower = better)')
    lns2 = ax2.plot(list(Ks), silhs, 's--', color='#e87722', lw=2.5, ms=8,
                    markerfacecolor='white', markeredgewidth=2, label='Silhouette coefficient')

    # Vertical line at optimal K
    ax.axvline(optimal_k, color='#c5003e', ls=':', lw=2.0, alpha=0.8, label=f'Optimal K = {optimal_k}')
    ax2.axvline(optimal_k, color='#c5003e', ls=':', lw=2.0, alpha=0.8)

    ax.set_xlabel('Number of clusters (K)', fontsize=12)
    ax.set_ylabel('Normalised BIC (lower is better)', fontsize=11, color='#1a6faf')
    ax2.set_ylabel('Silhouette coefficient', fontsize=11, color='#e87722')
    ax.tick_params(axis='y', labelcolor='#1a6faf')
    ax2.tick_params(axis='y', labelcolor='#e87722')
    ax.set_xticks(list(Ks))
    ax.set_title(f'MDHS {year}', fontsize=13, fontweight='bold', pad=10)
    ax.set_facecolor('#F8F8F8')
    ax.spines[['top']].set_visible(False)
    ax2.spines[['top']].set_visible(False)

    # Combined legend
    lns = lns1 + lns2 + [Line2D([0],[0], color='#c5003e', ls=':', lw=2, label=f'Optimal K = {optimal_k}')]
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=9, loc='upper right', framealpha=0.85)

fig1.suptitle('Figure 1. Model selection: BIC and silhouette coefficient across K = 2–6 for MDHS 2015 and 2024.\n'
              'The optimal K was selected by composite BIC (60%) and silhouette (40%) weighting.',
              fontsize=9, style='italic', y=-0.04)

fig1.savefig(f'{OUT}/Fig1_model_selection.png', dpi=300, bbox_inches='tight', facecolor='white')
fig1.savefig(f'{OUT}/Fig1_model_selection.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("Figure 1 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Cluster profiles — grouped bar chart (reference paper style)
# ─────────────────────────────────────────────────────────────────────────────
FEAT_LABELS = {
    'first_anc_month':   'First ANC\nvisit (months)',
    'anc_visits':        'ANC visits\n(mean)',
    'early_anc':         'Early ANC\ninitiation (%)',
    'adequate_anc':      'Adequate ANC\n≥4 visits (%)',
    'optimal_anc':       'Optimal ANC\n≥8 visits (%)',
    'skilled_anc':       'Skilled ANC\nprovider (%)',
    'facility_delivery': 'Facility\ndelivery (%)',
    'skilled_delivery':  'Skilled birth\nattendant (%)',
    'caesarean':         'Caesarean\nsection (%)',
    'pnc_received':      'PNC\nreceived (%)',
}

for df, year, cl_labels, cl_colors, suffix in [
    (df15, 2015, CLUSTER_LABELS_15, COLORS_15, 'a'),
    (df24, 2024, CLUSTER_LABELS_24, COLORS_24, 'b'),
]:
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    fig2.patch.set_facecolor('white')

    clusters = sorted(df['cluster'].unique())
    n_cl = len(clusters)
    feat_keys = list(FEAT_LABELS.keys())
    feat_names = list(FEAT_LABELS.values())
    n_feat = len(feat_keys)

    # Compute cluster means (convert proportions to %)
    profiles = {}
    for c in clusters:
        sub = df[df['cluster']==c]
        vals = []
        for fk in feat_keys:
            v = sub[fk].mean()
            if fk not in ('first_anc_month', 'anc_visits'):
                v = v * 100  # convert to %
            vals.append(v)
        profiles[c] = vals

    x = np.arange(n_feat)
    bar_w = 0.75 / n_cl
    offsets = np.linspace(-(n_cl-1)/2 * bar_w, (n_cl-1)/2 * bar_w, n_cl)

    for i, c in enumerate(clusters):
        short_label = cl_labels[c].replace('\n', ' ')
        n_c = (df['cluster']==c).sum()
        pct_c = n_c / len(df) * 100
        ax.bar(x + offsets[i], profiles[c], width=bar_w,
               color=cl_colors[c], alpha=0.85,
               label=f"{short_label}  (n={n_c:,}; {pct_c:.1f}%)",
               edgecolor='white', linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, fontsize=9, rotation=25, ha='right')
    ax.set_ylabel('Mean value (months / count) or proportion (%)', fontsize=11)
    ax.set_title(f'MDHS {year}  (K\u00a0=\u00a0{n_cl})', fontsize=14, fontweight='bold', pad=12)
    ax.set_facecolor('#F8F8F8')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', labelsize=9)
    ax.yaxis.grid(True, alpha=0.4, color='white')
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='#CCCCCC',
              title='Cluster', title_fontsize=9.5, ncol=1)

    fig_label = f'2{suffix}'
    fig2.suptitle(
        f'Figure {fig_label}. Cluster profiles of maternal care utilisation indicators, MDHS {year} (K\u00a0=\u00a0{n_cl}).\n'
        'Bars show mean values per cluster; percentages reflect proportional indicators.',
        fontsize=9, style='italic', y=-0.04
    )

    fig2.savefig(f'{OUT}/Fig2{suffix}_cluster_profiles_{year}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig(f'{OUT}/Fig2{suffix}_cluster_profiles_{year}.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"Figure 2{suffix} ({year}) saved.")

print("All done.")
