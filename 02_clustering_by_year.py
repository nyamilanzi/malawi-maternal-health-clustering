"""
=============================================================================
SCRIPT 02: UNSUPERVISED CLUSTERING BY SURVEY YEAR (2015 AND 2024)
=============================================================================
Apply Gaussian Mixture Models (GMM) independently to 2015 and 2024
harmonised data. Select the optimal number of clusters using BIC, AIC,
and interpretability criteria. Validate with silhouette scores.

Output:
  results/clusters_2015.csv
  results/clusters_2024.csv
  results/cluster_profiles_2015.csv
  results/cluster_profiles_2024.csv
  results/model_selection_summary.csv
  figures/02_model_selection_*.png
  figures/02_cluster_profiles_*.png
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

BASE = Path(__file__).parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

# ============================================================
# 1. LOAD HARMONISED DATA
# ============================================================

print("=" * 70)
print("GMM CLUSTERING BY SURVEY YEAR")
print("=" * 70)

df15 = pd.read_csv(RESULTS / "harmonised_2015.csv")
df24 = pd.read_csv(RESULTS / "harmonised_2024.csv")

# Identify clustering feature columns (exclude sociodemographics and metadata)
SOCIO_COLS = ['age', 'age_group', 'residence', 'wealth_quintile', 'education',
              'births_last5', 'parity', 'region', 'married', 'distance_problem',
              'survey_year']

def get_feature_cols(df):
    return [c for c in df.columns if c not in SOCIO_COLS]

feat_cols = get_feature_cols(df15)
print(f"\nClustering features ({len(feat_cols)}): {feat_cols}")

# ============================================================
# 2. STANDARDISATION
# ============================================================

def standardise(df, feat_cols, year):
    X = df[feat_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n  {year}: Scaled {X_scaled.shape[0]:,} obs × {X_scaled.shape[1]} features")
    return X_scaled, scaler

X15, scaler15 = standardise(df15, feat_cols, '2015')
X24, scaler24 = standardise(df24, feat_cols, '2024')

# ============================================================
# 3. GMM MODEL SELECTION (K = 2 TO 6)
# ============================================================

def run_gmm_selection(X, year_label, k_range=range(2, 7), n_init=10, random_state=42):
    """
    Fit GMM for a range of K and compute selection criteria.
    """
    print(f"\n{'='*60}")
    print(f"  GMM MODEL SELECTION: {year_label}")
    print(f"{'='*60}")

    results = []
    models = {}

    for k in k_range:
        bics, aics, sils = [], [], []
        best_model = None
        best_bic = np.inf

        for seed in range(n_init):
            gm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                n_init=3,
                random_state=seed * 100 + 42,
                max_iter=500
            )
            gm.fit(X)
            b = gm.bic(X)
            a = gm.aic(X)
            bics.append(b)
            aics.append(a)
            if b < best_bic:
                best_bic = b
                best_model = gm

        labels = best_model.predict(X)
        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)),
                               random_state=42) if len(set(labels)) > 1 else 0
        dbi = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else np.inf
        chi = calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else 0

        # Cluster size balance
        counts = pd.Series(labels).value_counts().sort_index()
        min_pct = counts.min() / len(labels) * 100
        max_pct = counts.max() / len(labels) * 100

        row = {
            'year': year_label, 'k': k,
            'bic_mean': np.mean(bics), 'bic_std': np.std(bics),
            'aic_mean': np.mean(aics), 'aic_std': np.std(aics),
            'silhouette': round(sil, 4),
            'davies_bouldin': round(dbi, 4),
            'calinski_harabasz': round(chi, 1),
            'min_cluster_pct': round(min_pct, 1),
            'max_cluster_pct': round(max_pct, 1)
        }
        results.append(row)
        models[k] = best_model

        print(f"  K={k}: BIC={np.mean(bics):.0f} (±{np.std(bics):.0f}), "
              f"Sil={sil:.3f}, DBI={dbi:.3f}, min_cluster={min_pct:.1f}%")

    return pd.DataFrame(results), models


sel15, models15 = run_gmm_selection(X15, '2015')
sel24, models24 = run_gmm_selection(X24, '2024')

# ============================================================
# 4. SELECT OPTIMAL K
# ============================================================

def select_k(sel_df, models, X, year_label, min_cluster_pct=5.0):
    """
    Select K based on:
    1. BIC elbow (largest decrease from k to k+1)
    2. All clusters >= min_cluster_pct
    3. Silhouette maximisation (tie-break)
    Prefer k=3 for interpretability if metrics are close.
    """
    valid = sel_df[sel_df['min_cluster_pct'] >= min_cluster_pct].copy()
    if valid.empty:
        valid = sel_df.copy()

    # Normalise BIC (lower is better → invert)
    valid['bic_norm'] = (valid['bic_mean'] - valid['bic_mean'].min()) / \
                        (valid['bic_mean'].max() - valid['bic_mean'].min() + 1e-9)
    valid['sil_norm'] = (valid['silhouette'] - valid['silhouette'].min()) / \
                        (valid['silhouette'].max() - valid['silhouette'].min() + 1e-9)
    valid['score'] = -0.6 * valid['bic_norm'] + 0.4 * valid['sil_norm']

    k_opt = int(valid.loc[valid['score'].idxmax(), 'k'])
    print(f"\n  {year_label}: Optimal K = {k_opt}")
    print(valid[['k', 'bic_mean', 'silhouette', 'min_cluster_pct', 'score']].to_string(index=False))

    model = models[k_opt]
    labels = model.predict(X)
    probs = model.predict_proba(X)
    return k_opt, labels, probs, model


k15, labels15, probs15, model15 = select_k(sel15, models15, X15, '2015')
k24, labels24, probs24, model24 = select_k(sel24, models24, X24, '2024')

# ============================================================
# 5. CLUSTER PROFILES
# ============================================================

def compute_profiles(df, feat_cols, labels, year_label):
    """Compute mean/proportion of each feature by cluster."""
    df_c = df[feat_cols].copy()
    df_c['cluster'] = labels

    rows = []
    for c in sorted(df_c['cluster'].unique()):
        sub = df_c[df_c['cluster'] == c]
        row = {'cluster': c, 'n': len(sub),
               'pct': round(len(sub) / len(df_c) * 100, 2)}
        for col in feat_cols:
            row[col] = round(sub[col].mean(), 4)
        rows.append(row)

    prof = pd.DataFrame(rows)
    print(f"\n  Cluster profiles ({year_label}):")
    print(prof[['cluster', 'n', 'pct'] + feat_cols].to_string(index=False))
    return prof


prof15 = compute_profiles(df15, feat_cols, labels15, '2015')
prof24 = compute_profiles(df24, feat_cols, labels24, '2024')

# ============================================================
# 6. LABEL CLUSTERS SEMANTICALLY (BASED ON PROFILES)
# ============================================================

def assign_semantic_labels(profiles, feat_cols, year_label):
    """
    Assign human-readable labels based on dominant characteristics.
    Strategy:
      - High early_anc + adequate_anc → 'Comprehensive ANC'
      - High facility_delivery + skilled_delivery → 'Facility-focused'
      - Low across all → 'Minimal utilisation'
      - Etc.
    """
    labels_map = {}
    for _, row in profiles.iterrows():
        c = row['cluster']
        early = row.get('early_anc', 0)
        adequate = row.get('adequate_anc', 0)
        facility = row.get('facility_delivery', 0)
        skilled_d = row.get('skilled_delivery', 0)
        pnc = row.get('pnc_received', row.get('pnc_48h', 0))
        anc_vis = row.get('anc_visits', 4)

        if early >= 0.6 and adequate >= 0.8 and skilled_d >= 0.85:
            label = 'High coverage across continuum'
        elif early >= 0.5 and adequate >= 0.5 and skilled_d < 0.8:
            label = 'Adequate ANC, limited skilled delivery'
        elif early < 0.4 and adequate < 0.5 and skilled_d >= 0.85:
            label = 'Late/inadequate ANC but facility delivery'
        elif early < 0.35 and adequate < 0.35 and facility < 0.7:
            label = 'Minimal utilisation'
        elif pnc > 0.3 and skilled_d >= 0.85:
            label = 'High facility + postnatal care'
        elif anc_vis >= 5 and skilled_d >= 0.85:
            label = 'High ANC intensity + skilled delivery'
        else:
            label = f'Mixed utilisation (Cluster {c})'

        labels_map[c] = label
        print(f"    {year_label} Cluster {c}: '{label}'")

    return labels_map


print("\n  --- Semantic cluster labelling ---")
labels_map15 = assign_semantic_labels(prof15, feat_cols, '2015')
labels_map24 = assign_semantic_labels(prof24, feat_cols, '2024')

prof15['label'] = prof15['cluster'].map(labels_map15)
prof24['label'] = prof24['cluster'].map(labels_map24)

# ============================================================
# 7. MODEL SELECTION VISUALISATION
# ============================================================

def plot_model_selection(sel15, sel24):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ('bic_mean', 'BIC (lower = better)', True),
        ('silhouette', 'Silhouette Score (higher = better)', False),
        ('min_cluster_pct', 'Smallest Cluster Size (%)', False)
    ]

    colors = {'2015': '#2196F3', '2024': '#F44336'}
    markers = {'2015': 'o', '2024': 's'}

    for ax, (metric, title, lower_better) in zip(axes, metrics):
        for sel, year in [(sel15, '2015'), (sel24, '2024')]:
            ax.plot(sel['k'], sel[metric],
                    color=colors[year], marker=markers[year],
                    linewidth=2, markersize=7, label=year)
        ax.set_xlabel('Number of Clusters (K)', fontsize=11)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(title='Survey year', fontsize=9)
        ax.set_xticks(sel15['k'].values)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if lower_better:
            ax.annotate('Lower is better', xy=(0.02, 0.05),
                        xycoords='axes fraction', fontsize=8, color='grey')
        else:
            ax.annotate('Higher is better', xy=(0.02, 0.05),
                        xycoords='axes fraction', fontsize=8, color='grey')

    plt.suptitle('GMM Model Selection Criteria: MDHS 2015 and 2024',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES / "02_model_selection.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\n  ✓ Figure: 02_model_selection.png")


plot_model_selection(sel15, sel24)

# ============================================================
# 8. CLUSTER PROFILE HEATMAPS
# ============================================================

def plot_profile_heatmap(prof, feat_cols, year_label, k_opt, out_path):
    """
    Heatmap of standardised cluster means, with raw values annotated.
    """
    heat_data = prof.set_index('cluster')[feat_cols].copy()
    heat_data.index = [f"Cluster {c}\n({prof.loc[prof['cluster']==c,'pct'].values[0]:.1f}%)"
                       for c in heat_data.index]

    # Standardise columns for colour mapping
    heat_norm = (heat_data - heat_data.mean()) / (heat_data.std() + 1e-9)

    col_labels = [c.replace('_', '\n') for c in feat_cols]

    fig, ax = plt.subplots(figsize=(max(12, len(feat_cols) * 1.6), 4))
    sns.heatmap(heat_norm, annot=heat_data.round(2),
                fmt='.2f', cmap='RdYlGn', center=0,
                linewidths=0.5, linecolor='white',
                ax=ax, cbar_kws={'label': 'Z-score (row-wise)'})
    ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(f'Cluster Profiles (K={k_opt}) — MDHS {year_label}',
                 fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


plot_profile_heatmap(prof15, feat_cols, '2015', k15,
                     FIGURES / "02_cluster_profiles_2015.png")
plot_profile_heatmap(prof24, feat_cols, '2024', k24,
                     FIGURES / "02_cluster_profiles_2024.png")
print("  ✓ Figures: 02_cluster_profiles_*.png")

# ============================================================
# 9. PCA VISUALISATION OF CLUSTERS
# ============================================================

def plot_pca_clusters(X, labels, year_label, out_path):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100

    n_clusters = len(set(labels))
    palette = sns.color_palette("Set2", n_clusters)
    color_map = {c: palette[i] for i, c in enumerate(sorted(set(labels)))}

    fig, ax = plt.subplots(figsize=(9, 7))
    for c in sorted(set(labels)):
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color_map[c]], alpha=0.35, s=12, label=f'Cluster {c}',
                   rasterized=True)
    # Centroids
    for c in sorted(set(labels)):
        mask = labels == c
        cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
        ax.scatter(cx, cy, c=[color_map[c]], s=200, marker='*',
                   edgecolors='black', linewidth=0.8, zorder=5)

    ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% variance)', fontsize=11)
    ax.set_title(f'PCA Projection of Clusters — MDHS {year_label}',
                 fontsize=12, fontweight='bold')
    ax.legend(title='Cluster', fontsize=9, markerscale=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


plot_pca_clusters(X15, labels15, '2015', FIGURES / "02_pca_2015.png")
plot_pca_clusters(X24, labels24, '2024', FIGURES / "02_pca_2024.png")
print("  ✓ Figures: 02_pca_*.png")

# ============================================================
# 10. SAVE OUTPUTS
# ============================================================

# Cluster assignments
out15 = df15.copy()
out15['cluster'] = labels15
out15['cluster_label'] = out15['cluster'].map(labels_map15)
for c in range(probs15.shape[1]):
    out15[f'prob_cluster_{c}'] = probs15[:, c]

out24 = df24.copy()
out24['cluster'] = labels24
out24['cluster_label'] = out24['cluster'].map(labels_map24)
for c in range(probs24.shape[1]):
    out24[f'prob_cluster_{c}'] = probs24[:, c]

out15.to_csv(RESULTS / "clusters_2015.csv", index=False)
out24.to_csv(RESULTS / "clusters_2024.csv", index=False)

# Profiles
prof15.to_csv(RESULTS / "cluster_profiles_2015.csv", index=False)
prof24.to_csv(RESULTS / "cluster_profiles_2024.csv", index=False)

# Model selection summary
sel_all = pd.concat([sel15, sel24], ignore_index=True)
sel_all.to_csv(RESULTS / "model_selection_summary.csv", index=False)

print("\n" + "=" * 70)
print("CLUSTERING COMPLETE")
print(f"  2015: K={k15}, n={len(df15):,}")
print(f"  2024: K={k24}, n={len(df24):,}")
print("  Outputs saved to results/ and figures/")
print("=" * 70)
