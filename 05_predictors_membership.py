"""
=============================================================================
SCRIPT 05: PREDICTORS OF CLUSTER MEMBERSHIP
=============================================================================
Identify sociodemographic predictors of cluster membership using:
  1. Multinomial logistic regression (interpretable; ORs + 95% CIs)
  2. Random Forest classifier (variable importance; non-linear effects)
  3. Temporal comparison of predictor importance across years

Each model is run separately for 2015 and 2024, and results are compared.

Output:
  results/multinomial_results_*.csv
  results/rf_importance_*.csv
  results/predictor_comparison.csv
  figures/05_*.png
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import statsmodels.api as sm
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
print("PREDICTORS OF CLUSTER MEMBERSHIP")
print("=" * 70)

df15 = pd.read_csv(RESULTS / "clusters_2015.csv")
df24 = pd.read_csv(RESULTS / "clusters_2024.csv")

SOCIO_COLS = ['age', 'age_group', 'residence', 'wealth_quintile', 'education',
              'births_last5', 'parity', 'region', 'married', 'distance_problem',
              'survey_year', 'cluster', 'cluster_label']

feat_cols = [c for c in df15.columns
             if c not in SOCIO_COLS and not c.startswith('prob_cluster')]

# Predictors (sociodemographic + background)
PREDICTOR_CANDIDATES = [
    'age', 'residence', 'wealth_quintile', 'education',
    'parity', 'married', 'distance_problem', 'region'
]

# ============================================================
# 2. FEATURE PREPARATION
# ============================================================

def prepare_predictors(df, predictor_cols):
    """
    Prepare predictor matrix with dummy encoding of categoricals.
    Returns X (DataFrame), y (Series), and feature names.
    """
    keep = [c for c in predictor_cols if c in df.columns]
    df_sub = df[keep + ['cluster']].dropna()

    y = df_sub['cluster']
    X_raw = df_sub[keep].copy()

    # Encode categorical variables — drop_first=True removes perfect multicollinearity
    cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

    return X_enc, y, X_raw


X15, y15, X15_raw = prepare_predictors(df15, PREDICTOR_CANDIDATES)
X24, y24, X24_raw = prepare_predictors(df24, PREDICTOR_CANDIDATES)

print(f"\n2015: {len(y15):,} obs, {X15.shape[1]} predictors, "
      f"{y15.nunique()} clusters")
print(f"2024: {len(y24):,} obs, {X24.shape[1]} predictors, "
      f"{y24.nunique()} clusters")

# ============================================================
# 3. MULTINOMIAL LOGISTIC REGRESSION
# ============================================================

def run_multinomial(X, y, year_label):
    """
    Fit multinomial logistic regression via statsmodels (MNLogit).
    Uses newton-cg with regularisation for numerical stability.
    Falls back to sklearn if convergence fails.
    Returns a results DataFrame with ORs, CIs, and p-values.
    """
    print(f"\n  --- Multinomial logistic regression: {year_label} ---")

    # Reference category = cluster with largest N
    ref_cat = y.value_counts().idxmax()
    print(f"    Reference cluster: {ref_cat}")

    # Remap y: ref_cat → 0, non-ref clusters → 1, 2, 3...
    non_ref = sorted([c for c in y.unique() if c != ref_cat])
    remap = {ref_cat: 0}
    remap.update({c: i + 1 for i, c in enumerate(non_ref)})
    # params cols are 0-indexed (0,1,2 for 3 non-ref outcomes);
    # conf_int uses string of the y_r value ('1','2','3')
    # col_idx i → y_r value (i+1) → original cluster non_ref[i]
    yr_to_orig = {i + 1: c for i, c in enumerate(non_ref)}

    y_r = y.map(remap)
    X_sm = sm.add_constant(X.astype(float))

    try:
        model = sm.MNLogit(y_r, X_sm)
        result = model.fit(maxiter=500, method='bfgs', disp=False)
        _ = result.bse  # triggers ValueError if covariance unavailable
        conf = result.conf_int(alpha=0.05)

        rows = []
        for col_idx in result.params.columns:   # col_idx = 0, 1, 2 (0-indexed)
            yr_val = col_idx + 1                  # corresponding y_r value
            orig_cluster = yr_to_orig[yr_val]     # back to original cluster ID
            coef  = result.params[col_idx]
            pvals = result.pvalues[col_idx]
            ci_sub = conf.loc[str(yr_val)]        # conf_int indexed by string of y_r value
            for var in coef.index:
                if var == 'const':
                    continue
                rows.append({
                    'year': year_label, 'cluster': orig_cluster,
                    'ref_cluster': ref_cat, 'predictor': var,
                    'coef': round(coef[var], 4),
                    'OR': round(np.exp(coef[var]), 4),
                    'CI_lo': round(np.exp(ci_sub.loc[var, 'lower']), 4),
                    'CI_hi': round(np.exp(ci_sub.loc[var, 'upper']), 4),
                    'p_value': round(pvals[var], 5)
                })
        df_res = pd.DataFrame(rows)
        print(f"    Converged. McFadden R² = {1 - result.llf/result.llnull:.4f}")
        return df_res, result

    except Exception as e:
        print(f"    ⚠ MNLogit failed ({type(e).__name__}: {e}). Using sklearn fallback.")
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X.astype(float))
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, solver='lbfgs')
        clf.fit(X_s, y)
        rows = []
        for i, c in enumerate(clf.classes_):
            if c == ref_cat:
                continue
            for j, var in enumerate(X.columns):
                coef_val = clf.coef_[i, j]
                rows.append({
                    'year': year_label, 'cluster': c,
                    'ref_cluster': ref_cat, 'predictor': var,
                    'coef': round(coef_val, 4),
                    'OR': round(np.exp(coef_val), 4),
                    'CI_lo': np.nan, 'CI_hi': np.nan, 'p_value': np.nan
                })
        return pd.DataFrame(rows), None


mnl15, res15 = run_multinomial(X15, y15, '2015')
mnl24, res24 = run_multinomial(X24, y24, '2024')

mnl_all = pd.concat([mnl15, mnl24], ignore_index=True)
mnl_all.to_csv(RESULTS / "multinomial_results_all.csv", index=False)
mnl15.to_csv(RESULTS / "multinomial_results_2015.csv", index=False)
mnl24.to_csv(RESULTS / "multinomial_results_2024.csv", index=False)

# ============================================================
# 4. RANDOM FOREST — VARIABLE IMPORTANCE
# ============================================================

def run_random_forest(X, y, year_label, n_estimators=500, cv_folds=5):
    """
    Fit Random Forest and compute:
      - Mean Decrease in Impurity (MDI) importance
      - Cross-validated macro F1 score
    """
    print(f"\n  --- Random Forest: {year_label} ---")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X.astype(float))

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_s, y)

    # Cross-validated performance
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    f1_scores = cross_val_score(rf, X_s, y, cv=cv,
                                scoring='f1_macro', n_jobs=-1)
    print(f"    CV macro F1 = {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    # Feature importance
    imp = pd.DataFrame({
        'predictor': X.columns,
        'importance': rf.feature_importances_,
        'year': year_label
    }).sort_values('importance', ascending=False)

    print(f"    Top-5 predictors:")
    print(imp.head(5)[['predictor', 'importance']].to_string(index=False))

    return rf, imp, f1_scores.mean()


rf15, imp15, f1_15 = run_random_forest(X15, y15, '2015')
rf24, imp24, f1_24 = run_random_forest(X24, y24, '2024')

imp_all = pd.concat([imp15, imp24], ignore_index=True)
imp_all.to_csv(RESULTS / "rf_importance_all.csv", index=False)

# ============================================================
# 5. CONFUSION MATRICES (CV PREDICTIONS)
# ============================================================

def plot_confusion_matrix(X, y, rf, year_label, out_path):
    from sklearn.model_selection import cross_val_predict
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X.astype(float))
    y_pred = cross_val_predict(rf, X_s, y, cv=5)
    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels, normalize='true')

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix (Normalised) — RF Classifier\nMDHS {year_label}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure: {out_path.name}")


plot_confusion_matrix(X15, y15, rf15, '2015',
                      FIGURES / "05_confusion_matrix_2015.png")
plot_confusion_matrix(X24, y24, rf24, '2024',
                      FIGURES / "05_confusion_matrix_2024.png")

# ============================================================
# 6. FOREST PLOT — ODDS RATIOS FROM MULTINOMIAL REGRESSION
# ============================================================

def plot_or_forest(mnl_df, year_label, out_path, top_n=12):
    """
    Forest plot of ORs from multinomial regression for top predictors.
    """
    # Keep top-N most extreme (by absolute log OR)
    df = mnl_df.copy()
    df['log_or'] = np.log(df['OR'].clip(0.01, 100))
    df['abs_log_or'] = df['log_or'].abs()
    top_preds = (df.groupby('predictor')['abs_log_or'].max()
                   .nlargest(top_n).index.tolist())
    df_plot = df[df['predictor'].isin(top_preds)].copy()

    n_clusters = df_plot['cluster'].nunique()
    clusters = sorted(df_plot['cluster'].unique())
    palette = sns.color_palette("Set2", n_clusters)

    fig, ax = plt.subplots(figsize=(10, max(6, len(top_preds) * 0.8)))

    y_pos = {}
    base_y = 0
    gap = 0.3
    cluster_gap = 0.7

    for j, c in enumerate(clusters):
        sub = df_plot[df_plot['cluster'] == c].sort_values('log_or')
        for i, (_, row) in enumerate(sub.iterrows()):
            y_val = base_y + i * gap
            ax.errorbar(row['OR'], y_val,
                        xerr=[[row['OR'] - row['CI_lo']],
                               [row['CI_hi'] - row['OR']]],
                        fmt='o', color=palette[j], markersize=6,
                        capsize=3, linewidth=1.5,
                        label=f'Cluster {c}' if i == 0 else '')
            if not pd.isna(row.get('p_value', np.nan)):
                star = '***' if row['p_value'] < 0.001 else \
                       '**' if row['p_value'] < 0.01 else \
                       '*' if row['p_value'] < 0.05 else ''
                ax.text(row['CI_hi'] * 1.02, y_val, star,
                        va='center', fontsize=8, color=palette[j])
        base_y += len(sub) * gap + cluster_gap

    ax.axvline(1, color='black', linewidth=0.8, linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('Odds Ratio (log scale)', fontsize=11)
    ax.set_title(f'Predictors of Cluster Membership — MDHS {year_label}\n'
                 f'(Reference: largest cluster; * p<0.05, ** p<0.01, *** p<0.001)',
                 fontsize=11, fontweight='bold')
    ax.legend(title='Cluster (vs. ref)', fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure: {out_path.name}")


if not mnl15.empty:
    plot_or_forest(mnl15, '2015', FIGURES / "05_or_forest_2015.png")
if not mnl24.empty:
    plot_or_forest(mnl24, '2024', FIGURES / "05_or_forest_2024.png")

# ============================================================
# 7. VARIABLE IMPORTANCE COMPARISON PLOT
# ============================================================

def plot_importance_comparison(imp15, imp24, out_path, top_n=15):
    """
    Side-by-side bar chart comparing variable importance
    between 2015 and 2024 (top N predictors by average importance).
    """
    top_vars = (pd.concat([imp15, imp24])
                  .groupby('predictor')['importance'].mean()
                  .nlargest(top_n).index.tolist())

    df15_top = imp15[imp15['predictor'].isin(top_vars)].set_index('predictor')
    df24_top = imp24[imp24['predictor'].isin(top_vars)].set_index('predictor')

    combined = pd.DataFrame({
        '2015': df15_top.get('importance', pd.Series(dtype=float)),
        '2024': df24_top.get('importance', pd.Series(dtype=float))
    }).fillna(0).reindex(top_vars).sort_values('2015', ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(6, len(top_vars) * 0.5)))
    y_pos = np.arange(len(combined))
    w = 0.35
    ax.barh(y_pos - w/2, combined['2015'], w,
            label='MDHS 2015', color='#2196F3', edgecolor='white')
    ax.barh(y_pos + w/2, combined['2024'], w,
            label='MDHS 2024', color='#F44336', edgecolor='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([v.replace('_', ' ').replace('wealth quintile ', 'WQ: ')
                        for v in combined.index], fontsize=9)
    ax.set_xlabel('Mean Decrease in Impurity (Feature Importance)', fontsize=10)
    ax.set_title('Variable Importance for Predicting Cluster Membership\n'
                 'MDHS 2015 vs 2024 (Random Forest)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure: {out_path.name}")


plot_importance_comparison(imp15, imp24,
                           FIGURES / "05_importance_comparison.png")

# ============================================================
# 8. PREDICTABILITY SUMMARY TABLE
# ============================================================

def compute_predictability(X, y, rf, year_label):
    from sklearn.model_selection import cross_val_predict
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X.astype(float))
    y_pred = cross_val_predict(rf, X_s, y, cv=5)
    report = classification_report(y, y_pred, output_dict=True)

    rows = []
    for label, metrics in report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            try:
                c = int(label)
                n = (y == c).sum()
                rows.append({
                    'year': year_label, 'cluster': c,
                    'n': n, 'pct': round(n/len(y)*100, 1),
                    'precision': round(metrics['precision'], 3),
                    'recall': round(metrics['recall'], 3),
                    'f1': round(metrics['f1-score'], 3),
                    'support': metrics['support']
                })
            except ValueError:
                pass

    # Macro average
    macro = report.get('macro avg', {})
    rows.append({
        'year': year_label, 'cluster': 'macro avg', 'n': len(y),
        'pct': 100,
        'precision': round(macro.get('precision', 0), 3),
        'recall': round(macro.get('recall', 0), 3),
        'f1': round(macro.get('f1-score', 0), 3),
        'support': len(y)
    })
    return pd.DataFrame(rows)


pred15 = compute_predictability(X15, y15, rf15, '2015')
pred24 = compute_predictability(X24, y24, rf24, '2024')
pred_all = pd.concat([pred15, pred24], ignore_index=True)
pred_all.to_csv(RESULTS / "predictability_summary.csv", index=False)

print("\nPredictability summary:")
print(pred_all.to_string(index=False))

# ============================================================
# 9. F1 COMPARISON PLOT ACROSS YEARS
# ============================================================

def plot_f1_comparison(pred_all, out_path):
    """Plot per-cluster F1 scores per year (handles different K per year)."""
    df = pred_all[pred_all['cluster'] != 'macro avg'].copy()
    df['cluster'] = df['cluster'].astype(str)
    macro = pred_all[pred_all['cluster'] == 'macro avg'][['year', 'f1']].rename(
        columns={'f1': 'macro_f1'})

    years = sorted(df['year'].unique())
    palette = {'2015': '#2196F3', '2024': '#F44336'}

    fig, axes = plt.subplots(1, len(years), figsize=(5 * len(years), 5),
                             sharey=True)
    if len(years) == 1:
        axes = [axes]

    for ax, yr in zip(axes, years):
        sub = df[df['year'] == yr].sort_values('cluster')
        x = np.arange(len(sub))
        bars = ax.bar(x, sub['f1'], color=palette.get(str(yr), '#888'),
                      edgecolor='white', width=0.6)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01, f'{bar.get_height():.2f}',
                    ha='center', va='bottom', fontsize=9)
        macro_val = macro[macro['year'] == yr]['macro_f1'].values
        if len(macro_val):
            ax.axhline(macro_val[0], color=palette.get(str(yr), '#888'),
                       linewidth=1.8, linestyle='--', alpha=0.8,
                       label=f"Macro F1 = {macro_val[0]:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{c}' for c in sub['cluster']], fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title(f'MDHS {yr}', fontsize=11, fontweight='bold',
                     color=palette.get(str(yr), 'black'))
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Per-Cluster F1 Score for Predicting Cluster Membership\n'
                 '(Random Forest, 5-fold CV)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure: {out_path.name}")


plot_f1_comparison(pred_all, FIGURES / "05_f1_comparison.png")

print("\n" + "=" * 70)
print("PREDICTORS ANALYSIS COMPLETE")
print(f"  2015 CV macro F1 = {f1_15:.3f}")
print(f"  2024 CV macro F1 = {f1_24:.3f}")
print("  Outputs saved to results/ and figures/")
print("=" * 70)
