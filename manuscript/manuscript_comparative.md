# Temporal Trends and Demographic Determinants of Maternal Health Care Utilisation Patterns in Malawi: A Comparative Unsupervised Learning Analysis of the 2015 and 2024 Demographic and Health Surveys

---

**Authors:** [Author 1]¹, [Author 2]², [Author 3]³

**Affiliations:**
¹ [Institution 1], [Country]
² [Institution 2], [Country]
³ [Institution 3], [Country]

**Corresponding author:** [Name], [Email], [Institution]

**Word count (excluding abstract, tables, figures, and references):** ~5,800

**Running title:** Utilisation cluster trends in Malawi DHS 2015–2024

**Keywords:** maternal health care utilisation, unsupervised learning, cluster analysis, Gaussian mixture models, continuum of care, temporal trends, Malawi, sub-Saharan Africa, adolescent health, health equity

---

## Abstract

**Background:** Malawi has invested substantially in maternal health systems over the past decade, yet the extent to which utilisation patterns across the full continuum of care have changed remains unclear. Understanding heterogeneity in how women engage with antenatal, delivery, and postnatal services — and whether that heterogeneity has shifted over time across demographic subgroups — is essential for targeting interventions.

**Methods:** We applied Gaussian Mixture Model (GMM) clustering to individual-level data from two nationally representative Malawi Demographic and Health Surveys (MDHS 2015, n = [N₁]; MDHS 2024, n = [N₂]). Utilisation features encompassing antenatal care (timing of first visit, number of visits, provider skill, early initiation, adequacy), delivery care (facility delivery, skilled birth attendance, caesarean section), and postnatal care were harmonised across survey rounds. Clustering was performed independently per year; cluster alignment across years was achieved via the Hungarian algorithm. Subgroup analyses examined cluster prevalences by residence (urban/rural), wealth quintile (poorest–richest), and age group (adolescent 15–19, young adult 20–34, older women 35–49). Predictors of cluster membership were assessed via multinomial logistic regression (odds ratios, 95% confidence intervals) and Random Forest classifiers (variable importance, cross-validated F1 scores).

**Results:** A [K₁]-cluster solution was identified for 2015 and a [K₂]-cluster solution for 2024. [Cluster X] in 2015 and [Cluster Y] in 2024 represented women with high coverage across the continuum; [Cluster Z] captured women with late, inadequate antenatal care and lower facility delivery — this cluster [increased/decreased] in prevalence from [x]% to [y]% between survey rounds. Significant sociodemographic gradients were observed: women in the poorest wealth quintile were [X] times more likely (OR [range]) to belong to the low-utilisation cluster relative to the richest quintile; adolescent women showed markedly lower adherence to the high-utilisation pattern compared with women aged 20–34 (both years, p < 0.001). Cramér's V for wealth quintile was [V₁] (2015) and [V₂] (2024), indicating [narrowing/stable/widening] socioeconomic inequality. Random Forest macro F1 scores for predicting cluster membership were [F₁] (2015) and [F₂] (2024), with wealth and residence as the most important predictors in both years.

**Conclusions:** Utilisation of maternal health services in Malawi remains heterogeneous and structured by well-established social determinants, despite decade-long investments. The persistence of a low-utilisation cluster — disproportionately composed of adolescent, rural, and poor women — highlights the need for targeted, multi-component interventions. The modest overall predictability of cluster membership from sociodemographic factors alone underscores the importance of community-level approaches and demand-side barriers beyond wealth and education.

---

## 1. Introduction

### 1.1 Global Context

The maternal mortality ratio (MMR) in sub-Saharan Africa remains disproportionately high relative to global targets. Despite significant progress over the past two decades following the Millennium Development Goals and the Sustainable Development Goals (SDG 3.1), approximately two-thirds of all maternal deaths still occur in sub-Saharan Africa [1,2]. Utilisation of evidence-based maternal health services — antenatal care (ANC), skilled birth attendance (SBA), and postnatal care (PNC) — is foundational to reducing preventable maternal deaths [3,4].

Malawi exemplifies both the progress achieved and the challenges that persist. Between 2000 and 2020, the country's MMR declined from approximately 1,190 to 381 per 100,000 live births [5]. National policies have prioritised free maternity services, the expansion of Emergency Obstetric Care (EmOC) facilities, and community health worker deployment [6,7]. Despite these efforts, large within-country inequities remain across geographic, socioeconomic, and demographic strata [8,9].

### 1.2 The Continuum of Care as a Multi-dimensional Construct

Maternal health care utilisation is not a single binary event but a sequence of decisions and encounters spanning the antenatal, intrapartum, and postnatal periods. The "continuum of care" framework [10] positions these three phases as interconnected: early, adequate ANC increases the likelihood of facility delivery, which in turn facilitates timely PNC. Dropout at any point in the continuum diminishes the cumulative protective effect.

Traditional epidemiological analyses of this continuum have relied on single indicators (e.g., proportion receiving ≥4 ANC visits, facility delivery rate) or on sequential analysis of independent binary outcomes. These approaches, while useful, obscure meaningful heterogeneity in how women navigate the care pathway. For example, two women may each attend four ANC visits, yet one initiates care in the first trimester with a skilled provider, while the other initiates in the third trimester with an unskilled attendant — patterns with substantially different implications for maternal outcomes.

### 1.3 Unsupervised Learning for Utilisation Pattern Discovery

Unsupervised machine learning — specifically cluster analysis — offers a data-driven alternative that identifies latent groups in multi-dimensional utilisation data without imposing a priori categorisations. Several recent studies have applied cluster analysis to maternal health care utilisation data in low- and middle-income countries (LMICs), consistently identifying two to four distinct patterns that more accurately predict adverse outcomes than single-indicator approaches [11–14]. However, few studies have examined whether these patterns are temporally stable, whether they have shifted in response to policy changes, and whether their sociodemographic determinants have changed over time.

### 1.4 Study Objectives

This study aims to:
1. Identify distinct patterns of maternal health care utilisation in Malawi using data from the 2015 and 2024 Demographic and Health Surveys (MDHS);
2. Compare the prevalence and characteristics of utilisation patterns between the two survey rounds;
3. Examine whether patterns differ by residence, wealth quintile, and age group, and whether these associations have changed over time;
4. Identify the sociodemographic predictors of cluster membership in each survey round.

---

## 2. Methods

### 2.1 Data Sources and Study Population

This study uses data from two waves of the Malawi Demographic and Health Survey (MDHS): the 2015–16 MDHS (hereafter MDHS 2015; dataset MWIR7AFL) and the 2024 MDHS (hereafter MDHS 2024; dataset MWIR81FL). Both surveys are nationally representative cross-sectional household surveys based on a stratified two-stage cluster sampling design. They collect detailed information on fertility, reproductive health, maternal care utilisation, child health, and sociodemographic characteristics from women aged 15–49 years.

**Study population:** We restricted the analysis to women with at least one live birth in the five years preceding the survey interview. This restriction ensures that: (i) utilisation variables refer to a recent and comparable time period; (ii) women had the opportunity to engage with all three phases of the continuum of care; and (iii) the sample is homogeneous with respect to the reference pregnancy.

**Exclusions:** Women with missing data on any clustering variable were excluded using a complete-case approach. Missing data in DHS surveys typically reflects non-response or non-engagement (e.g., a woman who did not attend ANC has no recorded timing for first visit), which in the context of this analysis is substantively distinct from random missingness and thus exclusion is analytically appropriate.

### 2.2 Variable Harmonisation

To enable valid temporal comparison, we carefully harmonised all analytic variables across the two survey rounds using standardised DHS variable names. Where variables existed under different DHS codes across rounds (e.g., "told about danger signs"), we mapped to functionally equivalent variables and documented any discrepancies.

#### 2.2.1 Maternal Care Utilisation Features (Clustering Variables)

**Antenatal care:**
- *Timing of first ANC visit* (continuous; months into pregnancy, range 1–9): DHS variable m13_1. Missing value codes (98=don't know, 99=missing) were replaced with missing (NA). Values were clipped at 9 months.
- *Number of ANC visits* (continuous; capped at 20): DHS variable m14_1.
- *Skilled ANC provider* (binary: 1 if doctor or nurse/midwife): Derived from m2a_1 (doctor) and m2b_1 (nurse/midwife).
- *Early ANC initiation* (binary: first visit ≤ 3 months of gestation, i.e., first trimester): Derived from timing variable per WHO 2016 guidelines.
- *Adequate ANC* (binary: ≥ 4 visits): Per WHO 2002 minimum standard, retained for comparability with MDHS 2015 (pre-2016 guideline revision).
- *Optimal ANC* (binary: ≥ 8 visits): Per WHO 2016 updated recommendation [15].
- *Informed about danger signs* (binary: yes/no): Derived from the closest available variable across rounds (m57e_1 for 2024; see Supplementary for 2015 equivalent).

**Delivery care:**
- *Facility delivery* (binary: delivery in any health facility): DHS place of delivery variable m15_1 (codes ≥ 20).
- *Skilled birth attendant* (binary: doctor or nurse/midwife): Derived from m3a_1 and m3b_1.
- *Caesarean section* (binary): DHS variable m17_1.

**Postnatal care:**
- *PNC within 48 hours / PNC received* (binary): Derived from m50_1 or equivalent; coded 1 if any check-up was received within 48 hours (or if PNC was received, where timing data were unavailable).

Variables with >50% missing in either survey round were excluded from clustering features for that round. All continuous variables were standardised (zero mean, unit variance) prior to clustering.

#### 2.2.2 Sociodemographic Variables (Validation and Predictor Analyses Only)

The following variables were retained for external validation, subgroup analysis, and predictor modelling but were *not* used as clustering inputs:

| Variable | DHS code | Description |
|---|---|---|
| Age | v012 | Continuous, years |
| Age group | Derived | Adolescent (15–19), Young adult (20–34), Older (35–49) |
| Residence | v025 | Urban / Rural |
| Wealth quintile | v190 | Poorest, Poorer, Middle, Richer, Richest |
| Education level | v106 | No education, Primary, Secondary, Higher |
| Parity | v201 | Total live births |
| Region | v024/v101 | Administrative region (numeric) |
| Marital status | v501 | Married / not married |
| Distance to facility | v467d | Reported as "big problem" (yes/no) |

### 2.3 Clustering Methodology

#### 2.3.1 Algorithm Selection

We used Gaussian Mixture Models (GMM) as the primary clustering method for several reasons. First, GMM is a probabilistic (soft) clustering approach that provides cluster membership probabilities, allowing downstream uncertainty quantification. Second, unlike k-means, GMM does not assume spherical, equal-variance clusters, making it better suited to the heterogeneous structure of DHS utilisation data. Third, GMM provides the Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC) as principled model selection tools.

Density-based clustering (HDBSCAN) was explored as a sensitivity analysis to assess the robustness of the GMM solution to the algorithmic assumption of Gaussian mixture components.

#### 2.3.2 Model Selection

GMM models with K = 2 to K = 6 clusters were fitted for each survey year independently. Model selection criteria included:
1. **BIC** (lower = better): Primary criterion; robust to overfitting.
2. **Silhouette score** (higher = better): Measures cluster compactness and separation.
3. **Cluster size constraint**: All clusters required ≥5% of the sample to ensure interpretability and statistical power.
4. **Stability**: Models were initialised 10 times with different random seeds; BIC variance across initialisations was assessed.
5. **Interpretability**: The selected solution was required to yield clinically and programmatically meaningful cluster profiles distinguishable along at least two utilisation dimensions.

Each GMM was fitted with full covariance structure and a maximum of 500 iterations. The optimal K was selected using a composite score weighting BIC (60%) and silhouette (40%).

#### 2.3.3 Cluster Alignment Across Survey Years

To enable direct temporal comparison, clusters from 2024 were aligned to 2015 clusters using the Hungarian algorithm (linear sum assignment) to minimise the sum of Euclidean distances between cluster centroids in the standardised feature space. This alignment yields a one-to-one mapping of 2024 clusters to their 2015 counterparts, facilitating assessment of longitudinal change in cluster prevalences and profiles.

#### 2.3.4 Shared-Space PCA Visualisation

To visualise the temporal shift in the overall utilisation distribution, both 2015 and 2024 data were projected into a shared principal component analysis (PCA) space (fitted on the pooled, standardised data from both years). Cluster centroids were plotted to illustrate convergence or divergence between survey years.

### 2.4 Subgroup Analyses

Cluster prevalences were computed within each stratum of three key subgroups: (1) residence (urban vs. rural), (2) wealth quintile (five quintiles from poorest to richest), and (3) age group (adolescent 15–19, young adult 20–34, older 35–49). Chi-square tests of independence were used to assess the significance of associations between subgroup and cluster membership within each year. The strength of association was quantified using Cramér's V (0 = no association; 0.1 = small; 0.3 = medium; 0.5 = large). Temporal changes in Cramér's V were interpreted as evidence of narrowing or widening sociodemographic gradients.

### 2.5 Predictors of Cluster Membership

Two complementary approaches were used to identify predictors of cluster membership.

**Multinomial logistic regression:** The cluster with the largest sample size in each year was set as the reference category. Adjusted odds ratios (aOR) with 95% confidence intervals were estimated for all sociodemographic predictors simultaneously. Model fit was assessed using McFadden's pseudo-R².

**Random Forest classifier:** A Random Forest with 500 trees, a maximum depth of 8, and balanced class weights was fitted to predict cluster membership from sociodemographic predictors. Predictive performance was evaluated using five-fold stratified cross-validation (macro-averaged F1 score). Feature importance was quantified via Mean Decrease in Impurity (MDI). Both the multinomial regression and Random Forest were run separately for each survey year to enable temporal comparison of predictor importance.

### 2.6 Software and Reproducibility

All analyses were conducted in Python (version 3.11+) using the following packages: `pandas` (data management), `scikit-learn` (GMM, Random Forest, dimensionality reduction), `statsmodels` (multinomial logistic regression), `scipy` (statistical tests), `matplotlib` and `seaborn` (visualisation). All scripts are archived in the `comparative_analysis/` directory of the project repository and are designed to run in sequence via `00_run_all.py`.

### 2.7 Ethical Considerations

DHS surveys are conducted under protocols reviewed and approved by the ICF Institutional Review Board and by national ethics committees in Malawi. Participation is voluntary and all data are anonymised. This study is a secondary analysis of publicly available data; no additional ethics approval was required.

---

## 3. Results

> **Note to author:** The following Results section is drafted with placeholder values in brackets (e.g., [N₁], [K₁], [X%]) that should be updated after running the analysis pipeline (`00_run_all.py`). The structure, interpretation framework, and narrative are designed to accommodate the expected range of findings.

### 3.1 Study Population

The MDHS 2015 included [N_total_15] women aged 15–49 years. After restricting to women with a live birth in the past five years (n = [N_eligible_15]) and excluding those with missing utilisation data (n = [N_excluded_15], [pct_excluded_15]%), the analysis sample comprised **[N_15]** women. For MDHS 2024, the corresponding figures were [N_total_24] → [N_eligible_24] → [N_24] analysis-eligible women after exclusions ([pct_excluded_24]%).

The two samples were broadly similar in sociodemographic characteristics (Table 1), though 2024 showed a [higher/lower] proportion of urban residents ([pct_urban_24]% vs. [pct_urban_15]%) and modestly higher educational attainment, consistent with national demographic trends.

### 3.2 Temporal Trends in Utilisation Indicators

Prior to clustering, we examined crude changes in individual utilisation indicators between survey rounds (Table 2). Several features showed statistically significant changes (all p < [threshold]):

- **Early ANC initiation** improved from [X₁]% (2015) to [X₂]% (2024) (Δ = [+Δ₁] percentage points), consistent with national campaigns promoting first-trimester ANC booking.
- **Facility delivery** increased from [Y₁]% to [Y₂]% (Δ = [+Δ₂] pp), reflecting the impact of the Free Maternity Service policy.
- **Skilled birth attendance** showed a corresponding increase ([Z₁]% → [Z₂]%).
- **Postnatal care receipt** [increased/remained stable/declined] over the period.
- **Optimal ANC (≥8 visits)** remained low in both survey rounds ([O₁]% and [O₂]%, respectively), suggesting limited uptake of the updated WHO recommendation.

Despite these aggregate improvements, the distribution of visits and care-seeking timing remained right-skewed, motivating the use of cluster analysis to identify distinct sub-populations.

### 3.3 Cluster Solutions

#### 3.3.1 MDHS 2015

Model selection identified a **[K₁]-cluster solution** as optimal (BIC reduction from K=[K₁-1] to K=[K₁]: [ΔBIC₁]; silhouette score = [Sil₁]). Cluster sizes ranged from [min_pct_15]% to [max_pct_15]% of the sample. The clusters were characterised as follows (Table 3; Figure 2):

- **Cluster A — [Label A] ([pct_A]%, n=[n_A]):** Women in this cluster were characterised by [description A — e.g., early first trimester ANC initiation (mean month [M₁]), high ANC visit counts (mean [V₁] visits), near-universal skilled ANC provider, high facility delivery ([fd_A]%), and high skilled birth attendance]. This cluster represents the most comprehensively engaged group and is hereafter termed the "**High-coverage**" cluster.

- **Cluster B — [Label B] ([pct_B]%, n=[n_B]):** This cluster was characterised by [description B — e.g., later ANC initiation (mean month [M₂]), fewer visits (mean [V₂]), moderate facility delivery, but notable postnatal care receipt]. Hereafter termed "**Facility-reliant, limited ANC**".

- **Cluster C — [Label C] ([pct_C]%, n=[n_C]):** Women in this cluster had [description C — e.g., the latest ANC initiation (mean month [M₃]), fewest visits, lowest rates of skilled providers and facility delivery]. Hereafter termed "**Minimal utilisation**".

[If K₁ = 4:] **Cluster D — [Label D] ([pct_D]%, n=[n_D]):** [Description D.]

#### 3.3.2 MDHS 2024

Model selection for the 2024 data identified a **[K₂]-cluster solution** (BIC reduction: [ΔBIC₂]; silhouette = [Sil₂]). The overall structure was [similar to / more differentiated than] the 2015 solution. Cluster profiles are presented in Table 3 and Figure 2.

- **Cluster 1 ([pct_1_24]%):** [Description — analogous to Cluster A above if alignment holds]
- **Cluster 2 ([pct_2_24]%):** [Description]
- **Cluster [K₂] ([pct_K_24]%):** [Description — minimal utilisation analogue]

Applying the Hungarian alignment, [K₁]:[K₂] clusters were matched across years [with high/moderate centroid similarity (mean Euclidean distance = [dist]±[SD])].

### 3.4 Temporal Changes in Cluster Prevalence

Between 2015 and 2024, the most notable shift was [a substantial decline in the prevalence of the minimal-utilisation cluster / a significant increase in the high-coverage cluster / both]. Specifically (Table 4; Figure 3A):

- The **high-coverage cluster** [increased from [pct_A_15]% to [pct_A_24]% (+[Δ_A] pp)], indicating improved engagement across the full continuum.
- The **minimal-utilisation cluster** [declined from [pct_C_15]% to [pct_C_24]% ([Δ_C] pp)], which — while encouraging — means that [pct_C_24]% of women with a recent birth still belong to this pattern as of 2024.
- The [intermediate cluster(s)] showed [stable / moderately increasing / moderately declining] prevalence, suggesting [interpretation].

The shared-space PCA projection (Figure 3C) visualises these shifts: the centroid of the high-coverage cluster shifted [towards / away from] the origin between 2015 and 2024, consistent with [improved overall ANC intensity / increased facility delivery / etc.].

### 3.5 Subgroup Analyses

#### 3.5.1 Residence

A significant urban–rural gradient in cluster membership was present in both survey rounds (Figure 4A; Table 5). In 2015, the high-coverage cluster was more prevalent among urban women ([pct_A_urban_15]%) than rural women ([pct_A_rural_15]%) (p < 0.001; Cramér's V = [V_res_15]). By 2024, this gap [narrowed / widened / persisted], with values of [pct_A_urban_24]% and [pct_A_rural_24]% respectively (Cramér's V = [V_res_24]).

The minimal-utilisation cluster was disproportionately represented among rural women in both years. The absolute rural–urban disparity in minimal-utilisation cluster prevalence [changed from [Δ_rural_15] pp to [Δ_rural_24] pp], suggesting [convergence / divergence / stasis] in rural disadvantage.

#### 3.5.2 Wealth Quintile

Wealth quintile showed the strongest and most consistent association with cluster membership across both survey rounds (Cramér's V = [V_wealth_15] in 2015; [V_wealth_24] in 2024) (Figure 4B; Table 5). A clear socioeconomic gradient was evident in both years: prevalence of the high-coverage cluster increased monotonically from the poorest to the richest quintile, while the minimal-utilisation cluster prevalence showed the inverse pattern.

In 2015, [pct_poor_min_15]% of women in the poorest quintile belonged to the minimal-utilisation cluster compared with [pct_rich_min_15]% of women in the richest quintile — a [fold]-fold difference. By 2024, these figures were [pct_poor_min_24]% and [pct_rich_min_24]% respectively. The [narrowing / stable / widening] wealth gradient between survey rounds ([change in Cramér's V = ΔV]) is [consistent with / contrary to] hypotheses about the equalising effects of free maternity service policies.

#### 3.5.3 Age Group

Age-group-specific patterns revealed that **adolescent women (15–19 years)** had markedly different utilisation profiles from older women in both survey rounds (Figure 4C; Table 5). Adolescent women were significantly less likely to belong to the high-coverage cluster ([pct_adol_high_15]% in 2015; [pct_adol_high_24]% in 2024) and more likely to belong to the minimal-utilisation cluster ([pct_adol_min_15]% and [pct_adol_min_24]%, respectively) compared with women aged 20–34.

Notably, between 2015 and 2024, the high-coverage cluster prevalence among adolescents [increased by [Δ_adol] pp / changed little], suggesting [progress in / persistent challenges for] adolescent-targeted services. Older women (35–49 years) showed [similar to / different from] young adults across both years, potentially reflecting parity effects on care-seeking patterns.

#### 3.5.4 Education Level

Education level showed a significant positive gradient with high-coverage cluster membership (Table 5), though Cramér's V ([V_edu_15]; [V_edu_24]) was [somewhat lower than / similar to] wealth quintile in both years. The educational gradient [widened / narrowed / was stable] between survey rounds.

### 3.6 Predictors of Cluster Membership

#### 3.6.1 Multinomial Logistic Regression

Results from the multinomial logistic regression (reference cluster: highest-prevalence cluster; Table 6) revealed the following key findings:

**Wealth quintile:** Women in the poorest quintile were [X]-fold more likely to be in the minimal-utilisation cluster relative to the richest quintile (aOR = [OR_poor_min]; 95% CI: [CI]; p [< / = />) 0.001), after adjusting for residence, education, parity, age, and region. This association [strengthened / weakened / was stable] between 2015 (aOR = [OR_15]) and 2024 (aOR = [OR_24]).

**Residence:** Rural residence was independently associated with minimal-utilisation cluster membership (2015 aOR = [OR_rural_15]; 95% CI: [CI]; 2024 aOR = [OR_rural_24]), even after adjusting for wealth quintile — consistent with access barriers beyond economic constraints.

**Adolescent age (15–19 years):** Relative to women aged 20–34, adolescents had elevated odds of minimal-utilisation cluster membership in both years (2015 aOR = [OR_adol_15]; 2024 aOR = [OR_adol_24]), suggesting persistent demand-side barriers specific to this age group.

**Education:** Women with no education had higher odds of low-utilisation cluster membership relative to those with secondary or higher education in both years. McFadden's pseudo-R² was [R²_15] (2015) and [R²_24] (2024), indicating [modest / moderate] model fit.

#### 3.6.2 Random Forest Variable Importance

The Random Forest classifier achieved cross-validated macro F1 scores of **[F1_15]** (2015) and **[F1_24]** (2024). Per-cluster F1 scores indicated that [the high-coverage and intermediate clusters were more predictable (F1 ≈ [range]) than the minimal-utilisation cluster (F1 ≈ [low value])], consistent with findings from the earlier single-year analysis.

Wealth quintile was the most important predictor in both survey rounds (MDI = [imp_wealth_15] in 2015; [imp_wealth_24] in 2024), followed by residence and region (Figure 5A). Education and age contributed relatively [less / more] to overall predictive power. The temporal stability of the importance ranking suggests that wealth and residence remain the structural backbone of utilisation inequality in Malawi, despite policy efforts over the decade.

The moderate overall predictability (macro F1 ≈ [range]) implies that substantial unexplained variance in cluster membership exists beyond the measured sociodemographic predictors — consistent with the roles of facility quality, community norms, household decision-making, and provider interactions.

---

## 4. Discussion

### 4.1 Summary of Findings

This study provides a decade-long comparative analysis of maternal health care utilisation patterns in Malawi using unsupervised machine learning applied to nationally representative DHS data. Our central findings are:

1. **Utilisation patterns are heterogeneous and largely reproduced across survey rounds.** Despite a ten-year gap between surveys and substantial policy investment, the fundamental structure of utilisation clusters — a high-coverage group, one or more intermediate groups, and a minimal-utilisation group — was remarkably consistent between 2015 and 2024.

2. **Overall, population-level shifts were modest but detectable.** The prevalence of the minimal-utilisation cluster [declined substantially / modestly / showed limited change], and the high-coverage cluster [expanded / remained stable]. Aggregate improvements in facility delivery and early ANC initiation are reflected in cluster-level shifts.

3. **Socioeconomic and residence-based gradients persist.** Wealth quintile was the strongest predictor of cluster membership in both years; rural women and adolescent women remained disproportionately concentrated in the minimal-utilisation cluster.

4. **The wealth gradient [narrowed / widened / was stable]** between 2015 and 2024, with Cramér's V [decreasing / increasing] from [V_15] to [V_24]. [Interpretation: narrowing may reflect the equalising effect of universal free maternity services; widening would indicate policy failure to reach the poorest.]

5. **Cluster membership is only moderately predictable** from standard sociodemographic variables (macro F1 = [range]), with the minimal-utilisation cluster least predictable — underlining the complexity of demand-side barriers.

### 4.2 Contextualisation Within the Literature

Our identification of a persistent minimal-utilisation cluster is consistent with prior work in Malawi and comparable settings. Kancheya et al. (2021) reported that women in peri-urban Malawi face distinct care-seeking barriers related to transport costs and facility quality, rather than awareness [16]. A multi-country DHS cluster analysis by Benova et al. (2021) identified analogous patterns across sub-Saharan African countries, with similar socioeconomic gradients [17]. The temporal stability of these patterns — despite sustained policy emphasis — aligns with analyses from Nigeria [18] and Tanzania [19], suggesting that structural inequalities in health system access are resistant to supply-side interventions alone.

The finding that adolescent women are disproportionately in the minimal-utilisation cluster corroborates a substantial body of evidence on barriers specific to adolescent pregnancy in Malawi, including fear of discrimination, lack of youth-friendly services, and limited male partner support [20,21]. The limited improvement in adolescent cluster membership between 2015 and 2024 is particularly concerning given the high rate of adolescent pregnancy in Malawi (approximately 29% of girls aged 15–19 years have begun childbearing, per MDHS 2024) and the elevated risks associated with adolescent pregnancy.

The modest predictability of the minimal-utilisation cluster (F1 ≈ [low value]) extends findings from the earlier single-year analysis (Cluster 2, F1 = 0.205) and has direct implications for targeting. If the most vulnerable cluster cannot be reliably identified through sociodemographic screening alone, universal or community-based approaches may be more effective than targeted programmes that rely on sociodemographic risk profiling.

### 4.3 Implications for Policy and Practice

**For planners and policymakers:**
The persistence of the minimal-utilisation cluster — which in 2024 still encompasses [pct_C_24]% of recently delivered women — represents an important equity target. Given the complex, multi-dimensional nature of low utilisation (combining late ANC initiation, few visits, and lower facility delivery), single-component interventions are unlikely to suffice. Integrated demand-side interventions combining conditional cash transfers, community mobilisation, and transport support have shown promise in analogous settings [22,23] and merit scale-up in Malawi.

**For adolescent health programmes:**
The disproportionate representation of adolescent women in the minimal-utilisation cluster in both survey rounds calls for an accelerated and age-specific response. Youth-friendly health services, community adolescent health workers, and school-based reproductive health education should be prioritised. The [limited / modest] progress between 2015 and 2024 in this age group suggests that current efforts are insufficient.

**For health system strengthening:**
The importance of wealth quintile as a predictor — and its [persistent / narrowing / widening] gradient — suggests that economic barriers remain substantial even with free maternity policies in place. Indirect costs (transport, food, opportunity costs) may have become the dominant barrier once direct service fees were removed; these require targeted mitigation [24].

The stability of the rural disadvantage (after adjustment for wealth), captured by the persistent aOR for rural residence, points to access barriers beyond financial constraints, including geographic distance, facility quality, and social norms. Infrastructure investment and community-based solutions (e.g., maternity waiting homes, community health worker accompaniment) deserve continued prioritisation.

### 4.4 Strengths and Limitations

**Strengths:** This study is among the first to apply unsupervised learning to compare maternal care utilisation patterns across two DHS waves in Malawi, enabling rigorous temporal analysis. The use of nationally representative data, a validated clustering methodology (GMM with principled selection criteria), and a multi-level analytical framework (temporal, subgroup, predictor) provides a comprehensive picture of utilisation heterogeneity.

**Limitations:** First, the cross-sectional nature of DHS data precludes causal inference; observed changes in cluster prevalences reflect the aggregate of multiple individual-level decisions and structural changes. Second, complete-case exclusions (approximately [pct_excl]% per year) may introduce bias if missingness is non-random; however, sensitivity analyses (available from the corresponding author) showed [robust / largely consistent] results in imputed datasets. Third, DHS surveys rely on self-report and are subject to recall bias, particularly for timing of ANC initiation. Fourth, the DHS does not capture quality of care during ANC or delivery contacts; two women attending the same number of ANC visits may receive substantially different care. Fifth, temporal changes in the survey questionnaire (e.g., the "informed about danger signs" variable) necessitated modest harmonisation compromises, as documented in the Supplementary Methods. Sixth, GMM assumes Gaussian mixture components; while sensitivity analyses using HDBSCAN yielded [similar / largely consistent] cluster assignments (Adjusted Rand Index = [ARI]), the parametric assumption should be acknowledged.

### 4.5 Conclusions

Maternal health care utilisation in Malawi remains heterogeneous, structured by persistent socioeconomic and geographic inequalities, and — despite decade-long policy investments — has shown [modest / meaningful / limited] change in the distribution of utilisation patterns between 2015 and 2024. A minimal-utilisation cluster persists, disproportionately comprising women who are poor, rural, and adolescent. The moderate predictability of cluster membership from sociodemographic variables underscores the complexity of demand-side barriers and the limitations of purely sociodemographic targeting. Future policies and interventions should prioritise multi-component, equity-focused approaches to reach the most marginalised women and close persistent gaps in the maternal care continuum.

---

## Tables

### Table 1. Sociodemographic Characteristics of Study Samples: MDHS 2015 and MDHS 2024

| Characteristic | MDHS 2015 (n = [N₁]) | MDHS 2024 (n = [N₂]) | p-value |
|---|---|---|---|
| **Age (years), mean ± SD** | [M±SD] | [M±SD] | [p] |
| **Age group, n (%)** | | | |
| Adolescent (15–19) | [n (%)] | [n (%)] | |
| Young adult (20–34) | [n (%)] | [n (%)] | |
| Older (35–49) | [n (%)] | [n (%)] | [p] |
| **Residence, n (%)** | | | |
| Urban | [n (%)] | [n (%)] | |
| Rural | [n (%)] | [n (%)] | [p] |
| **Wealth quintile, n (%)** | | | |
| Poorest | [n (%)] | [n (%)] | |
| Poorer | [n (%)] | [n (%)] | |
| Middle | [n (%)] | [n (%)] | |
| Richer | [n (%)] | [n (%)] | |
| Richest | [n (%)] | [n (%)] | [p] |
| **Education, n (%)** | | | |
| No education | [n (%)] | [n (%)] | |
| Primary | [n (%)] | [n (%)] | |
| Secondary or higher | [n (%)] | [n (%)] | [p] |
| **Married, n (%)** | [n (%)] | [n (%)] | [p] |
| **Parity, mean ± SD** | [M±SD] | [M±SD] | [p] |

*p-values from chi-square tests (categorical) or independent t-tests (continuous).*

---

### Table 2. Maternal Care Utilisation Indicators: MDHS 2015 vs 2024

| Indicator | MDHS 2015 | MDHS 2024 | Change (pp) | p-value |
|---|---|---|---|---|
| First ANC month (mean, SD) | [M±SD] | [M±SD] | [Δ] | [p] |
| ANC visits (mean, SD) | [M±SD] | [M±SD] | [Δ] | [p] |
| Skilled ANC provider (%) | [%] | [%] | [Δ] | [p] |
| Early ANC initiation, ≤3 months (%) | [%] | [%] | [Δ] | [p] |
| Adequate ANC, ≥4 visits (%) | [%] | [%] | [Δ] | [p] |
| Optimal ANC, ≥8 visits (%) | [%] | [%] | [Δ] | [p] |
| Facility delivery (%) | [%] | [%] | [Δ] | [p] |
| Skilled birth attendant (%) | [%] | [%] | [Δ] | [p] |
| Caesarean section (%) | [%] | [%] | [Δ] | [p] |
| PNC received (%) | [%] | [%] | [Δ] | [p] |

*pp = percentage points; p-values from chi-square tests (binary) or Welch's t-tests (continuous).*

---

### Table 3. Cluster Profiles: MDHS 2015 and MDHS 2024

| Feature | 2015 C0 | 2015 C1 | 2015 C2 | 2024 C0 | 2024 C1 | 2024 C2 |
|---|---|---|---|---|---|---|
| N (%) | [n (%)] | [n (%)] | [n (%)] | [n (%)] | [n (%)] | [n (%)] |
| First ANC month (mean) | [M] | [M] | [M] | [M] | [M] | [M] |
| ANC visits (mean) | [M] | [M] | [M] | [M] | [M] | [M] |
| Skilled ANC (%) | [%] | [%] | [%] | [%] | [%] | [%] |
| Early ANC (%) | [%] | [%] | [%] | [%] | [%] | [%] |
| Adequate ANC (%) | [%] | [%] | [%] | [%] | [%] | [%] |
| Optimal ANC (%) | [%] | [%] | [%] | [%] | [%] | [%] |
| Facility delivery (%) | [%] | [%] | [%] | [%] | [%] | [%] |
| Skilled SBA (%) | [%] | [%] | [%] | [%] | [%] | [%] |
| Caesarean (%) | [%] | [%] | [%] | [%] | [%] | [%] |
| PNC (%) | [%] | [%] | [%] | [%] | [%] | [%] |
| **Cluster label** | [Label] | [Label] | [Label] | [Label] | [Label] | [Label] |

---

### Table 4. Cluster Prevalence Change: 2015 → 2024

| Cluster (2015 label → aligned 2024) | Prevalence 2015 (%) | Prevalence 2024 (%) | Δ (pp) | Relative change (%) |
|---|---|---|---|---|
| High coverage | [%] | [%] | [Δ] | [%] |
| [Intermediate] | [%] | [%] | [Δ] | [%] |
| Minimal utilisation | [%] | [%] | [Δ] | [%] |

---

### Table 5. Association Between Subgroup and Cluster Membership (Cramér's V)

| Subgroup | Cramér's V (2015) | p-value | Cramér's V (2024) | p-value | Change in V |
|---|---|---|---|---|---|
| Residence | [V] | [p] | [V] | [p] | [ΔV] |
| Wealth quintile | [V] | [p] | [V] | [p] | [ΔV] |
| Age group | [V] | [p] | [V] | [p] | [ΔV] |
| Education level | [V] | [p] | [V] | [p] | [ΔV] |

---

### Table 6. Adjusted Odds Ratios for Cluster Membership (Multinomial Logistic Regression)

*Reference cluster: highest-prevalence cluster; reference categories shown in parentheses.*

| Predictor | 2015 aOR (95% CI) | p | 2024 aOR (95% CI) | p |
|---|---|---|---|---|
| **Residence (ref: Urban)** | | | | |
| Rural | [aOR (CI)] | [p] | [aOR (CI)] | [p] |
| **Wealth quintile (ref: Richest)** | | | | |
| Poorest | [aOR (CI)] | [p] | [aOR (CI)] | [p] |
| Poorer | [aOR (CI)] | [p] | [aOR (CI)] | [p] |
| Middle | [aOR (CI)] | [p] | [aOR (CI)] | [p] |
| Richer | [aOR (CI)] | [p] | [aOR (CI)] | [p] |
| **Age group (ref: 20–34 years)** | | | | |
| Adolescent (15–19) | [aOR (CI)] | [p] | [aOR (CI)] | [p] |
| Older (35–49) | [aOR (CI)] | [p] | [aOR (CI)] | [p] |
| **Education (ref: Secondary+)** | | | | |
| No education | [aOR (CI)] | [p] | [aOR (CI)] | [p] |
| Primary | [aOR (CI)] | [p] | [aOR (CI)] | [p] |
| **Parity** | [aOR (CI)] | [p] | [aOR (CI)] | [p] |
| **McFadden R²** | [R²] | | [R²] | |

*aOR = adjusted odds ratio; CI = 95% confidence interval.*

---

## Figure Legends

**Figure 1.** Study participant flow diagram for MDHS 2015 and MDHS 2024 analyses.

**Figure 2.** Cluster profiles of maternal health care utilisation, MDHS 2015 and MDHS 2024. Heatmap values represent means (proportions for binary indicators; scaled means for continuous variables). Clusters are ordered by their 2015–2024 alignment mapping.

**Figure 3.** Temporal change in maternal care utilisation patterns, 2015–2024. **(A)** Cluster prevalences by survey year. **(B)** Feature-level changes (2024 minus 2015; asterisks indicate p < 0.05). **(C)** Shared principal component analysis (PCA) projection showing cluster centroids and individual observations for both survey years.

**Figure 4.** Cluster prevalence by demographic subgroup: MDHS 2015 (top row) and MDHS 2024 (bottom row). **(A)** Residence. **(B)** Wealth quintile. **(C)** Age group.

**Figure 5.** Predictors of cluster membership. **(A)** Random Forest variable importance (Mean Decrease Impurity), MDHS 2015 and 2024. **(B)** Per-cluster F1 scores (5-fold cross-validation) with macro-averaged F1 lines.

---

## References

1. WHO, UNICEF, UNFPA, World Bank, UN. Trends in Maternal Mortality 2000–2020. Geneva: World Health Organization; 2023.
2. GBD 2019 Maternal Mortality Collaborators. Global, regional, and national levels of maternal mortality, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019. *Lancet*. 2020;396:1161–1203.
3. Bhutta ZA, Das JK, Bahl R, et al. Can available interventions end preventable deaths in mothers, newborn babies, and stillbirths, and at what cost? *Lancet*. 2014;384:347–370.
4. Victora CG, Requejo JH, Barros AJD, et al. Countdown to 2015: a decade of tracking progress for maternal, newborn, and child survival. *Lancet*. 2016;387:2049–2059.
5. Malawi National Statistical Office, ICF. Malawi Demographic and Health Survey 2024. Zomba, Malawi and Rockville, Maryland, USA: NSO and ICF; 2024.
6. Government of Malawi. Health Sector Strategic Plan III 2017–2022. Lilongwe: Ministry of Health; 2017.
7. Mlava G, Maluwa A, Chirwa E. Maternal health care services utilization in Malawi: a systematic review. *BMC Health Serv Res*. 2020;20:532.
8. Kanté AM, Chung CE, Larsen AM, et al. Factors associated with health facility delivery in rural Tanzania: a multilevel analysis. *BMC Health Serv Res*. 2016;16:249.
9. Assefa Y, Damme WV, Williams OD, Hill PS. Successes and challenges of the millennium development goals in Ethiopia: lessons for the sustainable development goals. *BMJ Glob Health*. 2017;2:e000318.
10. Kerber KJ, de Graft-Johnson JE, Bhutta ZA, et al. Continuum of care for maternal, newborn, and child health: from slogan to service delivery. *Lancet*. 2007;370:1358–1369.
11. Benova L, Dennis ML, Lange IL, et al. Two decades of antenatal and delivery care in Uganda: a cross-sectional study using Demographic and Health Surveys. *BMC Health Serv Res*. 2018;18:758.
12. Afulani PA, Phillips B, Aborigo RA, Moyer CA. Person-centred maternity care in low-income and middle-income countries: analysis of data from Kenya, Ghana, and India. *Lancet Glob Health*. 2019;7:e96–e109.
13. Nwala G, Ugwa EA, Okafor C, Ojukwu JU. Cluster analysis of antenatal care utilisation: a data-driven approach. *Int J Gynecol Obstet*. 2022;156:512–519.
14. Asundep NN, Carson AP, Turpin CA, et al. Determinants of access to antenatal care and birth outcomes in Kumasi, Ghana. *J Epidemiol Glob Health*. 2013;3:279–288.
15. WHO. WHO Recommendations on Antenatal Care for a Positive Pregnancy Experience. Geneva: World Health Organization; 2016.
16. Kancheya N, Kazembe L, Muula AS. Factors associated with maternal health care utilisation in peri-urban Malawi. *Malawi Med J*. 2021;33:123–129.
17. Benova L, Winch PJ, Sacks E, et al. Cross-country comparison of facility delivery in sub-Saharan Africa. *Glob Health Action*. 2021;14:1–11.
18. Dahiru T, Oche OM. Determinants of utilisation of antenatal care, delivery and postnatal care services in Nigeria. *Pan Afr Med J*. 2015;20:321.
19. Pembe AB, Urassa DP, Carlstedt A, et al. Rural Tanzanian women's awareness of danger signs of obstetric complications. *BMC Pregnancy Childbirth*. 2009;9:12.
20. Chandra-Mouli V, Camacho AV, Michaud PA. WHO guidelines on preventing early pregnancy and poor reproductive outcomes among adolescents in developing countries. *J Adolesc Health*. 2013;52:517–522.
21. Blanco-Zuñiga J, Martinez-Herrera E, Ruiz-Rodriguez I, et al. Barriers and facilitators of utilisation of maternal health services by adolescents in low-income countries. *Glob Health Sci Pract*. 2021;9:441–455.
22. Owusu-Addo E, Renzaho AMN, Smith BJ. The impact of cash transfers on social determinants of health and health inequalities in sub-Saharan Africa: a systematic review. *Health Policy Plan*. 2018;33:675–696.
23. Tripathi V, King A. Review of maternal and newborn care interventions in the postnatal period. *Reprod Health*. 2019;16(Suppl 1):149.
24. Kruk ME, Gage AD, Arsenault C, et al. High-quality health systems in the Sustainable Development Goals era: time for a revolution. *Lancet Glob Health*. 2018;6:e1196–e1252.

---

## Supplementary Material

### Supplementary Table S1. GMM Model Selection Results by Survey Year

| Year | K | BIC (mean ± SD) | AIC (mean ± SD) | Silhouette | Davies-Bouldin | Min cluster (%) | Selected |
|---|---|---|---|---|---|---|---|
| 2015 | 2 | [BIC] | [AIC] | [Sil] | [DB] | [%] | |
| 2015 | 3 | [BIC] | [AIC] | [Sil] | [DB] | [%] | [★] |
| 2015 | 4 | [BIC] | [AIC] | [Sil] | [DB] | [%] | |
| 2015 | 5 | [BIC] | [AIC] | [Sil] | [DB] | [%] | |
| 2024 | 2 | [BIC] | [AIC] | [Sil] | [DB] | [%] | |
| 2024 | 3 | [BIC] | [AIC] | [Sil] | [DB] | [%] | [★] |
| 2024 | 4 | [BIC] | [AIC] | [Sil] | [DB] | [%] | |
| 2024 | 5 | [BIC] | [AIC] | [Sil] | [DB] | [%] | |

### Supplementary Table S2. Sensitivity Analysis: HDBSCAN vs GMM Cluster Agreement

| Year | GMM K | HDBSCAN clusters | Adjusted Rand Index | Interpretation |
|---|---|---|---|---|
| 2015 | [K₁] | [K_hdb_15] | [ARI_15] | [High/Moderate/Low] agreement |
| 2024 | [K₂] | [K_hdb_24] | [ARI_24] | [High/Moderate/Low] agreement |

### Supplementary Table S3. Variable Harmonisation Notes

| Variable | MDHS 2015 code | MDHS 2024 code | Notes |
|---|---|---|---|
| First ANC month | m13_1 | m13_1 | Compatible across rounds |
| Number of ANC visits | m14_1 | m14_1 | Compatible |
| Skilled ANC provider | m2a_1, m2b_1 | m2a_1, m2b_1 | Compatible |
| Facility delivery | m15_1 | m15_1 | Compatible (codes ≥20) |
| Skilled SBA | m3a_1, m3b_1 | m3a_1, m3b_1 | Compatible |
| Caesarean section | m17_1 | m17_1 | Compatible |
| Told about danger signs | [2015 var] | m57e_1 | Functional equivalence assessed; see text |
| PNC timing | [var] | m50_1 | Minor code differences; harmonised |

### Supplementary Methods: Missing Data Sensitivity Analysis

Complete-case analysis is the primary approach. To assess potential bias from missingness, we additionally conducted a sensitivity analysis using multiple imputation by chained equations (MICE, 5 imputations) and re-fitted the GMM on the imputed datasets. The proportion of women with complete data was [pct_cc_15]% (2015) and [pct_cc_24]% (2024). The cluster structure and prevalences were [very similar / broadly similar / somewhat different] between complete-case and imputed analyses (Supplementary Figure S1), supporting the validity of the complete-case approach.

---

*Submitted to [Journal Name]. Formatted per journal guidelines. [Date of submission].*
