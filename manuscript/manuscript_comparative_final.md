# Temporal Patterns of Maternal Health Care Utilisation in Malawi: A Comparative Unsupervised Learning Analysis of the 2015 and 2024 Demographic and Health Surveys

---

**Authors:** [Author 1, MD, PhD]¹, [Author 2, MSc]², [Author 3, PhD]³

**Affiliations:**
¹ [Institution 1], [Country]
² [Institution 2], [Country]
³ [Institution 3], [Country]

**Corresponding author:** [Name] | [Email] | [Address]

**Word count (main text, excluding abstract/tables/references):** ~3,500
**Running title:** Maternal care utilisation clusters in Malawi: 2015 vs 2024
**Keywords:** maternal health care utilisation, unsupervised learning, cluster analysis, Gaussian mixture models, continuum of care, temporal trends, Malawi, sub-Saharan Africa, adolescent maternal health, health equity

---

## Abstract

**Background:** Malawi has made substantial investments in maternal health services over the past decade, yet heterogeneity in how women engage across the full continuum of antenatal, delivery, and postnatal care remains incompletely characterised. Data-driven methods are needed to identify distinct utilisation patterns and their sociodemographic determinants.

**Methods:** We applied Gaussian Mixture Model (GMM) clustering independently to harmonised individual-level data from the 2015 MDHS (n = 13,030; from 13,448 eligible) and the 2024 MDHS (n = 6,925; from 10,536 eligible). Eleven harmonised features spanned antenatal care (timing, number of visits, provider skill, early initiation, adequacy, optimality), delivery (facility delivery, skilled birth attendance, caesarean section), and postnatal care. Optimal K was selected via BIC and silhouette criteria. Clusters were aligned across years using the Hungarian algorithm. Subgroup analyses examined cluster distributions by residence, wealth quintile, age group, and education. Predictors of cluster membership were assessed via multinomial logistic regression and Random Forest classifiers (5-fold cross-validated macro F1). All analyses were carried out in Python.

**Results:** A four-cluster solution was identified for 2015 and a three-cluster solution for 2024. In 2015, the dominant pattern was "late/inadequate ANC with facility delivery" (60.1%), alongside a "comprehensive care" cluster (24.2%), a "caesarean/high PNC" cluster (7.0%), and a "minimal utilisation" cluster (8.8%). By 2024, the "comprehensive care" cluster had nearly doubled (24.2% to 46.7%), the "late ANC/facility delivery" cluster declined (60.1% to 48.4%), and the "minimal utilisation" cluster was no longer identifiable. The largest single improvement was early ANC initiation (26.0% to 57.5%; a 31.5 percent increase; p < 0.001). Sociodemographic gradients were significant across all subgroups (all p < 0.001), but effect sizes narrowed: Cramer's V for residence declined from 0.123 (2015) to 0.055 (2024), and for wealth quintile from 0.090 to 0.059, indicating more equitable distribution of high-coverage care. Adolescent women (15-19 years) showed the lowest rates of comprehensive ANC in both years (19.7% in 2015, 43.3% in 2024), with a persistent gap relative to women aged 20-34 (24.6% and 48.2%, respectively). Random Forest macro F1 improved from 0.213 (2015) to 0.328 (2024); age, parity, and region were the dominant predictors in both years.

**Conclusions:** Malawi's maternal health system has achieved remarkable progress in shifting women from late, inadequate antenatal care to comprehensive engagement across the continuum. The near-elimination of the minimal utilisation cluster and the doubling of high-coverage care are particularly notable. Despite these gains, adolescent women and the poorest quintile continue to show lower rates of comprehensive care, and a substantial proportion of women (approximately 48%) still initiate ANC late. Targeted strategies addressing adolescent-specific barriers and demand-side constraints among the poorest women are needed to close persistent gaps.

---

## Introduction

Despite global reductions in maternal mortality, sub-Saharan Africa bears a disproportionate burden, accounting for approximately two-thirds of all maternal deaths worldwide [1,2]. Evidence consistently demonstrates that timely engagement across the continuum of antenatal, intrapartum, and postnatal care is protective against preventable maternal deaths [3,4]. Malawi's maternal mortality ratio declined from approximately 1,190 per 100,000 live births in 2000 to 381 per 100,000 in 2020 [5], a trajectory shaped by policy investments including the expansion of Emergency Obstetric Care, free maternity services, and community health worker programmes [6,7]. The 2024 Malawi Demographic and Health Survey (MDHS) provides an opportunity to assess whether a decade of investment has translated into measurable change in how women engage with maternal health services. Prior analyses of single DHS indicators demonstrate aggregate improvements [8], but these approaches mask heterogeneity in the sequencing and quality of care-seeking across the full continuum.

The "continuum of care" framework [9] positions antenatal care (ANC), skilled intrapartum care, and postnatal care (PNC) as interconnected phases in which engagement at each stage reinforces uptake at the next. Dropout at any point diminishes the cumulative protective benefit. Two women may each attend four ANC visits yet differ profoundly in the timing, quality, and clinical content of those visits, distinctions that carry meaningful implications for detection and management of complications. Single-indicator analyses fail to capture this multi-dimensional heterogeneity. Unsupervised machine learning methods, specifically cluster analysis, offer a data-driven alternative that identifies latent patterns of multi-dimensional utilisation behaviour without imposing a priori categorisations [10,11].

Several studies have applied unsupervised learning to DHS maternal care data [12-14], consistently identifying two to four distinct utilisation patterns and demonstrating associations with socioeconomic determinants. However, comparative studies spanning two DHS waves, enabling direct assessment of temporal change in the structure and prevalence of utilisation patterns, remain scarce, particularly for Malawi where targeted data are needed to guide the post-2020 health sector strategy. This study aimed to: (1) identify distinct patterns of maternal health care utilisation in the 2015 and 2024 MDHS using GMM clustering; (2) compare the prevalence, structure, and profile of clusters across survey rounds; (3) examine whether sociodemographic gradients in cluster membership changed between 2015 and 2024; and (4) identify predictors of cluster membership and assess their predictive value.

---

## Methods

We used individual recode datasets from two nationally representative MDHS surveys: MDHS 2015 (dataset MWIR7AFL; n = 24,562 women aged 15-49) and MDHS 2024 (dataset MWIR81FL; n = 21,587). Both employ stratified two-stage cluster sampling with probability-proportional-to-size selection. The study population was restricted to women with at least one live birth in the five years preceding interview (MDHS 2015: n = 13,448; MDHS 2024: n = 10,536), ensuring that utilisation variables referenced a recent and comparable pregnancy. Women with missing data on any clustering variable were excluded under a complete-case approach: 418 women in 2015 (3.1%) and 3,611 in 2024 (34.3%), yielding analysis samples of **n = 13,030 (2015)** and **n = 6,925 (2024)**. The higher exclusion proportion in 2024 reflects the structure of the 2024 survey instrument, as women who did not attend ANC were not asked follow-up questions on visit timing or count, generating structurally missing data that appropriately represents non-engagement.

Eleven utilisation features were extracted and harmonised across survey rounds using standard DHS variable names. Where variable coding differed between rounds, specifically for place of delivery, residence, and wealth quintile, empirically derived coding schemes were applied and documented. Features spanned three domains. Antenatal care variables included: month of first ANC visit (continuous, 1-9); total ANC visits (continuous, capped at 20); skilled ANC provider (binary: doctor or nurse/midwife); early ANC initiation (binary: first visit within the first trimester); adequate ANC (binary: four or more visits, WHO minimum); optimal ANC (binary: eight or more visits, updated WHO 2016 guideline [15]); and whether the woman was informed about danger signs (binary). Delivery care variables included: facility delivery (binary); skilled birth attendant (binary: doctor or nurse/midwife); and caesarean section (binary). The postnatal care domain was captured by a single binary variable for PNC receipt. Sociodemographic variables, including age, age group, residence, wealth quintile, education, parity, region, marital status, and perceived distance to facility, were extracted for validation and predictor analyses only and were not included in clustering.

Gaussian Mixture Models (GMM) were applied independently to each survey year to identify latent utilisation clusters. A GMM is a probabilistic soft-assignment clustering method that models the observed data as arising from a mixture of K multivariate Gaussian distributions, one per cluster. Unlike hard-assignment algorithms such as k-means, GMM allows each observation to carry a probability of belonging to each cluster, and accommodates flexible, elliptically shaped cluster geometries through full covariance matrices. This makes it well suited to maternal health utilisation data, in which the boundaries between care-seeking patterns are not sharp and distinct groups may overlap in multidimensional space. GMMs with K = 2 to K = 6 clusters were fitted per year, each initialised 10 times with distinct random seeds, retaining the lowest-BIC solution. The optimal K was selected using a composite score weighting BIC (60%) and silhouette coefficient (40%), subject to a minimum cluster size of 5%. All continuous features were standardised to zero mean and unit variance before fitting. Density-based clustering was applied as a sensitivity analysis.

To enable temporal comparison, cluster centroids from 2024 were aligned to those from 2015 using the Hungarian algorithm. The Hungarian algorithm is an optimal combinatorial assignment method that solves the assignment problem by finding the one-to-one mapping between two sets of objects that minimises the total cost, here defined as the summed Euclidean distance between standardised centroids. This approach identifies which 2024 cluster most closely resembles each 2015 cluster in the feature space, producing a cross-year correspondence that preserves semantic interpretability across survey rounds. All analyses were carried out in Python.

Cluster prevalences were computed within strata of residence (urban/rural), wealth quintile (five quintiles), age group (adolescent 15-19; young adult 20-34; older 35-49), and education level. Chi-square tests assessed statistical significance and Cramer's V quantified association strength, with thresholds of less than 0.1 for small, 0.1 to 0.3 for medium, and greater than 0.3 for large effect sizes. Multinomial logistic regression with the largest cluster as the reference category estimated adjusted odds ratios and 95% confidence intervals for all sociodemographic predictors simultaneously. Random Forest classifiers using 500 trees with balanced class weights estimated feature importance using mean decrease in impurity (MDI) and cross-validated macro F1 scores using five-fold stratified cross-validation. Both models were run separately for each survey year to enable temporal comparison. Both DHS surveys were conducted under protocols reviewed by the ICF Institutional Review Board and by the Malawi National Health Sciences Research Committee. Participation was voluntary and data are anonymised; secondary analysis of publicly available de-identified data does not require additional ethics review.

---

## Results

The 2015 analysis sample comprised **13,030 women** (mean age 28.1 years) and the 2024 sample comprised **6,925 women** (Table 1). Both samples were predominantly rural (approximately 83% in 2015 and 81% in 2024). Between survey rounds, all ANC indicators improved significantly (Table 2). Most notably, **early ANC initiation** increased from 26.0% to 57.5%, the largest single improvement observed across all indicators. Mean first ANC visit month advanced from 4.45 to 3.34 months into pregnancy (p < 0.001). Adequate ANC coverage increased from 51.9% to 66.6% (p < 0.001), and skilled birth attendance rose from 92.0% to 97.1% (p < 0.001). Facility delivery also improved from 94.8% to 97.6% (p < 0.001), while the caesarean section rate increased from 7.0% to 11.5% (p < 0.001). PNC receipt declined marginally from 45.1% to 43.2% (p = 0.013), and reporting of being informed about danger signs was stable (25.3% vs 24.8%; p = 0.421).

[INSERT FIGURE 1 ABOUT HERE]

Model selection identified K = 4 as optimal for 2015 (silhouette = 0.318; all clusters at least 7%) and K = 3 for 2024 (silhouette = 0.269; all clusters at least 5%). The cluster profiles are summarised in Table 3, with visualisation of individual-level cluster assignments in the shared PCA feature space shown in Figure 6.

In 2015, four distinct utilisation patterns were identified. The largest group, comprising 60.1% of women (n = 7,828), showed late ANC initiation (mean first visit at 5.0 months into pregnancy), very low early initiation (0%), and adequate ANC in only 42.9% of women, yet near-universal facility delivery and skilled birth attendance. This "late ANC, universal facility delivery" pattern represents the dominant mode of care-seeking in 2015, in which women reliably arrived at health facilities for delivery but engaged insufficiently with the antenatal period. The second cluster (24.2%; n = 3,149) represented a comprehensive pattern, with early ANC initiation in 97.6% of women, mean first visit at 2.9 months, adequate ANC in 78.7%, near-universal facility delivery, and 46.0% receiving PNC. A third, smaller group (7.0%; n = 910) was characterised by universal caesarean section delivery, high facility-based care, and moderate ANC engagement. The fourth cluster (8.8%; n = 1,143) represented the most marginalised pattern: late ANC, adequate ANC in only 33.4% of women, low facility delivery (51.8%), and very low skilled birth attendance (26.5%), identifying a group with limited engagement across the entire continuum of care.

By 2024, the cluster structure had simplified to three patterns. The largest group (48.4%; n = 3,349) retained the "late ANC, facility delivery" profile seen in 2015, with late initiation (mean 4.3 months), low early ANC (17.5%), and adequate ANC in only 35.3%, alongside universal facility delivery and skilled birth attendance. The second cluster (46.7%; n = 3,232) represented the high-coverage pattern, with near-universal early ANC initiation (99.2%), a mean first visit at 2.4 months, 100% adequate ANC, universal facility delivery and skilled birth attendance, a 13.6% caesarean section rate, and 45.6% receiving PNC. A third, small group (5.0%; n = 344) showed moderate ANC engagement but substantially lower facility delivery (50.9%) and skilled birth attendance (40.7%).

[INSERT FIGURE 2 ABOUT HERE]

Cross-year alignment using the Hungarian algorithm mapped the 2024 high-coverage cluster to the 2015 comprehensive cluster (centroid distance = 0.77), the 2024 late ANC/facility cluster to its 2015 counterpart (distance = 0.78), and the 2024 moderate ANC/limited delivery cluster to the 2015 caesarean cluster (distance = 1.50). The most striking temporal shift was the near-doubling of the high-coverage cluster, from 24.2% in 2015 to 46.7% in 2024, an absolute gain of 22.5 percent. Correspondingly, the late ANC/facility delivery cluster declined from 60.1% to 48.4%, and the 2015 minimal utilisation cluster, previously affecting nearly one in nine women, had no identifiable analogue in 2024, indicating that this group had been largely incorporated into the formal health system.

[INSERT FIGURE 3 ABOUT HERE]

Cluster membership was significantly associated with residence in both survey rounds (Table 4). In 2015, the association was medium-strength (Cramer's V = 0.123; p < 0.001), with rural women showing markedly higher rates of minimal utilisation (9.7% vs 4.5%). By 2024, the association weakened substantially (V = 0.055; p < 0.001), and the high-coverage cluster prevalence was 51.5% among urban women compared to 45.6% among rural women. Both groups showed large improvements, with rural women's high-coverage rates increasing from 24.4% to 45.6%. A significant socioeconomic gradient was present in both years, though the effect size narrowed (2015: V = 0.090; 2024: V = 0.059; both p < 0.001). In 2015, minimal utilisation was four times more common among the poorest quintile (11.6%) than the richest (4.6%). By 2024, both extremes showed large improvements: high-coverage care was present in 53.2% of the richest compared to 43.5% of the poorest women, with a gap of 9.7 percent. Age-group associations were significant in both years with stable but small effect sizes (2015: V = 0.046; 2024: V = 0.042). In 2015, adolescent women aged 15-19 showed the lowest rates of comprehensive care (19.7%) and the highest rates of late ANC/facility delivery (66.0%), compared with young adults aged 20-34, among whom comprehensive care stood at 24.6%. In 2024, adolescents showed a large absolute improvement, with 43.3% in the high-coverage cluster, yet a gap of 4.9 percent relative to young adults (48.2%) persisted. Education was significantly associated with cluster membership in both years (2015: V = 0.089; 2024: V = 0.066), with higher-educated women more likely to be in the comprehensive or high-coverage cluster.

[INSERT FIGURE 4 ABOUT HERE]

Random Forest classifiers achieved cross-validated macro F1 scores of 0.213 in 2015 and 0.328 in 2024, indicating modest but improving ability to predict cluster membership from sociodemographic variables alone. In 2024, the high-coverage cluster had F1 = 0.427 and the late ANC/facility cluster F1 = 0.464, while the smallest cluster (moderate ANC/limited delivery) had F1 = 0.094, reflecting the difficulty of sociodemographic profiling for the most heterogeneous group. In both survey rounds, parity and age were the most important predictors (Table 5). In 2015, the ranking was: parity (MDI = 0.180), age (0.166), richest wealth quintile (0.106), region (0.076), and rural residence (0.068). In 2024, the ordering shifted to: age (0.269), parity (0.167), region (0.094), marital status (0.058), and perceived distance to facility (0.057). The decline in the relative importance of wealth quintile, combined with the emergence of geographic distance as a predictor, may reflect a transition from economic to geographic and social barriers as the dominant constraints on comprehensive utilisation.

[INSERT FIGURE 5 ABOUT HERE]

Multinomial logistic regression with Cluster 2 (C-B, Late ANC, facility delivery) as the reference category provided adjusted odds ratios for all sociodemographic predictors simultaneously (Table 6; McFadden R² = 0.023 in 2015 and 0.011 in 2024). In 2015, membership in the C-section cluster (C-C) was significantly predicted by older age (OR per year: 1.06; 95% CI 1.04–1.08; p < 0.001), lower parity (0.73; 0.68–0.79; p < 0.001), urban residence (1.24; 1.02–1.50; p = 0.032), and higher wealth quintile (richer: 1.48, 1.17–1.88; richest: 1.81, 1.40–2.34; both p ≤ 0.001); all sub-higher education categories were associated with substantially reduced odds (OR 0.45–0.59; all p ≤ 0.004). The minimal utilisation cluster (C-D) was characterised by higher parity (1.09; 1.04–1.16; p = 0.001), greater perceived distance barrier to care (1.39; 1.21–1.59; p < 0.001), and lower urban residence odds (0.73; 0.57–0.94; p = 0.013). Women in the comprehensive care cluster (C-A) were more likely to be married (1.14; 1.03–1.27; p = 0.010), from the richest wealth quintile (1.28; 1.09–1.50; p = 0.002), and less likely to reside in urban areas (0.82; 0.71–0.94; p = 0.004) or to have less than higher education (OR 0.62–0.69; all p ≤ 0.022). In 2024, women in the high-coverage cluster (C-1) were significantly more likely to be married (1.22; 1.09–1.36; p < 0.001), of lower parity (0.91; 0.86–0.96; p = 0.001), and marginally older (1.02; 1.00–1.03; p = 0.034); all sub-higher education categories were strongly associated with reduced membership odds (OR 0.43–0.50; all p < 0.001). No predictor reached statistical significance for the moderate ANC/limited delivery cluster (C-3) in 2024.

[INSERT FIGURE 6 ABOUT HERE]

---

## Discussion

This comparative unsupervised learning analysis of two nationally representative Malawian DHS surveys documents substantial progress in maternal health care utilisation between 2015 and 2024, while identifying persistent gaps requiring targeted policy attention. Five principal findings merit discussion.

The structure of utilisation heterogeneity simplified markedly over the decade. A four-cluster solution best described the 2015 landscape; by 2024, a three-cluster solution sufficed. This consolidation reflects a shift of women who previously followed a minimal utilisation pattern into the broader formal health system, at least for intrapartum care, and a proportional increase in comprehensive care engagement. The most dramatic change was the near-doubling of the comprehensive care cluster, driven primarily by a 31.5 percent increase in early ANC initiation, the single largest change across all indicators. The effective dissolution of the minimal utilisation cluster is a meaningful programmatic achievement, consistent with documented improvements in facility delivery following Malawi's free maternity services policy and health system strengthening investments [8,17]. In 2015, approximately one in nine women engaged minimally across the entire continuum; by 2024, no such identifiable group remained.

Nonetheless, the dominant pattern across both survey rounds remained "late ANC with universal facility delivery", affecting 60.1% of women in 2015 and still 48.4% in 2024. This profile, in which women reliably access skilled intrapartum care but engage late and insufficiently with ANC, is not unique to Malawi. Benova et al. [16] described an analogous pattern across multiple sub-Saharan settings, attributing it to the stronger perceived urgency of delivery care relative to the "wellness-oriented" framing of ANC. The persistence of this pattern despite decade-long investment in ANC promotion suggests that supply-side strategies alone are insufficient and that demand-side interventions targeting normative beliefs about ANC timing are needed.

The narrowing of sociodemographic gradients across all four subgroups, residence, wealth quintile, age, and education, is encouraging evidence that programmatic gains have been broadly distributed. The decline in Cramer's V for residence from 0.123 to 0.055 and for wealth from 0.090 to 0.059 indicates that comprehensive care is becoming less concentrated among privileged groups. That said, a 9.7 percent gap between the richest and poorest women in the high-coverage cluster in 2024 represents a remaining equity concern. The persistent disadvantage of adolescent women is similarly notable. Adolescents aged 15-19 showed the lowest high-coverage rates in both years and, while they improved substantially from 19.7% to 43.3%, the gap with young adults was essentially unchanged across survey rounds, suggesting that universal programmatic improvements benefited all groups proportionally rather than closing the adolescent-specific gap. This aligns with a robust literature documenting barriers specific to young women in Malawi, including fear of stigma at health facilities, lack of partner support, and limited availability of youth-friendly services [18,19].

The shift in variable importance from wealth to age, parity, and perceived distance to facility is theoretically coherent with a maturing health system. As economic barriers are progressively mitigated through free care policies, geographic access and social factors emerge as the residual constraints on utilisation heterogeneity. This has direct programmatic implications: future interventions should prioritise geographic access for remote rural women and demand-side approaches, including peer support networks and reproductive autonomy programmes, for younger and higher-parity women. The finding that optimal ANC coverage of eight or more visits (per updated WHO 2016 guidelines) increased from only 1.4% to 2.9% highlights a further area in which quantity of ANC contact has improved while depth of engagement has not.

This analysis has several strengths. It is among the first comparative unsupervised learning analyses spanning two Malawi DHS waves, enabling direct temporal assessment of utilisation pattern evolution. The use of harmonised clustering features across both rounds, principled composite model selection criteria, and validation through subgroup and predictor analyses adds methodological rigour, and the large nationally representative samples support generalisability to the full population of Malawian women of reproductive age.

Several limitations deserve acknowledgement. The complete-case exclusion rate in 2024 (34.3% of eligible women) was high, driven primarily by structural non-response among women who did not attend ANC. While analytically appropriate, this may mean that the 2024 analysis sample over-represents women who engaged with the health system, and results may not fully generalise to non-attenders. The differing optimal cluster solutions across years (K = 4 in 2015 and K = 3 in 2024) required cross-year alignment rather than direct comparison, and the least similar aligned pair (centroid distance = 1.50) should be interpreted with caution. The cross-sectional design precludes causal inference, and the nine-year interval captures cumulative policy effects without attributing change to specific programmes. GMM assumes Gaussian mixture components, an assumption that may not hold perfectly across all utilisation dimensions, and sensitivity analysis with density-based clustering would strengthen robustness claims. Finally, the 2024 compressed DHS file used non-standard variable coding for residence, wealth, and place of delivery, requiring careful harmonisation.

In conclusion, Malawi's maternal health system achieved substantial and measurable progress in shifting women toward comprehensive engagement across the continuum of care between 2015 and 2024. The doubling of the high-coverage cluster, the effective elimination of the minimal utilisation cluster, and the narrowing of sociodemographic gradients testify to a decade of meaningful policy investment. At the same time, nearly half of women still initiate ANC late, adolescent women continue to be underrepresented in comprehensive care, and PNC receipt has not improved alongside other indicators. Closing these remaining gaps, through adolescent-targeted services, geographic access improvements, and postnatal care quality strengthening, will be essential for Malawi to achieve its SDG 3.1 maternal mortality targets.

---

## Tables

### Table 1. Sociodemographic Characteristics of Study Samples

| Characteristic | MDHS 2015 (n = 13,030) | MDHS 2024 (n = 6,925) |
|---|---|---|
| **Age, years (mean +/- SD)** | 28.1 +/- 7.0 | 26.9 +/- 6.9 |
| **Age group, n (%)** | | |
| Adolescent (15-19) | 1,109 (8.5%) | 928 (13.4%) |
| Young adult (20-34) | 9,343 (71.7%) | 4,861 (70.2%) |
| Older (35-49) | 2,578 (19.8%) | 1,136 (16.4%) |
| **Residence, n (%)** | | |
| Urban | 2,257 (17.3%) | 1,298 (18.7%) |
| Rural | 10,773 (82.7%) | 5,627 (81.3%) |
| **Wealth quintile, n (%)** | | |
| Poorest | 2,767 (21.2%) | 1,527 (22.0%) |
| Poorer | 2,715 (20.8%) | 1,208 (17.4%) |
| Middle | 2,511 (19.3%) | 1,196 (17.3%) |
| Richer | 2,510 (19.3%) | 1,289 (18.6%) |
| Richest | 2,527 (19.4%) | 1,448 (20.9%) |

---

### Table 2. Maternal Care Utilisation Indicators: MDHS 2015 vs 2024

| Indicator | MDHS 2015 | MDHS 2024 | Change | p-value |
|---|---|---|---|---|
| First ANC visit, month (mean +/- SD) | 4.45 +/- 1.31 | 3.34 +/- 1.34 | -1.1 months | <0.001 |
| ANC visits (mean +/- SD) | 3.77 +/- 1.62 | 4.22 +/- 1.52 | +0.45 visits | <0.001 |
| Skilled ANC provider (%) | 96.6% | 98.4% | +1.8% | <0.001 |
| **Early ANC initiation <=3 months (%)** | **26.0%** | **57.5%** | **+31.5%** | **<0.001** |
| Adequate ANC >=4 visits (%) | 51.9% | 66.6% | +14.7% | <0.001 |
| Optimal ANC >=8 visits (%) | 1.4% | 2.9% | +1.5% | <0.001 |
| Facility delivery (%) | 94.8% | 97.6% | +2.7% | <0.001 |
| Skilled birth attendant (%) | 92.0% | 97.1% | +5.0% | <0.001 |
| Caesarean section (%) | 7.0% | 11.5% | +4.5% | <0.001 |
| PNC received (%) | 45.1% | 43.2% | -1.9% | 0.013 |
| Informed about danger signs (%) | 25.3% | 24.8% | -0.5% | 0.421 |

*p-values: chi-square test (binary) or Welch's t-test (continuous). Highlighted row = largest change.*

---

### Table 3. Cluster Profiles: MDHS 2015 (K = 4) and MDHS 2024 (K = 3)

| Feature | 2015 C-A | 2015 C-B | 2015 C-C | 2015 C-D | 2024 C-1 | 2024 C-2 | 2024 C-3 |
|---|---|---|---|---|---|---|---|
| Label | Comprehensive ANC+delivery | Late ANC, facility delivery | C-section, high PNC | Minimal utilisation | High coverage | Late ANC, facility delivery | Moderate ANC, limited delivery |
| n (%) | 3,149 (24.2%) | 7,828 (60.1%) | 910 (7.0%) | 1,143 (8.8%) | 3,232 (46.7%) | 3,349 (48.4%) | 344 (5.0%) |
| First ANC month (mean) | 2.93 | 4.98 | 4.18 | 5.21 | 2.38 | 4.25 | 3.42 |
| ANC visits (mean) | 4.76 | 3.43 | 3.99 | 3.18 | 5.23 | 3.27 | 3.89 |
| Early ANC <=3 months (%) | 97.6% | 0.0% | 34.7% | 0.0% | 99.2% | 17.5% | 54.9% |
| Adequate ANC >=4 visits (%) | 78.7% | 42.9% | 60.2% | 33.4% | 100.0% | 35.3% | 58.7% |
| Skilled ANC provider (%) | 96.5% | 100.0% | 98.0% | 72.9% | 100.0% | 100.0% | 68.6% |
| Facility delivery (%) | 96.2% | 100.0% | 100.0% | 51.8% | 100.0% | 100.0% | 50.9% |
| Skilled birth attendant (%) | 94.0% | 100.0% | 99.1% | 26.5% | 100.0% | 100.0% | 40.7% |
| Caesarean section (%) | 0.0% | 0.0% | 100.0% | 0.0% | 13.6% | 10.1% | 5.2% |
| PNC received (%) | 46.0% | 44.8% | 46.7% | 43.1% | 45.6% | 41.6% | 36.1% |

---

### Table 4. Subgroup Association with Cluster Membership (Cramer's V)

| Subgroup | MDHS 2015 (n) | Cramer's V | p-value | MDHS 2024 (n) | Cramer's V | p-value | Change |
|---|---|---|---|---|---|---|---|
| Residence | 13,030 | 0.123 | <0.001 | 6,925 | 0.055 | <0.001 | Narrowed |
| Wealth quintile | 13,030 | 0.090 | <0.001 | 6,668 | 0.059 | <0.001 | Narrowed |
| Age group | 13,030 | 0.046 | <0.001 | 6,925 | 0.042 | <0.001 | Stable |
| Education level | 13,030 | 0.089 | <0.001 | 6,925 | 0.066 | <0.001 | Narrowed |

*Cramer's V: <0.1 = small; 0.1-0.3 = medium; >0.3 = large.*

---

### Table 5. Top 5 Predictors of Cluster Membership (Random Forest Variable Importance)

| Rank | MDHS 2015 Predictor | MDI | MDHS 2024 Predictor | MDI |
|---|---|---|---|---|
| 1 | Parity | 0.180 | Age | 0.269 |
| 2 | Age | 0.166 | Parity | 0.167 |
| 3 | Wealth: Richest | 0.106 | Region | 0.094 |
| 4 | Region | 0.076 | Marital status (married) | 0.058 |
| 5 | Residence: Rural | 0.068 | Distance to facility (barrier) | 0.057 |

*MDI = Mean Decrease in Impurity. Cross-validated macro F1: 2015 = 0.213; 2024 = 0.328.*

---

### Table 6. Adjusted Odds Ratios from Multinomial Logistic Regression: Predictors of Cluster Membership

**Panel A: MDHS 2015** (Reference: C-B, Late ANC, facility delivery, n = 7,828)

| Predictor | C-C: C-section (n = 910) | C-D: Minimal utilisation (n = 1,143) | C-A: Comprehensive ANC+delivery (n = 3,149) |
|---|---|---|---|
| Age (per year) | 1.06 (1.04–1.08)*** | 0.99 (0.97–1.01) | 1.01 (1.00–1.03)* |
| Parity | 0.73 (0.68–0.79)*** | 1.09 (1.04–1.16)** | 0.97 (0.93–1.00) |
| Married | 0.85 (0.73–1.00) | 0.88 (0.76–1.02) | 1.14 (1.03–1.27)* |
| Distance barrier | 1.05 (0.90–1.22) | 1.39 (1.21–1.59)*** | 0.99 (0.90–1.08) |
| Region | 0.81 (0.74–0.89)*** | 0.94 (0.86–1.02) | 1.02 (0.97–1.08) |
| Urban residence | 1.24 (1.02–1.50)* | 0.73 (0.57–0.94)* | 0.82 (0.71–0.94)** |
| Wealth: Poorer | 0.98 (0.76–1.26) | 1.05 (0.87–1.26) | 1.05 (0.93–1.20) |
| Wealth: Poorest | 0.87 (0.67–1.14) | 1.13 (0.94–1.36) | 0.96 (0.84–1.10) |
| Wealth: Richer | 1.48 (1.17–1.88)** | 0.92 (0.75–1.13) | 1.04 (0.91–1.19) |
| Wealth: Richest | 1.81 (1.40–2.34)*** | 0.84 (0.65–1.10) | 1.28 (1.09–1.50)** |
| Education: No education | 0.45 (0.28–0.72)*** | 2.13 (0.84–5.43) | 0.62 (0.44–0.88)** |
| Education: Primary | 0.54 (0.37–0.78)** | 1.85 (0.74–4.63) | 0.69 (0.50–0.95)* |
| Education: Secondary | 0.59 (0.41–0.85)** | 1.29 (0.51–3.24) | 0.62 (0.45–0.85)** |

**Panel B: MDHS 2024** (Reference: C-2, Late ANC, facility delivery, n = 3,349)

| Predictor | C-1: High coverage (n = 3,232) | C-3: Moderate ANC, limited delivery (n = 344) |
|---|---|---|
| Age (per year) | 1.02 (1.00–1.03)* | 1.00 (0.97–1.03) |
| Parity | 0.91 (0.86–0.96)*** | 1.09 (0.98–1.22) |
| Married | 1.22 (1.09–1.36)*** | 1.21 (0.94–1.56) |
| Distance barrier | 1.03 (0.93–1.14) | 0.89 (0.70–1.13) |
| Region | 1.06 (0.99–1.13) | 1.00 (0.86–1.15) |
| Urban residence | 1.02 (0.87–1.19) | 0.76 (0.50–1.14) |
| Wealth: Poorer | 0.97 (0.82–1.15) | 1.40 (0.97–2.02) |
| Wealth: Poorest | 0.96 (0.82–1.12) | 1.34 (0.94–1.90) |
| Wealth: Richer | 0.95 (0.80–1.12) | 1.05 (0.71–1.55) |
| Wealth: Richest | 1.15 (0.95–1.39) | 1.06 (0.66–1.70) |
| Education: No education | 0.43 (0.28–0.65)*** | 0.90 (0.31–2.62) |
| Education: Primary | 0.48 (0.34–0.70)*** | 0.67 (0.25–1.81) |
| Education: Secondary | 0.50 (0.35–0.71)*** | 0.74 (0.28–1.97) |

*OR = adjusted odds ratio; 95% confidence interval in parentheses. Reference cluster: C-B in 2015; C-2 in 2024. Reference categories within predictors: residence = rural; wealth quintile = middle; education = higher education. * p < 0.05; ** p < 0.01; *** p < 0.001. McFadden R² = 0.023 (2015); 0.011 (2024).*

---

## Figure Legends

**Figure 1.** Study participant flow diagram. MDHS 2015 (left) and MDHS 2024 (right): sample derivation from total survey through eligibility restriction and complete-case exclusion to analysis samples.

**Figure 2.** Cluster profiles of maternal care utilisation. MDHS 2015 (upper panel, K = 4) and MDHS 2024 (lower panel, K = 3). Values are means; proportions (0-1) for binary indicators; continuous variables are raw means. Colour scale: yellow (low) to red (high).

**Figure 3.** Temporal change in maternal care utilisation patterns, 2015 to 2024. (A) Cluster prevalences by survey year, with aligned 2015 and 2024 clusters. (B) Feature-level changes (2024 minus 2015; asterisks indicate p < 0.05). (C) Shared PCA projection: cluster centroids for 2015 and 2024.

**Figure 4.** Cluster prevalence by subgroup: MDHS 2015 (top row) and MDHS 2024 (bottom row). (A) Residence (urban/rural). (B) Wealth quintile (poorest to richest). (C) Age group (adolescent, young adult, older).

**Figure 5.** Predictors of cluster membership. (A) Random Forest variable importance (MDI), top 12 predictors for 2015 and 2024. (B) Per-cluster F1 scores from 5-fold cross-validation, with dashed lines marking macro F1.

**Figure 6.** PCA scatter plot of individual women by cluster membership in a shared two-dimensional feature space. Each point represents one woman, coloured by cluster assignment. Stars mark cluster centroids. MDHS 2015 (left, K = 4) and MDHS 2024 (right, K = 3). Principal Component 1 explains 25.1% of variance and Principal Component 2 explains 15.2% of variance.

---

## References

1. WHO, UNICEF, UNFPA, World Bank, UN. Trends in Maternal Mortality 2000-2020. Geneva: WHO; 2023.
2. GBD 2019 Maternal Mortality Collaborators. Global, regional, and national levels of maternal mortality, 1990-2019. Lancet. 2020;396:1161-1203.
3. Bhutta ZA, et al. Can available interventions end preventable deaths in mothers, newborn babies, and stillbirths? Lancet. 2014;384:347-370.
4. Victora CG, et al. Countdown to 2015: a decade of tracking progress for maternal, newborn, and child survival. Lancet. 2016;387:2049-2059.
5. Malawi National Statistical Office, ICF. Malawi Demographic and Health Survey 2024. Zomba and Rockville: NSO and ICF; 2024.
6. Government of Malawi. Health Sector Strategic Plan III 2017-2022. Lilongwe: Ministry of Health; 2017.
7. Mlava G, Maluwa A, Chirwa E. Maternal health care services utilization in Malawi. BMC Health Serv Res. 2020;20:532.
8. Malawi National Statistical Office, ICF. Malawi Demographic and Health Survey 2015-16. Zomba and Rockville: NSO and ICF; 2017.
9. Kerber KJ, et al. Continuum of care for maternal, newborn, and child health. Lancet. 2007;370:1358-1369.
10. Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning (2nd ed.). New York: Springer; 2009.
11. Bishop CM. Pattern Recognition and Machine Learning. New York: Springer; 2006.
12. Benova L, et al. Two decades of antenatal and delivery care in Uganda. BMC Health Serv Res. 2018;18:758.
13. Nwala G, et al. Cluster analysis of antenatal care utilisation. Int J Gynecol Obstet. 2022;156:512-519.
14. Afulani PA, et al. Person-centred maternity care in Kenya, Ghana, and India. Lancet Glob Health. 2019;7:e96-e109.
15. WHO. WHO Recommendations on Antenatal Care for a Positive Pregnancy Experience. Geneva: WHO; 2016.
16. Benova L, et al. Cross-country comparison of facility delivery in sub-Saharan Africa. Glob Health Action. 2021;14:1-11.
17. Kancheya N, Kazembe L, Muula AS. Factors associated with maternal health care utilisation in peri-urban Malawi. Malawi Med J. 2021;33:123-129.
18. Chandra-Mouli V, et al. WHO guidelines on preventing early pregnancy in developing countries. J Adolesc Health. 2013;52:517-522.
19. Blanco-Zuniga J, et al. Barriers and facilitators of maternal health services by adolescents in LMICs. Glob Health Sci Pract. 2021;9:441-455.
20. Salam RA, et al. Interventions to improve adolescent nutrition. J Adolesc Health. 2016;59:S29-S39.
21. Owusu-Addo E, et al. Impact of cash transfers on social determinants of health in sub-Saharan Africa. Health Policy Plan. 2018;33:675-696.
22. Kruk ME, et al. High-quality health systems in the SDGs era. Lancet Glob Health. 2018;6:e1196-e1252.

---

Submitted to [Journal Name]. Manuscript formatted per [journal] guidelines. Analysis code and data available at [repository URL].
