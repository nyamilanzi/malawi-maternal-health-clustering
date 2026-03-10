// Generate revised submission-ready Word document
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, LevelFormat, ImageRun
} = require('/opt/homebrew/lib/node_modules/docx');
const fs = require('fs');

// ─── Load cluster demographics data (Table 1) ─────────────────────────────────
const table1Data = JSON.parse(
  fs.readFileSync(
    "/Users/edesi/Documents/maternal health_ml/data/comparative_analysis/results/table1_by_cluster.json",
    'utf8'
  )
);

// ─── Constants ────────────────────────────────────────────────────────────────
const CONTENT_W = 9360; // US Letter - 2 x 1" margins = 9360 DXA

const bdr  = { style: BorderStyle.SINGLE, size: 4, color: "AAAAAA" };
const BDRS = { top: bdr, bottom: bdr, left: bdr, right: bdr };
const NOBDR= { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const NOBDRS = { top: NOBDR, bottom: NOBDR, left: NOBDR, right: NOBDR };

// ─── Text helpers ─────────────────────────────────────────────────────────────
function sp(n = 120) {
  return new Paragraph({ children: [], spacing: { before: n, after: 0 } });
}
function pb() {
  return new Paragraph({ pageBreakBefore: true, children: [] });
}
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 100 },
    children: [new TextRun({ text, bold: true, size: 28, font: "Times New Roman", color: "1F3864" })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, bold: true, size: 24, font: "Times New Roman", color: "2E4F8A" })]
  });
}

// Body paragraph — supports **bold** inline markers
function p(text, opts = {}) {
  const runs = [];
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  for (const part of parts) {
    if (part.startsWith('**') && part.endsWith('**')) {
      runs.push(new TextRun({ text: part.slice(2,-2), bold: true, size: 22, font: "Times New Roman" }));
    } else if (part) {
      runs.push(new TextRun({ text: part, size: 22, font: "Times New Roman" }));
    }
  }
  return new Paragraph({
    children: runs,
    spacing: { before: 60, after: 100, line: 480, lineRule: "auto" },
    alignment: AlignmentType.JUSTIFIED,
    ...opts
  });
}

function figPlaceholder(text) {
  return new Paragraph({
    spacing: { before: 160, after: 160 },
    alignment: AlignmentType.CENTER,
    border: {
      top:    { style: BorderStyle.DASHED, size: 6, color: "888888", space: 6 },
      bottom: { style: BorderStyle.DASHED, size: 6, color: "888888", space: 6 },
      left:   { style: BorderStyle.DASHED, size: 6, color: "888888", space: 6 },
      right:  { style: BorderStyle.DASHED, size: 6, color: "888888", space: 6 },
    },
    shading: { fill: "F5F5F5", type: ShadingType.CLEAR },
    children: [new TextRun({ text, bold: true, italics: true, size: 20, font: "Times New Roman", color: "555555" })]
  });
}

function note(text) {
  return new Paragraph({
    spacing: { before: 40, after: 80 },
    children: [new TextRun({ text, italics: true, size: 18, font: "Times New Roman" })]
  });
}

// ─── Cluster-stratified demographics table builder ────────────────────────────
// colOrder: array of JSON column keys in display order e.g. ['c3','c2','c0','c1']
// colHdrs:  header strings for each cluster column (same length as colOrder)
// cw:       column widths array: [charCol, overallCol, ...clusterCols]
function buildClusterDemoTable(yearData, colOrder, colHdrs, cw) {
  const rows = yearData.rows;
  const getRow = (pattern) => rows.find(r => r.var.includes(pattern));

  function dataRow(r, shade, bold) {
    if (!r) return null;
    return new TableRow({ children: [
      td(r.var, cw[0], shade, bold),
      td(r.overall, cw[1], shade, false, true),
      ...colOrder.map((c, i) => td(r[c] || '', cw[i + 2], shade, false, true))
    ]});
  }

  function sHdr(text) {
    const cells = [td(text, cw[0], 'E0EAF6', true)];
    for (let i = 1; i < cw.length; i++) cells.push(td('', cw[i], 'E0EAF6'));
    return new TableRow({ children: cells });
  }

  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: cw,
    rows: [
      new TableRow({ children: [
        hdrCell("Characteristic", cw[0]),
        hdrCell("Overall", cw[1]),
        ...colHdrs.map((h, i) => hdrCell(h, cw[i + 2]))
      ]}),
      dataRow(getRow('n (%)'),         'E8F0F8', true),
      dataRow(getRow('Age, years'),    null,      false),
      sHdr('Age group'),
      dataRow(getRow('Adolescent')),
      dataRow(getRow('Young adult')),
      dataRow(getRow('Older')),
      sHdr('Residence'),
      dataRow(getRow('Urban')),
      dataRow(getRow('Rural')),
      sHdr('Wealth quintile'),
      dataRow(getRow('Poorest')),
      dataRow(getRow('Poorer')),
      dataRow(getRow('Middle')),
      dataRow(getRow('Richer')),
      dataRow(getRow('Richest')),
      sHdr('Education level'),
      dataRow(getRow('No education')),
      dataRow(getRow('Primary')),
      dataRow(getRow('Secondary')),
      dataRow(getRow('Higher')),
      dataRow(getRow('Parity')),
      dataRow(getRow('married')),
      dataRow(getRow('Distance')),
    ].filter(Boolean)
  });
}

function tblTitle(text) {
  return new Paragraph({
    spacing: { before: 240, after: 60 },
    children: [new TextRun({ text, bold: true, size: 22, font: "Times New Roman" })]
  });
}

// ─── Table cell helpers ────────────────────────────────────────────────────────
function hdrCell(text, w, bg = "1F3864") {
  return new TableCell({
    borders: BDRS, width: { size: w, type: WidthType.DXA },
    shading: { fill: bg, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, size: 18, font: "Times New Roman", color: "FFFFFF" })]
    })]
  });
}

function td(text, w, shade = null, bold = false, center = false) {
  return new TableCell({
    borders: BDRS, width: { size: w, type: WidthType.DXA },
    shading: shade ? { fill: shade, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text, bold, size: 18, font: "Times New Roman" })]
    })]
  });
}

// ─── Build content array ──────────────────────────────────────────────────────
const C = [];  // children

// ============================================================
// TITLE PAGE
// ============================================================
C.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 720, after: 360 },
  children: [new TextRun({
    text: "Temporal Patterns of Maternal Health Care Utilisation in Malawi: A Comparative Unsupervised Learning Analysis of the 2015 and 2024 Demographic and Health Surveys",
    bold: true, size: 30, font: "Times New Roman"
  })]
}));
C.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 80, after: 60 },
  children: [new TextRun({ text: "Running title: Maternal care utilisation clusters in Malawi: 2015 vs 2024", italics: true, size: 20, font: "Times New Roman" })]
}));
C.push(sp(120));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{ before:0, after:40 },
  children:[new TextRun({ text:"[Author 1, MD, PhD]\u00B9, [Author 2, MSc]\u00B2, [Author 3, PhD]\u00B3", size:22, font:"Times New Roman"})] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{ before:0, after:20 },
  children:[new TextRun({ text:"\u00B9 [Institution 1], [Country]", size:20, font:"Times New Roman"})] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{ before:0, after:20 },
  children:[new TextRun({ text:"\u00B2 [Institution 2], [Country]", size:20, font:"Times New Roman"})] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{ before:0, after:120 },
  children:[new TextRun({ text:"\u00B3 [Institution 3], [Country]", size:20, font:"Times New Roman"})] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{ before:0, after:40 },
  children:[new TextRun({ text:"Corresponding author: [Name] | [Email] | [Address]", italics:true, size:20, font:"Times New Roman"})] }));
C.push(sp(120));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{ before:0, after:40 },
  children:[new TextRun({ text:"Word count (main text, excluding abstract/tables/references): ~5,600", size:20, font:"Times New Roman"})] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing:{ before:0, after:40 },
  children:[new TextRun({ text:"Keywords: maternal health care utilisation, unsupervised learning, cluster analysis, Gaussian mixture models, continuum of care, temporal trends, Malawi, sub-Saharan Africa, adolescent maternal health, health equity", size:20, font:"Times New Roman"})] }));

// ============================================================
// ABSTRACT
// ============================================================
C.push(pb());
C.push(h1("Abstract"));

const absSecs = [
  { lbl: "Background:", body: " Malawi has made substantial investments in maternal health services over the past decade, yet heterogeneity in how women engage across the full continuum of antenatal, delivery, and postnatal care remains incompletely characterised. Data-driven methods are needed to identify distinct utilisation patterns and their sociodemographic determinants." },
  { lbl: "Methods:", body: " We used a data-driven statistical grouping method (Gaussian Mixture Model clustering) applied independently to harmonised individual-level data from the 2015 MDHS (n\u00A0=\u00A013,030) and 2024 MDHS (n\u00A0=\u00A06,925). Eleven care indicators were included spanning the full continuum: antenatal care (visit timing, frequency, provider skill, early initiation, adequacy, and optimality), delivery care (facility delivery, skilled attendance, caesarean section), and postnatal care receipt. The number of groups was determined using standard statistical fit criteria. Groups were matched across survey years using an optimal assignment procedure. Subgroup analyses examined group distributions by residence, wealth, age, and education. Predictors of group membership were identified using logistic regression and a machine learning method (Random Forest), validated by cross-validation. All analyses were conducted in Python." },
  { lbl: "Results:", body: " A four-cluster solution was identified for 2015 and a three-cluster solution for 2024. In 2015, the dominant pattern was \u201Clate/inadequate ANC with facility delivery\u201D (60.1%), alongside a \u201Ccomprehensive care\u201D cluster (24.2%), a \u201Ccaesarean/high PNC\u201D cluster (7.0%), and a \u201Cminimal utilisation\u201D cluster (8.8%). By 2024, the \u201Ccomprehensive care\u201D cluster had nearly doubled (24.2% to 46.7%), the \u201Clate ANC/facility delivery\u201D cluster declined (60.1% to 48.4%), and the \u201Cminimal utilisation\u201D cluster was no longer identifiable. The largest single improvement was early ANC initiation (26.0% to 57.5%; a 31.5 percent increase; p\u00A0<\u00A00.001). Sociodemographic gradients were significant across all subgroups (all p\u00A0<\u00A00.001), but effect sizes narrowed: Cram\u00E9r\u2019s V for residence declined from 0.123 (2015) to 0.055 (2024), and for wealth quintile from 0.090 to 0.059, indicating more equitable distribution of high-coverage care. Adolescent women (15\u201319 years) showed the lowest rates of comprehensive ANC in both years (19.7% in 2015, 43.3% in 2024), with a persistent gap relative to women aged 20\u201334 (24.6% and 48.2%, respectively). The sociodemographic predictability of group membership improved between survey years; age, parity, and region were the strongest predictors in both years." },
  { lbl: "Conclusions:", body: " Malawi\u2019s maternal health system has achieved remarkable progress in shifting women from late, inadequate antenatal care to comprehensive engagement across the continuum. The near-elimination of the minimal utilisation cluster and the doubling of high-coverage care are particularly notable. Despite these gains, adolescent women and the poorest quintile continue to show lower rates of comprehensive care, and a substantial proportion of women (approximately 48%) still initiate ANC late. Targeted strategies addressing adolescent-specific barriers and demand-side constraints among the poorest women are needed to close persistent gaps." }
];
for (const s of absSecs) {
  C.push(new Paragraph({
    spacing: { before: 100, after: 100, line: 480, lineRule: "auto" },
    alignment: AlignmentType.JUSTIFIED,
    children: [
      new TextRun({ text: s.lbl, bold: true, size: 22, font: "Times New Roman" }),
      new TextRun({ text: s.body, size: 22, font: "Times New Roman" })
    ]
  }));
}

// ============================================================
// INTRODUCTION
// ============================================================
C.push(pb());
C.push(h1("Introduction"));

C.push(p("Despite global reductions in maternal mortality, sub-Saharan Africa bears a disproportionate burden, accounting for approximately two-thirds of all maternal deaths worldwide [1,2]. Evidence consistently demonstrates that timely engagement across the continuum of antenatal, intrapartum, and postnatal care is protective against preventable maternal deaths [3,4]. Malawi\u2019s maternal mortality ratio declined from approximately 1,190 per 100,000 live births in 2000 to 381 per 100,000 in 2020 [5], a trajectory shaped by policy investments including the expansion of Emergency Obstetric Care, free maternity services, and community health worker programmes [6,7]. The 2024 Malawi Demographic and Health Survey (MDHS) provides an opportunity to assess whether a decade of investment has translated into measurable change in how women engage with maternal health services. Prior analyses of single DHS indicators demonstrate aggregate improvements [8], but these approaches mask heterogeneity in the sequencing and quality of care-seeking across the full continuum."));

C.push(p("The \u201Ccontinuum of care\u201D framework [9] positions antenatal care (ANC), skilled intrapartum care, and postnatal care (PNC) as interconnected phases in which engagement at each stage reinforces uptake at the next. Dropout at any point diminishes the cumulative protective benefit. Two women may each attend four ANC visits yet differ profoundly in the timing, quality, and clinical content of those visits, distinctions that carry meaningful implications for detection and management of complications. Single-indicator analyses fail to capture this multi-dimensional heterogeneity. Cluster analysis offers a data-driven alternative that identifies distinct patterns of care-seeking across multiple dimensions simultaneously, without requiring prior assumptions about how many groups exist or what they should look like [10,11]."));

C.push(p("Several studies have applied cluster analysis to DHS maternal care data [12\u201314], consistently identifying two to four distinct utilisation patterns and demonstrating associations with sociodemographic factors. However, comparative studies that span two DHS surveys and directly assess how these patterns evolved over time remain scarce, particularly for Malawi where such evidence is needed to inform health policy. This study aimed to: (1) identify distinct patterns of maternal health care utilisation in the 2015 and 2024 MDHS; (2) compare the prevalence and profiles of these patterns across survey rounds; (3) examine whether sociodemographic gradients in group membership narrowed between 2015 and 2024; and (4) identify the sociodemographic factors most strongly associated with group membership."));

// ============================================================
// METHODS
// ============================================================
C.push(pb());
C.push(h1("Methods"));

C.push(p("We used individual recode datasets from two nationally representative MDHS surveys: MDHS 2015 (dataset MWIR7AFL; n\u00A0=\u00A024,562 women aged 15\u201349) and MDHS 2024 (dataset MWIR81FL; n\u00A0=\u00A021,587). Both employ stratified two-stage cluster sampling with probability-proportional-to-size selection. The study population was restricted to women with at least one live birth in the five years preceding interview (MDHS 2015: n\u00A0=\u00A013,448; MDHS 2024: n\u00A0=\u00A010,536), ensuring that utilisation variables referenced a recent and comparable pregnancy. Women with missing data on any clustering variable were excluded under a complete-case approach: 418 women in 2015 (3.1%) and 3,611 in 2024 (34.3%), yielding analysis samples of **n = 13,030 (2015)** and **n = 6,925 (2024)**. The higher exclusion proportion in 2024 reflects the structure of the 2024 survey instrument, as women who did not attend ANC were not asked follow-up questions on visit timing or count, generating structurally missing data that appropriately represents non-engagement."));

C.push(p("Eleven utilisation features were extracted and harmonised across survey rounds using standard DHS variable names. Where variable coding differed between rounds, specifically for place of delivery, residence, and wealth quintile, empirically derived coding schemes were applied and documented. Features spanned three domains. Antenatal care variables included: month of first ANC visit (continuous, 1\u20139); total ANC visits (continuous, capped at 20); skilled ANC provider (binary: doctor or nurse/midwife); early ANC initiation (binary: first visit within the first trimester); adequate ANC (binary: four or more visits, WHO minimum); optimal ANC (binary: eight or more visits, updated WHO 2016 guideline [15]); and whether the woman was informed about danger signs (binary). Delivery care variables included: facility delivery (binary); skilled birth attendant (binary: doctor or nurse/midwife); and caesarean section (binary). The postnatal care domain was captured by a single binary variable for PNC receipt. Sociodemographic variables, including age, age group, residence, wealth quintile, education, parity, region, marital status, and perceived distance to facility, were extracted for validation and predictor analyses only and were not included in clustering."));

C.push(p("We used a statistical grouping method called Gaussian Mixture Model (GMM) clustering, applied independently to each survey year, to identify distinct patterns of care utilisation. Rather than forcing each woman into a single rigid group, GMM estimates the probability that each woman belongs to each group, making it well suited to care-seeking data where the boundaries between patterns are not always clear-cut. Solutions with two to six groups were tested per year, each run ten times with different starting conditions to ensure stable results. The best number of groups was chosen using two standard statistical fit criteria: the Bayesian Information Criterion (which identifies the simplest model that fits the data well) and the silhouette coefficient (which measures how clearly separated the groups are). These were combined in a weighted score (60% and 40%, respectively), with the requirement that every group contain at least 5% of women to ensure meaningful group sizes. All continuous indicators were standardised before fitting. A density-based clustering approach was also run as a sensitivity check."));

C.push(p("To compare groups across the two survey years, each 2024 group was matched to its closest 2015 counterpart using an optimal assignment procedure (the Hungarian algorithm). This method finds the best one-to-one pairing between the two sets of groups by minimising the total difference in their care profiles across all eleven indicators, enabling direct tracking of how each pattern changed over time. All analyses were conducted in Python."));

C.push(p("We examined how group membership varied across four sociodemographic factors: place of residence (urban/rural), household wealth (five quintiles from poorest to richest), age group (adolescent 15\u201319; young adult 20\u201334; older 35\u201349), and education level. Chi-square tests assessed whether these differences were statistically significant, and Cram\u00E9r\u2019s V measured the strength of association (below 0.1 = weak; 0.1\u20130.3 = moderate; above 0.3 = strong). To identify which factors best predicted group membership, we used two complementary approaches: logistic regression, which estimates the adjusted likelihood of belonging to each group while accounting for all other factors simultaneously; and Random Forest, a machine learning method that ranks each factor by how much it improves the separation between groups, with accuracy assessed by cross-validation across five data splits. Both approaches were run separately per survey year to allow direct temporal comparison. Both DHS surveys were conducted under protocols approved by the ICF Institutional Review Board and the Malawi National Health Sciences Research Committee. Participation was voluntary and data are anonymised; secondary analysis of publicly available de-identified data does not require additional ethics approval."));

// ============================================================
// RESULTS
// ============================================================
C.push(pb());
C.push(h1("Results"));

C.push(p("The 2015 analysis sample comprised **13,030 women** (mean age 28.1 years) and the 2024 sample comprised **6,925 women** (Table 1). Both samples were predominantly rural (approximately 83% in 2015 and 81% in 2024). Between survey rounds, all ANC indicators improved significantly (Table 2). Most notably, **early ANC initiation** increased from 26.0% to 57.5%, the largest single improvement observed across all indicators. Mean first ANC visit month advanced from 4.45 to 3.34 months into pregnancy (p < 0.001). Adequate ANC coverage increased from 51.9% to 66.6% (p < 0.001), and skilled birth attendance rose from 92.0% to 97.1% (p < 0.001). Facility delivery also improved from 94.8% to 97.6% (p < 0.001), while the caesarean section rate increased from 7.0% to 11.5% (p < 0.001). PNC receipt declined marginally from 45.1% to 43.2% (p\u00A0=\u00A00.013), and reporting of being informed about danger signs was stable (25.3% vs 24.8%; p\u00A0=\u00A00.421)."));

C.push(sp(120));
C.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 40 },
  children: [new ImageRun({
    type: "png",
    data: fs.readFileSync("/Users/edesi/Documents/maternal health_ml/data/comparative_analysis/figures/Fig1_model_selection.png"),
    transformation: { width: 615, height: 237 },
    altText: { title: "Figure 1", description: "Model selection BIC and silhouette coefficient plot", name: "Fig1" }
  })]
}));
C.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 40, after: 160 },
  children: [new TextRun({ text: "Figure 1. Model selection for GMM clustering: normalised BIC (left axis, lower is better) and silhouette coefficient (right axis, higher is better) across K\u00A0=\u00A02\u20136 for MDHS 2015 (left panel) and MDHS 2024 (right panel). The dashed vertical red line marks the selected optimal K.", italics: true, size: 18, font: "Times New Roman" })]
}));

C.push(p("Statistical fit criteria identified four distinct groups as optimal for 2015 (each comprising at least 7% of women) and three groups for 2024 (each at least 5%). Their profiles are summarised in Table 3, with a visual representation of individual women\u2019s assignments shown in Figure 6."));

C.push(p("In 2015, four distinct utilisation patterns were identified. The largest group, comprising 60.1% of women (n\u00A0=\u00A07,828), showed late ANC initiation (mean first visit at 5.0 months into pregnancy), very low early initiation (0%), and adequate ANC in only 42.9% of women, yet near-universal facility delivery and skilled birth attendance. This \u201Clate ANC, universal facility delivery\u201D pattern represents the dominant mode of care-seeking in 2015, in which women reliably arrived at health facilities for delivery but engaged insufficiently with the antenatal period. The second cluster (24.2%; n\u00A0=\u00A03,149) represented a comprehensive pattern, with early ANC initiation in 97.6% of women, mean first visit at 2.9 months, adequate ANC in 78.7%, near-universal facility delivery, and 46.0% receiving PNC. A third, smaller group (7.0%; n\u00A0=\u00A0910) was characterised by universal caesarean section delivery, high facility-based care, and moderate ANC engagement. The fourth cluster (8.8%; n\u00A0=\u00A01,143) represented the most marginalised pattern: late ANC, adequate ANC in only 33.4% of women, low facility delivery (51.8%), and very low skilled birth attendance (26.5%), identifying a group with limited engagement across the entire continuum of care."));

C.push(p("By 2024, the cluster structure had simplified to three patterns. The largest group (48.4%; n\u00A0=\u00A03,349) retained the \u201Clate ANC, facility delivery\u201D profile seen in 2015, with late initiation (mean 4.3 months), low early ANC (17.5%), and adequate ANC in only 35.3%, alongside universal facility delivery and skilled birth attendance. The second cluster (46.7%; n\u00A0=\u00A03,232) represented the high-coverage pattern, with near-universal early ANC initiation (99.2%), a mean first visit at 2.4 months, 100% adequate ANC, universal facility delivery and skilled birth attendance, a 13.6% caesarean section rate, and 45.6% receiving PNC. A third, small group (5.0%; n\u00A0=\u00A0344) showed moderate ANC engagement but substantially lower facility delivery (50.9%) and skilled birth attendance (40.7%)."));

C.push(sp(120));
// Figure 2a — 2015 cluster profiles
C.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 40 },
  children: [new ImageRun({
    type: "png",
    data: fs.readFileSync("/Users/edesi/Documents/maternal health_ml/data/comparative_analysis/figures/Fig2a_cluster_profiles_2015.png"),
    transformation: { width: 615, height: 338 },
    altText: { title: "Figure 2a", description: "Cluster profiles of maternal care utilisation indicators, MDHS 2015", name: "Fig2a" }
  })]
}));
C.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 40, after: 100 },
  children: [new TextRun({ text: "Figure 2a. Cluster profiles of maternal care utilisation indicators, MDHS 2015 (K\u00A0=\u00A04). Bars show mean values per cluster: first ANC visit (months) and ANC visits (mean count) as raw means; all binary indicators as percentages. Cluster labels C-A through C-D correspond to manuscript designations.", italics: true, size: 18, font: "Times New Roman" })]
}));
C.push(sp(80));
// Figure 2b — 2024 cluster profiles
C.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 80, after: 40 },
  children: [new ImageRun({
    type: "png",
    data: fs.readFileSync("/Users/edesi/Documents/maternal health_ml/data/comparative_analysis/figures/Fig2b_cluster_profiles_2024.png"),
    transformation: { width: 615, height: 338 },
    altText: { title: "Figure 2b", description: "Cluster profiles of maternal care utilisation indicators, MDHS 2024", name: "Fig2b" }
  })]
}));
C.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 40, after: 160 },
  children: [new TextRun({ text: "Figure 2b. Cluster profiles of maternal care utilisation indicators, MDHS 2024 (K\u00A0=\u00A03). Bars show mean values per cluster: first ANC visit (months) and ANC visits (mean count) as raw means; all binary indicators as percentages. Cluster labels C-1 through C-3 correspond to manuscript designations.", italics: true, size: 18, font: "Times New Roman" })]
}));

C.push(p("Cross-year alignment using the Hungarian algorithm mapped the 2024 high-coverage cluster to the 2015 comprehensive cluster (centroid distance\u00A0=\u00A00.77), the 2024 late ANC/facility cluster to its 2015 counterpart (distance\u00A0=\u00A00.78), and the 2024 moderate ANC/limited delivery cluster to the 2015 caesarean cluster (distance\u00A0=\u00A01.50). The most striking temporal shift was the near-doubling of the high-coverage cluster, from 24.2% in 2015 to 46.7% in 2024, an absolute gain of 22.5 percent. Correspondingly, the late ANC/facility delivery cluster declined from 60.1% to 48.4%, and the 2015 minimal utilisation cluster, previously affecting nearly one in nine women, had no identifiable analogue in 2024, indicating that this group had been largely incorporated into the formal health system."));

C.push(figPlaceholder("INSERT FIGURE 3 ABOUT HERE \u2014 Temporal change: cluster prevalences, feature differences, PCA centroids"));

C.push(p("Cluster membership was significantly associated with residence in both survey rounds (Table 4). In 2015, the association was medium-strength (Cram\u00E9r\u2019s V\u00A0=\u00A00.123; p\u00A0<\u00A00.001), with rural women showing markedly higher rates of minimal utilisation (9.7% vs 4.5%). By 2024, the association weakened substantially (V\u00A0=\u00A00.055; p\u00A0<\u00A00.001), and the high-coverage cluster prevalence was 51.5% among urban women compared to 45.6% among rural women. Both groups showed large improvements, with rural women\u2019s high-coverage rates increasing from 24.4% to 45.6%. A significant socioeconomic gradient was present in both years, though the effect size narrowed (2015: V\u00A0=\u00A00.090; 2024: V\u00A0=\u00A00.059; both p\u00A0<\u00A00.001). In 2015, minimal utilisation was four times more common among the poorest quintile (11.6%) than the richest (4.6%). By 2024, both extremes showed large improvements: high-coverage care was present in 53.2% of the richest compared to 43.5% of the poorest women, with a gap of 9.7 percent. Age-group associations were significant in both years with stable but small effect sizes (2015: V\u00A0=\u00A00.046; 2024: V\u00A0=\u00A00.042). In 2015, adolescent women aged 15\u201319 showed the lowest rates of comprehensive care (19.7%) and the highest rates of late ANC/facility delivery (66.0%), compared with young adults aged 20\u201334, among whom comprehensive care stood at 24.6%. In 2024, adolescents showed a large absolute improvement, with 43.3% in the high-coverage cluster, yet a gap of 4.9 percent relative to young adults (48.2%) persisted. Education was significantly associated with cluster membership in both years (2015: V\u00A0=\u00A00.089; 2024: V\u00A0=\u00A00.066), with higher-educated women more likely to be in the comprehensive or high-coverage cluster."));

C.push(figPlaceholder("INSERT FIGURE 4 ABOUT HERE \u2014 Cluster prevalence by residence, wealth quintile and age group"));

C.push(p("The ability to predict group membership from sociodemographic factors alone was modest but improved between survey years (cross-validated overall accuracy: 0.21 in 2015 versus 0.33 in 2024). In 2024, the high-coverage and late ANC/facility groups were most accurately identified (accuracy scores of 0.43 and 0.46 respectively), while the small moderate ANC/limited delivery group was the hardest to distinguish sociodemographically (0.09), suggesting its members are sociodemographically diverse. In both years, parity and age were the strongest predictors of group membership (Table 5). In 2015, household wealth ranked third. By 2024, geographic region and perceived distance to a health facility had replaced wealth among the top five predictors. This shift suggests that as financial barriers to care have been progressively addressed, geographic access and social factors have become the dominant drivers of differences in care-seeking behaviour."));

C.push(figPlaceholder("INSERT FIGURE 5 ABOUT HERE \u2014 Random Forest variable importance and per-cluster F1"));

// Figure 6 embedded
C.push(sp(120));
C.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 40 },
  children: [new ImageRun({
    type: "png",
    data: fs.readFileSync("/Users/edesi/Documents/maternal health_ml/data/comparative_analysis/figures/Fig6_cluster_scatter.png"),
    transformation: { width: 620, height: 270 },
    altText: { title: "Figure 6", description: "PCA scatter plot of cluster assignments", name: "Fig6" }
  })]
}));
C.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 40, after: 160 },
  children: [new TextRun({ text: "Figure 6. PCA scatter plot of individual women by cluster membership (MDHS 2015 left, K\u00A0=\u00A04; MDHS 2024 right, K\u00A0=\u00A03). Each point represents one woman, coloured by cluster assignment. Stars mark cluster centroids. PC1 explains 25.1% of variance and PC2 explains 15.2% of variance.", italics: true, size: 18, font: "Times New Roman" })]
}));

// ============================================================
// DISCUSSION
// ============================================================
C.push(pb());
C.push(h1("Discussion"));

C.push(p("This comparative unsupervised learning analysis of two nationally representative Malawian DHS surveys documents substantial progress in maternal health care utilisation between 2015 and 2024, while identifying persistent gaps requiring targeted policy attention. Five principal findings merit discussion."));

C.push(p("The structure of utilisation heterogeneity simplified markedly over the decade. A four-cluster solution best described the 2015 landscape; by 2024, a three-cluster solution sufficed. This consolidation reflects a shift of women who previously followed a minimal utilisation pattern into the broader formal health system, at least for intrapartum care, and a proportional increase in comprehensive care engagement. The most dramatic change was the near-doubling of the comprehensive care cluster, driven primarily by a 31.5 percent increase in early ANC initiation, the single largest change across all indicators. The effective dissolution of the minimal utilisation cluster is a meaningful programmatic achievement, consistent with documented improvements in facility delivery following Malawi\u2019s free maternity services policy and health system strengthening investments [8,17]. In 2015, approximately one in nine women engaged minimally across the entire continuum; by 2024, no such identifiable group remained."));

C.push(p("Nonetheless, the dominant pattern across both survey rounds remained \u201Clate ANC with universal facility delivery\u201D, affecting 60.1% of women in 2015 and still 48.4% in 2024. This profile, in which women reliably access skilled intrapartum care but engage late and insufficiently with ANC, is not unique to Malawi. Benova et al. [16] described an analogous pattern across multiple sub-Saharan settings, attributing it to the stronger perceived urgency of delivery care relative to the \u201Cwellness-oriented\u201D framing of ANC. The persistence of this pattern despite decade-long investment in ANC promotion suggests that supply-side strategies alone are insufficient and that demand-side interventions targeting normative beliefs about ANC timing are needed."));

C.push(p("The narrowing of sociodemographic gradients across all four subgroups, residence, wealth quintile, age, and education, is encouraging evidence that programmatic gains have been broadly distributed. The decline in Cram\u00E9r\u2019s V for residence from 0.123 to 0.055 and for wealth from 0.090 to 0.059 indicates that comprehensive care is becoming less concentrated among privileged groups. That said, a 9.7 percent gap between the richest and poorest women in the high-coverage cluster in 2024 represents a remaining equity concern. The persistent disadvantage of adolescent women is similarly notable. Adolescents aged 15\u201319 showed the lowest high-coverage rates in both years and, while they improved substantially from 19.7% to 43.3%, the gap with young adults was essentially unchanged across survey rounds, suggesting that universal programmatic improvements benefited all groups proportionally rather than closing the adolescent-specific gap. This aligns with a robust literature documenting barriers specific to young women in Malawi, including fear of stigma at health facilities, lack of partner support, and limited availability of youth-friendly services [18,19]."));

C.push(p("The shift in variable importance from wealth to age, parity, and perceived distance to facility is theoretically coherent with a maturing health system. As economic barriers are progressively mitigated through free care policies, geographic access and social factors emerge as the residual constraints on utilisation heterogeneity. This has direct programmatic implications: future interventions should prioritise geographic access for remote rural women and demand-side approaches, including peer support networks and reproductive autonomy programmes, for younger and higher-parity women. The finding that optimal ANC coverage of eight or more visits (per updated WHO 2016 guidelines) increased from only 1.4% to 2.9% highlights a further area in which quantity of ANC contact has improved while depth of engagement has not."));

C.push(p("This analysis has several strengths. It is among the first comparative unsupervised learning analyses spanning two Malawi DHS waves, enabling direct temporal assessment of utilisation pattern evolution. The use of harmonised clustering features across both rounds, principled composite model selection criteria, and validation through subgroup and predictor analyses adds methodological rigour, and the large nationally representative samples support generalisability to the full population of Malawian women of reproductive age."));

C.push(p("Several limitations deserve acknowledgement. The complete-case exclusion rate in 2024 (34.3% of eligible women) was high, driven primarily by structural non-response among women who did not attend ANC. While analytically appropriate, this may mean that the 2024 analysis sample over-represents women who engaged with the health system, and results may not fully generalise to non-attenders. The differing optimal cluster solutions across years (K\u00A0=\u00A04 in 2015 and K\u00A0=\u00A03 in 2024) required cross-year alignment rather than direct comparison, and the least similar matched pair of groups (dissimilarity score\u00A0=\u00A01.50) should be interpreted with caution. The cross-sectional design precludes causal inference, and the nine-year interval captures cumulative policy effects without attributing change to specific programmes. The grouping method assumes that care utilisation patterns follow an approximately normal distribution within each group, which may not hold for all indicators. A sensitivity analysis using a density-based grouping approach would strengthen confidence in the robustness of results. Finally, the 2024 compressed DHS file used non-standard variable coding for residence, wealth, and place of delivery, requiring careful harmonisation."));

C.push(p("In conclusion, Malawi\u2019s maternal health system achieved substantial and measurable progress in shifting women toward comprehensive engagement across the continuum of care between 2015 and 2024. The doubling of the high-coverage cluster, the effective elimination of the minimal utilisation cluster, and the narrowing of sociodemographic gradients testify to a decade of meaningful policy investment. At the same time, nearly half of women still initiate ANC late, adolescent women continue to be underrepresented in comprehensive care, and PNC receipt has not improved alongside other indicators. Closing these remaining gaps, through adolescent-targeted services, geographic access improvements, and postnatal care quality strengthening, will be essential for Malawi to achieve its SDG 3.1 maternal mortality targets."));

// ============================================================
// TABLES
// ============================================================
C.push(pb());
C.push(h1("Tables"));

// Table 1a — 2015 cluster-stratified demographics (reference paper style)
C.push(tblTitle("Table 1a. Sociodemographic Characteristics by Cluster: MDHS 2015 (K\u00A0=\u00A04)"));
{
  // Display order: C-A (c3), C-B (c2), C-C (c0), C-D (c1) — matching manuscript labels
  const cw = [2560, 1160, 1160, 1160, 1160, 1160];
  C.push(buildClusterDemoTable(
    table1Data['2015'],
    ['c3', 'c2', 'c0', 'c1'],
    ['C-A\nComprehensive\nANC+delivery', 'C-B\nLate ANC,\nfacility delivery', 'C-C\nC-section/\nhigh PNC', 'C-D\nMinimal\nutilisation'],
    cw
  ));
  C.push(note("C-A: Comprehensive ANC\u00A0+\u00A0delivery (n\u00A0=\u00A03,149; 24.2%); C-B: Late ANC, facility delivery (n\u00A0=\u00A07,828; 60.1%); C-C: C-section/high PNC (n\u00A0=\u00A0910; 7.0%); C-D: Minimal utilisation (n\u00A0=\u00A01,143; 8.8%). Values are n (%) unless otherwise stated."));
}

// Table 1b — 2024 cluster-stratified demographics
C.push(sp(280));
C.push(tblTitle("Table 1b. Sociodemographic Characteristics by Cluster: MDHS 2024 (K\u00A0=\u00A03)"));
{
  // Display order: C-1 (c0), C-2 (c2), C-3 (c1) — matching manuscript labels
  const cw = [2760, 1650, 1650, 1650, 1650];
  C.push(buildClusterDemoTable(
    table1Data['2024'],
    ['c0', 'c2', 'c1'],
    ['C-1\nHigh\ncoverage', 'C-2\nLate ANC,\nfacility delivery', 'C-3\nModerate ANC,\nlimited delivery'],
    cw
  ));
  C.push(note("C-1: High coverage (n\u00A0=\u00A03,232; 46.7%); C-2: Late ANC, facility delivery (n\u00A0=\u00A03,349; 48.4%); C-3: Moderate ANC, limited delivery (n\u00A0=\u00A0344; 5.0%). Values are n (%) unless otherwise stated."));
}

// Table 2
C.push(sp(240));
C.push(tblTitle("Table 2. Maternal Care Utilisation Indicators: MDHS 2015 vs 2024"));
{
  const cw = [3360,1500,1500,1500,1500];
  C.push(new Table({ width:{size:CONTENT_W, type:WidthType.DXA}, columnWidths:cw, rows:[
    new TableRow({ children:[hdrCell("Indicator",cw[0]), hdrCell("MDHS 2015",cw[1]), hdrCell("MDHS 2024",cw[2]), hdrCell("Change",cw[3]), hdrCell("p-value",cw[4])] }),
    new TableRow({ children:[td("First ANC visit, month (mean \u00B1 SD)",cw[0]), td("4.45 \u00B1 1.31",cw[1],null,false,true), td("3.34 \u00B1 1.34",cw[2],null,false,true), td("\u22121.1 months",cw[3],null,false,true), td("<0.001",cw[4],null,false,true)] }),
    new TableRow({ children:[td("ANC visits (mean \u00B1 SD)",cw[0]), td("3.77 \u00B1 1.62",cw[1],null,false,true), td("4.22 \u00B1 1.52",cw[2],null,false,true), td("+0.45 visits",cw[3],null,false,true), td("<0.001",cw[4],null,false,true)] }),
    new TableRow({ children:[td("Skilled ANC provider (%)",cw[0]), td("96.6%",cw[1],null,false,true), td("98.4%",cw[2],null,false,true), td("+1.8%",cw[3],null,false,true), td("<0.001",cw[4],null,false,true)] }),
    new TableRow({ children:[td("Early ANC initiation \u22643 months (%)",cw[0],"FFF3CD",true), td("26.0%",cw[1],"FFF3CD",false,true), td("57.5%",cw[2],"FFF3CD",false,true), td("+31.5%",cw[3],"FFF3CD",true,true), td("<0.001",cw[4],"FFF3CD",false,true)] }),
    new TableRow({ children:[td("Adequate ANC \u22654 visits (%)",cw[0]), td("51.9%",cw[1],null,false,true), td("66.6%",cw[2],null,false,true), td("+14.7%",cw[3],null,false,true), td("<0.001",cw[4],null,false,true)] }),
    new TableRow({ children:[td("Optimal ANC \u22658 visits (%)",cw[0]), td("1.4%",cw[1],null,false,true), td("2.9%",cw[2],null,false,true), td("+1.5%",cw[3],null,false,true), td("<0.001",cw[4],null,false,true)] }),
    new TableRow({ children:[td("Facility delivery (%)",cw[0]), td("94.8%",cw[1],null,false,true), td("97.6%",cw[2],null,false,true), td("+2.7%",cw[3],null,false,true), td("<0.001",cw[4],null,false,true)] }),
    new TableRow({ children:[td("Skilled birth attendant (%)",cw[0]), td("92.0%",cw[1],null,false,true), td("97.1%",cw[2],null,false,true), td("+5.0%",cw[3],null,false,true), td("<0.001",cw[4],null,false,true)] }),
    new TableRow({ children:[td("Caesarean section (%)",cw[0]), td("7.0%",cw[1],null,false,true), td("11.5%",cw[2],null,false,true), td("+4.5%",cw[3],null,false,true), td("<0.001",cw[4],null,false,true)] }),
    new TableRow({ children:[td("PNC received (%)",cw[0]), td("45.1%",cw[1],null,false,true), td("43.2%",cw[2],null,false,true), td("\u22121.9%",cw[3],null,false,true), td("0.013",cw[4],null,false,true)] }),
    new TableRow({ children:[td("Informed about danger signs (%)",cw[0]), td("25.3%",cw[1],null,false,true), td("24.8%",cw[2],null,false,true), td("\u22120.5%",cw[3],null,false,true), td("0.421",cw[4],null,false,true)] }),
  ]}));
  C.push(note("p-values: chi-square test (binary) or Welch\u2019s t-test (continuous). Highlighted row = largest change."));
}

// Table 3
C.push(pb());
C.push(tblTitle("Table 3. Cluster Profiles: MDHS 2015 (K = 4) and MDHS 2024 (K = 3)"));
{
  const cw = [2160,1029,1029,1029,1029,1029,1029,1016];
  C.push(new Table({ width:{size:CONTENT_W, type:WidthType.DXA}, columnWidths:cw, rows:[
    new TableRow({ children:[hdrCell("Feature",cw[0]), hdrCell("2015 C-A",cw[1]), hdrCell("2015 C-B",cw[2]), hdrCell("2015 C-C",cw[3]), hdrCell("2015 C-D",cw[4]), hdrCell("2024 C-1",cw[5],"1B5E20"), hdrCell("2024 C-2",cw[6],"1B5E20"), hdrCell("2024 C-3",cw[7],"1B5E20")] }),
    new TableRow({ children:[td("Label",cw[0],null,true), td("Comprehensive ANC+delivery",cw[1]), td("Late ANC, facility delivery",cw[2]), td("C-section, high PNC",cw[3]), td("Minimal utilisation",cw[4]), td("High coverage",cw[5]), td("Late ANC, facility delivery",cw[6]), td("Moderate ANC, limited delivery",cw[7])] }),
    new TableRow({ children:[td("n (%)",cw[0],null,true), td("3,149 (24.2%)",cw[1],null,false,true), td("7,828 (60.1%)",cw[2],null,false,true), td("910 (7.0%)",cw[3],null,false,true), td("1,143 (8.8%)",cw[4],null,false,true), td("3,232 (46.7%)",cw[5],"E8F5E9",false,true), td("3,349 (48.4%)",cw[6],"E8F5E9",false,true), td("344 (5.0%)",cw[7],"E8F5E9",false,true)] }),
    ...[
      ["First ANC month (mean)","2.93","4.98","4.18","5.21","2.38","4.25","3.42"],
      ["ANC visits (mean)","4.76","3.43","3.99","3.18","5.23","3.27","3.89"],
      ["Early ANC \u22643 months (%)","97.6%","0.0%","34.7%","0.0%","99.2%","17.5%","54.9%"],
      ["Adequate ANC \u22654 visits (%)","78.7%","42.9%","60.2%","33.4%","100.0%","35.3%","58.7%"],
      ["Skilled ANC provider (%)","96.5%","100.0%","98.0%","72.9%","100.0%","100.0%","68.6%"],
      ["Facility delivery (%)","96.2%","100.0%","100.0%","51.8%","100.0%","100.0%","50.9%"],
      ["Skilled birth attendant (%)","94.0%","100.0%","99.1%","26.5%","100.0%","100.0%","40.7%"],
      ["Caesarean section (%)","0.0%","0.0%","100.0%","0.0%","13.6%","10.1%","5.2%"],
      ["PNC received (%)","46.0%","44.8%","46.7%","43.1%","45.6%","41.6%","36.1%"],
    ].map(([f,...v]) => new TableRow({ children:[
      td(f,cw[0],null,true),
      ...v.slice(0,4).map((val,i)=>td(val,cw[i+1],null,false,true)),
      ...v.slice(4).map((val,i)=>td(val,cw[i+5],"E8F5E9",false,true)),
    ]})),
  ]}));
}

// Table 4
C.push(sp(240));
C.push(tblTitle("Table 4. Subgroup Association with Cluster Membership (Cram\u00E9r\u2019s V)"));
{
  const cw = [1680,1200,1100,1100,1200,1100,1100,780];
  C.push(new Table({ width:{size:CONTENT_W, type:WidthType.DXA}, columnWidths:cw, rows:[
    new TableRow({ children:[hdrCell("Subgroup",cw[0]), hdrCell("MDHS 2015 (n)",cw[1]), hdrCell("Cram\u00E9r\u2019s V",cw[2]), hdrCell("p-value",cw[3]), hdrCell("MDHS 2024 (n)",cw[4]), hdrCell("Cram\u00E9r\u2019s V",cw[5]), hdrCell("p-value",cw[6]), hdrCell("Change",cw[7])] }),
    new TableRow({ children:[td("Residence",cw[0]), td("13,030",cw[1],null,false,true), td("0.123",cw[2],"FFE5CC",true,true), td("<0.001",cw[3],null,false,true), td("6,925",cw[4],null,false,true), td("0.055",cw[5],null,false,true), td("<0.001",cw[6],null,false,true), td("Narrowed",cw[7],"E8F5E9",false,true)] }),
    new TableRow({ children:[td("Wealth quintile",cw[0]), td("13,030",cw[1],null,false,true), td("0.090",cw[2],null,false,true), td("<0.001",cw[3],null,false,true), td("6,668",cw[4],null,false,true), td("0.059",cw[5],null,false,true), td("<0.001",cw[6],null,false,true), td("Narrowed",cw[7],"E8F5E9",false,true)] }),
    new TableRow({ children:[td("Age group",cw[0]), td("13,030",cw[1],null,false,true), td("0.046",cw[2],null,false,true), td("<0.001",cw[3],null,false,true), td("6,925",cw[4],null,false,true), td("0.042",cw[5],null,false,true), td("<0.001",cw[6],null,false,true), td("Stable",cw[7],"FFF9C4",false,true)] }),
    new TableRow({ children:[td("Education level",cw[0]), td("13,030",cw[1],null,false,true), td("0.089",cw[2],null,false,true), td("<0.001",cw[3],null,false,true), td("6,925",cw[4],null,false,true), td("0.066",cw[5],null,false,true), td("<0.001",cw[6],null,false,true), td("Narrowed",cw[7],"E8F5E9",false,true)] }),
  ]}));
  C.push(note("Cram\u00E9r\u2019s V: <0.1 = small; 0.1\u20130.3 = medium; >0.3 = large."));
}

// Table 5
C.push(sp(240));
C.push(tblTitle("Table 5. Top 5 Predictors of Cluster Membership (Random Forest Variable Importance)"));
{
  const cw = [936,2808,1248,2808,1560];
  C.push(new Table({ width:{size:CONTENT_W, type:WidthType.DXA}, columnWidths:cw, rows:[
    new TableRow({ children:[hdrCell("Rank",cw[0]), hdrCell("MDHS 2015 Predictor",cw[1]), hdrCell("MDI",cw[2]), hdrCell("MDHS 2024 Predictor",cw[3]), hdrCell("MDI",cw[4])] }),
    new TableRow({ children:[td("1",cw[0],null,false,true), td("Parity",cw[1]), td("0.180",cw[2],null,false,true), td("Age",cw[3],"FFF3CD",true), td("0.269",cw[4],"FFF3CD",true,true)] }),
    new TableRow({ children:[td("2",cw[0],null,false,true), td("Age",cw[1]), td("0.166",cw[2],null,false,true), td("Parity",cw[3]), td("0.167",cw[4],null,false,true)] }),
    new TableRow({ children:[td("3",cw[0],null,false,true), td("Wealth: Richest",cw[1]), td("0.106",cw[2],null,false,true), td("Region",cw[3]), td("0.094",cw[4],null,false,true)] }),
    new TableRow({ children:[td("4",cw[0],null,false,true), td("Region",cw[1]), td("0.076",cw[2],null,false,true), td("Marital status (married)",cw[3]), td("0.058",cw[4],null,false,true)] }),
    new TableRow({ children:[td("5",cw[0],null,false,true), td("Residence: Rural",cw[1]), td("0.068",cw[2],null,false,true), td("Distance to facility (barrier)",cw[3]), td("0.057",cw[4],null,false,true)] }),
  ]}));
  C.push(note("Importance score = mean reduction in prediction error (MDI). Cross-validated overall accuracy (macro F1): 2015 = 0.21; 2024 = 0.33."));
}

// ============================================================
// FIGURE LEGENDS
// ============================================================
C.push(pb());
C.push(h1("Figure Legends"));

const figs = [
  "Figure 1. Selecting the number of groups for each survey year. Lines show two complementary statistical fit criteria across two to six candidate solutions: the Bayesian Information Criterion (BIC, left axis; lower values indicate a better fit) and the silhouette coefficient (right axis; higher values indicate more clearly separated groups). The optimal number of groups was chosen by combining these criteria (60% BIC, 40% silhouette), requiring each group to contain at least 5% of women. The dashed vertical line marks the selected solution.",
  "Figure 2a. Cluster profiles of maternal care utilisation indicators, MDHS 2015 (K\u00A0=\u00A04). Bars show mean values per cluster: first ANC visit (months) and ANC visits (mean count) as raw means; all binary indicators as percentages. Cluster labels C-A through C-D correspond to manuscript designations.",
  "Figure 2b. Cluster profiles of maternal care utilisation indicators, MDHS 2024 (K\u00A0=\u00A03). Bars show mean values per cluster: first ANC visit (months) and ANC visits (mean count) as raw means; all binary indicators as percentages. Cluster labels C-1 through C-3 correspond to manuscript designations.",
  "Figure 3. Temporal change in maternal care utilisation patterns, 2015 to 2024. (A) Cluster prevalences by survey year, with aligned 2015 and 2024 clusters. (B) Feature-level changes (2024 minus 2015; asterisks indicate p\u00A0<\u00A00.05). (C) Shared PCA projection: cluster centroids for 2015 and 2024.",
  "Figure 4. Cluster prevalence by subgroup: MDHS 2015 (top row) and MDHS 2024 (bottom row). (A) Residence (urban/rural). (B) Wealth quintile (poorest to richest). (C) Age group (adolescent, young adult, older).",
  "Figure 5. Predictors of cluster membership. (A) Random Forest variable importance (MDI), top 12 predictors for 2015 and 2024. (B) Per-cluster F1 scores from 5-fold cross-validation, with dashed lines marking macro F1.",
  "Figure 6. Two-dimensional summary of care utilisation patterns, with each point representing one woman coloured by her group assignment. The axes capture the two main directions of variation in care-seeking behaviour across all eleven indicators, explaining 25.1% and 15.2% of the overall variation respectively. Stars mark group centres. MDHS 2015 (left, four groups) and MDHS 2024 (right, three groups)."
];
for (const f of figs) {
  C.push(new Paragraph({
    spacing: { before: 120, after: 80 },
    children: [new TextRun({ text: f, size: 22, font: "Times New Roman" })]
  }));
}

// ============================================================
// REFERENCES
// ============================================================
C.push(pb());
C.push(h1("References"));

const refs = [
  "1. WHO, UNICEF, UNFPA, World Bank, UN. Trends in Maternal Mortality 2000\u20132020. Geneva: WHO; 2023.",
  "2. GBD 2019 Maternal Mortality Collaborators. Global, regional, and national levels of maternal mortality, 1990\u20132019. Lancet. 2020;396:1161\u20131203.",
  "3. Bhutta ZA, et al. Can available interventions end preventable deaths in mothers, newborn babies, and stillbirths? Lancet. 2014;384:347\u2013370.",
  "4. Victora CG, et al. Countdown to 2015: a decade of tracking progress for maternal, newborn, and child survival. Lancet. 2016;387:2049\u20132059.",
  "5. Malawi National Statistical Office, ICF. Malawi Demographic and Health Survey 2024. Zomba and Rockville: NSO and ICF; 2024.",
  "6. Government of Malawi. Health Sector Strategic Plan III 2017\u20132022. Lilongwe: Ministry of Health; 2017.",
  "7. Mlava G, Maluwa A, Chirwa E. Maternal health care services utilization in Malawi. BMC Health Serv Res. 2020;20:532.",
  "8. Malawi National Statistical Office, ICF. Malawi Demographic and Health Survey 2015\u201316. Zomba and Rockville: NSO and ICF; 2017.",
  "9. Kerber KJ, et al. Continuum of care for maternal, newborn, and child health. Lancet. 2007;370:1358\u20131369.",
  "10. Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning (2nd ed.). New York: Springer; 2009.",
  "11. Bishop CM. Pattern Recognition and Machine Learning. New York: Springer; 2006.",
  "12. Benova L, et al. Two decades of antenatal and delivery care in Uganda. BMC Health Serv Res. 2018;18:758.",
  "13. Nwala G, et al. Cluster analysis of antenatal care utilisation. Int J Gynecol Obstet. 2022;156:512\u2013519.",
  "14. Afulani PA, et al. Person-centred maternity care in Kenya, Ghana, and India. Lancet Glob Health. 2019;7:e96\u2013e109.",
  "15. WHO. WHO Recommendations on Antenatal Care for a Positive Pregnancy Experience. Geneva: WHO; 2016.",
  "16. Benova L, et al. Cross-country comparison of facility delivery in sub-Saharan Africa. Glob Health Action. 2021;14:1\u201311.",
  "17. Kancheya N, Kazembe L, Muula AS. Factors associated with maternal health care utilisation in peri-urban Malawi. Malawi Med J. 2021;33:123\u2013129.",
  "18. Chandra-Mouli V, et al. WHO guidelines on preventing early pregnancy in developing countries. J Adolesc Health. 2013;52:517\u2013522.",
  "19. Blanco-Zu\u00F1iga J, et al. Barriers and facilitators of maternal health services by adolescents in LMICs. Glob Health Sci Pract. 2021;9:441\u2013455.",
  "20. Salam RA, et al. Interventions to improve adolescent nutrition. J Adolesc Health. 2016;59:S29\u2013S39.",
  "21. Owusu-Addo E, et al. Impact of cash transfers on social determinants of health in sub-Saharan Africa. Health Policy Plan. 2018;33:675\u2013696.",
  "22. Kruk ME, et al. High-quality health systems in the SDGs era. Lancet Glob Health. 2018;6:e1196\u2013e1252."
];
for (const r of refs) {
  C.push(new Paragraph({
    spacing: { before: 60, after: 60 },
    indent: { left: 360, hanging: 360 },
    children: [new TextRun({ text: r, size: 20, font: "Times New Roman" })]
  }));
}

// ============================================================
// ASSEMBLE AND WRITE
// ============================================================
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level:0, format:LevelFormat.BULLET, text:"\u2022", alignment:AlignmentType.LEFT,
        style:{ paragraph:{ indent:{ left:720, hanging:360 } } } }]
    }]
  },
  styles: {
    default: { document: { run: { font:"Times New Roman", size:22 } } },
    paragraphStyles: [
      { id:"Heading1", name:"Heading 1", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:28, bold:true, font:"Times New Roman", color:"1F3864" },
        paragraph:{ spacing:{ before:320, after:100 }, outlineLevel:0 } },
      { id:"Heading2", name:"Heading 2", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:24, bold:true, font:"Times New Roman", color:"2E4F8A" },
        paragraph:{ spacing:{ before:200, after:80 }, outlineLevel:1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width:12240, height:15840 },
        margin: { top:1440, right:1440, bottom:1440, left:1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom:{ style:BorderStyle.SINGLE, size:4, color:"AAAAAA", space:1 } },
          children: [
            new TextRun({ text:"Maternal care utilisation clusters in Malawi: 2015 vs 2024", italics:true, size:18, font:"Times New Roman", color:"666666" }),
            new TextRun({ text:"\t", size:18 }),
            new TextRun({ children:[PageNumber.CURRENT], size:18, font:"Times New Roman", color:"666666" })
          ],
          tabStops: [{ type:"right", position:9360 }]
        })]
      })
    },
    children: C
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/Users/edesi/Documents/maternal health_ml/data/comparative_analysis/manuscript/maternal_health_malawi_comparative_2015_2024.docx", buf);
  console.log("SUCCESS");
}).catch(e => { console.error("ERROR:", e); process.exit(1); });
