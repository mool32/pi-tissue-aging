# Transcriptomic noise accumulates within tissue identity across human aging: a systemic signature distinct from cell-composition drift

## Authors

Theodor Spiro¹

¹Vaika, Inc., East Aurora, NY, USA. Correspondence: theospirin@gmail.com

**Draft v4** — 2026-04-19

---

## Abstract

Aging is often described in two competing languages: accumulation of specific damaged or senescent cell populations versus systemic regulatory erosion affecting every cell. We provide a direct quantitative test of this dichotomy using variance-decomposition of bulk transcriptomes across age. In 263 GTEx v8 donors (20–79 years) with matched samples in six tissues, tissue identity accounts for ≈ 0.73 of transcriptomic variance and declines by only 0.031 over forty years (variancePartition REML 0.789 → 0.758, PERMANOVA R² = 0.858; observed π is 243-fold above a permutation null). The small decline is absorbed almost entirely by *within-tissue, within-donor residual variance* (π_residual 0.168 → 0.194), not by between-donor systemic factors (π_donor stable at ~0.064). The signature is therefore systemic noise accumulating within every cell type, not a shift toward outlier populations. At the single-cell level, the residual growth is partly compositional (cell-type proportions shift) and partly cell-intrinsic: two platforms from the Tabula Muris Senis (Smart-seq2 and 10x Chromium) give complementary reads, with 10x — which has lower age-related gene-detection bias — showing Δπ ≈ −0.07 per cell type in balanced analysis. Per-tissue, noise accumulates at very different rates: hematopoietic tissue grows in variance ~3-fold faster than skeletal muscle (Δvar = +0.079 vs +0.028 over 40 years), but rates do not cleanly track a dividing/post-mitotic axis — left ventricular myocardium accumulates the largest Δvar of any tissue tested (+0.121) despite its post-mitotic nature. Across four mammalian species (mouse, rat, macaque, human), the rate of π_tissue decline scales inversely with maximum lifespan (α = −1.02 ± 0.24, R² = 0.90, Spearman ρ = −1.0). Caloric restriction in rat partially reverses aging-associated π loss in marrow by reducing residual variance rather than restoring tissue-specific variance (87% rescue; bootstrap 95% CI 82–91%; mechanism confirmed in 100% of iterations). Our transcriptomic measurement captures tissue-specific noise accumulation rates that are not represented in methylation clocks, which are themselves largely tissue-invariant; the two signals are therefore complementary. We interpret the results as direct quantitative support for the systemic-noise view of aging, without denying the reality of specific senescent populations, and identify the hematopoietic compartment and left-ventricular myocardium as tissues worth prioritising for mechanistic follow-up.

**Keywords:** aging, tissue identity, variance decomposition, systemic noise, GTEx, single-cell RNA-seq, caloric restriction, cross-species scaling, methylation clocks.

---

## 1. Introduction

Current research on the mechanisms of aging divides, broadly, between two perspectives. On one view, aging reflects the accumulation of specific damaged or senescent cell populations whose growth disrupts tissue function; removing those cells should slow or reverse age-related decline. On another view, aging is a systemic process affecting every cell, manifesting as progressive regulatory noise that cannot be localized to a minority subpopulation. Both views have accumulated evidence — senolytic interventions produce functional benefit in mice (Baker et al. 2016); at the same time, DNA-methylation clocks (Horvath 2013; Hannum et al. 2013) track age with striking precision in essentially every tissue tested, consistent with a universal and systemic drift. What has been missing is a direct quantitative comparison of these two views within one dataset, using a metric that distinguishes "variance between cells/populations" from "variance within each cell type".

Here we provide that test. We apply a three-level variance decomposition (V_total = V_tissue + V_donor + V_residual) to bulk transcriptomic profiles from 263 GTEx v8 donors with matched samples in six tissues. Tissue identity (π_tissue = V_tissue / V_total) captures how much of total variation is explained by tissue of origin; the donor component (π_donor) captures inter-individual systemic factors; the residual (π_residual) captures within-donor, within-tissue stochastic variation. If aging is driven primarily by outlier populations within each tissue, it should appear in the residual (increased within-tissue heterogeneity) without strongly changing donor-level variance. If aging is driven by systemic donor-level factors, it should appear in the donor component. If aging simply converges tissues toward a shared degraded state, π_tissue itself should decline substantially.

We show that, over forty years of human aging, the picture is well-defined: π_tissue shifts very little (Δπ ≈ −0.031), π_donor is essentially unchanged (Δ < 0.005), and the change is carried almost entirely by the residual (+0.026). This is, quantitatively, the signature predicted by the systemic-noise view: every cell is getting noisier, and neither the tissue's identity nor the individual's systemic profile is where the action is. The signature is distinct from what methylation clocks see: DNAm-based age predictors are remarkably consistent *across* tissues, whereas transcriptomic residual noise is markedly *tissue-specific* in its rate, with hematopoietic tissue accumulating noise ~3-fold faster than skeletal muscle. The two kinds of signal are therefore complementary: DNAm clocks as universal time-keepers, transcriptomic residual variance as a tissue-resolved readout of regulatory reserve.

We then use this framework to evaluate three biological questions: (i) whether the bulk residual growth is cell-intrinsic, compositional, or both; (ii) which tissues accumulate noise fastest and whether rates track a simple dividing-vs-post-mitotic axis; (iii) whether rates scale with species lifespan. The resulting picture, which we present below, supports an interpretation in which aging produces progressive, tissue-specific, systemic noise within an essentially preserved compositional architecture, with the rate of decline calibrated to maximum lifespan and partially reversed by caloric restriction. Prior work established that tissue differences dominate transcriptomic variation (Melé et al. 2015), that tissues may converge during mouse aging (Izgi et al. 2022), and that tissue-specific gene downregulation occurs in many tissues (Chatsirisupachai et al. 2023). Our contribution is to put these observations into a single variance-decomposition framework, link that framework to the systemic-noise hypothesis, and test it with interventions.

---

## 2. Data sources

This paper draws on five datasets, each addressing a specific sub-question. We list them here once, before the results, so the reader can keep the landscape in view.

- **GTEx v8** (human; bulk RNA-seq; 263 donors aged 20–79 with matched samples in 6 tissues; additional 15-tissue panel for per-tissue noise rates). Source: Mele et al. 2015; GTEx Consortium 2020. Used for the primary variance decomposition (π_tissue, π_donor, π_residual), per-tissue noise-rate analysis, and sex stratification (Sections 3.1–3.3, 3.5, 3.6).

- **Tabula Muris Senis — FACS** (mouse; Smart-seq2 scRNA-seq; 110,824 cells across 23 tissues; ages 3 and 24 months). Source: Schaum et al. 2020. Used for single-cell validation of compositional versus cell-intrinsic origin of the bulk residual growth (Section 3.4).

- **Tabula Muris Senis — Droplet** (mouse; 10x Chromium scRNA-seq; 245,389 cells across 16 tissues; ages 1, 3, 18, 21, 24, 30 months). Source: Schaum et al. 2020. Used as the independent-platform single-cell validation with lower age-related gene-detection bias (Section 3.4).

- **Calico rat aging atlas** (rat; 10x Chromium scRNA-seq; 218,971 cells across 9 tissues; three conditions: young 5 mo, old ad libitum 27 mo, old caloric restriction 27 mo; n = 2 animals per condition). Source: Zou et al. 2022. Used for the CR intervention test on π_tissue (Section 3.7).

- **Macaque cross-species atlas** (rhesus macaque; bulk RNA-seq; 500 samples from 17 animals across 32 tissues; ages 3–27 years). Source: Li & Kong 2025. Used for cross-species scaling of π erosion rate alongside mouse, rat, and human (Section 3.8).

In what follows, the primary analysis uses GTEx (Sections 3.1–3.3, 3.5, 3.6). Single-cell validation uses the two TMS arms (Section 3.4). The CR intervention test uses the Calico rat atlas (Section 3.7). The cross-species scaling combines all four species (Section 3.8).

---

## 3. Results

### 3.1 Transcriptomic noise accumulates within tissue identity across human aging

We first characterize the central phenomenon — where, in the variance structure of the aging transcriptome, the signal of aging actually lives. For 18,000 expressed genes across 263 GTEx v8 donors matched for six tissues (skeletal muscle, whole blood, sun-exposed skin, subcutaneous adipose, tibial artery, thyroid), we decomposed total transcriptomic variance as V_total = V_tissue + V_donor + V_residual and tracked the proportions (π) across four age bins (20–39, 40–49, 50–59, 60–79 years; Fig. 1A).

Over four decades of aging:

- **π_tissue** declined from 0.764 to 0.733 — a small change of −0.031.
- **π_donor** changed from 0.062 to 0.066 — a shift of +0.004, effectively no change.
- **π_residual** grew from 0.168 to 0.194 — a gain of +0.026, capturing virtually all of what π_tissue lost.

Put another way: tissue identity is preserved at ≈ 73% of total variance across the adult lifespan, inter-individual systemic differences remain a stable ~6% at every age, and the change with aging shows up as within-tissue, within-donor residual noise. This is the signature predicted by the systemic-noise view of aging: in every tissue, in every donor, cells are getting somewhat noisier, and the change does not concentrate into either a systemic donor-level signal or a collapse of tissue identity.

The decomposition is confirmed by independent methods. variancePartition REML on 2,000 randomly sampled genes per age bin yielded π_tissue = 0.789 in young donors and 0.758 in old (identical Δ to the ANOVA estimate). PERMANOVA on the full sample-by-gene matrix returned R² = 0.858 for the tissue factor (p = 0.001 from 999 permutations). A tissue-label-shuffle null gave π_null = 0.003 ± 0.001, placing the observed signal 243-fold above chance (Fig. 1C). Gene subsampling shows convergence by ≈ 3,000 genes; removing any single tissue changes π by at most 0.07 (blood most influential); the age trajectory is stable for 4, 5, or 6 tissues (Δπ_tissue = −0.031 to −0.036).

Batch effects account for 25.7% of total variance but are not confounded with age (Kruskal–Wallis p = 0.80 for age distribution across batches; Spearman ρ (batch, age) = 0.19); RNA integrity is nearly uncorrelated with age (ρ = −0.064). Sex-stratified decomposition shows a modest asymmetry — females decline from 0.776 to 0.725 (Δ = −0.051) versus males 0.768 to 0.743 (Δ = −0.025) — consistent with hormonal transitions at menopause but based on modest sample sizes within age-sex strata.

The decline is concentrated in the 20–39 → 40–49 transition (0.789 → 0.753 by variancePartition), with subsequent decades essentially flat (0.757, 0.758). The compositional story is, at the aggregate level, a small early settling followed by a plateau — with all the ongoing change living in residual noise.

### 3.2 Relationship to DNA methylation clocks

DNA-methylation–based age predictors — the Horvath clock and its successors — track chronological age with striking precision, and a central feature is their *tissue invariance*: the same panel of CpG sites predicts age in blood, skin, brain, and liver alike (Horvath 2013; Hannum et al. 2013; Field et al. 2018). Our transcriptomic measurement, by contrast, is strongly tissue-specific in its rate: per-tissue residual variance growth ranges from −0.012 (artery) to +0.121 (left-ventricular myocardium), a factor-of-10 spread within the same donors (Section 3.5 below). These two readouts of aging are therefore complementary — methylation clocks capture a universal timekeeper, transcriptomic residual variance captures tissue-resolved regulatory reserve. The underlying phenomenon is plausibly the same: methylation heterogeneity at the locus scale is known to increase with age (Jones 2015; Slieker et al. 2016), and our framework reports its downstream transcriptomic manifestation. What is new is the tissue-specific rate, which is not visible to tissue-invariant methylation clocks.

### 3.3 The decline is near-invariant: tissue identity is preserved even as noise grows

At an observed rate of ≈ 0.08% per year, reaching π_tissue = 0.5 from the young value would require roughly 500 years — a timescale not encountered in any mammalian lifespan. Tissue identity is, for practical purposes, preserved across the adult lifespan. The biological signal of aging lies in the residual noise that accumulates *while* tissue identity holds, not in its collapse. This reframes one of the background debates in the field (do aging tissues lose identity, converge, or remain distinct?) as a question not quite about identity loss but about the rate at which systemic noise grows inside a preserved identity.

### 3.4 The residual growth is partly compositional, partly cell-intrinsic

We next asked whether the residual growth reflects shifts in cell-type composition within each tissue or within-cell regulatory drift. We applied the same variance framework to single-cell pseudobulk profiles from the Tabula Muris Senis — first to the FACS (Smart-seq2) arm, then to the Droplet (10x Chromium) arm, which has substantially lower age-related technical bias.

**FACS (Smart-seq2).** Seven cell types present in ≥ 4 tissues (macrophages, endothelial cells, B cells, T cells, NK cells, MSC adipose cells, myeloid cells) were analyzed. In the initial unbalanced comparison, all seven showed π increase with age (binomial p = 0.016). After sample-size balancing between age groups, the signal collapsed: only 3 of 7 showed an increase, the mean Δπ was −0.01, and the binomial test was non-significant (p = 0.69). However, Smart-seq2 exhibited a 29% age-related decline in gene detection (2,824 genes/cell at 3 mo vs 2,002 at 24 mo in macrophages), large enough to mask or fabricate cell-intrinsic signals.

**Droplet (10x Chromium).** With UMI correction, detection decline was only 10% (1,845 vs 1,657 genes/cell), allowing a cleaner comparison. In cross-balanced analysis — matching tissues and subsampling to the same number of pseudobulk samples per tissue — all four cell types with sufficient data showed negative Δπ (Fig. 2):

| Cell type | π_young | π_old | Δπ |
|---|---|---|---|
| Macrophage | 0.648 | 0.560 | −0.088 |
| Endothelial | 0.493 | 0.382 | −0.110 |
| B cell | 0.342 | 0.305 | −0.038 |
| T cell | 0.384 | 0.331 | −0.053 |

Mean Δπ = −0.072. This is a real but modest cell-intrinsic component that was invisible on Smart-seq2 because of detection bias. A 10% residual detection bias remains even on 10x, so the true cell-intrinsic Δ may be smaller than −0.07.

**Synthesis.** The bulk residual growth in GTEx (≈ +0.026) is consistent with both compositional drift and a modest cell-intrinsic component. Our best interpretation is that composition changes dominate — roughly two-thirds of the effect — while cell-intrinsic drift contributes the remaining third. This is also consistent with a tau-stratified decomposition (tissue-specific genes decline 4-fold less than ubiquitous genes; Fig. 2C).

### 3.5 Per-tissue noise accumulation is highly heterogeneous and not cleanly predicted by turnover architecture

If aging drives systemic noise through a universal mechanism, we would expect similar rates across tissues. Conversely, if it is coupled to tissue architecture, we should see systematic differences — for example, between continuously dividing epithelia (which reassemble their chromatin at every division) and post-mitotic tissues (which must maintain chromatin without disassembly). We tested both hypotheses by extending GTEx analysis to 15 tissues spanning three architectural classes: continuously dividing (whole blood, colon, esophagus mucosa, sun-exposed skin, non-sun-exposed skin), intermediate (thyroid, breast, lung, adipose), and post-mitotic / stationary (muscle, heart left ventricle, heart atrial appendage, tibial artery, aorta, tibial nerve). For each tissue we compute the median per-gene change in expression variance between young (20–39) and old (60–79) donors, with bootstrap 95% CI over genes (Fig. 3).

The results are more textured than either simple hypothesis predicts:

- **Hematopoietic tissue is exceptional in rate.** Whole blood shows Δvar = +0.079 [+0.075, +0.083], the second-largest value in the panel. Esophageal mucosa (+0.062) and colon (+0.033) also sit in the fast-accumulation region, consistent with noise growth in continuously dividing epithelia.
- **Skin shows little residual growth** (−0.0015 for sun-exposed; −0.006 for non-sun-exposed) despite being a dividing tissue. One plausible explanation is bulk skin being dominated by long-lived differentiated keratinocytes and adipocytes, with the dividing basal layer a small fraction of the signal; another is the specific GTEx collection protocol. This is a counter-example to a simple "dividing ⇒ high noise growth" hypothesis.
- **Post-mitotic tissues are heterogeneous.** Left-ventricular myocardium shows the *largest* Δvar in the panel (+0.121 [+0.117, +0.124]), atrial appendage is high (+0.063), but aorta, tibial artery, and tibial nerve are near zero or slightly negative. The heart result is striking given that cardiomyocytes renew at <1% per year; the likely drivers are age-related macrophage infiltration, fibroblast activation, and progressive fibrosis within a nominally post-mitotic parenchyma — processes that generate within-donor heterogeneity without requiring cardiomyocyte division.
- **Group-level contrast is weak.** Mean Δvar in the dividing group (n = 5) is +0.033 and in the post-mitotic group (n = 6) is +0.036 (Mann–Whitney U = 15, p = 0.53 for dividing > post-mitotic). A simple architectural dichotomy does not explain the data.

The reading we favor is that tissues differ in noise-accumulation rate for multiple distinct reasons — clonal hematopoiesis in blood, age-related fibrosis and immune infiltration in heart, continuous turnover in gut epithelium — rather than a single "dividing vs post-mitotic" axis. The practical consequence is that rate ordering provides a map of tissues worth mechanistic follow-up: **whole blood, left-ventricular myocardium, and esophageal mucosa** are the three largest Δvar signals, each with distinct biological hypotheses for what drives the growth.

We also confirmed, for whole blood specifically, that the noise growth is consistent with clonal hematopoiesis: eight known CH driver genes (DNMT3A, TET2, ASXL1, JAK2, TP53, SF3B1, SRSF2, PPM1D) show 1.7-fold to 2.4-fold variance increases with age specifically in blood, not in solid tissues. This is the expected signature of donor-specific clonal expansion inflating residual variance.

### 3.6 Chromatin remodeling genes lose tissue specificity faster than expression-matched controls

As a mechanistic probe of what gene classes drive the residual growth, we examined whether genes in canonical regulatory categories lose tissue specificity at different rates than expression-level–matched controls. Chromatin remodeling genes (DNMT1, DNMT3A, DNMT3B, TET1–3, HDAC1–6, EZH2, SIRT1/6/7, SMARCA4, ARID1A; n = 30) showed the most negative Δπ (median = −0.057). Expression-matched controls (10 matched genes per chromatin gene) showed Δπ = −0.023. Chromatin genes therefore erode tissue specificity 2.5-fold faster than matched controls (Mann-Whitney U p = 0.009). Transcription factors show a similar trend (Δπ = −0.039 vs controls −0.024; p = 0.19, n = 18). Downstream targets are relatively protected (Δπ = −0.012 vs controls −0.030, p = 0.09). Housekeeping genes match their expression-level expectation exactly (p = 0.96). The expression-independent hierarchy — chromatin fastest, TFs intermediate, targets protected, housekeeping on baseline — is consistent with a top-down cascade in which loss of tissue-specific chromatin machinery precedes and induces downstream drift.

Given the dual-source decomposition (Section 3.4), the chromatin signal may reflect either differential sensitivity to composition changes or genuine within-cell erosion of chromatin gene tissue specificity; single-cell variance decomposition of the chromatin gene panel is the clean test, which we flag as follow-up.

### 3.7 Caloric restriction partially reverses noise accumulation by reducing residual variance

We used the Calico rat aging atlas (Zou et al. 2022; young 5 mo, old ad libitum 27 mo, old caloric restriction 27 mo; n = 2 animals per condition) to test whether residual-variance growth is reversible by intervention. In the rat, π_tissue declined from 0.893 (young) to 0.842 (old_AL) — a 5.1 percentage point loss in 22 months. Caloric restriction restored π to 0.886 (Fig. 4).

We bootstrapped the rescue estimate 1,000 times over genes: rescue = 87.0% [95% CI 82.4%, 91.3%] — a narrow interval despite the small animal number because each GSM contains thousands of cells. To resolve *how* CR acts, we examined absolute variance components. Aging decreased V_tissue (structural weakening) while V_residual grew. CR left V_tissue essentially unchanged compared to old_AL (ΔV_tissue = −0.000056; 95% CI [−0.000104, −0.000009]) but reduced V_residual substantially (ΔV_residual = −0.000061; 95% CI [−0.000071, −0.000052]). V_residual decreased in 100% of bootstrap iterations; V_tissue increased in only 1.1%.

CR therefore operates as a **noise filter**, reducing within-tissue stochastic variation without restoring tissue-specific expression programs. CR-mimetic compounds (rapamycin, metformin) are predicted by this framework to share the same signature. Restoring tissue-specific programs themselves — rather than filtering their noise — would likely require interventions targeting chromatin state directly, such as partial epigenetic reprogramming.

The caveat is n = 2 biological replicates per condition. The bootstrap CI is a gene-level, not animal-level, uncertainty. We report the direction (rescue is reliably positive on both metrics) with more confidence than we report the magnitude.

### 3.8 Cross-species scaling: erosion rate scales inversely with lifespan

We repeated the π_tissue computation in four mammals spanning ~30× in maximum lifespan: mouse (2.5 yr), rat (3 yr), rhesus macaque (40 yr), human (80 yr). For the macaque, we used the Li & Kong 2025 dataset (500 samples, 17 animals, ages 3–27 yr), excluding a batch-confounded Middle_aged group (all Middle_aged animals in batch 1, all others in batch 2), and applied balanced subsampling of n = 4 animals per group (100 iterations). The macaque declined from π = 0.860 (young adult, 7.5 yr) to 0.753 (elderly, 25 yr), Δπ = −0.107 over 17.5 years (bootstrap 95% CI −0.116, −0.097). The mouse trajectory was recomputed from bulk RNA-seq (10 age points, 17 tissues) rather than FACS pseudobulk, which had produced an artifactually positive slope; corrected mouse Δπ = −0.119 over 24 mo (dπ/dt = −0.060/yr).

Putting the four species together:

| Species | Lifespan (yr) | π_young | π_old | dπ/dt (/yr) |
|---|---|---|---|---|
| Mouse | 2.5 | 0.607 | 0.488 | −0.060 |
| Rat | 3.0 | 0.893 | 0.842 | −0.028 |
| Macaque | 40 | 0.860 | 0.753 | −0.006 |
| Human | 80 | 0.764 | 0.733 | −0.001 |

A log-log fit gives α = −1.02 ± 0.24 (R² = 0.90, p = 0.052), consistent with simple inverse proportionality to lifespan; the Spearman rank correlation is ρ = −1.0 across all four species. Absolute π values are not directly comparable across species because they depend on the number of tissues included; only within-species rates are used in the fit.

Species with longer lifespan accumulate transcriptomic noise more slowly by a factor that tracks lifespan — a quantitative echo of earlier scaling laws linking life-history variables (metabolism, heartbeat, cell turnover) to maximum lifespan. The interpretation we favor is that the maintenance machinery in long-lived species is tuned to keep cumulative noise at a roughly similar level by the end of life, with the per-year rate compressed to match; but this remains a structural observation, not a mechanism.

### 3.9 Killed hypothesis

We also tested, and rejected, a stronger formulation: π_tissue(t) + D(t) = const, where D would be some complementary organizational principle whose growth compensates for the decline in π_tissue. Random gene sets achieve the same variance redistribution, indicating no conservation beyond the trivial constraint that variance proportions sum to 1. We report this here for transparency rather than as a result.

---

## 4. Discussion

### 4.1 What the data support

The central finding is a quantitative signature of systemic, tissue-specific noise accumulation superimposed on a largely preserved tissue-identity architecture. Across 40 years of human aging, only ~3% of total transcriptomic variance leaves the tissue component; nearly all of that ~3% is absorbed by within-tissue residual noise, not by donor-level systemic factors. This is the pattern expected if aging operates primarily as regulatory noise within every cell type rather than through the rise of outlier populations. The existence of specific senescent and clonally expanded populations is not in question — they are well documented and mechanistically important — but in the aggregate variance structure of the aging transcriptome, their contribution is a fraction of a slowly growing residual, not the dominant term.

This result connects naturally with DNA-methylation–clock literature (Horvath 2013; Hannum et al. 2013) without duplicating it. Methylation clocks predict chronological age using tissue-invariant CpG panels — the universal timekeeper. Our transcriptomic residual variance exposes the *tissue-specific consequences* of that timekeeper: blood, heart, and esophagus accumulate noise at rates that differ by an order of magnitude, and neither clock nor framework alone captures the full picture. Joint application should be able to distinguish animals that are chronologically old from animals with tissue-specific regulatory reserve compromised to different degrees.

### 4.2 Tissue-specific rates reveal candidate drivers, not a single architectural axis

The simple architectural hypothesis — that continuously dividing tissues accumulate noise faster than post-mitotic ones because chromatin reassembly at mitosis provides error-correction opportunities unavailable to stationary cells — is supported by hematopoietic and digestive epithelium but not by skin or heart. Skin is a dividing tissue with near-zero Δvar, and left-ventricular myocardium — post-mitotic — shows the largest Δvar in the panel. The honest reading is that per-tissue rates reflect a mixture of distinct biological drivers rather than a single architectural dichotomy. Clonal hematopoiesis is the established driver in blood, and our results confirm the classical CH gene signature. In heart, the likely driver is a combination of macrophage infiltration and progressive fibrosis within a non-dividing parenchyma — a process that generates within-donor heterogeneity without requiring cardiomyocyte turnover. In esophagus, basal-layer proliferation with age-related stem-cell drift is a plausible candidate. These are testable predictions for follow-up, rather than conclusions from the current data.

### 4.3 Caloric restriction as a noise filter, not a structure repair

CR does not restore aged V_tissue toward young levels; it reduces aged V_residual. Mechanistically, this distinguishes CR from interventions that would have to restore the structural component — for example, Yamanaka-factor–based partial epigenetic reprogramming, which targets chromatin state and should be expected to act on V_tissue. Our framework provides a way to distinguish these mechanisms in transcriptomic readouts: interventions that reduce V_residual behave as "noise filters" (CR, rapamycin, metformin are predicted to fall here); interventions that increase V_tissue behave as "structure restorers" (partial reprogramming is predicted here). This mechanistic partition is not obvious from bulk transcriptomic measures other than variance decomposition.

The rat CR result is numerically strong (87% rescue of Δπ) but based on n = 2 per condition; the gene-level bootstrap CI is narrow because each GSM contains thousands of cells, but it does not capture inter-animal variance. We report direction (robust) with more confidence than we report magnitude (not yet definitive).

### 4.4 Cross-species scaling of noise accumulation

Inverse proportionality of π erosion rate to maximum lifespan (α ≈ −1.02, R² = 0.90 across four species) is structurally reminiscent of metabolic-rate and cell-division scaling laws. Its most parsimonious reading is that maintenance machinery across mammals is tuned to cap lifetime cumulative noise at roughly similar levels, with per-year rates compressed to match lifespan. We do not have mechanism for this scaling; we report its presence and robustness across species spanning ~30× in lifespan.

### 4.5 Future work

1. **Applying the framework to cancer properly.** The current framework compares compositions of healthy tissues across age. Its extension to cancer requires matching tumor cells not to bulk tissue-of-origin but to the appropriate differentiation stage of the cell type of origin. For example, a B-cell lymphoblastic leukaemia should be compared to B-cells at the specific differentiation stage from which transformation occurred, rather than to bulk blood. Single-cell resolution to isolate transformed cells from the microenvironment is additionally required. We propose this as a separate investigation, and caution that naive tumor-vs-tissue comparisons using π_tissue compare categorically incommensurable quantities (cells versus tissues) and should not be interpreted biologically.
2. **Heart and hematopoiesis as mechanistic priority tissues.** Δvar in blood and heart exceeds all others by a substantial margin. Decomposing each signal into compositional vs cell-intrinsic contributions at single-cell resolution — CH for blood, macrophage/fibroblast infiltration for heart — would connect the bulk-variance framework to specific cellular mechanisms and test specific interventions (senolytics for CH; anti-fibrotics for heart).
3. **Intervention predictions.** CR-mimetics should reduce V_residual; partial reprogramming should increase V_tissue. Applying the same decomposition to published rapamycin, metformin, and Yamanaka-factor datasets provides a direct test.
4. **Spatial transcriptomics to bypass dissociation.** Solid-tissue composition measurements via scRNA-seq are biased by age-dependent dissociation (aged tissue is more cross-linked; cell-type fragility varies). Spatial transcriptomics would sidestep this confound and allow direct measurement of cell-type-composition vs in-situ regulatory state contributions to the bulk variance signal.

### 4.6 Conclusion

Variance decomposition of bulk transcriptomes in 263 GTEx v8 donors across forty years of aging places nearly all age-related change in the residual component — within-tissue, within-donor stochastic variation — rather than in tissue identity or donor-level systemic factors. This is a direct quantitative test of the systemic-noise view of aging, and the data support it: aging manifests as progressive regulatory noise within every cell type, inside an essentially preserved tissue-identity architecture. The signature is complementary to DNA-methylation clocks — which are universal timekeepers — by exposing tissue-resolved rates of regulatory reserve erosion that methylation clocks cannot see. Per-tissue rates differ by an order of magnitude and do not track a simple dividing-vs-post-mitotic architectural axis; hematopoietic tissue and left-ventricular myocardium accumulate noise fastest for distinct mechanistic reasons. Caloric restriction in rat partially reverses noise accumulation by acting as a noise filter on residual variance, not as a restorer of tissue-specific variance — a mechanistic distinction that predicts a specific signature for CR-mimetic compounds and differentiates them from structure-restoring interventions such as partial reprogramming. Across four mammalian species spanning ~30× in lifespan, π-erosion rate scales inversely with maximum lifespan, placing the system alongside classical metabolic and life-history scaling laws. We offer this framework as a quantitative scaffold on which mechanistic hypotheses about aging can be tested, rather than as a mechanism itself.

### 4.7 Limitations

Six tissues in GTEx (chosen for maximum donor overlap — 263 donors with matched samples in all six) is a limited panel; key tissues such as brain were dropped for donor-coverage reasons. Single-cell validation uses mouse TMS to interpret human bulk; cross-species conservation of π as a near-invariant supports this inference, but direct human multi-tissue single-cell aging data would provide cleaner resolution. Cell-type annotation is asymmetric between TMS (fixed ontology) and Kimmel (de novo clustering); our methodological symmetry tests reconcile this for the kidney comparison but not in principle. GTEx donors are deceased; peri-mortem effects on tissue composition cannot be fully excluded. The 20s-to-40s concentration of π decline could partly reflect completion of development rather than onset of aging. The rat CR analysis has n = 2 biological replicates per condition; the direction of effect is robust, the magnitude is provisional. Sex-stratified results (female π declining ≈ 2-fold faster than male) rest on modest sample sizes within age-sex strata and should be considered preliminary. The dividing-vs-post-mitotic hypothesis (Section 3.5) was tested and rejected in its simple form; per-tissue rates should be interpreted through tissue-specific biology, not an architectural dichotomy.

---

## 5. Methods

### 5.1 GTEx data processing

GTEx v8 gene-level TPM values (GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz) and accompanying sample/subject annotations. Six tissues were selected for maximum donor overlap (263 donors with matched samples in all six): Muscle – Skeletal, Whole Blood, Skin – Sun Exposed (Lower leg), Adipose – Subcutaneous, Artery – Tibial, and Thyroid. The constraint on tissue count reflects a trade-off between donor overlap and tissue breadth: adding a seventh tissue would drop the matched-donor count by roughly half. Expression values were log₂(TPM + 1) transformed; genes with median TPM below 0.5 across all samples were excluded, yielding ~18,000 genes. Age bins were defined as 20–39, 40–49, 50–59, and 60–79 years (matching GTEx annotation bins).

For the 15-tissue per-tissue noise analysis (Section 3.5), we used all tissues in GTEx with ≥ 30 donors in both the young (20–39) and old (60–79) bins, without restricting to a matched-donor panel; this expands the tissue count at the cost of not sharing donors across tissues.

### 5.2 Three-level ANOVA variance decomposition

For each gene, V_total = V_tissue + V_donor + V_residual, where V_tissue = n_donors × Σ(tissue_mean − grand_mean)², V_donor = n_tissues × Σ(donor_mean − grand_mean)², and V_residual = V_total − V_tissue − V_donor. π_tissue is V_tissue / V_total per gene; the summary statistic is the median π across all expressed genes. Per-decade analyses apply the same decomposition restricted to donors within one age bin.

### 5.3 variancePartition REML validation

variancePartition R package (v1.40.1) with formula ~ (1|tissue) + (1|donor), REML, on 2,000 randomly sampled genes per age bin. REML rather than ML is used because ML estimators of variance components are biased downward with few factor levels; REML profiles out fixed effects before estimation, yielding unbiased variance-component estimates. The overall age-including model adds age_mid as a fixed effect.

### 5.4 PERMANOVA validation

adonis2() from the vegan R package with Euclidean distances computed on the top 500 most variable genes, model d ~ tissue + age_bin + sex, 999 permutations. R² values per term are extracted from the PERMANOVA table. PERMANOVA is non-parametric and distribution-free; we use it here as a second independent check that the tissue signal dominates the bulk variance, without relying on variance-component modeling assumptions.

### 5.5 Permutation null

Tissue labels are shuffled 100 times (breaking the tissue-sample mapping while preserving all other structure) and π_tissue is recomputed. The null distribution (π_null = 0.003, SD = 0.001) establishes a baseline against which the observed π ≈ 0.73 is compared.

### 5.6 Tissue specificity index (τ)

For each gene, τ = Σ(1 − expr_i / max_expr) / (n_tissues − 1), where expr_i is the mean expression in tissue i. Genes with τ > 0.8 are classified tissue-specific; τ < 0.3 ubiquitous.

### 5.7 Single-cell validation: TMS FACS and TMS Droplet

We used the Tabula Muris Senis FACS dataset (Schaum et al. 2020; 110,824 cells, 23 tissues, Smart-seq2 platform, ages 3 and 24 months) and the Droplet dataset (245,389 cells, 16 tissues, 10x Chromium, ages 1–30 months). Cell type annotations from the original study. For each cell type, pseudobulk expression was computed per (tissue, age, mouse). π_tissue was computed using the standard ANOVA framework on these pseudobulk profiles. For cross-balanced analysis on the Droplet arm, tissues present in both young (1, 3 mo) and old (18, 21, 24, 30 mo) groups were identified per cell type, and both groups were subsampled to the minimum number of pseudobulk samples per tissue across both groups, iterated 100 times. Gene detection rates per cell (number of genes with non-zero expression) were computed to quantify platform-specific age bias (29% decline on FACS vs 10% on Droplet).

### 5.8 Expression-matched functional category analysis

Genes were assigned to four functional categories: chromatin remodeling (DNMT1, DNMT3A, DNMT3B, TET1, TET2, TET3, HDAC1-6, EZH2, SIRT1, SIRT6, SIRT7, SMARCA4, ARID1A; n = 30), transcription factors (n = 18 curated from Lambert et al. 2018), downstream targets (n = 50), and housekeeping genes (n = 100 from Eisenberg and Levanon 2013). For each focal gene, 10 expression-level-matched genes were selected from the remaining genome by nearest-neighbor matching on mean log₂(TPM+1). Δπ distributions were compared by two-sided Mann-Whitney U.

### 5.9 Per-tissue noise accumulation (15 tissues)

For 15 GTEx tissues with ≥ 30 donors in both young (20–39) and old (60–79) age bins, we computed per-gene variance in log₂(TPM+1) expression within each age group, then Δvar = var_old − var_young per gene. Per-tissue summaries report median Δvar across all 18,015 expressed genes, with bootstrap 95% CI obtained from 200 gene-level resamples (this captures gene-level uncertainty; animal-level uncertainty is not separately captured here because inter-donor variance is part of the signal). Tissues were classified a priori into continuously dividing, intermediate, and post-mitotic on the basis of published cell-turnover literature, not post-hoc.

### 5.10 Rat caloric restriction analysis

Calico rat aging atlas (Zou et al. 2022; 218,971 cells, 54 samples across 9 tissues). Conditions: young (5 mo, n = 18 samples across tissues), old ad libitum (27 mo, n = 18), old caloric restriction (27 mo, n = 18). Bone marrow has 2 samples per condition. After normalization (scanpy normalize_total, target 1e4, followed by log1p), we aggregated to pseudobulk per sample and computed π_tissue using the ANOVA framework. Bootstrap CIs for rescue percentage and variance-component changes came from 1,000 gene-resampling iterations; rescue fraction = (π_CR − π_old_AL) / (π_young − π_old_AL). Noise-reduction mechanism was assessed by the fraction of bootstrap iterations in which V_residual decreased (CR vs old_AL) and V_tissue increased (CR vs old_AL).

### 5.11 Cross-species scaling

Mouse: TMS bulk RNA-seq (GSE132040), 17 tissues, 10 age points (1–27 mo). Adult-only slope (3–27 mo) excludes developmental maturation. Macaque: Li & Kong 2025 (Figshare 26963386), 500 samples from 17 female rhesus macaques, 32 tissues, ages 3–27 years; batch 2 only (14 animals, 3 age groups: Juvenile, Young_adult, Elderly), 6 tissues common to all animals, balanced subsampling (n = 4 animals per group, 100 iterations). Rat: Calico atlas as above. Human: GTEx as above. All datasets were processed with log₂(CPM + 1) or log₂(TPM + 1) as appropriate. Within-species erosion rates were computed by linear regression on adult ages. The cross-species scaling law was fit as log|dπ/dt| = α × log(L) + β; Spearman rank correlation provides a non-parametric cross-check.

### 5.12 Sex-stratified analysis

GTEx donors were split by sex and π_tissue trajectories computed independently across age decades. Inter-sex differences in Δπ are reported descriptively given the modest sample sizes within age-sex strata.

### 5.13 Statistical tests

Bootstrap confidence intervals come from 100 resamples of donors (for GTEx tissue-level metrics) or 1,000 resamples of genes (for the rat CR rescue). Permutation null uses 100 shuffles of tissue labels. All p-values are two-sided; unless noted otherwise, no multiple-testing correction is applied because each comparison tests a distinct threat to validity (batch, permutation, gene subsampling, tissue subsampling, platform concordance, sex stratification, single-tissue sensitivity). This is deliberate: we use concordance across independent tests rather than adjusted p-values as the primary inferential tool.

### 5.14 Code and data availability

All analysis code, intermediate CSVs, and figures are released at
https://github.com/mool32/pi-tissue-aging (a local git repository
accompanies this submission and will be pushed to GitHub on preprint
deposit). Raw upstream data must be obtained separately: GTEx v8 from
dbGaP (phs000424.v8, requires approval); Tabula Muris Senis FACS and
Droplet from GEO (GSE132040) and figshare (doi:10.6084/m9.figshare.12654728);
Calico rat atlas from GEO (GSE141784); macaque data from figshare (26963386).

---

## Figure Legends

**Figure 1. Transcriptomic noise accumulates within tissue identity across human aging.** (A) Schematic of three-level variance decomposition V_total = V_tissue + V_donor + V_residual. (B) Stacked bar of π_tissue (green), π_donor (blue), π_residual (gray) across four age bins in GTEx v8 (263 donors, 6 tissues, 18,000 genes). π_tissue = 0.764 → 0.733, π_donor stable at ~0.064, π_residual growing from 0.168 to 0.194. (C) Permutation null: π_null = 0.003 from 100 tissue-label shuffles vs observed π = 0.73 (243-fold enrichment). (D) variancePartition REML confirmation: 0.789 → 0.758 (identical Δ to ANOVA).

**Figure 2. Cross-platform single-cell validation reveals predominantly compositional decline with a cell-intrinsic component.** (A) Gene-detection QC: Smart-seq2 29% decline vs 10x Chromium 10% decline in genes/cell with age. (B) Cross-balanced 10x Chromium analysis: all four cell types show negative Δπ (mean = −0.07). (C) Tau-stratified decomposition: tissue-specific genes (τ > 0.8) decline minimally (Δ = −0.011); ubiquitous genes (τ < 0.3) decline more (Δ = −0.041).

**Figure 3. Per-tissue noise accumulation across 15 GTEx tissues, classified by turnover architecture.** Median Δvar (old − young) per gene, with 95% bootstrap CI over genes. Tissues colored by a priori architectural class: continuously dividing (red), intermediate (orange), post-mitotic (blue). Whole blood (+0.079), esophagus mucosa (+0.062), heart left ventricle (+0.121) accumulate noise fastest; skin and tibial artery are near zero. A simple dividing-vs-post-mitotic contrast does not explain the pattern (Mann–Whitney p = 0.53).

**Figure 4. Caloric restriction partially reverses marrow noise accumulation as a noise filter, not a structure repair.** (A) π_tissue in the Calico rat atlas: young = 0.893, old_AL = 0.842, old_CR = 0.886 (87% rescue; 95% CI 82–91%). (B) Bootstrap rescue-fraction distribution (n = 1,000 gene resamples). (C) Mechanism: V_residual decreases in 100% of iterations (noise reduction); V_tissue increases in only 1.1% (not structure restoration).

**Figure 5. Cross-species scaling of π erosion rate with lifespan.** Four mammals (mouse, rat, macaque, human) in log-log (|dπ/dt|, lifespan) coordinates. Power-law fit α = −1.02 ± 0.24 (R² = 0.90); Spearman ρ = −1.0. Dashed line α = −1 (inverse proportionality).

---

## References

(Full reference list to follow. Key in-text citations: Baker et al. 2016 Nature 530:184 senolytics; Horvath 2013 Genome Biol 14:R115; Hannum et al. 2013 Mol Cell 49:359; Field et al. 2018 Mol Cell 71:882; Jones 2015 Nat Rev Genet 16:286; Slieker et al. 2016 Genome Biol 17:191; Mele et al. 2015 Science 348:660; Izgi et al. 2022 eLife 11:e68048; Chatsirisupachai et al. 2023 BMC Genomics 24:65; Schaum et al. 2020 Nature 583:596; Zou et al. 2022 bioRxiv; Li & Kong 2025 figshare 26963386; Eisenberg & Levanon 2013 Trends Genet 29:569; Lambert et al. 2018 Cell 172:650.)

---

## Acknowledgements

Andrei V. Gudkov provided scientific input on the biological framing, the dividing-vs-non-dividing architectural hypothesis, the positioning of this work in the systemic-noise vs selective-accumulation debate, and the recommendation to remove cancer comparisons and the suggestion of the Vaika affiliation. Computational analysis, code generation, and manuscript preparation were assisted by Claude (Anthropic). All scientific claims, analyses, and conclusions are the author's responsibility.

---
