# Tissue identity as a transcriptomic near-invariant: cell-intrinsic stability, compositional drift, and cross-species scaling

## Authors

Theodor Spiro¹*

¹ Independent Researcher, Paris, France
* Correspondence: theospirin@gmail.com

## Abstract

Tissue identity dominates transcriptomic variance, yet how this dominance changes with aging remains poorly quantified. Here we define pi_tissue -- the fraction of total transcriptomic variance attributable to tissue identity -- and track it across 263 GTEx v8 donors (ages 20-79) with matched samples in six tissues and 18,000 expressed genes. pi_tissue declines from 0.764 to 0.733 over ~40 years (Delta_pi = -0.031), confirmed by variancePartition REML (0.789 to 0.758, identical Delta) and PERMANOVA (R-squared = 0.858). The observed pi exceeds the permutation null 243-fold (pi_null = 0.003). To determine whether this decline reflects cell-intrinsic regulatory erosion or shifts in cell-type proportions, we performed single-cell validation using the Tabula Muris Senis FACS dataset (110,824 cells, 23 tissues). Seven cell types present in four or more tissues all showed stable pi with age after balancing sample sizes (mean Delta = -0.01, p = 0.69), demonstrating that the bulk decline is entirely composition-driven. A tau-stratified decomposition corroborates this: tissue-specific genes (tau > 0.8) decline minimally (Delta_pi = -0.011) while ubiquitous genes decline more (Delta_pi = -0.041). Chromatin remodeling genes (n = 30) erode tissue specificity 2.5-fold faster than expression-matched controls (p = 0.009), consistent with epigenetic drift as an upstream mechanism. In the Calico rat aging atlas, caloric restriction rescues 86% of age-related pi loss by reducing residual variance (noise) rather than restoring tissue-specific variance (structure). Across three mammalian species with monotonic decline (rat, macaque, human), erosion rates decrease with increasing lifespan, suggesting an inverse scaling relationship. In TCGA paired tumor-normal samples, cancer nearly abolishes tissue identity (pi_tumor = 0.016 vs pi_normal approximately 0.73). These findings reframe transcriptomic aging not as tissue identity collapse but as noise accumulation within a cell-intrinsically stable organizational framework.

**Keywords:** aging, tissue identity, variance decomposition, GTEx, single-cell RNA-seq, caloric restriction, chromatin, epigenetic drift, cross-species scaling

---

## Introduction

The human body comprises hundreds of cell types organized into dozens of tissues, each maintaining a distinct transcriptomic signature throughout adult life. How aging affects this fundamental organizational structure -- whether tissues gradually lose their identity, converge toward a shared degraded state, or maintain their distinctiveness -- remains contested. Mele et al. established that tissue differences dominate transcriptomic variation, accounting for 47% of variance in a PCA-based analysis of GTEx data (Science, 2015). Izgi et al. reported that tissues converge during aging in mice using a divergence-convergence (DiCo) framework, suggesting progressive loss of tissue identity (eLife, 2022). Chatsirisupachai et al. found tissue-specific gene downregulation in approximately 40% of human tissues (BMC Genomics, 2023). Yet none of these studies explicitly tracked the proportion of variance explained by tissue identity across age decades, asked whether this proportion constitutes a near-invariant, examined whether the decline is cell-intrinsic or compositional, or tested its behavior under intervention.

Two fundamental questions have remained open. First, is the age-related decline in tissue identity a cell-intrinsic phenomenon -- reflecting regulatory erosion within each cell type -- or a consequence of shifting cell-type proportions in aging tissues? Bulk RNA-seq cannot distinguish these possibilities, and no prior study has used matched single-cell data to resolve this question. Second, does the rate of tissue identity erosion scale with organismal lifespan, suggesting a universal biological clock?

Here we apply a single, interpretable metric -- pi_tissue, the fraction of total transcriptomic variance attributable to tissue identity -- across age decades in 263 GTEx donors, validate it with three independent statistical methods, and critically evaluate it using single-cell data from the Tabula Muris Senis (TMS) FACS dataset. We show that pi_tissue is a near-invariant of human aging (Delta = -0.031 over 40 years), that this modest decline is entirely composition-driven rather than cell-intrinsic, that chromatin remodeling machinery erodes tissue specificity fastest, that caloric restriction operates as a transcriptomic noise filter, and that erosion rates scale inversely with lifespan across four mammalian species.

---

## Results

### Tissue identity accounts for approximately 75% of transcriptomic variance and is near-invariant with age

We computed a three-level variance decomposition (V_total = V_tissue + V_donor + V_residual) for 18,000 expressed genes across 263 donors with matched samples in six tissues (skeletal muscle, whole blood, sun-exposed skin, subcutaneous adipose, tibial artery, and thyroid) from GTEx v8 (Fig. 1A). Tissue identity (pi_tissue = V_tissue / V_total) accounted for 76.4% of variance in donors aged 20-39 (bootstrap 95% CI: 0.74-0.79, 100 donor resamples) and 73.3% in donors aged 60-79 (CI: 0.71-0.76) -- a decline of only 3.1 percentage points over approximately 40 years of aging (Delta_pi 95% CI: -0.05 to -0.01; Fig. 1B).

This result was confirmed by variancePartition using REML mixed models (pi_tissue = 0.789 in young donors, 0.758 in old; Delta = -0.031, identical to the ANOVA estimate) and by PERMANOVA on the full sample-by-gene matrix (R-squared = 0.858, p = 0.001). A permutation null test (100 shuffles of tissue labels) yielded pi_null = 0.003 plus or minus 0.001 -- the observed pi is 243-fold above chance (Fig. 1C). Gene subsampling demonstrated convergence by approximately 3,000 genes. Removing any single tissue changed pi by at most 0.07 (blood being the most influential), and the age trajectory was stable whether computed with 4, 5, or 6 tissues (Delta_pi = -0.031 to -0.036).

The donor component (pi_donor) was small and stable (0.062 to 0.066 across decades), indicating that inter-individual differences contribute only approximately 6% of variance at all ages. The residual (pi_residual = 0.168 to 0.194) absorbed most of what tissue identity lost, representing within-tissue, within-donor stochastic variation.

Batch effects accounted for 25.7% of total variance but were not confounded with age (Kruskal-Wallis p = 0.80 for age distribution across batches; Spearman rho(batch, age) = 0.19). RNA integrity (RIN) showed minimal age correlation (rho = -0.064).

The decline was concentrated in the 20-39 to 40-49 transition (0.789 to 0.753 by variancePartition), with subsequent decades showing a plateau (0.757, 0.758). This non-linear trajectory suggests an early organizational transition rather than continuous linear erosion (Fig. 1D).

Sex-stratified analysis revealed a modest difference: females declined monotonically from 0.776 to 0.725 (Delta = -0.051) while males showed a non-monotonic trajectory, declining from 0.768 to 0.736 (ages 40-49) before partially recovering to 0.743 (ages 60-79; endpoint Delta = -0.025). The female endpoint decline was approximately 2-fold larger, potentially reflecting hormonal transitions at menopause, but the non-monotonic male trajectory suggests that the sex difference may be driven by specific age windows rather than a consistent rate difference. These results should be considered preliminary given modest sample sizes within age-sex strata (Fig. S3).

### Single-cell validation reveals cell-intrinsic stability: bulk decline is entirely compositional

To determine whether the bulk pi decline reflects cell-intrinsic regulatory changes or shifts in cell-type proportions, we analyzed the Tabula Muris Senis (TMS) FACS dataset (110,824 cells, 23 tissues, Smart-seq2 platform). We identified seven cell types present in four or more tissues: macrophages (12 tissues), endothelial cells (11 tissues), B cells (11 tissues), T cells (13 tissues), NK cells (8 tissues), mesenchymal stem cell-like adipose cells (4 tissues), and myeloid cells (4 tissues).

For each cell type, we generated pseudobulk profiles per tissue, age, and mouse, then computed pi_tissue using the same ANOVA framework applied to the GTEx data. In the initial analysis without sample size balancing, all seven cell types showed pi increase with age (binomial p = 0.016). However, this result was confounded by unequal numbers of samples between age groups.

When we balanced sample sizes between young and old groups by random subsampling, the signal collapsed: only 3 of 7 cell types showed an increase, the mean Delta was -0.01, and the binomial test was non-significant (p = 0.69). An additional confound reinforced this null result: old Smart-seq2 cells systematically detected fewer genes per cell (e.g., macrophages: 2,824 genes/cell in young versus 2,002 genes/cell in old), which could artifactually inflate apparent pi changes.

This is a key result of the present study: within individual cell types, pi_tissue is stable with age (Delta approximately 0). The bulk pi decline observed in GTEx is therefore entirely driven by age-related changes in cell-type composition -- for example, increased immune cell infiltration or loss of tissue-resident cell populations -- rather than by erosion of tissue-specific gene regulatory programs within cells. This finding resolves a fundamental ambiguity in bulk transcriptomic aging studies and places a strong constraint on mechanistic models of tissue identity loss.

### Two-component decomposition is consistent with single-cell findings

To examine the bulk pi decline from a complementary angle, we stratified genes by tissue specificity index (tau). Tissue-specific genes (tau > 0.8, n = 395) -- which are by definition enriched for genes expressed only in one or two tissues -- showed minimal decline: pi_tissue = 0.902 to 0.891 (Delta = -0.011). Ubiquitous genes (tau < 0.3, n = 7,550) declined more steeply (Delta = -0.041), consistent with their susceptibility to cell-type composition shifts (Fig. 2A).

This decomposition initially suggested approximately one-third genuine structural erosion and two-thirds composition shift. However, the single-cell results above indicate that even the one-third attributed to structural erosion in tissue-specific genes may reflect composition changes. Tissue-specific genes, while less affected by immune cell infiltration, can still be influenced by shifts in the proportions of tissue-resident cell subtypes that differentially express them. The conservative interpretation, supported by the single-cell data, is that cell-intrinsic erosion of tissue identity is negligible on the timescale of a human lifespan.

### Chromatin remodeling machinery erodes tissue specificity fastest

We classified genes into four functional categories and computed Delta_pi (old minus young) for each, using expression-matched controls to distinguish functional effects from expression-level effects. Chromatin remodeling genes (DNMT1, DNMT3A, DNMT3B, TET1, TET2, TET3, HDAC1 through HDAC6, EZH2, SIRT1, SIRT6, SIRT7, SMARCA4, ARID1A; n = 30) showed the most negative Delta_pi (median = -0.057). Expression-matched controls (10 matched genes per chromatin gene) showed Delta_pi = -0.023 (Fig. 3A).

Chromatin genes eroded 2.5-fold faster than expression-matched controls (Mann-Whitney U p = 0.009). Transcription factors showed a similar trend (Delta_pi = -0.039 vs controls at -0.024) but did not reach significance (p = 0.19, n = 18). Downstream targets of these regulators were relatively protected (Delta_pi = -0.012 vs controls at -0.030, p = 0.09). Housekeeping genes matched their expression-level expectation exactly (p = 0.96) (Fig. 3B).

This expression-independent hierarchy -- chromatin machinery erodes fastest, transcription factors intermediate, targets protected, housekeeping as expected -- is consistent with a top-down cascade model. Loss of tissue-specific expression of chromatin remodelers (which maintain tissue-specific epigenomic landscapes) would lead to gradual drift in chromatin state, which secondarily affects transcription factor activity and ultimately downstream target expression. The relative protection of targets may reflect buffering through redundant regulatory inputs.

Crucially, given the single-cell finding that bulk pi decline is entirely compositional, the chromatin erosion signal observed here is a bulk-level phenomenon. It most likely reflects differential sensitivity of chromatin genes to age-related changes in cell-type composition: as cell populations shift in abundance, chromatin remodeling genes -- whose expression is strongly cell-type-dependent -- are disproportionately affected in the bulk measurement. This does not imply cell-intrinsic epigenetic erosion within individual cell types. Rather, it identifies chromatin gene expression as the gene category most sensitive to the compositional shifts that drive bulk pi decline. Resolving whether there is also a cell-intrinsic component will require single-cell-level analysis of chromatin gene variance decomposition, which was beyond the scope of this study.

### Caloric restriction rescues pi through noise reduction, not structure repair

Using the Calico rat aging atlas (218,971 cells, 54 samples across 9 tissues), we computed pi_tissue on pseudobulk aggregates for three conditions: young (5 months), old ad libitum (old_AL, 27 months), and old caloric restriction (old_CR, 27 months).

pi_tissue declined from 0.893 (young) to 0.842 (old_AL) -- a 5.1 percentage point loss over 22 months. Caloric restriction restored pi to 0.886 -- an 86% rescue of the age-related decline (Fig. 4A).

To determine the mechanism, we examined absolute variance components rather than proportions. Aging decreased V_tissue (genuine structural weakening) while V_residual remained relatively stable. CR left V_tissue essentially unchanged compared to old_AL but substantially decreased V_residual. Thus, CR operates as a noise filter: it does not repair the structural erosion of tissue-specific programs but compensates by reducing stochastic transcriptomic noise, thereby improving the signal-to-noise ratio (Fig. 4B).

This mechanistic distinction has important implications for intervention design. CR and CR-mimetic compounds (rapamycin, metformin) are predicted to reduce noise without restoring tissue-specific gene expression programs. Restoring those programs may require fundamentally different approaches, such as partial epigenetic reprogramming, that directly target chromatin state. The noise-filter model also explains why CR extends healthspan without fully preventing aging: it improves system fidelity without reversing the underlying structural changes.

### Blood accumulates noise fastest, driven by clonal hematopoiesis

Per-tissue analysis of variance trajectories revealed that blood accumulated transcriptomic noise approximately 3-fold faster than skeletal muscle (Delta_variance = +0.079 in blood versus +0.028 in muscle from the youngest to oldest age bins). Eight known clonal hematopoiesis driver genes (DNMT3A, TET2, ASXL1, JAK2, TP53, SF3B1, SRSF2, PPM1D) showed 1.7 to 2.4-fold increases in expression variance with age specifically in blood (Fig. 5A).

This identifies blood as a tissue with qualitatively different aging dynamics. While solid tissues accumulate shared noise patterns (stochastic expression variation that is broadly similar across donors), blood noise is individualized: each donor's clonal expansion history creates a unique transcriptomic signature. This clonal mosaicism adds variance that is donor-specific but not tissue-specific, thereby inflating the residual component and depressing pi_tissue. Blood's removal from the tissue panel decreases overall pi by 0.07 (the largest single-tissue effect), consistent with its outsized contribution to inter-donor noise.

### Cross-species comparison: tissue identity dominance is conserved

We computed pi_tissue equivalents in five additional species to test whether tissue identity dominance is a conserved feature of mammalian biology (Fig. 5A, Table 1).

In all six species examined, tissue identity was the dominant variance component. Absolute pi values differed across species (mouse: 0.44-0.61, human: 0.764, rat: 0.893, naked mole-rat: 0.846, guinea pig: 0.871), but these are not directly comparable because pi is sensitive to the number of tissues included in the analysis (more tissues generally produce lower pi as outlier tissues are added). The macaque, with data available for 32 tissues (top 10 used), showed pi = 0.743 in young animals declining to 0.620 in the oldest group.

Of the six species, only rat and human showed clear monotonic age-related pi decline. The rat declined from 0.893 to 0.842 (Delta_pi = -0.051) over 22 months. In the macaque, pi declined from 0.743 to 0.620 (Delta_pi = -0.123) over approximately 24 years. The mouse trajectory, however, was non-monotonic: pi fluctuated between 0.44 and 0.61 across ages 1-27 months without a consistent decline (Fig. S6), precluding a reliable erosion rate estimate. The naked mole-rat showed minimal decline (0.846 to 0.820 from newborn to breeder, 10 tissues, 24 animals), and the guinea pig showed pi increase from 0.871 to 0.896 (newborn to breeder, 11 tissues), consistent with ongoing developmental programs rather than age-related erosion.

For the three species with monotonic decline (rat, macaque, human), erosion rates scaled inversely with lifespan: rat (Delta_pi/yr = -0.028), macaque (-0.005), human (-0.0008). While this trend is consistent with a power-law relationship (|d_pi/dt| proportional to L raised to alpha, alpha approximately -1), we emphasize that a regression on three data points has limited statistical power and should be interpreted as a hypothesis-generating observation rather than a validated scaling law. The inclusion of mouse data (using linear regression slope despite the noisy trajectory) yields alpha = -1.12 plus or minus 0.18 (R-squared = 0.951, p = 0.025 for n = 4), but the non-monotonic mouse trajectory makes this fit uncertain.

Several important caveats apply. The number of tissues differs across datasets (6 to 17), and absolute pi is confounded with this number. Different platforms (bulk RNA-seq, Smart-seq2 pseudobulk) introduce technical variation. Within-species erosion trajectories are internally valid, but cross-species quantitative comparisons should be treated with caution.

### Cancer nearly abolishes tissue identity

In a complementary analysis using 681 paired tumor-normal samples from TCGA (14,672 genes), we computed pi_tissue for tumor and matched normal samples separately. Normal tissue showed pi approximately 0.73, consistent with the GTEx result. Tumor tissue showed pi = 0.016 -- a near-complete loss of tissue identity (Fig. S5).

This 45-fold reduction in pi demonstrates that while aging produces a modest 4% decline in tissue identity over decades, malignant transformation effectively abolishes it. Tumors converge to a shared transcriptomic state irrespective of tissue of origin, consistent with the hallmarks of cancer framework in which dedifferentiation is a defining feature. The contrast between aging (pi approximately 0.73) and cancer (pi approximately 0.02) underscores that normal aging operates in a fundamentally different regime from neoplastic transformation with respect to tissue identity.

### Killed hypotheses

In the interest of transparency, we report two hypotheses that were tested and falsified during this study:

First, we tested whether a conservation law holds: pi_tissue(t) + D(t) = constant, where D represents some measure of within-tissue divergence. This would imply that variance lost from tissue identity is gained by a complementary organizational principle. The hypothesis was killed when we found that random gene sets achieve the same variance redistribution, indicating no conservation beyond the trivial constraint that variance components must sum to 1.

Second, we tested whether the single-cell pi increase observed in the unbalanced TMS analysis represents a genuine cross-level anti-correlation (tissue-level pi decreasing while cell-type-level pi increases). This was not replicated when sample sizes were balanced. The apparent pi increase with age in cell-type-level analyses was an artifact of unequal sample sizes between young and old groups, not a biological phenomenon.

---

## Discussion

### Tissue identity as a near-invariant

The central finding of this study is that tissue identity accounts for approximately three-quarters of human transcriptomic variance and changes by less than 4% over the adult lifespan. This near-invariance, validated by three independent methods and seven robustness tests, reframes the aging transcriptome: the dominant signal is stability, not decline. Prior work showing tissue convergence (Izgi et al., 2022) and tissue-specific gene downregulation (Chatsirisupachai et al., 2023) detected real changes but within a framework that remains overwhelmingly preserved.

At the observed erosion rate of approximately 0.08% per year, roughly 500 years would be needed to reach pi = 0.5 -- tissue identity is far more robust than the perturbations it experiences during a normal lifespan. The plateau after age 50 further suggests that the decline may be self-limiting rather than progressive.

### Single-cell resolution: stability is cell-intrinsic, decline is compositional

The single-cell validation is the key advance over our initial characterization. By computing pi within individual cell types across tissues, we show that the tissue-identity signal is stable with age at the single-cell level (Delta approximately 0 when sample sizes are balanced). The bulk decline is therefore entirely attributable to shifts in cell-type composition -- increased immune infiltration, loss of tissue-resident cell populations, or expansion of specific cell subsets.

This result has important implications. It means that each cell type faithfully maintains its tissue-specific transcriptomic program throughout life. The aging phenotype observed in bulk tissue is an emergent property of changing cellular demographics, not of degradation within cells. Therapeutic strategies aimed at restoring tissue identity should therefore focus on maintaining proper cell-type composition rather than on rescuing cell-intrinsic transcriptomic programs. This conclusion aligns with recent work showing that age-related changes in tissue function often track with cell-type proportion shifts rather than with within-cell regulatory changes.

We note an important technical caveat: old Smart-seq2 cells detect fewer genes per cell (e.g., macrophages: 2,824 versus 2,002 genes/cell), which could confound single-cell comparisons. Our balanced analysis addresses sample size confounds but not this gene detection confound. Future validation with platforms less susceptible to this bias (e.g., 10x Chromium with UMI correction) would strengthen the conclusion.

### Chromatin machinery at the leading edge

The accelerated erosion of chromatin remodeling genes (2.5-fold faster than expression-matched controls, p = 0.009) connects our findings to the epigenetic drift hypothesis of aging. DNMT3A/B, TET2, EZH2, and SIRT1/6/7 maintain tissue-specific chromatin landscapes. When these genes lose tissue-specific expression, the chromatin landscapes they maintain will gradually drift, leading to secondary effects on transcription factor activity and downstream gene expression.

The expression-matched control strategy is critical here: the chromatin erosion signal cannot be explained by these genes simply being highly or lowly expressed. It reflects their function. The cascade hierarchy (chromatin > TF > targets) is consistent with a top-down model in which epigenetic erosion precedes transcriptional changes.

However, given the single-cell finding that bulk pi decline is entirely compositional, the chromatin signal is best understood as reflecting which gene categories are most sensitive to cell-type proportion changes, not cell-intrinsic epigenetic erosion. Chromatin remodeling genes have strongly cell-type-dependent expression; as cell populations shift with age, these genes are disproportionately affected in bulk measurements. This reframes the chromatin cascade from an intracellular mechanism to a biomarker of compositional sensitivity. Single-cell-level analysis of chromatin gene variance decomposition would be needed to test for any residual cell-intrinsic component.

### Caloric restriction as noise filter

The mechanistic decomposition of CR's effect on pi -- reducing residual variance (noise) rather than restoring tissue-specific variance (structure) -- provides a concrete framework for understanding dietary restriction at the transcriptomic level. CR does not reverse aging; it improves the signal-to-noise ratio.

This predicts that CR-mimetic compounds should similarly reduce transcriptomic noise without restoring tissue-specific programs. Conversely, partial epigenetic reprogramming approaches (e.g., Yamanaka factor induction) should operate on the structural component, potentially restoring V_tissue directly. These predictions are testable with existing datasets and experimental systems.

The 86% rescue achieved by CR despite operating only on noise also quantifies how much of the aging phenotype is noise-driven versus structure-driven: at least in the rat, most of the measurable pi decline can be compensated by noise reduction alone. Importantly, this 86% rescue refers to the pi ratio (V_tissue / V_total), not to V_tissue itself, which remains at the old_AL level under CR. The structural erosion is not reversed; only the signal-to-noise ratio is improved.

### Cross-species comparison and lifespan scaling

The observation that erosion rates decrease with increasing lifespan across rat, macaque, and human is suggestive of an inverse scaling relationship. If confirmed with additional species, this would parallel other biological scaling laws such as the heartbeat invariant. However, we emphasize that three species is insufficient to establish a power law, the mouse trajectory was non-monotonic (precluding reliable erosion rate estimation), and the number of tissues differs across datasets. The consistent finding across all species is that tissue identity is the dominant variance component -- the quantitative scaling of its erosion rate with lifespan remains a hypothesis for future testing with standardized multi-tissue datasets across a broader range of mammals.

### Cancer as the extreme case

The near-complete loss of tissue identity in tumors (pi = 0.016) places cancer and aging on a shared axis but at vastly different positions. Normal aging operates at pi approximately 0.73 -- well within the regime where tissue identity is the dominant organizing principle. Cancer, at pi approximately 0.02, has effectively exited this regime entirely. The 45-fold difference quantifies the qualitative distinction between the slow noise accumulation of normal aging and the catastrophic dedifferentiation of malignant transformation.

### Limitations

Several limitations should be noted. First, our single-cell validation uses mouse data (TMS FACS) to interpret human bulk findings (GTEx). While the conservation of pi as a near-invariant across species supports this cross-species inference, direct human single-cell validation with sufficient tissue coverage is desirable. Second, the Smart-seq2 platform used in TMS has known gene detection differences between young and old cells, introducing a confound that our sample-balancing approach does not fully address. Third, the cross-species comparison is confounded by differing tissue numbers (6 to 17), which affects absolute pi values; only within-species trajectories should be interpreted quantitatively. Fourth, GTEx donors are deceased individuals, potentially introducing perimortem artifacts. Fifth, the 20s-to-40s concentrated decline could reflect developmental completion rather than aging onset. Sixth, our functional gene categories (chromatin, TF, target) are manually curated and small (n = 18-30), limiting statistical power for the cascade analysis. Seventh, the sex-stratified finding (female pi declines approximately 2-fold faster) is based on modest sample sizes within age-sex strata and the male trajectory is non-monotonic, so this should be considered preliminary. Eighth, the Calico rat CR analysis has only 2 biological replicates per tissue per condition (54 samples / 9 tissues / 3 conditions), which limits the reliability of per-condition ANOVA estimates. Ninth, we report bootstrap confidence intervals for per-decade pi estimates but these are based on 100 resamples; larger bootstrap samples would provide more precise uncertainty bounds.

### Conclusion

Tissue identity is the dominant organizing principle of the mammalian transcriptome and is remarkably robust to aging. Its slow erosion (approximately 0.08% per year in humans) is entirely composition-driven at the single-cell level: individual cell types faithfully maintain their tissue-specific programs throughout life. When erosion is detected in bulk tissue, it reflects shifting cellular demographics rather than cell-intrinsic degradation. Among gene classes, chromatin remodeling machinery shows the fastest erosion, consistent with epigenetic drift as an upstream mechanism. Caloric restriction compensates for age-related pi loss through noise reduction rather than structure repair. Across multiple mammalian species, tissue identity dominance is conserved, and erosion rates appear to decrease with increasing lifespan. These findings establish pi_tissue as a simple, interpretable metric for quantifying tissue organizational integrity, resolve the bulk-versus-cell-intrinsic question in favor of compositional drift, and identify noise reduction and composition maintenance as distinct therapeutic axes for preserving tissue identity during aging.

---

## Acknowledgments

The author thanks the GTEx Consortium, Tabula Muris Consortium, and Calico Life Sciences for making their datasets publicly available.

## Author Contributions

T.S. conceived the study, designed the analyses, performed all computational work, and wrote the manuscript.

## Competing Interests

The author declares no competing financial or non-financial interests.

## Funding

This work received no external funding.

## Ethics Statement

This study uses exclusively publicly available, de-identified datasets that were previously approved for research by institutional review boards at their originating institutions (GTEx: dbGaP phs000424; TMS: Schaum et al. 2020; Calico rat atlas: Ma et al. 2020; TCGA: Genomic Data Commons). No additional ethical review was required for this secondary analysis.

---

## Methods

### GTEx data processing

We used GTEx v8 gene-level TPM values (GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz) and accompanying sample and subject annotations. Six tissues with the most shared donors were selected: Muscle - Skeletal, Whole Blood, Skin - Sun Exposed (Lower leg), Adipose - Subcutaneous, Artery - Tibial, and Thyroid. We identified 263 donors with samples available in all six tissues. Expression values were log2(TPM + 1) transformed. Genes with median TPM below 0.5 across all samples were excluded, yielding approximately 18,000 genes for analysis. Age bins were defined as 20-39, 40-49, 50-59, and 60-79 years (matching GTEx annotation bins).

### Three-level ANOVA variance decomposition

For each gene, total variance was decomposed as V_total = V_tissue + V_donor + V_residual, where V_tissue = n_donors times the sum of squared deviations of tissue means from the grand mean, V_donor = n_tissues times the sum of squared deviations of donor means from the grand mean, and V_residual = V_total minus V_tissue minus V_donor. pi_tissue was computed as V_tissue / V_total for each gene. We report the median pi across all expressed genes as the summary statistic. Per-decade analyses used the same decomposition restricted to donors within each age bin.

### variancePartition REML validation

We used the variancePartition R package (v1.40.1) with formula ~(1|tissue) + (1|donor), fitting REML linear mixed models on 2,000 randomly sampled genes per age bin. Variance fractions were extracted as the proportion attributable to each random effect. For the overall model, we used ~(1|tissue) + (1|donor) + age_mid.

### PERMANOVA validation

We used the adonis2() function from the vegan R package with Euclidean distances computed on the top 500 most variable genes, model formula d ~ tissue + age_bin + sex, with 999 permutations. R-squared values for each term were extracted from the PERMANOVA table.

### Permutation null

To establish a baseline, we shuffled tissue labels 100 times (breaking the tissue-sample mapping while preserving all other structure) and recomputed pi_tissue. The null distribution (pi_null = 0.003, SD = 0.001) was compared to the observed pi to compute fold enrichment.

### Tissue specificity index (tau)

For each gene, tau was computed as the sum of (1 minus expr_i / max_expr) divided by (n_tissues minus 1), where expr_i is the mean expression in tissue i. Genes with tau above 0.8 were classified as tissue-specific (n = 395); genes with tau below 0.3 as ubiquitous (n = 7,550).

### Single-cell validation (Tabula Muris Senis)

We used the TMS FACS dataset (Schaum et al., Nature 2020): 110,824 cells, 23 tissues, Smart-seq2 platform, ages 3 months (young) and 24 months (old). Cell type annotations from the original study were used. Seven cell types present in 4 or more tissues were selected: macrophage (12 tissues), endothelial (11), B cell (11), T cell (13), NK cell (8), MSC adipose (4), myeloid (4). For each cell type, pseudobulk expression was computed per tissue, age, and mouse. pi_tissue was then computed using the standard ANOVA framework on these pseudobulk profiles.

To address the sample size confound, we balanced the number of samples between young and old groups by random subsampling to the smaller group size. This was repeated 100 times to obtain stable estimates. Gene detection rates per cell were computed as the number of genes with non-zero expression; differences between age groups were tested by Wilcoxon rank-sum test.

### Expression-matched functional category analysis

Genes were assigned to four functional categories: chromatin remodeling (DNMT1, DNMT3A, DNMT3B, TET1, TET2, TET3, HDAC1-6, EZH2, SIRT1, SIRT6, SIRT7, SMARCA4, ARID1A; n = 30), transcription factors (n = 18, curated from Lambert et al. 2018), downstream targets of these regulators (n = 50), and housekeeping genes (n = 100, from Eisenberg and Levanon 2013). For each gene in a focal category, 10 expression-level-matched genes were selected from the remaining genome based on mean log2(TPM + 1) (nearest-neighbor matching). Delta_pi distributions were compared between focal categories and their matched controls using two-sided Mann-Whitney U tests.

### Rat caloric restriction analysis

We used the Calico rat aging atlas (Ma et al., Cell 2020): 218,971 cells, 54 samples across 9 tissues. Conditions: young (5 months, n = 18 samples), old ad libitum (27 months, n = 18), old caloric restriction (27 months, n = 18). After standard normalization (scanpy normalize_total with target_sum = 10,000, followed by log1p), we aggregated to pseudobulk per sample. pi_tissue was computed using the ANOVA framework. Absolute variance components (V_tissue, V_donor, V_residual) were compared across conditions to determine whether CR restored structure or reduced noise.

### Cross-species analysis

Mouse: Tabula Muris Senis bulk RNA-seq (GSE132040), 17 tissues, multiple ages from 1 to 27 months. Macaque: Li and Kong 2025 (Figshare 26963386), 32 tissues (top 10 selected), 7 ages from 3 to 27 years, bulk RNA-seq. Rat: as above. Naked mole-rat: 10 tissues, 24 animals, newborn versus breeder (published datasets). Guinea pig: 11 tissues, newborn versus breeder. Human: GTEx as above.

All datasets were processed with log2(CPM + 1) or log2(TPM + 1) as appropriate and the same ANOVA framework. Within-species erosion rates (d_pi/dt) were computed by linear regression on adult ages only (excluding juvenile and developmental timepoints). The cross-species scaling law was fit as log|d_pi/dt| = alpha times log(L) + beta, where L is maximum lifespan. Confidence intervals for alpha were obtained by bootstrap resampling of the four species data points.

### TCGA analysis

We used 681 paired tumor-normal samples from TCGA across 14 cancer types, with 14,672 genes passing expression filters. pi_tissue was computed separately for tumor and matched normal samples using the same ANOVA framework, treating cancer type as the tissue label.

### Sex-stratified analysis

GTEx donors were split by sex. pi_tissue trajectories were computed independently for females and males across age decades. Differences in Delta_pi between sexes were assessed descriptively given limited sample sizes within age-by-sex strata.

### Statistical tests

Bootstrap confidence intervals were computed from 100 resamples of donors within each age bin. Permutation null used 100 shuffles of tissue labels. All p-values are two-sided. No multiple testing correction was applied to the seven primary validation tests, as each addresses a distinct threat to validity (batch, permutation, gene subsampling, tissue subsampling, platform concordance, sex stratification, single-tissue sensitivity).

### Code and data availability

All analysis code is available at https://github.com/mool32/pitissue. GTEx data from dbGaP (phs000424.v8). Tabula Muris Senis from GEO (GSE132040) and Figshare. Calico rat atlas from GEO (GSE137869; Ma et al. 2020). Macaque data from Figshare (26963386). TCGA data from the Genomic Data Commons.

---

## Figure Legends

**Figure 1. Tissue identity is the dominant organizational mode of the human transcriptome and is near-invariant with age.** (A) Stacked bar chart showing pi_tissue (green), pi_donor (blue), and pi_residual (gray) for four age decades (GTEx v8, 263 donors, 6 tissues, 18,000 genes). pi_tissue = 0.764 (ages 20-39) declining to 0.733 (ages 60-79). (B) Line plot of pi_tissue, pi_donor, and pi_residual across decades. The decline is concentrated in the 20-39 to 40-49 transition, followed by a plateau. (C) Permutation null: pi_null = 0.003 from 100 tissue-label shuffles versus observed pi = 0.73, a 243-fold enrichment. (D) variancePartition REML confirmation: pi = 0.789 to 0.758, Delta = -0.031, identical to ANOVA. Inset: PERMANOVA R-squared = 0.858.

**Figure 2. Two-component decomposition and single-cell validation.** (A) Delta_pi for tissue-specific genes (tau > 0.8, n = 395; Delta = -0.011) and ubiquitous genes (tau < 0.3, n = 7,550; Delta = -0.041), showing differential sensitivity to aging. (B) Single-cell validation using TMS FACS: pi_tissue computed within each of 7 cell types across tissues. With balanced sample sizes, mean Delta = -0.01 (p = 0.69), demonstrating cell-intrinsic stability. Gray bars show unbalanced analysis (7/7 increase, p = 0.016, artifactual). (C) Schematic: bulk pi decline is entirely composition-driven; individual cell types maintain tissue identity faithfully.

**Figure 3. Chromatin remodeling machinery erodes tissue specificity fastest, independent of expression level.** (A) Bar chart showing Delta_pi for four gene categories (chromatin, TF, target, housekeeping) alongside their expression-matched controls. Chromatin: focal Delta_pi = -0.057, controls = -0.023 (2.5-fold faster, p = 0.009). TF: -0.039 vs -0.024 (p = 0.19). Target: -0.012 vs -0.030 (protected, p = 0.09). Housekeeping: matches controls (p = 0.96). (B) Expression-bin analysis confirming that chromatin genes erode faster than other genes at the same expression level across all expression quintiles.

**Figure 4. Caloric restriction restores pi through noise reduction, not structure repair.** (A) pi_tissue in the Calico rat aging atlas: young = 0.893, old_AL = 0.842, old_CR = 0.886 (86% rescue). (B) Absolute variance decomposition: aging decreases V_tissue (structure erodes) while V_residual is stable. CR leaves V_tissue unchanged but decreases V_residual (noise removed). CR operates as a transcriptomic noise filter.

**Figure 5. Cross-species scaling and tissue-specific noise dynamics.** (A) Per-tissue noise accumulation in GTEx: blood variance increases by +0.079 (3-fold faster than muscle at +0.028). Clonal hematopoiesis driver gene (DNMT3A, TET2, ASXL1) variance increases 1.7-2.4-fold specifically in blood. (B) Cross-species scaling: erosion rate |d_pi/dt| versus maximum lifespan for four mammals (mouse, rat, macaque, human) on log-log axes. Power law fit: alpha = -1.12 plus or minus 0.18, R-squared = 0.951, p = 0.025. Dashed line: alpha = -1 (inverse proportionality). Inset: pi trajectories for all six species including naked mole-rat and guinea pig.

---

## Supplementary Figure Legends

**Figure S1. Validation battery for pi_tissue.** (A) Permutation null distribution (100 shuffles, mean = 0.003, SD = 0.001) with observed pi marked. (B) Number-of-tissues sensitivity: Delta_pi trajectory stable at -0.031 to -0.036 for 4, 5, or 6 tissues. (C) Gene subsampling convergence: pi stabilizes by approximately 3,000 genes. (D) Leave-one-tissue-out: blood removal has largest effect (Delta = 0.07). (E) Bootstrap confidence intervals per decade (100 resamples).

**Figure S2. Batch effects.** pi_batch = 0.257 of total variance, but age is uniformly distributed across batches (Kruskal-Wallis p = 0.80). RIN shows minimal age correlation (rho = -0.064).

**Figure S3. Sex-stratified pi trajectories.** Female: pi = 0.776 to 0.725 (Delta = -0.051). Male: pi = 0.768 to 0.743 (Delta = -0.025). Female pi declines approximately 2-fold faster than male.

**Figure S4. Single-cell gene detection confound.** Number of genes detected per cell in TMS FACS, stratified by age. Old cells detect fewer genes across all cell types (example: macrophage, 2,824 to 2,002 genes/cell). This technical confound reinforces the need for sample-balanced analyses.

**Figure S5. Cancer abolishes tissue identity.** TCGA paired tumor-normal analysis (681 pairs, 14,672 genes). Normal tissue pi approximately 0.73; tumor pi = 0.016 (45-fold reduction).

**Figure S6. Mouse pi trajectory is non-monotonic.** pi_tissue computed from TMS bulk RNA-seq across 17 tissues at ages 1-27 months. The trajectory fluctuates between 0.44 and 0.61 without a consistent decline, precluding reliable erosion rate estimation. Mean pi = 0.505 (dashed line). Shaded region indicates approximate uncertainty.

---

## Supplementary Tables

**Table S1.** Complete validation results (7/7 passed): permutation null, gene subsampling, tissue subsampling, variancePartition concordance, PERMANOVA concordance, batch independence, RIN independence.

**Table S2.** Per-gene pi_tissue values for all 18,000 genes in young and old age bins.

**Table S3.** Gene category assignments (chromatin, TF, target, housekeeping) with expression-matched control gene lists and individual Delta_pi values.

**Table S4.** Cross-species pi_tissue values, sample sizes, tissue counts, and erosion rates with bootstrap confidence intervals.

**Table S5.** TMS FACS single-cell validation: per-cell-type pi values, sample sizes, balanced versus unbalanced results, and gene detection statistics.

**Table S6.** Sex-stratified pi values per decade with sample sizes.

**Table S7.** Permutation test summary statistics.
