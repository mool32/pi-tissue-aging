# Supplementary Tables

---

## Table 1: Cross-Species Scaling of Tissue Identity Erosion

pi_tissue values and erosion rates across mammalian species. The erosion rate (|dpi/dt|) scales inversely with organismal lifespan (L) according to a power law: |dpi/dt| proportional to L^alpha where alpha = -1.12 +/- 0.18 (R^2 = 0.951, p = 0.025).

| Species | Lifespan | Young Age | Young pi_tissue | Old Age | Old pi_tissue | Delta_pi | Time Span | Erosion Rate |dpi/dt| | n_tissues | n_samples |
|---|---|---|---|---|---|---|---|---|---|---|
| Mouse (C57BL/6) | ~2.5 yr | 1 mo | 0.440 | 27 mo | 0.488 | +0.048^a^ | 26 mo | non-monotonic^a^ | 17 | 55 |
| Rat (Fischer 344, AL) | ~2.5 yr | 5 mo | 0.893 | 27 mo | 0.842 | -0.051 | 22 mo | -0.0278 | 9 | 36 |
| Macaque | ~25 yr | Young | 0.715 | Oldest | 0.620 | -0.095 | ~24 yr | -0.00395 | 10 | 40 |
| Human (GTEx) | ~80 yr | 20-39 | 0.764 | 60-79 | 0.733 | -0.031 | 40 yr | -0.000775 | 6 | 1,033 |
| Naked mole-rat | ~30 yr | Newborn | 0.846 | Breeder | 0.820 | -0.026 | ~3 yr | -0.0087 | 10 | 24 |
| Guinea pig | ~5 yr | Newborn | 0.871 | Breeder | 0.896 | +0.025^b^ | ~1 yr | +0.025 | 11 | -- |

### Power Law Scaling

| Parameter | Value | 95% CI | Note |
|---|---|---|---|
| Exponent (alpha) | -1.12 | -1.30 to -0.94 | Consistent with inverse proportionality (alpha = -1) |
| R^2 | 0.951 | -- | Fit quality across four species with complete aging data |
| p-value | 0.025 | -- | Significant (p < 0.05) |
| Relationship | |dpi/dt| = 0.295 x L^-1.12 | -- | Power-law form |

**Notes:**
- Lifespan: approximate maximum lifespan in laboratory conditions.
- ^a^ Mouse trajectory is non-monotonic (pi fluctuates 0.44-0.61 across 1-27 months; see Fig. S6), precluding reliable erosion rate estimation. Endpoint difference is positive but not representative of overall trend.
- ^b^ Guinea pig shows increase with maturation, consistent with developmental programs rather than aging.
- Cross-species comparison caveats: pi_tissue is sensitive to tissue number (mouse: 17 tissues vs human: 6 tissues). Different platforms used (bulk RNA-seq for human/rat vs. pseudobulk Smart-seq2 for mouse/macaque). Scaling law should be regarded as suggestive; within-species erosion trajectories are internally valid.

---

## Table S1: Validation Results

Complete validation results for pi_tissue across seven independent validation tests.

| Validation Test | Result | Key Finding |
|---|---|---|
| Permutation null (tissue shuffling) | PASSED | pi_observed = 0.729, pi_null = 0.003; 243-fold above chance (p < 0.001) |
| Permutation null (donor shuffling) | PASSED | Donor-shuffled null matches real value, confirming donor structure preserved |
| Gene subsampling robustness | PASSED | pi_tissue stable across random gene subsets; converges by ~3,000 genes |
| Tissue subsampling robustness | PASSED | pi_tissue conserved across 4-, 5-, and 6-tissue panels (Delta_pi = -0.031 to -0.036) |
| variancePartition concordance | PASSED | REML-based pi = 0.789 to 0.758 (Delta = -0.031, identical to ANOVA) |
| PERMANOVA concordance | PASSED | R^2 = 0.858, p = 0.001 |
| Batch independence | PASSED | Batch-age correlation rho = 0.187; Kruskal-Wallis p = 0.80 |
| RIN independence | PASSED | RIN-age correlation rho = -0.064 |

**Notes:** pi_tissue values are overall medians across all ages (0.729) and per-decade values (0.764 young, 0.733 old). Batch effects account for 25.7% of total variance but are not confounded with age.

---

## Table S2: Per-Gene pi_tissue Values

Per-gene pi_tissue values in the youngest age bin (20-39), age-trajectory slopes, and significance for 5,000 expressed genes.

| Gene | pi_tissue (Young) | Slope (rho) | Slope p-value |
|---|---|---|---|
| WASH7P | 0.488 | 1.000 | 0.000 |
| RP11-34P13.15 | 0.893 | -1.000 | 0.000 |
| SAMD11 | 0.703 | 1.000 | 0.000 |
| PLEKHN1 | 0.939 | 0.800 | 0.200 |
| ... | ... | ... | ... |

**Notes:** Complete table (5,000 genes) available as Supplementary Data File (per_gene_stability.csv). Slope (rho): Spearman correlation between gene's pi_tissue and age decade. pi_tissue ranges 0-1, where 1 = perfectly tissue-locked expression.

---

## Table S3: Gene Category Analysis

Gene category assignments with pi_tissue values per age decade and expression-matched control comparisons.

| Gene Category | 20-39 | 40-49 | 50-59 | 60-79 | Delta_pi (young-old) | n_genes |
|---|---|---|---|---|---|---|
| Chromatin remodeling | 0.732 | 0.684 | 0.643 | 0.652 | -0.080 | 31 |
| Transcription factors | 0.725 | 0.673 | 0.664 | 0.680 | -0.045 | 32 |
| TF target genes | 0.752 | 0.672 | 0.696 | 0.689 | -0.063 | 29 |
| Housekeeping genes | 0.776 | 0.750 | 0.743 | 0.769 | -0.006 | 15 |

### Expression-Matched Control Comparison

| Gene Category | Focal Delta_pi | Control Delta_pi | Fold difference | p-value (Mann-Whitney U) |
|---|---|---|---|---|
| Chromatin remodeling | -0.057 | -0.023 | 2.5x | 0.009 |
| Transcription factors | -0.039 | -0.024 | 1.6x | 0.19 |
| TF target genes | -0.012 | -0.030 | 0.4x (protected) | 0.09 |
| Housekeeping genes | -0.023 | -0.023 | 1.0x | 0.96 |

**Notes:** Expression-matched controls: 10 genes per focal gene, matched by mean log2(TPM + 1). Chromatin remodeling genes include DNMT1, DNMT3A/B, TET1-3, HDAC1-6, EZH2, SIRT1/6/7, SMARCA4, ARID1A. Transcription factors curated from Lambert et al. 2018. Housekeeping genes from Eisenberg and Levanon 2013.

---

## Table S4: Cross-Species pi_tissue Comparison

Detailed pi_tissue values and sample statistics across species.

| Species | Age | pi_tissue | Q1-Q3 | n_genes | n_samples | Erosion rate (Dpi/yr) |
|---|---|---|---|---|---|---|
| Human (GTEx) | 20-39 | 0.764 | 0.61-0.92 | 18,000 | 772 | -0.000775 |
| Human (GTEx) | 60-79 | 0.733 | 0.59-0.89 | 18,000 | 261 | -- |
| Mouse (Bulk RNA) | 1 month | 0.440 | -- | -- | -- | -- |
| Mouse (Bulk RNA) | 27 months | 0.488 | -- | -- | -- | -- |
| Rat (AL diet) | 5 months | 0.893 | 0.78-0.95 | 20,884 | 18 | -0.028 |
| Rat (AL diet) | 27 months | 0.842 | 0.70-0.93 | 20,884 | 18 | -- |
| Rat (CR diet) | 5 months | 0.886 | 0.75-0.95 | 20,902 | 18 | -- |
| Rat (CR diet) | 27 months | 0.886 | -- | 20,902 | 18 | -- |

**Notes:** pi_tissue: median across all genes (0-1 scale). Q1-Q3: 25th and 75th percentiles of gene-level distributions.

---

## Table S5: Single-Cell Validation (TMS FACS)

Per-cell-type pi_tissue values from Tabula Muris Senis FACS dataset in young (3 months) and old (24 months) mice.

| Cell Type | Age | pi_tissue | n_tissues | Delta_pi |
|---|---|---|---|---|
| Macrophage | 3 months | 0.442 | 9 | +0.010 |
| Macrophage | 24 months | 0.452 | 9 | |
| Endothelial cell | 3 months | 0.344 | 11 | +0.035 |
| Endothelial cell | 24 months | 0.379 | 10 | |
| B cell | 3 months | 0.253 | 10 | +0.116 |
| B cell | 24 months | 0.369 | 11 | |
| T cell | 3 months | 0.332 | 9 | +0.052 |
| T cell | 24 months | 0.384 | 12 | |
| NK cell | 3 months | 0.331 | 7 | +0.072 |
| NK cell | 24 months | 0.404 | 4 | |
| MSC adipose | 3 months | 0.163 | 4 | +0.148 |
| MSC adipose | 24 months | 0.312 | 4 | |
| Myeloid cell | 3 months | 0.231 | 4 | +0.038 |
| Myeloid cell | 24 months | 0.269 | 4 | |

**Summary:** Without sample size balancing, 7/7 cell types show pi increase (binomial p = 0.016). With balanced sample sizes (100 random subsamplings), mean Delta = -0.01, p = 0.69 (non-significant). The unbalanced increase is an artifact of unequal sample sizes between age groups.

### Cross-Level Validation

| Age | pi_tissue | pi_celltype | n_cells | n_tissues |
|---|---|---|---|---|
| 3 months | 0.031 | 0.021 | 44,518 | 19 |
| 18 months | 0.024 | 0.023 | 34,027 | 20 |
| 24 months | 0.021 | 0.018 | 31,551 | 18 |

**Notes:** Single-cell pi_tissue values (0.02-0.03) are much lower than bulk (0.73) because cell-type heterogeneity is resolved. Old Smart-seq2 cells detect fewer genes per cell (e.g., macrophages: 2,824 vs 2,002 genes/cell), a technical confound that reinforces the need for sample-balanced analyses.

---

## Table S6: Sex-Stratified Analysis

pi_tissue values stratified by biological sex across age decades.

| Age Decade | Female pi_tissue | Male pi_tissue | Difference (F-M) |
|---|---|---|---|
| 20-39 | 0.776 | 0.768 | +0.008 |
| 40-49 | 0.748 | 0.736 | +0.012 |
| 50-59 | 0.737 | 0.738 | -0.002 |
| 60-79 | 0.725 | 0.744 | -0.018 |

**Notes:** Female Delta_pi = -0.051; Male Delta_pi = -0.025. Females show approximately 2-fold faster decline, potentially reflecting hormonal transitions at menopause. Absolute sex differences are small (< 0.02 at all decades).

---

## Table S7: Permutation Test Summary

Summary statistics from permutation-based validation.

| Test | Statistic | Value | p-value |
|---|---|---|---|
| Tissue shuffling null | Observed pi_tissue | 0.729 | < 0.001 |
| Tissue shuffling null | Null mean +/- SD | 0.003 +/- 0.001 | -- |
| Tissue shuffling null | Fold enrichment | 243x | -- |
| Donor shuffling null | Donor-shuffled pi | 0.729 | -- |
| Batch independence | Batch-age correlation (rho) | 0.187 | 0.80 (Kruskal-Wallis) |
| RIN independence | RIN-age correlation (rho) | -0.064 | n.s. |

**Notes:** Expected by chance (random tissue labels for 6 tissues): 1/6 = 0.167. Observed pi_tissue exceeds this 4.4-fold and exceeds permutation null 243-fold.
