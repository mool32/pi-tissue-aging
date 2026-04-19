#!/usr/bin/env Rscript
# Test 1.1: variancePartition (REML mixed model) — field standard
# Test 1.2: PERMANOVA (adonis2) — distribution-free validation
#
# Compare with our ANOVA π_tissue ≈ 0.73
# If variancePartition gives ~0.47 (like Melé) → method-dependent
# If ~0.73 → method-robust

suppressMessages({
  library(variancePartition)
  library(lme4)
  library(vegan)
})

cat(strrep("=", 70), "\n")
cat("variancePartition + PERMANOVA validation\n")
cat(strrep("=", 70), "\n\n")

# ── Load data ────────────────────────────────────────────
cat("Loading GTEx data (pre-processed by Python)...\n")

# We need to prepare data from Python first — save a subset as CSV
# For speed, use 5000 genes × 263 donors × 6 tissues = 1578 samples
# Python will prepare this

data_dir <- "/Users/teo/Desktop/research/coupling_atlas/results/step18_vp"
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

# Check if Python has prepared the data
expr_file <- file.path(data_dir, "expr_matrix.csv.gz")
meta_file <- file.path(data_dir, "metadata.csv")

if (!file.exists(expr_file)) {
  cat("Expression matrix not found. Run Python prep script first.\n")
  cat("Expected:", expr_file, "\n")
  quit(status = 1)
}

cat("Loading expression matrix...\n")
expr <- read.csv(gzfile(expr_file), check.names = FALSE)
# Handle duplicate gene names
rownames(expr) <- make.unique(as.character(expr[,1]))
expr <- expr[,-1]
meta <- read.csv(meta_file, row.names = 1, stringsAsFactors = TRUE)

cat(sprintf("Expression: %d genes × %d samples\n", nrow(expr), ncol(expr)))
cat(sprintf("Metadata: %d samples, columns: %s\n", nrow(meta), paste(colnames(meta), collapse=", ")))
cat(sprintf("Tissues: %s\n", paste(levels(meta$tissue), collapse=", ")))
cat(sprintf("Age bins: %s\n", paste(sort(unique(meta$age_bin)), collapse=", ")))

# ══════════════════════════════════════════════════════════
# TEST 1.1: variancePartition
# ══════════════════════════════════════════════════════════
cat("\n", strrep("=", 60), "\n")
cat("TEST 1.1: variancePartition (REML mixed model)\n")
cat(strrep("=", 60), "\n\n")

# Model: ~ (1|tissue) + (1|donor) + age_mid
# tissue and donor as random effects
meta$tissue <- factor(meta$tissue)
meta$donor <- factor(meta$donor)

# Subset for speed — 2000 genes
set.seed(42)
if (nrow(expr) > 2000) {
  gene_idx <- sample(nrow(expr), 2000)
  expr_sub <- expr[gene_idx, ]
} else {
  expr_sub <- expr
}

cat(sprintf("Running variancePartition on %d genes...\n", nrow(expr_sub)))

# Formula
form <- ~ (1|tissue) + (1|donor) + age_mid

t0 <- proc.time()
varPart <- fitExtractVarPartModel(expr_sub, form, meta)
elapsed <- (proc.time() - t0)[3]
cat(sprintf("Done in %.0f seconds\n", elapsed))

# Summary
cat("\nMedian variance explained:\n")
medians <- apply(varPart, 2, median)
for (i in seq_along(medians)) {
  cat(sprintf("  %s: %.4f (%.1f%%)\n", names(medians)[i], medians[i], medians[i]*100))
}

# Per age bin
cat("\nvariancePartition per age bin:\n")
for (ab in sort(unique(meta$age_bin))) {
  mask <- meta$age_bin == ab
  samples_ab <- rownames(meta)[mask]
  samples_ab <- samples_ab[samples_ab %in% colnames(expr_sub)]

  if (length(samples_ab) < 50) {
    cat(sprintf("  %s: n=%d — skipped (too few)\n", ab, length(samples_ab)))
    next
  }

  expr_ab <- expr_sub[, samples_ab, drop=FALSE]
  meta_ab <- meta[samples_ab, , drop=FALSE]
  meta_ab$tissue <- droplevels(meta_ab$tissue)
  meta_ab$donor <- droplevels(meta_ab$donor)

  form_ab <- ~ (1|tissue) + (1|donor)

  tryCatch({
    vp_ab <- fitExtractVarPartModel(expr_ab, form_ab, meta_ab)
    med_tissue <- median(vp_ab$tissue)
    med_donor <- median(vp_ab$donor)
    med_resid <- median(vp_ab$Residuals)
    cat(sprintf("  %s (n=%d): π_tissue=%.4f, π_donor=%.4f, π_resid=%.4f\n",
                ab, length(samples_ab), med_tissue, med_donor, med_resid))
  }, error = function(e) {
    cat(sprintf("  %s: ERROR — %s\n", ab, e$message))
  })
}

# Save variancePartition results
write.csv(as.data.frame(varPart), file.path(data_dir, "variancePartition_results.csv"))

# ══════════════════════════════════════════════════════════
# TEST 1.2: PERMANOVA
# ══════════════════════════════════════════════════════════
cat("\n", strrep("=", 60), "\n")
cat("TEST 1.2: PERMANOVA (adonis2)\n")
cat(strrep("=", 60), "\n\n")

# Use top 500 most variable genes for PERMANOVA (speed)
gene_vars <- apply(expr, 1, var)
top500 <- names(sort(gene_vars, decreasing=TRUE))[1:500]
expr_perm <- t(expr[top500, ])  # samples × genes

cat(sprintf("PERMANOVA on %d samples × %d genes\n", nrow(expr_perm), ncol(expr_perm)))

# Distance matrix
d <- vegdist(expr_perm, method = "euclidean")

# PERMANOVA model
perm_result <- adonis2(d ~ tissue + age_bin + sex, data = meta, permutations = 999)
cat("\nPERMANOVA results:\n")
print(perm_result)

# Extract R² values
cat("\nR² (variance explained):\n")
for (term in rownames(perm_result)) {
  if (term != "Total") {
    r2 <- perm_result[term, "R2"]
    p <- perm_result[term, "Pr(>F)"]
    cat(sprintf("  %s: R²=%.4f (%.1f%%), p=%.4f\n", term, r2, r2*100, p))
  }
}

# Save PERMANOVA
write.csv(as.data.frame(perm_result), file.path(data_dir, "permanova_results.csv"))

cat("\n", strrep("=", 60), "\n")
cat("SUMMARY\n")
cat(strrep("=", 60), "\n")
cat(sprintf("ANOVA π_tissue:              ~0.73\n"))
cat(sprintf("variancePartition π_tissue:  %.4f\n", median(varPart$tissue)))
cat(sprintf("PERMANOVA R²_tissue:         %.4f\n", perm_result["tissue", "R2"]))
cat(sprintf("\nAll results saved to: %s\n", data_dir))
