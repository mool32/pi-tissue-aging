"""
TCGA paired tumor/normal coupling analysis.

Download TCGA data via GDC API for cancer types with most paired samples.
Compute TF→target coupling in normal vs tumor tissue.
Same panel of 30+ pairs as GTEx.
"""

import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data" / "tcga"
RESULTS_DIR = BASE / "results" / "step3_tcga"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg):
    print(msg, flush=True)


def spearman_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan, 1.0
    return stats.spearmanr(x, y)


# Standard TF→target panel (HUGO symbols)
PAIRS = [
    # Production / structural
    ("SMAD3", "COL1A1", "SMAD3→COL1A1", "production"),
    ("SMAD3", "COL3A1", "SMAD3→COL3A1", "production"),
    ("SMAD3", "FN1", "SMAD3→FN1", "production"),
    ("SMAD3", "SERPINE1", "SMAD3→SERPINE1", "production"),
    ("ESR1", "COL1A1", "ESR1→COL1A1", "production"),
    ("ESR1", "ELN", "ESR1→ELN", "production"),
    ("AR", "COL1A1", "AR→COL1A1", "production"),
    ("SOX9", "COL2A1", "SOX9→COL2A1", "production"),
    ("PPARG", "FABP4", "PPARG→FABP4", "production"),
    ("PPARG", "ADIPOQ", "PPARG→ADIPOQ", "production"),
    ("RUNX2", "SPP1", "RUNX2→SPP1", "production"),
    ("HNF4A", "ALB", "HNF4A→ALB", "production"),
    ("FOXO1", "PCK1", "FOXO1→PCK1", "production"),
    # Detection / signaling
    ("RELA", "ICAM1", "RELA→ICAM1", "detection"),
    ("RELA", "NFKBIA", "RELA→NFKBIA", "detection"),
    ("RELA", "CCL2", "RELA→CCL2", "detection"),
    ("RELA", "IL6", "RELA→IL6", "detection"),
    ("NFKB1", "TNF", "NFKB1→TNF", "detection"),
    ("TP53", "CDKN1A", "TP53→CDKN1A", "detection"),
    ("TP53", "MDM2", "TP53→MDM2", "detection"),
    ("HIF1A", "VEGFA", "HIF1A→VEGFA", "detection"),
    ("HIF1A", "SLC2A1", "HIF1A→SLC2A1", "detection"),
    ("STAT1", "IRF1", "STAT1→IRF1", "detection"),
    ("NFE2L2", "NQO1", "NFE2L2→NQO1", "detection"),
    ("NFE2L2", "HMOX1", "NFE2L2→HMOX1", "detection"),
    ("HSF1", "HSPA1A", "HSF1→HSPA1A", "detection"),
    ("ATF4", "DDIT3", "ATF4→DDIT3", "detection"),
    # Housekeeping
    ("ACTB", "GAPDH", "ACTB↔GAPDH", "housekeeping"),
    ("B2M", "PPIA", "B2M↔PPIA", "housekeeping"),
    ("HPRT1", "TBP", "HPRT1↔TBP", "housekeeping"),
    ("RPL13A", "RPS18", "RPL13A↔RPS18", "housekeeping"),
]


def download_tcga_data():
    """Download TCGA RNA-seq data using UCSC Xena hub (pre-processed, easy to access)."""
    import urllib.request

    # UCSC Xena has pre-processed TCGA data — log2(TPM+1) — perfect for us
    # Gene expression: toil RSEM expected counts / TPM
    xena_base = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download"

    # Use TCGA TARGET GTEx combined dataset from Xena
    # Or use GDC directly. Let's check if we can use a simpler approach.

    # Actually, let's use the TCGA pan-cancer atlas files from GDC
    # Simplest: download from cBioPortal or use existing processed files

    # For now, let's try the Xena approach
    tpm_url = f"{xena_base}/tcga_RSEM_gene_tpm.gz"
    pheno_url = f"{xena_base}/TCGA_phenotype_denseDataOnlyDownload.tsv.gz"

    tpm_file = DATA_DIR / "tcga_RSEM_gene_tpm.gz"
    pheno_file = DATA_DIR / "TCGA_phenotype.tsv.gz"

    if not tpm_file.exists():
        _log(f"  Downloading TCGA TPM (~1.2GB)...")
        _log(f"  URL: {tpm_url}")
        try:
            urllib.request.urlretrieve(tpm_url, tpm_file)
            _log(f"  Downloaded: {tpm_file.stat().st_size / 1e6:.0f} MB")
        except Exception as e:
            _log(f"  Download failed: {e}")
            _log(f"  Trying alternative approach...")
            return None, None

    if not pheno_file.exists():
        _log(f"  Downloading TCGA phenotype...")
        try:
            urllib.request.urlretrieve(pheno_url, pheno_file)
            _log(f"  Downloaded: {pheno_file.stat().st_size / 1e6:.0f} MB")
        except Exception as e:
            _log(f"  Download failed: {e}")
            return None, None

    return tpm_file, pheno_file


def load_tcga_genes(tpm_file, target_genes):
    """Load only target genes from TCGA TPM file."""
    import gzip

    _log(f"  Loading {len(target_genes)} genes from TCGA TPM...")
    data = {}
    remaining = set(target_genes)

    with gzip.open(tpm_file, "rt") as f:
        header = f.readline().strip().split("\t")
        sample_ids = header[1:]

        for line in f:
            parts = line.strip().split("\t", 1)
            gene = parts[0]
            if gene in remaining:
                vals = np.array(parts[1].split("\t"), dtype=np.float32)
                data[gene] = vals
                remaining.discard(gene)
                _log(f"    Found {gene} ({len(target_genes) - len(remaining)}/{len(target_genes)})")
                if not remaining:
                    break

    _log(f"  Loaded {len(data)}/{len(target_genes)} genes, {len(sample_ids)} samples")
    if remaining:
        _log(f"  Missing: {remaining}")

    return pd.DataFrame(data, index=sample_ids), sample_ids


def main():
    t0 = time.time()
    _log("=" * 70)
    _log("TCGA PAIRED TUMOR/NORMAL COUPLING ANALYSIS")
    _log("=" * 70)

    # Download
    tpm_file, pheno_file = download_tcga_data()
    if tpm_file is None:
        _log("  Cannot proceed without data. Exiting.")
        return

    # Get all genes we need
    all_genes = set()
    for tf, tgt, _, _ in PAIRS:
        all_genes.add(tf)
        all_genes.add(tgt)

    # Load expression
    expr_df, sample_ids = load_tcga_genes(tpm_file, all_genes)

    # Load phenotype
    import gzip
    _log(f"\n  Loading phenotype...")
    pheno = pd.read_csv(pheno_file, sep="\t", compression="gzip")
    _log(f"  Phenotype: {pheno.shape[0]} samples, columns: {pheno.columns.tolist()[:10]}")

    # Identify sample type from TCGA barcode
    # TCGA-XX-XXXX-01 = tumor, TCGA-XX-XXXX-11 = normal
    expr_df["sample_id"] = expr_df.index
    expr_df["patient"] = expr_df["sample_id"].str[:12]
    expr_df["sample_type_code"] = expr_df["sample_id"].str[13:15]
    expr_df["is_tumor"] = expr_df["sample_type_code"].astype(str).str.startswith("0")  # 01-09 = tumor
    expr_df["is_normal"] = expr_df["sample_type_code"] == "11"  # 11 = solid tissue normal

    # Merge with phenotype for cancer type
    if "sample" in pheno.columns:
        pheno = pheno.rename(columns={"sample": "sample_id"})
    if "sample_id" in pheno.columns:
        expr_df = expr_df.merge(pheno[["sample_id", "_primary_disease"]].drop_duplicates(),
                                 on="sample_id", how="left")
    elif "sampleID" in pheno.columns:
        pheno = pheno.rename(columns={"sampleID": "sample_id"})
        expr_df = expr_df.merge(pheno[["sample_id", "_primary_disease"]].drop_duplicates(),
                                 on="sample_id", how="left")

    tumor = expr_df[expr_df["is_tumor"]].copy()
    normal = expr_df[expr_df["is_normal"]].copy()
    _log(f"\n  Tumor samples: {len(tumor)}")
    _log(f"  Normal samples: {len(normal)}")

    # Find cancer types with paired samples
    tumor_patients = set(tumor["patient"])
    normal_patients = set(normal["patient"])
    paired_patients = tumor_patients & normal_patients
    _log(f"  Paired patients (tumor + normal): {len(paired_patients)}")

    if "_primary_disease" in normal.columns:
        cancer_counts = normal[normal["patient"].isin(paired_patients)].groupby("_primary_disease")["patient"].nunique()
        cancer_counts = cancer_counts.sort_values(ascending=False)
        _log(f"\n  Cancer types with paired samples:")
        for ct, n in cancer_counts.head(15).items():
            _log(f"    {ct}: {n} patients")
        top_cancers = cancer_counts[cancer_counts >= 20].index.tolist()
    else:
        # Infer cancer type from barcode project
        normal["project"] = normal["sample_id"].str[:12].str[5:7]
        cancer_counts = normal[normal["patient"].isin(paired_patients)].groupby("project")["patient"].nunique()
        top_cancers = cancer_counts[cancer_counts >= 20].index.tolist()

    # ── Coupling analysis per cancer type ─────────────────────────────
    _log("\n" + "=" * 70)
    _log("Coupling: tumor vs normal (paired)")
    _log("=" * 70)

    gene_cols = [g for g in all_genes if g in expr_df.columns]
    results = []

    cancer_col = "_primary_disease" if "_primary_disease" in normal.columns else "project"

    for cancer in top_cancers:
        c_normal = normal[(normal[cancer_col] == cancer) & (normal["patient"].isin(paired_patients))]
        c_tumor = tumor[(tumor[cancer_col] == cancer) & (tumor["patient"].isin(paired_patients))]

        if len(c_normal) < 20 or len(c_tumor) < 20:
            continue

        _log(f"\n  {cancer}: {len(c_normal)} normal, {len(c_tumor)} tumor")

        for tf, tgt, label, cat in PAIRS:
            if tf not in gene_cols or tgt not in gene_cols:
                continue

            # Normal coupling
            rho_n, p_n = spearman_safe(c_normal[tf].values.astype(float),
                                        c_normal[tgt].values.astype(float))
            # Tumor coupling
            rho_t, p_t = spearman_safe(c_tumor[tf].values.astype(float),
                                        c_tumor[tgt].values.astype(float))

            delta = rho_t - rho_n if not (np.isnan(rho_t) or np.isnan(rho_n)) else np.nan

            results.append({
                "cancer": cancer, "label": label, "category": cat,
                "n_normal": len(c_normal), "n_tumor": len(c_tumor),
                "rho_normal": rho_n, "p_normal": p_n,
                "rho_tumor": rho_t, "p_tumor": p_t,
                "delta": delta,
            })

        # Print key pairs
        for tf, tgt, label, cat in [("SMAD3", "COL1A1", "SMAD3→COL1A1", "production"),
                                     ("RELA", "ICAM1", "RELA→ICAM1", "detection"),
                                     ("ACTB", "GAPDH", "ACTB↔GAPDH", "housekeeping")]:
            if tf not in gene_cols or tgt not in gene_cols:
                continue
            rho_n, _ = spearman_safe(c_normal[tf].values.astype(float),
                                      c_normal[tgt].values.astype(float))
            rho_t, _ = spearman_safe(c_tumor[tf].values.astype(float),
                                      c_tumor[tgt].values.astype(float))
            d = rho_t - rho_n if not (np.isnan(rho_t) or np.isnan(rho_n)) else np.nan
            _log(f"    {label:<20s}: normal ρ={rho_n:+.3f}, tumor ρ={rho_t:+.3f}, Δ={d:+.3f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / "tcga_coupling.csv", index=False)

    # ── Summary ──────────────────────────────────────────────────────
    _log("\n" + "=" * 70)
    _log("Summary")
    _log("=" * 70)

    if len(df_results) > 0:
        for cat in ["production", "detection", "housekeeping"]:
            sub = df_results[df_results["category"] == cat]
            deltas = sub["delta"].dropna()
            n_down = (deltas < 0).sum()
            _log(f"\n  {cat.upper()} (n={len(deltas)}):")
            _log(f"    Median Δ(tumor-normal) = {deltas.median():+.3f}")
            _log(f"    Coupling decreased in tumor: {n_down}/{len(deltas)} ({n_down/len(deltas):.0%})")

        # Production vs detection in cancer
        p_deltas = df_results[df_results["category"] == "production"]["delta"].dropna()
        d_deltas = df_results[df_results["category"] == "detection"]["delta"].dropna()
        if len(p_deltas) > 5 and len(d_deltas) > 5:
            u, p = stats.mannwhitneyu(p_deltas, d_deltas)
            _log(f"\n  Production vs Detection Δ: Wilcoxon p = {p:.4e}")

    # ── Figure ───────────────────────────────────────────────────────
    if len(df_results) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        # Panel A: Delta distribution by category
        ax = axes[0]
        cats = ["production", "detection", "housekeeping"]
        cat_data = [df_results[df_results["category"] == c]["delta"].dropna().values for c in cats]
        cat_colors = ["tab:green", "tab:orange", "tab:gray"]
        for i, (d, c, color) in enumerate(zip(cat_data, cats, cat_colors)):
            if len(d) > 0:
                ax.violinplot([d], positions=[i], showmeans=True, showmedians=True)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Δρ (tumor - normal)")
        ax.set_title("A: Coupling change in cancer\nby channel type")

        # Panel B: Heatmap cancer × pair
        ax = axes[1]
        pivot = df_results.pivot_table(index="cancer", columns="label", values="delta", aggfunc="first")
        pivot = pivot.dropna(thresh=5, axis=0).dropna(thresh=3, axis=1)
        if pivot.shape[0] > 0 and pivot.shape[1] > 0:
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
            ax.set_yticks(range(len(pivot)))
            cancer_short = [c[:30] for c in pivot.index]
            ax.set_yticklabels(cancer_short, fontsize=6)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
            plt.colorbar(im, ax=ax, shrink=0.5, label="Δρ (tumor-normal)")
        ax.set_title("B: Coupling change by cancer type × pair")

        # Panel C: Scatter — SMAD3→COL1A1 normal vs tumor
        ax = axes[2]
        smad_data = df_results[df_results["label"] == "SMAD3→COL1A1"]
        if len(smad_data) > 0:
            ax.scatter(smad_data["rho_normal"], smad_data["rho_tumor"], s=50, alpha=0.7,
                       edgecolors="gray", linewidth=0.3, c="tab:green")
            for _, r in smad_data.iterrows():
                ax.annotate(r["cancer"][:15], (r["rho_normal"], r["rho_tumor"]),
                            fontsize=5, alpha=0.7)
            lim = [-0.5, 0.8]
            ax.plot(lim, lim, "k--", alpha=0.3)
            ax.set_xlabel("Normal ρ (SMAD3→COL1A1)")
            ax.set_ylabel("Tumor ρ (SMAD3→COL1A1)")
            ax.set_title("C: SMAD3→COL1A1 coupling\nnormal vs tumor per cancer type")

        fig.suptitle("TCGA Paired Tumor/Normal Coupling Analysis",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "tcga_coupling.png", dpi=150)
        plt.close(fig)
        _log(f"  Saved tcga_coupling.png")

    _log(f"\n  Time: {time.time()-t0:.0f}s")
    _log("=" * 70)


if __name__ == "__main__":
    main()
