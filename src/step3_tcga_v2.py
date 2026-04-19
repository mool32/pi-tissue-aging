"""
TCGA v2 — parse sample types from barcodes directly, no phenotype file needed.
TCGA barcode: TCGA-XX-XXXX-SSV-... where SS = sample type (01=tumor, 11=normal)
Project = first 12 chars → TCGA-XX (2-letter project code after TCGA-)
"""

import gzip
import time
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg):
    print(msg, flush=True)


def spearman_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan, 1.0
    return stats.spearmanr(x, y)


PAIRS = [
    ("SMAD3", "COL1A1", "SMAD3→COL1A1", "production"),
    ("SMAD3", "COL3A1", "SMAD3→COL3A1", "production"),
    ("SMAD3", "FN1", "SMAD3→FN1", "production"),
    ("SMAD3", "SERPINE1", "SMAD3→SERPINE1", "production"),
    ("ESR1", "COL1A1", "ESR1→COL1A1", "production"),
    ("ESR1", "ELN", "ESR1→ELN", "production"),
    ("AR", "COL1A1", "AR→COL1A1", "production"),
    ("SOX9", "COL2A1", "SOX9→COL2A1", "production"),
    ("PPARG", "FABP4", "PPARG→FABP4", "production"),
    ("RUNX2", "SPP1", "RUNX2→SPP1", "production"),
    ("HNF4A", "ALB", "HNF4A→ALB", "production"),
    ("FOXO1", "PCK1", "FOXO1→PCK1", "production"),
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
    ("ACTB", "GAPDH", "ACTB↔GAPDH", "housekeeping"),
    ("B2M", "PPIA", "B2M↔PPIA", "housekeeping"),
    ("HPRT1", "TBP", "HPRT1↔TBP", "housekeeping"),
    ("RPL13A", "RPS18", "RPL13A↔RPS18", "housekeeping"),
]

# TCGA project codes → cancer names
PROJECT_NAMES = {
    "BRCA": "Breast", "LUAD": "Lung Adeno", "LUSC": "Lung Squam",
    "PRAD": "Prostate", "THCA": "Thyroid", "HNSC": "Head&Neck",
    "KIRC": "Kidney Clear", "KIRP": "Kidney Pap", "KICH": "Kidney Chromo",
    "LIHC": "Liver", "STAD": "Stomach", "COAD": "Colon",
    "READ": "Rectum", "BLCA": "Bladder", "UCEC": "Uterine",
    "ESCA": "Esophagus", "PCPG": "Pheochromocytoma", "CHOL": "Bile Duct",
    "PAAD": "Pancreas", "CESC": "Cervical", "OV": "Ovarian",
    "GBM": "Glioblastoma", "SKCM": "Melanoma", "SARC": "Sarcoma",
    "THYM": "Thymoma", "DLBC": "Lymphoma", "ACC": "Adrenocortical",
    "MESO": "Mesothelioma", "UCS": "Uterine Sarcoma", "UVM": "Uveal Melanoma",
    "TGCT": "Testicular",
}


def main():
    t0 = time.time()
    _log("=" * 70)
    _log("TCGA v2 — PAIRED TUMOR/NORMAL COUPLING")
    _log("=" * 70)

    tpm_file = DATA_DIR / "tcga_RSEM_gene_tpm.gz"
    if not tpm_file.exists():
        _log("  TPM file not found. Exiting.")
        return

    # Get target genes
    all_genes = set()
    for tf, tgt, _, _ in PAIRS:
        all_genes.add(tf)
        all_genes.add(tgt)

    # Load genes
    _log(f"  Loading {len(all_genes)} genes...")
    data = {}
    remaining = set(all_genes)
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
                if not remaining:
                    break

    _log(f"  Found {len(data)}/{len(all_genes)} genes, {len(sample_ids)} samples")
    if remaining:
        _log(f"  Missing: {remaining}")

    # Build dataframe
    expr = pd.DataFrame(data, index=sample_ids)
    # Log-transform (data is already log2(TPM+0.001) in Xena, but let's check)
    # Xena RSEM gene TPM is log2(tpm+0.001), so values can be negative
    # For coupling analysis, use as-is (already log-transformed)

    # Parse barcodes
    expr["barcode"] = expr.index
    expr["patient"] = expr["barcode"].str[:12]
    expr["sample_type"] = expr["barcode"].str[13:15]

    # Extract project code from barcode
    # TCGA barcodes: TCGA-XX-XXXX-... where XX is tissue source site
    # Need to map patient to project differently
    # Actually, Xena sample IDs may be in format: TCGA-XX-XXXX-01A-...
    # The project is determined by the study, not barcode alone
    # Let's infer from the data structure

    _log(f"\n  Sample type distribution:")
    type_counts = expr["sample_type"].value_counts()
    for st, n in type_counts.head(10).items():
        label = "tumor" if st in ["01", "02", "03", "04", "05", "06"] else \
                "normal" if st == "11" else f"other({st})"
        _log(f"    {st}: {n:6d} ({label})")

    tumor = expr[expr["sample_type"].isin(["01", "02", "03", "04", "05", "06"])].copy()
    normal = expr[expr["sample_type"] == "11"].copy()
    _log(f"\n  Tumor: {len(tumor)}, Normal: {len(normal)}")

    paired = set(tumor["patient"]) & set(normal["patient"])
    _log(f"  Paired patients: {len(paired)}")

    # Group by TSS (tissue source site) — chars 5-6 of barcode
    # This groups by project indirectly
    normal_paired = normal[normal["patient"].isin(paired)].copy()

    # Try to identify project from batch/TSS patterns
    # Actually, simplest: group all paired normals by shared TSS
    normal_paired["tss"] = normal_paired["barcode"].str[5:7]

    # Group by TSS and only keep TSS with enough samples
    tss_counts = normal_paired.groupby("tss")["patient"].nunique()
    _log(f"\n  TSS sites with paired samples: {len(tss_counts)}")

    # Better approach: just process all paired samples together,
    # then also by major TSS groups
    gene_cols = [g for g in all_genes if g in expr.columns]

    # GLOBAL: all paired tumor vs all paired normal
    _log("\n" + "=" * 70)
    _log("GLOBAL: All paired tumor vs normal")
    _log("=" * 70)

    t_paired = tumor[tumor["patient"].isin(paired)]
    n_paired = normal[normal["patient"].isin(paired)]

    results = []
    _log(f"  {'Label':<22s} {'ρ_normal':>9s} {'ρ_tumor':>9s} {'Δ':>8s}")
    _log(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*8}")

    for tf, tgt, label, cat in PAIRS:
        if tf not in gene_cols or tgt not in gene_cols:
            continue
        rho_n, p_n = spearman_safe(n_paired[tf].values, n_paired[tgt].values)
        rho_t, p_t = spearman_safe(t_paired[tf].values, t_paired[tgt].values)
        delta = rho_t - rho_n if not (np.isnan(rho_t) or np.isnan(rho_n)) else np.nan
        results.append({"cancer": "ALL_PAIRED", "label": label, "category": cat,
                         "n_normal": len(n_paired), "n_tumor": len(t_paired),
                         "rho_normal": rho_n, "rho_tumor": rho_t, "delta": delta})
        _log(f"  {label:<22s} {rho_n:>+9.3f} {rho_t:>+9.3f} {delta:>+8.3f}")

    # BY TSS (proxy for cancer type)
    _log("\n" + "=" * 70)
    _log("BY TSS (proxy for cancer type)")
    _log("=" * 70)

    big_tss = tss_counts[tss_counts >= 15].index.tolist()
    _log(f"  TSS with ≥15 paired patients: {len(big_tss)}")

    for tss in sorted(big_tss):
        tss_patients = set(normal_paired[normal_paired["tss"] == tss]["patient"])
        tss_normal = normal[normal["patient"].isin(tss_patients)]
        tss_tumor = tumor[tumor["patient"].isin(tss_patients)]

        if len(tss_normal) < 15 or len(tss_tumor) < 15:
            continue

        _log(f"\n  TSS={tss}: {len(tss_normal)} normal, {len(tss_tumor)} tumor")

        for tf, tgt, label, cat in PAIRS:
            if tf not in gene_cols or tgt not in gene_cols:
                continue
            rho_n, _ = spearman_safe(tss_normal[tf].values, tss_normal[tgt].values)
            rho_t, _ = spearman_safe(tss_tumor[tf].values, tss_tumor[tgt].values)
            delta = rho_t - rho_n if not (np.isnan(rho_t) or np.isnan(rho_n)) else np.nan
            results.append({"cancer": f"TSS_{tss}", "label": label, "category": cat,
                             "n_normal": len(tss_normal), "n_tumor": len(tss_tumor),
                             "rho_normal": rho_n, "rho_tumor": rho_t, "delta": delta})

        # Print key pairs
        for tf, tgt, label, _ in [PAIRS[0], PAIRS[12], PAIRS[26]]:
            if tf not in gene_cols or tgt not in gene_cols:
                continue
            rho_n, _ = spearman_safe(tss_normal[tf].values, tss_normal[tgt].values)
            rho_t, _ = spearman_safe(tss_tumor[tf].values, tss_tumor[tgt].values)
            d = rho_t - rho_n if not (np.isnan(rho_t) or np.isnan(rho_n)) else np.nan
            _log(f"    {label:<20s}: N ρ={rho_n:+.3f}, T ρ={rho_t:+.3f}, Δ={d:+.3f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / "tcga_coupling.csv", index=False)

    # Summary
    _log("\n" + "=" * 70)
    _log("SUMMARY")
    _log("=" * 70)

    # Global all-paired
    global_r = df_results[df_results["cancer"] == "ALL_PAIRED"]
    for cat in ["production", "detection", "housekeeping"]:
        sub = global_r[global_r["category"] == cat]
        deltas = sub["delta"].dropna()
        if len(deltas) > 0:
            n_down = (deltas < 0).sum()
            _log(f"  {cat.upper():>15s}: median Δ = {deltas.median():+.3f}, "
                 f"coupling ↓ in tumor: {n_down}/{len(deltas)}")

    # Production vs detection
    tss_r = df_results[df_results["cancer"] != "ALL_PAIRED"]
    if len(tss_r) > 0:
        p_d = tss_r[tss_r["category"] == "production"]["delta"].dropna()
        d_d = tss_r[tss_r["category"] == "detection"]["delta"].dropna()
        if len(p_d) > 5 and len(d_d) > 5:
            u, p = stats.mannwhitneyu(p_d, d_d)
            _log(f"\n  Production vs Detection Δ across TSS groups: p = {p:.4e}")
            _log(f"    Production median Δ = {p_d.median():+.3f}")
            _log(f"    Detection median Δ = {d_d.median():+.3f}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel A: Global coupling bar chart
    ax = axes[0]
    g = global_r.dropna(subset=["delta"]).sort_values("delta")
    colors = {"production": "tab:green", "detection": "tab:orange", "housekeeping": "tab:gray"}
    ax.barh(range(len(g)), g["delta"].values,
            color=[colors.get(c, "gray") for c in g["category"]])
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g["label"].values, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δρ (tumor - normal)")
    ax.set_title(f"A: All paired samples\n(n={len(n_paired)} normal, {len(t_paired)} tumor)")

    # Panel B: Normal ρ vs Tumor ρ scatter
    ax = axes[1]
    for cat, color in colors.items():
        sub = global_r[global_r["category"] == cat].dropna(subset=["rho_normal", "rho_tumor"])
        if len(sub) > 0:
            ax.scatter(sub["rho_normal"], sub["rho_tumor"], c=color, s=60,
                       alpha=0.7, label=cat, edgecolors="gray", linewidth=0.3)
            for _, r in sub.iterrows():
                ax.annotate(r["label"].split("→")[-1][:6], (r["rho_normal"], r["rho_tumor"]),
                            fontsize=5)
    ax.plot([-0.3, 0.8], [-0.3, 0.8], "k--", alpha=0.3)
    ax.set_xlabel("Normal ρ")
    ax.set_ylabel("Tumor ρ")
    ax.set_title("B: Coupling: normal vs tumor")
    ax.legend(fontsize=8)

    # Panel C: Delta by category (violin)
    ax = axes[2]
    if len(tss_r) > 0:
        cat_data = []
        cat_labels = []
        cat_colors_list = []
        for cat in ["production", "detection", "housekeeping"]:
            vals = tss_r[tss_r["category"] == cat]["delta"].dropna().values
            if len(vals) > 3:
                cat_data.append(vals)
                cat_labels.append(f"{cat}\n(n={len(vals)})")
                cat_colors_list.append(colors[cat])

        if cat_data:
            parts = ax.violinplot(cat_data, showmeans=True, showmedians=True)
            for j, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(cat_colors_list[j])
                pc.set_alpha(0.6)
            ax.set_xticks(range(1, len(cat_labels) + 1))
            ax.set_xticklabels(cat_labels)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.set_ylabel("Δρ (tumor - normal)")
            ax.set_title("C: Coupling change by category\n(across TSS groups)")

    fig.suptitle("TCGA Paired Tumor/Normal — Coupling Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tcga_coupling.png", dpi=150)
    plt.close(fig)
    _log(f"  Saved tcga_coupling.png")

    _log(f"\n  Time: {time.time()-t0:.0f}s")
    _log("=" * 70)


if __name__ == "__main__":
    main()
