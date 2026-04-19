"""
Step 1: GTEx v8 Coupling Atlas — QC + Primary Analysis

For each of 54 tissues:
1. QC: N donors per age×sex, expression levels of panel genes
2. Coupling: Spearman(TF, target) for 30+ pairs, stratified by age decade and sex
3. Scatter plots for key pairs
4. A(t) = |ρ_D| / |ρ_P| per age decade

This is bulk RNA-seq — no zero-inflation, no dropout, no QC confound.
"""

import time
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data" / "gtex"
RESULTS_DIR = BASE / "results" / "step1_gtex"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Gene panel ────────────────────────────────────────────────────
PRODUCTION_PAIRS = [
    ("SMAD3", "COL1A1", "SMAD3→COL1A1"),
    ("SMAD3", "COL3A1", "SMAD3→COL3A1"),
    ("SMAD3", "FN1", "SMAD3→FN1"),
    ("SMAD3", "SERPINE1", "SMAD3→SERPINE1"),
    ("ESR1", "COL1A1", "ESR1→COL1A1"),
    ("ESR1", "ELN", "ESR1→ELN"),
    ("AR", "COL1A1", "AR→COL1A1"),
    ("SOX9", "ACAN", "SOX9→ACAN"),
    ("SOX9", "COL2A1", "SOX9→COL2A1"),
    ("PPARG", "FABP4", "PPARG→FABP4"),
    ("PPARG", "ADIPOQ", "PPARG→ADIPOQ"),
    ("RUNX2", "SPP1", "RUNX2→SPP1"),
    ("FOXO1", "PCK1", "FOXO1→PCK1"),
    ("HNF4A", "ALB", "HNF4A→ALB"),
    ("MYOD1", "MYH2", "MYOD1→MYH2"),
]

DETECTION_PAIRS = [
    ("RELA", "ICAM1", "RELA→ICAM1"),
    ("RELA", "IL6", "RELA→IL6"),
    ("RELA", "CCL2", "RELA→CCL2"),
    ("RELA", "NFKBIA", "RELA→NFKBIA"),
    ("NFKB1", "TNF", "NFKB1→TNF"),
    ("HIF1A", "VEGFA", "HIF1A→VEGFA"),
    ("HIF1A", "SLC2A1", "HIF1A→SLC2A1"),
    ("TP53", "CDKN1A", "TP53→CDKN1A"),
    ("TP53", "MDM2", "TP53→MDM2"),
    ("ATF4", "DDIT3", "ATF4→DDIT3"),
    ("XBP1", "DNAJB1", "XBP1→DNAJB1"),
    ("HSF1", "HSPA1A", "HSF1→HSPA1A"),
    ("STAT1", "IRF1", "STAT1→IRF1"),
    ("NFE2L2", "NQO1", "NFE2L2→NQO1"),
    ("NFE2L2", "HMOX1", "NFE2L2→HMOX1"),
]

HK_PAIRS = [
    ("ACTB", "GAPDH", "ACTB↔GAPDH"),
    ("RPL13A", "RPS18", "RPL13A↔RPS18"),
    ("HPRT1", "TBP", "HPRT1↔TBP"),
    ("B2M", "PPIA", "B2M↔PPIA"),
]

HORMONE_PAIRS = [
    ("ESR1", "PGR", "ESR1→PGR"),
    ("AR", "KLK3", "AR→KLK3"),
    ("AR", "SMAD3", "AR→SMAD3"),
]

ALL_PAIRS = PRODUCTION_PAIRS + DETECTION_PAIRS + HK_PAIRS + HORMONE_PAIRS


def _log(msg):
    print(msg, flush=True)


def spearman_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan, 1.0
    return stats.spearmanr(x, y)


def load_gtex_tpm(gene_list):
    """Load GTEx TPM for specific genes only (memory efficient)."""
    _log("  Loading GTEx TPM (streaming, gene-filtered)...")
    tpm_path = DATA_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"

    gene_set = set(gene_list)
    data = {}
    sample_ids = None

    with gzip.open(tpm_path, "rt") as f:
        # Skip first two lines (GCT header)
        f.readline()  # #1.2
        f.readline()  # nrows ncols
        # Header line
        header = f.readline().strip().split("\t")
        sample_ids = header[2:]  # First two cols: Name, Description

        for line in f:
            parts = line.strip().split("\t", 2)
            gene_id = parts[0]  # ENSG...
            gene_name = parts[1]  # Symbol

            if gene_name in gene_set:
                vals = np.array(parts[2].split("\t"), dtype=np.float32)
                data[gene_name] = vals
                gene_set.discard(gene_name)
                if not gene_set:
                    break

    _log(f"  Loaded {len(data)} genes, {len(sample_ids)} samples")
    if gene_set:
        _log(f"  Missing genes: {gene_set}")

    df = pd.DataFrame(data, index=sample_ids)
    return df


def main():
    t0 = time.time()
    _log("=" * 70)
    _log("GTEx v8 COUPLING ATLAS — Step 1")
    _log("=" * 70)

    # ── Load metadata ─────────────────────────────────────────────
    _log("\n[0] Loading metadata...")
    samples = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")

    # Extract subject ID from sample ID (GTEX-XXXX-...)
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")

    # Filter to RNA-seq samples only
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    _log(f"  RNA-seq samples: {len(samples)}")
    _log(f"  Unique donors: {samples['SUBJID'].nunique()}")
    _log(f"  Tissues: {samples['SMTSD'].nunique()}")

    # Sex mapping: 1=male, 2=female
    samples["sex"] = samples["SEX"].map({1: "male", 2: "female"})

    # Age decade midpoint for continuous analysis
    age_map = {"20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65, "70-79": 75}
    samples["age_mid"] = samples["AGE"].map(age_map)

    # Death classification
    # 0=ventilator case, 1=violent/fast, 2=fast natural, 3=intermediate, 4=slow
    # Exclude 1 (violent) per plan
    samples_clean = samples[samples["DTHHRDY"] != 1].copy()
    _log(f"  After excluding violent death: {len(samples_clean)} samples, "
         f"{samples_clean['SUBJID'].nunique()} donors")

    # ── Collect all needed genes ──────────────────────────────────
    all_genes = set()
    for pairs in [PRODUCTION_PAIRS, DETECTION_PAIRS, HK_PAIRS, HORMONE_PAIRS]:
        for tf, tgt, _ in pairs:
            all_genes.add(tf)
            all_genes.add(tgt)
    _log(f"\n  Panel genes: {len(all_genes)}")

    # ── Load TPM ──────────────────────────────────────────────────
    tpm = load_gtex_tpm(list(all_genes))
    # Log2(TPM+1) transform
    tpm_log = np.log2(tpm + 1)

    # Merge with metadata
    tpm_log["SAMPID"] = tpm_log.index
    merged = tpm_log.merge(
        samples_clean[["SAMPID", "SUBJID", "SMTSD", "sex", "AGE", "age_mid", "DTHHRDY"]],
        on="SAMPID", how="inner"
    )
    _log(f"  Merged samples with metadata: {len(merged)}")

    gene_cols = [c for c in tpm.columns if c in all_genes]
    _log(f"  Gene columns available: {len(gene_cols)}")

    # ── QC per tissue ─────────────────────────────────────────────
    _log("\n" + "=" * 70)
    _log("[1] QC: Tissue × Age × Sex")
    _log("=" * 70)

    tissue_counts = merged.groupby("SMTSD").agg(
        n_samples=("SAMPID", "count"),
        n_donors=("SUBJID", "nunique"),
        n_male=("sex", lambda x: (x == "male").sum()),
        n_female=("sex", lambda x: (x == "female").sum()),
    ).sort_values("n_samples", ascending=False)

    # Top tissues with enough samples
    good_tissues = tissue_counts[tissue_counts["n_samples"] >= 100].index.tolist()
    _log(f"\n  Tissues with ≥100 samples: {len(good_tissues)}")
    for t in good_tissues[:15]:
        row = tissue_counts.loc[t]
        _log(f"    {t:>45s}: n={row['n_samples']:4d} "
             f"(M={row['n_male']:3d}, F={row['n_female']:3d})")

    tissue_counts.to_csv(RESULTS_DIR / "tissue_sample_counts.csv")

    # ── Primary coupling analysis ─────────────────────────────────
    _log("\n" + "=" * 70)
    _log("[2] Coupling analysis — all tissues")
    _log("=" * 70)

    all_results = []

    # Priority tissues for detailed analysis
    priority_tissues = [
        "Skin - Sun Exposed (Lower leg)",
        "Muscle - Skeletal",
        "Liver",
        "Lung",
        "Heart - Left Ventricle",
        "Adipose - Subcutaneous",
        "Artery - Tibial",
        "Whole Blood",
        "Brain - Cortex",
        "Thyroid",
    ]

    for tissue in good_tissues:
        sub = merged[merged["SMTSD"] == tissue]
        if len(sub) < 50:
            continue

        is_priority = tissue in priority_tissues

        # Full cohort coupling
        for cat, pairs in [("production", PRODUCTION_PAIRS),
                            ("detection", DETECTION_PAIRS),
                            ("housekeeping", HK_PAIRS),
                            ("hormone", HORMONE_PAIRS)]:
            for tf, tgt, label in pairs:
                if tf not in gene_cols or tgt not in gene_cols:
                    continue

                # Check expression level
                tf_expr = sub[tf].values.astype(float)
                tgt_expr = sub[tgt].values.astype(float)
                tf_mean = np.mean(tf_expr)
                tgt_mean = np.mean(tgt_expr)

                # Skip if either gene not expressed
                if tf_mean < 0.1 or tgt_mean < 0.1:  # log2(TPM+1) < 0.1 ≈ TPM < 0.07
                    continue

                rho_all, p_all = spearman_safe(tf_expr, tgt_expr)

                row = {
                    "tissue": tissue, "category": cat, "label": label,
                    "tf": tf, "target": tgt,
                    "n_all": len(sub), "rho_all": rho_all, "p_all": p_all,
                    "tf_mean_log2tpm": tf_mean, "tgt_mean_log2tpm": tgt_mean,
                }

                # Age-stratified (decades)
                for age_bin in sorted(sub["AGE"].dropna().unique()):
                    age_sub = sub[sub["AGE"] == age_bin]
                    if len(age_sub) < 20:
                        continue
                    rho, p = spearman_safe(age_sub[tf].values.astype(float),
                                           age_sub[tgt].values.astype(float))
                    row[f"rho_{age_bin}"] = rho
                    row[f"p_{age_bin}"] = p
                    row[f"n_{age_bin}"] = len(age_sub)

                # Sex-stratified
                for sex in ["male", "female"]:
                    sex_sub = sub[sub["sex"] == sex]
                    if len(sex_sub) < 30:
                        continue
                    rho, p = spearman_safe(sex_sub[tf].values.astype(float),
                                           sex_sub[tgt].values.astype(float))
                    row[f"rho_{sex}"] = rho
                    row[f"p_{sex}"] = p
                    row[f"n_{sex}"] = len(sex_sub)

                # Age-decline: Spearman(age_mid, residual coupling)
                # Per-individual: compute TF*target product as proxy
                # Or: sliding window coupling
                if not np.isnan(rho_all):
                    all_results.append(row)

        if is_priority:
            _log(f"\n  === {tissue} (n={len(sub)}) ===")
            sub_results = [r for r in all_results if r["tissue"] == tissue]
            for r in sub_results:
                age_rhos = {k.replace("rho_", ""): v for k, v in r.items()
                            if k.startswith("rho_") and k != "rho_all"
                            and not k.endswith("male") and not k.endswith("female")
                            and not np.isnan(v)}
                age_str = " | ".join(f"{k}:{v:+.2f}" for k, v in sorted(age_rhos.items()))
                _log(f"    {r['label']:<20s} ρ_all={r['rho_all']:+.3f} (p={r['p_all']:.1e}) | {age_str}")

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(RESULTS_DIR / "gtex_coupling_all.csv", index=False)
    _log(f"\n  Total coupling measurements: {len(df_all)}")
    _log(f"  Tissues: {df_all['tissue'].nunique()}")

    # ── A(t) per tissue per age decade ────────────────────────────
    _log("\n" + "=" * 70)
    _log("[3] A(t) = |ρ_D| / |ρ_P| per tissue per age decade")
    _log("=" * 70)

    age_decades = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]
    A_results = []

    for tissue in good_tissues:
        t_data = df_all[df_all["tissue"] == tissue]
        if len(t_data) < 10:
            continue

        for age_bin in age_decades:
            rho_col = f"rho_{age_bin}"
            n_col = f"n_{age_bin}"

            p_vals = t_data[(t_data["category"] == "production") & t_data[rho_col].notna()][rho_col].abs()
            d_vals = t_data[(t_data["category"] == "detection") & t_data[rho_col].notna()][rho_col].abs()
            hk_vals = t_data[(t_data["category"] == "housekeeping") & t_data[rho_col].notna()][rho_col].abs()

            if len(p_vals) < 3 or len(d_vals) < 3:
                continue

            n_donors = t_data[t_data[n_col].notna()][n_col].iloc[0] if t_data[n_col].notna().any() else 0

            A_results.append({
                "tissue": tissue, "age": age_bin,
                "age_mid": age_map.get(age_bin, 0),
                "n_donors": n_donors,
                "mean_abs_rho_P": p_vals.mean(),
                "mean_abs_rho_D": d_vals.mean(),
                "mean_abs_rho_HK": hk_vals.mean() if len(hk_vals) > 0 else np.nan,
                "n_P_pairs": len(p_vals),
                "n_D_pairs": len(d_vals),
                "A": d_vals.mean() / p_vals.mean() if p_vals.mean() > 0.01 else np.nan,
            })

    df_A = pd.DataFrame(A_results)
    df_A.to_csv(RESULTS_DIR / "gtex_A_by_tissue_age.csv", index=False)

    # Print A trajectories for priority tissues
    for tissue in priority_tissues:
        sub = df_A[df_A["tissue"] == tissue].sort_values("age_mid")
        if len(sub) < 3:
            continue
        _log(f"\n  {tissue}:")
        for _, r in sub.iterrows():
            _log(f"    {r['age']:>5s}: |ρ_P|={r['mean_abs_rho_P']:.3f}, "
                 f"|ρ_D|={r['mean_abs_rho_D']:.3f}, A={r['A']:.3f} (n={r['n_donors']:.0f})")

        # Trend
        if len(sub) >= 4:
            rho_A, p_A = stats.spearmanr(sub["age_mid"], sub["A"])
            rho_P, p_P = stats.spearmanr(sub["age_mid"], sub["mean_abs_rho_P"])
            rho_D, p_D = stats.spearmanr(sub["age_mid"], sub["mean_abs_rho_D"])
            _log(f"    Trends: A vs age ρ={rho_A:+.3f} p={p_A:.3f}, "
                 f"|ρ_P| vs age ρ={rho_P:+.3f} p={p_P:.3f}, "
                 f"|ρ_D| vs age ρ={rho_D:+.3f} p={p_D:.3f}")

    # ── Summary across ALL tissues ────────────────────────────────
    _log("\n" + "=" * 70)
    _log("[4] Summary: A(t) trends across all tissues")
    _log("=" * 70)

    tissue_trends = []
    for tissue in df_A["tissue"].unique():
        sub = df_A[df_A["tissue"] == tissue].sort_values("age_mid")
        if len(sub) < 4:
            continue
        rho_A, p_A = stats.spearmanr(sub["age_mid"], sub["A"])
        rho_P, p_P = stats.spearmanr(sub["age_mid"], sub["mean_abs_rho_P"])
        rho_D, p_D = stats.spearmanr(sub["age_mid"], sub["mean_abs_rho_D"])
        tissue_trends.append({
            "tissue": tissue, "n_decades": len(sub),
            "rho_A_vs_age": rho_A, "p_A_vs_age": p_A,
            "rho_P_vs_age": rho_P, "p_P_vs_age": p_P,
            "rho_D_vs_age": rho_D, "p_D_vs_age": p_D,
        })

    df_trends = pd.DataFrame(tissue_trends)
    df_trends.to_csv(RESULTS_DIR / "gtex_A_trends.csv", index=False)

    n_A_up = (df_trends["rho_A_vs_age"] > 0).sum()
    n_A_sig = ((df_trends["rho_A_vs_age"] > 0) & (df_trends["p_A_vs_age"] < 0.05)).sum()
    n_P_down = (df_trends["rho_P_vs_age"] < 0).sum()
    n_D_stable = (df_trends["rho_D_vs_age"].abs() < 0.3).sum()

    _log(f"\n  Tissues analyzed: {len(df_trends)}")
    _log(f"  A increases with age: {n_A_up}/{len(df_trends)} ({n_A_up/len(df_trends):.0%})")
    _log(f"  A increases significantly (p<0.05): {n_A_sig}/{len(df_trends)}")
    _log(f"  |ρ_P| declines with age: {n_P_down}/{len(df_trends)} ({n_P_down/len(df_trends):.0%})")
    _log(f"  |ρ_D| stable (|ρ|<0.3): {n_D_stable}/{len(df_trends)}")

    # Median trends
    _log(f"\n  Median ρ(A, age) across tissues: {df_trends['rho_A_vs_age'].median():+.3f}")
    _log(f"  Median ρ(|ρ_P|, age): {df_trends['rho_P_vs_age'].median():+.3f}")
    _log(f"  Median ρ(|ρ_D|, age): {df_trends['rho_D_vs_age'].median():+.3f}")

    # Sign test: does A increase with age more often than chance?
    sign_stat, sign_p = stats.binomtest(n_A_up, len(df_trends), 0.5)
    _log(f"  Sign test (A↑ with age): {n_A_up}/{len(df_trends)}, p={sign_p.pvalue:.4f}")

    # ── Figure ────────────────────────────────────────────────────
    _log("\n" + "=" * 70)
    _log("[5] Figures")
    _log("=" * 70)

    # Figure 1: A(t) for priority tissues
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharey=True)
    axes = axes.flatten()

    for i, tissue in enumerate(priority_tissues):
        ax = axes[i]
        sub = df_A[df_A["tissue"] == tissue].sort_values("age_mid")
        if len(sub) < 2:
            ax.set_title(tissue.split(" - ")[-1] + "\n(insufficient data)", fontsize=9)
            continue

        ax.plot(sub["age_mid"], sub["mean_abs_rho_P"], "g-o", label="|ρ_P|", markersize=6, linewidth=2)
        ax.plot(sub["age_mid"], sub["mean_abs_rho_D"], "r-s", label="|ρ_D|", markersize=6, linewidth=2)
        ax.plot(sub["age_mid"], sub["mean_abs_rho_HK"], "k--^", label="|ρ_HK|", markersize=5, alpha=0.5)

        # A on secondary axis
        ax2 = ax.twinx()
        ax2.plot(sub["age_mid"], sub["A"], "b-D", alpha=0.4, markersize=4)
        ax2.set_ylabel("A", color="blue", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="blue", labelsize=7)

        short_name = tissue.replace("Skin - Sun Exposed (Lower leg)", "Skin (sun)")
        short_name = short_name.replace("Adipose - Subcutaneous", "Adipose (subcut)")
        short_name = short_name.replace("Heart - Left Ventricle", "Heart (LV)")
        short_name = short_name.replace("Muscle - Skeletal", "Muscle")
        short_name = short_name.replace("Artery - Tibial", "Artery")
        short_name = short_name.replace("Brain - Cortex", "Brain (cortex)")
        ax.set_title(short_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Age decade midpoint")
        if i % 5 == 0:
            ax.set_ylabel("Mean |Spearman ρ|")
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("GTEx v8 Coupling Atlas — |ρ_P| (production) vs |ρ_D| (detection) vs age\n"
                 "948 donors, 54 tissues, bulk RNA-seq",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "gtex_coupling_trajectories.png", dpi=150)
    plt.close(fig)
    _log(f"  Saved gtex_coupling_trajectories.png")

    # Figure 2: Scatter plots for skin (top pairs)
    skin_tissue = "Skin - Sun Exposed (Lower leg)"
    skin = merged[merged["SMTSD"] == skin_tissue]

    if len(skin) > 100:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        scatter_pairs = [
            ("SMAD3", "COL1A1", "SMAD3→COL1A1 (structural)"),
            ("ESR1", "COL1A1", "ESR1→COL1A1 (hormone→ECM)"),
            ("RELA", "ICAM1", "RELA→ICAM1 (detection)"),
            ("HIF1A", "VEGFA", "HIF1A→VEGFA (hypoxia)"),
            ("ACTB", "GAPDH", "ACTB↔GAPDH (HK control)"),
            ("AR", "SMAD3", "AR→SMAD3 (hormone cross-talk)"),
        ]

        for idx, (tf, tgt, title) in enumerate(scatter_pairs):
            ax = axes[idx // 3, idx % 3]
            if tf in gene_cols and tgt in gene_cols:
                sc = ax.scatter(skin[tf].astype(float), skin[tgt].astype(float),
                                c=skin["age_mid"].astype(float), cmap="coolwarm",
                                s=15, alpha=0.6, edgecolors="none")

                rho, p = spearman_safe(skin[tf].values.astype(float),
                                        skin[tgt].values.astype(float))
                ax.set_title(f"{title}\nρ={rho:+.3f}, p={p:.1e}, n={len(skin)}", fontsize=10)
                ax.set_xlabel(f"{tf} log2(TPM+1)")
                ax.set_ylabel(f"{tgt} log2(TPM+1)")

                if idx == 0:
                    plt.colorbar(sc, ax=ax, label="Age")
            else:
                ax.set_title(f"{title}\n(gene not in data)")

        fig.suptitle(f"GTEx Skin (Sun Exposed) — {len(skin)} samples\nEach dot = one donor",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "gtex_skin_scatter.png", dpi=150)
        plt.close(fig)
        _log(f"  Saved gtex_skin_scatter.png")

    # Figure 3: Heatmap of A(t) trend across all tissues
    if len(df_trends) > 5:
        fig, ax = plt.subplots(figsize=(8, max(6, len(df_trends) * 0.3)))
        df_sorted = df_trends.sort_values("rho_A_vs_age", ascending=True)
        colors = ["tab:green" if r < 0 else "tab:red" for r in df_sorted["rho_A_vs_age"]]
        sig = ["*" if p < 0.05 else "" for p in df_sorted["p_A_vs_age"]]

        bars = ax.barh(range(len(df_sorted)), df_sorted["rho_A_vs_age"].values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(df_sorted)))
        ylabels = [f"{t} {s}" for t, s in zip(df_sorted["tissue"].values, sig)]
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Spearman ρ(A, age)")
        ax.set_title(f"Does detection/production ratio increase with age?\n"
                     f"* = p<0.05. {n_A_up}/{len(df_trends)} tissues show A↑ (sign test p={sign_p.pvalue:.3f})")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "gtex_A_trend_heatmap.png", dpi=150)
        plt.close(fig)
        _log(f"  Saved gtex_A_trend_heatmap.png")

    # ── Sex-stratified coupling for skin ──────────────────────────
    _log("\n" + "=" * 70)
    _log("[6] Sex-stratified coupling — Skin")
    _log("=" * 70)

    if len(skin) > 100:
        for sex in ["male", "female"]:
            sex_sub = skin[skin["sex"] == sex]
            _log(f"\n  {sex.upper()} (n={len(sex_sub)}):")

            for age_bin in ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]:
                age_sub = sex_sub[sex_sub["AGE"] == age_bin]
                if len(age_sub) < 15:
                    continue

                key_pairs_sex = [("ESR1", "COL1A1"), ("SMAD3", "COL1A1"),
                                  ("RELA", "ICAM1"), ("ACTB", "GAPDH")]
                parts = []
                for tf, tgt in key_pairs_sex:
                    if tf in gene_cols and tgt in gene_cols:
                        rho, _ = spearman_safe(age_sub[tf].values.astype(float),
                                                age_sub[tgt].values.astype(float))
                        parts.append(f"{tf}→{tgt}:{rho:+.2f}")
                _log(f"    {age_bin} (n={len(age_sub):3d}): {' | '.join(parts)}")

    _log(f"\n  Total time: {time.time()-t0:.0f}s")
    _log(f"  Results: {RESULTS_DIR}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
