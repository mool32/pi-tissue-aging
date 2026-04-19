"""
Direction 1: COL1A1 systemic coordination — deep dive

Questions:
1. Age-stratified cross-tissue ρ: does coordination weaken with age?
2. Other ECM genes (COL3A1, FN1, ELN, COL1A2) — also systemic?
3. Non-ECM genes for comparison — is this ECM-specific or general?
4. Sex-stratified cross-tissue ρ
5. Partial correlation controlling for age — is coordination independent of age?
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
DATA_DIR = BASE / "data" / "gtex"
RESULTS_DIR = BASE / "results" / "step3_col1a1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg):
    print(msg, flush=True)


def spearman_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan, 1.0
    return stats.spearmanr(x, y)


def partial_spearman(x, y, z):
    """Partial Spearman correlation of x,y controlling for z."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 15:
        return np.nan, 1.0
    # Rank-transform then partial
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    # Residualize
    from numpy.polynomial.polynomial import polyfit
    cx = np.polyfit(rz, rx, 1)
    cy = np.polyfit(rz, ry, 1)
    res_x = rx - np.polyval(cx, rz)
    res_y = ry - np.polyval(cy, rz)
    return stats.spearmanr(res_x, res_y)


def main():
    t0 = time.time()
    _log("=" * 70)
    _log("DIRECTION 1: COL1A1 Systemic Coordination — Deep Dive")
    _log("=" * 70)

    # Load metadata
    samples = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    samples["sex"] = samples["SEX"].map({1: "male", 2: "female"})

    # Age midpoints
    age_map = {"20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65, "70-79": 75}
    samples["age_mid"] = samples["AGE"].map(age_map)

    # Load genes — ECM panel + controls
    target_genes = {
        # ECM / structural
        "COL1A1", "COL1A2", "COL3A1", "COL5A1", "FN1", "ELN", "LAMA1",
        "BGN", "DCN", "LUM", "VCAN",
        # TFs
        "SMAD3", "SMAD2", "ESR1", "AR", "RELA",
        # Signaling outputs
        "ICAM1", "IL6", "CCL2", "TNF",
        # Housekeeping
        "ACTB", "GAPDH", "B2M", "HPRT1",
        # Metabolic (control — not ECM)
        "ALB", "INS", "GCG", "PCK1",
        # Immune markers
        "CD3E", "CD19", "CD14", "PTPRC",
    }

    _log(f"  Loading {len(target_genes)} genes from GTEx TPM...")
    tpm_path = DATA_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    data = {}
    remaining = set(target_genes)
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline()
        header = f.readline().strip().split("\t")
        sample_ids = header[2:]
        for line in f:
            parts = line.strip().split("\t", 2)
            gene_name = parts[1]
            if gene_name in remaining:
                vals = np.array(parts[2].split("\t"), dtype=np.float32)
                data[gene_name] = vals
                remaining.discard(gene_name)
                if not remaining:
                    break

    _log(f"  Found: {len(data)}/{len(target_genes)} genes")
    if remaining:
        _log(f"  Missing: {remaining}")

    tpm_df = pd.DataFrame(data, index=sample_ids)
    tpm_log = np.log2(tpm_df + 1)
    tpm_log["SAMPID"] = tpm_log.index
    merged = tpm_log.merge(samples[["SAMPID", "SUBJID", "SMTSD", "sex", "AGE", "age_mid"]],
                           on="SAMPID", how="inner")

    # Top tissues by donor count
    tissue_counts = merged.groupby("SMTSD")["SUBJID"].nunique().sort_values(ascending=False)
    top_tissues = tissue_counts[tissue_counts >= 100].index.tolist()
    _log(f"  Tissues with ≥100 donors: {len(top_tissues)}")

    # ══════════════════════════════════════════════════════════════════
    # Q1: Which genes are cross-tissue coordinated?
    # ══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Q1] Cross-tissue coordination — which genes?")
    _log("=" * 70)

    # Pick 6 top tissues for pairwise comparison
    focus_tissues = top_tissues[:6]
    genes_to_test = [g for g in sorted(data.keys()) if g in merged.columns]

    gene_cross = []
    for gene in genes_to_test:
        rho_list = []
        for i, t1 in enumerate(focus_tissues):
            for t2 in focus_tissues[i+1:]:
                d1 = merged[merged["SMTSD"] == t1][["SUBJID", gene]].set_index("SUBJID")
                d2 = merged[merged["SMTSD"] == t2][["SUBJID", gene]].set_index("SUBJID")
                common = d1.index.intersection(d2.index)
                if len(common) < 50:
                    continue
                rho, p = spearman_safe(
                    d1.loc[common, gene].values.astype(float),
                    d2.loc[common, gene].values.astype(float))
                if not np.isnan(rho):
                    rho_list.append({"gene": gene, "t1": t1, "t2": t2,
                                      "n": len(common), "rho": rho, "p": p})

        if rho_list:
            rhos = [r["rho"] for r in rho_list]
            gene_cross.append({
                "gene": gene,
                "median_cross_rho": np.median(rhos),
                "mean_cross_rho": np.mean(rhos),
                "n_sig": sum(1 for r in rho_list if r["p"] < 0.05),
                "n_pairs": len(rho_list),
                "pct_sig": sum(1 for r in rho_list if r["p"] < 0.05) / len(rho_list),
            })

    df_gc = pd.DataFrame(gene_cross).sort_values("median_cross_rho", ascending=False)
    df_gc.to_csv(RESULTS_DIR / "gene_cross_tissue.csv", index=False)

    _log(f"\n  {'Gene':<12s} {'Median ρ':>10s} {'%Sig':>6s} {'Category':>12s}")
    _log(f"  {'-'*12} {'-'*10} {'-'*6} {'-'*12}")

    ecm_genes = {"COL1A1", "COL1A2", "COL3A1", "COL5A1", "FN1", "ELN", "BGN", "DCN", "LUM", "VCAN", "LAMA1"}
    hk_genes = {"ACTB", "GAPDH", "B2M", "HPRT1"}
    signal_genes = {"ICAM1", "IL6", "CCL2", "TNF", "RELA"}

    for _, r in df_gc.iterrows():
        cat = "ECM" if r["gene"] in ecm_genes else "HK" if r["gene"] in hk_genes else \
              "signal" if r["gene"] in signal_genes else "other"
        _log(f"  {r['gene']:<12s} {r['median_cross_rho']:>+10.3f} {r['pct_sig']:>5.0%} {cat:>12s}")

    # ══════════════════════════════════════════════════════════════════
    # Q2: Age-stratified cross-tissue coordination
    # ══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Q2] Age-stratified cross-tissue COL1A1")
    _log("=" * 70)

    ecm_focus = ["COL1A1", "COL3A1", "FN1", "ELN"]
    age_bins_cross = {"young": (20, 45), "old": (55, 80)}

    t1_name, t2_name = focus_tissues[0], focus_tissues[1]
    _log(f"  Focus tissue pair: {t1_name} × {t2_name}")

    age_cross_results = []
    for gene in ecm_focus:
        if gene not in merged.columns:
            continue
        for age_label, (lo, hi) in age_bins_cross.items():
            d1 = merged[(merged["SMTSD"] == t1_name) & (merged["age_mid"] >= lo) & (merged["age_mid"] <= hi)]
            d2 = merged[(merged["SMTSD"] == t2_name) & (merged["age_mid"] >= lo) & (merged["age_mid"] <= hi)]
            d1s = d1[["SUBJID", gene]].set_index("SUBJID")
            d2s = d2[["SUBJID", gene]].set_index("SUBJID")
            common = d1s.index.intersection(d2s.index)
            if len(common) < 20:
                continue
            rho, p = spearman_safe(d1s.loc[common, gene].values.astype(float),
                                    d2s.loc[common, gene].values.astype(float))
            age_cross_results.append({"gene": gene, "age": age_label, "n": len(common),
                                       "rho": rho, "p": p})
            _log(f"    {gene:<10s} {age_label:>5s} (n={len(common):3d}): ρ = {rho:+.3f} (p={p:.2e})")

    # All tissue pairs, COL1A1 only, age-stratified
    _log(f"\n  COL1A1 cross-tissue by age (all tissue pairs):")
    age_rho_young, age_rho_old = [], []
    for i, t1 in enumerate(focus_tissues):
        for t2 in focus_tissues[i+1:]:
            for age_label, (lo, hi) in age_bins_cross.items():
                d1 = merged[(merged["SMTSD"] == t1) & (merged["age_mid"] >= lo) & (merged["age_mid"] <= hi)]
                d2 = merged[(merged["SMTSD"] == t2) & (merged["age_mid"] >= lo) & (merged["age_mid"] <= hi)]
                d1s = d1[["SUBJID", "COL1A1"]].set_index("SUBJID")
                d2s = d2[["SUBJID", "COL1A1"]].set_index("SUBJID")
                common = d1s.index.intersection(d2s.index)
                if len(common) < 20:
                    continue
                rho, _ = spearman_safe(d1s.loc[common, "COL1A1"].values.astype(float),
                                        d2s.loc[common, "COL1A1"].values.astype(float))
                if not np.isnan(rho):
                    if age_label == "young":
                        age_rho_young.append(rho)
                    else:
                        age_rho_old.append(rho)

    if age_rho_young and age_rho_old:
        _log(f"    Young median ρ = {np.median(age_rho_young):+.3f} (n={len(age_rho_young)} pairs)")
        _log(f"    Old   median ρ = {np.median(age_rho_old):+.3f} (n={len(age_rho_old)} pairs)")
        u, p = stats.mannwhitneyu(age_rho_young, age_rho_old)
        _log(f"    Wilcoxon p = {p:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # Q3: Sex-stratified cross-tissue
    # ══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Q3] Sex-stratified cross-tissue COL1A1")
    _log("=" * 70)

    sex_rho = {"male": [], "female": []}
    for i, t1 in enumerate(focus_tissues):
        for t2 in focus_tissues[i+1:]:
            for sex_val in ["male", "female"]:
                d1 = merged[(merged["SMTSD"] == t1) & (merged["sex"] == sex_val)]
                d2 = merged[(merged["SMTSD"] == t2) & (merged["sex"] == sex_val)]
                d1s = d1[["SUBJID", "COL1A1"]].set_index("SUBJID")
                d2s = d2[["SUBJID", "COL1A1"]].set_index("SUBJID")
                common = d1s.index.intersection(d2s.index)
                if len(common) < 20:
                    continue
                rho, _ = spearman_safe(d1s.loc[common, "COL1A1"].values.astype(float),
                                        d2s.loc[common, "COL1A1"].values.astype(float))
                if not np.isnan(rho):
                    sex_rho[sex_val].append(rho)

    for sex_val in ["male", "female"]:
        if sex_rho[sex_val]:
            _log(f"  {sex_val}: median cross-tissue ρ = {np.median(sex_rho[sex_val]):+.3f} "
                 f"(n={len(sex_rho[sex_val])} pairs)")

    # ══════════════════════════════════════════════════════════════════
    # Q4: Partial correlation controlling for age
    # ══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Q4] COL1A1 cross-tissue — partial correlation (controlling for age)")
    _log("=" * 70)

    partial_results = []
    for i, t1 in enumerate(focus_tissues[:4]):
        for t2 in focus_tissues[:4]:
            if t1 >= t2:
                continue
            d1 = merged[merged["SMTSD"] == t1][["SUBJID", "COL1A1", "age_mid"]].set_index("SUBJID")
            d2 = merged[merged["SMTSD"] == t2][["SUBJID", "COL1A1"]].set_index("SUBJID")
            common = d1.index.intersection(d2.index)
            if len(common) < 50:
                continue

            raw_rho, raw_p = spearman_safe(d1.loc[common, "COL1A1"].values.astype(float),
                                            d2.loc[common, "COL1A1"].values.astype(float))
            part_rho, part_p = partial_spearman(
                d1.loc[common, "COL1A1"].values.astype(float),
                d2.loc[common, "COL1A1"].values.astype(float),
                d1.loc[common, "age_mid"].values.astype(float))

            t1s = t1.split(" - ")[-1][:15]
            t2s = t2.split(" - ")[-1][:15]
            _log(f"  {t1s:>15s} × {t2s:<15s}: raw ρ={raw_rho:+.3f}, partial(|age) ρ={part_rho:+.3f}")
            partial_results.append({"t1": t1, "t2": t2, "n": len(common),
                                     "raw_rho": raw_rho, "partial_rho": part_rho})

    if partial_results:
        raw_med = np.median([r["raw_rho"] for r in partial_results])
        part_med = np.median([r["partial_rho"] for r in partial_results])
        _log(f"\n  Median raw ρ = {raw_med:+.3f}, median partial(|age) ρ = {part_med:+.3f}")
        _log(f"  → Age explains {1 - part_med/raw_med:.0%} of cross-tissue coordination")

    # ══════════════════════════════════════════════════════════════════
    # Q5: Cross-tissue coordination — ECM vs non-ECM comparison
    # ══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Q5] ECM vs non-ECM cross-tissue coordination")
    _log("=" * 70)

    ecm_rhos = df_gc[df_gc["gene"].isin(ecm_genes)]["median_cross_rho"].values
    hk_rhos = df_gc[df_gc["gene"].isin(hk_genes)]["median_cross_rho"].values
    sig_rhos = df_gc[df_gc["gene"].isin(signal_genes)]["median_cross_rho"].values

    _log(f"  ECM genes (n={len(ecm_rhos)}):     median cross-tissue ρ = {np.median(ecm_rhos):+.3f}")
    _log(f"  HK genes (n={len(hk_rhos)}):       median cross-tissue ρ = {np.median(hk_rhos):+.3f}")
    _log(f"  Signal genes (n={len(sig_rhos)}):   median cross-tissue ρ = {np.median(sig_rhos):+.3f}")

    if len(ecm_rhos) >= 3 and len(hk_rhos) >= 3:
        u, p = stats.mannwhitneyu(ecm_rhos, hk_rhos, alternative="greater")
        _log(f"  ECM > HK: Wilcoxon p = {p:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # Figure
    # ══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel A: Gene ranking by cross-tissue ρ
    ax = axes[0, 0]
    df_sorted = df_gc.sort_values("median_cross_rho", ascending=True)
    colors = []
    for g in df_sorted["gene"]:
        if g in ecm_genes: colors.append("tab:green")
        elif g in hk_genes: colors.append("tab:gray")
        elif g in signal_genes: colors.append("tab:orange")
        else: colors.append("tab:purple")
    ax.barh(range(len(df_sorted)), df_sorted["median_cross_rho"].values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["gene"].values, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Median cross-tissue Spearman ρ")
    ax.set_title("A: Cross-tissue coordination by gene\nGreen=ECM, Gray=HK, Orange=Signal")

    # Panel B: ECM coordination heatmap
    ax = axes[0, 1]
    ecm_available = [g for g in ecm_focus if g in merged.columns]
    if ecm_available:
        ecm_matrix = np.full((len(ecm_available), len(focus_tissues[:5])), np.nan)
        for gi, gene in enumerate(ecm_available):
            for ti, tissue in enumerate(focus_tissues[:5]):
                rhos_t = []
                for t2 in focus_tissues[:5]:
                    if tissue == t2:
                        continue
                    d1 = merged[merged["SMTSD"] == tissue][["SUBJID", gene]].set_index("SUBJID")
                    d2 = merged[merged["SMTSD"] == t2][["SUBJID", gene]].set_index("SUBJID")
                    common = d1.index.intersection(d2.index)
                    if len(common) < 30:
                        continue
                    rho, _ = spearman_safe(d1.loc[common, gene].values.astype(float),
                                            d2.loc[common, gene].values.astype(float))
                    if not np.isnan(rho):
                        rhos_t.append(rho)
                if rhos_t:
                    ecm_matrix[gi, ti] = np.median(rhos_t)

        im = ax.imshow(ecm_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.6)
        ax.set_yticks(range(len(ecm_available)))
        ax.set_yticklabels(ecm_available, fontsize=8)
        tissue_labels = [t.split(" - ")[-1][:15] for t in focus_tissues[:5]]
        ax.set_xticks(range(len(tissue_labels)))
        ax.set_xticklabels(tissue_labels, rotation=45, ha="right", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Median cross-tissue ρ")
        ax.set_title("B: ECM genes × tissue\ncross-tissue coordination")

    # Panel C: Age-stratified COL1A1 cross-tissue
    ax = axes[0, 2]
    if age_rho_young and age_rho_old:
        ax.boxplot([age_rho_young, age_rho_old], labels=["Young (20-45)", "Old (55-79)"])
        ax.set_ylabel("Cross-tissue COL1A1 ρ")
        ax.set_title("C: COL1A1 coordination by age")
        ax.axhline(0, color="black", linewidth=0.3, linestyle="--")

    # Panel D: Sex-stratified
    ax = axes[1, 0]
    if sex_rho["male"] and sex_rho["female"]:
        ax.boxplot([sex_rho["male"], sex_rho["female"]], labels=["Male", "Female"])
        ax.set_ylabel("Cross-tissue COL1A1 ρ")
        ax.set_title("D: COL1A1 coordination by sex")

    # Panel E: Scatter — COL1A1 adipose vs skin
    ax = axes[1, 1]
    t1_pick = [t for t in focus_tissues if "Subcutaneous" in t]
    t2_pick = [t for t in focus_tissues if "Sun Exposed" in t or "Skin" in t]
    if t1_pick and t2_pick and "COL1A1" in merged.columns:
        d1 = merged[merged["SMTSD"] == t1_pick[0]][["SUBJID", "COL1A1", "age_mid"]].set_index("SUBJID")
        d2 = merged[merged["SMTSD"] == t2_pick[0]][["SUBJID", "COL1A1"]].set_index("SUBJID")
        common = d1.index.intersection(d2.index)
        if len(common) > 20:
            ages = d1.loc[common, "age_mid"].values
            scatter = ax.scatter(d1.loc[common, "COL1A1"].values.astype(float),
                                  d2.loc[common, "COL1A1"].values.astype(float),
                                  c=ages, cmap="coolwarm", s=10, alpha=0.5, edgecolors="none")
            rho, p = spearman_safe(d1.loc[common, "COL1A1"].values.astype(float),
                                    d2.loc[common, "COL1A1"].values.astype(float))
            ax.set_xlabel(f"COL1A1 — {t1_pick[0].split(' - ')[-1][:20]}")
            ax.set_ylabel(f"COL1A1 — {t2_pick[0].split(' - ')[-1][:20]}")
            ax.set_title(f"E: Same donor, different tissues\nρ = {rho:+.3f} (n={len(common)})")
            plt.colorbar(scatter, ax=ax, label="Age", shrink=0.7)

    # Panel F: Raw vs age-corrected
    ax = axes[1, 2]
    if partial_results:
        raw = [r["raw_rho"] for r in partial_results]
        part = [r["partial_rho"] for r in partial_results]
        ax.scatter(raw, part, s=40, alpha=0.7, edgecolors="gray", linewidth=0.3)
        for r in partial_results:
            t1s = r["t1"].split(" - ")[-1][:6]
            t2s = r["t2"].split(" - ")[-1][:6]
            ax.annotate(f"{t1s}×{t2s}", (r["raw_rho"], r["partial_rho"]), fontsize=5)
        lim = [min(min(raw), min(part)) - 0.05, max(max(raw), max(part)) + 0.05]
        ax.plot(lim, lim, "k--", alpha=0.3)
        ax.set_xlabel("Raw cross-tissue ρ")
        ax.set_ylabel("Partial ρ (controlling for age)")
        ax.set_title("F: Age explains how much?\n(below diagonal = age-driven)")

    fig.suptitle("COL1A1 Systemic Coordination — Deep Dive\nGTEx, 948 donors, 6 tissues",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "col1a1_deep.png", dpi=150)
    plt.close(fig)
    _log(f"\n  Saved col1a1_deep.png")
    _log(f"  Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
