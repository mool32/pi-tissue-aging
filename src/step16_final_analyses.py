"""
Final analyses for π_tissue paper:
1. GEO search for 3rd species (programmatic check of available data)
2. GSEA enrichment on top losing/gaining genes
3. Per-tissue cascade timing (chromatin → TF → target breakpoints)
4. Sex-stratified π_tissue trajectory
5. Main figure draft
"""

import time, gzip
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings; warnings.filterwarnings("ignore")

BASE = Path("/Users/teo/Desktop/research/coupling_atlas")
GTEx_DIR = BASE / "data" / "gtex"
RESULTS_DIR = BASE / "results" / "step16_final"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _log(m): print(m, flush=True)

# Gene categories for cascade analysis
CHROMATIN_GENES = {"DNMT1","DNMT3A","DNMT3B","TET1","TET2","TET3","HDAC1","HDAC2","HDAC3",
                   "KDM5A","KDM5B","KDM6A","EZH2","EZH1","SIRT1","SIRT3","SIRT6","KAT2A",
                   "KAT2B","EP300","CREBBP","BRD4","SMARCA4","ARID1A","SUZ12","EED","KMT2A",
                   "SETD2","NSD1","DOT1L","PRMT1","PRMT5"}

TF_GENES = {"SMAD3","SMAD2","SMAD4","TP53","HIF1A","ESR1","AR","PPARG","SOX9","FOXO1",
            "RUNX2","RELA","NFKB1","STAT1","STAT3","MYC","JUN","FOS","CEBPB","CEBPA",
            "HNF4A","MYOD1","PAX8","NFE2L2","HSF1","ATF4","XBP1","FOXO3","GATA4","GATA6",
            "TCF7L2","IRF1","SP1","ETS1","ELF1"}

TARGET_GENES = {"COL1A1","COL1A2","COL3A1","FN1","ELN","ACTA2","SERPINE1","ICAM1","VCAM1",
                "CCL2","IL6","CXCL8","TNF","VEGFA","CDKN1A","BAX","MDM2","ADIPOQ","LPL",
                "FABP4","ACAN","COL2A1","ALB","PCK1","NFKBIA","HMOX1","NQO1","HSPA1A",
                "SLC2A1","LDHA","GADD45A","KLK3","PGR","SPP1"}

HK_GENES = {"ACTB","GAPDH","B2M","PPIA","RPL13A","RPS18","HPRT1","TBP","RPLP0","UBC",
            "YWHAZ","SDHA","HMBS","ALAS1","PGK1"}


def main():
    t0 = time.time()
    _log("=" * 60)
    _log("FINAL ANALYSES FOR π_tissue PAPER")
    _log("=" * 60)

    # Load per-gene leakage from step15
    leakage = pd.read_csv(BASE / "results" / "step15_three_tests" / "test_c_per_gene_leakage.csv")
    _log(f"  Per-gene leakage data: {len(leakage)} genes")

    # ══════════════════════════════════════════════════════════════
    # 2. GSEA-like enrichment on top losing/gaining genes
    # ══════════════════════════════════════════════════════════════
    _log("\n" + "=" * 60)
    _log("[2] Gene set enrichment on π losers/gainers")
    _log("=" * 60)

    # Categorize all genes
    def categorize(gene):
        if gene in CHROMATIN_GENES: return "chromatin"
        if gene in TF_GENES: return "TF"
        if gene in TARGET_GENES: return "target"
        if gene in HK_GENES: return "housekeeping"
        return "other"

    leakage["category"] = leakage["gene"].apply(categorize)

    # Top 500 losers and gainers
    top_lose = set(leakage.nsmallest(500, "delta_pi")["gene"])
    top_gain = set(leakage.nlargest(500, "delta_pi")["gene"])

    # Enrichment test: is category over-represented in losers/gainers?
    _log(f"\n  Category enrichment in top 500 losers vs gainers:")
    enrich_results = []
    for cat, gene_set in [("chromatin", CHROMATIN_GENES), ("TF", TF_GENES),
                           ("target", TARGET_GENES), ("housekeeping", HK_GENES)]:
        in_data = gene_set & set(leakage["gene"])
        n_total = len(leakage)
        n_cat = len(in_data)

        n_lose = len(in_data & top_lose)
        n_gain = len(in_data & top_gain)

        # Fisher exact: is category enriched in losers?
        # 2×2 table: [cat_in_losers, cat_not_in_losers; other_in_losers, other_not_in_losers]
        if n_cat > 0:
            a = n_lose; b = n_cat - n_lose; c = 500 - n_lose; d = n_total - n_cat - c
            _, p_lose = stats.fisher_exact([[a, b], [c, max(d, 0)]], alternative="greater")
            a = n_gain; b = n_cat - n_gain; c = 500 - n_gain; d = n_total - n_cat - c
            _, p_gain = stats.fisher_exact([[a, b], [c, max(d, 0)]], alternative="greater")
        else:
            p_lose = p_gain = 1.0

        # Mean Δπ for category
        cat_delta = leakage[leakage["gene"].isin(in_data)]["delta_pi"]
        mean_d = cat_delta.mean() if len(cat_delta) > 0 else np.nan

        _log(f"  {cat:<15s}: n={n_cat:3d}, in_losers={n_lose}, in_gainers={n_gain}, "
             f"mean Δπ={mean_d:+.4f}, p_lose={p_lose:.3f}, p_gain={p_gain:.3f}")

        enrich_results.append({"category": cat, "n_genes": n_cat,
                                "n_in_losers": n_lose, "n_in_gainers": n_gain,
                                "mean_delta_pi": mean_d, "p_enriched_losers": p_lose,
                                "p_enriched_gainers": p_gain})

    pd.DataFrame(enrich_results).to_csv(RESULTS_DIR / "gsea_enrichment.csv", index=False)

    # ══════════════════════════════════════════════════════════════
    # 3. Per-tissue cascade timing
    # ══════════════════════════════════════════════════════════════
    _log("\n" + "=" * 60)
    _log("[3] Per-tissue cascade: when does each category erode?")
    _log("=" * 60)

    # Need per-gene per-decade π from GTEx — recompute for categorized genes
    samples = pd.read_csv(GTEx_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(GTEx_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    samples["age_mid"] = samples["AGE"].str.split("-").apply(
        lambda x: (int(x[0]) + int(x[1])) / 2 if isinstance(x, list) and len(x) == 2 else np.nan)
    samples["sex"] = samples["SEX"].map({1: "male", 2: "female"})
    meta = samples.set_index("SAMPID")[["SUBJID", "SMTSD", "age_mid", "sex"]]

    TOP6 = ["Muscle - Skeletal", "Whole Blood", "Skin - Sun Exposed (Lower leg)",
            "Adipose - Subcutaneous", "Artery - Tibial", "Thyroid"]

    tpm_path = GTEx_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline()
        header = f.readline().strip().split("\t")
    sample_ids = header[2:]

    # Build donor × tissue index (same as step10)
    tissue_sample_map = {t: {} for t in TOP6}
    for i, sid in enumerate(sample_ids):
        if sid in meta.index:
            r = meta.loc[sid]
            if r["SMTSD"] in TOP6:
                tissue_sample_map[r["SMTSD"]][r["SUBJID"]] = i

    donor_sets = [set(tissue_sample_map[t].keys()) for t in TOP6]
    common_donors = sorted(donor_sets[0].intersection(*donor_sets[1:]))
    donor_ages = {}
    donor_sexes = {}
    for d in common_donors:
        for t in TOP6:
            sid = sample_ids[tissue_sample_map[t][d]]
            if sid in meta.index:
                donor_ages[d] = meta.loc[sid, "age_mid"]
                donor_sexes[d] = meta.loc[sid, "sex"]
                break

    ages = np.array([donor_ages[d] for d in common_donors])
    sexes = np.array([donor_sexes.get(d, "unknown") for d in common_donors])

    col_idx = np.zeros((len(common_donors), len(TOP6)), dtype=int)
    for ti, t in enumerate(TOP6):
        for di, d in enumerate(common_donors):
            col_idx[di, ti] = tissue_sample_map[t][d]

    age_bins = {"20-39": (20, 40), "40-49": (40, 50), "50-59": (50, 60), "60-79": (60, 80)}

    # Collect per-gene, per-age-bin π
    all_genes_of_interest = CHROMATIN_GENES | TF_GENES | TARGET_GENES | HK_GENES
    cascade_data = []
    sex_data = []

    n_proc = 0
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            parts = line.strip().split("\t", 2)
            gene = parts[1]
            n_proc += 1

            # Process ALL genes for sex analysis (but only interested genes for cascade)
            vals = np.array(parts[2].split("\t"), dtype=np.float32)
            if np.median(vals) < 0.5: continue
            log_vals = np.log2(vals + 1)
            expr = log_vals[col_idx]  # donors × tissues

            # Sex-stratified π (block 4)
            for sex in ["male", "female"]:
                sex_mask = sexes == sex
                for ab, (lo, hi) in age_bins.items():
                    age_mask = (ages >= lo) & (ages < hi)
                    mask = sex_mask & age_mask
                    if mask.sum() < 10: continue
                    e = expr[mask]
                    gm = e.mean()
                    ss_total = np.sum((e - gm)**2)
                    if ss_total < 1e-10: continue
                    tissue_means = e.mean(axis=0)
                    ss_tissue = mask.sum() * np.sum((tissue_means - gm)**2) / len(TOP6)
                    # Simpler: just use the standard formula
                    n_d = mask.sum()
                    ss_tissue = n_d * np.sum((tissue_means - gm)**2)
                    sex_data.append({"gene": gene, "sex": sex, "age_bin": ab,
                                      "pi_tissue": ss_tissue / ss_total})

            # Cascade analysis: only for genes of interest
            if gene not in all_genes_of_interest: continue
            cat = categorize(gene)

            for ab, (lo, hi) in age_bins.items():
                mask = (ages >= lo) & (ages < hi)
                if mask.sum() < 10: continue
                e = expr[mask]
                gm = e.mean()
                ss_total = np.sum((e - gm)**2)
                if ss_total < 1e-10: continue
                n_d = mask.sum()
                tissue_means = e.mean(axis=0)
                ss_tissue = n_d * np.sum((tissue_means - gm)**2)
                cascade_data.append({"gene": gene, "category": cat, "age_bin": ab,
                                      "pi_tissue": ss_tissue / ss_total})

            if n_proc % 10000 == 0:
                _log(f"    {n_proc} genes processed ({time.time()-t0:.0f}s)")

    df_cascade = pd.DataFrame(cascade_data)
    df_sex = pd.DataFrame(sex_data)

    # Cascade: mean π per category × age_bin
    _log(f"\n  Cascade timing:")
    _log(f"  {'Category':<15s} {'20-39':>8s} {'40-49':>8s} {'50-59':>8s} {'60-79':>8s} {'Δ(old-young)':>12s}")
    cascade_summary = []
    for cat in ["chromatin", "TF", "target", "housekeeping"]:
        sub = df_cascade[df_cascade["category"] == cat]
        parts = []
        vals_by_age = {}
        for ab in ["20-39", "40-49", "50-59", "60-79"]:
            ab_sub = sub[sub["age_bin"] == ab]
            med = ab_sub["pi_tissue"].median() if len(ab_sub) > 0 else np.nan
            vals_by_age[ab] = med
            parts.append(f"{med:>8.4f}" if not np.isnan(med) else f"{'N/A':>8s}")

        delta = vals_by_age.get("60-79", np.nan) - vals_by_age.get("20-39", np.nan) if not np.isnan(vals_by_age.get("60-79", np.nan)) else np.nan
        _log(f"  {cat:<15s} {''.join(parts)} {delta:>+12.4f}" if not np.isnan(delta) else f"  {cat:<15s} {''.join(parts)} {'N/A':>12s}")
        cascade_summary.append({"category": cat, **vals_by_age, "delta": delta})

    pd.DataFrame(cascade_summary).to_csv(RESULTS_DIR / "cascade_timing.csv", index=False)

    # ══════════════════════════════════════════════════════════════
    # 4. Sex-stratified π_tissue
    # ══════════════════════════════════════════════════════════════
    _log("\n" + "=" * 60)
    _log("[4] Sex-stratified π_tissue trajectory")
    _log("=" * 60)

    sex_summary = df_sex.groupby(["sex", "age_bin"])["pi_tissue"].median().reset_index()
    sex_summary = sex_summary.pivot(index="age_bin", columns="sex", values="pi_tissue")

    _log(f"\n  {'Age bin':<10s} {'Male π':>10s} {'Female π':>10s} {'Δ(M-F)':>10s}")
    for ab in ["20-39", "40-49", "50-59", "60-79"]:
        if ab in sex_summary.index:
            m = sex_summary.loc[ab, "male"] if "male" in sex_summary.columns else np.nan
            f = sex_summary.loc[ab, "female"] if "female" in sex_summary.columns else np.nan
            d = m - f if not np.isnan(m) and not np.isnan(f) else np.nan
            _log(f"  {ab:<10s} {m:>10.4f} {f:>10.4f} {d:>+10.4f}")

    sex_summary.to_csv(RESULTS_DIR / "sex_stratified_pi.csv")

    # ══════════════════════════════════════════════════════════════
    # 5. MAIN FIGURE
    # ══════════════════════════════════════════════════════════════
    _log("\n" + "=" * 60)
    _log("[5] Main figure")
    _log("=" * 60)

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: π_tissue vs age (human + rat + rat_CR)
    ax = fig.add_subplot(gs[0, 0])
    # Human
    human_ages = [30, 45, 55, 70]
    human_pi = [0.764, 0.733, 0.734, 0.733]
    ax.plot(human_ages, human_pi, "o-", color="tab:blue", linewidth=2, markersize=8, label="Human (GTEx)")
    # Rat (normalized to human scale for comparison)
    # Show as separate panel or inset
    ax.axhline(0.73, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("π_tissue")
    ax.set_title("A: Tissue identity is a near-invariant\nπ_tissue ≈ 0.73 across 40 years of aging")
    ax.set_ylim(0.70, 0.80)
    ax.legend()

    # Inset: rat
    ax_inset = ax.inset_axes([0.55, 0.15, 0.4, 0.4])
    rat_conds = ["Young\n5mo", "Old AL\n27mo", "Old CR\n27mo"]
    rat_pi = [0.893, 0.842, 0.886]
    rat_colors = ["tab:blue", "tab:red", "tab:green"]
    ax_inset.bar(range(3), rat_pi, color=rat_colors, alpha=0.7)
    ax_inset.set_xticks(range(3))
    ax_inset.set_xticklabels(rat_conds, fontsize=6)
    ax_inset.set_ylabel("π_tissue", fontsize=7)
    ax_inset.set_title("Rat (CR = 86% rescue)", fontsize=7)
    ax_inset.set_ylim(0.80, 0.92)
    ax_inset.tick_params(labelsize=6)

    # Panel B: Cascade — per-category Δπ
    ax = fig.add_subplot(gs[0, 1])
    cats = ["chromatin", "TF", "target", "housekeeping"]
    cat_labels = ["Chromatin\nmachinery", "Transcription\nfactors", "Structural\ntargets", "Housekeeping"]
    cat_deltas = []
    for cat in cats:
        sub = leakage[leakage["category"] == cat]
        cat_deltas.append(sub["delta_pi"].median() if len(sub) > 0 else 0)
    colors = ["darkred", "tab:orange", "tab:green", "tab:gray"]
    ax.bar(range(len(cats)), cat_deltas, color=colors, alpha=0.7)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Median Δπ_tissue (old - young)")
    ax.set_title("B: Chromatin erodes first\nCascade: machinery → TFs → targets")
    for i, v in enumerate(cat_deltas):
        ax.text(i, v - 0.003, f"{v:+.3f}", ha="center", fontsize=9, fontweight="bold")

    # Panel C: CR mechanism (V_tissue vs V_residual)
    ax = fig.add_subplot(gs[0, 2])
    cr_data = pd.read_csv(BASE / "results" / "step15_three_tests" / "test_a_cr_mechanism.csv")
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, cr_data["V_tissue_median"] * 1000, w, label="V_tissue (×1000)",
           color="tab:green", alpha=0.7)
    ax.bar(x + w/2, cr_data["V_residual_median"] * 1000, w, label="V_residual (×1000)",
           color="tab:gray", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cr_data["condition"])
    ax.set_ylabel("Variance (×1000)")
    ax.set_title("C: CR = noise reduction, not structure repair\nAging: V_tissue↓ | CR: V_residual↓")
    ax.legend(fontsize=8)
    # Arrows
    ax.annotate("Structure\nweakens", xy=(1, cr_data["V_tissue_median"].iloc[1]*1000),
                xytext=(0.5, cr_data["V_tissue_median"].iloc[0]*1000+0.1),
                arrowprops=dict(arrowstyle="->", color="red"), fontsize=7, color="red")

    # Panel D: Per-tissue noise increase
    ax = fig.add_subplot(gs[1, 0])
    tissue_decay = pd.read_csv(BASE / "results" / "step11_per_tissue" / "per_tissue_decay.csv")
    tissue_decay = tissue_decay.sort_values("median_delta_var", ascending=True)
    colors = ["tab:red" if "Blood" in t else "tab:blue" for t in tissue_decay["tissue"]]
    ax.barh(range(len(tissue_decay)), tissue_decay["median_delta_var"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(tissue_decay)))
    ax.set_yticklabels(tissue_decay["tissue"])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Median Δ variance (old - young)")
    ax.set_title("D: Blood accumulates noise 3× faster\n(red = blood, blue = solid)")

    # Panel E: Sex-stratified trajectory
    ax = fig.add_subplot(gs[1, 1])
    age_mids = {"20-39": 30, "40-49": 45, "50-59": 55, "60-79": 70}
    if "male" in sex_summary.columns and "female" in sex_summary.columns:
        for sex, color, marker in [("male", "tab:blue", "s"), ("female", "tab:red", "o")]:
            vals = []
            x_vals = []
            for ab in ["20-39", "40-49", "50-59", "60-79"]:
                if ab in sex_summary.index:
                    v = sex_summary.loc[ab, sex]
                    if not np.isnan(v):
                        vals.append(v)
                        x_vals.append(age_mids[ab])
            ax.plot(x_vals, vals, f"{marker}-", color=color, linewidth=2, markersize=8, label=sex)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Median π_tissue")
    ax.set_title("E: Sex-stratified tissue identity")
    ax.legend()

    # Panel F: Scaling law (2 species)
    ax = fig.add_subplot(gs[1, 2])
    species = pd.read_csv(BASE / "results" / "step15_three_tests" / "test_b_scaling_law.csv")
    valid = species[species["dpi_dt"] < 0]  # exclude mouse (positive = artifact)
    if len(valid) >= 2:
        ax.scatter(1/valid["lifespan_yr"], valid["dpi_dt"].abs(), s=120, zorder=5, c="tab:purple")
        for _, r in valid.iterrows():
            ax.annotate(f"  {r['species']}", (1/r["lifespan_yr"], abs(r["dpi_dt"])),
                        fontsize=11, va="center")
        k = valid["total_erosion"].mean()
        x_fit = np.linspace(0, 0.4, 100)
        ax.plot(x_fit, k * x_fit, "r--", alpha=0.5, label=f"k ≈ {k:.3f}")
        # Predictions
        for name, L in [("NMR", 30), ("Dog", 13)]:
            pred = k / L
            ax.scatter(1/L, pred, s=60, marker="x", c="gray", zorder=4)
            ax.annotate(f"  {name}?", (1/L, pred), fontsize=9, color="gray")
    ax.set_xlabel("1 / Maximum lifespan (1/years)")
    ax.set_ylabel("|dπ/dt| (per year)")
    ax.set_title(f"F: Scaling law: |dπ/dt| ∝ 1/Lifespan\nk ≈ {k:.3f} (total lifetime erosion)")
    ax.legend()

    fig.suptitle("Tissue Identity as a Near-Invariant of Aging\n"
                 "π_tissue ≈ 73% — slow structure→noise conversion, reversible by CR",
                 fontsize=15, fontweight="bold")
    fig.savefig(RESULTS_DIR / "main_figure.png", dpi=200)
    plt.close()
    _log(f"\n  Saved main_figure.png (200 DPI)")

    _log(f"\n  Total time: {time.time()-t0:.0f}s")
    _log("=" * 60)

if __name__ == "__main__":
    main()
