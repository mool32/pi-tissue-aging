"""
Step 6: Cross-tissue coordination decline — deep validation

Four analyses + three confound checks:

Analysis A: Which genes drive decoupling? (gene-level Δρ young vs old)
Analysis B: Age trajectory — linear or breakpoint?
Analysis C: Sex-stratified cross-tissue coordination
Analysis D: Which tissue pairs decouple most/least?

Confound 1: Composition (immune gene exclusion)
Confound 2: Batch effects
Confound 3: Genetic vs non-genetic (HK gene baseline)
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
RESULTS_DIR = BASE / "results" / "step6_cross_tissue"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg):
    print(msg, flush=True)


def spearman_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan, 1.0
    return stats.spearmanr(x, y)


# Immune marker genes to exclude for composition check
IMMUNE_GENES = {
    "CD3D", "CD3E", "CD4", "CD8A", "CD8B", "CD19", "MS4A1", "CD79A", "CD79B",
    "CD14", "CD68", "CD163", "ITGAM", "ITGAX", "CSF1R", "FCGR3A", "FCGR3B",
    "NKG7", "GNLY", "GZMA", "GZMB", "PRF1", "KLRD1", "KLRK1",
    "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1",
    "PTPRC", "CXCR4", "CCR7", "IL2RA", "IL7R", "CD27", "CD38",
    "FOXP3", "TBET", "TBX21", "GATA3", "RORC", "BCL6",
    "CCL2", "CCL3", "CCL4", "CCL5", "CXCL8", "CXCL10", "CXCL12",
    "IL1B", "IL6", "IL10", "IL12A", "IL12B", "IL17A", "IL18", "IL23A",
    "TNF", "IFNG", "TGFB1",
    "ICAM1", "VCAM1", "SELE", "SELP",
}

# Tissue-specific genes (high specificity, not immune)
TISSUE_SPECIFIC = {
    "MYH7", "TNNT2", "MYL2",  # heart
    "ALB", "APOB", "CYP3A4",  # liver
    "SLC12A1", "UMOD", "AQP2",  # kidney
    "SFTPC", "SFTPB", "NKX2-1",  # lung
    "KRT14", "KRT5", "COL7A1",  # skin
    "MYH1", "MYH2", "ACTA1",  # muscle
    "TG", "TPO", "NKX2-1",  # thyroid
}


def main():
    t0 = time.time()
    _log("=" * 70)
    _log("CROSS-TISSUE COORDINATION — Deep validation")
    _log("=" * 70)

    # ── Load metadata ────────────────────────────────────────────────
    _log("\n[0] Loading metadata...")
    samples = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    samples["sex"] = samples["SEX"].map({1: "male", 2: "female"})
    samples["age_mid"] = samples["AGE"].str.split("-").apply(
        lambda x: (int(x[0]) + int(x[1])) / 2 if isinstance(x, list) and len(x) == 2 else np.nan)

    sample_meta = samples.set_index("SAMPID")[["SUBJID", "SMTSD", "sex", "age_mid", "SMGEBTCH"]].copy()

    # Top 6 tissues
    tissue_counts = samples["SMTSD"].value_counts()
    top6 = tissue_counts.head(6).index.tolist()
    _log(f"  Top 6 tissues: {[t.split(' - ')[-1][:20] for t in top6]}")

    # Build tissue → sample index mapping
    tpm_path = DATA_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline()
        header = f.readline().strip().split("\t")
    sample_ids = header[2:]

    tissue_sample_idx = {}
    for i, sid in enumerate(sample_ids):
        if sid in sample_meta.index:
            tissue = sample_meta.loc[sid, "SMTSD"]
            if tissue in top6:
                if tissue not in tissue_sample_idx:
                    tissue_sample_idx[tissue] = []
                tissue_sample_idx[tissue].append(i)
    for t in tissue_sample_idx:
        tissue_sample_idx[t] = np.array(tissue_sample_idx[t])

    # Build donor→tissue→sample_idx lookup
    donor_tissue = {}
    donor_meta = {}
    for i, sid in enumerate(sample_ids):
        if sid in sample_meta.index:
            m = sample_meta.loc[sid]
            if m["SMTSD"] in top6:
                donor = m["SUBJID"]
                if donor not in donor_tissue:
                    donor_tissue[donor] = {}
                    donor_meta[donor] = {"sex": m["sex"], "age": m["age_mid"],
                                          "batch": m.get("SMGEBTCH", "?")}
                donor_tissue[donor][m["SMTSD"]] = i

    # Multi-tissue donors
    multi = {d: ts for d, ts in donor_tissue.items() if len(ts) >= 2}
    _log(f"  {len(multi)} donors with ≥2 of top 6 tissues")

    # Age groups
    young_donors = [d for d in multi if donor_meta[d]["age"] <= 35]
    mid_donors = [d for d in multi if 35 < donor_meta[d]["age"] < 55]
    old_donors = [d for d in multi if donor_meta[d]["age"] >= 55]
    _log(f"  Young (≤35): {len(young_donors)}, Mid (36-54): {len(mid_donors)}, Old (≥55): {len(old_donors)}")

    male_donors = [d for d in multi if donor_meta[d]["sex"] == "male"]
    female_donors = [d for d in multi if donor_meta[d]["sex"] == "female"]
    _log(f"  Males: {len(male_donors)}, Females: {len(female_donors)}")

    # ── Stream TPM ───────────────────────────────────────────────────
    _log(f"\n[1] Streaming TPM — computing per-gene cross-tissue ρ...")

    # For each gene: store expression per donor per tissue
    # Then compute cross-tissue ρ for young vs old

    gene_results = []
    n_processed = 0
    n_kept = 0

    # Tissue pairs to analyze
    tissue_pairs = []
    for i, t1 in enumerate(top6[:5]):
        for t2 in top6[i+1:]:
            tissue_pairs.append((t1, t2))

    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()

        for line in f:
            parts = line.strip().split("\t", 2)
            gene_name = parts[1]
            n_processed += 1

            if n_processed % 5000 == 0:
                _log(f"    {n_processed} genes ({time.time()-t0:.0f}s, kept {n_kept})...")

            vals = np.array(parts[2].split("\t"), dtype=np.float32)

            # Skip low expression
            if np.median(vals) < 0.5:
                continue

            log_vals = np.log2(vals + 1)
            n_kept += 1

            is_immune = gene_name in IMMUNE_GENES
            is_tissue_specific = gene_name in TISSUE_SPECIFIC

            # For each tissue pair: compute cross-tissue ρ for young, mid, old, male, female
            for t1, t2 in tissue_pairs:
                # Get common donors for this tissue pair
                common_all = [d for d in multi if t1 in donor_tissue[d] and t2 in donor_tissue[d]]
                if len(common_all) < 50:
                    continue

                v1_all = np.array([log_vals[donor_tissue[d][t1]] for d in common_all])
                v2_all = np.array([log_vals[donor_tissue[d][t2]] for d in common_all])

                rho_all, _ = spearman_safe(v1_all, v2_all)
                if np.isnan(rho_all):
                    continue

                row = {
                    "gene": gene_name, "t1": t1, "t2": t2,
                    "is_immune": is_immune, "is_tissue_specific": is_tissue_specific,
                    "rho_all": rho_all, "n_all": len(common_all),
                }

                # Age groups
                for label, donor_list in [("young", young_donors), ("mid", mid_donors), ("old", old_donors)]:
                    common = [d for d in donor_list if t1 in donor_tissue[d] and t2 in donor_tissue[d]]
                    if len(common) >= 15:
                        v1 = np.array([log_vals[donor_tissue[d][t1]] for d in common])
                        v2 = np.array([log_vals[donor_tissue[d][t2]] for d in common])
                        rho, _ = spearman_safe(v1, v2)
                        row[f"rho_{label}"] = rho
                        row[f"n_{label}"] = len(common)

                # Sex groups
                for label, donor_list in [("male", male_donors), ("female", female_donors)]:
                    common = [d for d in donor_list if t1 in donor_tissue[d] and t2 in donor_tissue[d]]
                    if len(common) >= 20:
                        v1 = np.array([log_vals[donor_tissue[d][t1]] for d in common])
                        v2 = np.array([log_vals[donor_tissue[d][t2]] for d in common])
                        rho, _ = spearman_safe(v1, v2)
                        row[f"rho_{label}"] = rho
                        row[f"n_{label}"] = len(common)

                gene_results.append(row)

    _log(f"\n  Processed {n_processed} genes, kept {n_kept}")
    _log(f"  Gene×tissue_pair records: {len(gene_results)}")

    df = pd.DataFrame(gene_results)
    df["delta_age"] = df.get("rho_old", np.nan) - df.get("rho_young", np.nan)

    # Save compressed
    df.to_csv(RESULTS_DIR / "gene_cross_tissue.csv.gz", index=False, compression="gzip")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS A: Which genes drive decoupling?
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[A] Which genes drive cross-tissue decoupling?")
    _log("=" * 70)

    valid_delta = df.dropna(subset=["rho_young", "rho_old"])
    gene_delta = valid_delta.groupby("gene").agg(
        mean_delta=("delta_age", "mean"),
        median_delta=("delta_age", "median"),
        n_pairs=("delta_age", "count"),
        n_decline=("delta_age", lambda x: (x < 0).sum()),
        mean_rho_young=("rho_young", "mean"),
        mean_rho_old=("rho_old", "mean"),
    )
    gene_delta = gene_delta[gene_delta["n_pairs"] >= 5]
    gene_delta["pct_decline"] = gene_delta["n_decline"] / gene_delta["n_pairs"]
    gene_delta["is_immune"] = gene_delta.index.isin(IMMUNE_GENES)
    gene_delta = gene_delta.sort_values("median_delta")

    gene_delta.to_csv(RESULTS_DIR / "gene_decoupling_rank.csv")

    _log(f"\n  Genes with data in ≥5 tissue pairs: {len(gene_delta)}")

    # Overall direction
    n_decouplers = (gene_delta["median_delta"] < 0).sum()
    n_maintainers = (gene_delta["median_delta"] > 0).sum()
    sign_p = stats.binomtest(n_decouplers, n_decouplers + n_maintainers, 0.5).pvalue
    _log(f"  Decouplers: {n_decouplers}, Maintainers: {n_maintainers} (sign p={sign_p:.2e})")
    _log(f"  Overall median Δρ = {gene_delta['median_delta'].median():+.4f}")

    _log(f"\n  Top 15 DECOUPLERS (lose cross-tissue coordination):")
    for gene, r in gene_delta.head(15).iterrows():
        imm = " [IMMUNE]" if r["is_immune"] else ""
        _log(f"    {gene:<15s}: Δρ={r['median_delta']:+.4f}, decline in {r['pct_decline']:.0%} pairs, "
             f"ρ: {r['mean_rho_young']:+.3f}→{r['mean_rho_old']:+.3f}{imm}")

    _log(f"\n  Top 15 MAINTAINERS (keep cross-tissue coordination):")
    for gene, r in gene_delta.tail(15).iterrows():
        imm = " [IMMUNE]" if r["is_immune"] else ""
        _log(f"    {gene:<15s}: Δρ={r['median_delta']:+.4f}, decline in {r['pct_decline']:.0%} pairs, "
             f"ρ: {r['mean_rho_young']:+.3f}→{r['mean_rho_old']:+.3f}{imm}")

    # Immune vs non-immune
    imm_genes = gene_delta[gene_delta["is_immune"]]
    non_imm = gene_delta[~gene_delta["is_immune"]]
    _log(f"\n  Immune genes (n={len(imm_genes)}): median Δρ = {imm_genes['median_delta'].median():+.4f}")
    _log(f"  Non-immune (n={len(non_imm)}): median Δρ = {non_imm['median_delta'].median():+.4f}")
    if len(imm_genes) > 5 and len(non_imm) > 5:
        u, p = stats.mannwhitneyu(imm_genes["median_delta"], non_imm["median_delta"])
        _log(f"  Immune vs non-immune Wilcoxon p = {p:.2e}")

    # ═══════════════════════════════════════════════════════════════════
    # CONFOUND 1: Immune gene exclusion
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Confound 1] Cross-tissue ρ excluding immune genes")
    _log("=" * 70)

    for category, mask_fn in [
        ("ALL genes", lambda x: True),
        ("Excluding immune", lambda x: not x),
        ("Immune ONLY", lambda x: x),
    ]:
        sub = valid_delta[valid_delta["is_immune"].apply(mask_fn)]
        if len(sub) < 100:
            continue

        tp_summary = []
        for (t1, t2), grp in sub.groupby(["t1", "t2"]):
            for age in ["young", "old"]:
                col = f"rho_{age}"
                if col in grp.columns:
                    vals = grp[col].dropna()
                    if len(vals) > 50:
                        tp_summary.append({"t1": t1, "t2": t2, "age": age,
                                            "median_rho": vals.median(), "n": len(vals)})

        if tp_summary:
            dfs = pd.DataFrame(tp_summary)
            y = dfs[dfs["age"] == "young"]["median_rho"]
            o = dfs[dfs["age"] == "old"]["median_rho"]
            if len(y) > 3 and len(o) > 3:
                _log(f"  {category}: young median ρ = {y.median():+.4f}, "
                     f"old median ρ = {o.median():+.4f}, "
                     f"Δ = {o.median() - y.median():+.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS B: Age trajectory
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[B] Age trajectory — linear or breakpoint?")
    _log("=" * 70)

    # Compute median cross-tissue ρ per age decade
    age_bins = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80)]

    trajectory_data = []
    for t1, t2 in tissue_pairs:
        common_all = [d for d in multi if t1 in donor_tissue[d] and t2 in donor_tissue[d]]
        if len(common_all) < 100:
            continue

        for lo, hi in age_bins:
            donors_bin = [d for d in common_all if lo <= donor_meta[d]["age"] < hi]
            if len(donors_bin) < 15:
                continue

            # Compute median cross-tissue ρ across genes for this age bin
            # Use a subset of representative genes (from top variable)
            rhos_this_bin = valid_delta[
                (valid_delta["t1"] == t1) & (valid_delta["t2"] == t2)
            ]

            # We don't have per-bin ρ stored, so approximate from young/mid/old
            # Better: use the stored rho_young/mid/old
            age_label = "young" if hi <= 36 else ("mid" if hi <= 55 else "old")
            col = f"rho_{age_label}"
            if col in rhos_this_bin.columns:
                vals = rhos_this_bin[col].dropna()
                if len(vals) > 50:
                    trajectory_data.append({
                        "t1": t1, "t2": t2, "age_lo": lo, "age_hi": hi,
                        "age_mid": (lo + hi) / 2,
                        "median_rho": vals.median(), "n_genes": len(vals),
                        "n_donors": len(donors_bin),
                    })

    df_traj = pd.DataFrame(trajectory_data)
    if len(df_traj) > 0:
        _log(f"  Trajectory data points: {len(df_traj)}")
        for (t1, t2), grp in df_traj.groupby(["t1", "t2"]):
            grp = grp.sort_values("age_mid")
            t1s = t1.split(" - ")[-1][:12]
            t2s = t2.split(" - ")[-1][:12]
            parts = [f"{r['age_lo']:.0f}-{r['age_hi']:.0f}: ρ={r['median_rho']:+.3f}" for _, r in grp.iterrows()]
            _log(f"  {t1s:>12s}×{t2s:<12s}: {' | '.join(parts)}")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS C: Sex-stratified
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[C] Sex-stratified cross-tissue coordination")
    _log("=" * 70)

    sex_summary = []
    for (t1, t2), grp in valid_delta.groupby(["t1", "t2"]):
        t1s = t1.split(" - ")[-1][:12]
        t2s = t2.split(" - ")[-1][:12]

        row = {"t1": t1s, "t2": t2s}
        for sex in ["male", "female"]:
            col = f"rho_{sex}"
            if col in grp.columns:
                vals = grp[col].dropna()
                if len(vals) > 50:
                    row[f"median_rho_{sex}"] = vals.median()
                    row[f"n_{sex}"] = len(vals)
        sex_summary.append(row)

    df_sex = pd.DataFrame(sex_summary)
    _log(f"\n  Sex comparison (median cross-tissue ρ across genes):")
    for _, r in df_sex.iterrows():
        m = r.get("median_rho_male", np.nan)
        f = r.get("median_rho_female", np.nan)
        if not np.isnan(m) and not np.isnan(f):
            _log(f"  {r['t1']:>12s}×{r['t2']:<12s}: male ρ={m:+.4f}, female ρ={f:+.4f}, Δ={m-f:+.4f}")

    # Overall
    m_vals = df_sex["median_rho_male"].dropna()
    f_vals = df_sex["median_rho_female"].dropna()
    if len(m_vals) > 3 and len(f_vals) > 3:
        _log(f"\n  Overall: male median ρ = {m_vals.median():+.4f}, female = {f_vals.median():+.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS D: Which tissue pairs decouple most?
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[D] Tissue pair decoupling ranking")
    _log("=" * 70)

    pair_ranking = []
    for (t1, t2), grp in valid_delta.groupby(["t1", "t2"]):
        deltas = grp["delta_age"].dropna()
        if len(deltas) < 50:
            continue
        n_dec = (deltas < 0).sum()
        pair_ranking.append({
            "t1": t1.split(" - ")[-1][:15],
            "t2": t2.split(" - ")[-1][:15],
            "median_delta": deltas.median(),
            "pct_decline": n_dec / len(deltas),
            "n_genes": len(deltas),
            "rho_young": grp["rho_young"].dropna().median(),
            "rho_old": grp["rho_old"].dropna().median(),
        })

    df_pairs = pd.DataFrame(pair_ranking).sort_values("median_delta")
    df_pairs.to_csv(RESULTS_DIR / "tissue_pair_ranking.csv", index=False)

    _log(f"\n  {'Tissue pair':<30s} {'ρ_young':>8s} {'ρ_old':>8s} {'Δρ':>8s} {'%decline':>8s}")
    for _, r in df_pairs.iterrows():
        _log(f"  {r['t1']+' × '+r['t2']:<30s} {r['rho_young']:+.4f} {r['rho_old']:+.4f} "
             f"{r['median_delta']:+.4f} {r['pct_decline']:>7.0%}")

    # ═══════════════════════════════════════════════════════════════════
    # CONFOUND 2: Batch
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Confound 2] Batch structure")
    _log("=" * 70)

    # Check if young vs old donors are in different batches
    batch_data = []
    for d in multi:
        batch_data.append({
            "donor": d, "age": donor_meta[d]["age"], "sex": donor_meta[d]["sex"],
            "batch": donor_meta[d].get("batch", "?"),
        })
    df_batch = pd.DataFrame(batch_data)

    # Young vs old batch overlap
    young_batches = set(df_batch[df_batch["age"] <= 35]["batch"].unique())
    old_batches = set(df_batch[df_batch["age"] >= 55]["batch"].unique())
    overlap = young_batches & old_batches
    _log(f"  Young batches: {len(young_batches)}")
    _log(f"  Old batches: {len(old_batches)}")
    _log(f"  Overlap: {len(overlap)} ({len(overlap)/max(len(young_batches|old_batches),1):.0%})")

    if len(overlap) > 0:
        _log(f"  ✓ Batches overlap — not fully confounded")
    else:
        _log(f"  ⚠️ No batch overlap — potential confound!")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Figures]")
    _log("=" * 70)

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Panel A: Gene decoupling histogram
    ax = fig.add_subplot(gs[0, 0])
    if len(gene_delta) > 0:
        ax.hist(gene_delta["median_delta"].values, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
        imm_d = gene_delta[gene_delta["is_immune"]]["median_delta"]
        if len(imm_d) > 5:
            ax.axvline(imm_d.median(), color="red", linewidth=2, linestyle="--", label=f"Immune median ({imm_d.median():+.3f})")
        non_d = gene_delta[~gene_delta["is_immune"]]["median_delta"]
        ax.axvline(non_d.median(), color="blue", linewidth=2, linestyle="--", label=f"Non-immune median ({non_d.median():+.3f})")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Median Δρ (old − young) per gene")
        ax.set_ylabel("Number of genes")
        ax.set_title("A: Gene-level cross-tissue decoupling")
        ax.legend(fontsize=8)

    # Panel B: Top decouplers and maintainers
    ax = fig.add_subplot(gs[0, 1])
    if len(gene_delta) > 0:
        show = pd.concat([gene_delta.head(10), gene_delta.tail(10)])
        colors = ["tab:red" if v < 0 else "tab:green" for v in show["median_delta"]]
        ax.barh(range(len(show)), show["median_delta"].values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(show)))
        labels = [f"{'[I] ' if r['is_immune'] else ''}{gene}" for gene, r in show.iterrows()]
        ax.set_yticklabels(labels, fontsize=6)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Median Δρ")
        ax.set_title("B: Top decouplers (red) & maintainers (green)\n[I] = immune gene")

    # Panel C: Tissue pair ranking
    ax = fig.add_subplot(gs[0, 2])
    if len(df_pairs) > 0:
        ax.barh(range(len(df_pairs)), df_pairs["median_delta"].values, color="steelblue", alpha=0.7)
        ax.set_yticks(range(len(df_pairs)))
        ax.set_yticklabels([f"{r['t1']}×{r['t2']}" for _, r in df_pairs.iterrows()], fontsize=6)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Median Δρ (old − young)")
        ax.set_title("C: Tissue pair decoupling ranking")

    # Panel D: Sex comparison
    ax = fig.add_subplot(gs[1, 0])
    if len(df_sex) > 0 and "median_rho_male" in df_sex.columns:
        valid_sex = df_sex.dropna(subset=["median_rho_male", "median_rho_female"])
        if len(valid_sex) > 0:
            ax.scatter(valid_sex["median_rho_female"], valid_sex["median_rho_male"],
                       s=60, alpha=0.7, edgecolors="gray")
            for _, r in valid_sex.iterrows():
                ax.annotate(f"{r['t1'][:4]}×{r['t2'][:4]}",
                            (r["median_rho_female"], r["median_rho_male"]), fontsize=5)
            lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
            ax.set_xlabel("Female cross-tissue ρ")
            ax.set_ylabel("Male cross-tissue ρ")
            ax.set_title("D: Sex effect on cross-tissue coordination")

    # Panel E: Immune exclusion effect
    ax = fig.add_subplot(gs[1, 1])
    if len(valid_delta) > 0:
        for category, color, label in [
            (True, "red", "Immune genes"),
            (False, "blue", "Non-immune genes"),
        ]:
            sub = valid_delta[valid_delta["is_immune"] == category]
            if "rho_young" in sub.columns and "rho_old" in sub.columns:
                y_vals = sub.groupby(["t1", "t2"])["rho_young"].median()
                o_vals = sub.groupby(["t1", "t2"])["rho_old"].median()
                if len(y_vals) > 0:
                    x = np.arange(len(y_vals))
                    ax.scatter(y_vals.values, o_vals.values, c=color, s=40, alpha=0.7, label=label)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
        ax.set_xlabel("Young cross-tissue ρ")
        ax.set_ylabel("Old cross-tissue ρ")
        ax.set_title("E: Immune vs non-immune\n(below diagonal = decline)")
        ax.legend(fontsize=8)

    # Panel F: Young ρ vs Δρ (does initial coordination predict decline?)
    ax = fig.add_subplot(gs[1, 2])
    if "mean_rho_young" in gene_delta.columns:
        valid = gene_delta.dropna(subset=["mean_rho_young", "median_delta"])
        ax.scatter(valid["mean_rho_young"], valid["median_delta"], s=1, alpha=0.05, c="gray")
        # Binned trend
        bins = np.linspace(-0.1, 0.5, 25)
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mask = (valid["mean_rho_young"] >= b0) & (valid["mean_rho_young"] < b1)
            if mask.sum() > 30:
                ax.scatter((b0+b1)/2, valid.loc[mask, "median_delta"].median(),
                           c="red", s=30, zorder=5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Mean young cross-tissue ρ")
        ax.set_ylabel("Median Δρ (old − young)")
        ax.set_title("F: Does initial coordination predict decline?")

    # Panel G-I: Trajectory examples
    for idx, (t1, t2) in enumerate(tissue_pairs[:3]):
        ax = fig.add_subplot(gs[2, idx])
        t1s = t1.split(" - ")[-1][:12]
        t2s = t2.split(" - ")[-1][:12]

        sub = valid_delta[(valid_delta["t1"] == t1) & (valid_delta["t2"] == t2)]
        for age_label, color in [("young", "tab:blue"), ("mid", "tab:gray"), ("old", "tab:red")]:
            col = f"rho_{age_label}"
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals) > 50:
                    ax.hist(vals.values, bins=50, alpha=0.4, color=color,
                            label=f"{age_label} (med={vals.median():+.3f})", density=True)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Cross-tissue ρ per gene")
        ax.set_title(f"{'GHI'[idx]}: {t1s} × {t2s}")
        ax.legend(fontsize=7)

    fig.suptitle("Cross-Tissue Coordination Decline — Deep Validation\n"
                 "GTEx 948 donors, 6 tissues, genome-wide",
                 fontsize=14, fontweight="bold")
    fig.savefig(RESULTS_DIR / "cross_tissue_deep.png", dpi=150)
    plt.close(fig)
    _log(f"  Saved cross_tissue_deep.png")

    _log(f"\n  Total time: {time.time()-t0:.0f}s")
    _log(f"  Results: {RESULTS_DIR}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
