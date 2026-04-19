"""
Step 5: Beyond coupling — three new information structure metrics on GTEx

1. Gene predictability: R²(expression ~ age + sex) per gene per tissue
   - Which genes become more/less predictable with age?

2. Inter-tissue predictability: does cross-tissue coordination change with age?
   - COL1A1 cross-tissue ρ in young vs old donors
   - Genome-wide: mean cross-tissue R² in young vs old

3. Inter-individual entropy: do individuals diverge or converge?
   - H(gene) across donors, young vs old
   - Does variance increase (divergence) or decrease (convergence)?

Strategy: stream GTEx TPM line by line, compute stats per gene, never hold full matrix.
"""

import time
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data" / "gtex"
RESULTS_DIR = BASE / "results" / "step5_info_structure"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg):
    print(msg, flush=True)


def entropy_from_vals(vals, n_bins=20):
    """Shannon entropy of continuous values (binned)."""
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return np.nan
    counts, _ = np.histogram(vals, bins=n_bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def main():
    t0 = time.time()
    _log("=" * 70)
    _log("INFORMATION STRUCTURE ANALYSIS — Beyond coupling")
    _log("=" * 70)

    # ── Load metadata ────────────────────────────────────────────────
    _log("\n[0] Loading metadata...")
    samples = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    samples["sex_num"] = samples["SEX"].map({1: 0, 2: 1})  # male=0, female=1

    # Parse age
    samples["age_mid"] = samples["AGE"].str.split("-").apply(
        lambda x: (int(x[0]) + int(x[1])) / 2 if isinstance(x, list) and len(x) == 2 else np.nan)
    samples["age_bin"] = "middle"
    samples.loc[samples["age_mid"] <= 35, "age_bin"] = "young"
    samples.loc[samples["age_mid"] >= 60, "age_bin"] = "old"

    # Top tissues by sample count
    tissue_counts = samples["SMTSD"].value_counts()
    top_tissues = tissue_counts[tissue_counts >= 100].index.tolist()
    _log(f"  {len(samples)} RNA-seq samples, {len(top_tissues)} tissues with ≥100 samples")

    # Build sample→metadata lookup
    sample_meta = samples.set_index("SAMPID")[["SUBJID", "SMTSD", "sex_num", "age_mid", "age_bin"]].copy()

    # ── Read TPM header ──────────────────────────────────────────────
    tpm_path = DATA_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    _log(f"\n[1] Reading TPM header...")
    with gzip.open(tpm_path, "rt") as f:
        f.readline()  # version
        f.readline()  # dimensions
        header = f.readline().strip().split("\t")
    sample_ids = header[2:]
    n_samples = len(sample_ids)
    _log(f"  {n_samples} samples in TPM matrix")

    # Map sample positions to metadata
    sample_idx_to_meta = {}
    for i, sid in enumerate(sample_ids):
        if sid in sample_meta.index:
            sample_idx_to_meta[i] = sample_meta.loc[sid]

    # Pre-compute tissue→sample_indices mapping
    tissue_sample_idx = {}
    for i, sid in enumerate(sample_ids):
        if sid in sample_meta.index:
            tissue = sample_meta.loc[sid, "SMTSD"]
            if tissue in top_tissues:
                if tissue not in tissue_sample_idx:
                    tissue_sample_idx[tissue] = []
                tissue_sample_idx[tissue].append(i)

    for t in list(tissue_sample_idx.keys()):
        tissue_sample_idx[t] = np.array(tissue_sample_idx[t])

    _log(f"  {len(tissue_sample_idx)} tissues mapped")

    # Pre-compute age and sex arrays per tissue
    tissue_age = {}
    tissue_sex = {}
    tissue_age_bin = {}
    for tissue, idx_arr in tissue_sample_idx.items():
        ages = np.array([sample_meta.iloc[sample_meta.index.get_loc(sample_ids[i])]["age_mid"]
                         if sample_ids[i] in sample_meta.index else np.nan
                         for i in idx_arr])
        sexes = np.array([sample_meta.iloc[sample_meta.index.get_loc(sample_ids[i])]["sex_num"]
                          if sample_ids[i] in sample_meta.index else np.nan
                          for i in idx_arr])
        age_bins = np.array([sample_meta.iloc[sample_meta.index.get_loc(sample_ids[i])]["age_bin"]
                             if sample_ids[i] in sample_meta.index else "?"
                             for i in idx_arr])
        tissue_age[tissue] = ages
        tissue_sex[tissue] = sexes
        tissue_age_bin[tissue] = age_bins

    # For cross-tissue analysis: identify donors with data in multiple tissues
    # Focus on top 6 tissues
    top6 = tissue_counts.head(6).index.tolist()
    donor_tissue_samples = {}  # donor → tissue → sample_idx_in_tpm
    for tissue in top6:
        for i in tissue_sample_idx[tissue]:
            sid = sample_ids[i]
            if sid in sample_meta.index:
                donor = sample_meta.loc[sid, "SUBJID"]
                if donor not in donor_tissue_samples:
                    donor_tissue_samples[donor] = {}
                donor_tissue_samples[donor][tissue] = i

    # Donors with data in at least 2 of top6 tissues
    multi_tissue_donors = {d: ts for d, ts in donor_tissue_samples.items() if len(ts) >= 2}
    _log(f"  {len(multi_tissue_donors)} donors with ≥2 of top 6 tissues")

    # ── Stream through TPM ───────────────────────────────────────────
    _log(f"\n[2] Streaming through TPM file...")

    # Storage for results
    gene_predictability = []  # per gene per tissue: R², age_coef, etc.
    gene_entropy_young = []   # per gene per tissue: H in young
    gene_entropy_old = []     # per gene per tissue: H in old
    gene_variance = []        # per gene per tissue: CV in young vs old

    # For cross-tissue: store expression of top variable genes per donor
    N_CROSS_GENES = 500
    cross_tissue_data = {}  # gene → {tissue: {donor: value}}
    gene_variances_for_selection = []  # to pick top variable genes

    n_genes_processed = 0
    n_genes_kept = 0

    # First pass: identify top variable genes (quick scan of first 5000 genes to estimate)
    # Actually, let's do it in one pass: compute everything per gene

    BATCH_SIZE = 1000
    batch_count = 0

    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()  # skip header

        for line in f:
            parts = line.strip().split("\t", 2)
            gene_name = parts[1]
            vals = np.array(parts[2].split("\t"), dtype=np.float32)

            n_genes_processed += 1
            if n_genes_processed % 5000 == 0:
                _log(f"    Processed {n_genes_processed} genes ({time.time()-t0:.0f}s)...")

            # Skip genes with very low expression
            if np.median(vals) < 0.1:
                continue

            log_vals = np.log2(vals + 1)
            n_genes_kept += 1

            # ── Analysis 1: Predictability per tissue ────────────
            for tissue in top6:  # Focus on top 6 for speed
                idx = tissue_sample_idx[tissue]
                expr = log_vals[idx]
                age = tissue_age[tissue]
                sex = tissue_sex[tissue]
                a_bin = tissue_age_bin[tissue]

                valid = np.isfinite(age) & np.isfinite(sex) & np.isfinite(expr)
                if valid.sum() < 50:
                    continue

                e, a, s = expr[valid], age[valid], sex[valid]

                # R²(expression ~ age + sex)
                X_reg = np.column_stack([a, s])
                lr = LinearRegression()
                lr.fit(X_reg, e)
                r2 = lr.score(X_reg, e)
                age_coef = lr.coef_[0]

                # Age coefficient significance
                from scipy.stats import pearsonr
                rho_age, p_age = pearsonr(a, e)

                gene_predictability.append({
                    "gene": gene_name, "tissue": tissue,
                    "R2": r2, "age_coef": age_coef,
                    "rho_age": rho_age, "p_age": p_age,
                    "mean_expr": np.mean(e), "cv": np.std(e) / (np.mean(e) + 1e-6),
                })

                # ── Analysis 3: Entropy young vs old ─────────────
                young_mask = a_bin[valid] == "young"
                old_mask = a_bin[valid] == "old"

                if young_mask.sum() >= 20 and old_mask.sum() >= 20:
                    h_young = entropy_from_vals(e[young_mask])
                    h_old = entropy_from_vals(e[old_mask])
                    cv_young = np.std(e[young_mask]) / (np.mean(e[young_mask]) + 1e-6)
                    cv_old = np.std(e[old_mask]) / (np.mean(e[old_mask]) + 1e-6)
                    var_young = np.var(e[young_mask])
                    var_old = np.var(e[old_mask])

                    gene_entropy_young.append({
                        "gene": gene_name, "tissue": tissue,
                        "H": h_young, "CV": cv_young, "var": var_young,
                        "n": int(young_mask.sum()),
                    })
                    gene_entropy_old.append({
                        "gene": gene_name, "tissue": tissue,
                        "H": h_old, "CV": cv_old, "var": var_old,
                        "n": int(old_mask.sum()),
                    })

            # ── Store for cross-tissue (collect all, select top later) ──
            global_var = np.var(log_vals[log_vals > 0]) if (log_vals > 0).sum() > 100 else 0
            gene_variances_for_selection.append((gene_name, global_var))

            # Store cross-tissue data for this gene (for top donors)
            if n_genes_kept <= N_CROSS_GENES * 5:  # keep buffer, will filter later
                gene_data = {}
                for tissue in top6:
                    idx = tissue_sample_idx[tissue]
                    tissue_data = {}
                    for i in idx:
                        sid = sample_ids[i]
                        if sid in sample_meta.index:
                            donor = sample_meta.loc[sid, "SUBJID"]
                            if donor in multi_tissue_donors:
                                tissue_data[donor] = log_vals[i]
                    if tissue_data:
                        gene_data[tissue] = tissue_data
                if gene_data:
                    cross_tissue_data[gene_name] = gene_data

    _log(f"\n  Processed {n_genes_processed} genes, kept {n_genes_kept} (median TPM ≥ 0.1)")
    _log(f"  Predictability records: {len(gene_predictability)}")
    _log(f"  Entropy records: {len(gene_entropy_young)} young, {len(gene_entropy_old)} old")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 1: Gene predictability
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Analysis 1] Gene predictability from age + sex")
    _log("=" * 70)

    df_pred = pd.DataFrame(gene_predictability)
    df_pred.to_csv(RESULTS_DIR / "gene_predictability.csv.gz", index=False, compression="gzip")

    _log(f"\n  Per-tissue summary:")
    for tissue in top6:
        sub = df_pred[df_pred["tissue"] == tissue]
        _log(f"\n  {tissue} ({len(sub)} genes):")
        _log(f"    Median R² = {sub['R2'].median():.4f}")
        _log(f"    Genes with R² > 0.05: {(sub['R2'] > 0.05).sum()} ({(sub['R2'] > 0.05).mean():.1%})")
        _log(f"    Genes with R² > 0.10: {(sub['R2'] > 0.10).sum()} ({(sub['R2'] > 0.10).mean():.1%})")

        # Top age-associated genes
        top_age = sub.nsmallest(5, "p_age")
        _log(f"    Top age-correlated genes:")
        for _, r in top_age.iterrows():
            _log(f"      {r['gene']:<15s}: ρ={r['rho_age']:+.3f}, R²={r['R2']:.3f}")

    # Overall: what fraction of transcriptome is age-predictable?
    _log(f"\n  Overall across tissues:")
    _log(f"    Median R² = {df_pred['R2'].median():.4f}")
    _log(f"    Mean R² = {df_pred['R2'].mean():.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 2: Inter-tissue predictability by age
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Analysis 2] Inter-tissue predictability by age")
    _log("=" * 70)

    # Select top variable genes with cross-tissue data
    top_var_genes = sorted(gene_variances_for_selection, key=lambda x: -x[1])[:N_CROSS_GENES]
    top_var_names = set(g for g, _ in top_var_genes)

    # Filter cross_tissue_data to top variable genes
    ct_data = {g: d for g, d in cross_tissue_data.items() if g in top_var_names}
    _log(f"  Cross-tissue genes: {len(ct_data)}")

    # For each tissue pair: compute genome-wide cross-tissue ρ, stratified by age
    cross_results = []
    donor_ages = {}
    for donor, tissues in multi_tissue_donors.items():
        # Get age from any tissue
        for t, idx in tissues.items():
            sid = sample_ids[idx]
            if sid in sample_meta.index:
                donor_ages[donor] = sample_meta.loc[sid, "age_mid"]
                break

    for i, t1 in enumerate(top6[:5]):
        for t2 in top6[i+1:6]:
            # Get common donors
            common_donors = []
            for d in multi_tissue_donors:
                if t1 in multi_tissue_donors[d] and t2 in multi_tissue_donors[d]:
                    common_donors.append(d)

            if len(common_donors) < 50:
                continue

            # Per-gene cross-tissue correlation
            gene_rhos = []
            for gene, gdata in ct_data.items():
                if t1 not in gdata or t2 not in gdata:
                    continue
                vals1, vals2 = [], []
                for d in common_donors:
                    if d in gdata[t1] and d in gdata[t2]:
                        vals1.append(gdata[t1][d])
                        vals2.append(gdata[t2][d])
                if len(vals1) >= 30:
                    rho, _ = stats.spearmanr(vals1, vals2)
                    if np.isfinite(rho):
                        gene_rhos.append(rho)

            if len(gene_rhos) < 50:
                continue

            overall_rho = np.median(gene_rhos)

            # Stratify by age
            young_donors = [d for d in common_donors if d in donor_ages and donor_ages[d] <= 35]
            old_donors = [d for d in common_donors if d in donor_ages and donor_ages[d] >= 60]

            for age_group, donors in [("young", young_donors), ("old", old_donors), ("all", common_donors)]:
                if len(donors) < 20:
                    continue

                age_rhos = []
                for gene, gdata in ct_data.items():
                    if t1 not in gdata or t2 not in gdata:
                        continue
                    v1, v2 = [], []
                    for d in donors:
                        if d in gdata[t1] and d in gdata[t2]:
                            v1.append(gdata[t1][d])
                            v2.append(gdata[t2][d])
                    if len(v1) >= 15:
                        r, _ = stats.spearmanr(v1, v2)
                        if np.isfinite(r):
                            age_rhos.append(r)

                if len(age_rhos) >= 30:
                    cross_results.append({
                        "tissue_1": t1, "tissue_2": t2,
                        "age_group": age_group, "n_donors": len(donors),
                        "n_genes": len(age_rhos),
                        "median_rho": np.median(age_rhos),
                        "mean_rho": np.mean(age_rhos),
                        "pct_positive": np.mean(np.array(age_rhos) > 0),
                    })

    df_cross = pd.DataFrame(cross_results)
    df_cross.to_csv(RESULTS_DIR / "inter_tissue_by_age.csv", index=False)

    _log(f"\n  Cross-tissue results: {len(df_cross)} entries")
    if len(df_cross) > 0:
        for (t1, t2), sub in df_cross.groupby(["tissue_1", "tissue_2"]):
            t1s = t1.split(" - ")[-1][:15]
            t2s = t2.split(" - ")[-1][:15]
            parts = []
            for _, r in sub.iterrows():
                parts.append(f"{r['age_group']}: ρ={r['median_rho']:+.3f} (n={r['n_donors']})")
            _log(f"  {t1s:>15s} × {t2s:<15s}: {' | '.join(parts)}")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 3: Inter-individual entropy
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Analysis 3] Inter-individual entropy — divergence or convergence?")
    _log("=" * 70)

    df_ey = pd.DataFrame(gene_entropy_young)
    df_eo = pd.DataFrame(gene_entropy_old)

    if len(df_ey) > 0 and len(df_eo) > 0:
        # Merge young and old
        df_entropy = df_ey.merge(df_eo, on=["gene", "tissue"], suffixes=("_young", "_old"))
        df_entropy["dH"] = df_entropy["H_old"] - df_entropy["H_young"]
        df_entropy["dCV"] = df_entropy["CV_old"] - df_entropy["CV_young"]
        df_entropy["dVar"] = df_entropy["var_old"] - df_entropy["var_young"]
        df_entropy["log_var_ratio"] = np.log2((df_entropy["var_old"] + 1e-6) /
                                               (df_entropy["var_young"] + 1e-6))

        df_entropy.to_csv(RESULTS_DIR / "entropy_young_vs_old.csv.gz", index=False, compression="gzip")

        _log(f"\n  Entropy comparison: {len(df_entropy)} gene×tissue records")

        for tissue in top6:
            sub = df_entropy[df_entropy["tissue"] == tissue]
            if len(sub) < 100:
                continue

            n_diverge = (sub["dH"] > 0).sum()
            n_converge = (sub["dH"] < 0).sum()
            med_dh = sub["dH"].median()
            med_dcv = sub["dCV"].median()
            med_lvr = sub["log_var_ratio"].median()

            # Sign test
            sign_p = stats.binomtest(n_diverge, n_diverge + n_converge, 0.5).pvalue

            _log(f"\n  {tissue} ({len(sub)} genes):")
            _log(f"    Entropy: {n_diverge} diverge / {n_converge} converge "
                 f"(sign test p={sign_p:.2e})")
            _log(f"    Median ΔH = {med_dh:+.4f}")
            _log(f"    Median ΔCV = {med_dcv:+.4f}")
            _log(f"    Median log₂(var_old/var_young) = {med_lvr:+.4f}")

            direction = "DIVERGE" if med_dh > 0 else "CONVERGE"
            _log(f"    → Individuals {direction} with age")

        # Overall
        _log(f"\n  OVERALL across all tissues:")
        n_div = (df_entropy["dH"] > 0).sum()
        n_conv = (df_entropy["dH"] < 0).sum()
        _log(f"    {n_div} diverge / {n_conv} converge "
             f"(sign test p={stats.binomtest(n_div, n_div+n_conv, 0.5).pvalue:.2e})")
        _log(f"    Median ΔH = {df_entropy['dH'].median():+.4f}")
        _log(f"    Median log₂(var_old/var_young) = {df_entropy['log_var_ratio'].median():+.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Figures]")
    _log("=" * 70)

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Panel A: R² distribution per tissue
    ax = fig.add_subplot(gs[0, 0])
    tissue_r2 = []
    for tissue in top6:
        sub = df_pred[df_pred["tissue"] == tissue]
        tissue_r2.append(sub["R2"].values)
    if tissue_r2:
        parts = ax.violinplot(tissue_r2, showmedians=True)
        ax.set_xticks(range(1, len(top6) + 1))
        ax.set_xticklabels([t.split(" - ")[-1][:12] for t in top6], rotation=45, fontsize=7)
        ax.set_ylabel("R² (expression ~ age + sex)")
        ax.set_title("A: Gene predictability from age+sex")
        ax.set_ylim(0, 0.3)

    # Panel B: Top age-correlated genes across tissues
    ax = fig.add_subplot(gs[0, 1])
    # Get genes with highest mean |rho_age| across tissues
    gene_mean_rho = df_pred.groupby("gene")["rho_age"].apply(lambda x: np.mean(np.abs(x))).nlargest(20)
    ax.barh(range(len(gene_mean_rho)), gene_mean_rho.values, color="steelblue", alpha=0.7)
    ax.set_yticks(range(len(gene_mean_rho)))
    ax.set_yticklabels(gene_mean_rho.index, fontsize=7)
    ax.set_xlabel("Mean |ρ(age)| across tissues")
    ax.set_title("B: Most age-predictable genes")

    # Panel C: Entropy change distribution
    ax = fig.add_subplot(gs[0, 2])
    if len(df_entropy) > 0:
        for i, tissue in enumerate(top6):
            sub = df_entropy[df_entropy["tissue"] == tissue]
            if len(sub) > 100:
                ax.hist(sub["dH"].values, bins=50, alpha=0.4,
                        label=tissue.split(" - ")[-1][:12], density=True)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("ΔH (old - young)")
        ax.set_ylabel("Density")
        ax.set_title("C: Inter-individual entropy change")
        ax.legend(fontsize=6)

    # Panel D: Variance ratio distribution
    ax = fig.add_subplot(gs[1, 0])
    if len(df_entropy) > 0:
        for tissue in top6:
            sub = df_entropy[df_entropy["tissue"] == tissue]
            if len(sub) > 100:
                med_lvr = sub["log_var_ratio"].median()
                ax.hist(sub["log_var_ratio"].values, bins=50, alpha=0.4,
                        label=f"{tissue.split(' - ')[-1][:10]} (med={med_lvr:+.2f})", density=True)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("log₂(var_old / var_young)")
        ax.set_ylabel("Density")
        ax.set_title("D: Variance ratio — divergence (+) or convergence (−)?")
        ax.legend(fontsize=6)

    # Panel E: Cross-tissue coordination young vs old
    ax = fig.add_subplot(gs[1, 1])
    if len(df_cross) > 0:
        young_vals = df_cross[df_cross["age_group"] == "young"]["median_rho"].values
        old_vals = df_cross[df_cross["age_group"] == "old"]["median_rho"].values
        all_vals = df_cross[df_cross["age_group"] == "all"]["median_rho"].values
        x = np.arange(len(young_vals))
        if len(young_vals) > 0:
            width = 0.25
            ax.bar(x - width, young_vals, width, label="Young (≤35)", color="tab:blue", alpha=0.7)
            ax.bar(x, all_vals[:len(x)], width, label="All", color="tab:gray", alpha=0.7)
            ax.bar(x + width, old_vals[:len(x)], width, label="Old (≥60)", color="tab:red", alpha=0.7)
            ax.set_xticks(x)
            labels = [f"{r['tissue_1'].split(' - ')[-1][:6]}×{r['tissue_2'].split(' - ')[-1][:6]}"
                      for _, r in df_cross[df_cross["age_group"] == "young"].iterrows()]
            ax.set_xticklabels(labels, rotation=45, fontsize=6)
            ax.set_ylabel("Median cross-tissue ρ")
            ax.set_title("E: Cross-tissue coordination by age")
            ax.legend(fontsize=8)

    # Panel F: R² vs mean expression
    ax = fig.add_subplot(gs[1, 2])
    sub_one = df_pred[df_pred["tissue"] == top6[0]]
    if len(sub_one) > 0:
        ax.scatter(sub_one["mean_expr"], sub_one["R2"], s=2, alpha=0.1, c="steelblue")
        ax.set_xlabel(f"Mean expression (log2 TPM+1)")
        ax.set_ylabel("R² (age + sex)")
        ax.set_title(f"F: Predictability vs expression level\n({top6[0].split(' - ')[-1]})")
        ax.set_ylim(0, 0.3)

    # Panel G: Per-tissue entropy summary
    ax = fig.add_subplot(gs[2, 0])
    if len(df_entropy) > 0:
        tissue_entropy_summary = []
        for tissue in top6:
            sub = df_entropy[df_entropy["tissue"] == tissue]
            if len(sub) > 100:
                tissue_entropy_summary.append({
                    "tissue": tissue.split(" - ")[-1][:12],
                    "median_dH": sub["dH"].median(),
                    "pct_diverge": (sub["dH"] > 0).mean(),
                })
        if tissue_entropy_summary:
            tes = pd.DataFrame(tissue_entropy_summary)
            ax.barh(range(len(tes)), tes["pct_diverge"] - 0.5, color="tab:red", alpha=0.7)
            ax.set_yticks(range(len(tes)))
            ax.set_yticklabels(tes["tissue"])
            ax.axvline(0, color="black", linewidth=1)
            ax.set_xlabel("Fraction diverging − 0.5 (>0 = more diverge)")
            ax.set_title("G: Per-tissue divergence tendency")

    # Panel H: Gene-level: most diverging vs converging genes
    ax = fig.add_subplot(gs[2, 1])
    if len(df_entropy) > 0:
        gene_dh = df_entropy.groupby("gene")["dH"].median().sort_values()
        top_conv = gene_dh.head(10)
        top_div = gene_dh.tail(10)
        combined = pd.concat([top_conv, top_div])
        colors = ["tab:blue"] * len(top_conv) + ["tab:red"] * len(top_div)
        ax.barh(range(len(combined)), combined.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels(combined.index, fontsize=7)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Median ΔH (old−young)")
        ax.set_title("H: Most converging (blue) vs diverging (red) genes")

    # Panel I: Predictability vs entropy change
    ax = fig.add_subplot(gs[2, 2])
    if len(df_entropy) > 0 and len(df_pred) > 0:
        merged = df_entropy.merge(df_pred[["gene", "tissue", "R2", "rho_age"]],
                                   on=["gene", "tissue"], how="inner")
        if len(merged) > 100:
            ax.scatter(merged["R2"], merged["dH"], s=1, alpha=0.05, c="gray")
            # Bin and show trend
            bins = np.linspace(0, 0.2, 20)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_means = []
            for b0, b1 in zip(bins[:-1], bins[1:]):
                mask = (merged["R2"] >= b0) & (merged["R2"] < b1)
                if mask.sum() > 50:
                    bin_means.append(merged.loc[mask, "dH"].median())
                else:
                    bin_means.append(np.nan)
            ax.plot(bin_centers, bin_means, "r-o", markersize=4, linewidth=2)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("R² (age + sex)")
            ax.set_ylabel("ΔH (old − young)")
            ax.set_title("I: Predictable genes diverge or converge?")

    fig.suptitle("Information Structure of Aging — GTEx 948 donors\n"
                 "Predictability, cross-tissue coordination, inter-individual entropy",
                 fontsize=14, fontweight="bold")
    fig.savefig(RESULTS_DIR / "information_structure.png", dpi=150)
    plt.close(fig)
    _log(f"\n  Saved information_structure.png")

    _log(f"\n  Total time: {time.time()-t0:.0f}s")
    _log(f"  Results: {RESULTS_DIR}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
