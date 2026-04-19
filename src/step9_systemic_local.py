"""
Step 9: Systemic vs Local variance decomposition

Core question: Do tissues converge because local regulation degrades
while systemic factors persist?

expression(gene, tissue, donor) = systemic(gene, donor) + local(gene, tissue, donor)
systemic(gene, donor) = mean across tissues for that donor
local = residual

Tests:
1. Variance decomposition: systemic vs local fraction by age
2. Per-gene systemic fraction: high-systemic vs low-systemic genes age differently?
3. Known covariates: do age/sex/BMI explain systemic growth?
4. Cross-tissue prediction: does systemic→local R² change with age?
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
RESULTS_DIR = BASE / "results" / "step9_systemic_local"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg):
    print(msg, flush=True)


def main():
    t0 = time.time()
    _log("=" * 70)
    _log("SYSTEMIC vs LOCAL — Variance decomposition")
    _log("=" * 70)

    # ── Load metadata ────────────────────────────────────────────────
    samples = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    samples["age_mid"] = samples["AGE"].str.split("-").apply(
        lambda x: (int(x[0]) + int(x[1])) / 2 if isinstance(x, list) and len(x) == 2 else np.nan)
    samples["sex"] = samples["SEX"].map({1: "male", 2: "female"})

    # Focus on 6 tissues with most donors having ALL 6
    top6 = ["Muscle - Skeletal", "Whole Blood", "Skin - Sun Exposed (Lower leg)",
            "Adipose - Subcutaneous", "Artery - Tibial", "Thyroid"]

    sample_meta = samples.set_index("SAMPID")[["SUBJID", "SMTSD", "age_mid", "sex"]].copy()

    tpm_path = DATA_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline()
        header = f.readline().strip().split("\t")
    sample_ids = header[2:]

    # Build tissue → sample index mapping
    tissue_sample_map = {t: {} for t in top6}  # tissue → {SUBJID: col_index}
    for i, sid in enumerate(sample_ids):
        if sid in sample_meta.index:
            row = sample_meta.loc[sid]
            t = row["SMTSD"]
            subj = row["SUBJID"]
            if t in top6:
                tissue_sample_map[t][subj] = i

    # Find donors with ALL 6 tissues
    donor_sets = [set(tissue_sample_map[t].keys()) for t in top6]
    common_donors = donor_sets[0]
    for ds in donor_sets[1:]:
        common_donors &= ds
    common_donors = sorted(common_donors)

    _log(f"  Donors with all 6 tissues: {len(common_donors)}")

    # Get ages for common donors
    donor_ages = {}
    donor_sex = {}
    for subj in common_donors:
        for t in top6:
            idx = tissue_sample_map[t][subj]
            sid = sample_ids[idx]
            if sid in sample_meta.index:
                donor_ages[subj] = sample_meta.loc[sid, "age_mid"]
                donor_sex[subj] = sample_meta.loc[sid, "sex"]
                break

    ages = np.array([donor_ages.get(d, np.nan) for d in common_donors])
    young_mask = ages <= 35
    old_mask = ages >= 55
    _log(f"  Young (≤35): {young_mask.sum()}, Old (≥55): {old_mask.sum()}")

    # Build column index matrix: donors × tissues
    col_idx = np.zeros((len(common_donors), len(top6)), dtype=int)
    for ti, t in enumerate(top6):
        for di, d in enumerate(common_donors):
            col_idx[di, ti] = tissue_sample_map[t][d]

    # ── Read TPM and decompose ────────────────────────────────────────
    _log(f"\n[1] Reading TPM and computing variance decomposition...")

    gene_results = []
    n_processed = 0

    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            parts = line.strip().split("\t", 2)
            gene = parts[1]
            vals = np.array(parts[2].split("\t"), dtype=np.float32)

            # Filter low-expression genes
            if np.median(vals) < 0.5:
                n_processed += 1
                continue

            log_vals = np.log2(vals + 1)
            n_processed += 1

            if n_processed % 5000 == 0:
                _log(f"    {n_processed} genes ({time.time()-t0:.0f}s)...")

            # Extract expression matrix: donors × tissues
            expr_mat = log_vals[col_idx]  # shape: (n_donors, 6)

            # Systemic component = donor mean across tissues
            systemic = expr_mat.mean(axis=1)  # (n_donors,)
            # Local component = tissue residual
            local = expr_mat - systemic[:, np.newaxis]  # (n_donors, 6)

            # Total variance across all donor-tissue combinations
            total_var = np.var(expr_mat)
            if total_var < 1e-10:
                continue

            # Systemic variance = variance of donor means
            sys_var = np.var(systemic)
            # Local variance = mean variance of residuals
            local_var = np.mean(np.var(local, axis=0))

            # By age group
            for age_label, mask in [("young", young_mask), ("old", old_mask), ("all", np.ones(len(common_donors), dtype=bool))]:
                if mask.sum() < 20:
                    continue
                e = expr_mat[mask]
                s = e.mean(axis=1)
                l = e - s[:, np.newaxis]

                tv = np.var(e)
                sv = np.var(s)
                lv = np.mean(np.var(l, axis=0))
                frac_sys = sv / tv if tv > 1e-10 else np.nan

                if age_label != "all":
                    gene_results.append({
                        "gene": gene, "age_group": age_label,
                        "total_var": tv, "systemic_var": sv, "local_var": lv,
                        "frac_systemic": frac_sys,
                    })

    df_genes = pd.DataFrame(gene_results)
    _log(f"  Processed {n_processed} genes, {len(df_genes)} measurements")

    # ── Analysis: systemic fraction young vs old ──────────────────────
    _log(f"\n" + "=" * 70)
    _log("[Test 1] Systemic fraction: young vs old")
    _log("=" * 70)

    # Pivot to get young and old side by side
    pivot = df_genes.pivot_table(index="gene", columns="age_group",
                                  values="frac_systemic", aggfunc="first")

    if "young" in pivot.columns and "old" in pivot.columns:
        valid = pivot.dropna(subset=["young", "old"])
        _log(f"  Genes with both young and old: {len(valid)}")

        frac_y = valid["young"]
        frac_o = valid["old"]

        _log(f"  Systemic fraction (median):")
        _log(f"    Young: {frac_y.median():.4f}")
        _log(f"    Old:   {frac_o.median():.4f}")
        _log(f"    Δ:     {(frac_o - frac_y).median():+.4f}")

        # Sign test: how many genes have higher systemic fraction in old?
        n_higher = (frac_o > frac_y).sum()
        n_total = len(valid)
        sign_p = stats.binomtest(n_higher, n_total, 0.5).pvalue
        _log(f"    {n_higher}/{n_total} genes have higher systemic fraction in old")
        _log(f"    Sign test p = {sign_p:.2e}")

        # Wilcoxon
        w_stat, w_p = stats.wilcoxon(frac_o - frac_y)
        _log(f"    Wilcoxon signed-rank p = {w_p:.2e}")

        # Effect size
        delta = (frac_o - frac_y)
        _log(f"    Mean Δ = {delta.mean():+.4f}")
        _log(f"    Cohen's d ≈ {delta.mean() / delta.std():.3f}")

        # Save per-gene results
        valid["delta_frac"] = frac_o - frac_y
        valid.to_csv(RESULTS_DIR / "per_gene_systemic_fraction.csv")

    # Variance components
    _log(f"\n  Variance components (medians):")
    for ag in ["young", "old"]:
        sub = df_genes[df_genes["age_group"] == ag]
        _log(f"    {ag}: total_var={sub['total_var'].median():.4f}, "
             f"systemic_var={sub['systemic_var'].median():.4f}, "
             f"local_var={sub['local_var'].median():.4f}")

    # ── Test 2: High-systemic vs low-systemic genes ───────────────────
    _log(f"\n" + "=" * 70)
    _log("[Test 2] High-systemic vs low-systemic genes — different aging?")
    _log("=" * 70)

    if "young" in pivot.columns and "old" in pivot.columns:
        valid = pivot.dropna(subset=["young", "old"])

        # Quartiles of baseline systemic fraction
        q25 = valid["young"].quantile(0.25)
        q75 = valid["young"].quantile(0.75)

        low_sys = valid[valid["young"] < q25]
        high_sys = valid[valid["young"] > q75]

        _log(f"  Low systemic (Q1, n={len(low_sys)}): median young frac = {low_sys['young'].median():.4f}")
        _log(f"    → Old: {low_sys['old'].median():.4f}, Δ = {(low_sys['old'] - low_sys['young']).median():+.4f}")

        _log(f"  High systemic (Q4, n={len(high_sys)}): median young frac = {high_sys['young'].median():.4f}")
        _log(f"    → Old: {high_sys['old'].median():.4f}, Δ = {(high_sys['old'] - high_sys['young']).median():+.4f}")

        # Statistical comparison
        delta_low = (low_sys["old"] - low_sys["young"])
        delta_high = (high_sys["old"] - high_sys["young"])
        u_stat, u_p = stats.mannwhitneyu(delta_low, delta_high)
        _log(f"  Low vs High Δ: Mann-Whitney p = {u_p:.2e}")

    # ── Test 3: Known covariates ──────────────────────────────────────
    _log(f"\n" + "=" * 70)
    _log("[Test 3] Do known covariates explain systemic growth?")
    _log("=" * 70)

    # For a subset of genes: does controlling for age/sex change systemic fraction?
    from sklearn.linear_model import LinearRegression

    # Simple: compute systemic component, then regress out age effect
    # If systemic growth = just shared aging (same age effect in all tissues)
    # then residualizing for age should eliminate it

    # Read a small subset of genes for this test
    test_genes = valid.nlargest(500, "young").index.tolist()[:200]  # top systemic genes

    _log(f"  Testing on {len(test_genes)} high-systemic genes...")

    # For each gene: compute systemic variance with and without age correction
    age_arr = ages.copy()
    sex_arr = np.array([1 if donor_sex.get(d) == "male" else 0 for d in common_donors])

    # Re-read these specific genes
    gene_test_data = {}
    remaining = set(test_genes)
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            parts = line.strip().split("\t", 2)
            gene = parts[1]
            if gene in remaining:
                vals = np.array(parts[2].split("\t"), dtype=np.float32)
                log_vals = np.log2(vals + 1)
                gene_test_data[gene] = log_vals[col_idx]
                remaining.discard(gene)
                if not remaining:
                    break

    _log(f"  Loaded {len(gene_test_data)} genes for covariate test")

    # For each gene: systemic var before/after age correction
    covar_results = []
    for gene, expr_mat in gene_test_data.items():
        systemic = expr_mat.mean(axis=1)

        # Residualize systemic for age + sex
        valid_mask = np.isfinite(age_arr)
        X_cov = np.column_stack([age_arr[valid_mask], sex_arr[valid_mask]])
        lr = LinearRegression()
        lr.fit(X_cov, systemic[valid_mask])
        resid = systemic.copy()
        resid[valid_mask] = systemic[valid_mask] - lr.predict(X_cov)

        # Variance before and after
        var_raw = np.var(systemic)
        var_resid = np.var(resid)
        frac_explained = 1 - var_resid / var_raw if var_raw > 0 else 0

        covar_results.append({
            "gene": gene,
            "systemic_var_raw": var_raw,
            "systemic_var_residual": var_resid,
            "frac_explained_by_age_sex": frac_explained,
        })

    df_cov = pd.DataFrame(covar_results)
    df_cov.to_csv(RESULTS_DIR / "test3_covariate_control.csv", index=False)

    _log(f"\n  Systemic variance explained by age+sex:")
    _log(f"    Median: {df_cov['frac_explained_by_age_sex'].median():.4f}")
    _log(f"    Mean: {df_cov['frac_explained_by_age_sex'].mean():.4f}")
    _log(f"    >10% explained: {(df_cov['frac_explained_by_age_sex'] > 0.1).sum()}/{len(df_cov)}")
    _log(f"    >50% explained: {(df_cov['frac_explained_by_age_sex'] > 0.5).sum()}/{len(df_cov)}")

    # Recompute systemic fraction after age-sex correction for young vs old
    _log(f"\n  Systemic fraction AFTER age-sex correction:")
    resid_results = []
    for gene, expr_mat in gene_test_data.items():
        # Correct each tissue for age+sex
        corrected = np.zeros_like(expr_mat)
        for ti in range(expr_mat.shape[1]):
            valid_mask = np.isfinite(age_arr)
            X_cov = np.column_stack([age_arr[valid_mask], sex_arr[valid_mask]])
            lr = LinearRegression()
            lr.fit(X_cov, expr_mat[valid_mask, ti])
            corrected[:, ti] = expr_mat[:, ti]
            corrected[valid_mask, ti] -= lr.predict(X_cov) - expr_mat[valid_mask, ti].mean()

        for age_label, mask in [("young", young_mask), ("old", old_mask)]:
            if mask.sum() < 20:
                continue
            e = corrected[mask]
            s = e.mean(axis=1)
            tv = np.var(e)
            sv = np.var(s)
            frac = sv / tv if tv > 1e-10 else np.nan
            resid_results.append({"gene": gene, "age_group": age_label, "frac_systemic_corrected": frac})

    df_resid = pd.DataFrame(resid_results)
    pivot_resid = df_resid.pivot_table(index="gene", columns="age_group",
                                         values="frac_systemic_corrected", aggfunc="first")
    if "young" in pivot_resid.columns and "old" in pivot_resid.columns:
        vr = pivot_resid.dropna()
        delta_corr = vr["old"] - vr["young"]
        _log(f"    Median Δ (corrected): {delta_corr.median():+.4f}")
        _log(f"    {(delta_corr > 0).sum()}/{len(delta_corr)} genes still have higher systemic in old")
        w, p = stats.wilcoxon(delta_corr)
        _log(f"    Wilcoxon p (corrected) = {p:.2e}")

    # ── FIGURES ────────────────────────────────────────────────────────
    _log(f"\n" + "=" * 70)
    _log("[Figures]")
    _log("=" * 70)

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Systemic fraction young vs old (scatter)
    ax = fig.add_subplot(gs[0, 0])
    if "young" in pivot.columns and "old" in pivot.columns:
        valid = pivot.dropna(subset=["young", "old"])
        # Subsample for plotting
        if len(valid) > 2000:
            plot_genes = valid.sample(2000, random_state=42)
        else:
            plot_genes = valid
        ax.scatter(plot_genes["young"], plot_genes["old"], s=2, alpha=0.1, c="gray")
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
        ax.set_xlabel("Systemic fraction (young)")
        ax.set_ylabel("Systemic fraction (old)")
        ax.set_title(f"A: Per-gene systemic fraction\nMedian Δ={delta.median():+.4f}, "
                     f"Wilcoxon p={w_p:.1e}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Panel B: Distribution of Δ(systemic fraction)
    ax = fig.add_subplot(gs[0, 1])
    if len(delta) > 0:
        ax.hist(delta.values, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(delta.median(), color="red", linewidth=2, linestyle="--",
                   label=f"median={delta.median():+.4f}")
        ax.set_xlabel("Δ systemic fraction (old − young)")
        ax.set_ylabel("Number of genes")
        ax.set_title(f"B: Distribution of systemic shift\n"
                     f"{n_higher}/{n_total} genes shift toward systemic")
        ax.legend()

    # Panel C: High vs low systemic genes
    ax = fig.add_subplot(gs[0, 2])
    if len(delta_low) > 0 and len(delta_high) > 0:
        bp = ax.boxplot([delta_low.values, delta_high.values],
                        labels=["Low systemic\n(Q1)", "High systemic\n(Q4)"],
                        patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor("lightcoral")
        bp["boxes"][1].set_facecolor("steelblue")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Δ systemic fraction (old − young)")
        ax.set_title(f"C: Low vs High systemic genes\nMann-Whitney p={u_p:.1e}")

    # Panel D: Variance components by age
    ax = fig.add_subplot(gs[1, 0])
    for ag, color in [("young", "tab:blue"), ("old", "tab:red")]:
        sub = df_genes[df_genes["age_group"] == ag]
        ax.scatter(sub["systemic_var"], sub["local_var"], s=1, alpha=0.05, c=color, label=ag)
    ax.set_xlabel("Systemic variance")
    ax.set_ylabel("Local variance")
    ax.set_title("D: Systemic vs Local variance")
    ax.legend()
    ax.set_xlim(0, ax.get_xlim()[1] * 0.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 0.5)

    # Panel E: Age+sex explains how much?
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(df_cov["frac_explained_by_age_sex"].values, bins=50,
            color="tab:orange", alpha=0.7, edgecolor="none")
    ax.axvline(df_cov["frac_explained_by_age_sex"].median(), color="red", linewidth=2,
               linestyle="--", label=f"median={df_cov['frac_explained_by_age_sex'].median():.3f}")
    ax.set_xlabel("Fraction of systemic variance explained by age+sex")
    ax.set_ylabel("Number of genes")
    ax.set_title("E: Known covariates explain systemic variance?")
    ax.legend()

    # Panel F: Summary — the principle
    ax = fig.add_subplot(gs[1, 2])
    ax.text(0.5, 0.85, "SYSTEMIC/LOCAL DECOMPOSITION", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.70, f"Donors with 6 tissues: {len(common_donors)}", ha="center", fontsize=10)
    ax.text(0.5, 0.58, f"Young (≤35): {young_mask.sum()} | Old (≥55): {old_mask.sum()}",
            ha="center", fontsize=10)
    ax.text(0.5, 0.42, f"Systemic fraction shift: {delta.median():+.4f}", ha="center", fontsize=14,
            color="red" if delta.median() > 0 else "blue", fontweight="bold")
    ax.text(0.5, 0.30, f"{n_higher}/{n_total} genes → MORE systemic in old",
            ha="center", fontsize=11)
    ax.text(0.5, 0.18, f"Wilcoxon p = {w_p:.1e}",
            ha="center", fontsize=11)
    ax.text(0.5, 0.06, f"Age+sex explain median {df_cov['frac_explained_by_age_sex'].median():.1%} of systemic var",
            ha="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle("Systemic vs Local: Does aging shift regulatory balance?\n"
                 f"GTEx — {len(common_donors)} donors × 6 tissues",
                 fontsize=14, fontweight="bold")
    fig.savefig(RESULTS_DIR / "systemic_local_decomposition.png", dpi=150)
    plt.close(fig)
    _log(f"\n  Saved systemic_local_decomposition.png")

    _log(f"\n  Total time: {time.time()-t0:.0f}s")
    _log(f"  Results: {RESULTS_DIR}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
