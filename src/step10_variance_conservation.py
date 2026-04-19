"""
Step 10: Variance conservation across levels — is there an invariant?

Three-level ANOVA decomposition:
  V_total = V_donor + V_tissue + V_residual

For each gene, in each age group:
  π_donor = V_donor / V_total
  π_tissue = V_tissue / V_total
  π_residual = V_residual / V_total

Questions:
1. Are proportions conserved across age? (invariant?)
2. Which level gains/loses variance share?
3. Is there a conservation law (sum = 1 trivially, but do individual
   proportions stay constant even as absolutes change)?
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
RESULTS_DIR = BASE / "results" / "step10_variance_conservation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg):
    print(msg, flush=True)


def main():
    t0 = time.time()
    _log("=" * 70)
    _log("VARIANCE CONSERVATION — Three-level ANOVA")
    _log("=" * 70)

    # ── Load metadata ────────────────────────────────────────────────
    samples = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    samples["age_mid"] = samples["AGE"].str.split("-").apply(
        lambda x: (int(x[0]) + int(x[1])) / 2 if isinstance(x, list) and len(x) == 2 else np.nan)

    sample_meta = samples.set_index("SAMPID")[["SUBJID", "SMTSD", "age_mid"]].copy()

    top6 = ["Muscle - Skeletal", "Whole Blood", "Skin - Sun Exposed (Lower leg)",
            "Adipose - Subcutaneous", "Artery - Tibial", "Thyroid"]

    tpm_path = DATA_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline()
        header = f.readline().strip().split("\t")
    sample_ids = header[2:]

    # Build donor×tissue index
    tissue_sample_map = {t: {} for t in top6}
    for i, sid in enumerate(sample_ids):
        if sid in sample_meta.index:
            row = sample_meta.loc[sid]
            t = row["SMTSD"]
            subj = row["SUBJID"]
            if t in top6:
                tissue_sample_map[t][subj] = i

    donor_sets = [set(tissue_sample_map[t].keys()) for t in top6]
    common_donors = sorted(donor_sets[0].intersection(*donor_sets[1:]))
    _log(f"  Donors with all 6 tissues: {len(common_donors)}")

    donor_ages = {}
    for subj in common_donors:
        for t in top6:
            idx = tissue_sample_map[t][subj]
            sid = sample_ids[idx]
            if sid in sample_meta.index:
                donor_ages[subj] = sample_meta.loc[sid, "age_mid"]
                break

    ages = np.array([donor_ages.get(d, np.nan) for d in common_donors])

    # Age bins: decades
    age_bins = {
        "20-39": (ages >= 20) & (ages < 40),
        "40-49": (ages >= 40) & (ages < 50),
        "50-59": (ages >= 50) & (ages < 60),
        "60-79": (ages >= 60) & (ages < 80),
    }
    for ab, mask in age_bins.items():
        _log(f"  {ab}: n={mask.sum()}")

    # Column index matrix
    col_idx = np.zeros((len(common_donors), len(top6)), dtype=int)
    for ti, t in enumerate(top6):
        for di, d in enumerate(common_donors):
            col_idx[di, ti] = tissue_sample_map[t][d]

    # ── ANOVA decomposition ──────────────────────────────────────────
    _log(f"\n[1] Three-level ANOVA decomposition...")

    results = []
    n_processed = 0

    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            parts = line.strip().split("\t", 2)
            gene = parts[1]
            vals = np.array(parts[2].split("\t"), dtype=np.float32)

            if np.median(vals) < 0.5:
                n_processed += 1
                continue

            log_vals = np.log2(vals + 1)
            n_processed += 1

            if n_processed % 5000 == 0:
                _log(f"    {n_processed} genes ({time.time()-t0:.0f}s)...")

            # Expression matrix: donors × tissues
            expr = log_vals[col_idx]  # (n_donors, 6)

            for ab, mask in age_bins.items():
                if mask.sum() < 15:
                    continue

                e = expr[mask]  # (n_donors_in_bin, 6)
                n_d, n_t = e.shape

                grand_mean = e.mean()

                # Donor means (across tissues)
                donor_means = e.mean(axis=1)  # (n_d,)
                # Tissue means (across donors)
                tissue_means = e.mean(axis=0)  # (n_t,)

                # SS decomposition (two-way without interaction for balanced design)
                SS_total = np.sum((e - grand_mean) ** 2)
                SS_donor = n_t * np.sum((donor_means - grand_mean) ** 2)
                SS_tissue = n_d * np.sum((tissue_means - grand_mean) ** 2)
                SS_residual = SS_total - SS_donor - SS_tissue

                if SS_total < 1e-10:
                    continue

                results.append({
                    "gene": gene, "age_bin": ab,
                    "n_donors": n_d,
                    "SS_total": SS_total,
                    "SS_donor": SS_donor,
                    "SS_tissue": SS_tissue,
                    "SS_residual": max(SS_residual, 0),
                    "pi_donor": SS_donor / SS_total,
                    "pi_tissue": SS_tissue / SS_total,
                    "pi_residual": max(SS_residual, 0) / SS_total,
                    "V_total": SS_total / (n_d * n_t),
                })

    df = pd.DataFrame(results)
    _log(f"\n  Total measurements: {len(df)}")
    _log(f"  Genes per age bin: {df.groupby('age_bin')['gene'].nunique().to_dict()}")

    # ── Results ───────────────────────────────────────────────────────
    _log(f"\n" + "=" * 70)
    _log("[2] Variance proportions by age")
    _log("=" * 70)

    _log(f"\n  {'Age bin':<10s} {'π_tissue':>10s} {'π_donor':>10s} {'π_residual':>10s} {'V_total':>10s}")
    summary_rows = []
    for ab in ["20-39", "40-49", "50-59", "60-79"]:
        sub = df[df["age_bin"] == ab]
        if len(sub) == 0:
            continue
        row = {
            "age_bin": ab,
            "pi_tissue_median": sub["pi_tissue"].median(),
            "pi_donor_median": sub["pi_donor"].median(),
            "pi_residual_median": sub["pi_residual"].median(),
            "V_total_median": sub["V_total"].median(),
            "pi_tissue_mean": sub["pi_tissue"].mean(),
            "pi_donor_mean": sub["pi_donor"].mean(),
            "pi_residual_mean": sub["pi_residual"].mean(),
            "n_genes": sub["gene"].nunique(),
        }
        summary_rows.append(row)
        _log(f"  {ab:<10s} {row['pi_tissue_median']:>10.4f} {row['pi_donor_median']:>10.4f} "
             f"{row['pi_residual_median']:>10.4f} {row['V_total_median']:>10.4f}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_DIR / "variance_proportions_summary.csv", index=False)

    # ── Statistical tests ─────────────────────────────────────────────
    _log(f"\n" + "=" * 70)
    _log("[3] Is π_tissue conserved?")
    _log("=" * 70)

    # Compare youngest vs oldest
    young_genes = df[df["age_bin"] == "20-39"].set_index("gene")
    old_genes = df[df["age_bin"] == "60-79"].set_index("gene")
    common_genes = young_genes.index.intersection(old_genes.index)

    _log(f"  Common genes: {len(common_genes)}")

    for pi_name in ["pi_tissue", "pi_donor", "pi_residual"]:
        y_vals = young_genes.loc[common_genes, pi_name]
        o_vals = old_genes.loc[common_genes, pi_name]
        delta = o_vals - y_vals

        n_higher = (delta > 0).sum()
        w_stat, w_p = stats.wilcoxon(delta)
        median_d = delta.median()
        mean_d = delta.mean()

        _log(f"\n  {pi_name}:")
        _log(f"    Young median: {y_vals.median():.5f}")
        _log(f"    Old median:   {o_vals.median():.5f}")
        _log(f"    Δ median:     {median_d:+.5f}")
        _log(f"    Δ mean:       {mean_d:+.5f}")
        _log(f"    Higher in old: {n_higher}/{len(common_genes)} ({n_higher/len(common_genes):.1%})")
        _log(f"    Wilcoxon p:   {w_p:.2e}")
        _log(f"    Cohen's d:    {mean_d / delta.std():.4f}")

    # ── Test: per-gene conservation ───────────────────────────────────
    _log(f"\n" + "=" * 70)
    _log("[4] Per-gene π_tissue stability")
    _log("=" * 70)

    # For each gene: Spearman ρ of π_tissue across age bins
    gene_stability = []
    for gene in common_genes[:5000]:  # sample for speed
        gene_data = df[df["gene"] == gene].sort_values("age_bin")
        if len(gene_data) < 3:
            continue
        age_mids = {"20-39": 30, "40-49": 45, "50-59": 55, "60-79": 70}
        x = [age_mids[ab] for ab in gene_data["age_bin"]]
        y = gene_data["pi_tissue"].values
        if len(x) >= 3:
            rho, p = stats.spearmanr(x, y)
            gene_stability.append({
                "gene": gene,
                "pi_tissue_slope_rho": rho,
                "pi_tissue_slope_p": p,
                "pi_tissue_young": gene_data[gene_data["age_bin"] == "20-39"]["pi_tissue"].values[0]
                if "20-39" in gene_data["age_bin"].values else np.nan,
            })

    df_stab = pd.DataFrame(gene_stability)
    df_stab.to_csv(RESULTS_DIR / "per_gene_stability.csv", index=False)

    n_sig_pos = ((df_stab["pi_tissue_slope_p"] < 0.05) & (df_stab["pi_tissue_slope_rho"] > 0)).sum()
    n_sig_neg = ((df_stab["pi_tissue_slope_p"] < 0.05) & (df_stab["pi_tissue_slope_rho"] < 0)).sum()
    _log(f"  Genes tested: {len(df_stab)}")
    _log(f"  π_tissue increasing with age (p<0.05): {n_sig_pos} ({n_sig_pos/len(df_stab):.1%})")
    _log(f"  π_tissue decreasing with age (p<0.05): {n_sig_neg} ({n_sig_neg/len(df_stab):.1%})")
    _log(f"  Stable (p≥0.05): {len(df_stab) - n_sig_pos - n_sig_neg} ({(len(df_stab) - n_sig_pos - n_sig_neg)/len(df_stab):.1%})")

    # ── FIGURES ────────────────────────────────────────────────────────
    _log(f"\n" + "=" * 70)
    _log("[5] Figures")
    _log("=" * 70)

    fig = plt.figure(figsize=(20, 12))
    gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Stacked bar — variance proportions by age
    ax = fig.add_subplot(gs_fig[0, 0])
    x = np.arange(len(df_summary))
    w = 0.6
    bottom = np.zeros(len(df_summary))
    for pi_name, color, label in [
        ("pi_tissue_median", "tab:green", "π_tissue"),
        ("pi_donor_median", "tab:blue", "π_donor"),
        ("pi_residual_median", "tab:gray", "π_residual"),
    ]:
        vals = df_summary[pi_name].values
        ax.bar(x, vals, w, bottom=bottom, color=color, alpha=0.7, label=label)
        # Add text
        for i, v in enumerate(vals):
            ax.text(x[i], bottom[i] + v / 2, f"{v:.3f}", ha="center", va="center", fontsize=8)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(df_summary["age_bin"].values)
    ax.set_ylabel("Variance proportion")
    ax.set_title("A: Variance decomposition by age\n(median across genes)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Panel B: π_tissue trend line
    ax = fig.add_subplot(gs_fig[0, 1])
    age_mids_plot = [30, 45, 55, 70]
    for pi_name, color, label in [
        ("pi_tissue_median", "tab:green", "π_tissue"),
        ("pi_donor_median", "tab:blue", "π_donor"),
        ("pi_residual_median", "tab:gray", "π_residual"),
    ]:
        vals = df_summary[pi_name].values
        ax.plot(age_mids_plot[:len(vals)], vals, "o-", color=color, label=label, linewidth=2, markersize=8)
    ax.set_xlabel("Age (midpoint)")
    ax.set_ylabel("Median variance proportion")
    ax.set_title("B: Variance proportions across decades")
    ax.legend()

    # Panel C: Young vs Old π_tissue scatter
    ax = fig.add_subplot(gs_fig[0, 2])
    if len(common_genes) > 0:
        y_vals = young_genes.loc[common_genes[:3000], "pi_tissue"]
        o_vals = old_genes.loc[common_genes[:3000], "pi_tissue"]
        ax.scatter(y_vals, o_vals, s=1, alpha=0.1, c="gray")
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
        rho_c, p_c = stats.spearmanr(y_vals, o_vals)
        ax.set_xlabel("π_tissue (young)")
        ax.set_ylabel("π_tissue (old)")
        ax.set_title(f"C: Per-gene tissue proportion\nρ={rho_c:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Panel D: Distribution of Δπ_tissue
    ax = fig.add_subplot(gs_fig[1, 0])
    if len(common_genes) > 0:
        delta_tissue = old_genes.loc[common_genes, "pi_tissue"] - young_genes.loc[common_genes, "pi_tissue"]
        ax.hist(delta_tissue.values, bins=100, color="tab:green", alpha=0.7, edgecolor="none")
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(delta_tissue.median(), color="red", linewidth=2, linestyle="--",
                   label=f"median={delta_tissue.median():+.5f}")
        ax.set_xlabel("Δ π_tissue (old − young)")
        ax.set_ylabel("Count")
        ax.set_title("D: Per-gene tissue proportion shift")
        ax.legend()

    # Panel E: V_total by age
    ax = fig.add_subplot(gs_fig[1, 1])
    for ab in ["20-39", "40-49", "50-59", "60-79"]:
        sub = df[df["age_bin"] == ab]
        ax.hist(np.log10(sub["V_total"].clip(1e-5)), bins=100, alpha=0.4,
                label=ab, density=True)
    ax.set_xlabel("log10(V_total)")
    ax.set_ylabel("Density")
    ax.set_title("E: Total variance distribution by age")
    ax.legend(fontsize=8)

    # Panel F: Summary
    ax = fig.add_subplot(gs_fig[1, 2])
    ax.text(0.5, 0.90, "VARIANCE CONSERVATION TEST", ha="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.78, f"{len(common_donors)} donors × 6 tissues × {df['gene'].nunique()} genes",
            ha="center", fontsize=10)

    # Get the key results
    y_tissue = df_summary[df_summary["age_bin"] == "20-39"]["pi_tissue_median"].values
    o_tissue = df_summary[df_summary["age_bin"] == "60-79"]["pi_tissue_median"].values
    if len(y_tissue) > 0 and len(o_tissue) > 0:
        delta_t = o_tissue[0] - y_tissue[0]
        ax.text(0.5, 0.62, f"π_tissue: {y_tissue[0]:.4f} → {o_tissue[0]:.4f} (Δ={delta_t:+.4f})",
                ha="center", fontsize=12, color="green", fontweight="bold")

    y_donor = df_summary[df_summary["age_bin"] == "20-39"]["pi_donor_median"].values
    o_donor = df_summary[df_summary["age_bin"] == "60-79"]["pi_donor_median"].values
    if len(y_donor) > 0 and len(o_donor) > 0:
        delta_d = o_donor[0] - y_donor[0]
        ax.text(0.5, 0.48, f"π_donor: {y_donor[0]:.4f} → {o_donor[0]:.4f} (Δ={delta_d:+.4f})",
                ha="center", fontsize=12, color="blue", fontweight="bold")

    y_resid = df_summary[df_summary["age_bin"] == "20-39"]["pi_residual_median"].values
    o_resid = df_summary[df_summary["age_bin"] == "60-79"]["pi_residual_median"].values
    if len(y_resid) > 0 and len(o_resid) > 0:
        delta_r = o_resid[0] - y_resid[0]
        ax.text(0.5, 0.34, f"π_residual: {y_resid[0]:.4f} → {o_resid[0]:.4f} (Δ={delta_r:+.4f})",
                ha="center", fontsize=12, color="gray", fontweight="bold")

    # Verdict
    if len(y_tissue) > 0 and len(o_tissue) > 0:
        max_delta = max(abs(delta_t), abs(delta_d), abs(delta_r))
        if max_delta < 0.01:
            verdict = "PROPORTIONS APPROXIMATELY CONSERVED"
            v_color = "green"
        elif max_delta < 0.03:
            verdict = "SMALL SHIFTS — NEAR-CONSERVATION"
            v_color = "orange"
        else:
            verdict = "NOT CONSERVED — SIGNIFICANT SHIFTS"
            v_color = "red"
        ax.text(0.5, 0.15, verdict, ha="center", fontsize=14, fontweight="bold", color=v_color)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle("Three-Level Variance Decomposition: Is There an Invariant?\n"
                 "V_total = V_tissue + V_donor + V_residual",
                 fontsize=14, fontweight="bold")
    fig.savefig(RESULTS_DIR / "variance_conservation.png", dpi=150)
    plt.close(fig)
    _log(f"\n  Saved variance_conservation.png")

    _log(f"\n  Total time: {time.time()-t0:.0f}s")
    _log(f"  Results: {RESULTS_DIR}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
