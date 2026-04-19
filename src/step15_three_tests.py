"""
Three critical tests for π_tissue near-invariant:

Test A: CR mechanism — noise reduction vs structure reinforcement (rat)
Test B: Scaling law — dπ/dt vs 1/lifespan (mouse TMS + rat + human)
Test C: Per-gene leakage — which programs erode? (GTEx GSEA)
"""

import time, gzip
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse
import scanpy as sc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings; warnings.filterwarnings("ignore")

BASE_CA = Path("/Users/teo/Desktop/research/coupling_atlas")
BASE_OSC = Path("/Users/teo/Desktop/research/oscilatory")
RESULTS_DIR = BASE_CA / "results" / "step15_three_tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GTEx_DIR = BASE_CA / "data" / "gtex"

def _log(m): print(m, flush=True)

# ══════════════════════════════════════════════════════════════════
# TEST A: CR mechanism on rat data
# ══════════════════════════════════════════════════════════════════
def test_a_cr_mechanism():
    _log("\n" + "=" * 60)
    _log("TEST A: CR mechanism — noise reduction vs structure reinforcement")
    _log("=" * 60)

    adata = sc.read_h5ad(BASE_OSC / "results/h010_rat_cr/data/rat_atlas.h5ad")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Pseudobulk per GSM
    pb = {}
    for gsm, grp in adata.obs.groupby("GSM"):
        if len(grp) < 30: continue
        idx = np.array([adata.obs.index.get_loc(i) for i in grp.index])
        mat = adata.X[idx]
        if issparse(mat): mat = np.asarray(mat.todense())
        pb[gsm] = {"condition": grp["condition"].iloc[0], "tissue": grp["tissue"].iloc[0],
                    "expr": mat.mean(axis=0).flatten()}

    n_genes = len(next(iter(pb.values()))["expr"])
    results = []

    for cond in ["young", "old_AL", "old_CR"]:
        gsms = {k: v for k, v in pb.items() if v["condition"] == cond}
        tissues = sorted(set(v["tissue"] for v in gsms.values()))
        gsm_list = sorted(gsms.keys())

        mat = np.zeros((len(gsm_list), n_genes))
        t_labels = []
        for i, g in enumerate(gsm_list):
            mat[i] = gsms[g]["expr"]
            t_labels.append(gsms[g]["tissue"])
        t_labels = np.array(t_labels)

        v_tissue_list, v_resid_list, v_total_list = [], [], []
        for g in range(n_genes):
            vals = mat[:, g]
            if np.std(vals) < 1e-6: continue
            gm = np.mean(vals)
            ss_total = np.sum((vals - gm)**2)
            if ss_total < 1e-10: continue
            ss_tissue = sum((t_labels==t).sum() * (np.mean(vals[t_labels==t]) - gm)**2 for t in tissues)
            ss_resid = max(ss_total - ss_tissue, 0)
            n = len(vals)
            v_tissue_list.append(ss_tissue / n)
            v_resid_list.append(ss_resid / n)
            v_total_list.append(ss_total / n)

        results.append({
            "condition": cond,
            "V_tissue_median": np.median(v_tissue_list),
            "V_residual_median": np.median(v_resid_list),
            "V_total_median": np.median(v_total_list),
            "pi_tissue": np.median([vt/(vt+vr) if (vt+vr)>0 else 0
                                     for vt, vr in zip(v_tissue_list, v_resid_list)]),
        })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "test_a_cr_mechanism.csv", index=False)

    _log(f"\n  {'Condition':<10s} {'V_tissue':>10s} {'V_residual':>10s} {'V_total':>10s} {'π_tissue':>10s}")
    for _, r in df.iterrows():
        _log(f"  {r['condition']:<10s} {r['V_tissue_median']:>10.4f} {r['V_residual_median']:>10.4f} "
             f"{r['V_total_median']:>10.4f} {r['pi_tissue']:>10.4f}")

    y = df[df["condition"]=="young"].iloc[0]
    o = df[df["condition"]=="old_AL"].iloc[0]
    c = df[df["condition"]=="old_CR"].iloc[0]

    _log(f"\n  Aging effect:")
    _log(f"    V_tissue:   {o['V_tissue_median'] - y['V_tissue_median']:+.4f}")
    _log(f"    V_residual: {o['V_residual_median'] - y['V_residual_median']:+.4f}")
    _log(f"\n  CR effect (vs old_AL):")
    _log(f"    V_tissue:   {c['V_tissue_median'] - o['V_tissue_median']:+.4f}")
    _log(f"    V_residual: {c['V_residual_median'] - o['V_residual_median']:+.4f}")

    # Verdict
    vt_cr = c['V_tissue_median'] - o['V_tissue_median']
    vr_cr = c['V_residual_median'] - o['V_residual_median']
    if vr_cr < 0 and abs(vt_cr) < abs(vr_cr):
        _log(f"\n  VERDICT: Hypothesis A (NOISE REDUCTION) — CR reduces residual noise")
    elif vt_cr > 0 and abs(vt_cr) > abs(vr_cr):
        _log(f"\n  VERDICT: Hypothesis B (STRUCTURE REINFORCEMENT) — CR strengthens tissue signal")
    elif vr_cr < 0 and vt_cr > 0:
        _log(f"\n  VERDICT: BOTH — CR reduces noise AND strengthens structure")
    else:
        _log(f"\n  VERDICT: Neither clearly dominant")

    del adata
    return df


# ══════════════════════════════════════════════════════════════════
# TEST B: Scaling law — mouse TMS pseudobulk
# ══════════════════════════════════════════════════════════════════
def test_b_scaling_law():
    _log("\n" + "=" * 60)
    _log("TEST B: Scaling law — dπ/dt vs 1/lifespan")
    _log("=" * 60)

    # Mouse TMS FACS — pseudobulk per mouse × tissue
    adata = sc.read_h5ad(BASE_OSC / "data/tms/tms_facs.h5ad")
    _log(f"  TMS: {adata.shape[0]} cells, {adata.shape[1]} genes")

    adata.obs["age_months"] = pd.to_numeric(
        adata.obs["age"].astype(str).str.replace("m","").str.strip(), errors="coerce")

    # Need mouse IDs — check if available
    mouse_col = None
    for c in ["mouse.id", "mouse_id", "individual", "channel"]:
        if c in adata.obs.columns:
            mouse_col = c
            break
    if mouse_col is None:
        # Try to construct from metadata
        _log(f"  Available columns: {adata.obs.columns.tolist()[:15]}")
        # Use combination of age + sex + batch as proxy for mouse
        adata.obs["mouse_proxy"] = (adata.obs["age"].astype(str) + "_" +
                                     adata.obs["sex"].astype(str) + "_" +
                                     adata.obs["tissue"].astype(str))
        mouse_col = "mouse_proxy"
        _log(f"  Using proxy: age×sex×tissue as sample unit")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Pseudobulk per mouse_proxy (= age×sex×tissue combination)
    # For ANOVA we need: per age group, samples grouped by tissue
    top_tissues = adata.obs["tissue"].value_counts().head(6).index.tolist()
    _log(f"  Top tissues: {top_tissues}")

    age_groups = {3: "3mo", 18: "18mo", 21: "21mo", 24: "24mo"}
    mouse_results = []

    for age_m, age_label in age_groups.items():
        mask = (adata.obs["age_months"] == age_m) & (adata.obs["tissue"].isin(top_tissues))
        sub = adata[mask]
        if sub.shape[0] < 100: continue

        # Pseudobulk per tissue×sex (= one "sample" per combination)
        pb_mat = {}
        for (tissue, sex), grp in sub.obs.groupby(["tissue", "sex"]):
            if len(grp) < 20: continue
            idx = np.array([sub.obs.index.get_loc(i) for i in grp.index])
            mat = sub.X[idx]
            if issparse(mat): mat = np.asarray(mat.todense())
            key = f"{tissue}_{sex}"
            pb_mat[key] = {"tissue": tissue, "expr": mat.mean(axis=0).flatten()}

        if len(pb_mat) < 6: continue

        # ANOVA
        tissues = sorted(set(v["tissue"] for v in pb_mat.values()))
        keys = sorted(pb_mat.keys())
        n_genes = len(next(iter(pb_mat.values()))["expr"])
        expr = np.zeros((len(keys), n_genes))
        t_labels = np.array([pb_mat[k]["tissue"] for k in keys])
        for i, k in enumerate(keys):
            expr[i] = pb_mat[k]["expr"]

        pi_vals = []
        for g in range(n_genes):
            vals = expr[:, g]
            if np.std(vals) < 1e-6: continue
            gm = np.mean(vals)
            ss_total = np.sum((vals - gm)**2)
            if ss_total < 1e-10: continue
            ss_tissue = sum((t_labels==t).sum() * (np.mean(vals[t_labels==t]) - gm)**2 for t in tissues)
            pi_vals.append(ss_tissue / ss_total)

        med_pi = np.median(pi_vals)
        _log(f"  Mouse {age_label}: π_tissue = {med_pi:.4f} (n={len(keys)} samples, {len(pi_vals)} genes)")
        mouse_results.append({"species": "mouse", "age_months": age_m, "age_label": age_label,
                              "pi_tissue": med_pi, "n_samples": len(keys), "n_genes": len(pi_vals)})

    del adata

    # Compute dπ/dt for mouse
    df_mouse = pd.DataFrame(mouse_results)
    if len(df_mouse) >= 2:
        ages_yr = df_mouse["age_months"].values / 12
        pis = df_mouse["pi_tissue"].values
        slope, intercept, r, p, se = stats.linregress(ages_yr, pis)
        _log(f"\n  Mouse: dπ/dt = {slope:+.4f}/year (p={p:.4f})")
    else:
        slope = np.nan

    # Compile three-species comparison
    _log(f"\n  THREE-SPECIES COMPARISON:")
    species_data = [
        {"species": "Mouse", "lifespan_yr": 2.5, "dpi_dt": slope,
         "observation_span_yr": (24-3)/12},
        {"species": "Rat", "lifespan_yr": 3.0, "dpi_dt": -0.051 / (22/12),
         "observation_span_yr": 22/12},
        {"species": "Human", "lifespan_yr": 80, "dpi_dt": -0.031 / 40,
         "observation_span_yr": 40},
    ]

    df_species = pd.DataFrame(species_data)
    df_species["total_erosion"] = df_species["dpi_dt"].abs() * df_species["lifespan_yr"]
    df_species["inv_lifespan"] = 1 / df_species["lifespan_yr"]
    df_species.to_csv(RESULTS_DIR / "test_b_scaling_law.csv", index=False)

    _log(f"\n  {'Species':<8s} {'Lifespan':>10s} {'dπ/dt':>12s} {'Total erosion':>15s} {'k=|dπ/dt|×L':>12s}")
    for _, r in df_species.iterrows():
        _log(f"  {r['species']:<8s} {r['lifespan_yr']:>8.1f} yr {r['dpi_dt']:>+12.5f}/yr "
             f"{r['total_erosion']:>12.4f}     {r['total_erosion']:>12.4f}")

    # Fit: |dπ/dt| = k / L
    valid = df_species.dropna(subset=["dpi_dt"])
    if len(valid) >= 2:
        rho, p = stats.spearmanr(valid["inv_lifespan"], valid["dpi_dt"].abs())
        _log(f"\n  Correlation |dπ/dt| vs 1/L: ρ = {rho:+.3f}, p = {p:.4f}")

        # Fit k
        k_values = valid["total_erosion"].values
        _log(f"  k estimates: {k_values}")
        _log(f"  k mean: {np.mean(k_values):.4f} ± {np.std(k_values):.4f}")

        # Predictions
        k = np.mean(k_values)
        for name, L in [("NMR", 30), ("Bowhead whale", 200), ("Dog", 13), ("Cat", 20)]:
            pred = k / L
            _log(f"  Prediction: {name} (L={L}yr): dπ/dt = {-pred:+.6f}/year")

    return df_species


# ══════════════════════════════════════════════════════════════════
# TEST C: Per-gene leakage — which programs erode?
# ══════════════════════════════════════════════════════════════════
def test_c_gene_leakage():
    _log("\n" + "=" * 60)
    _log("TEST C: Per-gene leakage — which programs erode?")
    _log("=" * 60)

    # Load pre-computed per-gene stability from step10
    stab_path = BASE_CA / "results" / "step10_variance_conservation" / "per_gene_stability.csv"
    if not stab_path.exists():
        _log(f"  Per-gene stability file not found, computing from GTEx...")
        # Recompute from raw data
        return None

    df_stab = pd.read_csv(stab_path)
    _log(f"  Per-gene data: {len(df_stab)} genes")

    # We need Δπ_tissue (old - young) per gene
    # Load full ANOVA results
    # Re-read step10 raw output
    samples = pd.read_csv(GTEx_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(GTEx_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    samples["age_mid"] = samples["AGE"].str.split("-").apply(
        lambda x: (int(x[0]) + int(x[1])) / 2 if isinstance(x, list) and len(x) == 2 else np.nan)
    meta = samples.set_index("SAMPID")[["SUBJID", "SMTSD", "age_mid"]]

    TOP6 = ["Muscle - Skeletal", "Whole Blood", "Skin - Sun Exposed (Lower leg)",
            "Adipose - Subcutaneous", "Artery - Tibial", "Thyroid"]

    tpm_path = GTEx_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline()
        header = f.readline().strip().split("\t")
    sample_ids = header[2:]

    # Find donors with all 6 tissues
    tissue_sample_map = {t: {} for t in TOP6}
    for i, sid in enumerate(sample_ids):
        if sid in meta.index:
            r = meta.loc[sid]
            if r["SMTSD"] in TOP6:
                tissue_sample_map[r["SMTSD"]][r["SUBJID"]] = i

    donor_sets = [set(tissue_sample_map[t].keys()) for t in TOP6]
    common_donors = sorted(donor_sets[0].intersection(*donor_sets[1:]))

    donor_ages = {}
    for d in common_donors:
        for t in TOP6:
            sid = sample_ids[tissue_sample_map[t][d]]
            if sid in meta.index:
                donor_ages[d] = meta.loc[sid, "age_mid"]
                break

    ages = np.array([donor_ages[d] for d in common_donors])
    young_mask = ages < 40
    old_mask = ages >= 60

    col_idx = np.zeros((len(common_donors), len(TOP6)), dtype=int)
    for ti, t in enumerate(TOP6):
        for di, d in enumerate(common_donors):
            col_idx[di, ti] = tissue_sample_map[t][d]

    _log(f"  Donors: {len(common_donors)} (young: {young_mask.sum()}, old: {old_mask.sum()})")

    # Per-gene Δπ
    gene_delta = []
    n_proc = 0
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            parts = line.strip().split("\t", 2)
            gene = parts[1]
            vals = np.array(parts[2].split("\t"), dtype=np.float32)
            if np.median(vals) < 0.5:
                n_proc += 1; continue
            log_vals = np.log2(vals + 1)
            n_proc += 1
            if n_proc % 10000 == 0:
                _log(f"    {n_proc} genes ({len(gene_delta)} kept)")

            expr = log_vals[col_idx]  # donors × tissues

            for age_label, mask in [("young", young_mask), ("old", old_mask)]:
                e = expr[mask]
                n_d, n_t = e.shape
                if n_d < 10: continue
                gm = e.mean()
                ss_total = np.sum((e - gm)**2)
                if ss_total < 1e-10: continue
                tissue_means = e.mean(axis=0)
                ss_tissue = n_d * np.sum((tissue_means - gm)**2)
                pi = ss_tissue / ss_total

                if age_label == "young":
                    pi_y = pi
                else:
                    pi_o = pi

            if "pi_y" in dir() and "pi_o" in dir():
                gene_delta.append({"gene": gene, "pi_young": pi_y, "pi_old": pi_o,
                                   "delta_pi": pi_o - pi_y})
            pi_y = pi_o = None  # reset

    df_delta = pd.DataFrame(gene_delta)
    df_delta = df_delta.sort_values("delta_pi")
    df_delta.to_csv(RESULTS_DIR / "test_c_per_gene_leakage.csv", index=False)
    _log(f"\n  Genes with Δπ: {len(df_delta)}")

    # Top losers and gainers
    _log(f"\n  Top 20 LOSING tissue identity (most negative Δπ):")
    for _, r in df_delta.head(20).iterrows():
        _log(f"    {r['gene']:<15s}: π {r['pi_young']:.3f} → {r['pi_old']:.3f} (Δ={r['delta_pi']:+.4f})")

    _log(f"\n  Top 20 GAINING tissue identity (most positive Δπ):")
    for _, r in df_delta.tail(20).iterrows():
        _log(f"    {r['gene']:<15s}: π {r['pi_young']:.3f} → {r['pi_old']:.3f} (Δ={r['delta_pi']:+.4f})")

    # Summary stats
    _log(f"\n  Overall: median Δπ = {df_delta['delta_pi'].median():+.5f}")
    _log(f"  Losing (Δπ < 0): {(df_delta['delta_pi'] < 0).sum()} ({(df_delta['delta_pi'] < 0).mean():.0%})")
    _log(f"  Gaining (Δπ > 0): {(df_delta['delta_pi'] > 0).sum()} ({(df_delta['delta_pi'] > 0).mean():.0%})")

    # Quick enrichment: categorize known gene sets
    known_sets = {
        "ECM/collagen": {"COL1A1","COL1A2","COL3A1","COL4A1","COL5A1","COL6A1","FN1","LAMA1","LAMB1"},
        "Inflammation": {"IL6","TNF","CCL2","CXCL8","IL1B","ICAM1","VCAM1","NFKBIA","RELA"},
        "Housekeeping": {"ACTB","GAPDH","B2M","PPIA","RPL13A","RPS18","HPRT1","TBP"},
        "TFs": {"SMAD3","SMAD2","TP53","HIF1A","ESR1","AR","PPARG","SOX9","FOXO1","RUNX2"},
        "Senescence": {"CDKN2A","CDKN1A","TP53","MDM2","RB1","SERPINE1"},
        "Chromatin": {"DNMT1","DNMT3A","DNMT3B","TET1","TET2","HDAC1","HDAC2","KDM5A","EZH2","SIRT1"},
    }

    _log(f"\n  Category-level Δπ:")
    for cat, genes in known_sets.items():
        sub = df_delta[df_delta["gene"].isin(genes)]
        if len(sub) >= 2:
            _log(f"    {cat:<18s}: n={len(sub)}, median Δπ = {sub['delta_pi'].median():+.5f}, "
                 f"mean = {sub['delta_pi'].mean():+.5f}")

    return df_delta


def main():
    t0 = time.time()
    _log("=" * 60)
    _log("THREE CRITICAL TESTS")
    _log("=" * 60)

    df_a = test_a_cr_mechanism()
    df_b = test_b_scaling_law()
    df_c = test_c_gene_leakage()

    # ── Summary figure ──
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: CR absolute variances
    ax = fig.add_subplot(gs[0, 0])
    if df_a is not None:
        x = np.arange(3)
        w = 0.35
        ax.bar(x - w/2, df_a["V_tissue_median"], w, label="V_tissue", color="tab:green", alpha=0.7)
        ax.bar(x + w/2, df_a["V_residual_median"], w, label="V_residual", color="tab:gray", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(df_a["condition"])
        ax.set_ylabel("Median variance")
        ax.set_title("A: CR mechanism\n(absolute variances)")
        ax.legend()

    # Panel B: Scaling law
    ax = fig.add_subplot(gs[0, 1])
    if df_b is not None and len(df_b) >= 2:
        valid = df_b.dropna(subset=["dpi_dt"])
        ax.scatter(1/valid["lifespan_yr"], valid["dpi_dt"].abs(), s=100, zorder=5)
        for _, r in valid.iterrows():
            ax.annotate(r["species"], (1/r["lifespan_yr"], abs(r["dpi_dt"])),
                        fontsize=10, ha="left", va="bottom")
        # Fit line
        if len(valid) >= 2:
            x_fit = np.linspace(0, max(1/valid["lifespan_yr"])*1.2, 100)
            k = valid["total_erosion"].mean()
            ax.plot(x_fit, k * x_fit, "r--", alpha=0.5, label=f"k={k:.3f}")
        ax.set_xlabel("1 / Lifespan (1/years)")
        ax.set_ylabel("|dπ/dt| (per year)")
        ax.set_title("B: Scaling law\n|dπ/dt| = k / Lifespan?")
        ax.legend()

    # Panel C: Δπ histogram
    ax = fig.add_subplot(gs[0, 2])
    if df_c is not None:
        ax.hist(df_c["delta_pi"], bins=100, color="steelblue", alpha=0.7, edgecolor="none")
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(df_c["delta_pi"].median(), color="red", linewidth=2, linestyle="--",
                   label=f"median={df_c['delta_pi'].median():+.4f}")
        ax.set_xlabel("Δπ_tissue (old - young)")
        ax.set_title("C: Per-gene tissue identity change")
        ax.legend()

    # Panel D: Top losers/gainers
    ax = fig.add_subplot(gs[1, 0])
    if df_c is not None:
        top_lose = df_c.head(15)
        top_gain = df_c.tail(15)
        combined = pd.concat([top_lose, top_gain])
        colors = ["tab:red"]*15 + ["tab:green"]*15
        ax.barh(range(30), combined["delta_pi"].values, color=colors, alpha=0.7)
        ax.set_yticks(range(30))
        ax.set_yticklabels(combined["gene"].values, fontsize=6)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Δπ_tissue")
        ax.set_title("D: Top losers (red) & gainers (green)")

    # Panel E: Species comparison table
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    if df_b is not None:
        table_data = []
        for _, r in df_b.iterrows():
            table_data.append([r["species"], f"{r['lifespan_yr']:.0f}",
                              f"{r['dpi_dt']:+.5f}" if not np.isnan(r['dpi_dt']) else "N/A",
                              f"{r['total_erosion']:.4f}" if not np.isnan(r['total_erosion']) else "N/A"])
        table = ax.table(cellText=table_data,
                         colLabels=["Species", "Lifespan (yr)", "dπ/dt (/yr)", "k = |dπ/dt|×L"],
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title("E: Scaling law data")

    fig.suptitle("Three Critical Tests for π_tissue Near-Invariant", fontsize=14, fontweight="bold")
    fig.savefig(RESULTS_DIR / "three_tests_summary.png", dpi=150)
    plt.close()
    _log(f"\n  Saved three_tests_summary.png")
    _log(f"  Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
