"""
Step 7: Two-Compartment Validation — Confounds + Level 1

Part A: Composition confound control
  - Estimate immune fraction per sample using marker gene signature scores
  - Residualize expression for immune fraction → recompute cross-tissue ρ
  - Does two-compartment finding survive?

Part B: Level 1 — Gene-gene coordination within tissue
  - For each tissue × age group: gene-gene Spearman ρ matrix (top 2000 genes)
  - Mean |ρ| = coordination index
  - Prediction: solid tissues increase, blood decreases

Part C: Level 4 confirmation — Donor-donor pairwise similarity
  - For each tissue × age decade: pairwise donor ρ
  - Mean donor-donor ρ = similarity index

Part D: Clonal hematopoiesis drivers in blood
  - Variance of CH drivers (DNMT3A, TET2, ASXL1) vs age
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
RESULTS_DIR = BASE / "results" / "step7_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg):
    print(msg, flush=True)


def spearman_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan, 1.0
    return stats.spearmanr(x, y)


# Immune signature genes (high specificity for immune cells)
IMMUNE_SIGNATURE = [
    "PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "CD19", "MS4A1",
    "CD14", "CD68", "ITGAM", "NKG7", "GNLY",
    "HLA-DRA", "HLA-DRB1", "CCL5", "GZMB",
]

# Fibroblast/stromal signature
FIBRO_SIGNATURE = [
    "COL1A1", "COL1A2", "COL3A1", "FN1", "DCN", "LUM",
    "FAP", "PDGFRA", "THY1", "VIM",
]

# CH driver genes
CH_DRIVERS = ["DNMT3A", "TET2", "ASXL1", "JAK2", "SF3B1", "SRSF2", "PPM1D", "TP53"]


def main():
    t0 = time.time()
    _log("=" * 70)
    _log("TWO-COMPARTMENT VALIDATION")
    _log("=" * 70)

    # ── Load metadata ────────────────────────────────────────────────
    samples = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    samples["sex"] = samples["SEX"].map({1: "male", 2: "female"})
    samples["age_mid"] = samples["AGE"].str.split("-").apply(
        lambda x: (int(x[0]) + int(x[1])) / 2 if isinstance(x, list) and len(x) == 2 else np.nan)
    samples["hardy"] = samples.get("DTHHRDY", np.nan)

    sample_meta = samples.set_index("SAMPID")[["SUBJID", "SMTSD", "sex", "age_mid", "hardy"]].copy()

    tissue_counts = samples["SMTSD"].value_counts()
    top6 = tissue_counts.head(6).index.tolist()

    tpm_path = DATA_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline()
        header = f.readline().strip().split("\t")
    sample_ids = header[2:]

    # Build tissue → sample indices
    tissue_samples = {}
    for i, sid in enumerate(sample_ids):
        if sid in sample_meta.index:
            tissue = sample_meta.loc[sid, "SMTSD"]
            if tissue in top6:
                if tissue not in tissue_samples:
                    tissue_samples[tissue] = []
                tissue_samples[tissue].append((i, sid))

    # Donor lookups
    donor_tissue = {}
    donor_meta = {}
    for i, sid in enumerate(sample_ids):
        if sid in sample_meta.index:
            m = sample_meta.loc[sid]
            if m["SMTSD"] in top6:
                d = m["SUBJID"]
                if d not in donor_tissue:
                    donor_tissue[d] = {}
                    donor_meta[d] = {"age": m["age_mid"], "sex": m["sex"], "hardy": m["hardy"]}
                donor_tissue[d][m["SMTSD"]] = i

    multi = {d: ts for d, ts in donor_tissue.items() if len(ts) >= 2}
    _log(f"  {len(multi)} multi-tissue donors, {len(top6)} tissues")

    # ── Load key genes + full matrix for top 2000 variable ──────────
    _log(f"\n[0] Streaming TPM — collecting expression matrices...")

    # First pass: collect signature genes + compute variance to select top 2000
    sig_genes = set(IMMUNE_SIGNATURE + FIBRO_SIGNATURE + CH_DRIVERS)
    sig_data = {}
    gene_vars = []

    # For Level 1: need donor × gene matrices per tissue
    # Strategy: collect top 2000 variable genes across all samples
    # Two-pass: pass 1 = get variances, pass 2 = collect data

    _log("  Pass 1: computing gene variances...")
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            parts = line.strip().split("\t", 2)
            gene_name = parts[1]
            vals = np.array(parts[2].split("\t"), dtype=np.float32)
            log_vals = np.log2(vals + 1)

            if np.median(vals) < 0.5:
                continue

            v = np.var(log_vals)
            gene_vars.append((gene_name, v))

            if gene_name in sig_genes:
                sig_data[gene_name] = log_vals

    gene_vars.sort(key=lambda x: -x[1])
    top2000 = set(g for g, _ in gene_vars[:2000])
    _log(f"  {len(gene_vars)} expressed genes, selected top 2000 variable")
    _log(f"  Signature genes found: {len(sig_data)}/{len(sig_genes)}")

    # Pass 2: collect top 2000 genes
    _log("  Pass 2: collecting top 2000 genes...")
    gene_matrix = {}
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            parts = line.strip().split("\t", 2)
            gene_name = parts[1]
            if gene_name in top2000:
                vals = np.array(parts[2].split("\t"), dtype=np.float32)
                gene_matrix[gene_name] = np.log2(vals + 1)
                if len(gene_matrix) >= 2000:
                    break

    _log(f"  Collected {len(gene_matrix)} genes")

    # ═══════════════════════════════════════════════════════════════════
    # PART A: Composition confound
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[A] Composition confound — immune fraction estimation")
    _log("=" * 70)

    # Compute immune score per sample (mean of immune signature genes)
    immune_genes_found = [g for g in IMMUNE_SIGNATURE if g in sig_data]
    fibro_genes_found = [g for g in FIBRO_SIGNATURE if g in sig_data]
    _log(f"  Immune signature genes: {len(immune_genes_found)}")
    _log(f"  Fibroblast signature genes: {len(fibro_genes_found)}")

    immune_score = np.zeros(len(sample_ids))
    for g in immune_genes_found:
        immune_score += sig_data[g]
    immune_score /= max(len(immune_genes_found), 1)

    fibro_score = np.zeros(len(sample_ids))
    for g in fibro_genes_found:
        fibro_score += sig_data[g]
    fibro_score /= max(len(fibro_genes_found), 1)

    # Does immune score change with age?
    _log(f"\n  Immune score vs age by tissue:")
    for tissue in top6:
        t_samples = tissue_samples[tissue]
        ages = [sample_meta.loc[sid, "age_mid"] for _, sid in t_samples]
        scores = [immune_score[i] for i, _ in t_samples]
        rho, p = spearman_safe(np.array(ages), np.array(scores))
        t_short = tissue.split(" - ")[-1][:20]
        _log(f"    {t_short:<20s}: ρ(age, immune_score) = {rho:+.3f}, p = {p:.2e}")

    # Residualize gene expression for immune score → recompute cross-tissue ρ
    _log(f"\n  Residualizing top 2000 genes for immune score...")

    residual_matrix = {}
    for gene, vals in gene_matrix.items():
        lr = LinearRegression()
        valid = np.isfinite(immune_score)
        lr.fit(immune_score[valid].reshape(-1, 1), vals[valid])
        resid = vals.copy()
        resid[valid] = vals[valid] - lr.predict(immune_score[valid].reshape(-1, 1))
        residual_matrix[gene] = resid

    # Recompute cross-tissue ρ on residuals
    _log(f"\n  Cross-tissue ρ: RAW vs IMMUNE-RESIDUALIZED")

    young_donors = [d for d in multi if donor_meta[d]["age"] <= 35]
    old_donors = [d for d in multi if donor_meta[d]["age"] >= 55]

    tissue_pairs = []
    for i, t1 in enumerate(top6[:5]):
        for t2 in top6[i+1:]:
            tissue_pairs.append((t1, t2))

    confound_results = []
    for t1, t2 in tissue_pairs:
        t1s = t1.split(" - ")[-1][:12]
        t2s = t2.split(" - ")[-1][:12]

        for age_label, donors in [("young", young_donors), ("old", old_donors)]:
            common = [d for d in donors if t1 in donor_tissue[d] and t2 in donor_tissue[d]]
            if len(common) < 20:
                continue

            raw_rhos = []
            resid_rhos = []
            for gene in list(gene_matrix.keys())[:500]:  # Use 500 for speed
                v1_raw = np.array([gene_matrix[gene][donor_tissue[d][t1]] for d in common])
                v2_raw = np.array([gene_matrix[gene][donor_tissue[d][t2]] for d in common])
                r_raw, _ = spearman_safe(v1_raw, v2_raw)
                if np.isfinite(r_raw):
                    raw_rhos.append(r_raw)

                v1_res = np.array([residual_matrix[gene][donor_tissue[d][t1]] for d in common])
                v2_res = np.array([residual_matrix[gene][donor_tissue[d][t2]] for d in common])
                r_res, _ = spearman_safe(v1_res, v2_res)
                if np.isfinite(r_res):
                    resid_rhos.append(r_res)

            if raw_rhos and resid_rhos:
                confound_results.append({
                    "t1": t1s, "t2": t2s, "age": age_label,
                    "median_raw": np.median(raw_rhos),
                    "median_resid": np.median(resid_rhos),
                    "delta": np.median(resid_rhos) - np.median(raw_rhos),
                    "n_donors": len(common), "n_genes": len(raw_rhos),
                })

    df_conf = pd.DataFrame(confound_results)
    df_conf.to_csv(RESULTS_DIR / "confound_immune_residual.csv", index=False)

    _log(f"\n  {'Pair':<25s} {'Age':>5s} {'Raw ρ':>8s} {'Resid ρ':>8s} {'Δ':>8s}")
    for _, r in df_conf.iterrows():
        _log(f"  {r['t1']+'×'+r['t2']:<25s} {r['age']:>5s} "
             f"{r['median_raw']:+.4f} {r['median_resid']:+.4f} {r['delta']:+.4f}")

    # Key test: does blood decoupling survive residualization?
    _log(f"\n  KEY TEST: Blood decoupling after immune residualization:")
    blood_pairs = df_conf[df_conf["t1"].str.contains("Blood") | df_conf["t2"].str.contains("Blood")]
    solid_pairs = df_conf[~(df_conf["t1"].str.contains("Blood") | df_conf["t2"].str.contains("Blood"))]

    for label, sub in [("Blood-involved", blood_pairs), ("Solid-solid", solid_pairs)]:
        y = sub[sub["age"] == "young"]["median_resid"]
        o = sub[sub["age"] == "old"]["median_resid"]
        if len(y) > 0 and len(o) > 0:
            _log(f"    {label}: young resid ρ = {y.median():+.4f}, old resid ρ = {o.median():+.4f}, "
                 f"Δ = {o.median()-y.median():+.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART B: Level 1 — Gene-gene coordination within tissue
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[B] Level 1 — Gene-gene coordination within tissue")
    _log("=" * 70)

    # For each tissue × age group: compute gene-gene ρ among top 500 genes
    # across donors in that age group
    N_GENES_L1 = 500
    gene_list = list(gene_matrix.keys())[:N_GENES_L1]

    level1_results = []
    for tissue in top6:
        t_short = tissue.split(" - ")[-1][:20]
        t_samples = tissue_samples[tissue]

        for age_label, age_lo, age_hi in [
            ("20-35", 20, 36), ("36-49", 36, 50), ("50-59", 50, 60), ("60-79", 60, 80)
        ]:
            # Get samples in this tissue × age
            samples_in = [(i, sid) for i, sid in t_samples
                          if sid in sample_meta.index and
                          age_lo <= sample_meta.loc[sid, "age_mid"] < age_hi]

            if len(samples_in) < 30:
                continue

            idx_arr = [i for i, _ in samples_in]

            # Build donor × gene matrix
            mat = np.zeros((len(idx_arr), len(gene_list)))
            for g_i, gene in enumerate(gene_list):
                for s_i, idx in enumerate(idx_arr):
                    mat[s_i, g_i] = gene_matrix[gene][idx]

            # Compute gene-gene correlation matrix
            # Use numpy for speed: standardize columns, then mat.T @ mat / N
            means = mat.mean(axis=0)
            stds = mat.std(axis=0)
            stds[stds < 1e-10] = 1
            mat_z = (mat - means) / stds
            corr = mat_z.T @ mat_z / len(idx_arr)

            # Extract upper triangle (exclude diagonal)
            triu_idx = np.triu_indices(len(gene_list), k=1)
            rhos = corr[triu_idx]

            mean_abs_rho = np.mean(np.abs(rhos))
            median_abs_rho = np.median(np.abs(rhos))
            pct_strong = np.mean(np.abs(rhos) > 0.3)

            level1_results.append({
                "tissue": t_short, "age": age_label,
                "n_samples": len(idx_arr), "n_genes": len(gene_list),
                "mean_abs_rho": mean_abs_rho,
                "median_abs_rho": median_abs_rho,
                "pct_strong": pct_strong,
            })

            _log(f"  {t_short:<20s} {age_label:>6s} (n={len(idx_arr):3d}): "
                 f"mean|ρ|={mean_abs_rho:.4f}, median|ρ|={median_abs_rho:.4f}, "
                 f"%strong={pct_strong:.1%}")

    df_l1 = pd.DataFrame(level1_results)
    df_l1.to_csv(RESULTS_DIR / "level1_gene_coordination.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════
    # PART C: Level 4 — Donor-donor pairwise similarity
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[C] Level 4 — Donor-donor similarity by age")
    _log("=" * 70)

    level4_results = []
    for tissue in top6:
        t_short = tissue.split(" - ")[-1][:20]
        t_samples = tissue_samples[tissue]

        for age_label, age_lo, age_hi in [
            ("20-35", 20, 36), ("36-49", 36, 50), ("50-59", 50, 60), ("60-79", 60, 80)
        ]:
            samples_in = [(i, sid) for i, sid in t_samples
                          if sid in sample_meta.index and
                          age_lo <= sample_meta.loc[sid, "age_mid"] < age_hi]

            if len(samples_in) < 20:
                continue

            idx_arr = [i for i, _ in samples_in]

            # Build donor × gene matrix (top 500 genes)
            mat = np.zeros((len(idx_arr), len(gene_list)))
            for g_i, gene in enumerate(gene_list):
                for s_i, idx in enumerate(idx_arr):
                    mat[s_i, g_i] = gene_matrix[gene][idx]

            # Donor-donor correlation: standardize rows (donors), then mat @ mat.T
            means = mat.mean(axis=1, keepdims=True)
            stds = mat.std(axis=1, keepdims=True)
            stds[stds < 1e-10] = 1
            mat_z = (mat - means) / stds
            donor_corr = mat_z @ mat_z.T / mat.shape[1]

            triu_idx = np.triu_indices(len(idx_arr), k=1)
            dd_rhos = donor_corr[triu_idx]

            mean_dd_rho = np.mean(dd_rhos)
            median_dd_rho = np.median(dd_rhos)

            level4_results.append({
                "tissue": t_short, "age": age_label,
                "n_donors": len(idx_arr),
                "mean_donor_rho": mean_dd_rho,
                "median_donor_rho": median_dd_rho,
            })

            _log(f"  {t_short:<20s} {age_label:>6s} (n={len(idx_arr):3d}): "
                 f"mean donor ρ={mean_dd_rho:.4f}, median={median_dd_rho:.4f}")

    df_l4 = pd.DataFrame(level4_results)
    df_l4.to_csv(RESULTS_DIR / "level4_donor_similarity.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════
    # PART D: CH drivers in blood
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[D] Clonal hematopoiesis drivers in blood")
    _log("=" * 70)

    blood_tissue = [t for t in top6 if "Blood" in t]
    if blood_tissue:
        bt = blood_tissue[0]
        b_samples = tissue_samples[bt]

        for gene in CH_DRIVERS:
            if gene not in sig_data:
                continue
            vals = sig_data[gene]
            ages = []
            exprs = []
            for i, sid in b_samples:
                if sid in sample_meta.index:
                    a = sample_meta.loc[sid, "age_mid"]
                    if np.isfinite(a):
                        ages.append(a)
                        exprs.append(vals[i])

            ages = np.array(ages)
            exprs = np.array(exprs)

            # Mean expression vs age
            rho_mean, p_mean = spearman_safe(ages, exprs)

            # Variance vs age (by decade)
            var_by_decade = []
            for lo, hi in [(20, 35), (35, 50), (50, 65), (65, 80)]:
                mask = (ages >= lo) & (ages < hi)
                if mask.sum() >= 20:
                    var_by_decade.append((lo, hi, np.var(exprs[mask]), np.mean(exprs[mask])))

            _log(f"  {gene:<10s}: ρ(age,expr)={rho_mean:+.3f} p={p_mean:.2e}")
            for lo, hi, v, m in var_by_decade:
                _log(f"    {lo}-{hi}: mean={m:.3f}, var={v:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[Figures]")
    _log("=" * 70)

    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Level 1 — gene-gene coordination by age
    ax = fig.add_subplot(gs[0, 0])
    for tissue in top6:
        t_short = tissue.split(" - ")[-1][:15]
        sub = df_l1[df_l1["tissue"] == t_short]
        if len(sub) >= 3:
            color = "tab:blue" if "Blood" in tissue else "tab:red"
            linestyle = "--" if "Blood" in tissue else "-"
            ax.plot(range(len(sub)), sub["mean_abs_rho"].values, f"{linestyle}o",
                    color=color, markersize=4, label=t_short, alpha=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["20-35", "36-49", "50-59", "60-79"])
    ax.set_xlabel("Age group")
    ax.set_ylabel("Mean |ρ| (gene-gene)")
    ax.set_title("A: Level 1 — Gene-gene coordination\nBlue=blood, Red=solid")
    ax.legend(fontsize=6, ncol=2)

    # Panel B: Level 4 — donor-donor similarity
    ax = fig.add_subplot(gs[0, 1])
    for tissue in top6:
        t_short = tissue.split(" - ")[-1][:15]
        sub = df_l4[df_l4["tissue"] == t_short]
        if len(sub) >= 3:
            color = "tab:blue" if "Blood" in tissue else "tab:red"
            linestyle = "--" if "Blood" in tissue else "-"
            ax.plot(range(len(sub)), sub["mean_donor_rho"].values, f"{linestyle}o",
                    color=color, markersize=4, label=t_short, alpha=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["20-35", "36-49", "50-59", "60-79"])
    ax.set_xlabel("Age group")
    ax.set_ylabel("Mean donor-donor ρ")
    ax.set_title("B: Level 4 — Donor similarity\nBlue=blood, Red=solid")
    ax.legend(fontsize=6, ncol=2)

    # Panel C: Confound — raw vs residualized
    ax = fig.add_subplot(gs[0, 2])
    if len(df_conf) > 0:
        ax.scatter(df_conf["median_raw"], df_conf["median_resid"], s=40, alpha=0.7,
                   c=["tab:blue" if "Blood" in r["t1"] or "Blood" in r["t2"] else "tab:red"
                      for _, r in df_conf.iterrows()],
                   edgecolors="gray", linewidth=0.3)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
        ax.set_xlabel("Raw cross-tissue ρ")
        ax.set_ylabel("Immune-residualized ρ")
        ax.set_title("C: Confound control\nBlue=blood, Red=solid")

    # Panel D: Level 1 summary — solid vs blood
    ax = fig.add_subplot(gs[1, 0])
    for tissue_type, color in [("solid", "tab:red"), ("blood", "tab:blue")]:
        if tissue_type == "blood":
            sub = df_l1[df_l1["tissue"].str.contains("Blood")]
        else:
            sub = df_l1[~df_l1["tissue"].str.contains("Blood")]

        if len(sub) > 0:
            means = sub.groupby("age")["mean_abs_rho"].mean()
            ax.plot(range(len(means)), means.values, "o-", color=color, label=tissue_type, linewidth=2)

    ax.set_xticks(range(4))
    ax.set_xticklabels(["20-35", "36-49", "50-59", "60-79"])
    ax.set_xlabel("Age group")
    ax.set_ylabel("Mean |ρ| averaged across tissues")
    ax.set_title("D: Level 1 summary — solid vs blood")
    ax.legend()

    # Panel E: Level 4 summary
    ax = fig.add_subplot(gs[1, 1])
    for tissue_type, color in [("solid", "tab:red"), ("blood", "tab:blue")]:
        if tissue_type == "blood":
            sub = df_l4[df_l4["tissue"].str.contains("Blood")]
        else:
            sub = df_l4[~df_l4["tissue"].str.contains("Blood")]

        if len(sub) > 0:
            means = sub.groupby("age")["mean_donor_rho"].mean()
            ax.plot(range(len(means)), means.values, "o-", color=color, label=tissue_type, linewidth=2)

    ax.set_xticks(range(4))
    ax.set_xticklabels(["20-35", "36-49", "50-59", "60-79"])
    ax.set_xlabel("Age group")
    ax.set_ylabel("Mean donor-donor ρ")
    ax.set_title("E: Level 4 summary — solid vs blood")
    ax.legend()

    # Panel F: Cross-level consistency
    ax = fig.add_subplot(gs[1, 2])
    # For each tissue: compute Δ(Level 1) and Δ(Level 4)
    cross_level = []
    for tissue in top6:
        t_short = tissue.split(" - ")[-1][:15]
        l1 = df_l1[df_l1["tissue"] == t_short]
        l4 = df_l4[df_l4["tissue"] == t_short]
        if len(l1) >= 3 and len(l4) >= 3:
            dl1 = l1["mean_abs_rho"].iloc[-1] - l1["mean_abs_rho"].iloc[0]
            dl4 = l4["mean_donor_rho"].iloc[-1] - l4["mean_donor_rho"].iloc[0]
            is_blood = "Blood" in tissue
            cross_level.append({"tissue": t_short, "dl1": dl1, "dl4": dl4, "blood": is_blood})

    df_cl = pd.DataFrame(cross_level)
    if len(df_cl) > 0:
        ax.scatter(df_cl["dl1"], df_cl["dl4"],
                   c=["tab:blue" if b else "tab:red" for b in df_cl["blood"]],
                   s=80, edgecolors="gray", linewidth=0.5)
        for _, r in df_cl.iterrows():
            ax.annotate(r["tissue"][:10], (r["dl1"], r["dl4"]), fontsize=7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Δ Level 1 (gene-gene |ρ|)")
        ax.set_ylabel("Δ Level 4 (donor-donor ρ)")
        ax.set_title("F: Cross-level consistency\nBlue=blood, Red=solid")

    fig.suptitle("Two-Compartment Validation — Confounds + Multi-Level\n"
                 "GTEx 948 donors, 6 tissues, 500 genes",
                 fontsize=14, fontweight="bold")
    fig.savefig(RESULTS_DIR / "two_compartment_validation.png", dpi=150)
    plt.close(fig)
    _log(f"\n  Saved two_compartment_validation.png")

    _log(f"\n  Total time: {time.time()-t0:.0f}s")
    _log(f"  Results: {RESULTS_DIR}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
