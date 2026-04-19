"""
Step 8: Tissue Identity Loss — the key mechanistic test

Test 1: Tissue-specific vs shared-aging gene pair decomposition
  - Define tissue-specific genes (high expression in one tissue, low elsewhere)
  - Define shared-aging genes (consistently upregulated with age across tissues)
  - Compute gene-gene ρ within each set, young vs old
  - Prediction: tissue-specific ρ↓, shared-aging ρ↑ or stable

Test 2: Aging signature overlap
  - Top 200 DEGs (old vs young) per tissue
  - Overlap matrix: how many DEGs shared between tissue pairs?
  - If high overlap → shared aging program

Test 3: Tissue identity score
  - For each donor: how well does their expression match tissue-specific program?
  - Score = mean z-score of tissue-specific genes
  - Does identity score decline with age? Blood vs solid?
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
RESULTS_DIR = BASE / "results" / "step8_tissue_identity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg):
    print(msg, flush=True)


def main():
    t0 = time.time()
    _log("=" * 70)
    _log("TISSUE IDENTITY LOSS — Mechanistic decomposition")
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

    tissue_counts = samples["SMTSD"].value_counts()
    top6 = tissue_counts.head(6).index.tolist()

    tpm_path = DATA_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline()
        header = f.readline().strip().split("\t")
    sample_ids = header[2:]

    tissue_sample_idx = {}
    for i, sid in enumerate(sample_ids):
        if sid in sample_meta.index:
            t = sample_meta.loc[sid, "SMTSD"]
            if t in top6:
                if t not in tissue_sample_idx:
                    tissue_sample_idx[t] = []
                tissue_sample_idx[t].append(i)
    for t in tissue_sample_idx:
        tissue_sample_idx[t] = np.array(tissue_sample_idx[t])

    # Age masks per tissue
    tissue_young_idx = {}
    tissue_old_idx = {}
    for t in top6:
        young, old = [], []
        for i in tissue_sample_idx[t]:
            sid = sample_ids[i]
            if sid in sample_meta.index:
                age = sample_meta.loc[sid, "age_mid"]
                if age <= 35:
                    young.append(i)
                elif age >= 55:
                    old.append(i)
        tissue_young_idx[t] = np.array(young)
        tissue_old_idx[t] = np.array(old)
        _log(f"  {t.split(' - ')[-1][:15]:>15s}: young={len(young)}, old={len(old)}")

    # ── Pass 1: Identify tissue-specific and shared-aging genes ──────
    _log(f"\n[1] Identifying tissue-specific and shared-aging genes...")

    gene_tissue_means = {}  # gene → {tissue: mean_expr}
    gene_age_effects = {}   # gene → {tissue: rho_with_age}
    gene_log_vals = {}      # gene → full log2(TPM+1) array

    n_processed = 0
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            parts = line.strip().split("\t", 2)
            gene = parts[1]
            vals = np.array(parts[2].split("\t"), dtype=np.float32)

            if np.median(vals) < 1.0:  # Higher threshold for robust genes
                n_processed += 1
                continue

            log_vals = np.log2(vals + 1)
            n_processed += 1

            if n_processed % 5000 == 0:
                _log(f"    {n_processed} genes ({time.time()-t0:.0f}s)...")

            # Mean per tissue
            t_means = {}
            for t in top6:
                idx = tissue_sample_idx[t]
                t_means[t] = np.mean(log_vals[idx])
            gene_tissue_means[gene] = t_means

            # Age effect per tissue (Pearson r with age)
            t_age = {}
            for t in top6:
                y_idx = tissue_young_idx[t]
                o_idx = tissue_old_idx[t]
                if len(y_idx) >= 30 and len(o_idx) >= 30:
                    y_mean = np.mean(log_vals[y_idx])
                    o_mean = np.mean(log_vals[o_idx])
                    t_age[t] = o_mean - y_mean  # positive = upregulated in old
            gene_age_effects[gene] = t_age

            # Store for later use (keep top genes only to save memory)
            if len(gene_log_vals) < 5000:
                gene_log_vals[gene] = log_vals

    _log(f"  Total expressed genes (median TPM ≥ 1): {len(gene_tissue_means)}")

    # ── Define tissue-specific genes ──────────────────────────────────
    _log(f"\n[2] Defining tissue-specific genes...")

    tissue_specific = {}  # tissue → list of genes
    N_SPECIFIC = 150

    for target_tissue in top6:
        scores = []
        for gene, t_means in gene_tissue_means.items():
            if target_tissue not in t_means:
                continue
            target_expr = t_means[target_tissue]
            other_exprs = [t_means[t] for t in top6 if t != target_tissue and t in t_means]
            if not other_exprs:
                continue
            # Tissue specificity = target - max(others)
            specificity = target_expr - max(other_exprs)
            scores.append((gene, specificity, target_expr))

        scores.sort(key=lambda x: -x[1])
        tissue_specific[target_tissue] = [g for g, s, e in scores[:N_SPECIFIC] if s > 0.5]

        t_short = target_tissue.split(" - ")[-1][:15]
        _log(f"  {t_short}: {len(tissue_specific[target_tissue])} tissue-specific genes")
        if tissue_specific[target_tissue]:
            _log(f"    Top 5: {tissue_specific[target_tissue][:5]}")

    # ── Define shared-aging genes ─────────────────────────────────────
    _log(f"\n[3] Defining shared-aging genes...")

    # Genes upregulated in old across ≥4/6 tissues
    shared_up = []
    shared_down = []
    for gene, t_age in gene_age_effects.items():
        if len(t_age) < 4:
            continue
        n_up = sum(1 for v in t_age.values() if v > 0.1)
        n_down = sum(1 for v in t_age.values() if v < -0.1)
        mean_change = np.mean(list(t_age.values()))

        if n_up >= 4:
            shared_up.append((gene, mean_change, n_up))
        if n_down >= 4:
            shared_down.append((gene, mean_change, n_down))

    shared_up.sort(key=lambda x: -x[1])
    shared_down.sort(key=lambda x: x[1])

    N_SHARED = 150
    shared_aging_up = [g for g, _, _ in shared_up[:N_SHARED]]
    shared_aging_down = [g for g, _, _ in shared_down[:N_SHARED]]
    shared_aging_all = shared_aging_up + shared_aging_down

    _log(f"  Shared aging UP (≥4/6 tissues, Δ>0.1): {len(shared_up)} total, using top {len(shared_aging_up)}")
    _log(f"    Top 10: {shared_aging_up[:10]}")
    _log(f"  Shared aging DOWN: {len(shared_down)} total, using top {len(shared_aging_down)}")
    _log(f"    Top 10: {shared_aging_down[:10]}")

    # ── Test 1: Gene-gene ρ within each set ───────────────────────────
    _log(f"\n" + "=" * 70)
    _log("[Test 1] Gene-gene coordination: tissue-specific vs shared-aging")
    _log("=" * 70)

    test1_results = []

    for tissue in top6:
        t_short = tissue.split(" - ")[-1][:15]
        y_idx = tissue_young_idx[tissue]
        o_idx = tissue_old_idx[tissue]

        if len(y_idx) < 30 or len(o_idx) < 30:
            continue

        for gene_set_name, gene_list in [
            ("tissue_specific", tissue_specific.get(tissue, [])),
            ("shared_aging", shared_aging_all),
        ]:
            # Filter to genes in our stored data
            available = [g for g in gene_list if g in gene_log_vals]
            if len(available) < 20:
                continue

            available = available[:100]  # Cap for speed

            # Build expression matrices
            mat_y = np.zeros((len(y_idx), len(available)))
            mat_o = np.zeros((len(o_idx), len(available)))
            for gi, gene in enumerate(available):
                mat_y[:, gi] = gene_log_vals[gene][y_idx]
                mat_o[:, gi] = gene_log_vals[gene][o_idx]

            # Compute gene-gene correlation matrices
            def mean_abs_rho(mat):
                n = mat.shape[1]
                if n < 5:
                    return np.nan
                means = mat.mean(axis=0)
                stds = mat.std(axis=0)
                stds[stds < 1e-10] = 1
                z = (mat - means) / stds
                corr = z.T @ z / mat.shape[0]
                triu = np.triu_indices(n, k=1)
                return np.mean(np.abs(corr[triu]))

            rho_y = mean_abs_rho(mat_y)
            rho_o = mean_abs_rho(mat_o)

            test1_results.append({
                "tissue": t_short, "gene_set": gene_set_name,
                "n_genes": len(available),
                "mean_abs_rho_young": rho_y,
                "mean_abs_rho_old": rho_o,
                "delta": rho_o - rho_y,
            })

            _log(f"  {t_short:<15s} {gene_set_name:<18s} (n={len(available):3d}): "
                 f"young={rho_y:.4f}, old={rho_o:.4f}, Δ={rho_o-rho_y:+.4f}")

    df_t1 = pd.DataFrame(test1_results)
    df_t1.to_csv(RESULTS_DIR / "test1_specifc_vs_shared.csv", index=False)

    # Summary
    _log(f"\n  SUMMARY:")
    for gs in ["tissue_specific", "shared_aging"]:
        sub = df_t1[df_t1["gene_set"] == gs]
        if len(sub) > 0:
            _log(f"    {gs}: median Δ = {sub['delta'].median():+.4f}, "
                 f"all decline: {(sub['delta'] < 0).all()}")

    # ── Test 2: Aging signature overlap ───────────────────────────────
    _log(f"\n" + "=" * 70)
    _log("[Test 2] Aging signature overlap between tissues")
    _log("=" * 70)

    # Top 200 age-upregulated genes per tissue
    tissue_aging_sig = {}
    for tissue in top6:
        t_short = tissue.split(" - ")[-1][:15]
        scores = []
        for gene, t_age in gene_age_effects.items():
            if tissue in t_age:
                scores.append((gene, t_age[tissue]))
        scores.sort(key=lambda x: -x[1])
        tissue_aging_sig[tissue] = set(g for g, _ in scores[:200])
        _log(f"  {t_short}: top 200 aging genes, top 5: {[g for g,_ in scores[:5]]}")

    # Overlap matrix
    _log(f"\n  Pairwise overlap (Jaccard index):")
    overlap_results = []
    for i, t1 in enumerate(top6):
        for t2 in top6[i+1:]:
            s1 = tissue_aging_sig[t1]
            s2 = tissue_aging_sig[t2]
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            jaccard = intersection / union if union > 0 else 0
            t1s = t1.split(" - ")[-1][:12]
            t2s = t2.split(" - ")[-1][:12]
            is_blood = "Blood" in t1 or "Blood" in t2
            overlap_results.append({
                "t1": t1s, "t2": t2s,
                "intersection": intersection, "union": union,
                "jaccard": jaccard, "blood_involved": is_blood,
            })
            _log(f"    {t1s:>12s} × {t2s:<12s}: {intersection:3d}/200 shared "
                 f"(J={jaccard:.3f}) {'[BLOOD]' if is_blood else ''}")

    df_overlap = pd.DataFrame(overlap_results)
    df_overlap.to_csv(RESULTS_DIR / "test2_aging_overlap.csv", index=False)

    # Summary
    blood_j = df_overlap[df_overlap["blood_involved"]]["jaccard"]
    solid_j = df_overlap[~df_overlap["blood_involved"]]["jaccard"]
    _log(f"\n  Blood-involved pairs: median Jaccard = {blood_j.median():.3f}")
    _log(f"  Solid-solid pairs: median Jaccard = {solid_j.median():.3f}")

    # ── Test 3: Tissue identity score vs age ──────────────────────────
    _log(f"\n" + "=" * 70)
    _log("[Test 3] Tissue identity score vs age")
    _log("=" * 70)

    # For each tissue: z-score of tissue-specific genes = identity score
    identity_results = []
    for tissue in top6:
        t_short = tissue.split(" - ")[-1][:15]
        t_genes = tissue_specific.get(tissue, [])
        available = [g for g in t_genes if g in gene_log_vals]
        if len(available) < 10:
            _log(f"  {t_short}: too few tissue-specific genes ({len(available)})")
            continue

        idx = tissue_sample_idx[tissue]
        # Compute mean of tissue-specific genes per sample
        identity_scores = np.zeros(len(idx))
        for g in available[:50]:
            identity_scores += gene_log_vals[g][idx]
        identity_scores /= len(available[:50])

        # Get ages
        ages = np.array([sample_meta.loc[sample_ids[i], "age_mid"]
                         if sample_ids[i] in sample_meta.index else np.nan
                         for i in idx])

        valid = np.isfinite(ages)
        rho, p = stats.spearmanr(ages[valid], identity_scores[valid])

        identity_results.append({
            "tissue": t_short, "n_genes": len(available[:50]),
            "rho_age": rho, "p_age": p,
            "mean_score_young": np.mean(identity_scores[ages <= 35]),
            "mean_score_old": np.mean(identity_scores[ages >= 55]),
        })

        _log(f"  {t_short:<15s}: ρ(age, identity)={rho:+.3f}, p={p:.2e}, "
             f"young={np.mean(identity_scores[ages<=35]):.3f}, "
             f"old={np.mean(identity_scores[ages>=55]):.3f}")

    df_id = pd.DataFrame(identity_results)
    df_id.to_csv(RESULTS_DIR / "test3_identity_score.csv", index=False)

    # ── FIGURES ────────────────────────────────────────────────────────
    _log(f"\n" + "=" * 70)
    _log("[Figures]")
    _log("=" * 70)

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Test 1 — tissue-specific vs shared-aging Δρ
    ax = fig.add_subplot(gs[0, 0])
    for gs_name, color, label in [("tissue_specific", "tab:blue", "Tissue-specific"),
                                    ("shared_aging", "tab:red", "Shared aging")]:
        sub = df_t1[df_t1["gene_set"] == gs_name]
        if len(sub) > 0:
            x = np.arange(len(sub))
            offset = -0.15 if gs_name == "tissue_specific" else 0.15
            ax.bar(x + offset, sub["delta"].values, width=0.3, color=color, alpha=0.7, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels(sub["tissue"].values, rotation=45, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Δ mean|ρ| (old − young)")
    ax.set_title("A: Gene-gene coordination change\nTissue-specific (blue) vs Shared aging (red)")
    ax.legend(fontsize=8)

    # Panel B: Test 1 — young vs old for each set
    ax = fig.add_subplot(gs[0, 1])
    for gs_name, marker, color in [("tissue_specific", "o", "tab:blue"),
                                     ("shared_aging", "s", "tab:red")]:
        sub = df_t1[df_t1["gene_set"] == gs_name]
        if len(sub) > 0:
            ax.scatter(sub["mean_abs_rho_young"], sub["mean_abs_rho_old"],
                       c=color, marker=marker, s=60, alpha=0.7, label=gs_name, edgecolors="gray")
            for _, r in sub.iterrows():
                ax.annotate(r["tissue"][:6], (r["mean_abs_rho_young"], r["mean_abs_rho_old"]),
                            fontsize=6, alpha=0.7)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
    ax.set_xlabel("Young mean|ρ|")
    ax.set_ylabel("Old mean|ρ|")
    ax.set_title("B: Young vs Old coordination\n(below diagonal = decline)")
    ax.legend(fontsize=8)

    # Panel C: Test 2 — overlap heatmap
    ax = fig.add_subplot(gs[0, 2])
    tissue_names = [t.split(" - ")[-1][:12] for t in top6]
    n_t = len(top6)
    overlap_mat = np.zeros((n_t, n_t))
    for _, r in df_overlap.iterrows():
        i = [i for i, t in enumerate(tissue_names) if r["t1"] in t or t in r["t1"]]
        j = [j for j, t in enumerate(tissue_names) if r["t2"] in t or t in r["t2"]]
        if i and j:
            overlap_mat[i[0], j[0]] = r["jaccard"]
            overlap_mat[j[0], i[0]] = r["jaccard"]
    np.fill_diagonal(overlap_mat, 1.0)
    im = ax.imshow(overlap_mat, cmap="YlOrRd", vmin=0, vmax=0.3)
    ax.set_xticks(range(n_t))
    ax.set_xticklabels(tissue_names, rotation=45, fontsize=7)
    ax.set_yticks(range(n_t))
    ax.set_yticklabels(tissue_names, fontsize=7)
    for i in range(n_t):
        for j in range(n_t):
            if i != j:
                ax.text(j, i, f"{overlap_mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="Jaccard index", shrink=0.7)
    ax.set_title("C: Aging signature overlap\n(shared aging genes between tissues)")

    # Panel D: Test 3 — identity score vs age
    ax = fig.add_subplot(gs[1, 0])
    if len(df_id) > 0:
        colors = ["tab:blue" if "Blood" in r["tissue"] else "tab:red" for _, r in df_id.iterrows()]
        ax.bar(range(len(df_id)), df_id["rho_age"].values, color=colors, alpha=0.7)
        ax.set_xticks(range(len(df_id)))
        ax.set_xticklabels(df_id["tissue"].values, rotation=45, fontsize=7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("ρ(age, identity score)")
        ax.set_title("D: Tissue identity vs age\n(negative = losing identity)")

    # Panel E: Schematic summary
    ax = fig.add_subplot(gs[1, 1])
    ax.text(0.5, 0.9, "YOUNG ORGANISM", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.2, 0.7, "Muscle\n(muscle genes)", ha="center", fontsize=9, color="tab:green",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))
    ax.text(0.5, 0.7, "Skin\n(skin genes)", ha="center", fontsize=9, color="tab:orange",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))
    ax.text(0.8, 0.7, "Blood\n(blood genes)", ha="center", fontsize=9, color="tab:blue",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    ax.text(0.5, 0.4, "OLD ORGANISM", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.2, 0.2, "Muscle\n(inflam+fibro)", ha="center", fontsize=9, color="tab:red",
            bbox=dict(boxstyle="round", facecolor="lightsalmon", alpha=0.5))
    ax.text(0.5, 0.2, "Skin\n(inflam+fibro)", ha="center", fontsize=9, color="tab:red",
            bbox=dict(boxstyle="round", facecolor="lightsalmon", alpha=0.5))
    ax.text(0.8, 0.2, "Blood\n(clonal X/Y/Z)", ha="center", fontsize=9, color="tab:purple",
            bbox=dict(boxstyle="round", facecolor="plum", alpha=0.5))

    ax.annotate("", xy=(0.35, 0.15), xytext=(0.35, 0.25),
                arrowprops=dict(arrowstyle="<->", color="red", lw=2))
    ax.text(0.35, 0.12, "CONVERGE", ha="center", fontsize=8, color="red")
    ax.annotate("", xy=(0.65, 0.15), xytext=(0.65, 0.25),
                arrowprops=dict(arrowstyle="<->", color="purple", lw=2))
    ax.text(0.65, 0.12, "DIVERGE", ha="center", fontsize=8, color="purple")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("E: Model — tissue identity loss")

    # Panel F: Blood vs Solid overlap
    ax = fig.add_subplot(gs[1, 2])
    if len(df_overlap) > 0:
        blood_data = df_overlap[df_overlap["blood_involved"]]["jaccard"]
        solid_data = df_overlap[~df_overlap["blood_involved"]]["jaccard"]
        data = [solid_data.values, blood_data.values]
        bp = ax.boxplot(data, labels=["Solid-Solid", "Blood-involved"], patch_artist=True)
        bp["boxes"][0].set_facecolor("tab:red")
        bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor("tab:blue")
        bp["boxes"][1].set_alpha(0.5)
        ax.set_ylabel("Jaccard index (aging signature overlap)")
        ax.set_title("F: Shared aging program\nSolid tissues share more aging genes than blood")

    fig.suptitle("Tissue Identity Loss with Aging\n"
                 "GTEx 948 donors — tissue-specific programs erode, shared aging program emerges",
                 fontsize=14, fontweight="bold")
    fig.savefig(RESULTS_DIR / "tissue_identity_loss.png", dpi=150)
    plt.close(fig)
    _log(f"\n  Saved tissue_identity_loss.png")

    _log(f"\n  Total time: {time.time()-t0:.0f}s")
    _log(f"  Results: {RESULTS_DIR}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
