"""
Step 2: GTEx Atlas Mining — Five analyses on existing coupling data

1. Tissue clustering by coupling trajectory
2. Pair ranking across tissues (universality of decline)
3. Cross-tissue coupling within donors
4. Sex × tissue interaction
5. Breakpoint detection per tissue

All from gtex_coupling_all.csv — no new data loading needed.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
RESULTS_DIR = BASE / "results" / "step2_mining"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STEP1_DIR = BASE / "results" / "step1_gtex"


def _log(msg):
    print(msg, flush=True)


def spearman_safe(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 6 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan, 1.0
    return stats.spearmanr(x, y)


def main():
    t0 = time.time()
    _log("=" * 70)
    _log("GTEx ATLAS MINING — Step 2")
    _log("=" * 70)

    df = pd.read_csv(STEP1_DIR / "gtex_coupling_all.csv")
    _log(f"  Loaded {len(df)} coupling measurements")
    _log(f"  Tissues: {df['tissue'].nunique()}, Pairs: {df['label'].nunique()}")

    # Age decades available
    age_cols = [c for c in df.columns if c.startswith("rho_") and c[4:5].isdigit()]
    age_decades = sorted(set(c.replace("rho_", "") for c in age_cols))
    _log(f"  Age decades: {age_decades}")

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 1: Tissue clustering by coupling trajectory
    # ══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[1] Tissue clustering by coupling trajectory")
    _log("=" * 70)

    # For each tissue × pair: compute slope (rho vs age decade midpoint)
    age_mids = {"20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65, "70-79": 75}

    slopes = []
    for _, row in df.iterrows():
        age_vals = []
        rho_vals = []
        for ad in age_decades:
            r = row.get(f"rho_{ad}", np.nan)
            if not np.isnan(r):
                age_vals.append(age_mids[ad])
                rho_vals.append(r)
        if len(age_vals) >= 4:
            slope, _, _, _, _ = stats.linregress(age_vals, rho_vals)
        else:
            slope = np.nan
        slopes.append(slope)

    df["slope"] = slopes

    # Pivot: tissue × pair → slope
    pivot = df.pivot_table(index="tissue", columns="label", values="slope", aggfunc="first")
    # Drop pairs/tissues with too many NaN
    pivot = pivot.dropna(thresh=pivot.shape[1] * 0.5, axis=0)
    pivot = pivot.dropna(thresh=pivot.shape[0] * 0.5, axis=1)
    pivot = pivot.fillna(0)

    _log(f"  Pivot: {pivot.shape[0]} tissues × {pivot.shape[1]} pairs")

    # Cluster tissues
    if pivot.shape[0] >= 4:
        dist = pdist(pivot.values, metric="euclidean")
        Z = linkage(dist, method="ward")
        clusters = fcluster(Z, t=3, criterion="maxclust")

        fig, axes = plt.subplots(1, 2, figsize=(24, max(8, pivot.shape[0] * 0.35)),
                                  gridspec_kw={"width_ratios": [1, 3]})

        # Dendrogram
        ax = axes[0]
        dn = dendrogram(Z, labels=pivot.index.tolist(), orientation="left", ax=ax,
                        leaf_font_size=8, color_threshold=Z[-2, 2])
        ax.set_title("Tissue clustering\n(by coupling slope vectors)")

        # Heatmap
        ax = axes[1]
        order = dn["leaves"]
        ordered = pivot.iloc[order]

        # Sort columns by mean slope
        col_order = ordered.mean().sort_values().index
        ordered = ordered[col_order]

        im = ax.imshow(ordered.values, aspect="auto", cmap="RdBu_r", vmin=-0.005, vmax=0.005)
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered.index, fontsize=7)
        ax.set_xticks(range(len(ordered.columns)))
        ax.set_xticklabels(ordered.columns, rotation=90, fontsize=6)
        plt.colorbar(im, ax=ax, label="Slope (Δρ per year)", shrink=0.5)
        ax.set_title("Coupling slope: red = increases with age, blue = decreases")

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "tissue_clustering.png", dpi=150)
        plt.close(fig)
        _log(f"  Saved tissue_clustering.png")

        # Print clusters
        for c in sorted(set(clusters)):
            tissues_in = pivot.index[clusters == c].tolist()
            _log(f"\n  Cluster {c}: {tissues_in}")
            cluster_data = pivot.loc[tissues_in]
            mean_slopes = cluster_data.mean()
            top_up = mean_slopes.nlargest(3)
            top_down = mean_slopes.nsmallest(3)
            _log(f"    Most increasing: {[(l, f'{v:+.4f}') for l, v in top_up.items()]}")
            _log(f"    Most decreasing: {[(l, f'{v:+.4f}') for l, v in top_down.items()]}")

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 2: Pair ranking across tissues
    # ══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[2] Pair ranking — universality of decline")
    _log("=" * 70)

    pair_stats = []
    for label in df["label"].unique():
        sub = df[df["label"] == label]
        valid_slopes = sub["slope"].dropna()
        if len(valid_slopes) < 5:
            continue

        n_decline = (valid_slopes < 0).sum()
        n_increase = (valid_slopes > 0).sum()
        n_total = len(valid_slopes)
        median_slope = valid_slopes.median()
        mean_slope = valid_slopes.mean()

        # Sign test
        if n_total > 0:
            sign_p = stats.binomtest(n_decline, n_total, 0.5).pvalue
        else:
            sign_p = 1.0

        cat = sub["category"].iloc[0]
        pair_stats.append({
            "label": label, "category": cat,
            "n_tissues": n_total, "n_decline": n_decline, "n_increase": n_increase,
            "pct_decline": n_decline / n_total,
            "median_slope": median_slope, "mean_slope": mean_slope,
            "sign_p": sign_p,
        })

    df_pairs = pd.DataFrame(pair_stats).sort_values("median_slope")
    df_pairs.to_csv(RESULTS_DIR / "pair_ranking.csv", index=False)

    _log(f"\n  {'Label':<25s} {'Cat':<12s} {'%Decline':>8s} {'Med slope':>10s} {'Sign p':>8s}")
    _log(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*10} {'-'*8}")
    for _, r in df_pairs.iterrows():
        sig = "*" if r["sign_p"] < 0.05 else " "
        _log(f"  {r['label']:<25s} {r['category']:<12s} "
             f"{r['pct_decline']:>7.0%} {r['median_slope']:>+10.5f} {r['sign_p']:>7.3f} {sig}")

    # Figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_pairs) * 0.3)))
    colors_map = {"production": "tab:green", "detection": "tab:orange",
                  "housekeeping": "tab:gray", "hormone": "tab:purple"}
    colors = [colors_map.get(r["category"], "gray") for _, r in df_pairs.iterrows()]
    sig_markers = ["*" if r["sign_p"] < 0.05 else "" for _, r in df_pairs.iterrows()]

    ax.barh(range(len(df_pairs)), df_pairs["median_slope"].values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_pairs)))
    ax.set_yticklabels([f"{r['label']} {s}" for (_, r), s in zip(df_pairs.iterrows(), sig_markers)],
                        fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Median slope (Δρ per year across tissues)")
    ax.set_title("Pair universality: which TF→target couplings decline with age?\n"
                 "Green=production, Orange=detection, Gray=HK, Purple=hormone. *=sign test p<0.05")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pair_ranking.png", dpi=150)
    plt.close(fig)
    _log(f"  Saved pair_ranking.png")

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 3: Cross-tissue coupling within donors
    # ══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[3] Cross-tissue within-donor coupling")
    _log("=" * 70)

    # Need to load raw TPM for this — use the merged data approach
    # Load GTEx TPM for key pairs, compute per-donor per-tissue values
    import gzip

    DATA_DIR = BASE / "data" / "gtex"
    samples = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    samples["sex"] = samples["SEX"].map({1: "male", 2: "female"})

    # Load key genes
    key_genes = {"SMAD3", "COL1A1", "RELA", "ICAM1", "ACTB", "GAPDH", "ESR1", "FN1"}
    _log("  Loading TPM for key genes...")

    tpm_path = DATA_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    data = {}
    remaining = set(key_genes)
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

    tpm_df = pd.DataFrame(data, index=sample_ids)
    tpm_log = np.log2(tpm_df + 1)
    tpm_log["SAMPID"] = tpm_log.index

    merged = tpm_log.merge(samples[["SAMPID", "SUBJID", "SMTSD", "sex", "AGE"]], on="SAMPID", how="inner")

    # For each donor: compute coupling (product or residual) per tissue
    # Simple approach: for each donor × tissue, store mean SMAD3 and mean COL1A1
    # Then for each tissue pair: correlate coupling across donors

    # Focus on top tissues with most donors
    tissue_donor_counts = merged.groupby("SMTSD")["SUBJID"].nunique().sort_values(ascending=False)
    top_tissues = tissue_donor_counts.head(10).index.tolist()

    # For each donor × tissue: store SMAD3 and COL1A1 values
    # (one sample per donor per tissue in GTEx)
    donor_tissue = merged[merged["SMTSD"].isin(top_tissues)].copy()

    # Cross-tissue: for donors with data in BOTH tissue A and tissue B
    cross_tissue_results = []
    tissue_pairs_done = set()

    key_pair = ("SMAD3", "COL1A1")
    for t1 in top_tissues[:6]:
        for t2 in top_tissues[:6]:
            if t1 >= t2:
                continue

            d1 = donor_tissue[donor_tissue["SMTSD"] == t1][["SUBJID", "SMAD3", "COL1A1"]].set_index("SUBJID")
            d2 = donor_tissue[donor_tissue["SMTSD"] == t2][["SUBJID", "SMAD3", "COL1A1"]].set_index("SUBJID")

            common = d1.index.intersection(d2.index)
            if len(common) < 30:
                continue

            # Cross-tissue correlation of SMAD3
            rho_smad3, p_smad3 = spearman_safe(
                d1.loc[common, "SMAD3"].values.astype(float),
                d2.loc[common, "SMAD3"].values.astype(float)
            )
            # Cross-tissue correlation of COL1A1
            rho_col1a1, p_col1a1 = spearman_safe(
                d1.loc[common, "COL1A1"].values.astype(float),
                d2.loc[common, "COL1A1"].values.astype(float)
            )
            # Cross-tissue: SMAD3 in tissue1 vs COL1A1 in tissue2
            rho_cross, p_cross = spearman_safe(
                d1.loc[common, "SMAD3"].values.astype(float),
                d2.loc[common, "COL1A1"].values.astype(float)
            )

            cross_tissue_results.append({
                "tissue_1": t1, "tissue_2": t2, "n_donors": len(common),
                "rho_SMAD3_cross": rho_smad3, "p_SMAD3_cross": p_smad3,
                "rho_COL1A1_cross": rho_col1a1, "p_COL1A1_cross": p_col1a1,
                "rho_SMAD3_t1_COL1A1_t2": rho_cross, "p_cross": p_cross,
            })

    df_cross = pd.DataFrame(cross_tissue_results)
    df_cross.to_csv(RESULTS_DIR / "cross_tissue_coupling.csv", index=False)

    _log(f"\n  Cross-tissue pairs: {len(df_cross)}")
    _log(f"\n  SMAD3 cross-tissue correlation (same donor, different tissues):")
    _log(f"    Median ρ = {df_cross['rho_SMAD3_cross'].median():+.3f}")
    _log(f"    {(df_cross['p_SMAD3_cross'] < 0.05).sum()}/{len(df_cross)} significant")

    _log(f"\n  COL1A1 cross-tissue correlation:")
    _log(f"    Median ρ = {df_cross['rho_COL1A1_cross'].median():+.3f}")
    _log(f"    {(df_cross['p_COL1A1_cross'] < 0.05).sum()}/{len(df_cross)} significant")

    for _, r in df_cross.iterrows():
        t1_short = r["tissue_1"].split(" - ")[-1][:15]
        t2_short = r["tissue_2"].split(" - ")[-1][:15]
        _log(f"    {t1_short:>15s} × {t2_short:<15s} (n={r['n_donors']:3d}): "
             f"SMAD3 ρ={r['rho_SMAD3_cross']:+.3f}, "
             f"COL1A1 ρ={r['rho_COL1A1_cross']:+.3f}")

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 4: Sex × tissue interaction
    # ══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[4] Sex × tissue interaction")
    _log("=" * 70)

    sex_results = []
    key_pairs_sex = [
        ("SMAD3", "COL1A1", "SMAD3→COL1A1", "production"),
        ("SMAD3", "FN1", "SMAD3→FN1", "production"),
        ("ESR1", "COL1A1", "ESR1→COL1A1", "production"),
        ("RELA", "ICAM1", "RELA→ICAM1", "detection"),
        ("ACTB", "GAPDH", "ACTB↔GAPDH", "housekeeping"),
    ]

    for _, row in df.iterrows():
        rho_m = row.get("rho_male", np.nan)
        rho_f = row.get("rho_female", np.nan)
        if np.isnan(rho_m) or np.isnan(rho_f):
            continue
        sex_results.append({
            "tissue": row["tissue"], "label": row["label"], "category": row["category"],
            "rho_male": rho_m, "rho_female": rho_f,
            "delta_sex": rho_m - rho_f,
            "abs_delta_sex": abs(rho_m - rho_f),
        })

    df_sex = pd.DataFrame(sex_results)
    df_sex.to_csv(RESULTS_DIR / "sex_tissue_interaction.csv", index=False)

    _log(f"\n  Sex-stratified measurements: {len(df_sex)}")

    # Summary: which pairs show biggest sex difference?
    sex_by_pair = df_sex.groupby("label").agg(
        mean_delta=("delta_sex", "mean"),
        median_delta=("delta_sex", "median"),
        n_male_stronger=("delta_sex", lambda x: (x > 0).sum()),
        n_female_stronger=("delta_sex", lambda x: (x < 0).sum()),
        n=("delta_sex", "count"),
    ).sort_values("median_delta")

    _log(f"\n  Sex effect by pair (positive = stronger in males):")
    for label, r in sex_by_pair.iterrows():
        _log(f"    {label:<25s}: median Δ(M-F)={r['median_delta']:+.3f}, "
             f"M stronger: {r['n_male_stronger']:.0f}/{r['n']:.0f}")

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 5: Breakpoint detection
    # ══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[5] Breakpoint detection — when does coupling break?")
    _log("=" * 70)

    # For pairs with clear decline (negative slope): find decade of steepest drop
    bp_results = []

    for _, row in df.iterrows():
        if np.isnan(row.get("slope", np.nan)) or row["slope"] > -0.001:
            continue  # Only declining pairs

        rhos = {}
        for ad in age_decades:
            r = row.get(f"rho_{ad}", np.nan)
            if not np.isnan(r):
                rhos[ad] = r

        if len(rhos) < 4:
            continue

        # Find steepest single-decade drop
        sorted_decades = sorted(rhos.keys())
        max_drop = 0
        bp_decade = None
        for i in range(len(sorted_decades) - 1):
            drop = rhos[sorted_decades[i]] - rhos[sorted_decades[i + 1]]
            if drop > max_drop:
                max_drop = drop
                bp_decade = sorted_decades[i + 1]

        # Also: first decade where rho drops below baseline/2
        baseline = rhos[sorted_decades[0]]
        half_decade = None
        if baseline > 0.05:
            for ad in sorted_decades[1:]:
                if rhos[ad] < baseline / 2:
                    half_decade = ad
                    break

        bp_results.append({
            "tissue": row["tissue"], "label": row["label"], "category": row["category"],
            "slope": row["slope"],
            "rho_youngest": rhos[sorted_decades[0]],
            "rho_oldest": rhos[sorted_decades[-1]],
            "breakpoint_decade": bp_decade,
            "max_drop": max_drop,
            "half_life_decade": half_decade,
        })

    df_bp = pd.DataFrame(bp_results)
    df_bp.to_csv(RESULTS_DIR / "breakpoints.csv", index=False)

    _log(f"\n  Declining pairs: {len(df_bp)}")

    if len(df_bp) > 0:
        _log(f"\n  Breakpoint distribution (decade of steepest drop):")
        bp_counts = df_bp["breakpoint_decade"].value_counts().sort_index()
        for decade, count in bp_counts.items():
            _log(f"    {decade}: {count} pairs ({count/len(df_bp):.0%})")

        # Top declining pairs with breakpoints
        _log(f"\n  Top 15 declining pairs:")
        top_decline = df_bp.nsmallest(15, "slope")
        for _, r in top_decline.iterrows():
            t_short = r["tissue"][:25]
            _log(f"    {t_short:<25s} {r['label']:<20s} "
                 f"slope={r['slope']:+.5f} bp={r['breakpoint_decade']} "
                 f"ρ: {r['rho_youngest']:+.2f}→{r['rho_oldest']:+.2f}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY FIGURE
    # ══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[6] Summary figure")
    _log("=" * 70)

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Pair ranking bar chart (already saved separately)
    ax = fig.add_subplot(gs[0, 0])
    if len(df_pairs) > 0:
        top_n = min(20, len(df_pairs))
        sub_p = df_pairs.head(top_n)
        colors = [colors_map.get(r["category"], "gray") for _, r in sub_p.iterrows()]
        ax.barh(range(top_n), sub_p["median_slope"].values, color=colors, alpha=0.7)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(sub_p["label"].values, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Median slope across tissues")
        ax.set_title("A: Most declining pairs")

    # Panel B: Cross-tissue SMAD3 correlation
    ax = fig.add_subplot(gs[0, 1])
    if len(df_cross) > 0:
        ax.bar(range(len(df_cross)), df_cross["rho_SMAD3_cross"].values,
               color="tab:green", alpha=0.7)
        ax.set_xticks(range(len(df_cross)))
        labels = [f"{r['tissue_1'].split(' - ')[-1][:8]}\n×\n{r['tissue_2'].split(' - ')[-1][:8]}"
                  for _, r in df_cross.iterrows()]
        ax.set_xticklabels(labels, fontsize=6, rotation=0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Spearman ρ (same donor, different tissues)")
        ax.set_title("B: Cross-tissue SMAD3 correlation\n(systemic vs tissue-autonomous?)")

    # Panel C: Cross-tissue COL1A1
    ax = fig.add_subplot(gs[0, 2])
    if len(df_cross) > 0:
        ax.bar(range(len(df_cross)), df_cross["rho_COL1A1_cross"].values,
               color="tab:blue", alpha=0.7)
        ax.set_xticks(range(len(df_cross)))
        ax.set_xticklabels(labels, fontsize=6, rotation=0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Spearman ρ")
        ax.set_title("C: Cross-tissue COL1A1 correlation")

    # Panel D: Sex effect — scatter male vs female rho
    ax = fig.add_subplot(gs[1, 0])
    if len(df_sex) > 0:
        for cat, color in colors_map.items():
            sub = df_sex[df_sex["category"] == cat]
            if len(sub) > 0:
                ax.scatter(sub["rho_female"], sub["rho_male"], c=color, s=15,
                           alpha=0.5, label=cat, edgecolors="none")
        lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
                  abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3)
        ax.set_xlabel("Female ρ")
        ax.set_ylabel("Male ρ")
        ax.set_title("D: Sex effect on coupling\n(above diagonal = stronger in males)")
        ax.legend(fontsize=7, markerscale=2)

    # Panel E: Breakpoint distribution
    ax = fig.add_subplot(gs[1, 1])
    if len(df_bp) > 0 and len(bp_counts) > 0:
        for cat, color in [("production", "tab:green"), ("detection", "tab:orange")]:
            sub = df_bp[df_bp["category"] == cat]
            if len(sub) > 0:
                bp_c = sub["breakpoint_decade"].value_counts().sort_index()
                ax.bar([i - 0.15 if cat == "production" else i + 0.15
                        for i in range(len(bp_c))],
                       bp_c.values, width=0.3, color=color, alpha=0.7, label=cat)
                ax.set_xticks(range(len(bp_c)))
                ax.set_xticklabels(bp_c.index, fontsize=8)
        ax.set_xlabel("Decade of steepest coupling drop")
        ax.set_ylabel("Number of tissue×pair combinations")
        ax.set_title("E: When does coupling break?\n(decade of steepest decline)")
        ax.legend()

    # Panel F: Production vs detection slope scatter per tissue
    ax = fig.add_subplot(gs[1, 2])
    tissue_pd = []
    for tissue in df["tissue"].unique():
        t_data = df[df["tissue"] == tissue]
        p_slopes = t_data[t_data["category"] == "production"]["slope"].dropna()
        d_slopes = t_data[t_data["category"] == "detection"]["slope"].dropna()
        if len(p_slopes) >= 3 and len(d_slopes) >= 3:
            tissue_pd.append({
                "tissue": tissue,
                "mean_P_slope": p_slopes.mean(),
                "mean_D_slope": d_slopes.mean(),
            })
    df_pd = pd.DataFrame(tissue_pd)
    if len(df_pd) > 0:
        ax.scatter(df_pd["mean_P_slope"], df_pd["mean_D_slope"], s=40, alpha=0.7,
                   edgecolors="gray", linewidth=0.3)
        for _, r in df_pd.iterrows():
            ax.annotate(r["tissue"].split(" - ")[-1][:10], (r["mean_P_slope"], r["mean_D_slope"]),
                        fontsize=5, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.3)
        ax.set_xlabel("Mean production slope (Δρ/year)")
        ax.set_ylabel("Mean detection slope (Δρ/year)")
        ax.set_title("F: Per-tissue production vs detection trends\n"
                     "Q1=both↑ Q2=P↓D↑ Q3=both↓ Q4=P↑D↓")

    fig.suptitle("GTEx Coupling Atlas — Mining Results\n948 donors, 26+ tissues, 30+ TF→target pairs",
                 fontsize=14, fontweight="bold")
    fig.savefig(RESULTS_DIR / "atlas_mining_summary.png", dpi=150)
    plt.close(fig)
    _log(f"  Saved atlas_mining_summary.png")

    _log(f"\n  Total time: {time.time()-t0:.0f}s")
    _log(f"  Results: {RESULTS_DIR}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
