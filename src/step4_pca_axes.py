"""
Step 4: PCA on tissue coupling fingerprints + tissue property annotation

Questions:
1. How many PCs explain >80% variance? (low-dimensional structure?)
2. Which tissue properties correlate with PC1/PC2?
3. Exposure? Turnover? Immune fraction? Mechanical load? Hormonal sensitivity?
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
RESULTS_DIR = BASE / "results" / "step4_pca"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STEP2_DIR = BASE / "results" / "step2_mining"


def _log(msg):
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════════
# TISSUE ANNOTATIONS — manual scoring based on biology
# ═══════════════════════════════════════════════════════════════════
# Each axis scored 1-3 (low/medium/high)

TISSUE_ANNOTATIONS = {
    # Barrier tissues — direct environmental contact
    "Skin - Sun Exposed (Lower leg)":       {"exposure": 3, "turnover": 2, "immune": 2, "mechanical": 1, "hormonal": 2, "info_load": 3},
    "Skin - Not Sun Exposed (Suprapubic)":  {"exposure": 2, "turnover": 2, "immune": 2, "mechanical": 1, "hormonal": 2, "info_load": 2},
    "Esophagus - Mucosa":                   {"exposure": 3, "turnover": 3, "immune": 2, "mechanical": 2, "hormonal": 1, "info_load": 2},
    "Esophagus - Muscularis":               {"exposure": 2, "turnover": 1, "immune": 1, "mechanical": 3, "hormonal": 1, "info_load": 2},
    "Esophagus - Gastroesophageal Junction": {"exposure": 3, "turnover": 3, "immune": 2, "mechanical": 2, "hormonal": 1, "info_load": 2},
    "Colon - Sigmoid":                      {"exposure": 3, "turnover": 3, "immune": 3, "mechanical": 2, "hormonal": 1, "info_load": 3},
    "Colon - Transverse":                   {"exposure": 3, "turnover": 3, "immune": 3, "mechanical": 2, "hormonal": 1, "info_load": 3},
    "Small Intestine - Terminal Ileum":      {"exposure": 3, "turnover": 3, "immune": 3, "mechanical": 2, "hormonal": 1, "info_load": 3},
    "Stomach":                              {"exposure": 3, "turnover": 3, "immune": 2, "mechanical": 2, "hormonal": 1, "info_load": 2},
    "Lung":                                 {"exposure": 3, "turnover": 2, "immune": 3, "mechanical": 2, "hormonal": 1, "info_load": 3},

    # Interface tissues — indirect contact
    "Artery - Aorta":                       {"exposure": 2, "turnover": 1, "immune": 2, "mechanical": 3, "hormonal": 1, "info_load": 2},
    "Artery - Tibial":                      {"exposure": 2, "turnover": 1, "immune": 1, "mechanical": 3, "hormonal": 1, "info_load": 2},
    "Whole Blood":                          {"exposure": 2, "turnover": 3, "immune": 3, "mechanical": 1, "hormonal": 2, "info_load": 3},
    "Breast - Mammary Tissue":              {"exposure": 2, "turnover": 2, "immune": 2, "mechanical": 1, "hormonal": 3, "info_load": 2},
    "Prostate":                             {"exposure": 2, "turnover": 2, "immune": 2, "mechanical": 1, "hormonal": 3, "info_load": 1},

    # Deep tissues — isolated
    "Muscle - Skeletal":                    {"exposure": 1, "turnover": 1, "immune": 1, "mechanical": 3, "hormonal": 2, "info_load": 2},
    "Heart - Left Ventricle":               {"exposure": 1, "turnover": 1, "immune": 1, "mechanical": 3, "hormonal": 1, "info_load": 2},
    "Heart - Atrial Appendage":             {"exposure": 1, "turnover": 1, "immune": 1, "mechanical": 3, "hormonal": 1, "info_load": 2},
    "Adipose - Subcutaneous":               {"exposure": 1, "turnover": 1, "immune": 2, "mechanical": 1, "hormonal": 2, "info_load": 1},
    "Adipose - Visceral (Omentum)":         {"exposure": 1, "turnover": 1, "immune": 2, "mechanical": 1, "hormonal": 2, "info_load": 2},
    "Liver":                                {"exposure": 2, "turnover": 2, "immune": 2, "mechanical": 1, "hormonal": 1, "info_load": 2},
    "Pancreas":                             {"exposure": 1, "turnover": 1, "immune": 1, "mechanical": 1, "hormonal": 2, "info_load": 1},
    "Adrenal Gland":                        {"exposure": 1, "turnover": 2, "immune": 1, "mechanical": 1, "hormonal": 3, "info_load": 1},
    "Thyroid":                              {"exposure": 1, "turnover": 2, "immune": 2, "mechanical": 1, "hormonal": 3, "info_load": 1},
    "Nerve - Tibial":                       {"exposure": 1, "turnover": 1, "immune": 1, "mechanical": 2, "hormonal": 1, "info_load": 2},
    "Spleen":                               {"exposure": 1, "turnover": 3, "immune": 3, "mechanical": 1, "hormonal": 1, "info_load": 2},
    "Testis":                               {"exposure": 1, "turnover": 3, "immune": 1, "mechanical": 1, "hormonal": 3, "info_load": 1},
    "Pituitary":                            {"exposure": 1, "turnover": 2, "immune": 1, "mechanical": 1, "hormonal": 3, "info_load": 2},
    "Minor Salivary Gland":                 {"exposure": 2, "turnover": 2, "immune": 2, "mechanical": 1, "hormonal": 1, "info_load": 1},
    "Cells - Cultured fibroblasts":         {"exposure": 1, "turnover": 2, "immune": 1, "mechanical": 1, "hormonal": 1, "info_load": 1},
    "Cells - EBV-transformed lymphocytes":  {"exposure": 1, "turnover": 3, "immune": 3, "mechanical": 1, "hormonal": 1, "info_load": 1},
    "Vagina":                               {"exposure": 3, "turnover": 2, "immune": 2, "mechanical": 1, "hormonal": 3, "info_load": 2},
    "Uterus":                               {"exposure": 2, "turnover": 2, "immune": 2, "mechanical": 2, "hormonal": 3, "info_load": 2},
    "Ovary":                                {"exposure": 1, "turnover": 2, "immune": 1, "mechanical": 1, "hormonal": 3, "info_load": 2},
    "Kidney - Cortex":                      {"exposure": 2, "turnover": 2, "immune": 2, "mechanical": 1, "hormonal": 2, "info_load": 2},
    "Kidney - Medulla":                     {"exposure": 1, "turnover": 1, "immune": 1, "mechanical": 1, "hormonal": 2, "info_load": 1},
    "Bladder":                              {"exposure": 2, "turnover": 2, "immune": 2, "mechanical": 2, "hormonal": 1, "info_load": 2},
}


def main():
    t0 = time.time()
    _log("=" * 70)
    _log("TISSUE PCA — Hidden axes of coupling change")
    _log("=" * 70)

    # Load coupling data from step2
    df_coupling = pd.read_csv(BASE / "results" / "step1_gtex" / "gtex_coupling_all.csv")

    # Compute slopes (same as step2)
    age_mids = {"20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65, "70-79": 75}
    age_decades = sorted(set(c.replace("rho_", "") for c in df_coupling.columns
                              if c.startswith("rho_") and c[4:5].isdigit()))

    slopes = []
    for _, row in df_coupling.iterrows():
        age_vals, rho_vals = [], []
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

    df_coupling["slope"] = slopes

    # Also compute: mean absolute coupling (baseline strength)
    mean_rhos = []
    for _, row in df_coupling.iterrows():
        rho_vals = [row.get(f"rho_{ad}", np.nan) for ad in age_decades]
        rho_vals = [r for r in rho_vals if not np.isnan(r)]
        mean_rhos.append(np.mean(rho_vals) if rho_vals else np.nan)
    df_coupling["mean_rho"] = mean_rhos

    # Pivot: tissue × pair → slope
    pivot = df_coupling.pivot_table(index="tissue", columns="label", values="slope", aggfunc="first")
    pivot = pivot.dropna(thresh=pivot.shape[1] * 0.5, axis=0)
    pivot = pivot.dropna(thresh=pivot.shape[0] * 0.5, axis=1)
    pivot_filled = pivot.fillna(0)

    _log(f"  Coupling matrix: {pivot_filled.shape[0]} tissues × {pivot_filled.shape[1]} pairs")

    # ═══════════════════════════════════════════════════════════════════
    # Step 1: PCA
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[1] PCA on coupling slope matrix")
    _log("=" * 70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot_filled.values)

    pca = PCA()
    scores = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_explained)

    _log(f"\n  Variance explained:")
    for i, (v, c) in enumerate(zip(var_explained, cum_var)):
        marker = " ← 80%" if c >= 0.8 and (i == 0 or cum_var[i-1] < 0.8) else ""
        _log(f"    PC{i+1}: {v:.1%} (cum: {c:.1%}){marker}")
        if c > 0.95:
            break

    n_80 = np.argmax(cum_var >= 0.8) + 1
    _log(f"\n  PCs for 80% variance: {n_80}")
    _log(f"  PCs for 90% variance: {np.argmax(cum_var >= 0.9) + 1}")

    # PC loadings — what pairs drive each PC?
    _log(f"\n  PC1 loadings (top 5 positive, top 5 negative):")
    loadings = pd.DataFrame(pca.components_.T, index=pivot_filled.columns, columns=[f"PC{i+1}" for i in range(len(var_explained))])
    for pc in ["PC1", "PC2", "PC3"]:
        _log(f"\n  {pc} loadings:")
        top_pos = loadings[pc].nlargest(5)
        top_neg = loadings[pc].nsmallest(5)
        for label, v in top_pos.items():
            cat = df_coupling[df_coupling["label"] == label]["category"].iloc[0] if label in df_coupling["label"].values else "?"
            _log(f"    + {label:<25s} ({cat:<12s}): {v:+.3f}")
        for label, v in top_neg.items():
            cat = df_coupling[df_coupling["label"] == label]["category"].iloc[0] if label in df_coupling["label"].values else "?"
            _log(f"    − {label:<25s} ({cat:<12s}): {v:+.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 2: Annotate tissues
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[2] Tissue annotations")
    _log("=" * 70)

    tissues = pivot_filled.index.tolist()
    annot_data = []
    for tissue in tissues:
        if tissue in TISSUE_ANNOTATIONS:
            row = {"tissue": tissue}
            row.update(TISSUE_ANNOTATIONS[tissue])
            annot_data.append(row)
        else:
            _log(f"  ⚠️ No annotation for: {tissue}")
            annot_data.append({"tissue": tissue, "exposure": np.nan, "turnover": np.nan,
                               "immune": np.nan, "mechanical": np.nan, "hormonal": np.nan,
                               "info_load": np.nan})

    df_annot = pd.DataFrame(annot_data).set_index("tissue")
    df_annot = df_annot.loc[tissues]  # align order

    # Add PC scores
    for i in range(min(5, scores.shape[1])):
        df_annot[f"PC{i+1}"] = scores[:, i]

    # Add aggregate slope stats
    for tissue in tissues:
        t_slopes = pivot_filled.loc[tissue]
        df_annot.loc[tissue, "mean_slope"] = t_slopes.mean()
        df_annot.loc[tissue, "n_declining"] = (t_slopes < 0).sum()
        df_annot.loc[tissue, "coupling_volatility"] = t_slopes.std()

    df_annot.to_csv(RESULTS_DIR / "tissue_annotations_with_pcs.csv")
    _log(f"  Annotated {len(df_annot)} tissues")

    # ═══════════════════════════════════════════════════════════════════
    # Step 3: Correlate PCs with tissue properties
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[3] PC × tissue property correlations")
    _log("=" * 70)

    axes = ["exposure", "turnover", "immune", "mechanical", "hormonal", "info_load"]

    corr_results = []
    header = f"  {'':>15s}" + "".join(f" {ax:>12s}" for ax in axes)
    _log(header)

    for pc in ["PC1", "PC2", "PC3", "PC4", "PC5", "mean_slope", "coupling_volatility"]:
        parts = [f"  {pc:>15s}"]
        for ax in axes:
            valid = df_annot[[pc, ax]].dropna()
            if len(valid) < 8:
                parts.append(f" {'N/A':>12s}")
                continue
            rho, p = stats.spearmanr(valid[pc], valid[ax])
            sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
            parts.append(f" {rho:+.3f}{sig:>3s}     ")
            corr_results.append({"metric": pc, "axis": ax, "rho": rho, "p": p})
        _log("".join(parts))

    df_corr = pd.DataFrame(corr_results)
    df_corr.to_csv(RESULTS_DIR / "pc_property_correlations.csv", index=False)

    # Best predictor for each PC
    _log(f"\n  Best predictor for each PC:")
    for pc in ["PC1", "PC2", "PC3"]:
        sub = df_corr[df_corr["metric"] == pc]
        if len(sub) > 0:
            best = sub.loc[sub["p"].idxmin()]
            _log(f"    {pc}: {best['axis']} (ρ={best['rho']:+.3f}, p={best['p']:.4f})")

    # ═══════════════════════════════════════════════════════════════════
    # Step 4: Multiple regression — which axes explain PCs?
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[4] Multiple regression: tissue properties → PCs")
    _log("=" * 70)

    from sklearn.linear_model import LinearRegression

    for pc in ["PC1", "PC2", "PC3"]:
        valid = df_annot[[pc] + axes].dropna()
        if len(valid) < 10:
            continue

        X_reg = valid[axes].values
        y_reg = valid[pc].values

        lr = LinearRegression()
        lr.fit(X_reg, y_reg)
        r2 = lr.score(X_reg, y_reg)

        _log(f"\n  {pc} ~ {' + '.join(axes)}")
        _log(f"    R² = {r2:.3f} (n={len(valid)})")
        for ax, coef in zip(axes, lr.coef_):
            _log(f"    {ax:>15s}: β = {coef:+.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 5: Figures
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[5] Figures")
    _log("=" * 70)

    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Scree plot
    ax = fig.add_subplot(gs[0, 0])
    n_pcs = min(15, len(var_explained))
    ax.bar(range(1, n_pcs + 1), var_explained[:n_pcs] * 100, color="steelblue", alpha=0.7)
    ax.plot(range(1, n_pcs + 1), cum_var[:n_pcs] * 100, "ro-", markersize=4)
    ax.axhline(80, color="red", linestyle="--", alpha=0.3, label="80%")
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(f"A: Scree plot — {n_80} PCs for 80%")
    ax.legend()

    # Panels B-F: PC1 vs PC2 colored by each axis
    for i, (ax_name, cmap) in enumerate([
        ("exposure", "YlOrRd"), ("turnover", "YlGnBu"), ("immune", "PuRd"),
        ("mechanical", "YlOrBr"), ("hormonal", "RdPu"), ("info_load", "viridis")
    ]):
        row, col = divmod(i + 1, 3)
        ax = fig.add_subplot(gs[row, col])

        valid = df_annot[["PC1", "PC2", ax_name]].dropna()
        sc = ax.scatter(valid["PC1"], valid["PC2"], c=valid[ax_name],
                        cmap=cmap, s=80, edgecolors="gray", linewidth=0.5, vmin=1, vmax=3)

        for tissue in valid.index:
            short = tissue.split(" - ")[-1][:12]
            ax.annotate(short, (valid.loc[tissue, "PC1"], valid.loc[tissue, "PC2"]),
                        fontsize=5, alpha=0.7)

        plt.colorbar(sc, ax=ax, label=ax_name, shrink=0.7)

        # Add correlation text
        rho1, p1 = stats.spearmanr(valid["PC1"], valid[ax_name])
        rho2, p2 = stats.spearmanr(valid["PC2"], valid[ax_name])
        ax.set_xlabel(f"PC1 ({var_explained[0]:.0%} var) | ρ={rho1:+.2f} p={p1:.3f}")
        ax.set_ylabel(f"PC2 ({var_explained[1]:.0%} var) | ρ={rho2:+.2f} p={p2:.3f}")
        ax.set_title(f"{'BCDEFG'[i]}: colored by {ax_name}")

    # Panel G: Correlation heatmap (PC × axis)
    ax = fig.add_subplot(gs[2, 0])
    corr_pivot = df_corr[df_corr["metric"].isin(["PC1", "PC2", "PC3"])].pivot_table(
        index="metric", columns="axis", values="rho")
    if len(corr_pivot) > 0:
        corr_pivot = corr_pivot[axes]
        im = ax.imshow(corr_pivot.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_yticks(range(len(corr_pivot)))
        ax.set_yticklabels(corr_pivot.index)
        ax.set_xticks(range(len(corr_pivot.columns)))
        ax.set_xticklabels(corr_pivot.columns, rotation=45, ha="right")
        plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.7)

        # Add text values
        for i in range(corr_pivot.shape[0]):
            for j in range(corr_pivot.shape[1]):
                v = corr_pivot.values[i, j]
                p_val = df_corr[(df_corr["metric"] == corr_pivot.index[i]) &
                                (df_corr["axis"] == corr_pivot.columns[j])]["p"].values
                sig = "*" if len(p_val) > 0 and p_val[0] < 0.05 else ""
                ax.text(j, i, f"{v:.2f}{sig}", ha="center", va="center", fontsize=8)
        ax.set_title("G: PC × tissue property correlations")

    # Panel H: Discriminating cases
    ax = fig.add_subplot(gs[2, 1])
    disc_tissues = ["Lung", "Artery - Tibial", "Colon - Sigmoid", "Muscle - Skeletal",
                    "Skin - Sun Exposed (Lower leg)", "Heart - Left Ventricle",
                    "Nerve - Tibial", "Spleen"]
    disc_data = []
    for t in disc_tissues:
        if t in df_annot.index:
            disc_data.append({
                "tissue": t.split(" - ")[-1][:15],
                "mean_slope": df_annot.loc[t, "mean_slope"] * 1000,  # scale for visibility
                "exposure": df_annot.loc[t, "exposure"],
                "turnover": df_annot.loc[t, "turnover"],
            })
    if disc_data:
        dd = pd.DataFrame(disc_data)
        x = np.arange(len(dd))
        width = 0.25
        ax.bar(x - width, dd["exposure"], width, label="Exposure", color="tab:orange", alpha=0.7)
        ax.bar(x, dd["turnover"], width, label="Turnover", color="tab:blue", alpha=0.7)
        ax.bar(x + width, dd["mean_slope"], width, label="Mean slope (×1000)", color="tab:green", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(dd["tissue"], rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("H: Discriminating cases\n(exposure vs turnover vs coupling change)")

    # Panel I: Mean slope vs each axis (best predictor)
    ax = fig.add_subplot(gs[2, 2])
    # Find best predictor of mean_slope
    best_axis = None
    best_p = 1.0
    for ax_name in axes:
        valid = df_annot[["mean_slope", ax_name]].dropna()
        if len(valid) > 8:
            rho, p = stats.spearmanr(valid["mean_slope"], valid[ax_name])
            if p < best_p:
                best_p = p
                best_axis = ax_name
                best_rho = rho

    if best_axis:
        valid = df_annot[["mean_slope", best_axis]].dropna()
        ax.scatter(valid[best_axis], valid["mean_slope"] * 1000, s=60, alpha=0.7,
                   edgecolors="gray", linewidth=0.3)
        for tissue in valid.index:
            short = tissue.split(" - ")[-1][:12]
            ax.annotate(short, (valid.loc[tissue, best_axis],
                                valid.loc[tissue, "mean_slope"] * 1000),
                        fontsize=5, alpha=0.7)
        ax.set_xlabel(f"{best_axis} score")
        ax.set_ylabel("Mean coupling slope (×1000)")
        ax.set_title(f"I: Best predictor of overall coupling change\n"
                     f"{best_axis}: ρ={best_rho:+.3f}, p={best_p:.4f}")

    fig.suptitle("Tissue PCA — Hidden axes of age-related coupling change\n"
                 "948 GTEx donors, 26+ tissues, 30+ TF→target pairs",
                 fontsize=14, fontweight="bold")
    fig.savefig(RESULTS_DIR / "pca_tissue_axes.png", dpi=150)
    plt.close(fig)
    _log(f"  Saved pca_tissue_axes.png")

    # ═══════════════════════════════════════════════════════════════════
    # Step 6: Discriminating cases — exposure vs turnover
    # ═══════════════════════════════════════════════════════════════════
    _log("\n" + "=" * 70)
    _log("[6] Discriminating cases")
    _log("=" * 70)

    cases = [
        ("Lung", "barrier + partially renewable", "exposure predicts decline, turnover predicts stability"),
        ("Artery - Tibial", "interface + slow-renewing", "exposure predicts stability, turnover predicts decline"),
        ("Colon - Sigmoid", "barrier + renewable", "both predict stability"),
        ("Muscle - Skeletal", "deep + post-mitotic", "both predict decline"),
        ("Spleen", "deep + high-turnover", "exposure: decline, turnover: stability"),
    ]

    for tissue, desc, prediction in cases:
        if tissue in df_annot.index:
            ms = df_annot.loc[tissue, "mean_slope"]
            nd = df_annot.loc[tissue, "n_declining"]
            total = pivot_filled.shape[1]
            direction = "DECLINING" if ms < 0 else "INCREASING"
            _log(f"\n  {tissue} ({desc}):")
            _log(f"    Mean slope: {ms*1000:+.2f}×10⁻³, {nd:.0f}/{total} pairs decline → {direction}")
            _log(f"    Prediction: {prediction}")
            _log(f"    Exposure={df_annot.loc[tissue, 'exposure']:.0f}, "
                 f"Turnover={df_annot.loc[tissue, 'turnover']:.0f}, "
                 f"Immune={df_annot.loc[tissue, 'immune']:.0f}")

    _log(f"\n  Time: {time.time()-t0:.0f}s")
    _log(f"  Results: {RESULTS_DIR}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
