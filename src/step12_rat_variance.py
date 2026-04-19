"""
Tests 2+3: Cross-species π_tissue (rat) + CR effect
Single-cell approach: subsample cells per tissue, compute V_tissue/V_total
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse
import scanpy as sc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

RAT_PATH = Path("/Users/teo/Desktop/research/oscilatory/results/h010_rat_cr/data/rat_atlas.h5ad")
RESULTS_DIR = Path("/Users/teo/Desktop/research/coupling_atlas/results/step12_rat")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CELLS_PER_TISSUE = 300
MIN_TISSUES = 5

def _log(m): print(m, flush=True)

def compute_pi_tissue(X_sub, tissue_labels, n_subsample=CELLS_PER_TISSUE):
    """Compute π_tissue for a matrix of cells × genes with tissue labels."""
    tissues = sorted(set(tissue_labels))
    if len(tissues) < MIN_TISSUES:
        return None, None, None

    # Subsample per tissue
    indices = []
    tissue_ids = []
    for t in tissues:
        t_idx = np.where(tissue_labels == t)[0]
        if len(t_idx) < 50:
            continue
        chosen = np.random.choice(t_idx, min(n_subsample, len(t_idx)), replace=False)
        indices.extend(chosen)
        tissue_ids.extend([t] * len(chosen))

    indices = np.array(indices)
    tissue_ids = np.array(tissue_ids)
    tissues_used = sorted(set(tissue_ids))
    if len(tissues_used) < MIN_TISSUES:
        return None, None, None

    # Get expression
    mat = X_sub[indices]
    if issparse(mat):
        mat = np.asarray(mat.todense())
    mat = np.log2(mat + 1)

    # Per-gene ANOVA
    n_genes = mat.shape[1]
    pi_tissue_vals = []

    for g in range(n_genes):
        vals = mat[:, g]
        if np.std(vals) < 1e-6:
            continue

        grand_mean = np.mean(vals)
        ss_total = np.sum((vals - grand_mean) ** 2)
        if ss_total < 1e-10:
            continue

        # SS between tissues
        ss_tissue = 0
        for t in tissues_used:
            mask = tissue_ids == t
            t_mean = np.mean(vals[mask])
            ss_tissue += mask.sum() * (t_mean - grand_mean) ** 2

        pi_tissue_vals.append(ss_tissue / ss_total)

    return np.median(pi_tissue_vals), np.mean(pi_tissue_vals), len(pi_tissue_vals)

def main():
    t0 = time.time()
    _log("=" * 60)
    _log("TESTS 2+3: Rat π_tissue + CR effect")
    _log("=" * 60)

    adata = sc.read_h5ad(RAT_PATH)
    _log(f"  {adata.shape[0]} cells, {adata.shape[1]} genes")
    _log(f"  Conditions: {adata.obs['condition'].value_counts().to_dict()}")
    _log(f"  Tissues: {sorted(adata.obs['tissue'].unique())}")

    X = adata.X
    # Normalize if not already
    sc.pp.normalize_total(adata, target_sum=1e4)
    X = adata.X

    # PSEUDOBULK approach: aggregate per GSM, then ANOVA like GTEx
    _log("\n  Building pseudobulk per GSM...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    pb_data = {}
    for gsm, grp in adata.obs.groupby("GSM"):
        if len(grp) < 30:
            continue
        cell_idx = np.array([adata.obs.index.get_loc(i) for i in grp.index])
        mat = adata.X[cell_idx]
        if issparse(mat):
            mat = np.asarray(mat.todense())
        pb_data[gsm] = {
            "condition": grp["condition"].iloc[0],
            "tissue": grp["tissue"].iloc[0],
            "n_cells": len(grp),
            "expr": mat.mean(axis=0).flatten(),
        }

    _log(f"  Pseudobulk samples: {len(pb_data)}")

    # For each condition: ANOVA on pseudobulk (GSM × tissue)
    # Since each GSM = 1 tissue, treat as: samples grouped by tissue
    results = []
    for condition in ["young", "old_AL", "old_CR"]:
        cond_gsms = {k: v for k, v in pb_data.items() if v["condition"] == condition}
        tissues = sorted(set(v["tissue"] for v in cond_gsms.values()))
        _log(f"\n  {condition}: {len(cond_gsms)} GSMs, {len(tissues)} tissues")

        # Build matrix: GSMs × genes, with tissue labels
        gsm_list = sorted(cond_gsms.keys())
        n_genes = len(next(iter(cond_gsms.values()))["expr"])
        expr_mat = np.zeros((len(gsm_list), n_genes))
        tissue_labels = []
        for i, gsm in enumerate(gsm_list):
            expr_mat[i] = cond_gsms[gsm]["expr"]
            tissue_labels.append(cond_gsms[gsm]["tissue"])
        tissue_labels = np.array(tissue_labels)

        # Per-gene ANOVA: V_tissue / V_total
        pi_vals = []
        for g in range(n_genes):
            vals = expr_mat[:, g]
            if np.std(vals) < 1e-6:
                continue
            grand_mean = np.mean(vals)
            ss_total = np.sum((vals - grand_mean) ** 2)
            if ss_total < 1e-10:
                continue
            ss_tissue = 0
            for t in tissues:
                mask = tissue_labels == t
                if mask.sum() == 0:
                    continue
                t_mean = np.mean(vals[mask])
                ss_tissue += mask.sum() * (t_mean - grand_mean) ** 2
            pi_vals.append(ss_tissue / ss_total)

        med_pi = np.median(pi_vals)
        _log(f"  π_tissue (pseudobulk): {med_pi:.4f} (n_genes={len(pi_vals)})")

        results.append({
            "condition": condition,
            "pi_tissue_median": med_pi,
            "pi_tissue_mean": np.mean(pi_vals),
            "pi_tissue_q25": np.percentile(pi_vals, 25),
            "pi_tissue_q75": np.percentile(pi_vals, 75),
            "n_genes": len(pi_vals),
            "n_gsms": len(gsm_list),
        })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "rat_pi_tissue.csv", index=False)

    _log("\n  RESULTS:")
    for _, r in df.iterrows():
        _log(f"    {r['condition']:>8s}: π_tissue = {r['pi_tissue_median']:.4f} "
             f"[Q25={r['pi_tissue_q25']:.4f}, Q75={r['pi_tissue_q75']:.4f}]")

    _log(f"\n  Human GTEx reference: π_tissue ≈ 0.73")
    _log(f"  NOTE: Rat uses pseudobulk per GSM (2 samples per tissue per condition)")
    _log(f"  GTEx uses 47-90 donors per tissue. Different N → different π scale is expected.")
    if len(df) == 3:
        y = df[df["condition"] == "young"]["pi_tissue_median"].values[0]
        o = df[df["condition"] == "old_AL"]["pi_tissue_median"].values[0]
        c = df[df["condition"] == "old_CR"]["pi_tissue_median"].values[0]
        _log(f"  Aging effect: {o - y:+.4f} (old_AL - young)")
        _log(f"  CR effect:    {c - o:+.4f} (old_CR - old_AL)")
        if abs(y - o) > 0.001:
            _log(f"  CR rescue:    {(c - o) / (y - o):.1%}")
        else:
            _log(f"  No aging effect to rescue")

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"young": "tab:blue", "old_AL": "tab:red", "old_CR": "tab:green"}
    for i, (_, r) in enumerate(df.iterrows()):
        ax.bar(i, r["pi_tissue_median"], color=colors.get(r["condition"], "gray"),
               yerr=[[r["pi_tissue_median"] - r["pi_tissue_q25"]],
                     [r["pi_tissue_q75"] - r["pi_tissue_median"]]],
               capsize=5, alpha=0.7)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["condition"])
    ax.axhline(0.73, color="purple", linestyle="--", alpha=0.5, label="Human GTEx (0.73)")
    ax.set_ylabel("π_tissue (median across genes)")
    ax.set_title("Rat: Tissue identity proportion\nyoung (5mo) vs old_AL (27mo) vs old_CR (27mo)")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rat_pi_tissue.png", dpi=150)
    plt.close()

    _log(f"\n  Time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
