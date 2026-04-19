"""
Test 5: Embryogenesis — does π_cell_type increase during development?
MOCA: 2M cells, 38 cell types, E9.5-E13.5
MatrixMarket sparse format (genes × cells)
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import mmread
from scipy.sparse import issparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

MOCA_DIR = Path("/Users/teo/Desktop/research/oscilatory/data/moca")
RESULTS_DIR = Path("/Users/teo/Desktop/research/coupling_atlas/results/step14_embryo")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CELLS_PER_TYPE = 150
MIN_TYPES = 8

def _log(m): print(m, flush=True)

def main():
    t0 = time.time()
    _log("=" * 60)
    _log("TEST 5: Embryogenesis π_cell_type trajectory")
    _log("=" * 60)

    # Load annotations
    annot = pd.read_csv(MOCA_DIR / "GSE119945_cell_annotate.csv.gz")
    _log(f"  Annotations: {len(annot)} cells")
    _log(f"  Stages: {sorted(annot['day'].unique())}")

    # Load sparse matrix
    _log(f"  Loading MatrixMarket sparse matrix...")
    mat = mmread(MOCA_DIR / "GSE119945_gene_count.txt.gz")  # genes × cells
    _log(f"  Matrix: {mat.shape} (genes × cells), nnz={mat.nnz}")
    mat = mat.tocsr()  # for row slicing (genes)

    # Random gene subset for speed
    n_genes = mat.shape[0]
    np.random.seed(42)
    gene_idx = sorted(np.random.choice(n_genes, min(3000, n_genes), replace=False))

    # Library sizes from annotation
    lib_sizes = annot["Total_mRNAs"].values.astype(float)
    lib_sizes[lib_sizes == 0] = 1

    # Process per stage
    results = []
    for stage in sorted(annot["day"].unique()):
        stage_mask = annot["day"] == stage
        stage_cells = np.where(stage_mask.values)[0]
        ct_labels = annot.loc[stage_mask, "Main_Cluster"].values

        valid_types = [t for t in set(ct_labels) if (ct_labels == t).sum() >= 50]
        if len(valid_types) < MIN_TYPES:
            _log(f"  E{stage}: only {len(valid_types)} valid types, skipping")
            continue

        _log(f"  E{stage}: {len(stage_cells)} cells, {len(valid_types)} types (≥50 cells)")

        # Subsample per type
        pi_reps = []
        for rep in range(5):
            np.random.seed(rep * 13 + 7)
            indices = []
            type_ids = []
            for t in valid_types:
                t_idx = stage_cells[ct_labels == t]
                chosen = np.random.choice(t_idx, min(CELLS_PER_TYPE, len(t_idx)), replace=False)
                indices.extend(chosen)
                type_ids.extend([t] * len(chosen))

            indices = np.array(indices)
            type_ids = np.array(type_ids)

            # Extract expression for these cells, subset genes
            # mat is genes × cells, need cells subset
            sub_mat = mat[gene_idx][:, indices]  # (n_genes_sub, n_cells_sub)
            sub_mat = sub_mat.toarray().T  # (n_cells_sub, n_genes_sub)

            # Normalize
            sub_lib = lib_sizes[indices]
            sub_mat = sub_mat / sub_lib[:, None] * 10000
            sub_mat = np.log2(sub_mat + 1)

            # ANOVA per gene
            types_used = sorted(set(type_ids))
            pi_vals = []
            for g in range(sub_mat.shape[1]):
                vals = sub_mat[:, g]
                if np.std(vals) < 1e-6:
                    continue
                gm = np.mean(vals)
                ss_total = np.sum((vals - gm) ** 2)
                if ss_total < 1e-10:
                    continue
                ss_type = sum(
                    (type_ids == t).sum() * (np.mean(vals[type_ids == t]) - gm) ** 2
                    for t in types_used
                )
                pi_vals.append(ss_type / ss_total)

            if pi_vals:
                pi_reps.append(np.median(pi_vals))

        if pi_reps:
            results.append({
                "stage": f"E{stage}",
                "day": stage,
                "pi_celltype": np.mean(pi_reps),
                "pi_ci_lo": np.percentile(pi_reps, 10),
                "pi_ci_hi": np.percentile(pi_reps, 90),
                "n_cells": len(stage_cells),
                "n_types": len(valid_types),
            })
            _log(f"    π_cell_type = {np.mean(pi_reps):.4f} [{np.percentile(pi_reps,10):.4f}, {np.percentile(pi_reps,90):.4f}]")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "embryo_pi_celltype.csv", index=False)

    if len(df) >= 3:
        rho, p = stats.spearmanr(df["day"], df["pi_celltype"])
        _log(f"\n  Trend: ρ = {rho:+.3f}, p = {p:.4f}")
        _log(f"  Direction: {'INCREASING' if rho > 0 else 'DECREASING'}")

        pi_first = df["pi_celltype"].iloc[0]
        pi_last = df["pi_celltype"].iloc[-1]
        delta = pi_last - pi_first
        rate_per_day = delta / (df["day"].iloc[-1] - df["day"].iloc[0])
        _log(f"  Δπ = {delta:+.4f} over {df['day'].iloc[-1]-df['day'].iloc[0]:.0f} days")
        _log(f"  Rate: {rate_per_day:+.5f}/day")
        _log(f"  Human aging rate: -0.00077/year ≈ -0.0000021/day")
        if rate_per_day > 0:
            _log(f"  Asymmetry: building {abs(rate_per_day/0.0000021):.0f}× faster than erosion")

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(df) > 0:
        ax.errorbar(df["day"], df["pi_celltype"],
                     yerr=[df["pi_celltype"]-df["pi_ci_lo"], df["pi_ci_hi"]-df["pi_celltype"]],
                     fmt="o-", color="tab:green", capsize=5, markersize=8, linewidth=2)
        ax.set_xlabel("Developmental stage (embryonic day)")
        ax.set_ylabel("π_cell_type")
        ax.set_title(f"Embryogenesis: Cell type identity\nMOCA, {df['n_cells'].sum()} cells")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "embryo_pi.png", dpi=150)
    plt.close()
    _log(f"\n  Time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
