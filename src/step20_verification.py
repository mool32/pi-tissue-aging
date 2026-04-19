"""
Step 20: Additional Verification Analyses for π_tissue paper

V1: Macaque cross-species scaling (adds 4th point)
V2: Fix mouse dπ/dt (adult-only slope)
V3: Bootstrap CI for CR rescue
V4: TMS Droplet 10x validation (download + compute)

Addresses reviewer concerns:
- Cross-species scaling on 3 points → 4+ points
- Mouse positive dπ/dt artifact → adult-only correction
- CR rescue 86% without CI → bootstrap CI
- Smart-seq2 gene detection confound → 10x Chromium validation
"""

import time, gzip, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse
import warnings; warnings.filterwarnings("ignore")

BASE = Path("/Users/teo/Desktop/research/pi_tissue_paper")
RESULTS_DIR = BASE / "results" / "step20_verification"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE / "data"

# Also need paths from prior analyses
BASE_CA = Path("/Users/teo/Desktop/research/coupling_atlas")
BASE_OSC = Path("/Users/teo/Desktop/research/paper1/oscilatory")
GTEx_DIR = BASE_CA / "data" / "gtex"

def _log(m): print(m, flush=True)

def compute_pi_tissue_anova(expr_matrix, tissue_labels):
    """
    Compute per-gene π_tissue via one-way ANOVA.

    expr_matrix: (n_samples, n_genes) array
    tissue_labels: (n_samples,) array of tissue names

    Returns: array of π_tissue per gene
    """
    tissues = np.unique(tissue_labels)
    n_samples, n_genes = expr_matrix.shape
    pi_vals = np.full(n_genes, np.nan)

    for g in range(n_genes):
        vals = expr_matrix[:, g]
        if np.std(vals) < 1e-8:
            continue
        gm = np.mean(vals)
        ss_total = np.sum((vals - gm)**2)
        if ss_total < 1e-12:
            continue
        ss_tissue = sum(
            (tissue_labels == t).sum() * (np.mean(vals[tissue_labels == t]) - gm)**2
            for t in tissues
        )
        pi_vals[g] = ss_tissue / ss_total

    return pi_vals


# ══════════════════════════════════════════════════════════════════
# V1: MACAQUE CROSS-SPECIES SCALING
# ══════════════════════════════════════════════════════════════════
def v1_macaque_scaling():
    _log("\n" + "=" * 60)
    _log("V1: MACAQUE — Add 4th species to cross-species scaling")
    _log("=" * 60)

    # Load metadata
    info_path = DATA_DIR / "macaque" / "extracted" / "rnaData_rawCounts.whole.info.csv"
    meta = pd.read_csv(info_path, index_col=0)
    _log(f"  Metadata: {len(meta)} samples, {meta['id'].nunique()} animals")
    _log(f"  Ages: {sorted(meta['age'].unique())} years")
    _log(f"  Age groups: {meta['type'].value_counts().to_dict()}")
    _log(f"  Tissues: {meta['tissues'].nunique()}")

    # Load counts
    counts_path = DATA_DIR / "macaque" / "extracted" / "rnaData_rawCounts.whole.csv.gz"
    _log(f"  Loading counts from {counts_path}...")
    counts = pd.read_csv(counts_path, index_col=0)
    _log(f"  Counts: {counts.shape} (genes × samples)")

    # Load gene annotations
    headers_path = DATA_DIR / "macaque" / "extracted" / "headers.csv.gz"
    headers = pd.read_csv(headers_path)
    _log(f"  Gene annotations: {len(headers)} genes")

    # Filter to protein-coding genes
    protein_coding = set(headers[headers["gene_biotype"] == "protein_coding"]["gene_id"])
    gene_mask = counts.index.isin(protein_coding)
    counts = counts[gene_mask]
    _log(f"  Protein-coding: {counts.shape[0]} genes")

    # Normalize: CPM + log2
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    log_expr = np.log2(cpm + 1)

    # Filter low-expressed genes (median CPM > 0.5)
    median_cpm = cpm.median(axis=1)
    expressed = median_cpm > 0.5
    log_expr = log_expr[expressed]
    _log(f"  After expression filter: {log_expr.shape[0]} genes")

    # Select top 10 tissues by sample count
    tissue_counts = meta['tissues'].value_counts()
    top_tissues = tissue_counts.head(10).index.tolist()
    _log(f"  Top 10 tissues: {top_tissues}")

    # Find animals with samples in all top tissues
    animal_tissue_matrix = meta[meta['tissues'].isin(top_tissues)].groupby('id')['tissues'].apply(set)
    complete_animals = [a for a in animal_tissue_matrix.index
                       if len(animal_tissue_matrix[a].intersection(top_tissues)) >= 6]
    _log(f"  Animals with ≥6 of top 10 tissues: {len(complete_animals)}")

    # Use top 6 tissues with most complete animals
    tissue_animal_count = {}
    for t in top_tissues:
        n = sum(1 for a in complete_animals
                if t in meta[(meta['id'] == a) & (meta['tissues'].isin(top_tissues))]['tissues'].values)
        tissue_animal_count[t] = n

    best_tissues = sorted(tissue_animal_count, key=tissue_animal_count.get, reverse=True)[:6]
    _log(f"  Selected 6 tissues: {best_tissues}")

    # Build per-age-group π_tissue
    # Age groupings for trajectory
    age_groups = {
        'Juvenile': (3,),
        'Young_adult': (7, 8),
        'Middle_aged': (14,),
        'Elderly': (23, 25, 27),
    }
    age_midpoints = {
        'Juvenile': 3,
        'Young_adult': 7.5,
        'Middle_aged': 14,
        'Elderly': 25,
    }

    macaque_results = []

    for age_group, ages_in_group in age_groups.items():
        # Get samples for this age group in selected tissues
        group_meta = meta[
            (meta['age'].isin(ages_in_group)) &
            (meta['tissues'].isin(best_tissues))
        ]

        if len(group_meta) < 6:
            _log(f"  {age_group}: skipped (only {len(group_meta)} samples)")
            continue

        # Build expression matrix (samples × genes)
        sample_names = group_meta.index.tolist()
        valid_samples = [s for s in sample_names if s in log_expr.columns]

        if len(valid_samples) < 6:
            _log(f"  {age_group}: skipped (only {len(valid_samples)} valid samples)")
            continue

        expr_mat = log_expr[valid_samples].values.T  # samples × genes
        tissue_labels = np.array([group_meta.loc[s, 'tissues'] for s in valid_samples])

        # Compute π_tissue
        pi_vals = compute_pi_tissue_anova(expr_mat, tissue_labels)
        valid_pi = pi_vals[~np.isnan(pi_vals)]
        med_pi = np.median(valid_pi)

        _log(f"  {age_group} (mid={age_midpoints[age_group]}yr): π={med_pi:.4f} "
             f"(n_samples={len(valid_samples)}, n_tissues={len(np.unique(tissue_labels))}, "
             f"n_genes={len(valid_pi)})")

        macaque_results.append({
            'age_group': age_group,
            'age_midpoint': age_midpoints[age_group],
            'pi_tissue': med_pi,
            'n_samples': len(valid_samples),
            'n_tissues': len(np.unique(tissue_labels)),
            'n_genes': len(valid_pi),
        })

    df_mac = pd.DataFrame(macaque_results)
    df_mac.to_csv(RESULTS_DIR / "v1_macaque_pi.csv", index=False)

    # Compute dπ/dt (adult only: exclude Juvenile if developmental)
    # Use all points for slope since macaque matures by 3-4 years
    if len(df_mac) >= 2:
        ages = df_mac['age_midpoint'].values
        pis = df_mac['pi_tissue'].values
        slope, intercept, r, p, se = stats.linregress(ages, pis)
        _log(f"\n  Macaque dπ/dt = {slope:+.6f}/year (R²={r**2:.3f}, p={p:.4f})")
        _log(f"  Over observation span: Δπ = {slope * (ages[-1] - ages[0]):+.4f}")

        # Adult-only (exclude Juvenile)
        adult_mask = df_mac['age_group'] != 'Juvenile'
        if adult_mask.sum() >= 2:
            ages_a = df_mac.loc[adult_mask, 'age_midpoint'].values
            pis_a = df_mac.loc[adult_mask, 'pi_tissue'].values
            slope_a, _, r_a, p_a, _ = stats.linregress(ages_a, pis_a)
            _log(f"  Adult-only dπ/dt = {slope_a:+.6f}/year (R²={r_a**2:.3f}, p={p_a:.4f})")

    return df_mac


# ══════════════════════════════════════════════════════════════════
# V2: FIX MOUSE dπ/dt — ADULT-ONLY SLOPE
# ══════════════════════════════════════════════════════════════════
def v2_mouse_adult_slope():
    _log("\n" + "=" * 60)
    _log("V2: MOUSE — Adult-only dπ/dt (excluding developmental ages)")
    _log("=" * 60)

    # Read mouse_bulk_pi.csv from step16
    mouse_bulk_path = BASE / "results" / "step16_final" / "mouse_bulk_pi.csv"
    if not mouse_bulk_path.exists():
        mouse_bulk_path = Path("/Users/teo/Desktop/research/coupling_atlas/results/step16_final/mouse_bulk_pi.csv")

    if mouse_bulk_path.exists():
        df_mouse_bulk = pd.read_csv(mouse_bulk_path)
        _log(f"  Mouse bulk data: {len(df_mouse_bulk)} age points")
        _log(f"  {df_mouse_bulk.to_string()}")
    else:
        _log(f"  Mouse bulk data not found at expected paths, computing from TMS FACS...")
        df_mouse_bulk = None

    # Recompute from TMS FACS with careful age handling
    import scanpy as sc
    tms_path = BASE_OSC / "data/tms/tms_facs.h5ad"
    if not tms_path.exists():
        tms_path = DATA_DIR / "tms_facs" / "tms_facs.h5ad"

    if not tms_path.exists():
        # Check for any h5ad
        import glob
        h5ad_files = glob.glob(str(DATA_DIR / "tms_facs" / "*.h5ad"))
        if h5ad_files:
            tms_path = Path(h5ad_files[0])
        else:
            _log(f"  TMS FACS data not found. Skipping mouse recomputation.")
            _log(f"  Checked: {BASE_OSC / 'data/tms/tms_facs.h5ad'}")
            _log(f"  Checked: {DATA_DIR / 'tms_facs'}")
            return None

    _log(f"  Loading TMS FACS from {tms_path}...")
    adata = sc.read_h5ad(tms_path)
    _log(f"  Shape: {adata.shape}")

    # Parse ages
    adata.obs["age_months"] = pd.to_numeric(
        adata.obs["age"].astype(str).str.replace("m","").str.strip(), errors="coerce")

    available_ages = sorted(adata.obs["age_months"].dropna().unique())
    _log(f"  Available ages (months): {available_ages}")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Get top tissues
    top_tissues = adata.obs["tissue"].value_counts().head(6).index.tolist()
    _log(f"  Top 6 tissues: {top_tissues}")

    # Compute π per age
    all_results = []
    for age_m in available_ages:
        mask = (adata.obs["age_months"] == age_m) & (adata.obs["tissue"].isin(top_tissues))
        sub = adata[mask]
        if sub.shape[0] < 100:
            continue

        # Pseudobulk per tissue×sex
        pb_mat = {}
        for (tissue, sex), grp in sub.obs.groupby(["tissue", "sex"]):
            if len(grp) < 20:
                continue
            idx = np.array([sub.obs.index.get_loc(i) for i in grp.index])
            mat = sub.X[idx]
            if issparse(mat):
                mat = np.asarray(mat.todense())
            pb_mat[f"{tissue}_{sex}"] = {"tissue": tissue, "expr": mat.mean(axis=0).flatten()}

        if len(pb_mat) < 6:
            _log(f"  Age {age_m}m: skipped (only {len(pb_mat)} pseudobulk samples)")
            continue

        keys = sorted(pb_mat.keys())
        n_genes = len(next(iter(pb_mat.values()))["expr"])
        expr = np.zeros((len(keys), n_genes))
        t_labels = np.array([pb_mat[k]["tissue"] for k in keys])
        for i, k in enumerate(keys):
            expr[i] = pb_mat[k]["expr"]

        pi_vals = compute_pi_tissue_anova(expr, t_labels)
        valid_pi = pi_vals[~np.isnan(pi_vals)]
        med_pi = np.median(valid_pi)

        _log(f"  Age {age_m}m: π={med_pi:.4f} (n_samples={len(keys)}, "
             f"n_tissues={len(np.unique(t_labels))}, n_genes={len(valid_pi)})")

        all_results.append({
            'age_months': age_m,
            'age_years': age_m / 12,
            'pi_tissue': med_pi,
            'n_samples': len(keys),
            'n_tissues': len(np.unique(t_labels)),
        })

    del adata

    df_mouse = pd.DataFrame(all_results)
    df_mouse.to_csv(RESULTS_DIR / "v2_mouse_pi_all_ages.csv", index=False)

    # Full slope (all ages)
    if len(df_mouse) >= 2:
        ages = df_mouse['age_years'].values
        pis = df_mouse['pi_tissue'].values
        slope_all, _, r_all, p_all, _ = stats.linregress(ages, pis)
        _log(f"\n  All ages: dπ/dt = {slope_all:+.4f}/yr (R²={r_all**2:.3f}, p={p_all:.4f})")

    # Adult-only slope (≥ 3 months)
    adult = df_mouse[df_mouse['age_months'] >= 3]
    if len(adult) >= 2:
        ages_a = adult['age_years'].values
        pis_a = adult['pi_tissue'].values
        slope_adult, _, r_a, p_a, _ = stats.linregress(ages_a, pis_a)
        _log(f"  Adult (≥3m): dπ/dt = {slope_adult:+.4f}/yr (R²={r_a**2:.3f}, p={p_a:.4f})")

    # Aging-only slope (≥ 6 months, definitely post-development)
    aging = df_mouse[df_mouse['age_months'] >= 6]
    if len(aging) >= 2:
        ages_ag = aging['age_years'].values
        pis_ag = aging['pi_tissue'].values
        slope_aging, _, r_ag, p_ag, _ = stats.linregress(ages_ag, pis_ag)
        _log(f"  Aging (≥6m): dπ/dt = {slope_aging:+.4f}/yr (R²={r_ag**2:.3f}, p={p_ag:.4f})")

    return df_mouse


# ══════════════════════════════════════════════════════════════════
# V3: BOOTSTRAP CI FOR CR RESCUE
# ══════════════════════════════════════════════════════════════════
def v3_bootstrap_cr():
    _log("\n" + "=" * 60)
    _log("V3: BOOTSTRAP CI for CR rescue estimate")
    _log("=" * 60)

    import scanpy as sc

    # Load rat CR atlas
    rat_path = BASE_OSC / "results/h010_rat_cr/data/rat_atlas.h5ad"
    if not rat_path.exists():
        _log(f"  Rat atlas not found at {rat_path}")
        return None

    _log(f"  Loading rat CR atlas...")
    adata = sc.read_h5ad(rat_path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Pseudobulk per GSM
    pb = {}
    for gsm, grp in adata.obs.groupby("GSM"):
        if len(grp) < 30:
            continue
        idx = np.array([adata.obs.index.get_loc(i) for i in grp.index])
        mat = adata.X[idx]
        if issparse(mat):
            mat = np.asarray(mat.todense())
        pb[gsm] = {
            "condition": grp["condition"].iloc[0],
            "tissue": grp["tissue"].iloc[0],
            "expr": mat.mean(axis=0).flatten()
        }

    del adata

    n_genes = len(next(iter(pb.values()))["expr"])
    _log(f"  Pseudobulk: {len(pb)} GSMs, {n_genes} genes")

    # Build full gene-level data for bootstrap
    def compute_pi_for_condition(pb_dict, condition, gene_indices=None):
        """Compute π_tissue for a given condition, optionally on a subset of genes."""
        gsms = {k: v for k, v in pb_dict.items() if v["condition"] == condition}
        tissues = sorted(set(v["tissue"] for v in gsms.values()))
        gsm_list = sorted(gsms.keys())

        n_g = len(gene_indices) if gene_indices is not None else n_genes
        mat = np.zeros((len(gsm_list), n_g))
        t_labels = []
        for i, g in enumerate(gsm_list):
            expr = gsms[g]["expr"]
            if gene_indices is not None:
                expr = expr[gene_indices]
            mat[i] = expr
            t_labels.append(gsms[g]["tissue"])
        t_labels = np.array(t_labels)

        v_tissue_list = []
        v_resid_list = []

        for g_idx in range(n_g):
            vals = mat[:, g_idx]
            if np.std(vals) < 1e-6:
                continue
            gm = np.mean(vals)
            ss_total = np.sum((vals - gm)**2)
            if ss_total < 1e-10:
                continue
            ss_tissue = sum(
                (t_labels == t).sum() * (np.mean(vals[t_labels == t]) - gm)**2
                for t in tissues
            )
            ss_resid = max(ss_total - ss_tissue, 0)
            n = len(vals)
            v_tissue_list.append(ss_tissue / n)
            v_resid_list.append(ss_resid / n)

        pi_per_gene = [vt / (vt + vr) if (vt + vr) > 0 else 0
                       for vt, vr in zip(v_tissue_list, v_resid_list)]

        return {
            'pi_tissue': np.median(pi_per_gene),
            'V_tissue': np.median(v_tissue_list),
            'V_residual': np.median(v_resid_list),
            'n_genes_used': len(pi_per_gene),
        }

    # Point estimate
    _log(f"\n  Point estimates:")
    for cond in ["young", "old_AL", "old_CR"]:
        res = compute_pi_for_condition(pb, cond)
        _log(f"    {cond}: π = {res['pi_tissue']:.4f} (V_t={res['V_tissue']:.6f}, V_r={res['V_residual']:.6f})")

    # Bootstrap: resample genes
    n_boot = 1000
    _log(f"\n  Running {n_boot} bootstrap iterations (gene resampling)...")

    rng = np.random.RandomState(42)
    boot_rescue = []
    boot_pi = {'young': [], 'old_AL': [], 'old_CR': []}
    boot_mechanism = {'V_tissue_CR_effect': [], 'V_residual_CR_effect': []}

    for b in range(n_boot):
        if b % 100 == 0:
            _log(f"    Bootstrap {b}/{n_boot}...")

        # Resample gene indices
        gene_idx = rng.choice(n_genes, size=n_genes, replace=True)

        res_y = compute_pi_for_condition(pb, "young", gene_idx)
        res_o = compute_pi_for_condition(pb, "old_AL", gene_idx)
        res_c = compute_pi_for_condition(pb, "old_CR", gene_idx)

        boot_pi['young'].append(res_y['pi_tissue'])
        boot_pi['old_AL'].append(res_o['pi_tissue'])
        boot_pi['old_CR'].append(res_c['pi_tissue'])

        # Rescue %
        decline = res_y['pi_tissue'] - res_o['pi_tissue']
        if abs(decline) > 1e-6:
            recovery = res_c['pi_tissue'] - res_o['pi_tissue']
            rescue_pct = (recovery / decline) * 100
        else:
            rescue_pct = np.nan
        boot_rescue.append(rescue_pct)

        # Mechanism: CR effect on V_tissue vs V_residual
        boot_mechanism['V_tissue_CR_effect'].append(res_c['V_tissue'] - res_o['V_tissue'])
        boot_mechanism['V_residual_CR_effect'].append(res_c['V_residual'] - res_o['V_residual'])

    boot_rescue = np.array(boot_rescue)
    boot_rescue = boot_rescue[~np.isnan(boot_rescue)]

    # Results
    _log(f"\n  BOOTSTRAP RESULTS ({n_boot} iterations, gene resampling):")
    _log(f"  ─────────────────────────────────────────")

    for cond in ['young', 'old_AL', 'old_CR']:
        vals = np.array(boot_pi[cond])
        _log(f"  π_{cond}: {np.median(vals):.4f} [{np.percentile(vals, 2.5):.4f}, {np.percentile(vals, 97.5):.4f}]")

    _log(f"\n  CR rescue: {np.median(boot_rescue):.1f}% "
         f"[{np.percentile(boot_rescue, 2.5):.1f}%, {np.percentile(boot_rescue, 97.5):.1f}%]")

    vt_cr = np.array(boot_mechanism['V_tissue_CR_effect'])
    vr_cr = np.array(boot_mechanism['V_residual_CR_effect'])

    _log(f"\n  CR mechanism (V_tissue change): {np.median(vt_cr):+.6f} "
         f"[{np.percentile(vt_cr, 2.5):+.6f}, {np.percentile(vt_cr, 97.5):+.6f}]")
    _log(f"  CR mechanism (V_residual change): {np.median(vr_cr):+.6f} "
         f"[{np.percentile(vr_cr, 2.5):+.6f}, {np.percentile(vr_cr, 97.5):+.6f}]")

    # Noise reduction fraction
    noise_frac = np.mean(vr_cr < 0)
    _log(f"  V_residual decreases in {noise_frac*100:.1f}% of bootstraps (noise reduction)")
    struct_frac = np.mean(vt_cr > 0)
    _log(f"  V_tissue increases in {struct_frac*100:.1f}% of bootstraps (structure reinforcement)")

    # Save
    boot_df = pd.DataFrame({
        'pi_young': boot_pi['young'],
        'pi_old_AL': boot_pi['old_AL'],
        'pi_old_CR': boot_pi['old_CR'],
        'rescue_pct': list(boot_rescue) + [np.nan] * (n_boot - len(boot_rescue)),
        'V_tissue_CR_effect': boot_mechanism['V_tissue_CR_effect'],
        'V_residual_CR_effect': boot_mechanism['V_residual_CR_effect'],
    })
    boot_df.to_csv(RESULTS_DIR / "v3_bootstrap_cr.csv", index=False)

    # Summary table
    summary = pd.DataFrame([{
        'metric': 'CR_rescue_pct',
        'median': np.median(boot_rescue),
        'ci_2.5': np.percentile(boot_rescue, 2.5),
        'ci_97.5': np.percentile(boot_rescue, 97.5),
        'n_boot': n_boot,
    }, {
        'metric': 'V_residual_CR_effect',
        'median': np.median(vr_cr),
        'ci_2.5': np.percentile(vr_cr, 2.5),
        'ci_97.5': np.percentile(vr_cr, 97.5),
        'n_boot': n_boot,
    }, {
        'metric': 'V_tissue_CR_effect',
        'median': np.median(vt_cr),
        'ci_2.5': np.percentile(vt_cr, 2.5),
        'ci_97.5': np.percentile(vt_cr, 97.5),
        'n_boot': n_boot,
    }])
    summary.to_csv(RESULTS_DIR / "v3_bootstrap_summary.csv", index=False)

    return summary


# ══════════════════════════════════════════════════════════════════
# V4: TMS DROPLET 10x VALIDATION
# ══════════════════════════════════════════════════════════════════
def v4_tms_droplet():
    _log("\n" + "=" * 60)
    _log("V4: TMS DROPLET (10x Chromium) — SC validation")
    _log("=" * 60)

    import scanpy as sc

    # Check if data exists locally
    droplet_paths = [
        DATA_DIR / "tms_droplet" / "tms_droplet.h5ad",
        DATA_DIR / "tms_droplet" / "tabula-muris-senis-droplet-processed-official-annotations.h5ad",
        BASE_OSC / "data/tms/tms_droplet.h5ad",
    ]

    droplet_path = None
    for p in droplet_paths:
        if p.exists():
            droplet_path = p
            break

    if droplet_path is None:
        # Download from figshare
        _log(f"  TMS Droplet data not found locally. Downloading from figshare...")
        dl_dir = DATA_DIR / "tms_droplet"
        dl_dir.mkdir(parents=True, exist_ok=True)
        droplet_path = dl_dir / "tms_droplet.h5ad"

        url = "https://figshare.com/ndownloader/files/24351086"
        _log(f"  Downloading from {url} to {droplet_path}...")
        _log(f"  (This is ~2GB, may take a while)")

        import urllib.request
        urllib.request.urlretrieve(url, droplet_path)
        _log(f"  Download complete.")

    _log(f"  Loading TMS Droplet from {droplet_path}...")
    adata = sc.read_h5ad(droplet_path)
    _log(f"  Shape: {adata.shape}")
    _log(f"  Columns: {adata.obs.columns.tolist()[:20]}")

    # Parse ages
    age_col = None
    for c in ['age', 'age_months', 'mouse.age']:
        if c in adata.obs.columns:
            age_col = c
            break

    if age_col is None:
        _log(f"  WARNING: No age column found! Available: {adata.obs.columns.tolist()}")
        return None

    _log(f"  Age column: {age_col}")
    _log(f"  Age values: {adata.obs[age_col].value_counts().to_dict()}")

    adata.obs["age_months"] = pd.to_numeric(
        adata.obs[age_col].astype(str).str.replace("m","").str.strip(), errors="coerce")

    # Check tissue column
    tissue_col = None
    for c in ['tissue', 'organ', 'tissue_type']:
        if c in adata.obs.columns:
            tissue_col = c
            break
    _log(f"  Tissue column: {tissue_col}")
    _log(f"  Tissues: {adata.obs[tissue_col].nunique()}")

    # Cell type column
    ct_col = None
    for c in ['cell_ontology_class', 'cell_type', 'celltype', 'free_annotation']:
        if c in adata.obs.columns:
            ct_col = c
            break
    _log(f"  Cell type column: {ct_col}")

    # QC: gene detection per cell by age
    _log(f"\n  Gene detection QC by age:")
    for age in sorted(adata.obs["age_months"].dropna().unique()):
        sub = adata[adata.obs["age_months"] == age]
        mat = sub.X
        if issparse(mat):
            n_genes_per_cell = np.array((mat > 0).sum(axis=1)).flatten()
        else:
            n_genes_per_cell = np.sum(mat > 0, axis=1)
        _log(f"    {int(age)}m: {sub.shape[0]} cells, median genes/cell = {np.median(n_genes_per_cell):.0f}")

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Find cell types in ≥4 tissues (matching FACS approach)
    ct_tissue = adata.obs.groupby(ct_col)[tissue_col].apply(lambda x: len(x.unique()))
    multi_tissue_cts = ct_tissue[ct_tissue >= 4].index.tolist()
    _log(f"\n  Cell types in ≥4 tissues: {len(multi_tissue_cts)}")

    # Select top 7 by cell count
    ct_counts = adata.obs[adata.obs[ct_col].isin(multi_tissue_cts)][ct_col].value_counts()
    selected_cts = ct_counts.head(7).index.tolist()
    _log(f"  Selected cell types: {selected_cts}")

    # For each cell type: compute π_tissue young vs old
    young_ages = [1, 3]  # months
    old_ages = [18, 21, 24, 30]  # months

    results = []

    for ct in selected_cts:
        ct_mask = adata.obs[ct_col] == ct

        for age_label, ages in [("young", young_ages), ("old", old_ages)]:
            age_mask = adata.obs["age_months"].isin(ages)
            sub = adata[ct_mask & age_mask]

            if sub.shape[0] < 50:
                continue

            # Pseudobulk per tissue × mouse (or tissue if no mouse ID)
            mouse_col = None
            for c in ['mouse.id', 'mouse_id', 'individual', 'channel']:
                if c in sub.obs.columns:
                    mouse_col = c
                    break

            if mouse_col:
                group_cols = [tissue_col, mouse_col]
            else:
                # Use tissue × sex × age as proxy
                sex_col = 'sex' if 'sex' in sub.obs.columns else None
                if sex_col:
                    group_cols = [tissue_col, sex_col, age_col]
                else:
                    group_cols = [tissue_col, age_col]

            pb_mat = {}
            for group_key, grp in sub.obs.groupby(group_cols):
                if len(grp) < 10:
                    continue
                idx = np.array([sub.obs.index.get_loc(i) for i in grp.index])
                mat = sub.X[idx]
                if issparse(mat):
                    mat = np.asarray(mat.todense())
                tissue = grp[tissue_col].iloc[0]
                key = str(group_key)
                pb_mat[key] = {"tissue": tissue, "expr": mat.mean(axis=0).flatten()}

            if len(pb_mat) < 4:
                continue

            keys = sorted(pb_mat.keys())
            n_g = len(next(iter(pb_mat.values()))["expr"])
            expr = np.zeros((len(keys), n_g))
            t_labels = np.array([pb_mat[k]["tissue"] for k in keys])
            for i, k in enumerate(keys):
                expr[i] = pb_mat[k]["expr"]

            pi_vals = compute_pi_tissue_anova(expr, t_labels)
            valid_pi = pi_vals[~np.isnan(pi_vals)]

            if len(valid_pi) < 100:
                continue

            med_pi = np.median(valid_pi)

            results.append({
                'cell_type': ct,
                'age_group': age_label,
                'pi_tissue': med_pi,
                'n_samples': len(keys),
                'n_tissues': len(np.unique(t_labels)),
                'n_genes': len(valid_pi),
                'n_cells': sub.shape[0],
            })

            _log(f"  {ct} ({age_label}): π={med_pi:.4f} "
                 f"(n_samples={len(keys)}, n_tissues={len(np.unique(t_labels))})")

    del adata

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / "v4_tms_droplet_pi.csv", index=False)

    # Compare young vs old per cell type
    if len(df_results) > 0:
        _log(f"\n  COMPARISON: Young vs Old per cell type (10x Chromium)")
        _log(f"  {'Cell type':<25s} {'π_young':>8s} {'π_old':>8s} {'Δπ':>8s}")
        _log(f"  {'─'*55}")

        deltas = []
        for ct in selected_cts:
            y = df_results[(df_results['cell_type'] == ct) & (df_results['age_group'] == 'young')]
            o = df_results[(df_results['cell_type'] == ct) & (df_results['age_group'] == 'old')]
            if len(y) > 0 and len(o) > 0:
                delta = o['pi_tissue'].iloc[0] - y['pi_tissue'].iloc[0]
                deltas.append(delta)
                _log(f"  {ct:<25s} {y['pi_tissue'].iloc[0]:>8.4f} {o['pi_tissue'].iloc[0]:>8.4f} {delta:>+8.4f}")

        if deltas:
            _log(f"\n  Mean Δπ across cell types: {np.mean(deltas):+.4f}")
            _log(f"  Median Δπ: {np.median(deltas):+.4f}")
            n_pos = sum(1 for d in deltas if d > 0)
            n_neg = sum(1 for d in deltas if d < 0)
            _log(f"  Direction: {n_pos} increase, {n_neg} decrease")

            # Compare with FACS result
            _log(f"\n  COMPARISON WITH FACS (Smart-seq2):")
            facs_path = BASE / "results" / "step39_sc_pi" / "sc_pi_tissue.csv"
            if facs_path.exists():
                facs = pd.read_csv(facs_path)
                _log(f"  FACS results:")
                for _, r in facs.iterrows():
                    _log(f"    {r['cell_type']}: {r['age']} π={r['pi']:.4f}")

    return df_results


# ══════════════════════════════════════════════════════════════════
# COMPILE: Updated cross-species scaling law
# ══════════════════════════════════════════════════════════════════
def compile_scaling_law(df_macaque, df_mouse):
    _log("\n" + "=" * 60)
    _log("COMPILE: Updated cross-species scaling law (4+ species)")
    _log("=" * 60)

    species_data = []

    # Human (GTEx) — from paper
    species_data.append({
        'species': 'Human',
        'lifespan_yr': 80,
        'pi_young': 0.764,
        'pi_old': 0.733,
        'observation_span_yr': 40,
    })

    # Rat (Calico) — from paper
    species_data.append({
        'species': 'Rat',
        'lifespan_yr': 3.0,
        'pi_young': 0.893,
        'pi_old': 0.842,
        'observation_span_yr': 22/12,
    })

    # Mouse — corrected
    if df_mouse is not None and len(df_mouse) >= 2:
        # Use adult-only (≥3 months) for aging slope
        adult = df_mouse[df_mouse['age_months'] >= 3]
        if len(adult) >= 2:
            pi_youngest = adult.iloc[0]['pi_tissue']
            pi_oldest = adult.iloc[-1]['pi_tissue']
            span = (adult.iloc[-1]['age_months'] - adult.iloc[0]['age_months']) / 12
            species_data.append({
                'species': 'Mouse',
                'lifespan_yr': 2.5,
                'pi_young': pi_youngest,
                'pi_old': pi_oldest,
                'observation_span_yr': span,
            })

    # Macaque — new
    if df_macaque is not None and len(df_macaque) >= 2:
        # Use adult ages only (exclude Juvenile=3yr which is pre-maturation)
        adult_mac = df_macaque[df_macaque['age_group'] != 'Juvenile']
        if len(adult_mac) >= 2:
            pi_y_mac = adult_mac.iloc[0]['pi_tissue']
            pi_o_mac = adult_mac.iloc[-1]['pi_tissue']
            span_mac = adult_mac.iloc[-1]['age_midpoint'] - adult_mac.iloc[0]['age_midpoint']
        else:
            pi_y_mac = df_macaque.iloc[0]['pi_tissue']
            pi_o_mac = df_macaque.iloc[-1]['pi_tissue']
            span_mac = df_macaque.iloc[-1]['age_midpoint'] - df_macaque.iloc[0]['age_midpoint']

        species_data.append({
            'species': 'Macaque',
            'lifespan_yr': 40,
            'pi_young': pi_y_mac,
            'pi_old': pi_o_mac,
            'observation_span_yr': span_mac,
        })

    df = pd.DataFrame(species_data)
    df['delta_pi'] = df['pi_old'] - df['pi_young']
    df['dpi_dt'] = df['delta_pi'] / df['observation_span_yr']
    df['abs_dpi_dt'] = df['dpi_dt'].abs()
    df['total_erosion'] = df['abs_dpi_dt'] * df['lifespan_yr']
    df['inv_lifespan'] = 1 / df['lifespan_yr']

    _log(f"\n  Updated species table:")
    _log(f"  {'Species':<10s} {'L(yr)':>6s} {'π_young':>8s} {'π_old':>8s} {'Δπ':>8s} {'dπ/dt':>10s} {'k':>8s}")
    _log(f"  {'─'*62}")
    for _, r in df.iterrows():
        _log(f"  {r['species']:<10s} {r['lifespan_yr']:>6.1f} {r['pi_young']:>8.4f} {r['pi_old']:>8.4f} "
             f"{r['delta_pi']:>+8.4f} {r['dpi_dt']:>+10.6f} {r['total_erosion']:>8.4f}")

    # Only use species with negative dπ/dt for scaling law
    negative = df[df['dpi_dt'] < 0]
    _log(f"\n  Species with negative dπ/dt: {len(negative)}")

    if len(negative) >= 3:
        # Log-log fit: log|dπ/dt| = α × log(L) + β
        log_L = np.log10(negative['lifespan_yr'].values)
        log_rate = np.log10(negative['abs_dpi_dt'].values)
        slope, intercept, r, p, se = stats.linregress(log_L, log_rate)

        _log(f"\n  POWER LAW FIT (log-log):")
        _log(f"  α = {slope:.3f} ± {se:.3f}")
        _log(f"  R² = {r**2:.3f}")
        _log(f"  p = {p:.4f}")
        _log(f"  Expected: α ≈ -1 (inverse proportionality)")

        # Spearman correlation
        rho, p_sp = stats.spearmanr(negative['inv_lifespan'], negative['abs_dpi_dt'])
        _log(f"  Spearman(1/L, |dπ/dt|): ρ = {rho:.3f}, p = {p_sp:.4f}")

    # Save
    df.to_csv(RESULTS_DIR / "v_scaling_law_updated.csv", index=False)

    return df


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    _log("=" * 60)
    _log("STEP 20: VERIFICATION ANALYSES")
    _log("=" * 60)

    # V1: Macaque
    df_macaque = v1_macaque_scaling()

    # V2: Mouse fix
    df_mouse = v2_mouse_adult_slope()

    # V3: Bootstrap CR
    v3_bootstrap_cr()

    # Compile scaling law with all species
    compile_scaling_law(df_macaque, df_mouse)

    # V4: 10x validation (runs last — may need download)
    v4_tms_droplet()

    _log(f"\n  Total time: {time.time()-t0:.0f}s")
    _log(f"  All results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
