"""
Step 17: Tier-1 Validation — Is π_tissue ≈ 0.73 real or an artifact?

Three critical tests:

TEST 6.1: PERMUTATION NULL
  - Shuffle tissue labels → null distribution of π
  - Shuffle donor labels → what happens to π?
  - If null π ≈ 0.73, our finding is mathematical, not biological

TEST 6.2: ABSOLUTE VARIANCE DECOMPOSITION
  - Track V_tissue, V_donor, V_residual in absolute terms by age
  - Distinguish: structure erodes vs noise grows vs differential growth

TEST 2.1: BATCH EFFECTS
  - Include batch (SMGEBTCH) in ANOVA
  - Compare π_tissue with/without batch
  - Test batch × age confound
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
RESULTS_DIR = BASE / "results" / "step17_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP6 = ["Muscle - Skeletal", "Whole Blood", "Skin - Sun Exposed (Lower leg)",
        "Adipose - Subcutaneous", "Artery - Tibial", "Thyroid"]


def _log(msg):
    print(msg, flush=True)


def load_metadata():
    """Load and merge sample + subject metadata."""
    samples = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    samples["age_mid"] = samples["AGE"].str.split("-").apply(
        lambda x: (int(x[0]) + int(x[1])) / 2 if isinstance(x, list) and len(x) == 2 else np.nan)
    return samples


def build_balanced_design(samples, sample_ids):
    """Build column index matrix for 263 donors × 6 tissues."""
    sample_meta = samples.set_index("SAMPID")[["SUBJID", "SMTSD", "age_mid", "SMGEBTCH", "SMNABTCH", "SMRIN"]].copy()

    tissue_sample_map = {t: {} for t in TOP6}
    for i, sid in enumerate(sample_ids):
        if sid in sample_meta.index:
            row = sample_meta.loc[sid]
            t = row["SMTSD"]
            subj = row["SUBJID"]
            if t in TOP6:
                tissue_sample_map[t][subj] = i

    donor_sets = [set(tissue_sample_map[t].keys()) for t in TOP6]
    common_donors = sorted(donor_sets[0].intersection(*donor_sets[1:]))

    # Donor ages
    donor_ages = {}
    for subj in common_donors:
        for t in TOP6:
            idx = tissue_sample_map[t][subj]
            sid = sample_ids[idx]
            if sid in sample_meta.index:
                donor_ages[subj] = sample_meta.loc[sid, "age_mid"]
                break

    # Column index matrix: (n_donors, 6)
    col_idx = np.zeros((len(common_donors), len(TOP6)), dtype=int)
    for ti, t in enumerate(TOP6):
        for di, d in enumerate(common_donors):
            col_idx[di, ti] = tissue_sample_map[t][d]

    ages = np.array([donor_ages.get(d, np.nan) for d in common_donors])

    # Batch info per sample
    batch_map = {}  # (donor_idx, tissue_idx) -> batch_id
    batch_labels_unique = set()
    rin_map = {}
    for ti, t in enumerate(TOP6):
        for di, d in enumerate(common_donors):
            sid = sample_ids[tissue_sample_map[t][d]]
            if sid in sample_meta.index:
                b = sample_meta.loc[sid, "SMGEBTCH"]
                batch_map[(di, ti)] = b
                batch_labels_unique.add(b)
                rin_map[(di, ti)] = sample_meta.loc[sid, "SMRIN"]

    return common_donors, col_idx, ages, tissue_sample_map, batch_map, batch_labels_unique, rin_map, sample_meta


def anova_3way(expr_matrix, n_donors, n_tissues):
    """Three-level ANOVA for balanced design: V = V_tissue + V_donor + V_residual.
    expr_matrix: (n_donors, n_tissues)
    Returns: SS_tissue, SS_donor, SS_residual, SS_total
    """
    grand_mean = expr_matrix.mean()
    donor_means = expr_matrix.mean(axis=1)
    tissue_means = expr_matrix.mean(axis=0)
    SS_total = np.sum((expr_matrix - grand_mean) ** 2)
    SS_donor = n_tissues * np.sum((donor_means - grand_mean) ** 2)
    SS_tissue = n_donors * np.sum((tissue_means - grand_mean) ** 2)
    SS_residual = max(SS_total - SS_donor - SS_tissue, 0)
    return SS_tissue, SS_donor, SS_residual, SS_total


def stream_gene_expressions(tpm_path, col_idx, min_median_tpm=0.5, max_genes=None):
    """Stream through GTEx GCT, yield (gene_name, expr_matrix) for expressed genes."""
    n_yielded = 0
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()  # skip GCT header
        for line in f:
            parts = line.strip().split("\t", 2)
            gene_name = parts[1]
            vals = np.array(parts[2].split("\t"), dtype=np.float32)
            if np.median(vals) < min_median_tpm:
                continue
            log_vals = np.log2(vals + 1)
            expr = log_vals[col_idx]  # (n_donors, n_tissues)
            n_yielded += 1
            yield gene_name, expr
            if max_genes and n_yielded >= max_genes:
                break


# ══════════════════════════════════════════════════════════════════
# TEST 6.1: PERMUTATION NULL
# ══════════════════════════════════════════════════════════════════
def test_6_1_permutation_null(col_idx, ages, tpm_path):
    _log("\n" + "=" * 70)
    _log("TEST 6.1: PERMUTATION NULL — Is π=0.73 mathematical artifact?")
    _log("=" * 70)

    n_donors, n_tissues = col_idx.shape
    n_perm = 100
    rng = np.random.RandomState(42)

    # First pass: collect all gene expression matrices into memory
    _log("  Loading gene expressions into memory...")
    gene_names = []
    all_expr = []
    for gname, expr in stream_gene_expressions(tpm_path, col_idx):
        gene_names.append(gname)
        all_expr.append(expr)
    all_expr = np.array(all_expr)  # (n_genes, n_donors, n_tissues)
    n_genes = len(gene_names)
    _log(f"  Loaded {n_genes} genes, {n_donors} donors, {n_tissues} tissues")

    # Real π_tissue
    real_pi = np.zeros(n_genes)
    for g in range(n_genes):
        e = all_expr[g]
        ss_tissue, ss_donor, ss_resid, ss_total = anova_3way(e, n_donors, n_tissues)
        if ss_total > 1e-10:
            real_pi[g] = ss_tissue / ss_total
        else:
            real_pi[g] = np.nan
    real_median_pi = np.nanmedian(real_pi)
    _log(f"\n  REAL π_tissue (median): {real_median_pi:.4f}")

    # --- Permutation A: Shuffle tissue labels ---
    _log(f"\n  Permutation A: Shuffle tissue labels ({n_perm} permutations)...")
    null_pi_tissue_shuffle = np.zeros(n_perm)

    for perm_i in range(n_perm):
        # Shuffle columns (tissue labels) independently for each donor
        perm_expr = all_expr.copy()
        for d in range(n_donors):
            perm_order = rng.permutation(n_tissues)
            perm_expr[:, d, :] = perm_expr[:, d, perm_order]

        perm_pi = np.zeros(n_genes)
        for g in range(n_genes):
            e = perm_expr[g]
            ss_tissue, ss_donor, ss_resid, ss_total = anova_3way(e, n_donors, n_tissues)
            if ss_total > 1e-10:
                perm_pi[g] = ss_tissue / ss_total
            else:
                perm_pi[g] = np.nan
        null_pi_tissue_shuffle[perm_i] = np.nanmedian(perm_pi)

        if (perm_i + 1) % 20 == 0:
            _log(f"    Perm {perm_i+1}/{n_perm}: median π = {null_pi_tissue_shuffle[perm_i]:.4f}")

    _log(f"\n  TISSUE SHUFFLE NULL:")
    _log(f"    Null π mean:   {null_pi_tissue_shuffle.mean():.4f}")
    _log(f"    Null π std:    {null_pi_tissue_shuffle.std():.4f}")
    _log(f"    Null π range:  [{null_pi_tissue_shuffle.min():.4f}, {null_pi_tissue_shuffle.max():.4f}]")
    _log(f"    Real π:        {real_median_pi:.4f}")
    p_value_tissue = (null_pi_tissue_shuffle >= real_median_pi).sum() / n_perm
    _log(f"    p-value (null ≥ real): {p_value_tissue:.4f}")
    _log(f"    Expected by chance (1/n_tissues): {1/n_tissues:.4f}")

    if null_pi_tissue_shuffle.mean() > 0.5:
        _log(f"    >>> DANGER: Null π ≈ {null_pi_tissue_shuffle.mean():.2f} — may be mathematical artifact!")
    elif null_pi_tissue_shuffle.mean() < 0.25:
        _log(f"    >>> SAFE: Null π ≈ {null_pi_tissue_shuffle.mean():.2f} — real π={real_median_pi:.2f} is biological")
    else:
        _log(f"    >>> AMBIGUOUS: Null π ≈ {null_pi_tissue_shuffle.mean():.2f} — partially artifact?")

    # --- Permutation B: Shuffle donor labels ---
    _log(f"\n  Permutation B: Shuffle donor labels ({n_perm} permutations)...")
    null_pi_donor_shuffle = np.zeros(n_perm)

    for perm_i in range(n_perm):
        # Shuffle rows (donor labels) — keeps tissue structure intact
        perm_expr = all_expr.copy()
        perm_order = rng.permutation(n_donors)
        perm_expr = perm_expr[:, perm_order, :]

        perm_pi = np.zeros(n_genes)
        for g in range(n_genes):
            e = perm_expr[g]
            ss_tissue, ss_donor, ss_resid, ss_total = anova_3way(e, n_donors, n_tissues)
            if ss_total > 1e-10:
                perm_pi[g] = ss_tissue / ss_total
            else:
                perm_pi[g] = np.nan
        null_pi_donor_shuffle[perm_i] = np.nanmedian(perm_pi)

        if (perm_i + 1) % 20 == 0:
            _log(f"    Perm {perm_i+1}/{n_perm}: median π = {null_pi_donor_shuffle[perm_i]:.4f}")

    _log(f"\n  DONOR SHUFFLE NULL:")
    _log(f"    Null π mean:   {null_pi_donor_shuffle.mean():.4f}")
    _log(f"    Null π std:    {null_pi_donor_shuffle.std():.4f}")
    _log(f"    Real π:        {real_median_pi:.4f}")
    _log(f"    Interpretation: Shuffling donors (keeping tissues intact) should preserve π_tissue")

    # Save results
    perm_results = {
        "real_pi_tissue": real_median_pi,
        "null_tissue_shuffle_mean": null_pi_tissue_shuffle.mean(),
        "null_tissue_shuffle_std": null_pi_tissue_shuffle.std(),
        "null_tissue_shuffle_p": p_value_tissue,
        "null_donor_shuffle_mean": null_pi_donor_shuffle.mean(),
        "null_donor_shuffle_std": null_pi_donor_shuffle.std(),
        "expected_by_chance": 1/n_tissues,
    }

    return perm_results, null_pi_tissue_shuffle, null_pi_donor_shuffle, real_pi, all_expr, gene_names


# ══════════════════════════════════════════════════════════════════
# TEST 6.2: ABSOLUTE VARIANCE DECOMPOSITION
# ══════════════════════════════════════════════════════════════════
def test_6_2_absolute_variance(all_expr, gene_names, ages, n_tissues):
    _log("\n" + "=" * 70)
    _log("TEST 6.2: ABSOLUTE VARIANCE DECOMPOSITION")
    _log("=" * 70)

    n_genes, n_donors_total, _ = all_expr.shape

    age_bins = {
        "20-39": (ages >= 20) & (ages < 40),
        "40-49": (ages >= 40) & (ages < 50),
        "50-59": (ages >= 50) & (ages < 60),
        "60-79": (ages >= 60) & (ages < 80),
    }

    rows = []
    for ab, mask in age_bins.items():
        n_d = mask.sum()
        if n_d < 15:
            _log(f"  {ab}: only {n_d} donors, skipping")
            continue

        v_tissue_list = []
        v_donor_list = []
        v_resid_list = []
        v_total_list = []
        pi_tissue_list = []

        for g in range(n_genes):
            e = all_expr[g][mask]  # (n_d, n_tissues)
            ss_tissue, ss_donor, ss_resid, ss_total = anova_3way(e, n_d, n_tissues)
            N = n_d * n_tissues
            if ss_total < 1e-10:
                continue
            v_tissue_list.append(ss_tissue / N)
            v_donor_list.append(ss_donor / N)
            v_resid_list.append(ss_resid / N)
            v_total_list.append(ss_total / N)
            pi_tissue_list.append(ss_tissue / ss_total)

        row = {
            "age_bin": ab, "n_donors": n_d, "n_genes": len(v_tissue_list),
            "V_tissue_median": np.median(v_tissue_list),
            "V_donor_median": np.median(v_donor_list),
            "V_residual_median": np.median(v_resid_list),
            "V_total_median": np.median(v_total_list),
            "pi_tissue_median": np.median(pi_tissue_list),
            "V_tissue_mean": np.mean(v_tissue_list),
            "V_donor_mean": np.mean(v_donor_list),
            "V_residual_mean": np.mean(v_resid_list),
            "V_total_mean": np.mean(v_total_list),
            "pi_tissue_mean": np.mean(pi_tissue_list),
            # Store IQR for tissue
            "V_tissue_q25": np.percentile(v_tissue_list, 25),
            "V_tissue_q75": np.percentile(v_tissue_list, 75),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "test6_2_absolute_variance.csv", index=False)

    _log(f"\n  {'Age bin':<10s} {'V_tissue':>10s} {'V_donor':>10s} {'V_resid':>10s} "
         f"{'V_total':>10s} {'π_tissue':>10s} {'n_donors':>8s}")
    for _, r in df.iterrows():
        _log(f"  {r['age_bin']:<10s} {r['V_tissue_median']:>10.5f} {r['V_donor_median']:>10.5f} "
             f"{r['V_residual_median']:>10.5f} {r['V_total_median']:>10.5f} "
             f"{r['pi_tissue_median']:>10.4f} {r['n_donors']:>8.0f}")

    # Compute changes relative to youngest bin
    if len(df) >= 2:
        young = df.iloc[0]
        _log(f"\n  CHANGES RELATIVE TO {young['age_bin']} (youngest):")
        for _, r in df.iterrows():
            dv_t = r['V_tissue_median'] - young['V_tissue_median']
            dv_d = r['V_donor_median'] - young['V_donor_median']
            dv_r = r['V_residual_median'] - young['V_residual_median']
            dv_tot = r['V_total_median'] - young['V_total_median']
            _log(f"  {r['age_bin']}: ΔV_tissue={dv_t:+.5f}, ΔV_donor={dv_d:+.5f}, "
                 f"ΔV_resid={dv_r:+.5f}, ΔV_total={dv_tot:+.5f}")

        old = df.iloc[-1]
        _log(f"\n  DIAGNOSIS (comparing {young['age_bin']} → {old['age_bin']}):")
        dv_t = old['V_tissue_median'] - young['V_tissue_median']
        dv_r = old['V_residual_median'] - young['V_residual_median']
        dv_d = old['V_donor_median'] - young['V_donor_median']

        if dv_t < 0 and dv_r > 0:
            _log(f"    V_tissue DECREASES ({dv_t:+.5f}), V_residual INCREASES ({dv_r:+.5f})")
            _log(f"    >>> GENUINE STRUCTURAL LOSS: tissue identity actively erodes")
        elif dv_t > 0 and dv_r > 0 and abs(dv_r) > abs(dv_t):
            _log(f"    V_tissue INCREASES ({dv_t:+.5f}) but V_residual grows FASTER ({dv_r:+.5f})")
            _log(f"    >>> NOISE OVERWHELMS: tissue signal grows but noise grows faster")
        elif dv_t > 0 and dv_r > 0 and abs(dv_t) > abs(dv_r):
            _log(f"    V_tissue INCREASES ({dv_t:+.5f}) faster than V_residual ({dv_r:+.5f})")
            _log(f"    >>> DIFFERENTIAL GROWTH: tissue signal strengthens more than noise")
        elif abs(dv_t) < 0.0001 and dv_r > 0:
            _log(f"    V_tissue FLAT ({dv_t:+.5f}), V_residual INCREASES ({dv_r:+.5f})")
            _log(f"    >>> NOISE ADDITION: tissue signal stable but noise accumulates")
        else:
            _log(f"    V_tissue: {dv_t:+.5f}, V_donor: {dv_d:+.5f}, V_residual: {dv_r:+.5f}")
            _log(f"    >>> COMPLEX PATTERN: mixed changes")

        # Percent changes
        if young['V_tissue_median'] > 0:
            pct_t = 100 * dv_t / young['V_tissue_median']
            pct_r = 100 * dv_r / young['V_residual_median'] if young['V_residual_median'] > 0 else np.nan
            pct_d = 100 * dv_d / young['V_donor_median'] if young['V_donor_median'] > 0 else np.nan
            _log(f"\n    Percent changes: V_tissue {pct_t:+.1f}%, V_donor {pct_d:+.1f}%, V_resid {pct_r:+.1f}%")

    return df


# ══════════════════════════════════════════════════════════════════
# TEST 2.1: BATCH EFFECTS
# ══════════════════════════════════════════════════════════════════
def test_2_1_batch_effects(all_expr, gene_names, col_idx, ages, batch_map, rin_map, sample_meta, sample_ids, common_donors):
    _log("\n" + "=" * 70)
    _log("TEST 2.1: BATCH EFFECTS — Does batch confound π_tissue?")
    _log("=" * 70)

    n_genes, n_donors, n_tissues = all_expr.shape

    # Construct batch label arrays
    # For each sample (donor_i, tissue_j), get batch
    batch_str = {}
    for (di, ti), b in batch_map.items():
        batch_str[(di, ti)] = str(b) if pd.notna(b) else "UNKNOWN"

    unique_batches = sorted(set(batch_str.values()))
    batch_to_int = {b: i for i, b in enumerate(unique_batches)}
    _log(f"  Unique batches (SMGEBTCH): {len(unique_batches)}")

    # Build batch ID matrix: (n_donors, n_tissues)
    batch_id_mat = np.zeros((n_donors, n_tissues), dtype=int)
    for (di, ti), b in batch_str.items():
        batch_id_mat[di, ti] = batch_to_int[b]

    # --- Compare π_tissue WITH and WITHOUT batch ---
    _log(f"\n  Computing ANOVA with and without batch term...")

    pi_no_batch = np.zeros(n_genes)
    pi_with_batch = np.zeros(n_genes)
    ss_batch_frac = np.zeros(n_genes)
    n_valid = 0

    for g in range(n_genes):
        e = all_expr[g]  # (n_donors, n_tissues)
        N = n_donors * n_tissues

        # Standard 3-way ANOVA (no batch)
        ss_tissue, ss_donor, ss_resid, ss_total = anova_3way(e, n_donors, n_tissues)
        if ss_total < 1e-10:
            pi_no_batch[g] = np.nan
            pi_with_batch[g] = np.nan
            continue

        pi_no_batch[g] = ss_tissue / ss_total

        # Extended ANOVA: add batch term
        # For each batch, compute batch mean and SS_batch
        flat_expr = e.flatten()
        flat_batch = batch_id_mat.flatten()
        grand_mean = flat_expr.mean()

        # SS_batch
        ss_batch = 0
        for b_id in range(len(unique_batches)):
            mask_b = flat_batch == b_id
            if mask_b.sum() > 0:
                batch_mean = flat_expr[mask_b].mean()
                ss_batch += mask_b.sum() * (batch_mean - grand_mean) ** 2

        # Adjusted: SS_residual_new = SS_total - SS_tissue - SS_donor - SS_batch
        ss_resid_new = max(ss_total - ss_tissue - ss_donor - ss_batch, 0)
        pi_with_batch[g] = ss_tissue / ss_total  # π_tissue unchanged by definition
        # But what fraction does batch explain?
        ss_batch_frac[g] = ss_batch / ss_total

        n_valid += 1

    valid = ~np.isnan(pi_no_batch)
    _log(f"  Valid genes: {valid.sum()}")

    # The real question: does batch steal from tissue variance?
    # Partial out batch: π_tissue_adjusted = SS_tissue / (SS_total - SS_batch)
    pi_adjusted = np.zeros(n_genes)
    for g in range(n_genes):
        e = all_expr[g]
        flat_expr = e.flatten()
        flat_batch = batch_id_mat.flatten()

        ss_tissue, ss_donor, ss_resid, ss_total = anova_3way(e, n_donors, n_tissues)
        if ss_total < 1e-10:
            pi_adjusted[g] = np.nan
            continue

        # Batch SS
        grand_mean = flat_expr.mean()
        ss_batch = 0
        for b_id in range(len(unique_batches)):
            mask_b = flat_batch == b_id
            if mask_b.sum() > 0:
                batch_mean = flat_expr[mask_b].mean()
                ss_batch += mask_b.sum() * (batch_mean - grand_mean) ** 2

        # Adjusted π: what fraction of NON-BATCH variance is tissue?
        ss_non_batch = ss_total - ss_batch
        if ss_non_batch > 1e-10:
            pi_adjusted[g] = ss_tissue / ss_non_batch
        else:
            pi_adjusted[g] = np.nan

    valid2 = ~np.isnan(pi_adjusted)

    _log(f"\n  RESULTS:")
    _log(f"    π_tissue (no batch):     median = {np.nanmedian(pi_no_batch):.4f}")
    _log(f"    π_batch:                 median = {np.nanmedian(ss_batch_frac):.4f}")
    _log(f"    π_tissue (batch-adjusted): median = {np.nanmedian(pi_adjusted):.4f}")
    delta_pi = np.nanmedian(pi_adjusted) - np.nanmedian(pi_no_batch)
    _log(f"    Δπ_tissue (adjusted - raw): {delta_pi:+.4f}")

    if abs(delta_pi) < 0.05:
        _log(f"    >>> SAFE: Batch adjustment changes π by only {delta_pi:+.4f} (<0.05)")
    else:
        _log(f"    >>> WARNING: Batch adjustment changes π by {delta_pi:+.4f} (≥0.05)")

    # --- Batch × Age confound ---
    _log(f"\n  BATCH × AGE CONFOUND:")
    # For each donor, find dominant batch across their tissues
    donor_batches = []
    for di in range(n_donors):
        batches_this_donor = [batch_str.get((di, ti), "UNKNOWN") for ti in range(n_tissues)]
        # Use the most common batch for this donor
        from collections import Counter
        most_common = Counter(batches_this_donor).most_common(1)[0][0]
        donor_batches.append(most_common)

    # Test: is batch correlated with age?
    # Encode batch as integer
    batch_encoded = np.array([batch_to_int[b] for b in donor_batches])
    valid_ages = ~np.isnan(ages)

    # Use Kruskal-Wallis: does age differ by batch?
    # Group ages by batch
    batch_age_groups = {}
    for di in range(n_donors):
        if not valid_ages[di]:
            continue
        b = donor_batches[di]
        if b not in batch_age_groups:
            batch_age_groups[b] = []
        batch_age_groups[b].append(ages[di])

    # Only test batches with >= 3 donors
    batch_groups_for_test = [np.array(v) for v in batch_age_groups.values() if len(v) >= 3]
    if len(batch_groups_for_test) >= 2:
        kw_stat, kw_p = stats.kruskal(*batch_groups_for_test)
        _log(f"    Kruskal-Wallis (age ~ batch): H={kw_stat:.2f}, p={kw_p:.4f}")
        if kw_p < 0.05:
            _log(f"    >>> WARNING: Batch IS correlated with age (p={kw_p:.4f})")
            _log(f"    >>> This means batch could confound age-related changes")
        else:
            _log(f"    >>> SAFE: No significant batch-age correlation (p={kw_p:.4f})")
    else:
        kw_stat, kw_p = np.nan, np.nan
        _log(f"    Not enough batches with ≥3 donors for Kruskal-Wallis test")

    # Spearman correlation between batch (encoded) and age
    rho_ba, p_ba = stats.spearmanr(batch_encoded[valid_ages], ages[valid_ages])
    _log(f"    Spearman(batch_id, age): ρ={rho_ba:+.3f}, p={p_ba:.4f}")

    # --- RIN quality check ---
    _log(f"\n  RNA INTEGRITY (SMRIN) CHECK:")
    rin_vals = []
    for di in range(n_donors):
        for ti in range(n_tissues):
            r = rin_map.get((di, ti))
            if pd.notna(r):
                rin_vals.append((ages[di], float(r)))
    if rin_vals:
        rin_arr = np.array(rin_vals)
        rho_rin, p_rin = stats.spearmanr(rin_arr[:, 0], rin_arr[:, 1])
        _log(f"    Spearman(age, RIN): ρ={rho_rin:+.3f}, p={p_rin:.4f}")
        _log(f"    Mean RIN: {rin_arr[:, 1].mean():.2f} (std={rin_arr[:, 1].std():.2f})")
        if abs(rho_rin) > 0.1 and p_rin < 0.05:
            _log(f"    >>> WARNING: RIN correlates with age — potential quality confound")
        else:
            _log(f"    >>> SAFE: No significant RIN-age correlation")
    else:
        rho_rin, p_rin = np.nan, np.nan

    # Save batch results
    batch_results = {
        "pi_tissue_raw": np.nanmedian(pi_no_batch),
        "pi_batch_median": np.nanmedian(ss_batch_frac),
        "pi_tissue_adjusted": np.nanmedian(pi_adjusted),
        "delta_pi": delta_pi,
        "batch_age_kruskal_H": kw_stat,
        "batch_age_kruskal_p": kw_p,
        "batch_age_spearman_rho": rho_ba,
        "batch_age_spearman_p": p_ba,
        "rin_age_spearman_rho": rho_rin,
        "rin_age_spearman_p": p_rin,
        "n_batches": len(unique_batches),
    }

    return batch_results, pi_no_batch, pi_adjusted, ss_batch_frac


# ══════════════════════════════════════════════════════════════════
# SUMMARY FIGURE
# ══════════════════════════════════════════════════════════════════
def make_figure(perm_results, null_tissue_shuffle, null_donor_shuffle,
                df_abs_var, batch_results, pi_no_batch, pi_adjusted, ss_batch_frac):
    _log("\n" + "=" * 70)
    _log("SUMMARY FIGURE")
    _log("=" * 70)

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Permutation null distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(null_tissue_shuffle, bins=25, color="steelblue", alpha=0.7, edgecolor="white",
            label="Tissue labels shuffled")
    ax.axvline(perm_results["real_pi_tissue"], color="red", linewidth=3, linestyle="-",
               label=f"Real π = {perm_results['real_pi_tissue']:.3f}")
    ax.axvline(perm_results["expected_by_chance"], color="orange", linewidth=2, linestyle="--",
               label=f"Expected by chance = {perm_results['expected_by_chance']:.3f}")
    # Add donor shuffle as separate distribution
    ax.hist(null_donor_shuffle, bins=25, color="green", alpha=0.4, edgecolor="white",
            label=f"Donor labels shuffled (μ={null_donor_shuffle.mean():.3f})")
    ax.set_xlabel("Median π_tissue", fontsize=12)
    ax.set_ylabel("Count (permutations)", fontsize=12)
    ax.set_title("A: Permutation Null — Is π=0.73 an artifact?", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: Absolute variance by age
    ax = fig.add_subplot(gs[0, 1])
    if df_abs_var is not None and len(df_abs_var) > 0:
        age_mids = {"20-39": 30, "40-49": 45, "50-59": 55, "60-79": 70}
        x = [age_mids.get(ab, 0) for ab in df_abs_var["age_bin"]]
        ax.plot(x, df_abs_var["V_tissue_median"], "g-o", linewidth=2, markersize=8,
                label=f"V_tissue")
        ax.plot(x, df_abs_var["V_donor_median"], "b-s", linewidth=2, markersize=8,
                label=f"V_donor")
        ax.plot(x, df_abs_var["V_residual_median"], "gray", marker="^", linewidth=2, markersize=8,
                label=f"V_residual")
        ax.plot(x, df_abs_var["V_total_median"], "k--D", linewidth=1.5, markersize=6, alpha=0.5,
                label=f"V_total")
        ax.set_xlabel("Age (midpoint)", fontsize=12)
        ax.set_ylabel("Median variance (absolute)", fontsize=12)
        ax.set_title("B: Absolute Variance — Structure vs Noise", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    # Panel C: Batch effect — π with vs without batch
    ax = fig.add_subplot(gs[1, 0])
    valid = ~np.isnan(pi_no_batch) & ~np.isnan(pi_adjusted)
    if valid.sum() > 0:
        ax.scatter(pi_no_batch[valid], pi_adjusted[valid], s=1, alpha=0.1, c="gray")
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5, linewidth=1)
        rho_adj, _ = stats.spearmanr(pi_no_batch[valid], pi_adjusted[valid])
        ax.set_xlabel("π_tissue (raw)", fontsize=12)
        ax.set_ylabel("π_tissue (batch-adjusted)", fontsize=12)
        ax.set_title(f"C: Batch Effect — Raw vs Adjusted\n"
                     f"ρ={rho_adj:.3f}, Δmedian={batch_results['delta_pi']:+.4f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel D: Summary verdicts
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")

    y_pos = 0.92
    ax.text(0.5, y_pos, "TIER-1 VALIDATION VERDICTS", ha="center", fontsize=16,
            fontweight="bold", transform=ax.transAxes)

    y_pos -= 0.12
    ax.text(0.05, y_pos, "TEST 6.1: Permutation Null", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    y_pos -= 0.06
    null_mean = null_tissue_shuffle.mean()
    if null_mean < 0.25:
        verdict_6_1 = f"PASS — Null π = {null_mean:.3f} << Real π = {perm_results['real_pi_tissue']:.3f}"
        color_6_1 = "green"
    else:
        verdict_6_1 = f"FAIL — Null π = {null_mean:.3f}, artifact suspected"
        color_6_1 = "red"
    ax.text(0.08, y_pos, verdict_6_1, fontsize=11, color=color_6_1, transform=ax.transAxes)

    y_pos -= 0.12
    ax.text(0.05, y_pos, "TEST 6.2: Absolute Variance", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    y_pos -= 0.06
    if df_abs_var is not None and len(df_abs_var) >= 2:
        young_v = df_abs_var.iloc[0]
        old_v = df_abs_var.iloc[-1]
        dv_t = old_v['V_tissue_median'] - young_v['V_tissue_median']
        dv_r = old_v['V_residual_median'] - young_v['V_residual_median']
        if dv_t < 0:
            verdict_6_2 = f"Structural loss: ΔV_tissue={dv_t:+.5f}, ΔV_resid={dv_r:+.5f}"
        elif dv_r > dv_t:
            verdict_6_2 = f"Noise grows faster: ΔV_tissue={dv_t:+.5f}, ΔV_resid={dv_r:+.5f}"
        else:
            verdict_6_2 = f"Mixed: ΔV_tissue={dv_t:+.5f}, ΔV_resid={dv_r:+.5f}"
        ax.text(0.08, y_pos, verdict_6_2, fontsize=10, color="black", transform=ax.transAxes)

    y_pos -= 0.12
    ax.text(0.05, y_pos, "TEST 2.1: Batch Effects", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    y_pos -= 0.06
    delta_batch = batch_results['delta_pi']
    if abs(delta_batch) < 0.05:
        verdict_2_1 = f"PASS — Batch changes π by only {delta_batch:+.4f}"
        color_2_1 = "green"
    else:
        verdict_2_1 = f"FAIL — Batch changes π by {delta_batch:+.4f}"
        color_2_1 = "red"
    ax.text(0.08, y_pos, verdict_2_1, fontsize=11, color=color_2_1, transform=ax.transAxes)
    y_pos -= 0.06
    ax.text(0.08, y_pos,
            f"π_batch = {batch_results['pi_batch_median']:.4f}, "
            f"batch-age ρ = {batch_results['batch_age_spearman_rho']:+.3f} (p={batch_results['batch_age_spearman_p']:.3f})",
            fontsize=9, color="gray", transform=ax.transAxes)

    y_pos -= 0.12
    ax.text(0.05, y_pos, "RIN Quality:", fontsize=12, fontweight="bold", transform=ax.transAxes)
    y_pos -= 0.06
    ax.text(0.08, y_pos,
            f"RIN-age ρ = {batch_results['rin_age_spearman_rho']:+.3f} (p={batch_results['rin_age_spearman_p']:.3f})",
            fontsize=10, transform=ax.transAxes)

    fig.suptitle("Step 17: Tier-1 Validation — Is π_tissue ≈ 0.73 Real?\n"
                 "263 donors × 6 tissues × ~18K genes",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(RESULTS_DIR / "step17_validation_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"  Saved step17_validation_summary.png")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    _log("=" * 70)
    _log("STEP 17: TIER-1 VALIDATION — Is π_tissue ≈ 0.73 real?")
    _log("=" * 70)

    # Load metadata
    _log("\n[0] Loading metadata...")
    samples = load_metadata()

    tpm_path = DATA_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline()
        header = f.readline().strip().split("\t")
    sample_ids = header[2:]

    common_donors, col_idx, ages, tissue_sample_map, batch_map, batch_labels_unique, rin_map, sample_meta = \
        build_balanced_design(samples, sample_ids)

    _log(f"  Donors with all 6 tissues: {len(common_donors)}")
    _log(f"  Age range: {np.nanmin(ages):.0f} - {np.nanmax(ages):.0f}")
    _log(f"  Unique batches: {len(batch_labels_unique)}")

    # TEST 6.1: Permutation null
    perm_results, null_tissue_shuffle, null_donor_shuffle, real_pi, all_expr, gene_names = \
        test_6_1_permutation_null(col_idx, ages, tpm_path)

    # TEST 6.2: Absolute variance decomposition
    n_tissues = col_idx.shape[1]
    df_abs_var = test_6_2_absolute_variance(all_expr, gene_names, ages, n_tissues)

    # TEST 2.1: Batch effects
    batch_results, pi_no_batch, pi_adjusted, ss_batch_frac = \
        test_2_1_batch_effects(all_expr, gene_names, col_idx, ages, batch_map, rin_map,
                               sample_meta, sample_ids, common_donors)

    # Save comprehensive results
    all_results = {**perm_results, **batch_results}
    pd.DataFrame([all_results]).to_csv(RESULTS_DIR / "step17_all_results.csv", index=False)

    # Summary figure
    make_figure(perm_results, null_tissue_shuffle, null_donor_shuffle,
                df_abs_var, batch_results, pi_no_batch, pi_adjusted, ss_batch_frac)

    # Final summary
    _log("\n" + "=" * 70)
    _log("FINAL SUMMARY")
    _log("=" * 70)
    _log(f"  Real π_tissue (median across {len(gene_names)} genes): {perm_results['real_pi_tissue']:.4f}")
    _log(f"")
    _log(f"  TEST 6.1 — Permutation null:")
    _log(f"    Tissue shuffle: null π = {null_tissue_shuffle.mean():.4f} ± {null_tissue_shuffle.std():.4f}")
    _log(f"    Donor shuffle:  null π = {null_donor_shuffle.mean():.4f} ± {null_donor_shuffle.std():.4f}")
    _log(f"    Expected by chance: {1/n_tissues:.4f}")
    _log(f"")
    _log(f"  TEST 6.2 — Absolute variance:")
    if len(df_abs_var) >= 2:
        _log(f"    V_tissue: {df_abs_var.iloc[0]['V_tissue_median']:.5f} → {df_abs_var.iloc[-1]['V_tissue_median']:.5f}")
        _log(f"    V_resid:  {df_abs_var.iloc[0]['V_residual_median']:.5f} → {df_abs_var.iloc[-1]['V_residual_median']:.5f}")
    _log(f"")
    _log(f"  TEST 2.1 — Batch effects:")
    _log(f"    π_tissue raw:      {batch_results['pi_tissue_raw']:.4f}")
    _log(f"    π_tissue adjusted: {batch_results['pi_tissue_adjusted']:.4f}")
    _log(f"    Δπ:                {batch_results['delta_pi']:+.4f}")
    _log(f"    Batch-age ρ:       {batch_results['batch_age_spearman_rho']:+.3f}")
    _log(f"")
    _log(f"  Total time: {time.time()-t0:.0f}s")
    _log(f"  Results: {RESULTS_DIR}")
    _log("=" * 70)


if __name__ == "__main__":
    main()
