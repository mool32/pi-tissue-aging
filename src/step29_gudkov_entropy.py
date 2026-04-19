#!/usr/bin/env python3
"""
Step 29: Exact Paper 1 per-gene entropy replication on Gudkov bone marrow.

Uses IDENTICAL method to h020_compute_gene_entropy.py:
  p_g^i = x_g^i / Σ_j(x_j^i)
  h_g^i = -p_g^i * log2(p_g^i)
  H̄_g = mean_i(h_g^i) across cells in group
  ΔH_g = H̄_g(old) - H̄_g(young)

Test 1: ΔH_g(Nfkbia) in Gudkov young_untreated vs old_untreated, per cluster
Test 2: ΔH_g for all NF-κB targets + upstream
Test 3: SEQ (IκBα/Nfkbia) vs ENZ (kinases/phosphatases) divergence
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

GUDKOV = Path('/Users/teo/Desktop/research/RQ022491-Gudkov/gudkov_merged.h5ad')
OUTDIR = Path('/Users/teo/Desktop/research/coupling_atlas/results/step29_gudkov_entropy')
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── NF-κB pathway genes (mouse) ──
NFKB_TARGETS = [
    'Nfkbia', 'Nfkbib', 'Nfkbie', 'Tnfaip3', 'Bcl2l1', 'Ccl2', 'Cxcl10',
    'Il6', 'Tnf', 'Icam1', 'Birc3', 'Ptgs2', 'Mmp9', 'Ccl5', 'Il1b',
    'Cxcl1', 'Socs3',
]

NFKB_UPSTREAM = [
    'Rela', 'Nfkb1', 'Nfkb2', 'Ikbkb', 'Ikbkg', 'Chuk',
    'Traf6', 'Myd88', 'Irak1', 'Irak4', 'Tlr4', 'Tlr5',
]

# SEQ (sequestration) vs ENZ (enzymatic) — Paper 1 core distinction
SEQ_GENES = ['Nfkbia', 'Nfkbib', 'Nfkbie']  # IκB family — sequestration inhibitors
ENZ_GENES = ['Tnfaip3', 'Ptgs2', 'Mmp9', 'Socs3']  # enzymatic/catalytic effectors


def gene_entropy_contrib(X_mat):
    """Per-gene entropy contribution: -p_g * log2(p_g), averaged across cells.
    EXACT copy of h020_compute_gene_entropy.py::_gene_entropy_contrib()
    """
    if hasattr(X_mat, 'toarray'):
        X_mat = X_mat.toarray()
    X_mat = X_mat.astype(np.float64)

    row_sums = X_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = X_mat / row_sums

    with np.errstate(divide='ignore', invalid='ignore'):
        H_g = -P * np.log2(np.where(P > 0, P, 1))
    H_g = np.nan_to_num(H_g, 0.0)

    return H_g.mean(axis=0)  # mean across cells → per-gene entropy contribution


def gene_entropy_per_cell(X_mat):
    """Per-gene entropy contribution per cell (not averaged).
    Returns (n_cells, n_genes) matrix.
    """
    if hasattr(X_mat, 'toarray'):
        X_mat = X_mat.toarray()
    X_mat = X_mat.astype(np.float64)

    row_sums = X_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = X_mat / row_sums

    with np.errstate(divide='ignore', invalid='ignore'):
        H_g = -P * np.log2(np.where(P > 0, P, 1))
    H_g = np.nan_to_num(H_g, 0.0)

    return H_g


# ── Load data ──
print("Loading Gudkov data...")
adata = sc.read_h5ad(GUDKOV)
raw_genes = list(adata.raw.var_names)
print(f"  {adata.shape[0]} cells, {len(raw_genes)} raw genes")

# Check which pathway genes exist
found_targets = [g for g in NFKB_TARGETS if g in raw_genes]
found_upstream = [g for g in NFKB_UPSTREAM if g in raw_genes]
print(f"  NF-κB targets found: {len(found_targets)}/{len(NFKB_TARGETS)}")
print(f"  NF-κB upstream found: {len(found_upstream)}/{len(NFKB_UPSTREAM)}")

# ── Test 1: ΔH_g(Nfkbia) per cluster ──
print("\n" + "=" * 70)
print("TEST 1: Per-gene entropy ΔH_g(Nfkbia) per cluster")
print("=" * 70)

conditions_compare = [
    ('young_untreated', 'old_untreated', 'aging'),
    ('young_untreated', 'young_GP532_6h', 'GP532_6h_young'),
    ('young_untreated', 'young_GP532_24h', 'GP532_24h_young'),
    ('old_untreated', 'old_GP532_6h', 'GP532_6h_old'),
    ('old_untreated', 'old_GP532_24h', 'GP532_24h_old'),
]

nfkbia_idx = raw_genes.index('Nfkbia')
all_pathway = found_targets + found_upstream
pathway_idxs = [raw_genes.index(g) for g in all_pathway]

results_per_cluster = []

for cl in sorted(adata.obs['leiden'].unique(), key=int):
    cl_mask = adata.obs['leiden'].values == cl
    n_cl = cl_mask.sum()

    for ref_cond, alt_cond, label in conditions_compare:
        ref_mask = cl_mask & (adata.obs['condition'].values == ref_cond)
        alt_mask = cl_mask & (adata.obs['condition'].values == alt_cond)

        n_ref = ref_mask.sum()
        n_alt = alt_mask.sum()

        if n_ref < 20 or n_alt < 20:
            continue

        # Extract raw counts for all genes in this cluster
        X_ref = adata.raw.X[ref_mask]
        X_alt = adata.raw.X[alt_mask]

        # Compute per-gene entropy (mean across cells)
        H_ref = gene_entropy_contrib(X_ref)
        H_alt = gene_entropy_contrib(X_alt)

        # ΔH for Nfkbia
        delta_h_nfkbia = H_alt[nfkbia_idx] - H_ref[nfkbia_idx]

        # ΔH for all pathway genes
        row = {
            'cluster': int(cl), 'n_total': n_cl,
            'comparison': label,
            'ref_cond': ref_cond, 'alt_cond': alt_cond,
            'n_ref': n_ref, 'n_alt': n_alt,
            'H_nfkbia_ref': H_ref[nfkbia_idx],
            'H_nfkbia_alt': H_alt[nfkbia_idx],
            'delta_H_nfkbia': delta_h_nfkbia,
        }

        # All pathway genes
        for g, idx in zip(all_pathway, pathway_idxs):
            row[f'delta_H_{g}'] = H_alt[idx] - H_ref[idx]
            row[f'H_{g}_ref'] = H_ref[idx]
            row[f'H_{g}_alt'] = H_alt[idx]

        # Genome-wide ΔH stats for context
        delta_H_all = H_alt - H_ref
        row['delta_H_genome_mean'] = np.mean(delta_H_all)
        row['delta_H_genome_std'] = np.std(delta_H_all)
        row['delta_H_genome_median'] = np.median(delta_H_all)
        row['nfkbia_z_score'] = ((delta_h_nfkbia - np.mean(delta_H_all))
                                  / np.std(delta_H_all)) if np.std(delta_H_all) > 0 else 0

        # Percentile of Nfkbia ΔH among all genes
        row['nfkbia_percentile'] = np.mean(delta_H_all < delta_h_nfkbia) * 100

        results_per_cluster.append(row)

    if int(cl) % 5 == 0:
        print(f"  Processed cluster {cl}...")

df = pd.DataFrame(results_per_cluster)

# ── Print aging results ──
aging = df[df['comparison'] == 'aging'].copy()
print(f"\nΔH_g(Nfkbia) = H(old) - H(young) per cluster:")
print("-" * 90)
print(f"{'Cluster':>8} {'N_young':>8} {'N_old':>8} {'H_young':>10} {'H_old':>10} "
      f"{'ΔH_g':>10} {'Z-score':>8} {'%ile':>6} {'Genome_ΔH':>10}")
print("-" * 90)

for _, r in aging.sort_values('delta_H_nfkbia', ascending=False).iterrows():
    print(f"{r['cluster']:>8d} {r['n_ref']:>8d} {r['n_alt']:>8d} "
          f"{r['H_nfkbia_ref']:>10.6f} {r['H_nfkbia_alt']:>10.6f} "
          f"{r['delta_H_nfkbia']:>10.6f} {r['nfkbia_z_score']:>8.2f} "
          f"{r['nfkbia_percentile']:>6.1f} {r['delta_H_genome_mean']:>10.6f}")

n_positive = (aging['delta_H_nfkbia'] > 0).sum()
n_total = len(aging)
mean_delta = aging['delta_H_nfkbia'].mean()
median_delta = aging['delta_H_nfkbia'].median()
mean_z = aging['nfkbia_z_score'].mean()

print(f"\nΔH_g(Nfkbia) > 0: {n_positive}/{n_total} clusters ({100*n_positive/n_total:.0f}%)")
print(f"Mean ΔH_g(Nfkbia): {mean_delta:.6f}")
print(f"Median ΔH_g(Nfkbia): {median_delta:.6f}")
print(f"Mean Z-score: {mean_z:.2f}")

# Sign test
sign_p = stats.binomtest(n_positive, n_total, 0.5).pvalue
print(f"Sign test p = {sign_p:.4f}")

# Wilcoxon on ΔH values (H0: median = 0)
if n_total >= 5:
    w_stat, w_p = stats.wilcoxon(aging['delta_H_nfkbia'])
    print(f"Wilcoxon signed-rank (H0: ΔH=0): p = {w_p:.4f}")

# ── Test 2: All pathway genes ΔH ──
print("\n" + "=" * 70)
print("TEST 2: ΔH_g for all NF-κB pathway genes (aging, averaged across clusters)")
print("=" * 70)

pathway_results = []
for g in all_pathway:
    col = f'delta_H_{g}'
    if col in aging.columns:
        vals = aging[col].values
        n_pos = (vals > 0).sum()
        category = 'TARGET' if g in NFKB_TARGETS else 'UPSTREAM'
        pathway_results.append({
            'gene': g, 'category': category,
            'mean_delta_H': np.mean(vals),
            'median_delta_H': np.median(vals),
            'n_positive': n_pos, 'n_total': len(vals),
            'frac_positive': n_pos / len(vals),
        })

df_pathway = pd.DataFrame(pathway_results).sort_values('mean_delta_H', ascending=False)

print(f"\n{'Gene':<12} {'Category':<10} {'Mean_ΔH':>12} {'Med_ΔH':>12} {'N_pos':>6} {'Frac+':>6}")
print("-" * 60)
for _, r in df_pathway.iterrows():
    marker = ' ***' if r['gene'] == 'Nfkbia' else ''
    print(f"{r['gene']:<12} {r['category']:<10} {r['mean_delta_H']:>12.6f} "
          f"{r['median_delta_H']:>12.6f} {r['n_positive']:>6.0f} {r['frac_positive']:>6.2f}{marker}")

# Target vs upstream comparison
target_deltas = df_pathway[df_pathway['category'] == 'TARGET']['mean_delta_H'].values
upstream_deltas = df_pathway[df_pathway['category'] == 'UPSTREAM']['mean_delta_H'].values
print(f"\nTargets mean ΔH: {np.mean(target_deltas):.6f}")
print(f"Upstream mean ΔH: {np.mean(upstream_deltas):.6f}")
if len(target_deltas) >= 3 and len(upstream_deltas) >= 3:
    u_stat, u_p = stats.mannwhitneyu(target_deltas, upstream_deltas, alternative='greater')
    print(f"Targets > Upstream: Mann-Whitney p = {u_p:.4f}")

# Nfkbia rank among targets
nfkbia_rank = list(df_pathway[df_pathway['category'] == 'TARGET']
                   .sort_values('mean_delta_H', ascending=False)['gene']).index('Nfkbia') + 1
print(f"Nfkbia rank among targets: #{nfkbia_rank}/{len(target_deltas)}")

# ── Test 3: SEQ vs ENZ divergence ──
print("\n" + "=" * 70)
print("TEST 3: SEQ (IκB family) vs ENZ (enzymatic) divergence")
print("=" * 70)

seq_found = [g for g in SEQ_GENES if f'delta_H_{g}' in aging.columns]
enz_found = [g for g in ENZ_GENES if f'delta_H_{g}' in aging.columns]

# Per-cluster SEQ vs ENZ
seq_deltas_per_cluster = []
enz_deltas_per_cluster = []

for _, r in aging.iterrows():
    seq_vals = [r[f'delta_H_{g}'] for g in seq_found]
    enz_vals = [r[f'delta_H_{g}'] for g in enz_found]
    seq_deltas_per_cluster.append(np.mean(seq_vals))
    enz_deltas_per_cluster.append(np.mean(enz_vals))

seq_arr = np.array(seq_deltas_per_cluster)
enz_arr = np.array(enz_deltas_per_cluster)
divergence = seq_arr - enz_arr  # positive = SEQ increases more

print(f"SEQ genes: {seq_found}")
print(f"ENZ genes: {enz_found}")
print(f"\nMean ΔH(SEQ): {np.mean(seq_arr):.6f}")
print(f"Mean ΔH(ENZ): {np.mean(enz_arr):.6f}")
print(f"SEQ - ENZ divergence: {np.mean(divergence):.6f}")
print(f"SEQ > ENZ in {(divergence > 0).sum()}/{len(divergence)} clusters")

if len(divergence) >= 5:
    w_stat, w_p = stats.wilcoxon(divergence)
    print(f"Wilcoxon (H0: SEQ=ENZ): p = {w_p:.4f}")

# ── GP532 entropy effect ──
print("\n" + "=" * 70)
print("GP532 EFFECT on Nfkbia entropy")
print("=" * 70)

for label in ['GP532_6h_young', 'GP532_24h_young', 'GP532_6h_old', 'GP532_24h_old']:
    subset = df[df['comparison'] == label]
    if len(subset) == 0:
        continue
    n_pos = (subset['delta_H_nfkbia'] > 0).sum()
    mean_d = subset['delta_H_nfkbia'].mean()
    print(f"  {label:<20}: ΔH(Nfkbia) = {mean_d:+.6f}, positive in {n_pos}/{len(subset)} clusters")

# ── FIGURE ──
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Panel A: ΔH(Nfkbia) per cluster (aging)
ax = axes[0, 0]
aging_sorted = aging.sort_values('delta_H_nfkbia', ascending=False)
colors = ['#d62728' if d > 0 else '#2ca02c' for d in aging_sorted['delta_H_nfkbia']]
ax.bar(range(len(aging_sorted)), aging_sorted['delta_H_nfkbia'].values * 1e3, color=colors, alpha=0.7)
ax.axhline(0, color='black', ls='-', lw=0.5)
ax.set_xlabel('Cluster (sorted)')
ax.set_ylabel('ΔH_g(Nfkbia) × 10³')
ax.set_title(f'Nfkbia entropy: old − young\n{n_positive}/{n_total} clusters positive')
ax.set_xticks(range(len(aging_sorted)))
ax.set_xticklabels(aging_sorted['cluster'].astype(int).values, fontsize=6, rotation=45)

# Panel B: Nfkbia Z-score per cluster
ax = axes[0, 1]
aging_z = aging.sort_values('nfkbia_z_score', ascending=False)
colors_z = ['#d62728' if z > 0 else '#2ca02c' for z in aging_z['nfkbia_z_score']]
ax.bar(range(len(aging_z)), aging_z['nfkbia_z_score'].values, color=colors_z, alpha=0.7)
ax.axhline(0, color='black', ls='-', lw=0.5)
ax.set_xlabel('Cluster (sorted)')
ax.set_ylabel('Z-score (Nfkbia vs genome)')
ax.set_title(f'Nfkbia ΔH_g Z-score\nMean Z = {mean_z:.2f}')

# Panel C: All pathway genes ranked by ΔH
ax = axes[0, 2]
df_pw_sorted = df_pathway.sort_values('mean_delta_H', ascending=True)
colors_pw = []
for _, r in df_pw_sorted.iterrows():
    if r['gene'] == 'Nfkbia':
        colors_pw.append('#d62728')
    elif r['category'] == 'TARGET':
        colors_pw.append('#ff7f0e')
    else:
        colors_pw.append('#1f77b4')
ax.barh(range(len(df_pw_sorted)), df_pw_sorted['mean_delta_H'].values * 1e3,
        color=colors_pw, alpha=0.7)
ax.set_yticks(range(len(df_pw_sorted)))
ax.set_yticklabels(df_pw_sorted['gene'].values, fontsize=7)
ax.axvline(0, color='black', ls='-', lw=0.5)
ax.set_xlabel('Mean ΔH_g × 10³')
ax.set_title('NF-κB pathway: ΔH_g (old−young)')
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#d62728', label='Nfkbia'),
    Patch(color='#ff7f0e', label='Targets'),
    Patch(color='#1f77b4', label='Upstream'),
], fontsize=7, loc='lower right')

# Panel D: SEQ vs ENZ per cluster
ax = axes[1, 0]
ax.scatter(enz_arr * 1e3, seq_arr * 1e3, c='steelblue', s=50, alpha=0.7, edgecolors='black', lw=0.5)
lims = [min(enz_arr.min(), seq_arr.min()) * 1e3 * 1.1, max(enz_arr.max(), seq_arr.max()) * 1e3 * 1.1]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
ax.set_xlabel('ΔH(ENZ) × 10³')
ax.set_ylabel('ΔH(SEQ) × 10³')
ax.set_title(f'SEQ vs ENZ divergence\nSEQ > ENZ in {(divergence > 0).sum()}/{len(divergence)} clusters')
for i, (_, r) in enumerate(aging.iterrows()):
    ax.annotate(str(int(r['cluster'])), (enz_arr[i]*1e3, seq_arr[i]*1e3), fontsize=6)

# Panel E: GP532 effect on Nfkbia entropy across conditions
ax = axes[1, 1]
cond_data = {}
for label in ['aging', 'GP532_6h_young', 'GP532_24h_young', 'GP532_6h_old', 'GP532_24h_old']:
    subset = df[df['comparison'] == label]
    if len(subset) > 0:
        cond_data[label] = subset['delta_H_nfkbia'].values * 1e3

positions = list(range(len(cond_data)))
bp = ax.boxplot(cond_data.values(), positions=positions, patch_artist=True)
colors_box = ['#d62728', '#90EE90', '#2ca02c', '#FFA07A', '#FF6347']
for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax.set_xticks(positions)
ax.set_xticklabels([k.replace('_', '\n') for k in cond_data.keys()], fontsize=7)
ax.axhline(0, color='black', ls='--', lw=0.5)
ax.set_ylabel('ΔH_g(Nfkbia) × 10³')
ax.set_title('Nfkbia entropy change by condition')

# Panel F: H(Nfkbia) absolute values across all 6 conditions
ax = axes[1, 2]
cond_order = ['young_untreated', 'young_GP532_6h', 'young_GP532_24h',
              'old_untreated', 'old_GP532_6h', 'old_GP532_24h']
cond_labels = ['Y_UT', 'Y_GP6', 'Y_GP24', 'O_UT', 'O_GP6', 'O_GP24']

# Collect mean H(Nfkbia) per condition across all cells
for i, cond in enumerate(cond_order):
    mask = adata.obs['condition'].values == cond
    if mask.sum() < 20:
        continue
    X = adata.raw.X[mask]
    H = gene_entropy_contrib(X)
    h_nfkbia = H[nfkbia_idx]
    ax.bar(i, h_nfkbia * 1e3, color=['#2ca02c', '#90EE90', '#98FB98',
           '#d62728', '#FF6347', '#FFA07A'][i], alpha=0.7, edgecolor='black', lw=0.5)

ax.set_xticks(range(len(cond_order)))
ax.set_xticklabels(cond_labels, fontsize=9)
ax.set_ylabel('H̄(Nfkbia) × 10³')
ax.set_title('Absolute Nfkbia entropy by condition')

plt.tight_layout()
plt.savefig(OUTDIR / 'gudkov_entropy_replication.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTDIR / 'gudkov_entropy_replication.pdf', bbox_inches='tight')
print(f"\nFigure saved to {OUTDIR}")

# ── Save ──
df.to_csv(OUTDIR / 'entropy_per_cluster_all_comparisons.csv', index=False)
aging.to_csv(OUTDIR / 'entropy_aging_per_cluster.csv', index=False)
df_pathway.to_csv(OUTDIR / 'pathway_delta_H_summary.csv', index=False)

# ── FINAL VERDICT ──
print("\n" + "=" * 70)
print("VERDICT: Paper 1 replication on Gudkov bone marrow")
print("=" * 70)

print(f"\nTest 1 — ΔH_g(Nfkbia) positive:")
print(f"  {n_positive}/{n_total} clusters ({100*n_positive/n_total:.0f}%)")
print(f"  Mean ΔH = {mean_delta:.6f}, sign test p = {sign_p:.4f}")
if mean_delta > 0 and sign_p < 0.05:
    print("  → SCENARIO A: REPLICATES ✓")
elif mean_delta <= 0 or (sign_p > 0.05 and n_positive < n_total / 2):
    print("  → SCENARIO B: DOES NOT REPLICATE ✗")
else:
    print("  → SCENARIO C: MIXED / INCONCLUSIVE")

print(f"\nTest 2 — Nfkbia rank among targets: #{nfkbia_rank}/{len(target_deltas)}")
nfkbia_top = nfkbia_rank <= 3
print(f"  {'Top 3 → consistent with outlier status' if nfkbia_top else 'Not top 3 → not clear outlier'}")

print(f"\nTest 3 — SEQ > ENZ:")
print(f"  SEQ > ENZ in {(divergence > 0).sum()}/{len(divergence)} clusters")
print(f"  Mean divergence: {np.mean(divergence):.6f}")

print(f"\nKey context:")
print(f"  - Single tissue (BM), N=1 biological replicate")
print(f"  - Paper 1 TMS FACS: 143 tissue×age groups, 23 tissues")
print(f"  - GP532 mimics aging (ρ=0.43, 16/16) remains STRONG regardless")
