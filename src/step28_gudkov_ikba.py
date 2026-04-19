#!/usr/bin/env python3
"""
Step 28: IκBα (Nfkbia) variance analysis on Gudkov bone marrow data.

Analysis A for Paper 1 integration:
- Per-cluster CV(Nfkbia) in young_untreated vs old_untreated
- If CV increases with age → direct replication of central claim
- Also: GP532 effect on Nfkbia variance (does NF-κB activation increase its own inhibitor's variance?)

Analysis C: NF-κB target outlier test replication
- Compare Nfkbia CV increase to genome-wide distribution
- Is Nfkbia an outlier among NF-κB pathway genes?

Analysis D: Upstream stability replication
- NF-κB upstream regulators: stable or variable?
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
GUDKOV = Path('/Users/teo/Desktop/research/RQ022491-Gudkov/gudkov_merged.h5ad')
OUTDIR = Path('/Users/teo/Desktop/research/coupling_atlas/results/step28_gudkov_ikba')
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── NF-κB pathway genes (mouse orthologs) ──
NFKB_TARGETS = [
    'Nfkbia',   # IκBα — the star
    'Nfkbib',   # IκBβ
    'Nfkbie',   # IκBε
    'Tnfaip3',  # A20
    'Bcl2l1',   # Bcl-xL
    'Ccl2',     # MCP-1
    'Cxcl10',   # IP-10
    'Il6',      # IL-6
    'Tnf',      # TNFα
    'Icam1',    # ICAM-1
    'Birc3',    # cIAP2
    'Ptgs2',    # COX-2
    'Mmp9',     # MMP-9
    'Ccl5',     # RANTES
    'Il1b',     # IL-1β
    'Cxcl1',    # KC/GROα
    'Socs3',    # SOCS3
]

NFKB_UPSTREAM = [
    'Rela',     # p65
    'Nfkb1',    # p50/p105
    'Nfkb2',    # p52/p100
    'Ikbkb',    # IKKβ
    'Ikbkg',    # NEMO
    'Chuk',     # IKKα
    'Traf6',    # TRAF6
    'Myd88',    # MyD88
    'Irak1',    # IRAK1
    'Irak4',    # IRAK4
    'Tlr4',     # TLR4
    'Tlr5',     # TLR5 (GP532 target!)
]

# ── Load data ──
print("Loading Gudkov data...")
adata = sc.read_h5ad(GUDKOV)
raw_var = list(adata.raw.var_names)
print(f"  {adata.shape[0]} cells, {len(raw_var)} raw genes")

# ── Helper: get raw expression for a gene ──
def get_raw_expr(gene):
    if gene in raw_var:
        idx = raw_var.index(gene)
        return np.asarray(adata.raw.X[:, idx].todense()).flatten()
    return None

# ── Analysis A: Nfkbia CV per cluster, young vs old ──
print("\n=== ANALYSIS A: Nfkbia variance per cluster ===")

nfkbia = get_raw_expr('Nfkbia')
conditions = ['young_untreated', 'old_untreated', 'young_GP532_6h', 'old_GP532_6h',
              'young_GP532_24h', 'old_GP532_24h']

results_a = []
for cl in sorted(adata.obs['leiden'].unique(), key=int):
    cl_mask = adata.obs['leiden'] == cl
    n_total = cl_mask.sum()

    row = {'cluster': int(cl), 'n_total': n_total}

    for cond in conditions:
        mask = cl_mask & (adata.obs['condition'] == cond)
        n = mask.sum()
        if n < 20:  # minimum cells for reliable CV
            row[f'n_{cond}'] = n
            row[f'cv_{cond}'] = np.nan
            row[f'mean_{cond}'] = np.nan
            row[f'frac_nonzero_{cond}'] = np.nan
            continue

        expr = nfkbia[mask]
        nonzero = expr > 0

        row[f'n_{cond}'] = n
        row[f'mean_{cond}'] = np.mean(expr)
        row[f'cv_{cond}'] = np.std(expr) / np.mean(expr) if np.mean(expr) > 0 else np.nan
        row[f'frac_nonzero_{cond}'] = np.mean(nonzero)

    # Young vs old comparison (untreated only)
    if not np.isnan(row.get(f'cv_young_untreated', np.nan)) and not np.isnan(row.get(f'cv_old_untreated', np.nan)):
        row['cv_ratio_old_young'] = row['cv_old_untreated'] / row['cv_young_untreated']

        # Levene test for variance difference
        y_mask = cl_mask & (adata.obs['condition'] == 'young_untreated')
        o_mask = cl_mask & (adata.obs['condition'] == 'old_untreated')
        stat, pval = stats.levene(nfkbia[y_mask], nfkbia[o_mask])
        row['levene_stat'] = stat
        row['levene_p'] = pval

        # Brown-Forsythe (median-based, more robust)
        stat_bf, pval_bf = stats.levene(nfkbia[y_mask], nfkbia[o_mask], center='median')
        row['bf_stat'] = stat_bf
        row['bf_p'] = pval_bf

    results_a.append(row)

df_a = pd.DataFrame(results_a)

# Print summary
print("\nNfkbia CV: young_untreated vs old_untreated per cluster")
print("-" * 80)
cols = ['cluster', 'n_young_untreated', 'n_old_untreated',
        'cv_young_untreated', 'cv_old_untreated', 'cv_ratio_old_young', 'levene_p']
valid = df_a.dropna(subset=['cv_ratio_old_young'])
print(valid[cols].to_string(index=False, float_format='{:.3f}'.format))

n_increase = (valid['cv_ratio_old_young'] > 1).sum()
n_decrease = (valid['cv_ratio_old_young'] < 1).sum()
n_sig = (valid['levene_p'] < 0.05).sum()
mean_ratio = valid['cv_ratio_old_young'].mean()
median_ratio = valid['cv_ratio_old_young'].median()

print(f"\nCV increases with age: {n_increase}/{len(valid)} clusters")
print(f"CV decreases with age: {n_decrease}/{len(valid)} clusters")
print(f"Mean CV ratio (old/young): {mean_ratio:.3f}")
print(f"Median CV ratio: {median_ratio:.3f}")
print(f"Significant (Levene p<0.05): {n_sig}/{len(valid)}")

# Sign test
sign_p = stats.binomtest(n_increase, len(valid), 0.5).pvalue
print(f"Sign test (H0: 50/50): p = {sign_p:.4f}")

# ── GP532 effect on Nfkbia variance ──
print("\n=== GP532 EFFECT on Nfkbia variance ===")
for age in ['young', 'old']:
    untreated = f'{age}_untreated'
    gp6 = f'{age}_GP532_6h'
    gp24 = f'{age}_GP532_24h'

    cvs_ut = valid[f'cv_{untreated}'].values
    cvs_6 = valid[f'cv_{gp6}'].dropna().values
    cvs_24 = valid[f'cv_{gp24}'].dropna().values

    if len(cvs_6) > 0:
        t, p = stats.wilcoxon(cvs_ut[:len(cvs_6)], cvs_6)
        ratio = np.mean(cvs_6) / np.mean(cvs_ut[:len(cvs_6)])
        print(f"  {age}: GP532_6h/untreated CV ratio = {ratio:.3f} (Wilcoxon p={p:.4f})")
    if len(cvs_24) > 0:
        t, p = stats.wilcoxon(cvs_ut[:len(cvs_24)], cvs_24)
        ratio = np.mean(cvs_24) / np.mean(cvs_ut[:len(cvs_24)])
        print(f"  {age}: GP532_24h/untreated CV ratio = {ratio:.3f} (Wilcoxon p={p:.4f})")

# ── Analysis C: NF-κB target outlier test ──
print("\n=== ANALYSIS C: NF-κB target outlier test ===")

# Compute CV ratio (old/young) for all expressed genes
print("Computing genome-wide CV ratios...")
young_mask = adata.obs['condition'] == 'young_untreated'
old_mask = adata.obs['condition'] == 'old_untreated'

gene_cv_ratios = {}
for gene in NFKB_TARGETS + NFKB_UPSTREAM:
    expr = get_raw_expr(gene)
    if expr is None:
        continue

    y_expr = expr[young_mask]
    o_expr = expr[old_mask]

    mean_y = np.mean(y_expr)
    mean_o = np.mean(o_expr)

    if mean_y > 0.01 and mean_o > 0.01:
        cv_y = np.std(y_expr) / mean_y
        cv_o = np.std(o_expr) / mean_o
        gene_cv_ratios[gene] = {
            'cv_young': cv_y, 'cv_old': cv_o, 'cv_ratio': cv_o / cv_y,
            'mean_young': mean_y, 'mean_old': mean_o,
            'category': 'target' if gene in NFKB_TARGETS else 'upstream'
        }

# Genome-wide background (sample 2000 random genes for comparison)
np.random.seed(42)
all_genes = [g for g in raw_var if g not in NFKB_TARGETS + NFKB_UPSTREAM]
sample_genes = np.random.choice(all_genes, size=min(2000, len(all_genes)), replace=False)

bg_ratios = []
for gene in sample_genes:
    expr = get_raw_expr(gene)
    y_expr = expr[young_mask]
    o_expr = expr[old_mask]
    mean_y = np.mean(y_expr)
    mean_o = np.mean(o_expr)
    if mean_y > 0.01 and mean_o > 0.01:
        cv_y = np.std(y_expr) / mean_y
        cv_o = np.std(o_expr) / mean_o
        bg_ratios.append(cv_o / cv_y)

bg_ratios = np.array(bg_ratios)
bg_mean = np.mean(bg_ratios)
bg_std = np.std(bg_ratios)

print(f"\nBackground CV ratio (old/young): {bg_mean:.3f} ± {bg_std:.3f} (n={len(bg_ratios)} genes)")
print(f"Background median: {np.median(bg_ratios):.3f}")

print(f"\nNF-κB pathway genes:")
print(f"{'Gene':<12} {'Category':<10} {'CV_young':<10} {'CV_old':<10} {'Ratio':<8} {'Z-score':<8}")
print("-" * 60)
for gene, d in sorted(gene_cv_ratios.items(), key=lambda x: -x[1]['cv_ratio']):
    z = (d['cv_ratio'] - bg_mean) / bg_std
    print(f"{gene:<12} {d['category']:<10} {d['cv_young']:<10.3f} {d['cv_old']:<10.3f} {d['cv_ratio']:<8.3f} {z:<8.2f}")

# Nfkbia percentile
if 'Nfkbia' in gene_cv_ratios:
    nfkbia_ratio = gene_cv_ratios['Nfkbia']['cv_ratio']
    percentile = np.mean(bg_ratios < nfkbia_ratio) * 100
    print(f"\nNfkbia CV ratio: {nfkbia_ratio:.3f} (percentile: {percentile:.1f}%)")

# ── Analysis D: Upstream stability ──
print("\n=== ANALYSIS D: Upstream stability ===")
target_ratios = [d['cv_ratio'] for g, d in gene_cv_ratios.items() if d['category'] == 'target']
upstream_ratios = [d['cv_ratio'] for g, d in gene_cv_ratios.items() if d['category'] == 'upstream']

if target_ratios and upstream_ratios:
    print(f"Target mean CV ratio: {np.mean(target_ratios):.3f} ± {np.std(target_ratios):.3f} (n={len(target_ratios)})")
    print(f"Upstream mean CV ratio: {np.mean(upstream_ratios):.3f} ± {np.std(upstream_ratios):.3f} (n={len(upstream_ratios)})")
    print(f"Background mean CV ratio: {bg_mean:.3f}")

    # Targets vs upstream
    if len(target_ratios) >= 3 and len(upstream_ratios) >= 3:
        t, p = stats.mannwhitneyu(target_ratios, upstream_ratios, alternative='greater')
        print(f"Targets > Upstream: Mann-Whitney p = {p:.4f}")

    # Targets vs background
    t_bg = [(r - bg_mean) / bg_std for r in target_ratios]
    print(f"Targets mean Z-score vs background: {np.mean(t_bg):.2f}")
    u_bg = [(r - bg_mean) / bg_std for r in upstream_ratios]
    print(f"Upstream mean Z-score vs background: {np.mean(u_bg):.2f}")

# ── Figure ──
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: CV ratio per cluster (young vs old)
ax = axes[0, 0]
valid_sorted = valid.sort_values('cv_ratio_old_young', ascending=False)
colors = ['#d62728' if r > 1 else '#2ca02c' for r in valid_sorted['cv_ratio_old_young']]
bars = ax.bar(range(len(valid_sorted)), valid_sorted['cv_ratio_old_young'], color=colors, alpha=0.7)
ax.axhline(1.0, color='black', ls='--', lw=1)
ax.set_xlabel('Cluster (sorted)')
ax.set_ylabel('CV ratio (old/young)')
ax.set_title(f'Nfkbia CV: old vs young per cluster\n{n_increase}/{len(valid)} clusters show increase')
ax.set_xticks(range(len(valid_sorted)))
ax.set_xticklabels(valid_sorted['cluster'].values, fontsize=7, rotation=45)

# Panel B: CV young vs CV old scatter
ax = axes[0, 1]
ax.scatter(valid['cv_young_untreated'], valid['cv_old_untreated'],
           c='steelblue', s=60, alpha=0.7, edgecolors='black', lw=0.5)
lims = [min(valid['cv_young_untreated'].min(), valid['cv_old_untreated'].min()) * 0.9,
        max(valid['cv_young_untreated'].max(), valid['cv_old_untreated'].max()) * 1.1]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
ax.set_xlabel('CV(Nfkbia) young')
ax.set_ylabel('CV(Nfkbia) old')
ax.set_title('Nfkbia CV: young vs old')
# Add cluster labels
for _, row in valid.iterrows():
    ax.annotate(str(int(row['cluster'])), (row['cv_young_untreated'], row['cv_old_untreated']),
                fontsize=6, ha='center', va='bottom')

# Panel C: NF-κB pathway CV ratios vs background
ax = axes[1, 0]
ax.hist(bg_ratios, bins=50, density=True, alpha=0.5, color='gray', label='Background')
for gene, d in gene_cv_ratios.items():
    color = '#d62728' if gene == 'Nfkbia' else ('#ff7f0e' if d['category'] == 'target' else '#1f77b4')
    lw = 3 if gene == 'Nfkbia' else 1.5
    ax.axvline(d['cv_ratio'], color=color, lw=lw, alpha=0.8)
    if gene in ['Nfkbia', 'Rela', 'Nfkb1', 'Tnf', 'Il6', 'Tnfaip3']:
        ax.text(d['cv_ratio'], ax.get_ylim()[1] * 0.9, gene, rotation=90, fontsize=7, va='top')

ax.set_xlabel('CV ratio (old/young)')
ax.set_ylabel('Density')
ax.set_title('Nfkbia vs genome-wide CV change')
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#d62728', label='Nfkbia'),
    Patch(color='#ff7f0e', label='NF-κB targets'),
    Patch(color='#1f77b4', label='NF-κB upstream'),
    Patch(color='gray', alpha=0.5, label='Background'),
], fontsize=8)

# Panel D: GP532 effect — CV across conditions for top clusters
ax = axes[1, 1]
cond_order = ['young_untreated', 'young_GP532_6h', 'young_GP532_24h',
              'old_untreated', 'old_GP532_6h', 'old_GP532_24h']
cond_labels = ['Y_UT', 'Y_GP6', 'Y_GP24', 'O_UT', 'O_GP6', 'O_GP24']

# Plot mean CV across clusters for each condition
mean_cvs = []
sem_cvs = []
for cond in cond_order:
    col = f'cv_{cond}'
    if col in df_a.columns:
        vals = df_a[col].dropna()
        mean_cvs.append(vals.mean())
        sem_cvs.append(vals.std() / np.sqrt(len(vals)))
    else:
        mean_cvs.append(np.nan)
        sem_cvs.append(np.nan)

colors_bar = ['#2ca02c', '#90EE90', '#98FB98', '#d62728', '#FF6347', '#FFA07A']
ax.bar(range(len(cond_order)), mean_cvs, yerr=sem_cvs, color=colors_bar,
       alpha=0.7, capsize=3, edgecolor='black', lw=0.5)
ax.set_xticks(range(len(cond_order)))
ax.set_xticklabels(cond_labels, fontsize=9)
ax.set_ylabel('Mean CV(Nfkbia) across clusters')
ax.set_title('Nfkbia CV by condition')

plt.tight_layout()
plt.savefig(OUTDIR / 'nfkbia_variance_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig(OUTDIR / 'nfkbia_variance_analysis.pdf', bbox_inches='tight')
print(f"\nFigure saved to {OUTDIR / 'nfkbia_variance_analysis.png'}")

# ── Save results ──
df_a.to_csv(OUTDIR / 'nfkbia_cv_per_cluster.csv', index=False)
pd.DataFrame(gene_cv_ratios).T.to_csv(OUTDIR / 'nfkb_pathway_cv_ratios.csv')
print(f"Results saved to {OUTDIR}")

print("\n=== SUMMARY ===")
print(f"Nfkbia CV increase with age: {n_increase}/{len(valid)} clusters ({100*n_increase/len(valid):.0f}%)")
print(f"Mean CV ratio (old/young): {mean_ratio:.3f}")
print(f"Sign test p = {sign_p:.4f}")
if 'Nfkbia' in gene_cv_ratios:
    print(f"Nfkbia CV ratio percentile vs genome: {percentile:.1f}%")
    print(f"Nfkbia Z-score: {(nfkbia_ratio - bg_mean) / bg_std:.2f}")
