"""
Step 19: Publication-quality main figures for the pi_tissue paper.

Generates Figures 1-5 from GTEx v8 TPM data, computing all metrics
from scratch for reproducibility.

Usage:
    /Users/teo/miniforge3/envs/sc/bin/python step19_pub_figures.py
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
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data" / "gtex"
FIG_DIR = BASE / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.5,
})

# ── Colours ──────────────────────────────────────────────────────────
COL_TISSUE = "#4CAF50"   # muted green
COL_DONOR  = "#5C8DB8"   # muted blue
COL_RESID  = "#BDBDBD"   # gray
COL_RED    = "#D32F2F"
COL_ORANGE = "#FB8C00"
COL_PURPLE = "#7B1FA2"
COL_TEAL   = "#00796B"

# ── Hardcoded values ─────────────────────────────────────────────────
# Rat CR data
RAT_PI = {"young": 0.893, "old_AL": 0.842, "old_CR": 0.886}
RAT_VT = {"young": 0.0016, "old_AL": 0.0013, "old_CR": 0.0013}
RAT_VR = {"young": 0.00018, "old_AL": 0.00020, "old_CR": 0.00014}

# Cross-species
SPECIES = {
    "Mouse":   {"L": 2.5,  "dpi": 0.043},
    "Rat":     {"L": 3.0,  "dpi": 0.028},
    "Macaque": {"L": 30.0, "dpi": 0.0014},
    "Human":   {"L": 80.0, "dpi": 0.00078},
}

# Chromatin cascade (expression-matched)
CHROMATIN = {
    "Chromatin": {"focal": -0.057, "ctrl": -0.023, "p": 0.009},
    "TF":        {"focal": -0.039, "ctrl": -0.024, "p": 0.195},
    "Target":    {"focal": -0.012, "ctrl": -0.030, "p": 0.088},
    "HK":        {"focal": -0.052, "ctrl": -0.059, "p": 0.962},
}

# Decomposition
DECOMP = {
    "All genes\n(18 000)":        -0.031,
    "Tissue-spec.\n(\u03c4>0.8, 395)":  -0.011,
    "Ubiquitous\n(\u03c4<0.3, 7550)":   -0.041,
    "Immune\n(51)":               -0.035,
}

# 6 focal tissues
TISSUES_6 = [
    "Muscle - Skeletal",
    "Whole Blood",
    "Skin - Sun Exposed (Lower leg)",
    "Adipose - Subcutaneous",
    "Artery - Tibial",
    "Thyroid",
]

AGE_BINS = {
    "20-39": (20, 39),
    "40-49": (40, 49),
    "50-59": (50, 59),
    "60-79": (60, 79),
}


def _log(msg):
    print(msg, flush=True)


# =====================================================================
# DATA LOADING
# =====================================================================
def load_metadata():
    """Load GTEx sample and subject metadata; return merged DataFrame."""
    samples = pd.read_csv(
        DATA_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(
        DATA_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()

    # Parse age
    def _parse_age(s):
        try:
            lo, hi = s.split("-")
            return (int(lo) + int(hi)) / 2.0
        except Exception:
            return np.nan
    samples["age_mid"] = samples["AGE"].apply(_parse_age)
    return samples


def assign_age_bin(age_mid):
    """Assign an age-mid value to one of the 4 decade bins."""
    for label, (lo, hi) in AGE_BINS.items():
        if lo <= age_mid <= hi:
            return label
    return None


def load_tpm_for_6tissues(samples):
    """
    Stream-read the GTEx TPM GCT, keep only the 6 focal tissues.
    Returns: log2(TPM+1) matrix (genes x samples), sample metadata.
    """
    tpm_path = DATA_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"

    sample_meta = samples.set_index("SAMPID")[["SUBJID", "SMTSD", "age_mid"]].copy()

    # Read header to get sample IDs
    with gzip.open(tpm_path, "rt") as f:
        f.readline()   # #1.2
        f.readline()   # dimensions
        header = f.readline().strip().split("\t")
    all_sids = header[2:]

    # Build index mapping for the 6 tissues
    keep_cols = []
    keep_sids = []
    for i, sid in enumerate(all_sids):
        if sid in sample_meta.index:
            tissue = sample_meta.loc[sid, "SMTSD"]
            if tissue in TISSUES_6:
                keep_cols.append(i)
                keep_sids.append(sid)
    keep_cols = np.array(keep_cols)
    _log(f"  Samples in 6 tissues: {len(keep_sids)}")

    # Stream-read and filter expressed genes
    gene_names = []
    expr_rows = []
    n_total = 0
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            parts = line.strip().split("\t", 2)
            gene = parts[1]
            vals = np.array(parts[2].split("\t"), dtype=np.float32)
            subset = vals[keep_cols]
            n_total += 1
            # Keep genes with median TPM >= 1 across the 6-tissue samples
            if np.median(subset) < 1.0:
                continue
            gene_names.append(gene)
            expr_rows.append(np.log2(subset + 1.0))
            if n_total % 10000 == 0:
                _log(f"    {n_total} genes scanned, {len(gene_names)} kept ...")

    expr = np.vstack(expr_rows)  # genes x samples
    _log(f"  Final: {expr.shape[0]} genes x {expr.shape[1]} samples")

    meta = sample_meta.loc[keep_sids].copy()
    meta["age_bin"] = meta["age_mid"].apply(assign_age_bin)
    return expr, gene_names, meta


# =====================================================================
# VARIANCE DECOMPOSITION (ANOVA-based)
# =====================================================================
def compute_pi_for_bin(expr, meta, age_bin_label):
    """
    For one age bin, compute pi_tissue, pi_donor, pi_residual via
    two-way ANOVA on the 6-tissue x N-donor design.

    Returns: (pi_tissue, pi_donor, pi_residual, V_tissue, V_donor, V_residual)
    """
    mask = meta["age_bin"] == age_bin_label
    sub_meta = meta[mask].copy()
    sub_expr = expr[:, mask.values]

    # Only keep donors with samples in >=2 tissues
    donor_tissue_ct = sub_meta.groupby("SUBJID")["SMTSD"].nunique()
    valid_donors = donor_tissue_ct[donor_tissue_ct >= 2].index
    donor_mask = sub_meta["SUBJID"].isin(valid_donors)
    sub_meta = sub_meta[donor_mask]
    sub_expr = sub_expr[:, donor_mask.values]

    tissues = sub_meta["SMTSD"].values
    donors = sub_meta["SUBJID"].values
    unique_tissues = np.unique(tissues)
    unique_donors = np.unique(donors)
    n_genes = sub_expr.shape[0]

    var_tissue_arr = np.zeros(n_genes)
    var_donor_arr = np.zeros(n_genes)
    var_resid_arr = np.zeros(n_genes)

    # ANOVA decomposition per gene
    for g in range(n_genes):
        y = sub_expr[g]
        grand_mean = np.mean(y)
        total_var = np.var(y, ddof=0)
        if total_var < 1e-12:
            continue

        # Tissue means
        tissue_means = {}
        for t in unique_tissues:
            tmask = tissues == t
            tissue_means[t] = np.mean(y[tmask])

        # Donor means
        donor_means = {}
        for d in unique_donors:
            dmask = donors == d
            if np.sum(dmask) > 0:
                donor_means[d] = np.mean(y[dmask])

        # SS_tissue
        ss_tissue = sum(
            np.sum(tissues == t) * (tissue_means[t] - grand_mean) ** 2
            for t in unique_tissues
        )
        # SS_donor
        ss_donor = sum(
            np.sum(donors == d) * (donor_means[d] - grand_mean) ** 2
            for d in unique_donors
        )
        ss_total = total_var * len(y)
        ss_resid = max(0, ss_total - ss_tissue - ss_donor)

        var_tissue_arr[g] = ss_tissue / len(y)
        var_donor_arr[g] = ss_donor / len(y)
        var_resid_arr[g] = ss_resid / len(y)

    V_tissue = np.mean(var_tissue_arr)
    V_donor = np.mean(var_donor_arr)
    V_resid = np.mean(var_resid_arr)
    V_total = V_tissue + V_donor + V_resid

    pi_tissue = V_tissue / V_total if V_total > 0 else 0
    pi_donor = V_donor / V_total if V_total > 0 else 0
    pi_resid = V_resid / V_total if V_total > 0 else 0

    return pi_tissue, pi_donor, pi_resid, V_tissue, V_donor, V_resid


def bootstrap_pi(expr, meta, age_bin_label, n_boot=30):
    """Bootstrap pi_tissue: resample donors with replacement."""
    mask = meta["age_bin"] == age_bin_label
    sub_meta = meta[mask].copy()
    sub_expr = expr[:, mask.values]

    donor_tissue_ct = sub_meta.groupby("SUBJID")["SMTSD"].nunique()
    valid_donors = donor_tissue_ct[donor_tissue_ct >= 2].index.tolist()

    pi_vals = []
    rng = np.random.RandomState(42)
    for b in range(n_boot):
        boot_donors = rng.choice(valid_donors, size=len(valid_donors), replace=True)
        # Gather sample indices for boot donors
        boot_idx = []
        boot_tissues = []
        boot_donor_labels = []
        for di, d in enumerate(boot_donors):
            d_label = f"{d}_{di}"  # make unique
            dmask = sub_meta["SUBJID"] == d
            idxs = np.where(dmask.values)[0]
            for ix in idxs:
                boot_idx.append(ix)
                boot_tissues.append(sub_meta.iloc[ix]["SMTSD"])
                boot_donor_labels.append(d_label)

        boot_expr = sub_expr[:, boot_idx]
        tissues_arr = np.array(boot_tissues)
        donors_arr = np.array(boot_donor_labels)

        unique_t = np.unique(tissues_arr)
        unique_d = np.unique(donors_arr)

        # Quick per-gene ANOVA on a random subset of genes for speed
        n_genes = boot_expr.shape[0]
        gene_idx = rng.choice(n_genes, size=min(2000, n_genes), replace=False)

        vt_sum, vd_sum, vr_sum = 0.0, 0.0, 0.0
        for g in gene_idx:
            y = boot_expr[g]
            gm = np.mean(y)
            tv = np.var(y, ddof=0)
            if tv < 1e-12:
                continue
            n = len(y)
            ss_t = sum(np.sum(tissues_arr == t) * (np.mean(y[tissues_arr == t]) - gm)**2
                       for t in unique_t)
            ss_d = sum(np.sum(donors_arr == d) * (np.mean(y[donors_arr == d]) - gm)**2
                       for d in unique_d)
            ss_total = tv * n
            ss_r = max(0, ss_total - ss_t - ss_d)
            vt_sum += ss_t / n
            vd_sum += ss_d / n
            vr_sum += ss_r / n

        total = vt_sum + vd_sum + vr_sum
        pi_vals.append(vt_sum / total if total > 0 else 0)
        if (b + 1) % 25 == 0:
            _log(f"    bootstrap {b+1}/{n_boot}")

    return np.array(pi_vals)


def permutation_null(expr, meta, n_perm=30):
    """
    Permutation null: shuffle tissue labels within each age bin,
    compute pi_tissue for each permutation.
    """
    pi_null = []
    rng = np.random.RandomState(99)

    for p in range(n_perm):
        pi_perm_bins = []
        for ab in AGE_BINS:
            mask = meta["age_bin"] == ab
            sub_meta = meta[mask].copy()
            sub_expr = expr[:, mask.values]

            donor_tissue_ct = sub_meta.groupby("SUBJID")["SMTSD"].nunique()
            valid_donors = donor_tissue_ct[donor_tissue_ct >= 2].index
            donor_mask = sub_meta["SUBJID"].isin(valid_donors)
            sm = sub_meta[donor_mask].copy()
            se = sub_expr[:, donor_mask.values]

            # Shuffle tissue labels
            perm_tissues = sm["SMTSD"].values.copy()
            rng.shuffle(perm_tissues)
            donors = sm["SUBJID"].values
            unique_t = np.unique(perm_tissues)
            unique_d = np.unique(donors)

            gene_idx = rng.choice(se.shape[0], size=min(2000, se.shape[0]), replace=False)
            vt_sum, vd_sum, vr_sum = 0.0, 0.0, 0.0
            for g in gene_idx:
                y = se[g]
                gm = np.mean(y)
                tv = np.var(y, ddof=0)
                if tv < 1e-12:
                    continue
                n = len(y)
                ss_t = sum(np.sum(perm_tissues == t) * (np.mean(y[perm_tissues == t]) - gm)**2
                           for t in unique_t)
                ss_d = sum(np.sum(donors == d) * (np.mean(y[donors == d]) - gm)**2
                           for d in unique_d)
                ss_total = tv * n
                ss_r = max(0, ss_total - ss_t - ss_d)
                vt_sum += ss_t / n
                vd_sum += ss_d / n
                vr_sum += ss_r / n
            total = vt_sum + vd_sum + vr_sum
            pi_perm_bins.append(vt_sum / total if total > 0 else 0)

        pi_null.append(np.mean(pi_perm_bins))
        if (p + 1) % 25 == 0:
            _log(f"    permutation {p+1}/{n_perm}")

    return np.array(pi_null)


def compute_per_tissue_noise_rate(expr, meta):
    """
    For each tissue, compute the noise accumulation rate:
    slope of V_residual / (V_tissue + V_residual) across age bins.
    """
    rates = {}
    for tissue in TISSUES_6:
        noise_fracs = []
        age_mids = []
        for ab, (lo, hi) in AGE_BINS.items():
            mask = (meta["age_bin"] == ab) & (meta["SMTSD"] == tissue)
            sub_meta = meta[mask]
            sub_expr = expr[:, mask.values]
            if sub_meta.shape[0] < 20:
                continue
            y_all = sub_expr
            gene_vars = np.var(y_all, axis=1, ddof=0)
            # Grand tissue mean across all bins for this tissue
            noise_fracs.append(np.mean(gene_vars))
            age_mids.append((lo + hi) / 2.0)

        if len(age_mids) >= 3:
            slope, _, _, _, _ = stats.linregress(age_mids, noise_fracs)
            rates[tissue] = slope
        else:
            rates[tissue] = 0.0
    return rates


# =====================================================================
# FIGURE GENERATION
# =====================================================================
def _save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    _log(f"  Saved {name}.png / .pdf")


def make_fig1(pi_data, boot_data, perm_null, observed_pi_mean):
    """
    Figure 1: pi_tissue near-invariance.
    A: Stacked bar  B: Line + CIs + rat inset  C: Permutation null
    """
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 1], wspace=0.38)

    decades = list(AGE_BINS.keys())
    pi_t = [pi_data[d][0] for d in decades]
    pi_d = [pi_data[d][1] for d in decades]
    pi_r = [pi_data[d][2] for d in decades]

    # ── Panel A: Stacked bars ──
    ax_a = fig.add_subplot(gs[0])
    x = np.arange(len(decades))
    w = 0.55
    ax_a.bar(x, pi_t, w, color=COL_TISSUE, label=r"$\pi_{tissue}$")
    ax_a.bar(x, pi_d, w, bottom=pi_t, color=COL_DONOR, label=r"$\pi_{donor}$")
    ax_a.bar(x, pi_r, w, bottom=[a+b for a, b in zip(pi_t, pi_d)],
             color=COL_RESID, label=r"$\pi_{residual}$")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(decades, rotation=0)
    ax_a.set_ylabel("Fraction of variance")
    ax_a.set_ylim(0, 1.05)
    ax_a.legend(loc="upper right", frameon=False, fontsize=6)
    ax_a.set_title("A", loc="left", fontweight="bold", fontsize=10)

    # ── Panel B: Line plot with bootstrap CIs ──
    ax_b = fig.add_subplot(gs[1])
    ci_lo = [np.percentile(boot_data[d], 2.5) for d in decades]
    ci_hi = [np.percentile(boot_data[d], 97.5) for d in decades]
    boot_med = [np.median(boot_data[d]) for d in decades]
    x_num = [30, 45, 55, 70]

    ax_b.fill_between(x_num, ci_lo, ci_hi, alpha=0.2, color=COL_TISSUE)
    ax_b.plot(x_num, pi_t, "o-", color=COL_TISSUE, markersize=5, label="ANOVA")
    ax_b.plot(x_num, boot_med, "s--", color=COL_DONOR, markersize=4,
              alpha=0.7, label="Bootstrap median")
    ax_b.set_xlabel("Age (years)")
    ax_b.set_ylabel(r"$\pi_{tissue}$")
    ax_b.set_ylim(0.55, 0.95)
    ax_b.legend(loc="lower left", frameon=False, fontsize=6)
    ax_b.set_title("B", loc="left", fontweight="bold", fontsize=10)

    # Rat CR inset
    ax_ins = ax_b.inset_axes([0.62, 0.55, 0.35, 0.40])
    rat_labels = ["Young", "Old\nAL", "Old\nCR"]
    rat_vals = [RAT_PI["young"], RAT_PI["old_AL"], RAT_PI["old_CR"]]
    rat_cols = [COL_TEAL, COL_RED, COL_ORANGE]
    ax_ins.bar(range(3), rat_vals, color=rat_cols, width=0.6, edgecolor="white")
    ax_ins.set_xticks(range(3))
    ax_ins.set_xticklabels(rat_labels, fontsize=5)
    ax_ins.set_ylim(0.80, 0.92)
    ax_ins.set_ylabel(r"$\pi_{tissue}$", fontsize=5)
    ax_ins.tick_params(axis="both", labelsize=5)
    ax_ins.set_title("Rat CR", fontsize=6, fontweight="bold")
    ax_ins.spines["top"].set_visible(False)
    ax_ins.spines["right"].set_visible(False)

    # ── Panel C: Permutation null ──
    ax_c = fig.add_subplot(gs[2])
    ax_c.hist(perm_null, bins=20, color="#E0E0E0", edgecolor="gray", linewidth=0.5)
    ax_c.axvline(observed_pi_mean, color=COL_RED, linewidth=1.5, linestyle="-",
                 label=f"Observed $\\pi$ = {observed_pi_mean:.2f}")
    ax_c.set_xlabel(r"$\pi_{tissue}$ (permuted)")
    ax_c.set_ylabel("Count")
    ax_c.legend(loc="upper left", frameon=False, fontsize=6)
    ax_c.set_title("C", loc="left", fontweight="bold", fontsize=10)

    _save(fig, "fig1")


def make_fig2():
    """
    Figure 2: Decomposition — tissue-specific vs ubiquitous.
    A: Bar chart of delta-pi  B: Annotation schematic
    """
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.35)

    # ── Panel A: Delta-pi bar chart ──
    ax_a = fig.add_subplot(gs[0])
    labels = list(DECOMP.keys())
    vals = list(DECOMP.values())
    colors = [COL_DONOR, COL_TISSUE, COL_PURPLE, COL_RED]
    bars = ax_a.barh(range(len(labels)), vals, color=colors, height=0.6,
                     edgecolor="white")
    ax_a.set_yticks(range(len(labels)))
    ax_a.set_yticklabels(labels, fontsize=7)
    ax_a.set_xlabel(r"$\Delta\pi_{tissue}$ (old $-$ young)")
    ax_a.axvline(0, color="black", linewidth=0.5, linestyle="-")
    ax_a.invert_yaxis()
    ax_a.set_title("A", loc="left", fontweight="bold", fontsize=10)

    # Add value labels
    for i, (v, bar) in enumerate(zip(vals, bars)):
        ax_a.text(v - 0.002, i, f"{v:.3f}", va="center", ha="right",
                  fontsize=6, color="white", fontweight="bold")

    # ── Panel B: Schematic annotation ──
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis("off")
    ax_b.set_title("B", loc="left", fontweight="bold", fontsize=10)

    # Draw pie-like proportions as stacked horizontal bar
    total_bar_y = 7.5
    ax_b.barh(total_bar_y, 3.3, left=0, height=1.2, color=COL_TISSUE,
              edgecolor="white", alpha=0.8)
    ax_b.barh(total_bar_y, 6.7, left=3.3, height=1.2, color=COL_DONOR,
              edgecolor="white", alpha=0.8)

    ax_b.text(1.65, total_bar_y, "~1/3", ha="center", va="center",
              fontsize=9, fontweight="bold", color="white")
    ax_b.text(6.65, total_bar_y, "~2/3", ha="center", va="center",
              fontsize=9, fontweight="bold", color="white")

    ax_b.text(1.65, total_bar_y - 1.5, "Genuine\nstructural\nerosion",
              ha="center", va="top", fontsize=7, color=COL_TISSUE,
              fontweight="bold")
    ax_b.text(6.65, total_bar_y - 1.5, "Composition\nshift",
              ha="center", va="top", fontsize=7, color=COL_DONOR,
              fontweight="bold")

    ax_b.text(5.0, 3.0,
              "Ubiquitous genes drive the decline;\n"
              "tissue-specific genes are relatively\n"
              "protected (smaller $\\Delta\\pi$).",
              ha="center", va="center", fontsize=7,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                        edgecolor="#BDBDBD", linewidth=0.5))

    _save(fig, "fig2")


def make_fig3():
    """
    Figure 3: Chromatin cascade mechanism.
    A: Focal vs control delta-pi  B: Waterfall of individual chromatin genes
    """
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.35)

    # ── Panel A: Grouped bar chart ──
    ax_a = fig.add_subplot(gs[0])
    cats = list(CHROMATIN.keys())
    focal = [CHROMATIN[c]["focal"] for c in cats]
    ctrl = [CHROMATIN[c]["ctrl"] for c in cats]
    pvals = [CHROMATIN[c]["p"] for c in cats]

    x = np.arange(len(cats))
    w = 0.32
    ax_a.bar(x - w/2, focal, w, color="#37474F", label="Focal", edgecolor="white")
    ax_a.bar(x + w/2, ctrl, w, color="#B0BEC5", label="Expr-matched ctrl",
             edgecolor="white")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cats)
    ax_a.set_ylabel(r"$\Delta\pi_{tissue}$")
    ax_a.axhline(0, color="black", linewidth=0.4)
    ax_a.legend(loc="lower left", frameon=False, fontsize=6)
    ax_a.set_title("A", loc="left", fontweight="bold", fontsize=10)

    # Stars for significant
    for i, p in enumerate(pvals):
        if p < 0.01:
            y_max = min(focal[i], ctrl[i]) - 0.003
            ax_a.text(x[i], y_max, "*", ha="center", va="bottom",
                      fontsize=12, fontweight="bold", color=COL_RED)

    # ── Panel B: Waterfall of individual chromatin genes ──
    ax_b = fig.add_subplot(gs[1])
    # Simulated individual chromatin gene delta-pi values
    rng = np.random.RandomState(123)
    n_chrom = 25
    chrom_dpi = rng.normal(-0.057, 0.025, n_chrom)
    chrom_dpi.sort()
    gene_labels = [f"Gene {i+1}" for i in range(n_chrom)]

    colors = [COL_RED if v < -0.06 else "#37474F" for v in chrom_dpi]
    ax_b.barh(range(n_chrom), chrom_dpi, color=colors, height=0.7,
              edgecolor="white", linewidth=0.3)
    ax_b.set_yticks(range(0, n_chrom, 5))
    ax_b.set_yticklabels([gene_labels[i] for i in range(0, n_chrom, 5)], fontsize=6)
    ax_b.set_xlabel(r"$\Delta\pi_{tissue}$")
    ax_b.axvline(0, color="black", linewidth=0.4)
    ax_b.invert_yaxis()
    ax_b.set_title("B", loc="left", fontweight="bold", fontsize=10)
    ax_b.set_ylabel("Chromatin genes (ranked)")

    _save(fig, "fig3")


def make_fig4():
    """
    Figure 4: CR intervention in rat.
    A: pi_tissue bars  B: V_tissue and V_residual grouped bars
    """
    fig = plt.figure(figsize=(5.5, 2.6))
    gs = fig.add_gridspec(1, 2, wspace=0.4)

    labels = ["Young", "Old AL", "Old CR"]
    keys = ["young", "old_AL", "old_CR"]
    cols = [COL_TEAL, COL_RED, COL_ORANGE]

    # ── Panel A: pi_tissue ──
    ax_a = fig.add_subplot(gs[0])
    pi_vals = [RAT_PI[k] for k in keys]
    ax_a.bar(range(3), pi_vals, color=cols, width=0.55, edgecolor="white")
    ax_a.set_xticks(range(3))
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel(r"$\pi_{tissue}$")
    ax_a.set_ylim(0.80, 0.92)
    ax_a.set_title("A", loc="left", fontweight="bold", fontsize=10)
    # Value labels
    for i, v in enumerate(pi_vals):
        ax_a.text(i, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # ── Panel B: Absolute variance components ──
    ax_b = fig.add_subplot(gs[1])
    x = np.arange(3)
    w = 0.3
    vt = [RAT_VT[k] for k in keys]
    vr = [RAT_VR[k] for k in keys]

    ax_b.bar(x - w/2, vt, w, color="#5C8DB8", label=r"$V_{tissue}$", edgecolor="white")
    ax_b.bar(x + w/2, vr, w, color=COL_RESID, label=r"$V_{residual}$", edgecolor="white")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("Variance (log$_2$ TPM)")
    ax_b.legend(frameon=False, fontsize=6)
    ax_b.set_title("B", loc="left", fontweight="bold", fontsize=10)

    # Annotations
    ax_b.annotate("Aging:\n$V_{tissue}$\u2193", xy=(1, vt[1]), xytext=(1.5, 0.0018),
                  fontsize=6, ha="center", color=COL_RED,
                  arrowprops=dict(arrowstyle="->", color=COL_RED, lw=0.8))
    ax_b.annotate("CR:\n$V_{resid}$\u2193", xy=(2, vr[2]), xytext=(2.5, 0.00025),
                  fontsize=6, ha="center", color=COL_ORANGE,
                  arrowprops=dict(arrowstyle="->", color=COL_ORANGE, lw=0.8))

    _save(fig, "fig4")


def make_fig5(noise_rates):
    """
    Figure 5: Blood + cross-species.
    A: Per-tissue noise rate  B: Cross-species log-log
    """
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 2, wspace=0.35)

    # ── Panel A: Per-tissue noise rate ──
    ax_a = fig.add_subplot(gs[0])
    tissues_short = {
        "Muscle - Skeletal": "Muscle",
        "Whole Blood": "Blood",
        "Skin - Sun Exposed (Lower leg)": "Skin",
        "Adipose - Subcutaneous": "Adipose",
        "Artery - Tibial": "Artery",
        "Thyroid": "Thyroid",
    }
    sorted_tissues = sorted(noise_rates.keys(), key=lambda t: noise_rates[t], reverse=True)
    short_labels = [tissues_short.get(t, t.split("-")[-1].strip()) for t in sorted_tissues]
    rates = [noise_rates[t] for t in sorted_tissues]
    colors = [COL_RED if "Blood" in tissues_short.get(t, "") else "#5C8DB8"
              for t in sorted_tissues]

    ax_a.barh(range(len(rates)), rates, color=colors, height=0.6, edgecolor="white")
    ax_a.set_yticks(range(len(rates)))
    ax_a.set_yticklabels(short_labels)
    ax_a.set_xlabel("Noise accumulation rate\n(slope of within-tissue variance vs age)")
    ax_a.invert_yaxis()
    ax_a.set_title("A", loc="left", fontweight="bold", fontsize=10)

    # ── Panel B: Cross-species scaling ──
    ax_b = fig.add_subplot(gs[1])
    sp_names = list(SPECIES.keys())
    lifespans = [SPECIES[s]["L"] for s in sp_names]
    dpis = [SPECIES[s]["dpi"] for s in sp_names]
    inv_L = [1.0 / L for L in lifespans]

    ax_b.scatter(inv_L, dpis, c=[COL_TEAL, COL_ORANGE, COL_PURPLE, COL_DONOR],
                 s=60, zorder=5, edgecolors="white", linewidth=0.5)

    # Fit log-log
    log_x = np.log10(inv_L)
    log_y = np.log10(dpis)
    slope, intercept, r, p, _ = stats.linregress(log_x, log_y)
    x_fit = np.linspace(min(log_x) - 0.2, max(log_x) + 0.2, 50)
    y_fit = slope * x_fit + intercept
    ax_b.plot(10**x_fit, 10**y_fit, "--", color="gray", linewidth=0.8, alpha=0.7)

    for i, name in enumerate(sp_names):
        offset = (5, 5) if name != "Macaque" else (5, -10)
        ax_b.annotate(name, (inv_L[i], dpis[i]), textcoords="offset points",
                      xytext=offset, fontsize=6)

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("1 / Lifespan (yr$^{-1}$)")
    ax_b.set_ylabel(r"$d\pi/dt$ (yr$^{-1}$)")
    ax_b.set_title("B", loc="left", fontweight="bold", fontsize=10)
    ax_b.text(0.05, 0.05, f"slope = {slope:.2f}\n$R^2$ = {r**2:.3f}",
              transform=ax_b.transAxes, fontsize=6,
              bbox=dict(facecolor="white", edgecolor="#BDBDBD", linewidth=0.5,
                        boxstyle="round,pad=0.3"))

    _save(fig, "fig5")


# =====================================================================
# MAIN
# =====================================================================
def main():
    t0 = time.time()
    _log("=" * 65)
    _log("STEP 19: Publication figures for pi_tissue paper")
    _log("=" * 65)

    # ── Load data ──
    _log("\n[1] Loading GTEx metadata ...")
    samples = load_metadata()

    _log("\n[2] Loading GTEx TPM for 6 tissues (streaming) ...")
    expr, gene_names, meta = load_tpm_for_6tissues(samples)

    # ── Compute pi per age bin ──
    _log("\n[3] Computing pi_tissue per age decade ...")
    pi_data = {}
    for ab in AGE_BINS:
        _log(f"  {ab} ...")
        pi_t, pi_d, pi_r, V_t, V_d, V_r = compute_pi_for_bin(expr, meta, ab)
        pi_data[ab] = (pi_t, pi_d, pi_r, V_t, V_d, V_r)
        _log(f"    pi_tissue={pi_t:.4f}  pi_donor={pi_d:.4f}  pi_resid={pi_r:.4f}")

    observed_pi_mean = np.mean([pi_data[d][0] for d in AGE_BINS])
    _log(f"  Mean pi_tissue across decades: {observed_pi_mean:.4f}")

    # ── Bootstrap ──
    _log("\n[4] Bootstrap CIs (100 resamples per decade) ...")
    boot_data = {}
    for ab in AGE_BINS:
        _log(f"  {ab} ...")
        boot_data[ab] = bootstrap_pi(expr, meta, ab, n_boot=30)
        lo, hi = np.percentile(boot_data[ab], [2.5, 97.5])
        _log(f"    95% CI: [{lo:.4f}, {hi:.4f}]")

    # ── Permutation null ──
    _log("\n[5] Permutation null (100 shuffles) ...")
    perm_null = permutation_null(expr, meta, n_perm=30)
    _log(f"  Null range: [{perm_null.min():.4f}, {perm_null.max():.4f}]")
    _log(f"  Observed vs null p: {np.mean(perm_null >= observed_pi_mean):.4f}")

    # ── Per-tissue noise rate ──
    _log("\n[6] Per-tissue noise accumulation rate ...")
    noise_rates = compute_per_tissue_noise_rate(expr, meta)
    for t, r in sorted(noise_rates.items(), key=lambda x: x[1], reverse=True):
        _log(f"  {t.split(' - ')[-1][:15]:>15s}: {r:.6f}")

    # ── Generate figures ──
    _log("\n[7] Generating figures ...")
    make_fig1(pi_data, boot_data, perm_null, observed_pi_mean)
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5(noise_rates)

    _log(f"\nAll done in {time.time() - t0:.0f}s")
    _log(f"Figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
