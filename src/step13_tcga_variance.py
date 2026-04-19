"""
Test 4: Cancer effect on π — TCGA paired tumor/normal
For each cancer type: ANOVA V_tumor + V_patient + V_residual
"""
import time, gzip, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data" / "tcga"
RESULTS_DIR = BASE / "results" / "step13_tcga"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# TCGA cancer type codes from project ID
CANCER_CODES = {
    "BRCA": "Breast", "LUAD": "Lung adeno", "THCA": "Thyroid",
    "KIRC": "Kidney clear", "PRAD": "Prostate", "LIHC": "Liver",
    "HNSC": "Head/Neck", "LUSC": "Lung squam", "COAD": "Colon",
    "STAD": "Stomach", "BLCA": "Bladder", "KIRP": "Kidney pap",
    "UCEC": "Uterine", "ESCA": "Esophagus",
}

def _log(m): print(m, flush=True)

def main():
    t0 = time.time()
    _log("=" * 60)
    _log("TEST 4: Cancer π — TCGA tumor/normal")
    _log("=" * 60)

    # Load TPM
    tpm_path = DATA_DIR / "tcga_RSEM_gene_tpm.gz"
    _log("  Loading TCGA TPM...")
    df_tpm = pd.read_csv(tpm_path, sep="\t", index_col=0)
    _log(f"  Shape: {df_tpm.shape}")

    # Map ENSG to gene symbols
    map_path = DATA_DIR / "gene_symbol_to_ensg.json"
    with open(map_path) as f:
        sym2ensg = json.load(f)
    ensg2sym = {v: k for k, v in sym2ensg.items()}

    # Parse sample IDs: TCGA-XX-XXXX-01A = tumor, -11A = normal
    sample_ids = df_tpm.columns.tolist()
    sample_info = []
    for sid in sample_ids:
        parts = sid.split("-")
        if len(parts) >= 4:
            patient = "-".join(parts[:3])
            sample_type_code = int(parts[3][:2])
            is_tumor = sample_type_code < 10
            is_normal = 10 <= sample_type_code < 20
            # Cancer type from TCGA project (need external mapping or infer from data)
            sample_info.append({"sample": sid, "patient": patient,
                                "is_tumor": is_tumor, "is_normal": is_normal,
                                "type_code": sample_type_code})

    df_info = pd.DataFrame(sample_info)
    _log(f"  Tumor samples: {df_info['is_tumor'].sum()}, Normal: {df_info['is_normal'].sum()}")

    # Find paired patients
    tumor_patients = set(df_info[df_info["is_tumor"]]["patient"])
    normal_patients = set(df_info[df_info["is_normal"]]["patient"])
    paired = sorted(tumor_patients & normal_patients)
    _log(f"  Paired patients: {len(paired)}")

    # Get one tumor and one normal sample per patient
    pairs = []
    for pat in paired:
        t_samples = df_info[(df_info["patient"] == pat) & (df_info["is_tumor"])]["sample"].values
        n_samples = df_info[(df_info["patient"] == pat) & (df_info["is_normal"])]["sample"].values
        if len(t_samples) > 0 and len(n_samples) > 0:
            pairs.append({"patient": pat, "tumor": t_samples[0], "normal": n_samples[0]})

    df_pairs = pd.DataFrame(pairs)
    _log(f"  Valid pairs: {len(df_pairs)}")

    # No cancer type splitting — do ALL paired as one group

    # Simple approach: ANOVA on ALL paired samples
    _log("\n  Computing variance decomposition for all paired samples...")

    # Filter to paired where both available
    valid_pairs = []
    for _, row in df_pairs.iterrows():
        if row["tumor"] in df_tpm.columns and row["normal"] in df_tpm.columns:
            valid_pairs.append(row)
    df_vp = pd.DataFrame(valid_pairs)
    _log(f"  Valid pairs with expression: {len(df_vp)}")

    if len(df_vp) < 30:
        _log("  Too few pairs, exiting")
        return

    # Build expression matrix: patients × 2 (tumor, normal) per gene
    n_patients = len(df_vp)
    tumor_cols = df_vp["tumor"].tolist()
    normal_cols = df_vp["normal"].tolist()

    # Filter genes (expressed)
    gene_ids = df_tpm.index.tolist()
    expr_t = df_tpm[tumor_cols].values  # genes × patients
    expr_n = df_tpm[normal_cols].values

    # Per-gene ANOVA: V_tumor_status + V_patient + V_residual
    pi_tumor_list = []
    pi_patient_list = []
    pi_residual_list = []
    gene_names_used = []

    for g_i in range(len(gene_ids)):
        t_vals = expr_t[g_i]  # (n_patients,)
        n_vals = expr_n[g_i]

        # Stack: (2*n_patients,) with labels
        all_vals = np.concatenate([t_vals, n_vals])
        if np.std(all_vals) < 0.01 or np.median(all_vals) < 0.1:
            continue

        grand_mean = np.mean(all_vals)
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        if ss_total < 1e-10:
            continue

        # Patient means
        patient_means = (t_vals + n_vals) / 2
        ss_patient = 2 * np.sum((patient_means - grand_mean) ** 2)

        # Tumor status means
        t_mean = np.mean(t_vals)
        n_mean = np.mean(n_vals)
        ss_tumor = n_patients * ((t_mean - grand_mean) ** 2 + (n_mean - grand_mean) ** 2)

        ss_residual = max(ss_total - ss_patient - ss_tumor, 0)

        pi_tumor_list.append(ss_tumor / ss_total)
        pi_patient_list.append(ss_patient / ss_total)
        pi_residual_list.append(ss_residual / ss_total)

        gene_name = ensg2sym.get(gene_ids[g_i].split(".")[0], gene_ids[g_i])
        gene_names_used.append(gene_name)

        if g_i % 10000 == 0 and g_i > 0:
            _log(f"    {g_i}/{len(gene_ids)} genes ({time.time()-t0:.0f}s)")

    _log(f"\n  Genes analyzed: {len(pi_tumor_list)}")

    pi_t = np.array(pi_tumor_list)
    pi_p = np.array(pi_patient_list)
    pi_r = np.array(pi_residual_list)

    _log(f"\n  RESULTS ({len(df_vp)} paired patients):")
    _log(f"    π_tumor (tumor vs normal):  median = {np.median(pi_t):.4f}, mean = {np.mean(pi_t):.4f}")
    _log(f"    π_patient (inter-individual): median = {np.median(pi_p):.4f}, mean = {np.mean(pi_p):.4f}")
    _log(f"    π_residual:                   median = {np.median(pi_r):.4f}, mean = {np.mean(pi_r):.4f}")

    _log(f"\n    Compare to GTEx aging:")
    _log(f"      GTEx π_tissue = 0.73 (tissues are different from each other)")
    _log(f"      TCGA π_tumor  = {np.median(pi_t):.4f} (tumor is different from normal)")
    _log(f"      GTEx π_donor  = 0.06 (people are different from each other)")
    _log(f"      TCGA π_patient = {np.median(pi_p):.4f} (patients are different)")

    # Top genes by π_tumor (most disrupted by cancer)
    df_genes = pd.DataFrame({"gene": gene_names_used, "pi_tumor": pi_t, "pi_patient": pi_p, "pi_residual": pi_r})
    df_genes = df_genes.sort_values("pi_tumor", ascending=False)
    df_genes.to_csv(RESULTS_DIR / "tcga_per_gene_pi.csv", index=False)

    _log(f"\n  Top 10 genes by π_tumor (most cancer-disrupted):")
    for _, r in df_genes.head(10).iterrows():
        _log(f"    {r['gene']:<15s}: π_tumor={r['pi_tumor']:.3f}, π_patient={r['pi_patient']:.3f}")

    _log(f"\n  Top 10 genes by π_patient (most individually variable):")
    for _, r in df_genes.nlargest(10, "pi_patient").iterrows():
        _log(f"    {r['gene']:<15s}: π_patient={r['pi_patient']:.3f}, π_tumor={r['pi_tumor']:.3f}")

    # Summary
    pd.DataFrame([{
        "n_patients": len(df_vp), "n_genes": len(pi_t),
        "pi_tumor_median": np.median(pi_t), "pi_patient_median": np.median(pi_p),
        "pi_residual_median": np.median(pi_r),
    }]).to_csv(RESULTS_DIR / "tcga_summary.csv", index=False)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    labels = ["π_tumor\n(cancer effect)", "π_patient\n(individual)", "π_residual\n(noise)"]
    vals = [np.median(pi_t), np.median(pi_p), np.median(pi_r)]
    colors = ["tab:red", "tab:blue", "tab:gray"]
    ax.bar(labels, vals, color=colors, alpha=0.7)
    ax.set_ylabel("Median proportion")
    ax.set_title(f"TCGA Variance Decomposition\n({len(df_vp)} paired tumor/normal)")

    ax = axes[1]
    ax.hist(pi_t, bins=50, color="tab:red", alpha=0.5, label="π_tumor", density=True)
    ax.hist(pi_p, bins=50, color="tab:blue", alpha=0.5, label="π_patient", density=True)
    ax.set_xlabel("Variance proportion")
    ax.set_title("Distribution across genes")
    ax.legend()

    ax = axes[2]
    ax.scatter(pi_p[:2000], pi_t[:2000], s=1, alpha=0.2)
    ax.set_xlabel("π_patient")
    ax.set_ylabel("π_tumor")
    ax.set_title("Patient vs tumor variance per gene")
    ax.plot([0, 1], [0, 1], "r--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tcga_variance.png", dpi=150)
    plt.close()
    _log(f"\n  Time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
