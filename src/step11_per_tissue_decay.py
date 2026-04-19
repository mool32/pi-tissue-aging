"""
Test 1: Per-tissue π_tissue decay rate
Which tissues erode fastest? Correlate with turnover/immune.
"""
import time, gzip
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data" / "gtex"
RESULTS_DIR = BASE / "results" / "step11_per_tissue"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP6 = ["Muscle - Skeletal", "Whole Blood", "Skin - Sun Exposed (Lower leg)",
        "Adipose - Subcutaneous", "Artery - Tibial", "Thyroid"]
TURNOVER = {"Muscle - Skeletal": 1, "Whole Blood": 3, "Skin - Sun Exposed (Lower leg)": 2,
            "Adipose - Subcutaneous": 1, "Artery - Tibial": 1, "Thyroid": 2}
IMMUNE = {"Muscle - Skeletal": 1, "Whole Blood": 3, "Skin - Sun Exposed (Lower leg)": 2,
          "Adipose - Subcutaneous": 2, "Artery - Tibial": 1, "Thyroid": 2}

def _log(m): print(m, flush=True)

def main():
    t0 = time.time()
    _log("=" * 60)
    _log("TEST 1: Per-tissue π decay rate")
    _log("=" * 60)

    samples = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t")
    subjects = pd.read_csv(DATA_DIR / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep="\t")
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    samples = samples.merge(subjects, on="SUBJID", how="left")
    samples = samples[samples["SMAFRZE"] == "RNASEQ"].copy()
    samples["age_mid"] = samples["AGE"].str.split("-").apply(
        lambda x: (int(x[0]) + int(x[1])) / 2 if isinstance(x, list) and len(x) == 2 else np.nan)
    meta = samples.set_index("SAMPID")[["SUBJID", "SMTSD", "age_mid"]]

    tpm_path = DATA_DIR / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline()
        header = f.readline().strip().split("\t")
    sample_ids = header[2:]

    # Build tissue→age_bin→sample_indices
    tissue_age_idx = {}
    age_bins = {"20-39": (20, 40), "40-49": (40, 50), "50-59": (50, 60), "60-79": (60, 80)}
    for t in TOP6:
        tissue_age_idx[t] = {}
        for ab, (lo, hi) in age_bins.items():
            indices = []
            for i, sid in enumerate(sample_ids):
                if sid in meta.index:
                    r = meta.loc[sid]
                    if r["SMTSD"] == t and lo <= r["age_mid"] < hi:
                        indices.append(i)
            tissue_age_idx[t][ab] = np.array(indices)

    for t in TOP6:
        ts = t.split(" - ")[-1][:15]
        sizes = {ab: len(idx) for ab, idx in tissue_age_idx[t].items()}
        _log(f"  {ts}: {sizes}")

    # Per-tissue within-tissue ANOVA: V_donor + V_residual (using age bins as proxy)
    # Actually simpler: for each tissue, compute inter-donor variance per age bin
    results = []
    n_proc = 0
    with gzip.open(tpm_path, "rt") as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            parts = line.strip().split("\t", 2)
            gene = parts[1]
            vals = np.array(parts[2].split("\t"), dtype=np.float32)
            if np.median(vals) < 0.5:
                n_proc += 1; continue
            log_vals = np.log2(vals + 1)
            n_proc += 1
            if n_proc % 5000 == 0:
                _log(f"  {n_proc} genes ({time.time()-t0:.0f}s)")

            for t in TOP6:
                for ab in ["20-39", "60-79"]:
                    idx = tissue_age_idx[t][ab]
                    if len(idx) < 20: continue
                    v = log_vals[idx]
                    results.append({"gene": gene, "tissue": t, "age_bin": ab,
                                    "var": np.var(v), "mean": np.mean(v), "cv": np.std(v)/(np.mean(v)+0.01)})

    df = pd.DataFrame(results)

    # Compute per-tissue: mean variance young vs old
    _log("\n  Per-tissue variance change:")
    tissue_stats = []
    for t in TOP6:
        ts = t.split(" - ")[-1][:15]
        y = df[(df["tissue"] == t) & (df["age_bin"] == "20-39")]
        o = df[(df["tissue"] == t) & (df["age_bin"] == "60-79")]
        common = set(y["gene"]) & set(o["gene"])
        y = y[y["gene"].isin(common)].set_index("gene")
        o = o[o["gene"].isin(common)].set_index("gene")
        common = sorted(list(common))
        delta_var = o.loc[common, "var"].values - y.loc[common, "var"].values
        delta_cv = o.loc[common, "cv"].values - y.loc[common, "cv"].values
        med_dv = np.median(delta_var)
        pct_increase = (delta_var > 0).mean()
        w, p = stats.wilcoxon(delta_var)
        _log(f"  {ts:>15s}: median Δvar={med_dv:+.4f}, {pct_increase:.0%} increase, p={p:.2e}")
        tissue_stats.append({"tissue": ts, "full_tissue": t,
                             "median_delta_var": med_dv, "pct_var_increase": pct_increase, "p": p,
                             "median_delta_cv": np.median(delta_cv),
                             "turnover": TURNOVER[t], "immune": IMMUNE[t]})

    df_stats = pd.DataFrame(tissue_stats).sort_values("median_delta_var", ascending=False)
    df_stats.to_csv(RESULTS_DIR / "per_tissue_decay.csv", index=False)

    # Correlation with turnover
    rho_t, p_t = stats.spearmanr(df_stats["median_delta_var"], df_stats["turnover"])
    rho_i, p_i = stats.spearmanr(df_stats["median_delta_var"], df_stats["immune"])
    _log(f"\n  Correlation Δvar × turnover: ρ={rho_t:+.3f}, p={p_t:.3f}")
    _log(f"  Correlation Δvar × immune:   ρ={rho_i:+.3f}, p={p_i:.3f}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    ax.barh(range(len(df_stats)), df_stats["median_delta_var"].values,
            color=["tab:red" if "Blood" in t else "tab:blue" for t in df_stats["tissue"]])
    ax.set_yticks(range(len(df_stats)))
    ax.set_yticklabels(df_stats["tissue"])
    ax.set_xlabel("Median Δ variance (old - young)")
    ax.set_title("Per-tissue noise increase")
    ax.axvline(0, color="black", linewidth=0.5)

    ax = axes[1]
    ax.scatter(df_stats["turnover"], df_stats["median_delta_var"], s=80)
    for _, r in df_stats.iterrows():
        ax.annotate(r["tissue"][:8], (r["turnover"], r["median_delta_var"]), fontsize=7)
    ax.set_xlabel("Turnover score")
    ax.set_ylabel("Median Δ variance")
    ax.set_title(f"Turnover ρ={rho_t:+.2f}")

    ax = axes[2]
    ax.scatter(df_stats["immune"], df_stats["median_delta_var"], s=80)
    for _, r in df_stats.iterrows():
        ax.annotate(r["tissue"][:8], (r["immune"], r["median_delta_var"]), fontsize=7)
    ax.set_xlabel("Immune score")
    ax.set_ylabel("Median Δ variance")
    ax.set_title(f"Immune ρ={rho_i:+.2f}")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "per_tissue_decay.png", dpi=150)
    plt.close()
    _log(f"\n  Time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
