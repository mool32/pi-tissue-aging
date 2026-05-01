# Transcriptomic noise accumulates within tissue identity across human aging

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19944444.svg)](https://zenodo.org/records/19944444)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC-BY 4.0](https://img.shields.io/badge/Data%20%26%20Manuscript-CC--BY%204.0-lightgrey.svg)](LICENSE)

A systemic signature of mammalian aging, distinct from cell-composition drift —
analysis pipeline, intermediate data, figures, and manuscript.

**Spiro T. (2026)** *Transcriptomic noise accumulates within tissue identity across
human aging: a systemic signature distinct from cell-composition drift.* Preprint,
Vaika Inc. Zenodo DOI: [10.5281/zenodo.19944444](https://zenodo.org/records/19944444).

📄 **Preprint PDF**: [`paper/pi_tissue_paper_v4.pdf`](paper/pi_tissue_paper_v4.pdf)
🔬 **Main analysis script**: [`src/step10_variance_conservation.py`](src/step10_variance_conservation.py) (three-level ANOVA on GTEx v8)
📦 **Build PDF from scratch**: [`src/step23_v4_pdf.py`](src/step23_v4_pdf.py)

---

## Brief paper summary

We performed a three-level variance decomposition (V_total = V_tissue + V_donor + V_residual)
on bulk transcriptomes from 263 GTEx v8 donors (ages 20–79) with matched samples in six tissues,
combined with single-cell data from two Tabula Muris Senis platforms (mouse, ages 1–30 months),
the Calico rat caloric-restriction atlas, and a rhesus macaque cross-species atlas, to ask
*where* in variance space age-related transcriptomic change is located.

**Five findings:**

1. **Tissue identity is preserved** across forty years of human aging. π_tissue declines from
   0.764 to 0.733 (Δ = −0.031); π_donor is flat (Δ < 0.005); the change is absorbed almost
   entirely by within-tissue, within-donor residual variance (π_residual: 0.168 → 0.194).
2. **The signature is systemic noise, not selective accumulation.** Aging adds within-cell-type
   stochastic variance, not between-population shifts — directly adjudicating between
   senescent-cell-accumulation and generalized-regulatory-erosion views of aging.
3. **Per-tissue rates differ by an order of magnitude** and do not track a simple
   dividing-vs-post-mitotic axis. Whole blood (Δvar = +0.079, hematopoietic) and
   left-ventricular myocardium (+0.121, post-mitotic but fibrosis/infiltration-driven)
   accumulate noise fastest.
4. **Caloric restriction acts as a noise filter, not a structure restorer.** In rat bone
   marrow, CR reverses 87% of the aging π loss (95% CI 82–91%) by reducing V_residual,
   not by restoring V_tissue. The mechanistic distinction predicts a specific signature
   for CR-mimetics that differs from partial-reprogramming-style structure restorers.
5. **Cross-species scaling.** Across mouse, rat, macaque, and human (~30× lifespan range),
   π erosion rate scales inversely with maximum lifespan (α = −1.02 ± 0.24, R² = 0.90,
   Spearman ρ = −1.0).

The framework is complementary to DNA-methylation clocks: where methylation clocks track
chronological age with tissue-invariant CpG panels, transcriptomic residual variance
exposes tissue-specific rates of regulatory reserve erosion that methylation clocks
cannot see. Joint application is predicted to stratify individuals of the same biological
age into functionally distinct subgroups.

---

## Repository layout

```
pi_tissue_paper/
├── paper/
│   └── pi_tissue_paper_v4.pdf       # Final preprint PDF (open me)
├── manuscript/
│   ├── pi_tissue_paper_v4.md        # Source markdown
│   └── figures/                     # Publication figures (PNG + PDF, 300 DPI)
├── src/                             # Analysis pipeline (numbered step00–step23)
├── results/
│   ├── step10_variance_conservation/   # Main GTEx three-level ANOVA results
│   ├── step11_per_tissue/              # Per-tissue Δvar (6-tissue matched panel)
│   ├── step12_rat/                     # Calico rat CR atlas
│   ├── step15_three_tests/             # Scaling law + CR mechanism + gene leakage
│   ├── step18_vp/                      # variancePartition REML (R)
│   ├── step20_verification/            # Macaque + Kimmel independent validation
│   ├── step22_dividing/                # 15-tissue dividing vs post-mitotic
│   └── step39_sc_pi/                   # Single-cell π_tissue (TMS FACS, Droplet)
├── data/                            # NOT in git; obtain from upstream sources
├── README.md
├── LICENSE
└── .gitignore
```

---

## Pipeline (numerical order)

| Step | File | Purpose |
|------|------|---------|
| 10 | [`step10_variance_conservation.py`](src/step10_variance_conservation.py) | **Main analysis.** Three-level ANOVA on GTEx (V_tissue + V_donor + V_residual). |
| 11 | [`step11_per_tissue_decay.py`](src/step11_per_tissue_decay.py) | Per-tissue Δvar on 6 matched-donor tissues. |
| 12 | [`step12_rat_variance.py`](src/step12_rat_variance.py) | Calico rat CR atlas variance decomposition. |
| 15 | [`step15_three_tests.py`](src/step15_three_tests.py) | CR mechanism, scaling law, per-gene leakage. |
| 17 | [`step17_tier1_validation.py`](src/step17_tier1_validation.py) | Permutation nulls + validation battery. |
| 18 | [`step18_variancePartition.R`](src/step18_variancePartition.R) | REML variance components (R). |
| 20 | [`step20_verification.py`](src/step20_verification.py) | Macaque + Kimmel independent validation. |
| 22 | [`step22_dividing_vs_nondividing.py`](src/step22_dividing_vs_nondividing.py) | 15-tissue noise rates by turnover class. |
| 23 | [`src/step23_v4_pdf.py`](src/step23_v4_pdf.py) | Compile manuscript markdown to PDF. |

---

## Reproducing the analysis

### Data dependencies (not committed; obtain separately)

- **GTEx v8** — dbGaP phs000424.v8 (requires approval). Files needed:
  - `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz`
  - `GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt`
  - `GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt`
  Place in `data/gtex/`.
- **Tabula Muris Senis** (FACS and Droplet h5ad) — figshare doi:10.6084/m9.figshare.12654728.
  Place in `data/tms_facs/` and `data/tms_droplet/`.
- **Calico rat aging atlas** — GEO GSE141784. Place in `results/h010_rat_cr/data/`.
- **Macaque cross-species atlas** (Li & Kong 2025) — figshare 26963386.
  Place in `data/macaque/extracted/`.

### Environment

Python 3.12 and R 4.3+. Key packages:

```bash
pip install scanpy anndata numpy pandas scipy matplotlib statsmodels fpdf2
# R:  install variancePartition + BiocParallel from Bioconductor
```

### Run

After updating data paths in scripts as needed:

```bash
cd pi_tissue_paper
python src/step10_variance_conservation.py
python src/step11_per_tissue_decay.py
python src/step12_rat_variance.py
python src/step15_three_tests.py
python src/step17_tier1_validation.py
Rscript src/step18_variancePartition.R
python src/step20_verification.py
python src/step22_dividing_vs_nondividing.py
python src/step23_v4_pdf.py            # Build PDF
```

Total runtime ≈ 4 hours on a workstation; peak memory ≈ 16 GB (GTEx TPM matrix loaded).

---

## Citation

Please cite the preprint and the Zenodo release together. BibTeX:

```bibtex
@article{spiro2026pitissue,
  author  = {Spiro, Theodor},
  title   = {Transcriptomic noise accumulates within tissue identity across
             human aging: a systemic signature distinct from cell-composition drift},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.5281/zenodo.19944444},
  url     = {https://zenodo.org/records/19944444}
}
```

Please also cite the underlying data sources per their own policies (GTEx Consortium 2020;
Tabula Muris Consortium / Schaum et al. 2020; Zou et al. 2022; Li & Kong 2025).

---

## License

- **Code** (`src/`): MIT License — see [LICENSE](LICENSE)
- **Intermediate data** (`results/**/*.csv`): CC-BY 4.0, with upstream attribution honored
- **Figures** (`manuscript/figures/`): CC-BY 4.0
- **Manuscript** (`paper/pi_tissue_paper_v4.*`, `manuscript/pi_tissue_paper_v4.md`): CC-BY 4.0

---

## Contact

**Theodor Spiro**
Vaika, Inc., 1933 Sweet Rd., East Aurora, NY 14052-3016, USA
[tspiro@vaika.org](mailto:tspiro@vaika.org)

## Acknowledgements

Andrei V. Gudkov (Roswell Park Comprehensive Cancer Center) provided scientific input
on the biological framing, the dividing-vs-non-dividing architectural hypothesis, and
the positioning of this work in the systemic-noise versus selective-accumulation debate.
Katerina Andrianova (Vaika, Inc.) provided administrative support. All analyses were
performed by the author.
