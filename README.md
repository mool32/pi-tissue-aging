# Transcriptomic noise accumulates within tissue identity across human aging

This repository contains the analysis pipeline, intermediate CSVs, and
figures supporting the preprint:

**Spiro T. (2026)** *Transcriptomic noise accumulates within tissue
identity across human aging: a systemic signature distinct from
cell-composition drift.* Preprint, Vaika Inc.
[bioRxiv DOI TBD on deposit]

## What's in here

```
pi_tissue_paper/
├── manuscript/
│   ├── pi_tissue_paper_v4.md    # Current draft
│   ├── pi_tissue_paper_v4.pdf   # Compiled PDF
│   ├── figures/                 # Publication figures (PNG + PDF, 300 DPI)
│   └── ...
├── src/                         # Analysis pipeline (step0 – step23)
├── results/
│   ├── step10_variance_conservation/   # Main GTEx variance decomposition
│   ├── step11_per_tissue/              # Per-tissue noise (6 tissues)
│   ├── step12_rat/                     # Rat CR atlas
│   ├── step15_three_tests/             # Scaling + CR + leakage
│   ├── step18_vp/                      # variancePartition REML
│   ├── step20_verification/            # Independent validation (Kimmel, macaque)
│   ├── step22_dividing/                # 15-tissue dividing vs post-mitotic
│   └── step39_sc_pi/                   # Single-cell π
├── data/                        # (NOT in git; see instructions below)
├── README.md                    # This file
└── .gitignore
```

## Pipeline

Scripts in `src/` run in numerical order. Key steps:

| Step | File | Purpose |
|------|------|---------|
| 10 | `step10_variance_conservation.py` | Three-level ANOVA on GTEx |
| 11 | `step11_per_tissue_decay.py` | Per-tissue Δvar on 6 tissues |
| 12 | `step12_rat_variance.py` | Calico rat CR atlas |
| 15 | `step15_three_tests.py` | CR mechanism, scaling law, gene leakage |
| 17 | `step17_tier1_validation.py` | Permutation nulls + validation |
| 18 | `step18_variancePartition.R` | REML variance components |
| 20 | `step20_verification.py` | Macaque + Kimmel independent validation |
| 22 | `step22_dividing_vs_nondividing.py` | **NEW v4**: 15-tissue noise rates by turnover class |
| 23 | `step23_v4_pdf.py` | Build v4 PDF |

## Reproducing the analysis

### Data (not committed; obtain separately)

- **GTEx v8**: dbGaP phs000424.v8 (requires approval). Files needed:
  `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz`,
  `GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt`,
  `GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt`.
  Place in `data/gtex/`.
- **Tabula Muris Senis**: figshare doi:10.6084/m9.figshare.12654728
  (FACS and Droplet h5ad). Place in `data/tms_facs/` and
  `data/tms_droplet/`.
- **Calico rat atlas**: GEO GSE141784. Place in `results/h010_rat_cr/data/`.
- **Macaque atlas**: figshare 26963386 (Li & Kong 2025). Place in
  `data/macaque/extracted/`.

### Environment

Python 3.12, R 4.3+. Key packages:

```
scanpy, anndata, numpy, pandas, scipy, matplotlib, statsmodels
variancePartition (R), BiocParallel (R)
fpdf2 (for PDF build)
```

### Run

Edit `src/*.py` data paths as needed, then:

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
python src/step23_v4_pdf.py  # Build PDF
```

Full pipeline runs in ~4 hours on a workstation; peak memory ~16 GB
(GTEx TPM matrix).

## Citation

If you use this code or data, please cite:

```bibtex
@article{spiro2026pitissue,
  author  = {Spiro, Theodor},
  title   = {Transcriptomic noise accumulates within tissue identity
             across human aging: a systemic signature distinct from
             cell-composition drift},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {[TBD on deposit]}
}
```

Please also cite the underlying data sources per their own policies
(GTEx, TMS, Calico, Li & Kong macaque atlas).

## License

- **Code** (`src/`): MIT License
- **Intermediate data** (`results/**/*.csv`): CC-BY 4.0, upstream
  citation honored
- **Figures** (`manuscript/figures/`): CC-BY 4.0
- **Manuscript** (`manuscript/pi_tissue_paper_v4.*`): CC-BY 4.0

## Contact

Theodor Spiro · Vaika, Inc., East Aurora, NY, USA ·
theospirin@gmail.com

## Acknowledgements

Andrei V. Gudkov provided scientific input on the biological framing,
positioning in the systemic-noise debate, and the dividing-vs-
non-dividing hypothesis. Computational analysis and manuscript
preparation were assisted by Claude (Anthropic). All scientific claims
and conclusions are the author's responsibility.
