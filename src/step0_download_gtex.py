"""
Step 0: Download GTEx v8 data for Coupling Atlas

Downloads:
1. Gene TPM matrix (GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz)
2. Sample annotations (GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt)
3. Subject phenotypes (GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt)

All files are publicly available from the GTEx Portal / Google Cloud Storage.
"""

import os
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "gtex"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# GTEx v8 files on Google Cloud Storage
FILES = {
    # Gene TPM matrix (~1.5 GB compressed)
    "gene_tpm": {
        "url": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz",
        "filename": "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz",
    },
    # Gene read counts matrix
    "gene_counts": {
        "url": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz",
        "filename": "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz",
    },
    # Sample attributes (tissue, sample ID, ischemic time, etc.)
    "sample_attributes": {
        "url": "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
        "filename": "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
    },
    # Subject phenotypes (age, sex, death classification)
    "subject_phenotypes": {
        "url": "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
        "filename": "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
    },
}


def download_file(url, dest_path):
    """Download using wget with progress."""
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / 1e6
        print(f"  Already exists: {dest_path.name} ({size_mb:.1f} MB)")
        return True

    print(f"  Downloading: {dest_path.name}")
    print(f"  URL: {url}")

    try:
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(dest_path), url],
            capture_output=False,
            timeout=3600,
        )
        if result.returncode == 0:
            size_mb = dest_path.stat().st_size / 1e6
            print(f"  ✓ Done ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  ✗ wget failed (exit code {result.returncode})")
            # Try curl as fallback
            print(f"  Trying curl...")
            result = subprocess.run(
                ["curl", "-L", "-o", str(dest_path), "--progress-bar", url],
                capture_output=False,
                timeout=3600,
            )
            if result.returncode == 0:
                size_mb = dest_path.stat().st_size / 1e6
                print(f"  ✓ Done with curl ({size_mb:.1f} MB)")
                return True
            else:
                print(f"  ✗ curl also failed")
                return False
    except FileNotFoundError:
        # wget not available, try curl
        print(f"  wget not found, trying curl...")
        result = subprocess.run(
            ["curl", "-L", "-o", str(dest_path), "--progress-bar", url],
            capture_output=False,
            timeout=3600,
        )
        if result.returncode == 0:
            size_mb = dest_path.stat().st_size / 1e6
            print(f"  ✓ Done ({size_mb:.1f} MB)")
            return True
        return False


def main():
    print("=" * 70)
    print("GTEx v8 Data Download")
    print(f"Target directory: {DATA_DIR}")
    print("=" * 70)

    # Start with annotations (small, fast)
    for key in ["sample_attributes", "subject_phenotypes", "gene_tpm", "gene_counts"]:
        info = FILES[key]
        dest = DATA_DIR / info["filename"]
        print(f"\n[{key}]")
        success = download_file(info["url"], dest)
        if not success:
            print(f"  ⚠️ Failed to download {key}")
            if key in ["gene_tpm"]:
                print(f"  This is required. You may need to download manually from:")
                print(f"  https://gtexportal.org/home/datasets")

    # Quick validation
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)

    for key, info in FILES.items():
        path = DATA_DIR / info["filename"]
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"  ✓ {info['filename']} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {info['filename']} MISSING")

    # Quick peek at annotations
    sample_path = DATA_DIR / FILES["sample_attributes"]["filename"]
    subject_path = DATA_DIR / FILES["subject_phenotypes"]["filename"]

    if sample_path.exists():
        import pandas as pd
        samples = pd.read_csv(sample_path, sep="\t")
        print(f"\n  Sample annotations: {len(samples)} rows, columns: {list(samples.columns)}")
        if "SMTSD" in samples.columns:
            print(f"  Tissues: {samples['SMTSD'].nunique()}")
            print(f"  Top tissues: {samples['SMTSD'].value_counts().head(10).to_dict()}")

    if subject_path.exists():
        import pandas as pd
        subjects = pd.read_csv(subject_path, sep="\t")
        print(f"\n  Subject phenotypes: {len(subjects)} rows, columns: {list(subjects.columns)}")
        if "AGE" in subjects.columns:
            print(f"  Age bins: {subjects['AGE'].value_counts().sort_index().to_dict()}")
        if "SEX" in subjects.columns:
            print(f"  Sex: {subjects['SEX'].value_counts().to_dict()}")
        if "DTHHRDY" in subjects.columns:
            print(f"  Death classification: {subjects['DTHHRDY'].value_counts().sort_index().to_dict()}")

    print("\n" + "=" * 70)
    print("Done. Next: run step1_qc_gtex.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
