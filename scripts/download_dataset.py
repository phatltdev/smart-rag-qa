"""Download the Zalo AI 2021 Legal Text Retrieval dataset from Kaggle.

Usage:
    python scripts/download_dataset.py
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CORPUS_FILENAME,
    QNA_FILENAME,
    RAW_DIR,
    STOPWORDS_FILENAME,
    TEST_QUESTION_FILENAME,
)

DATASET = "hariwh0/zaloai2021-legal-text-retrieval"
REQUIRED_FILES = [
    CORPUS_FILENAME,
    QNA_FILENAME,
    TEST_QUESTION_FILENAME,
    STOPWORDS_FILENAME,
]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in REQUIRED_FILES if not (RAW_DIR / f).exists()]
    if not missing:
        print("[OK] All dataset files already present in data/raw/.")
        return

    print(f"[INFO] Missing files: {missing}")
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[ERROR] kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    print(f"[INFO] Downloading {DATASET} ...")
    kaggle.api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=False)

    # Kaggle saves the archive as "<dataset-slug>.zip" (e.g. zaloai2021-legal-text-retrieval.zip)
    zip_candidates = sorted(RAW_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not zip_candidates:
        print("[ERROR] No zip archive found after download.")
        sys.exit(1)
    zip_path = zip_candidates[-1]

    print("[INFO] Extracting required files ...")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if base in REQUIRED_FILES:
                target = RAW_DIR / base
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"      extracted {base}")

    zip_path.unlink(missing_ok=True)

    missing = [f for f in REQUIRED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        print(f"[ERROR] Still missing after download: {missing}")
        sys.exit(1)
    print("[OK] Dataset ready in data/raw/.")


if __name__ == "__main__":
    main()
