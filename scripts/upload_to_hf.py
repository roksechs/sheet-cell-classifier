"""Upload cell_classifier.joblib to HuggingFace Hub.

Usage:
    .venv/bin/python scripts/upload_to_hf.py [--model PATH]

Requires a write token: huggingface-cli login  (or HF_TOKEN env var)
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "roksechs/sheet-cell-classifier"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "cell_classifier.joblib"


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to cell_classifier.joblib")
    args = parser.parse_args()

    model_path: Path = args.model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(REPO_ID, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="cell_classifier.joblib",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Upload {model_path.name} ({model_path.stat().st_size // 1024} KB)",
    )
    print(f"Uploaded {model_path.name} ({model_path.stat().st_size / 1024:.0f} KB) → {REPO_ID}")


if __name__ == "__main__":
    main()
