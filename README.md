# sheet-cell-classifier

Training pipeline for the cell role classifier used by [sheet-call-tree](https://github.com/roksechs/sheet-call-tree).

Trains a RandomForest that predicts whether a spreadsheet cell is a **header** or **data** cell.
The trained model is published to [roksechs/sheet-cell-classifier](https://huggingface.co/roksechs/sheet-cell-classifier) on HuggingFace Hub.

## Scripts

| script | purpose |
|--------|---------|
| `scripts/train_cell_classifier.py` | Train on CIUS + ENTRANT, eval on SAUS (held-out) |
| `scripts/eval_deco.py` | Evaluate on DECO dataset (generalization, not used for training) |
| `scripts/bench_labeler.py` | Benchmark sheet-call-tree labeler on real files |
| `scripts/upload_to_hf.py` | Upload trained model to HuggingFace Hub |

## Setup

```bash
uv sync --dev
```

## Training

Requires data in:
- `/tmp/entrant/output_CTC/` — CTC format (CIUS, SAUS)
- `/tmp/entrant_data/` — ENTRANT format (18-K, 10-KT, 485BPOS, 497, S-1)

```bash
.venv/bin/python scripts/train_cell_classifier.py
```

Outputs `cell_classifier.joblib` in the current directory.

## Upload to HuggingFace

```bash
.venv/bin/python scripts/upload_to_hf.py
```

Requires `huggingface_hub` and a write token (`huggingface-cli login` or `HF_TOKEN` env var).
