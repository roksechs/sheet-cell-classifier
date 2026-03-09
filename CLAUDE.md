# sheet-cell-classifier

## Environment

- Use `uv` for all Python environment and package management
- Install dependencies: `uv sync --dev`
- Run commands in venv: `.venv/bin/python`

## Training data (not in repo)

- CTC format (CIUS, SAUS): `/tmp/entrant/output_CTC/{cius,saus}.json`
- ENTRANT format (18-K, 10-KT, 485BPOS, 497, S-1): `/tmp/entrant_data/<subset>/`
- DECO (eval only, license pending): `/tmp/deco/completed/`

## Workflow

```bash
# Train (CIUS + ENTRANT → model, SAUS = held-out eval)
.venv/bin/python scripts/train_cell_classifier.py

# Evaluate on DECO (generalization check, not used for training)
.venv/bin/python scripts/eval_deco.py /tmp/deco/completed

# Upload trained model to HuggingFace
.venv/bin/python scripts/upload_to_hf.py
```

## Feature set (25 features)

| idx | name | description |
|-----|------|-------------|
| 0-1 | dist_above, dist_left | gap-proximity (cap=20) |
| 2-3 | dist_above/20, dist_left/20 | normalized |
| 4-6 | is_first_row, is_first_col, is_in_top_2 | position flags |
| 7-9 | is_empty, is_numeric, is_long_text | value type |
| 10-11 | row_numeric_frac, col_numeric_frac | fraction of numeric cells |
| 12-22 | fmt_bold…fmt_data_type | 11 format fields from openpyxl |
| 23-24 | row_text_frac, col_text_frac | fraction of non-numeric non-empty cells |

## HuggingFace repo

`roksechs/sheet-cell-classifier` — model consumed by `sheet-call-tree >= 0.1.2`
