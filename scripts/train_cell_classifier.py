"""Train a lightweight cell-role classifier on CTC + ENTRANT data.

Data sources:
  - CTC format (CIUS, SAUS): dict of tables with string_matrix, label_matrix, format_matrix
  - ENTRANT format (18-K, 10-KT, 485BPOS, 497, S-1): list of tables with Cells array

Train/eval split: CIUS + ENTRANT subsets = train, SAUS = eval (held-out).

Label mapping:
  CTC:     0=metadata→header, 3=left_attr→header, 4=top_attr→header, 2=data, 5=derived→data
  ENTRANT: is_header=true→header, is_attribute=true→header, else→data
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# CTC label → our 2-class scheme
_CTC_LABEL_MAP = {
    -1: -1,  0: 0, 1: -1, 2: 1, 3: 0, 4: 0, 5: 1,
}

# format_matrix field order in CTC (11 fields)
_ENTRANT_FMT_KEYS = ["FB", "I", "FC", "BC", "LB", "TB", "BB", "RB", "HA", "VA", "DT"]

_CAP = 20.0


def _is_numeric(s: str) -> bool:
    if not s:
        return False
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    if s.startswith("-"):
        s = s[1:]
    return s.replace(".", "", 1).isdigit()


def _make_features(
    dist_above: int,
    dist_left: int,
    val: str,
    fmt: list[int],
    row_num_frac: float = 0.0,
    col_num_frac: float = 0.0,
    row_text_frac: float = 0.0,
    col_text_frac: float = 0.0,
) -> list[float]:
    da = min(float(dist_above), _CAP)
    dl = min(float(dist_left), _CAP)
    features = [
        da, dl,
        da / _CAP, dl / _CAP,
        float(da == 0), float(dl == 0), float(da < 2),
        float(len(val) == 0),
        float(_is_numeric(val)),
        float(len(val) > 50),
        row_num_frac, col_num_frac,
    ]
    features.extend(fmt[:11] if len(fmt) >= 11 else fmt + [0] * (11 - len(fmt)))
    features.extend([row_text_frac, col_text_frac])
    return features


def _precompute_dist(sm: list[list[str]]) -> tuple[list[list[int]], list[list[int]]]:
    """Compute dist_above and dist_left matrices from a string matrix."""
    n_rows = len(sm)
    n_cols = len(sm[0]) if sm else 0
    dist_above = [[0] * n_cols for _ in range(n_rows)]
    dist_left = [[0] * n_cols for _ in range(n_rows)]
    for r in range(n_rows):
        for c in range(n_cols):
            if r == 0 or sm[r - 1][c] == "":
                dist_above[r][c] = 0
            else:
                dist_above[r][c] = dist_above[r - 1][c] + 1
            if c == 0 or sm[r][c - 1] == "":
                dist_left[r][c] = 0
            else:
                dist_left[r][c] = dist_left[r][c - 1] + 1
    return dist_above, dist_left


def _precompute_fracs(
    sm: list[list[str]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Compute row/col numeric and text fractions from a string matrix.

    Returns (row_nf, col_nf, row_tf, col_tf) where:
    - row_nf[r]  = fraction of numeric cells in row r (non-empty only)
    - col_nf[c]  = fraction of numeric cells in col c (non-empty only)
    - row_tf[r]  = fraction of non-numeric non-empty cells in row r
    - col_tf[c]  = fraction of non-numeric non-empty cells in col c
    """
    n_rows = len(sm)
    n_cols = len(sm[0]) if sm else 0
    row_nf, row_tf = [], []
    for r in range(n_rows):
        ne = [v for v in sm[r] if v != ""]
        num = sum(1 for v in ne if _is_numeric(v))
        n = len(ne)
        row_nf.append(num / n if n else 0.0)
        row_tf.append((n - num) / n if n else 0.0)
    col_nf, col_tf = [], []
    for c in range(n_cols):
        col_vals = [sm[r][c] for r in range(n_rows)]
        ne = [v for v in col_vals if v != ""]
        num = sum(1 for v in ne if _is_numeric(v))
        n = len(ne)
        col_nf.append(num / n if n else 0.0)
        col_tf.append((n - num) / n if n else 0.0)
    return row_nf, col_nf, row_tf, col_tf


def extract_from_ctc(table: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from a CTC-format table."""
    sm = table["string_matrix"]
    lm = table["label_matrix"]
    fm = table.get("format_matrix", [])
    n_rows = len(sm)
    n_cols = len(sm[0]) if sm else 0
    if n_rows == 0 or n_cols == 0:
        return np.empty((0, 0)), np.empty((0,))

    dist_above, dist_left = _precompute_dist(sm)
    row_nf, col_nf, row_tf, col_tf = _precompute_fracs(sm)

    rows_X, rows_y = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            label = _CTC_LABEL_MAP.get(lm[r][c], -1)
            if label == -1:
                continue
            val = sm[r][c] if sm[r][c] else ""
            fmt = fm[r][c] if fm and r < len(fm) and c < len(fm[r]) else [0] * 11
            rows_X.append(_make_features(
                dist_above[r][c], dist_left[r][c], val, fmt,
                row_nf[r], col_nf[c], row_tf[r], col_tf[c],
            ))
            rows_y.append(label)
    if not rows_X:
        return np.empty((0, 0)), np.empty((0,))
    return np.array(rows_X, dtype=np.float32), np.array(rows_y, dtype=np.int8)


def extract_from_entrant(table: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from an ENTRANT-format table."""
    cells = table.get("Cells")
    if not cells:
        return np.empty((0, 0)), np.empty((0,))
    n_rows = len(cells)
    n_cols = len(cells[0]) if cells else 0
    if n_rows == 0 or n_cols == 0:
        return np.empty((0, 0)), np.empty((0,))

    # Build string matrix
    sm = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            cell = cells[r][c]
            val = cell.get("T") or cell.get("V") or ""
            if val == "None":
                val = ""
            row.append(val)
        sm.append(row)

    dist_above, dist_left = _precompute_dist(sm)
    row_nf, col_nf, row_tf, col_tf = _precompute_fracs(sm)

    rows_X, rows_y = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            cell = cells[r][c]
            if cell.get("is_header"):
                label = 0
            elif cell.get("is_attribute"):
                label = 0
            else:
                label = 1
            val = sm[r][c]
            if not val and label == 1:
                continue
            fmt = [int(cell.get(k, 0) or 0) for k in _ENTRANT_FMT_KEYS]
            rows_X.append(_make_features(
                dist_above[r][c], dist_left[r][c], val, fmt,
                row_nf[r], col_nf[c], row_tf[r], col_tf[c],
            ))
            rows_y.append(label)
    if not rows_X:
        return np.empty((0, 0)), np.empty((0,))
    return np.array(rows_X, dtype=np.float32), np.array(rows_y, dtype=np.int8)



FEATURE_NAMES = [
    "dist_above", "dist_left", "dist_above_norm", "dist_left_norm",
    "is_first_row", "is_first_col", "is_in_top_2",
    "is_empty", "is_numeric", "is_long_text",
    "row_numeric_frac", "col_numeric_frac",
    "fmt_bold", "fmt_italic", "fmt_font_color", "fmt_bg_color",
    "fmt_left_border", "fmt_top_border", "fmt_bottom_border", "fmt_right_border",
    "fmt_h_align", "fmt_v_align", "fmt_data_type",
    "row_text_frac", "col_text_frac",
]


def load_ctc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    all_X, all_y = [], []
    with open(path) as f:
        data = json.load(f)
    for table in data.values():
        X, y = extract_from_ctc(table)
        if X.size > 0:
            all_X.append(X)
            all_y.append(y)
    return np.vstack(all_X), np.concatenate(all_y)


def load_entrant_dir(base_dir: Path, max_files: int = 0) -> tuple[np.ndarray, np.ndarray]:
    all_X, all_y = [], []
    json_files = sorted(base_dir.rglob("*.json"))
    if max_files > 0:
        json_files = json_files[:max_files]
    for jp in json_files:
        try:
            with open(jp, encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict):
            data = [data]
        for table in data:
            X, y = extract_from_entrant(table)
            if X.size > 0:
                all_X.append(X)
                all_y.append(y)
    if not all_X:
        return np.empty((0, 0)), np.empty((0,))
    return np.vstack(all_X), np.concatenate(all_y)


def main():
    ctc_dir = Path("/tmp/entrant/output_CTC")
    entrant_dir = Path("/tmp/entrant_data")

    # --- Load train data ---
    print("Loading train data...")
    train_parts: list[tuple[str, np.ndarray, np.ndarray]] = []

    # CTC: CIUS
    cius_path = ctc_dir / "cius.json"
    if cius_path.exists():
        X, y = load_ctc(cius_path)
        train_parts.append(("CIUS", X, y))
        print(f"  CIUS:    {len(y):>8,d} samples (header={np.sum(y==0):,}, data={np.sum(y==1):,})")

    # ENTRANT subsets
    for subdir in ["18-K", "10-KT", "485BPOS", "497", "S-1"]:
        path = entrant_dir / subdir
        if path.exists():
            X, y = load_entrant_dir(path, max_files=0)
            if X.size > 0:
                train_parts.append((subdir, X, y))
                print(f"  {subdir:<8s} {len(y):>8,d} samples (header={np.sum(y==0):,}, data={np.sum(y==1):,})")

    if not train_parts:
        print("No training data found. Exiting.")
        return

    X_train = np.vstack([p[1] for p in train_parts])
    y_train = np.concatenate([p[2] for p in train_parts])
    print(f"  TOTAL:   {len(y_train):>8,d} samples (header={np.sum(y_train==0):,}, data={np.sum(y_train==1):,})")

    # --- Load eval data (SAUS, held-out) ---
    print("\nLoading eval data (SAUS)...")
    saus_path = ctc_dir / "saus.json"
    X_eval, y_eval = load_ctc(saus_path)
    print(f"  SAUS:    {len(y_eval):>8,d} samples (header={np.sum(y_eval==0):,}, data={np.sum(y_eval==1):,})")

    # --- Train ---
    print("\nTraining RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # --- Eval on held-out SAUS ---
    y_pred = clf.predict(X_eval)
    print("\n=== Eval on SAUS (held-out) ===")
    print(classification_report(y_eval, y_pred, target_names=["header", "data"]))

    # --- Train report ---
    y_pred_train = clf.predict(X_train)
    print("=== Train ===")
    print(classification_report(y_train, y_pred_train, target_names=["header", "data"]))

    # --- Feature importances ---
    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
        if imp > 0.005:
            print(f"  {name}: {imp:.3f}")

    # --- Retrain on all data ---
    print("\nRetraining on ALL data for final model...")
    X_all = np.vstack([X_train, X_eval])
    y_all = np.concatenate([y_train, y_eval])
    clf.fit(X_all, y_all)

    out_path = Path(__file__).resolve().parent.parent / "src" / "sheet_call_tree" / "cell_classifier.joblib"
    joblib.dump(clf, out_path)
    print(f"Model saved to {out_path}")
    print(f"Model size: {out_path.stat().st_size / 1024:.1f} KB")
    print(f"Total training samples: {len(y_all):,}")


if __name__ == "__main__":
    main()
