# Branch Edit Summary — EXP-003-Add-Data-Drift-Tracking

## Summary
Added a `save_data_stats()` function to capture descriptive statistics from both the raw and clean DataFrames on every run. Stats are written to JSON files in `data/` and serve as a baseline for data drift tracking over time. Also added `docs/template.md` and `docs/checklist.md` as project documentation scaffolding.

---

## Files Modified

### `src/data_preprocessing.py`
- **Added imports:** `os`, `json`, `datetime`
- **Added `save_data_stats(df, path, label)`:**
  - Computes per-column stats: `dtype`, `null_count`, `null_pct`, `unique_count`
  - Numeric columns: additionally computes `mean`, `std`, `min`, `p25`, `p50`, `p75`, `max`
  - Categorical columns: additionally captures `top_value` (most frequent)
  - Writes a timestamped JSON file to `path` (creates parent dirs if needed)
  - Justified: enables run-to-run comparison to detect data drift before training

### `main.py`
- **Added import:** `save_data_stats` from `src.data_preprocessing`
- **Added call after load/sample:** `save_data_stats(df, 'data/stats_raw.json', label='raw')`
  - Justified: captures distribution of input data before any cleaning
- **Added call after preprocess:** `save_data_stats(data['df'], 'data/stats_clean.json', label='clean')`
  - Justified: captures distribution after cleaning to compare with raw and prior runs

## Files Added

### `docs/template.md`
- New project README template pre-filled with this project's business case, folder structure, pipeline overview, and API overview sections.

### `docs/checklist.md`
- New project checklist tracking which ML best-practice items are implemented (✅) vs outstanding ([ ]).

---

## What Was Added
| Item | Purpose |
|---|---|
| `save_data_stats()` | Captures per-column statistics for drift tracking |
| `data/stats_raw.json` (generated) | Snapshot of raw data distribution each run |
| `data/stats_clean.json` (generated) | Snapshot of clean data distribution each run |
| `docs/template.md` | Reusable README template for this project |
| `docs/checklist.md` | ML best-practices checklist with completion status |
