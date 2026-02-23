# RESULTS — EXP-003

**Branch:** `experiments/EXP-003-Add-Data-Drift-Tracking`  
**Goal:** Add data statistics capture for both raw and clean DataFrames on every run, enabling run-to-run comparison for data drift detection. Also add project documentation scaffolding.

## Changes Made
- Added `save_data_stats(df, path, label)` to `src/data_preprocessing.py`
- Called twice in `main.py`: once on raw sampled data, once on cleaned data output
- Stats written to `data/stats_raw.json` and `data/stats_clean.json` each run
- Added `docs/template.md` (project README template)
- Added `docs/checklist.md` (ML best-practices checklist)

## Model Results
_No model changes made in this experiment — RMSE unchanged from EXP-002._

**Test RMSE (unchanged):** 2,232.51
