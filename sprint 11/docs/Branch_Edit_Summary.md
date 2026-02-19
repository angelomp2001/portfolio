
# EXP-011: Sprint 11 Readme & Code Refactor

**Status**: ✅ Success
**Date**: 2026-02-19

## Summary
Refactored `main.py` to clarify objectives, fix regression target logic, improve code quality, and generate clean, comparable outputs for all objectives. Organized project structure by moving documentation to `docs/`.

## Key Changes
1.  **Project Structure**:
    *   Created `docs/` folder.
    *   Moved `agent_readme.md`, `experiment_log.md`, `RESULTS.md` to `docs/`.
2.  **Code Improvements (`main.py` & `src/data_preprocessing.py`)**:
    *   **Objective 1 (Customer Similarity)**: Implemented `get_knn` demo in `main.py` that excludes self-match and displays ORIGINAL values (not scaled).
    *   **Objective 2 (Classification)**: Fixed `F1 Score` calculation by ensuring target column passed to `data_preprocessor` is the binary one.
    *   **Objective 3 (Regression)**: Fixed logic to use CONTINUOUS target (`insurance_benefits`) instead of binary. Added clean comparison table for Unscaled vs Scaled.
    *   **Objective 4 (Obfuscation)**: Added clean comparison table for Original vs Obfuscated performance.
    *   **General**: Removed duplicate code, fixed `FutureWarning` in scaler, and added a shared `split_random_state` variable.

## Results
See `docs/RESULTS.md` for the full output.
*   **Classification**: Scaled data significantly outperforms Unscaled (F1 ~0.97 vs ~0.60).
*   **Regression**: Scaling has no effect on performance (RMSE 0.34, R2 0.43), but Obfuscation is proven to yield identical results to original data.
