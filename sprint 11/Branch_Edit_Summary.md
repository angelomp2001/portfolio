# Branch Edit Summary

- **Branch**: experiments/EXP-012-Refactor-Demo-Modules
- **Experiment Description**: Refactor Objectives 3 and 4 out of `main.py` into dedicated demo modules to improve readability and separation of concerns.
- **Date**: 2026-02-20

## Motivation
`main.py` contained too much low-level fitting logic for Objectives 3 (Linear Regression scaling) and 4 (Obfuscation proof). The goal was to make `main.py` a clean orchestrator, with detailed implementation delegated to focused modules.

## Changes

### `main.py` (Modified)
- Added imports: `from demo_scaling import run_scaling_demo` and `from demo_obfuscation import run_obfuscation_demo`
- Replaced ~38 lines of Objective 3 (LR fitting, comparison table) with a single call: `run_scaling_demo(df, features, split_random_state)`
- Replaced ~36 lines of Objective 4 (P-matrix, obfuscation, fitting, comparison table) with a single call: `run_obfuscation_demo(X_train_reg, X_test_reg, y_train_reg, y_test_reg, rmse_unscaled)`
- Total: reduced from 165 lines to 105 lines

### `demo_scaling.py` (Created)
- Encapsulates Objective 3 in `run_scaling_demo(df, features, split_random_state)`
- Performs the regression train/test split, fits LR on unscaled and scaled data, prints comparison table
- Returns `(X_train_reg, X_test_reg, y_train_reg, y_test_reg, rmse_unscaled)` so Objective 4 can reuse the splits without re-splitting

### `demo_obfuscation.py` (Created)
- Encapsulates Objective 4 in `run_obfuscation_demo(X_train_reg, X_test_reg, y_train_reg, y_test_reg, rmse_unscaled)`
- Generates random invertible matrix P, obfuscates features (X' = X @ P), trains LR, and prints comparison table vs. the original baseline RMSE

## Files Created/Modified
- `main.py` (Modified — 165 → 105 lines)
- `demo_scaling.py` (Created)
- `demo_obfuscation.py` (Created)
- `Branch_Edit_Summary.md` (Updated)

## Next Steps
- Run `python main.py` to verify output is identical to pre-refactor.
- If successful, commit with ✅ and merge to master.
