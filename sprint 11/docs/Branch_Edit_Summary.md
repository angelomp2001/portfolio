# EXP-012: Update Readme

**Status**: ✅ Success
**Date**: 2026-03-08

## Summary
Updated `README.md` to accurately reflect the mathematical and privacy/obfuscation focus of Sprint 11, instead of the generic boilerplate template.

## Key Changes
1.  **Project Structure Updates**:
    *   Removed mentions of non-existent files like `app.py` and `catboost_info/` from the folder structure.
    *   Replaced `eda.ipynb` with the actual file name `project_11.ipynb`.
    *   Updated the `src` directory references to mirror the actual contents correctly (`demo_logic.py`, `data_preprocessing.py`).
2.  **API References Update**:
    *   Left the headings intact (e.g. "How to run inference via API" and "API Overview") to comply with the project template, but added explicit notes stating that a live inference API is not built or used for this demonstration.
3.  **Pipeline Overview Adjustments**:
    *   Replaced the boilerplate pipeline flow with standard steps that reflect what `main.py` is doing. Specifically, added steps detailing the loading of scaling, searching for similar users via KNN, applying Linear Regression across scaled and non-scaled data, and the Obfuscation Proof via matrix multiplication.
4.  **Training Instructions Fix**:
    *   Updated the `main.py` execution instructions to note the presence of an `--eda` flag.

## Results
See `docs/RESULTS.md` for output logs regarding script-based evaluations.
The README is now accurate and removes false promises of an inference API while maintaining the required template sections.
