# Branch Edit Summary — EXP-014-Refactor-Preprocessor

## Summary
Refactored the `data_preprocessing.py` script to inline the pipeline for numerical models inside the `ColumnTransformer`. The user also added several explanatory inline comments to the preprocessing steps to detail the behavior of the `OrdinalEncoder`, `OneHotEncoder`, `StandardScaler`, and `PolynomialFeatures` blocks.

---

## Files Modified

### `src/data_preprocessing.py`
- Inlined the numerical pipeline variable directly into the `ColumnTransformer` under the `else` block for linear/NN models to reduce the number of discrete code blocks.
- Added various inline comments to detail step-by-step logic, specifically explaining why we split columns (`ColumnTransformer`), and describing how each component processes the data (`OrdinalEncoder` unknown value handling, `StandardScaler`, and `PolynomialFeatures`).

### `main.py`
- Triggered metrics logging to properly save results in the latest format.
