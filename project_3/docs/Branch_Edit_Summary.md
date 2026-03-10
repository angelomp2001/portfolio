# Branch Edit Summary
## Branch: experiment/structural-alignment
- Structural Alignment: finalized the project folder structure in the `README.md` to cleanly reflect a pure EDA and Hypothesis Testing script (removing API, ml paths). Implemented the missing `docs` and `scripts` directories.
- Feature Extraction & Checklist Formatting: Reviewed the codebase against `docs/checklist.md`. Formatted the checklist into valid markdown task syntax (`- [✅] Item` / `- [❌] Item`). 
- Missing Best Practices: Enforced several missing best practices that were conceptually suitable for an EDA/Data preprocessing project:
   1. Added Global random seed (`numpy.random.seed(42)`) in `main.py`
   2. Configured script to save out `raw_data_stats.csv` during `inspect_initial_data`
   3. Configured script to save out `clean_data_stats.csv` after preprocessing.
   4. Enforced explicit `print()` wrapper around the `df.sample(5)` so it can be seen in non-interactive sessions. 
   5. Created `tests.py` using `unittest` to ensure code is unit testable, checking the explicit datatype setting functionality from `src/data_preprocessing.py`.
- Documentation: Updated `README.md` with the newly polished details of Project 3 to correctly represent the project boundaries and structure.
