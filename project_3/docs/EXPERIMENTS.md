# EXPERIMENTS

## Experiment 2: Refine-DAG
**Goal**: Implement an AST-based parser to dynamically retrieve dependencies and module components, replacing hardcoded nodes in `DAG.py`. Formalize an experimental branch workflow.
**Outcome**: A flexible dependency visualization system that scans `.py` elements automatically. Created a robust workflow to continuously and modularly push project improvements.

### Branch Edit Summary
- DAG Refinement: Replaced hardcoded dependency nodes in `DAG.py` with dynamic AST parsing logic. The script now reads the Python files in the directory automatically, detects file contents including top-level classes, functions, variables, and cross-references their imports to programmatically generate an accurate component diagram.
- DAG Visual Polish: Implemented dynamic container heights, color styling, nested element boxes, custom layout parameters, and detailed legend for the new module graph format. The output is now saved dynamically as `dag.png` in the project root.
- Branching Workflow: Created `docs/branching_workflow.md` to define a systematic, step-by-step approach to running git-based experimental iterations on the project.

### Results
- Dynamic module parsing implemented via AST for accurate dependency mapping across the application.
- Established a concrete Git branching workflow to standardize experimental iteration and logging.
- `DAG.py` now robustly builds and saves a detailed modular component diagram `dag.png` to the project root.

## Experiment 1: structural-alignment
**Goal**: Align the project structure, generate comprehensive valid documentation, update checklist with existing techniques, and enforce missing logical core components.
**Outcome**: The `README.md` clearly reflects this EDA-only script project architecture. `checklist.md` reflects accurate mappings. `main.py` runs with initialized random states and properly dumps `raw_data_stats.csv` and `clean_data_stats.csv`. A testing module `tests.py` now exercises key type-assignment features to prove code testability. Branch edits compiled.

### Branch Edit Summary
- Structural Alignment: finalized the project folder structure in the `README.md` to cleanly reflect a pure EDA and Hypothesis Testing script (removing API, ml paths). Implemented the missing `docs` and `scripts` directories.
- Feature Extraction & Checklist Formatting: Reviewed the codebase against `docs/checklist.md`. Formatted the checklist into valid markdown task syntax (`- [✅] Item` / `- [❌] Item`). 
- Missing Best Practices: Enforced several missing best practices that were conceptually suitable for an EDA/Data preprocessing project:
   1. Added Global random seed (`numpy.random.seed(42)`) in `main.py`
   2. Configured script to save out `raw_data_stats.csv` during `inspect_initial_data`
   3. Configured script to save out `clean_data_stats.csv` after preprocessing.
   4. Enforced explicit `print()` wrapper around the `df.sample(5)` so it can be seen in non-interactive sessions. 
   5. Created `tests.py` using `unittest` to ensure code is unit testable, checking the explicit datatype setting functionality from `src/data_preprocessing.py`.
   6. Updated `inspect_initial_data` to output a `docs/data_statistics.md` markdown table containing column statistics.
   7. Added `df_to_markdown` and `visualize_raw_data` functions to `data_preprocessing.py` to save `docs/raw_df_col_hist.png` distribution charts.
   8. Implemented a `DAG.py` file to programmatically generate and save a `docs/component_diagram.png` outlining the project's data flow.
- Documentation: Updated `README.md` with the newly polished details of Project 3 to correctly represent the project boundaries and structure.

### Results
- Successfully cleaned, enriched, and structured Project 3 source files.
- Generates `docs/data_statistics.md`, multiple `docs/raw_*_hist.png` visualizations, and `docs/component_diagram.png`.
