# Experiment: FastAPI Analysis (EXP-002)

## Goal
Create a FastAPI web application to interactively upload new geological data, check for duplicates, and visualize analysis results against existing regions.

## Changes

1.  **Docs Reorganization**: Moved `agency_readme.md`, `EXPERIMENTS.md`, `RESULTS.md`, `experiment_log.md` into `docs/`.
2.  **Dependencies**: Added `fastapi`, `uvicorn`, `python-multipart`, `jinja2`, `aiofiles`.
3.  **New Modules**:
    *   `src/utils.py`: Implements `calculate_file_hash` and `check_duplicate` using SHA-256.
    *   `src/visualization.py`: Generates base64-encoded profit distribution (`KDE plot`) and risk (`Bar chart`) plots using `matplotlib`/`seaborn`.
    *   `app.py`: FastAPI implementation serving an HTML interface. Handles file upload, duplicate rejection, analysis orchestration, and report rendering.
4.  **Updated README.md**: Added instructions for running the FastAPI server.

## Logic Overview
*   **Upload**: User uploads a CSV.
*   **Validation**: Server hashes content, compares with `data/*.csv`. If match found, rejects with exact filename.
*   **Analysis**: Used existing `src/analysis` functions (`train_and_predict`, `bootstrap_profit`) but wrapped in a loop within `app.py` to aggregate results dynamically.
*   **Result**: Returns an HTML page with a summary table, Recommended Region, and embedded visualization plots.
