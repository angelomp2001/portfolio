# Branch Edit Summary
## Modified Files
- [x] main.py
- [x] src/data_preprocessing.py
- [x] README.md
- [x] app.py (New)

## Changes Description
- Refactored `main.py` to be a clean orchestrator with high-level function calls and comments.
- Moved implementation details (column definitions, plotting, anomaly removal) to `src/data_preprocessing.py`.
- Added model saving functionality (`joblib`) to `src/data_preprocessing.py`.
- Created `app.py`, a FastAPI dashboard that allows users to upload data, visualize analysis, and generate predictions using the pre-trained models.
- Updated `README.md` to include instructions for the new web dashboard and dependencies.
