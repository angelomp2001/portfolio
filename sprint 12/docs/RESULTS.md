# RESULTS — EXP-016

**Branch:** `experiments/EXP-016-Create-API`  
**Goal:** Create a FastAPI prediction endpoint (`/predict`) to run inference on the loaded regression model and automatically record input runs to logging.

## Changes Made
- Created `api.py` serving a `FastAPI` instance.
- Implemented `schema` with Pydantic for validation.
- Extracted and formatted inference logic, saving the JSON payload + result to `data/api_inference_logs.jsonl`.
- Maintained fallback handlers if the model doesn't load successfully on boot.

## Model Results (5-fold CV → best model tuned → test evaluation)

*No changes to performance metrics since the task was purely software engineering/deployment focus rather than AI model tuning/architecting.*
- Tests: successful prediction output when hit directly via `curl`.
