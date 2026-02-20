# Experiment Log

Running log of all experiments merged into `main`.  
Updated after each successful merge per the branching workflow in `docs/agent_readme.md`.

---

## EXP-001 — Organize Project Structure ✅

**Branch:** `experiments/EXP-001-Organize-Project-Structure`  
**Date:** 2026-02-20  
**Status:** ✅ Success (organizational — no model changes)

### Code Changes
- Added `README.md` at project root with business case, model results table, project structure, and how-to-run instructions
- Cleaned up `main.py`: removed raw-string comment, added named imports, config block, and section dividers
- Extracted `model_training()` from `src/data_preprocessing.py` into `src/model_training.py`
- Added workflow tracking files: `EXPERIMENTS.md`, `RESULTS.md`, `Branch_Edit_Summary.md`, `experiment_log.md`
- Added `.gitignore` (ignores `catboost_info/`, `__pycache__/`, `.pyc`, `.ipynb_checkpoints/`)

### Results
No model results changed — this was a structural/readability sprint.

---
