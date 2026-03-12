# Branch Edit Summary: EXP-project7-refactor-v2

## Overview
Replaced the old `project_7` code with the fully refactored `project_7_v2` codebase.

## Changes Made
- Deleted the old `project_7` directory.
- Renamed `project_7_v2` to `project_7`.
- The new codebase uses robust cross-validation (StratifiedKFold) and `sklearn.pipeline.Pipeline` with built-in preprocessing.
- Saved model is now a Pipeline instead of just the classifier.

## Results
- The new model predicts similarly to the old one (~95.33% match rate) while being much more robust to raw data inputs by doing preprocessing inside the pipeline.
