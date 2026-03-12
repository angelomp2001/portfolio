# Experiment Log

## Experiment ID: EXP-001-Refactor-Data-Preprocessing
**Date**: 2026-02-17
**Status**: Success ✅

### Branch Edit Summary
## Overview
This branch focused on refactoring the monolithic `data_preprocessing.py` script into a modular architecture to improve readability, maintainability, and reuse. We also optimized the application startup by decoupling model training from import.

## Changes Made

### 1. Modularization
-   Deleted `src/data_preprocessing.py`.
-   Created `src/data/loader.py`: Handles CSV data loading with error checking.
-   Created `src/data/explorer.py`: Encapsulates data exploration and summary logic.
-   Created `src/models/trainer.py`: Manages model training and splitting.
-   Created `src/models/tuner.py`: Implements hyperparameter tuning using `GridSearchCV`.
-   Created `src/pipeline.py`: Orchestrates the model selection and training workflow.

### 2. Main Script Update (`main.py`)
-   Updated to use the new modular components.
-   Encapsulated training logic in a `main()` function.
-   Added logic to save the best model to `model.joblib`.
-   Added logging of results to `RESULTS.md`.

### 3. Application Optimization (`app.py`)
-   Modified to load the pre-trained `model.joblib` on startup instead of importing `main.py`.
-   Removed dependency on runtime training, significantly speeding up startup.

### 4. Verification
-   Added `tests/test_refactoring.py` for unit testing new modules.
-   Verified that model performance remains consistent with the original implementation.

## Results
-   **Baseline Accuracy**: 0.6936 (Unchanged)
-   **Final Test Score**: ~0.8087 (Comparable to original ~0.8118)
-   **Startup Time**: Drastically reduced for `app.py`.


### Results Data
| Experiment | Accuracy | Baseline | Performance Ratio |
| Refactor-Data-Preprocessing | 0.8087 | 0.6936 | 2.6382 |

---

## Experiment ID: EXP-001-Project-Setup
**Date**: 2026-03-11
**Status**: Success ✅

### Branch Edit Summary
# Branch Edit Summary: EXP-001 Refactor Data Preprocessing

## Overview
This branch focused on refactoring the monolithic `data_preprocessing.py` script into a modular architecture to improve readability, maintainability, and reuse. We also optimized the application startup by decoupling model training from import.

## Changes Made

### 1. Modularization
-   Deleted `src/data_preprocessing.py`.
-   Created `src/data/loader.py`: Handles CSV data loading with error checking.
-   Created `src/data/explorer.py`: Encapsulates data exploration and summary logic.
-   Created `src/models/trainer.py`: Manages model training and splitting.
-   Created `src/models/tuner.py`: Implements hyperparameter tuning using `GridSearchCV`.
-   Created `src/pipeline.py`: Orchestrates the model selection and training workflow.

### 2. Main Script Update (`main.py`)
-   Updated to use the new modular components.
-   Encapsulated training logic in a `main()` function.
-   Added logic to save the best model to `model.joblib`.
-   Added logging of results to `RESULTS.md`.

### 3. Application Optimization (`app.py`)
-   Modified to load the pre-trained `model.joblib` on startup instead of importing `main.py`.
-   Removed dependency on runtime training, significantly speeding up startup.

### 4. Verification
-   Added `tests/test_refactoring.py` for unit testing new modules.
-   Verified that model performance remains consistent with the original implementation.

## Results
-   **Baseline Accuracy**: 0.6936 (Unchanged)
-   **Final Test Score**: ~0.8087 (Comparable to original ~0.8118)
-   **Startup Time**: Drastically reduced for `app.py`.


### Results Data
| Experiment | Accuracy | Baseline | Performance Ratio |
| Refactor-Data-Preprocessing | 0.8087 | 0.6936 | 2.6382 |
| Experiment | Accuracy | Baseline | Performance Ratio |
| Refactor-Data-Preprocessing | 0.8087 | 0.6936 | 2.6382 |
| Experiment | Accuracy | Baseline | Performance Ratio |
| Refactor-Data-Preprocessing | 0.8087 | 0.6936 | 2.6382 |
| Experiment | Accuracy | Baseline | Performance Ratio |
| Refactor-Data-Preprocessing | 0.8087 | 0.6936 | 2.6382 |

---

## Experiment ID: EXP-project7-refactor-v2
**Date**: 2026-03-12
**Status**: Success ✅

### Branch Edit Summary
## Overview
Replaced the old `project_7` code with the fully refactored `project_7_v2` codebase.

## Changes Made
- Deleted the old `project_7` directory.
- Renamed `project_7_v2` to `project_7`.
- The new codebase uses robust cross-validation (StratifiedKFold) and `sklearn.pipeline.Pipeline` with built-in preprocessing.
- Saved model is now a Pipeline instead of just the classifier.

## Results
- The new model predicts similarly to the old one (~95.33% match rate) while being much more robust to raw data inputs by doing preprocessing inside the pipeline.

### Results Data
**Best Model**: RandomForestClassifier
**Primary Metric (accuracy) Test Score**: 0.8087

#### Test Set Evaluation
| Metric | Score |
|---|---|
| accuracy | 0.8087 |
| precision | 0.8364 |
| recall | 0.4670 |
| f1 | 0.5993 |
| roc_auc | 0.8019 |

#### CV Results for Best Model
| Metric | Mean CV Score |
|---|---|
| accuracy | 0.7958 |
| precision | 0.7201 |
| recall | 0.5470 |
| f1 | 0.6213 |
| roc_auc | 0.7958 |
