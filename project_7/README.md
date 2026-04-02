# User Behavior Model Analysis

## Business Case
A telecom company needs a model to recommend the right plan ("Ultra" or "Smart") based on subscriber behavior.

## Task
* Predict whether a user is using the "Ultra" plan (`is_ultra`) based on their behavior data: `calls`, `minutes`, `messages`, and `mb_used`.

## Project Summary and Findings
Our analysis shows that accounting for the target distribution and performing systematic hyperparameter tuning is critical. The best model was tested to exceed the baseline and the target threshold (accuracy > 0.75).
* Random Forest Classifier achieved ~80.9% accuracy.
* Dummy Classifier Baseline achieved ~69.4% accuracy.

### Models Evaluated
| Base Model             |
|------------------------|
| DecisionTreeClassifier |
| RandomForestClassifier |
| LogisticRegression     |

## Environment
Dev ✅

## Project folder structure
├── data/
│   └── data.csv                       # Raw dataset
├── docs/
│   ├── branching_workflow.md          # Experiment branching workflow
│   ├── RESULTS.md                     # Results summary for current branch
│   ├── Branch_Edit_Summary.md         # Detailed code changes for current branch
│   ├── experiment_log.md              # Running log of all merged experiments
│   ├── checklist.md                   # Generic template checklist
│   ├── project_plan_v2.md             # Project plan documentation
│   ├── data_stats_raw.json            # Raw data statistics
│   ├── data_stats_clean.json          # Clean data statistics
│   ├── data_charts_raw_hist.png       # Raw data visualization
│   ├── data_charts_clean_pairplot.png # Clean data visualization
│   └── DAG.png                        # Component diagram visualization
├── models/
│   ├── RandomForestClassifier.joblib  # Serialized model
│   └── RandomForestClassifier_metadata.json # Best model metadata
├── src/
│   ├── config.py                      # Configuration file
│   ├── data_preprocessing.py          # Data preparation module
│   └── model_training.py              # Model training module
├── DAG.py                             # Script to generate component diagram
├── main.py                            # Entry point — orchestrates the pipeline
├── app.py                             # API Server
└── README.md                          # This file

## Project documentation
Readme: the main project document
branching_workflow.md: git strategy and workflow for the experiments
Branch_Edit_Summary.md: a description of the edits done on this branch.
checklist.md: a generic template checklist of what features should be in this project
experiment_log.md: a log of the results from the branch.
RESULTS.md: the latest results of the branch, just before it merges with main.
project_plan_v2.md: Project plan documentation.

To know what is happening, read the readme, then the results. If the results are inadequate, read the experiments.md to see how far along we are. The latest experiment_log and/or branch_edit_summary will tell you where we are right now. From this, you can guess what to do next to complete the branch experiment and complete the project.  

## How to train model
```bash
python main.py
```

## How to run inference via API
```bash
uvicorn app:app --reload
# Access docs at http://127.0.0.1:8000/docs
```

## Data Pipeline Overview
├── Load — Read data (`data.csv`)
├── Clean — Data exploration and summary statistics
├── Split — Stratified split into training, validation, test
├── Train — Fits candidate models and performs hyperparameter searching
├── Evaluate — Compare validation scores for candidate models
└── Save — Select the best model and evaluate on hold-out test set; serialize best model

## API Overview
├── Load — Loads the saved `RandomForestClassifier.joblib` and `RandomForestClassifier_metadata.json` on startup
├── Predict — Accepts POST requests with user behavior to generate predictions
└── Save — Log & save is missing.
