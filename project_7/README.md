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
│   └── users_behavior.csv             # Raw dataset
├── docs/
│   ├── branching_workflow.md          # Experiment branching workflow
│   ├── EXPERIMENTS.md                 # Log of all experiments run
│   ├── RESULTS.md                     # Results summary for current branch
│   ├── Branch_Edit_Summary.md         # Detailed code changes for current branch
│   └── experiment_log.md              # Running log of all merged experiments
├── models/
│   ├── best_model.joblib
│   └── metadata.json
├── src/
│   ├── data/
│   │   ├── loader.py                  # Load data
│   │   └── explorer.py                # Data exploration
│   ├── models/
│   │   ├── trainer.py                 # Model training module
│   │   └── tuner.py                   # Hyperparameter tuner module
│   └── pipeline.py                    # Orchestrates training and evaluation
├── tests/
│   └── test_refactoring.py            # Unit tests
├── main.py                            # Entry point — orchestrates the pipeline
├── app.py                             # API Server
├── requirements.txt                   # pip requirements
├── .gitignore                         # git ignores
└── README.md                          # This file

## Project documentation
Readme: the main project document
branching_workflow.md: git strategy and workflow for the experiments
Branch_Edit_Summary.md: a description of the edits done on this branch.
checklist.md: a generic template checklist of what features should be in this project
experiment_log.md: a log of the results from the branch.
RESULTS.md: the latest results of the branch, just before it merges with main.
EXPERIMENTS.md: a cumulative log of experiment logs that lives on the main branch only. 

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
├── Load — Read data (`users_behavior.csv`)
├── Clean — Data exploration and summary statistics
├── Split — Stratified split into training, validation, test
├── Train — Fits candidate models and performs hyperparameter searching
├── Evaluate — Compare validation scores for candidate models
└── Save — Select the best model and evaluate on hold-out test set; serialize best model

## API Overview
├── Load — Loads the saved `best_model.joblib` and `metadata.json` on startup
├── Predict — Accepts POST requests with user behavior to generate predictions
└── Save — Log & save is missing.
