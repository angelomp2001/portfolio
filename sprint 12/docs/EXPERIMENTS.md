# EXPERIMENTS

Log of all experiments run in this project.  
See `experiment_log.md` for merged results and `RESULTS.md` for the current branch's results.

| ID | Branch | Description | Status | RMSE |
|----|--------|-------------|--------|------|
| EXP-001 | experiments/EXP-001-Organize-Project-Structure | Organize project folder: README, split model_training into own module, add workflow tracking files, add .gitignore | ✅ | — |
| EXP-002 | experiments/EXP-002-Refactor-Model-Training-Pipeline | Refactor model_training into 4 helpers; model list in main.py; random search with live charts; auto-detect best model & params; silence streaming output | ✅ | 2,232.51 |
| EXP-003 | experiments/EXP-003-Add-Data-Drift-Tracking | Add save_data_stats() to capture raw and clean data statistics each run for drift tracking; add template.md and checklist.md | ✅ | — |
| EXP-004 | experiments/EXP-004-Complete-Checklist | Add Keras NN, 5-fold CV, sklearn Pipelines, PolynomialFeatures, visualizations, multi-metric eval, peak memory tracking, model saving | ✅ | 2,195.02 |
