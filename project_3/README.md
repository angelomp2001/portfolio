# Telecom Plan Analysis (Project 3)

## Business Case
The telecom operator Megaline wants to know which of two prepaid plans (Surf or Ultimate) brings in more revenue in order to adjust their advertising budget.
Objective: which of the plans brings in more revenue?

## Task
Perform Exploratory Data Analysis (EDA) and hypothesis testing on a sample of 500 Megaline clients to uncover patterns in client behavior, such as call durations, message counts, and internet usage, and to determine revenue differences between plans and regions. 

## Project Summary and Findings
The project loads client data on calls, messages, internet, and plans, cleans missing values and handles data types. Then, it aggregates usage into monthly revenue per user to test hypotheses.

### Key Insights
- **Revenue by Plan**: Ultimate tends to have a higher base revenue, while Surf has a wider distribution of extra fees due to clients exceeding limits.
- **Hypothesis Testing**: Statistical tests help determine if differences in revenue by plan or geography are significant.

## Environment
Dev ✅ 

## Project folder structure
```
project_3/
├── data/                            # Raw dataset CSV files
├── docs/
│   ├── branching_workflow.md        # Experiment branching workflow
│   ├── EXPERIMENTS.md               # Log of all experiments run
│   ├── RESULTS.md                   # Results summary for current branch
│   ├── Branch_Edit_Summary.md       # Detailed code changes for current branch
│   ├── checklist.md                 # Checklist of what features should be in this project
│   └── experiment_log.md            # Running log of all merged experiments
├── scripts/                         # Directory for python scripts, e.g. notebooks
├── src/
│   ├── config.py                    # Configuration and file paths
│   └── data_preprocessing.py        # EDA, cleaning, hypothesis testing
├── main.py                          # Entry point — orchestrates the pipeline
├── tests.py                         # Unit tests
├── requirements.txt                 # pip install requirements
├── .gitignore                       # list of things for github to ignore
└── README.md                        # This file
```

## Project documentation
Readme: the main project document
Branching_workflow: git strategy and workflow for the agent
Branch_Edit_Summary: a description of the edits done on this branch.
Checklist: a generic template checklist of what features should be in this project
Experiment_log: a log of the results from the branch.
Results: the latest results of the branch, just before it merges with main.
EXPERIMENTS: a cumulative log of experiment logs that lives on the main branch only. 

To know what is happening, read the readme, then the results. If the results are inadequate, read the experiments.md to see how far along we are. The latest experiment_log and/or branch_edit_summary will tell you where we are right now. From this, you can guess what to do next to complete the branch experiment and complete the project.  

## How to run pipeline
```bash
python main.py
```

## Data Pipeline Overview
├── Load — Read `calls.csv`, `internet.csv`, `messages.csv`, `users.csv`, `plans.csv`
├── Clean — Fix data types (`to_datetime`), missing values (`fill_missing`), remove duplicate rows
└── Evaluate — Visualize differences in plan behavior, aggregate monthly revenue, run t-tests