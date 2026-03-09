# Instacart Market Basket Analysis (Project 2)

## Business Case
Instacart wants to understand customer shopping habits to optimize recommendations and uncover patterns in order scheduling and product popularity.

## Task
Perform Exploratory Data Analysis (EDA) on the Instacart Market Basket dataset to clean the data, handle missing values, and uncover patterns in customer behavior, such as when customers place orders, what they buy, and how often they reorder.

## Project Summary and Findings
The project successfully loads, cleans, and analyzes the Instacart data structure to reveal customer ordering patterns and popular items.

### Key Insights
- **Missing Values**:
  - `days_since_prior_order`: Missing values correspond to a customer's first order.
  - `add_to_cart_order`: Missing for items added to the cart after the 64th item (likely a data collection artifact).
- **Order Patterns**:
  - Orders peak on days 0 and 1 (likely Sunday and Monday).
  - Most orders are placed between 10 AM and 4 PM.

## Environment
Dev ✅ 

## Project folder structure
```
project_2/
├── data/                            # Raw dataset CSV files
├── docs/
│   ├── agent_readme.md              # Experiment branching workflow
│   ├── EXPERIMENTS.md               # Log of all experiments run
│   ├── RESULTS.md                   # Results summary for current branch
│   ├── Branch_Edit_Summary.md       # Detailed code changes for current branch
│   ├── checklist.md                 # Checklist of what features should be in this project
│   └── experiment_log.md            # Running log of all merged experiments
├── models/                          # Storage for models
├── scripts/                         # Directory for python scripts, e.g. notebooks
│   └── test.ipynb                   # Test notebook
├── src/
│   ├── config.py                    # Configuration and file paths
│   ├── data_preprocessing.py        # EDA, cleaning
│   ├── analysis.py                  # Helper functions for visualization
│   └── model_training.py            # Placeholder for model training
├── main.py                          # Entry point — orchestrates the pipeline
├── eda.ipynb                        # Jupyter notebook for exploratory work
├── requirements.txt                 # pip install requirements
├── .gitignore                       # list of things for github to ignore
└── README.md                        # This file
```

## Project documentation
Readme: the main project document
Agent_readme: git strategy and workflow for the agent
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
├── Load — Read `instacart_orders.csv`, `products.csv`, etc.
├── Clean — Fix missing values (`fill_missing`), remove duplicate rows
└── Evaluate — Visualize differences in `order_hour_of_day`, popular items, and order distributions
