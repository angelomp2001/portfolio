# Video Game Sales Analysis

## Business Case
This project performs a comprehensive analysis of historical video game sales data to identify patterns that determine whether a game succeeds or fails. By analyzing platform trends, genre popularity, and user/critic reviews, we aim to spot potential big winners and plan advertising campaigns.

## Task
Perform a comprehensive exploratory data analysis (EDA) and hypothesis statistical testing on global video game sales data, investigating platform lifecycles, user vs. critic score correlations, and genre-specific sales drivers.

## Project Summary and Findings
Based on our analysis across platforms, genres, and regions, we established:
* **Sales Cycles**: Platforms typically have a lifecycle of about 10 years before obsolescence.
* **Peak Era**: New game releases and broad sales peaked around 2008-2009.
* **Platform Leaders**: For the modern scope, PS4, PS3, and X360 stood out as market leaders.
* **Score Correlations**: Critic scores generally correlate strongly with sales; user scores display larger variance and less direct impact.
* **Regional Preferences**: North America and Europe share similar genre preferences (favoring Action/Shooter), while Japan largely favors Role-Playing and alternative genres.
* **Statistical Insights**: T-tests revealed average user ratings differ significantly between Xbox One and PC platforms, but not between Action and Sports genres.

## Environment
Dev ✅

## Project folder structure/
```
project_5/
├── data/
│   └── games.csv                      # Raw dataset
├── docs/
│   ├── checklist.md                   # Feature template and tracking
│   ├── Branch_Edit_Summary.md         # Detailed code changes for current branch
│   ├── EXPERIMENTS.md                 # Log of all experiments run
│   ├── RESULTS.md                     # Results summary for current branch
│   ├── experiment_log.md              # Running log of all merged experiments
│   ├── raw_data_statistics.md         # Descriptive statistics of raw data
│   ├── clean_data_statistics.md       # Descriptive statistics of cleaned data
│   ├── dag.png                        # Dependency Component Diagram
│   ├── readme_template.md             # Standard document formatting template
│   └── branching_workflow.md          # Workflow logic and documentation
├── src/
│   ├── data_preprocessing.py          # Data loading, cleaning, formatting
│   ├── eda.py                         # Visualization and EDA patterns
│   └── analysis.py                    # Hypothesis and statistical tests
├── tests/
│   └── test_data_preprocessing.py     # Unit testing components
├── dag.py                             # Network/Module dependency graph visualizer
├── main.py                            # Main execution orchestrator
├── project_5.ipynb                    # Jupyter notebook for exploratory scratch work
├── requirements.txt                   # External dependencies
└── README.md                          # Project documentation
```

## Project documentation
* **Readme**: The main entry documentation for understanding the goals and outputs.
* **Checklist**: Tracking which standard machine learning and analytical best practices have been enforced across the repository.
* **Experiment Logs**: Tracks individual experiments during feature rollouts (RESULTS, EXPERIMENTS, Branch_Edit_Summary).
* **Statistics**: `raw_data_statistics.md` and `clean_data_statistics.md` are dynamically generated to give a snapshot of feature health pre- and post-processing.
* **dag.png**: Outlines how the different script files form a graph logic dependency structure for execution.

## How to execute analysis
Ensure the dependencies are installed:
```bash
pip install -r requirements.txt
```
To run the automated pipeline (including statistical data dumps and EDA graphic generation):
```bash
python main.py
```

## Data Pipeline Overview
├── Load — Read `data.csv`, randomly sample if defined (e.g. 10k rows) while preserving randomness state
├── Summarize Raw — Dump raw statistics into `raw_data_statistics.md`
├── Clean — Standardize names, manage missing string values, assign explicit dtypes, handle datetime columns, and remove duplicated data
├── Summarize Clean — Dump clean statistics into `clean_data_statistics.md`
├── Explore — Map out sales by genre, year, region, platform lifecycle mappings, and esrb impact tables
└── Test — Run Spearman correlation analyses and independent t-tests on defined hypotheses
