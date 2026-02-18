# Oil Well Profitability Analysis

## Project Overview

This project analyzes oil well data from three different regions to determine the most profitable location for a new oil well development. The analysis focuses on maximizing potential profit while minimizing the risk of loss, utilizing linear regression and bootstrapping techniques to model profitability and uncertainty.

## Objectives

-   Train a Linear Regression model to predict the volume of reserves in new wells.
-   Calculate potential profit for a set of selected wells based on model predictions.
-   Use bootstrapping to simulate 1000 scenarios to estimate the distribution of profit and the risk of loss for each region.
-   Recommend the region with the highest expected profit, subject to a risk of loss threshold (typically < 2.5%).

## Key Findings

After analyzing the three regions, **Region 2** is recommended for development.

-   **Highest Mean Profit:** Region 2 offered the most consistent and highest average profit ($\approx$ $6.45 Million) when accounting for realistic selection (top 200 wells).
-   **Lowest Risk:** Region 2 had the lowest risk of loss (0.6%), which is well below the 2.5% threshold. Other regions had higher risks.
-   **Model Reliability:** The model for Region 2 was highly predictive (RMSE score closest to 0 / high R2), whereas other regions had lower predictive quality.

## Project Structure

├── data/                   # Directory containing dataset CSVs (geo_data_0.csv, etc.)
├── docs/                   # Documentation and Experiment logs
├── src/                    # Source code for helper modules
│   ├── analysis.py         # Functions for statistical analysis and modeling
│   ├── data_explorers.py   # Functions for Exploratory Data Analysis (view, see)
│   ├── data_preprocessing.py # Functions for data loading and cleaning
│   ├── utils.py            # Utility functions (hashing, duplicates)
│   └── visualization.py    # Plotting functions for web interface
├── app.py                  # FastAPI Web Application
├── main.py                 # CLI execution script
├── requirements.txt        # Python dependencies
└── README.md               # This project documentation
```

## Setup and Installation

1.  **Clone the repository** (if applicable).
2.  **Install Dependencies**:
    Ensure you have Python installed. Install the required libraries using pip:

    ```bash
    pip install -r requirements.txt
    ```

    *Key libraries include: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`.*

## Usage

### CLI Analysis
To run the standard analysis in the terminal:
```bash
python main.py
```

### Web Interface (FastAPI)
To launch the interactive web dashboard where you can upload new data:
```bash
uvicorn app:app --reload
```
Then open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000).

The web interface allows you to:
1.  Upload a new CSV file.
2.  Automatically check for duplicates (rejected if found).
3.  Visualize profit distributions and risk analysis for all regions.

## Methodology

1.  **Data Preparation**: Data is loaded and checked for integrity. Missing values and duplicates are handled in the preprocessing stage.
2.  **EDA**: `src.data_explorers` provides tools (`view`, `see`) to inspect headers, distributions, and correlations.
3.  **Modeling**: A Linear Regression model is trained on 75% of the data/validation on 25% to predict reserve volumes.
4.  **Profit Calculation**:
    -   Budget: $100 Million
    -   Revenue per Unit: $4,500
    -   Wells to Select: Top 200 from 500 randomly sampled points.
5.  **Risk Assessment**: Bootstrapping (1000 samples) creates a distribution of potential profits used to calculate the 95% confidence interval and the probability of negative profit (risk of loss).
