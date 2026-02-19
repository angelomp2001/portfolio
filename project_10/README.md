# Gold Recovery Efficiency Analysis

## Project Goal
**Predict the amount of gold recovered from gold ore.**

The objective is to optimize the gold production process. We are building a machine learning model to predict two key metrics:
1.  **`rougher.output.recovery`**: The efficiency of the first "Rougher" flotation stage.
2.  **`final.output.recovery`**: The efficiency of the final purification stage.

By accurately predicting these recovery rates based on sensor data (inputs and state parameters), the business can optimize parameters to maximize gold yield.

## The Problem
The data comes from a gold extraction process.
-   **Input**: Raw ore feed (Au, Ag, Pb, Sol).
-   **Process**:
    1.  **Rougher Feed**: Raw material enters the first flotation bank.
    2.  **Rougher Concentrate**: Output of the first stage.
    3.  **Primary Cleaner**: First purification step.
    4.  **Secondary Cleaner**: Second purification step.
    5.  **Final Concentrate**: The final gold product.
-   **Challenge**: The test dataset does not contain target values (`recovery`) or output parameters (which are measured *after* the process). We must train a model using only the **input** and **state** parameters available at the start of the process to predict the final recovery.

## What `main.py` Does
The `main.py` script orchestrates the entire machine learning pipeline:

1.  **Data Loading**: Reads full, train, and test datasets.
2.  **Data Cleaning & Preprocessing**:
    *   **Verifies Recovery Calculation**: Checks that the `recovery` target in the training set was calculated correctly using the chemically defined formula.
    *   **Feature Engineering (Total Concentration)**: Sums the concentrations of all substances (Au + Ag + Pb + Sol) at each stage. Rows with a sum of **0** are considered anomalies (bad sensor data) and are removed.
    *   **Column Alignment**: Identifies `common_columns_sans_date`—the list of features present in both Train and Test sets. This ensures the model is trained only on features that will be available in production (no data leakage).
3.  **Exploratory Data Analysis (EDA)**:
    *   Visualizes how metal concentrations change across purification stages (e.g., Gold should increase, impurities should decrease).
    *   Compares feed particle size distributions between train and test sets to ensure the model will generalize well.
4.  **Model Training & Evaluation**:
    *   Trains three models: **Linear Regression**, **Decision Tree**, and **Random Forest**.
    *   Evaluates models using **sMAPE** (Symmetric Mean Absolute Percentage Error).
    *   Calculates the final weighted score:
        $$ \text{Final sMAPE} = 0.25 \times \text{sMAPE(rougher)} + 0.75 \times \text{sMAPE(final)} $$

## Web Dashboard (FastAPI)
In addition to the command-line analysis, this project includes a **FastAPI** web dashboard (`app.py`) for interactive use.

### Features:
1.  **File Upload**: Upload your own CSV datasets (Full, Train, Test).
2.  **Automated Preprocessing**: Runs the same robust data checks and cleaning pipeline as `main.py`.
3.  **Visualization**: Generates real-time plots for:
    *   Recovery Calculation Verification (MAE).
    *   Metal Concentration Changes across purification stages.
4.  **Instant Predictions**: Loads the **pre-trained Random Forest model** (saved by `main.py`) to generate predictions for `rougher.output.recovery` and `final.output.recovery` without retraining.

## Project Structure
```
sprint 10/
├── data/                       # Contains dataset CSVs (full, train, test)
├── models/                     # Saved models (scaler.pkl, model_rougher.pkl, model_final.pkl)
├── src/
│   ├── data_explorers.py       # Helper functions for data inspection (view, see)
│   ├── data_preprocessing.py   # Core logic for cleaning, feature engineering, modeling, and saving
│   └── H0_testing.py           # Statistical testing modules
├── app.py                      # FastAPI web application
├── main.py                     # Main execution entry point (Analysis + Model Training)
├── project_10.ipynb            # Jupyter notebook for interactive analysis
└── requirements.txt            # Project dependencies
```

## Setup and Usage

### Prerequisites
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `fastapi`, `uvicorn`, `python-multipart`, `joblib`, `jinja2`

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn fastapi uvicorn python-multipart joblib jinja2
   ```

### Running the Analysis
**Step 1: Train and Save Models**
First, run `main.py` to perform the analysis and save the best-performing models to the `models/` directory.
```bash
python main.py
```
*Output: Analysis logs, sMAPE scores, and saved model files in `models/`.*

**Step 2: Launch the Dashboard**
Start the FastAPI server to use the web interface.
```bash
uvicorn app:app --reload
```
*Access the dashboard at:* `http://127.0.0.1:8000`
