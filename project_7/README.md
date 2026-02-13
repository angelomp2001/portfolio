# User Behavior Model Analysis

## Project Overview
This project predicts whether a user is using the "Ultra" plan (`is_ultra`) based on their behavior data. We aim to develop a model with the highest possible accuracy, targeting a threshold of 0.75.

## Data Preprocessing & Modeling Strategy
The target variable (`is_ultra`) is imbalanced: roughly 30% are "Ultra" users (1s) and 70% are standard users (0s). We addressed this in two key ways:
1.  **Stratified Splitting**: We used the `stratify` parameter in `train_test_split()` to ensure our training, validation, and test sets all retain the original 30/70 proportion of target values.
2.  **Baseline Comparison**: We evaluated model quality not just by raw accuracy, but by comparing it to a "Dummy" baseline.

## Conclusion
Our analysis shows that accounting for the target distribution is critical.

*   **Baseline Model**: A naive "Dummy" predictor (which simply guesses the most frequent class) achieves an accuracy of ~**69%**. This represents the accuracy one gets by having "no model".
*   **Our Model**: After hyperparameter optimization, our best model (RandomForestClassifier) achieves a test accuracy of ~**81%**.

**Result**: Our model outperforms the baseline by **12%** (81% vs 69%). Given a success threshold of 5% outperformance, we exceeded this by over 2x, confirming the model provides significant predictive value.

## How to Run the Prediction Service

We have exposed the best model via a FastAPI application.

### 1. Install Dependencies
Ensure you have the required packages installed:
```bash
pip install -r requirements.txt
```

### 2. Start the Server
Run the following command in the terminal to start the FastAPI server:
```bash
uvicorn app:app --reload
```

### 3. Get Model Performance
Visit `http://127.0.0.1:8000/` to see the current model's performance metrics.

### 4. Get a Prediction
You can submit user behavior data to the `/predict` endpoint to get a prediction.

**Using the Interactive Docs:**
1.  Go to `http://127.0.0.1:8000/docs`
2.  Click on `POST /predict` -> `Try it out`
3.  Enter the values for `calls`, `minutes`, `messages`, and `mb_used`.
4.  Click `Execute`.
