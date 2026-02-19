# User Behavior Model Analysis

## Project Overview
This project predicts whether a user is using the "Ultra" plan (`is_ultra`) based on their behavior data. We aim to develop a model with the highest possible accuracy, targeting a threshold of 0.75.

## Data Preprocessing & Modeling Strategy
The target variable (`is_ultra`) is imbalanced: roughly 30% are "Ultra" users (1s) and 70% are standard users (0s). To ensure a robust and fair evaluation, we implemented the following strategy:

1.  **Data Partitioning**: The dataset was split into three distinct sets:
    *   **Training Set (60%)**: Used to train the models.
    *   **Validation Set (20%)**: Used for hyperparameter tuning and model selection.
    *   **Test Set (20%)**: Used for the final evaluation of the selected model.
2.  **Stratified Splitting**: We used the `stratify` parameter in `train_test_split()` to ensure our training, validation, and test sets all retain the original 30/70 proportion of target values.
3.  **Baseline Comparison**: We evaluated model quality not just by raw accuracy, but by comparing it to a `DummyClassifier` baseline that predicted based on the most frequent class.

## Model Selection & Hyperparameter Optimization
We compared three different machine learning models to identify the best predictor for user behavior:

| Model | Hyperparameters Optimized | Search Range |
| :--- | :--- | :--- |
| **Decision Tree** | `max_depth` | 1 - 20 |
| **Random Forest** | `max_depth`, `n_estimators` | `max_depth`: 1-20, `n_estimators`: 10-100 |
| **Logistic Regression** | *N/A (Baseline)* | default (solver='liblinear') |

### Optimization Process
We developed a custom **Hyperparameter Optimizer** that uses an iterative approach (similar to a binary search) to efficiently find the optimal values for the parameters listed above. The optimization follows these steps:
1.  Train the model on the **Training Set**.
2.  Evaluate the model on the **Validation Set**.
3.  Iteratively adjust parameters to maximize the validation accuracy.
4.  Once the best model is identified, it is refit on the training data and finally evaluated on the **Test Set**.

## Conclusion
Our analysis shows that accounting for the target distribution and performing systematic hyperparameter tuning is critical.

*   **Baseline Model**: A naive "Dummy" predictor (guessing the most frequent class) achieves an accuracy of ~**69.4%**.
*   **Our Best Model**: The **RandomForestClassifier** outperformed the others, achieving a test accuracy of ~**81.1%**.

**Result**: Our model outperforms the baseline by approximately **12%**. This exceeds our target threshold of 0.75 accuracy and confirms that the model provides significant predictive value beyond simple guessing.

## How to Run the Prediction Service

The project is split into two parts: **Model Training** and **API Service**.

### 1. Install Dependencies
Ensure you have the required packages installed:
```bash
pip install -r requirements.txt
```

### 2. Train and Save the Model
Before running the API, you must train the model. This script will perform hyperparameter optimization and save the best model and its metadata to the `models/` directory.
```bash
python main.py
```

### 3. Start the API Server
Once the model is saved, you can start the FastAPI server. The server loads the pre-trained model from disk, making it start instantly.
```bash
uvicorn app:app --reload
```

### 4. Get Model Performance
Visit `http://127.0.0.1:8000/` to see the performance metrics of the currently loaded model.

### 5. Get a Prediction
Submit user behavior data to the `/predict` endpoint.

**Using the Interactive Docs:**
1.  Go to `http://127.0.0.1:8000/docs`
2.  Click on `POST /predict` -> `Try it out`
3.  Enter the values for `calls`, `minutes`, `messages`, and `mb_used`.
4.  Click `Execute`.
