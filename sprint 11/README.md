# Insurance Solutions: Privacy & Prediction

This project analyzes insurance data to solve several key business problems while ensuring client data privacy. It implements machine learning models to identify similar customers, predict insurance benefits, and demonstrates data obfuscation techniques that protect personal information without compromising model performance.

## Project Objectives

1.  **Find Similar Customers**: identify customers who are similar to a specific customer using K-Nearest Neighbors (KNN). This helps in targeted marketing.
2.  **Predict Benefit Receipt**: Predict whether a new customer is likely to receive an insurance benefit using a classification approach.
3.  **Predict Number of Benefits**: Build a Linear Regression model to predict the number of insurance benefits a new customer is likely to receive.
4.  **Protect Client Data**: Implement a data obfuscation algorithm using matrix multiplication to mask personal data (gender, age, income, family members). The project mathematically and empirically proves that Linear Regression models trained on this obfuscated data perform identically to those trained on original data.

## Key Findings

*   **Scaling Matters for KNN**: Feature scaling (MaxAbsScaler) significantly improves the quality of K-Nearest Neighbors results by ensuring all features contribute equally to the distance metric.
*   **Obfuscation Works**: Multiplying the feature matrix $X$ by an invertible matrix $P$ protects the data. The weights of the Linear Regression model adjust ($w_P = P^{-1}w$) such that the predictions remain exactly the same ($y = X w = X P P^{-1} w$).

## Project Structure

```
.
├── data/
│   └── insurance_us.csv    # Dataset
├── src/
│   └── data_preprocessing.py # Helper functions for EDA, plotting, and custom Linear Regression class
├── main.py                 # Main execution script running the analysis and proofs
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

1.  Clone the repository and navigate to the project directory:
    ```bash
    git clone https://github.com/angelomp2001/portfolio.git
    cd "portfolio/sprint 11"
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main analysis script:

```bash
python main.py
```

This script will:
1.  Load and inspect the data.
2.  Perform Exploratory Data Analysis (EDA).
3.  Compare KNN results with and without feature scaling.
4.  Train a Linear Regression model (implemented from scratch) on original data.
5.  Obfuscate the data using a random invertible matrix.
6.  Train the model on obfuscated data and verify that RMSE and R2 scores remain unchanged.

## Technologies Used

*   **Python**: Core programming language.
*   **Pandas & NumPy**: Data manipulation and linear algebra operations.
*   **Scikit-learn**: Machine learning utilities (metrics, scalers, KNN).
*   **Seaborn/Matplotlib**: Data visualization.
