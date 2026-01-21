# Instacart Market Basket Analysis (Project 2)

## Overview
This project performs Exploratory Data Analysis (EDA) on the Instacart Market Basket dataset. The goal is to clean the data, handle missing values, and uncover patterns in customer behavior, such as:
- When do customers place orders?
- What are the most popular products?
- How often do customers reorder items?

## Project Structure
- **`data/`**: Contains the dataset CSV files (`instacart_orders.csv`, `products.csv`, etc.).
- **`scripts/`**: Contains the Jupyter Notebook (`Sprint_2_EDA.ipynb`) where the main analysis and visualizations are performed.
- **`src/`**: Source code for data loading and preprocessing.
  - `config.py`: Configuration and file paths.
  - `data_preprocessing.py`: Helper functions for loading, cleaning, and verifying data.
- **`main.py`**: A script version of the analysis workflow (Work In Progress).
- **`requirements.txt`**: List of Python dependencies.

## Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the analysis, launch the Jupyter Notebook:
```bash
jupyter notebook scripts/Sprint_2_EDA.ipynb
```
Follow the cells in the notebook to execute the data cleaning and visualization steps.

## Key Insights
- **Missing Values**:
  - `days_since_prior_order`: Missing values correspond to a customer's first order.
  - `add_to_cart_order`: Missing for items added to the cart after the 64th item (likely a data collection artifact).
- **Order Patterns**:
  - Orders peak on days 0 and 1 (likely Sunday and Monday).
  - Most orders are placed between 10 AM and 4 PM.

## License
[MIT](https://choosealicense.com/licenses/mit/)
