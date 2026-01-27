# Video Game Sales Analysis

This project performs a comprehensive analysis of historical video game sales data to identify patterns that determine whether a game succeeds or fails. By analyzing platform trends, genre popularity, and user/critic reviews, we aim to spot potential big winners and plan advertising campaigns.

## Project Overview

The analysis is broken down into several key stages:
1.  **Data Preprocessing**: Cleaning the raw data, handling missing values, identifying duplicates, and converting data types.
2.  **Exploratory Data Analysis (EDA)**: Visualizing sales trends over time, by platform, and by genre.
3.  **User & Critic Impact**: Investigating the correlation between review scores and global sales.
4.  **Hypothesis Testing**: Statistically testing assumptions about user ratings across different platforms and genres.

## Project Structure

```text
project_5/
├── data/
│   └── games.csv              # Raw dataset
├── src/
│   ├── data_preprocessing.py  # Functions for cleaning and preparing data
│   ├── eda.py                 # Visualization and analysis functions
│   └── analysis.py            # Statistical hypothesis testing
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Key Features

*   **Data Cleaning**: 
    *   Standardizes column names.
    *   Intellectually handles missing values (e.g., `tbd` in user scores).
    *   Converts types for memory optimization and correct date handling.
*   **Visualizations**:
    *   **Sales Trends**: Line charts and boxplots showing sales distribution over years.
    *   **Heatmaps**: Visualizing platform lifecycles and regional sales intensity.
    *   **Scatter Plots**: Correlating critic scores vs. user scores.
*   **Statistical Analysis**:
    *   Spearman correlation analysis.
    *   T-tests to compare average user ratings between platforms (Xbox One vs. PC) and genres (Action vs. Sports).

## Getting Started

### Prerequisites

Ensure you have Python installed along with the following libraries:
*   pandas
*   matplotlib
*   numpy
*   scipy

You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

*(Note: If `requirements.txt` is missing, install the packages directly: `pip install pandas matplotlib numpy scipy`)*

### Running the Analysis

To execute the full analysis pipeline, simply run the `main.py` script from the project root:

```bash
python main.py
```

## Findings & Conclusions

Based on the analysis (see `main.py` for detailed output):
*   **Sales Cycles**: Platforms typically have a sales lifecycle of about 10 years.
*   **Peak Era**: Game releases and sales peaked around 2008-2009.
*   **Platform Leaders**: In the relevant recent period, PS4, PS3, and X360 were distinct market leaders.
*   **Score Correlations**: Critic scores generally correlate with sales, though user scores show more variance.
*   **Regional Differences**: North America and Europe have similar market preferences, while Japan's market is distinct.

## Author

Angelo
