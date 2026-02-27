import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   StandardScaler, PolynomialFeatures)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data(path):
    return pd.read_csv(path)


# ── Clean ─────────────────────────────────────────────────────────────────────
def clean_data(df, target_col='Price', cols_to_drop=None,
               price_min=500, year_min=1900, year_max=2025,
               power_min=100, power_max=400):
    """
    Drop irrelevant columns, nullify out-of-range values,
    fill categorical NaN (any object col), dedup, and dropna.
    Returns a cleaned DataFrame (does not encode or scale).
    Cat/num column lists are NOT passed in — callers infer them from
    dtype on the cleaned data instead.
    """
    # drop irrelevant columns
    if cols_to_drop is None:
        cols_to_drop = []

    df = df.copy()
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Price: values below minimum are invalid
    df[target_col] = np.where(df[target_col] >= price_min, df[target_col], np.nan)

    # RegistrationYear: out-of-range → NaN (row dropped at dropna)
    df['RegistrationYear'] = df['RegistrationYear'].where(
        (df['RegistrationYear'] >= year_min) & (df['RegistrationYear'] <= year_max)
    )

    # Power: out-of-range → NaN (row dropped at dropna)
    df['Power'] = df['Power'].where(
        (df['Power'] >= power_min) & (df['Power'] <= power_max)
    )

    # Categorical missing: NaN → 'missing'
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('missing')

    # drop duplicates and dropna
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df


# ── Preprocessor factory ──────────────────────────────────────────────────────
def column_preprocessor(cat_cols, num_cols, is_tree=False):
    """
    Build a ColumnTransformer for use inside an sklearn Pipeline.

    is_tree=False (linear / NN):
        Categorical → OHE
        Numeric     → StandardScaler → PolynomialFeatures(2) → StandardScaler
                      (PolyFeatures applied only to numeric to avoid feature explosion) ✅

    is_tree=True (LGBM, RF, CatBoost, XGB):
        Categorical → OrdinalEncoder   (handles high cardinality efficiently)
        Numeric     → passthrough       (tree models don't need scaling)
    """
    # preprocess based on model type
    if is_tree:
        # Apply to tree models
        return ColumnTransformer([ # split cat and num columns and runs parallel processes
            ('cat', OrdinalEncoder( # encode categorical columns in the same column
                handle_unknown='use_encoded_value', unknown_value=-1), cat_cols), # unknown_value=-1 means that if a value is not found, it will be encoded as -1
            ('num', 'passthrough', num_cols), # numeric columns are passed through (don't need scaling or polynomial features for tree models)
        ], remainder='drop') # drop columns that are not specified
    else:
        # Apply to linear / NN models
        return ColumnTransformer([ 
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols), # dummy encoding
            ('num', Pipeline([ # runs num cols in serial process
                ('scale1', StandardScaler()), # 1. scale to mean 0 and variance 1
                ('poly',   PolynomialFeatures(degree=2, include_bias=False)), # 2. create polynomial features
                ('scale2', StandardScaler()), # 3. rescale
            ]), num_cols), # apply to numeric columns
        ], remainder='drop') # drop columns that are not specified


# ── Visualize data ────────────────────────────────────────────────────────────
def generate_distribution_figure(df, label):
    """
    Generate minimal distribution plots for all columns in a dataset.
    - Numeric columns: Visualized using histograms
    - Categorical columns: Visualized using horizontal bar charts (top 10 categories)
    
    This function is reusable for both raw and cleaned data. ✅
    """
    TOP_N_CATEGORIES = 10

    # 1. Identify column types to determine the appropriate plot type
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    all_cols = num_cols + cat_cols

    # 2. Set up a grid layout for the subplots
    ncols = 4 # We want 4 plots per row
    nrows = max(1, (len(all_cols) + ncols - 1) // ncols) # Calculate the number of rows needed to fit all columns
    
    # 3. Create the main figure and the subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    fig.suptitle(f"{label} Data — Distributions", fontsize=13, y=1.01)
    
    # Flatten the 2D array of axes into a 1D array for easier iteration
    axes_flat = np.array(axes).flatten()

    # 4. Iterate through all columns and plot the data
    for i, col in enumerate(all_cols):
        ax = axes_flat[i] # Get the specific subplot for this column
        
        # For numeric data: plot a histogram (removing NaN values first)
        if col in num_cols:
            # plot a histogram
            ax.hist(df[col].dropna(), bins=30)
            # style the axes
            ax.set_title(col, fontsize=10, pad=6)
            ax.set_xlabel(col, fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
        
        # For categorical data: get the top N most frequent categories
        else:
            # get the top N most frequent categories
            vc = df[col].value_counts().head(TOP_N_CATEGORIES)
            # plot a horizontal bar chart (reversing order to show largest at top)
            ax.barh(vc.index[::-1], vc.values[::-1])
            # style the axes
            ax.set_title(col, fontsize=10, pad=6)
            ax.set_xlabel('Count', fontsize=9)
            # make category labels slightly smaller
            ax.tick_params(axis='y', labelsize=7) 

    # 5. Hide extra unused subplots
    for j in range(len(all_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # 6. Adjust layout so plots do not overlap
    plt.tight_layout()
    
    return fig


# ── Save figure ───────────────────────────────────────────────────────────────
def save_figure(fig, label, out_path):
    """Save a Matplotlib figure to disk and close it."""
    # make sure the directory exists
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    # save the figure
    fig.savefig(out_path, bbox_inches='tight', facecolor="#1a1a2e", dpi=100)
    # close the figure
    plt.close(fig)
    print(f"[viz]   Saved {label} visualization → {out_path}")


# ── Save data statistics ──────────────────────────────────────────────────────
def save_data_stats(df, path, label="data"):
    """
    Compute and save descriptive statistics for a DataFrame to a JSON file.
    Call on both raw and clean data to enable data drift tracking across runs. ✅
    """
    # create a dictionary to store the statistics
    stats = {
        "label":     label,
        "timestamp": datetime.now().isoformat(),
        "shape":     {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns":   {},
    }

    # iterate through columns and compute statistics
    for col in df.columns:
        col_stats = {
            "dtype":        str(df[col].dtype),
            "null_count":   int(df[col].isna().sum()),
            "null_pct":     round(float(df[col].isna().mean()) * 100, 4),
            "unique_count": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                "mean": round(float(df[col].mean()),         4),
                "std":  round(float(df[col].std()),          4),
                "min":  round(float(df[col].min()),          4),
                "p25":  round(float(df[col].quantile(0.25)), 4),
                "p50":  round(float(df[col].quantile(0.50)), 4),
                "p75":  round(float(df[col].quantile(0.75)), 4),
                "max":  round(float(df[col].max()),          4),
            })
        else:
            top = df[col].mode()
            col_stats["mode"] = str(top.iloc[0]) if not top.empty else None
        # store the statistics
        stats["columns"][col] = col_stats

    # make sure the directory exists
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    # save the statistics
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    
    # print the statistics
    print(f"[stats] Saved {label} statistics → {path}  "
          f"({df.shape[0]:,} rows × {df.shape[1]} cols)")
