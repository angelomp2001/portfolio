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

from src import charts


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
    if cols_to_drop is None:
        cols_to_drop = []

    df = df.copy()
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Price: values below minimum are invalid
    df[target_col] = np.where(df[target_col] >= price_min, df[target_col], np.nan)

    # RegistrationYear: sentinel 0 for out-of-range (keeps row)
    df['RegistrationYear'] = df['RegistrationYear'].where(
        (df['RegistrationYear'] >= year_min) & (df['RegistrationYear'] <= year_max)
    ).fillna(0)

    # Power: out-of-range → NaN (row dropped at dropna)
    df['Power'] = df['Power'].where(
        (df['Power'] >= power_min) & (df['Power'] <= power_max)
    )

    # Categorical NaN → explicit 'missing' category (inferred from dtype)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('missing')

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df


# ── Preprocessor factory ──────────────────────────────────────────────────────
def build_preprocessor(cat_cols, num_cols, is_tree=False):
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
    if is_tree:
        return ColumnTransformer([
            ('cat', OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
            ('num', 'passthrough', num_cols),
        ], remainder='drop')
    else:
        # Apply PolynomialFeatures only to numeric columns to avoid OHE explosion
        num_pipeline = Pipeline([
            ('scale1', StandardScaler()),
            ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
            ('scale2', StandardScaler()),
        ])
        return ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', num_pipeline, num_cols),
        ], remainder='drop')


# ── Visualize data ────────────────────────────────────────────────────────────
def visualize_data(df, label, out_path):
    """
    Plot histograms for numeric columns and bar charts for categorical ones.
    Saves to out_path. Reusable for both raw and clean data. ✅
    """
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    all_cols = num_cols + cat_cols

    ncols = 4
    nrows = max(1, (len(all_cols) + ncols - 1) // ncols)
    fig, axes = charts.new_figure(nrows, ncols, title=f"{label} Data — Distributions",
                                  figsize=(ncols * 4, nrows * 3))
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(all_cols):
        ax = axes_flat[i]
        if col in num_cols:
            ax.hist(df[col].dropna(), bins=30,
                    color=charts.COLORS[0], edgecolor=charts.BORDER)
            charts.style_axes(ax, title=col, xlabel=col, ylabel='Count')
        else:
            vc = df[col].value_counts().head(10)
            ax.barh(vc.index[::-1], vc.values[::-1],
                    color=charts.COLORS[1], edgecolor=charts.BORDER)
            charts.style_axes(ax, title=col, xlabel='Count')
            ax.tick_params(axis='y', labelsize=7)

    for j in range(len(all_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', facecolor=charts.BG, dpi=100)
    plt.close(fig)
    print(f"[viz]   Saved {label} visualization → {out_path}")


# ── Save data statistics ──────────────────────────────────────────────────────
def save_data_stats(df, path, label="data"):
    """
    Compute and save descriptive statistics for a DataFrame to a JSON file.
    Call on both raw and clean data to enable data drift tracking across runs. ✅
    """
    stats = {
        "label":     label,
        "timestamp": datetime.now().isoformat(),
        "shape":     {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns":   {},
    }

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
            col_stats["top_value"] = str(top.iloc[0]) if not top.empty else None

        stats["columns"][col] = col_stats

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[stats] Saved {label} statistics → {path}  "
          f"({df.shape[0]:,} rows × {df.shape[1]} cols)")
