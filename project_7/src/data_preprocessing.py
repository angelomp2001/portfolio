import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PolynomialFeatures, MinMaxScaler
from src.config import RANDOM_STATE

def load_data(sample=None):
    df = pd.read_csv("data/data.csv")
    if sample:
        df = df.sample(n=sample, random_state=RANDOM_STATE).reset_index(drop=True)
    return df

def save_stats(df, label="raw"):
    os.makedirs("docs", exist_ok=True)
    stats = {
        "timestamp": datetime.now().isoformat(),
        "shape": df.shape,
    }
    # Describe includes everything
    describe_dict = df.describe(include="all").to_dict()
    # Handle NaN in dictionary for JSON serialization
    for k, v in describe_dict.items():
        for k2, v2 in v.items():
            if pd.isna(v2):
                describe_dict[k][k2] = None

    stats.update(describe_dict)
    
    with open(f"docs/data_stats_{label}.json", "w") as f:
        json.dump(stats, f, indent=4)

def save_charts(df, label="raw"):
    os.makedirs("docs", exist_ok=True)
    # pairplot might be too slow for raw if huge, but let's just do histograms first
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(exclude='number').columns
    
    if len(num_cols) > 0:
        if label == "raw":
            df[num_cols].hist(figsize=(10, 8))
            plt.tight_layout()
            plt.savefig(f"docs/data_charts_{label}_hist.png")
            plt.close()
        elif label == "clean":
            sns.pairplot(df[num_cols].sample(min(1000, len(df)), random_state=RANDOM_STATE))
            plt.savefig(f"docs/data_charts_{label}_pairplot.png")
            plt.close()
            
    if len(cat_cols) > 0:
        plt.figure(figsize=(10, 8))
        for i, col in enumerate(cat_cols):
            plt.subplot(len(cat_cols), 1, i + 1)
            sns.countplot(x=df[col].dropna())
        plt.tight_layout()
        plt.savefig(f"docs/data_charts_{label}_counts.png")
        plt.close()

def clean_data(df):
    """
    Clean data handling missing values, duplicates and type casts.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(exclude='number').columns
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # handle missing 
    if len(num_cols) > 0:
        df.dropna(subset=num_cols, inplace=True)
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna('missing')
        
    return df.reset_index(drop=True)

def data_types(df):
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()
    return num_cols, cat_cols

def column_preprocessor(num_cols, cat_cols, is_tree):
    """
    Return a scikit-learn ColumnTransformer
    """
    if is_tree:
        cat_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        num_transformer = 'passthrough'
    else:
        cat_transformer = OneHotEncoder(handle_unknown='ignore')
        # Simple pipeline for linear model variables
        num_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
    transformers = []
    if len(cat_cols) > 0:
        transformers.append(('cat', cat_transformer, cat_cols))
    if len(num_cols) > 0:
        transformers.append(('num', num_transformer, num_cols))
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
    return preprocessor

