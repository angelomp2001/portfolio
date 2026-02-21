import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────
# Price-related constants
PRICE_MIN       = 500

# Year range for registration
YEAR_MIN, YEAR_MAX = 1900, 2025

# Power range for cars
POWER_MIN, POWER_MAX = 100, 400

# Random state for reproducibility
RANDOM_STATE = 12345

# Ratios for splitting the dataset
TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.6, 0.2, 0.2

COLS_TO_DROP    = ['DateCrawled', 'RegistrationMonth', 'DateCreated',
                   'NumberOfPictures', 'PostalCode', 'LastSeen']
CATEGORICAL_COLS = ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'NotRepaired']


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data(path):
    return pd.read_csv(path)


# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess_data(df):
    """
    Clean, encode, split, and scale the car dataset.

    Returns a nested dict with keys: 'df', 'categorical', 'regression', 'ml'.
    Each model-type key contains: 'train', 'valid', 'test', and 'scaled'.
    'scaled' contains 'features' and 'targets' for each split.
    """

    # ── Drop irrelevant columns ───────────────────────────────────────────────
    df = df.drop(columns=COLS_TO_DROP, errors='ignore')

    # ── Clean numeric columns ─────────────────────────────────────────────────
    # Price: values below PRICE_MIN are likely invalid/missing
    df['Price'] = np.where(df['Price'] >= PRICE_MIN, df['Price'], np.nan)

    # RegistrationYear: out-of-range values treated as missing; sentinel 0 keeps the row
    df['RegistrationYear'] = df['RegistrationYear'].where(
        (df['RegistrationYear'] >= YEAR_MIN) & (df['RegistrationYear'] <= YEAR_MAX)
    ).fillna(0)

    # Power: out-of-range values treated as NaN (row removed later in dropna)
    df['Power'] = df['Power'].where(
        (df['Power'] >= POWER_MIN) & (df['Power'] <= POWER_MAX)
    )

    # ── Handle missing values in categorical columns ───────────────────────────
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna('missing')

    # ── Row-level cleanup ─────────────────────────────────────────────────────
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # ── Move target to first column ───────────────────────────────────────────
    target = 'Price'
    df = df[[target] + [col for col in df.columns if col != target]]

    # ── Train / Validation / Test split ──────────────────────────────────────
    df_temp, df_test = train_test_split(df, test_size=TEST_RATIO, random_state=RANDOM_STATE)
    adjusted_valid_ratio = VALID_RATIO / (1 - TEST_RATIO)
    df_train, df_valid = train_test_split(df_temp, test_size=adjusted_valid_ratio, random_state=RANDOM_STATE)

    categorical = CATEGORICAL_COLS

    # ── Encode: Regression (One-Hot Encoding) ─────────────────────────────────

    # Instantiate OneHotEncoder with specified options
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Creates an encoder to handle categorical features

    # Fit the encoder on the training data's categorical features and get feature names
    ohe_cols = ohe.fit(df_train[categorical]).get_feature_names_out(categorical)  # Returns array of new OHE column names

    # Define a function to transform a DataFrame using One-Hot Encoding
    def _ohe_transform(source_df, fit=False):
        # Apply OHE to source_df
        encoded = ohe.fit_transform(source_df[categorical]) if fit else ohe.transform(source_df[categorical])  # Encodes categorical features (dtype: ndarray)
        
        # Create DataFrame from the encoded array
        encoded_df = pd.DataFrame(encoded, columns=ohe_cols, index=source_df.index)  # New DataFrame with OHE columns (dtype: DataFrame)
        
        # Create new DataFrame by concatenating the original DataFrame (without categorical columns) with the newly encoded DataFrame
        return pd.concat([source_df.drop(columns=categorical), encoded_df], axis=1)  # Returns DataFrame with numeric and OHE columns

    # Fit the encoder on the training data's categorical features (prep for transforming)
    ohe.fit(df_train[categorical])  # Fits encoder to train data (captures categories)

    # Transform the training DataFrame using the One-Hot Encoder; do not fit again
    df_train_regressions = _ohe_transform(df_train, fit=False)  # DataFrame with OHE applied (e.g., shape: (n_samples, n_features))

    # Transform the validation DataFrame using the previously fitted One-Hot Encoder
    df_valid_regressions = _ohe_transform(df_valid)  # DataFrame with OHE applied, following same feature set as training

    # Transform the test DataFrame using the previously fitted One-Hot Encoder
    df_test_regressions  = _ohe_transform(df_test)  # DataFrame with OHE applied, ensures same encoding for comparison

    # ── Encode: ML (Label Encoding) ───────────────────────────────────────────
    df_train_ML = df_train.copy()
    df_valid_ML = df_valid.copy()
    df_test_ML  = df_test.copy()

    for each_df in [df_train_ML, df_valid_ML, df_test_ML]:
        for col in categorical:
            unique_values  = sorted(each_df[col].dropna().unique())
            mapping_dict   = {val: idx for idx, val in enumerate(unique_values)}
            each_df[col]   = each_df[col].map(mapping_dict)

    # ── Scale features ────────────────────────────────────────────────────────
    scaler = StandardScaler()

    # Regression
    df_train_regressions_scaled = df_train_regressions.copy()
    df_valid_regressions_scaled = df_valid_regressions.copy()
    df_test_regressions_scaled  = df_test_regressions.copy()

    # ensure column names are strings
    df_train_regressions_scaled.columns = df_train_regressions_scaled.columns.map(str)
    df_valid_regressions_scaled.columns = df_valid_regressions_scaled.columns.map(str)
    df_test_regressions_scaled.columns  = df_test_regressions_scaled.columns.map(str)

    # Get feature names (all columns except the first one, which is the target)
    features_name_reg = df_train_regressions_scaled.columns[1:]
    
    # Fit the scaler on the training data and transform it
    features_train_regressions_scaled = scaler.fit_transform(df_train_regressions_scaled[features_name_reg])
    
    # Transform the validation and test data using the same scaler
    feature_valid_regressions_scaled  = scaler.transform(df_valid_regressions_scaled[features_name_reg])
    feature_test_regressions_scaled   = scaler.transform(df_test_regressions_scaled[features_name_reg])

    # ML
    df_train_ML_scaled = df_train_ML.copy()
    df_valid_ML_scaled = df_valid_ML.copy()
    df_test_ML_scaled  = df_test_ML.copy()

    # Get feature names (all columns except the first one, which is the target)
    features_name_ML = df_train_ML_scaled.columns[1:]
    
    # Fit the scaler on the training data and transform it
    feature_train_ML_scaled = scaler.fit_transform(df_train_ML_scaled[features_name_ML])
    
    # Transform the validation and test data using the same scaler
    feature_valid_ML_scaled = scaler.transform(df_valid_ML_scaled[features_name_ML])
    feature_test_ML_scaled  = scaler.transform(df_test_ML_scaled[features_name_ML])

    # ── Vectorize targets ─────────────────────────────────────────────────────
    return {
        'df':          df,
        'categorical': categorical,
        'regression': {
            'train': df_train_regressions,
            'valid': df_valid_regressions,
            'test':  df_test_regressions,
            'scaled': {
                'train': df_train_regressions_scaled,
                'valid': df_valid_regressions_scaled,
                'test':  df_test_regressions_scaled,
                'features': {
                    'train': features_train_regressions_scaled,
                    'valid': feature_valid_regressions_scaled,
                    'test':  feature_test_regressions_scaled,
                },
                'targets': {
                    'train': df_train_regressions_scaled['Price'].to_numpy(),
                    'valid': df_valid_regressions_scaled['Price'].to_numpy(),
                    'test':  df_test_regressions_scaled['Price'].to_numpy(),
                },
            },
        },
        'ml': {
            'train': df_train_ML,
            'valid': df_valid_ML,
            'test':  df_test_ML,
            'scaled': {
                'train': df_train_ML_scaled,
                'valid': df_valid_ML_scaled,
                'test':  df_test_ML_scaled,
                'features': {
                    'train': feature_train_ML_scaled,
                    'valid': feature_valid_ML_scaled,
                    'test':  feature_test_ML_scaled,
                },
                'targets': {
                    'train': df_train_ML_scaled['Price'].to_numpy(),
                    'valid': df_valid_ML_scaled['Price'].to_numpy(),
                    'test':  df_test_ML_scaled['Price'].to_numpy(),
                },
            },
        },
    }

# model_training() has been moved to src/model_training.py
