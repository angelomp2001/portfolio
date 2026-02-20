# libraries
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

def load_data(path):
    return pd.read_csv(path)




def preprocess_data(df):
    ## EDA
    # track ordinal and categorical cols. 
    ordinal = []
    categorical = []

    # col 0
    column = 0
    print(df.columns[column]) # 'DateCrawled'

    # understand values:
    print(df.iloc[:,column].dtype)
    #dtype: int64
    print(df.iloc[:,column].head())
    #sns.histplot(df.iloc[:,column], bins=30, kde=False)
    #plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {354369 - len(df.iloc[:,column])}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # numberical


    # changes:
    # Edit values
    # Update data types
    # Remove missing data
    # Remove irrelevant columns
    df = df.drop(df.columns[column],axis = 1)
    # Feature engineering


    # QC
    print(df.columns.tolist())

    # EDA
    # new col 0:
    column = 0
    print(df.columns[column]) # 'Price'

    # understand values:
    print(df.iloc[:,column].dtype) #dtype: int64
    print(df.iloc[:,column].head())
    df.iloc[:, column].plot(kind='hist', bins=20)
    print(df.iloc[:,column].describe())
    print(f'n 0s: {(df.iloc[:,column] == 0).sum()}, {(df.iloc[:,column] == 0).sum()/(df.iloc[:,column]).count()}')
    print(f'missing: {354369 - len(df.iloc[:,column])}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # numberical

    # changes:
    # Edit values
    df.iloc[:,column] = np.where((df.iloc[:,column] >= 500), df.iloc[:,column], np.nan)
    # Update data types
    # Remove missing data
    # Remove irrelevant columns
    # Feature engineering

    # QC
    print(f'n 0s: {(df.iloc[:,column] == 0).sum()}, {(df.iloc[:,column] == 0).sum()/(df.iloc[:,column]).count()}')
    print(df.iloc[:,column].describe())


    # observations
    # missing might be set at zero.  Otherwise, no odd values.  No designated missing values.

    # EDA
    # col 1
    column = 1
    print(df.columns[column]) # 'VehicleType'

    # understand values:
    print(df.iloc[:,column].dtype)
    #dtype: object (text)
    print(df.iloc[:,column].head())
    df.iloc[:,column].value_counts().head(30).plot(kind='bar')
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # categorical
    categorical.append(df.columns[column])

    # changes:
    # Edit values
    # Update data types
    # Remove missing data
    df.iloc[:,column] = df.iloc[:,column].fillna('missing')
    # Remove irrelevant columns
    # Feature engineering

    # QC
    print(f'missing: {df.iloc[:, column].isnull().sum()}')

    # observations
    # missing values saved as 'missing' to include the row of data. 


    # EDA
    # col 2
    column = 2
    print(df.columns[column]) # 'RegistrationYear'

    # understand values:
    print(df.iloc[:,column].dtype)
    #dtype: int64
    print(df.iloc[:,column].head())
    df.iloc[:, column].plot(kind='hist', bins=30)
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # numerical


    # changes:
    # Edit values
    df.iloc[:,column] = df.iloc[:,column].where((df.iloc[:,column] >= 1900) & (df.iloc[:,column] <= 2025))
    # Update data types
    # Remove missing data
    df.iloc[:,column] = df.iloc[:,column].fillna(0) # this keeps the row, but 0 should clearly be understood as missing
    # Remove irrelevant columns
    # Feature engineering

    # QC
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')


    # observations
    # No missing, but certainly odd values. I turned all values outside of the range 1900-2025 to 0.  I didn't want to leave as NaN to lose the row.  I can't label it as 'missing' or it would change the dtype.  Not sure what a better solution is.


    # EDA
    # col 3
    column = 3
    print(df.columns[column]) # 'Gearbox'

    # understand values:
    print(df.iloc[:,column].dtype)
    #dtype: object
    print(df.iloc[:,column].head())
    df.iloc[:,column].value_counts().plot(kind='bar')
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # categorical
    categorical.append(df.columns[column])


    # changes:
    # Edit values
    # Update data types
    # Remove missing data
    df.iloc[:,column] = df.iloc[:,column].fillna('missing')
    # Remove irrelevant columns
    # Feature engineering

    # QC
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    df.iloc[:,column].value_counts().plot(kind='bar')

    # observations
    # missing saved as 'missing' to keep the row.   

    # EDA
    # col 4
    column = 4
    # print(df.columns[column]) # 'Power'

    # understand values:
    print(df.iloc[:,column].dtype)
    #dtype: int64
    # print(np.sort(df.iloc[:,column].unique()))

    df.iloc[:, column].plot(kind='hist', bins=20)
    plt.show()

    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # numerical



    # changes:
    # Edit values
    df.iloc[:,column] = df.iloc[:,column].where((df.iloc[:,column] >= 100) & (df.iloc[:,column] <= 400))
    # Update data types
    # Remove missing data
    # df.iloc[:,column] = df.iloc[:,column].fillna(0) # this keeps the row, but 0 should clearly be understood as missing
    # Remove irrelevant columns
    # Feature engineering

    # QC
    print(df.iloc[:,column].describe())
    df.iloc[:, column].plot(kind='hist', bins=20)
    plt.show()
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # observations
    # There is no missing, but these values don't make sense to me. I set all values outside of 100-400 to zero. 


    # EDA
    # col 5
    column = 5
    print(df.columns[column]) # 'Model'

    # understand values:
    print(df.iloc[:,column].dtype)
    #dtype: object
    print((df.iloc[:,column].value_counts().sort_values(ascending = False)))
    df.iloc[:,column].value_counts().plot(kind='bar')
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # categorical
    categorical.append(df.columns[column])


    # changes:
    # Edit values
    # Update data types
    # Remove missing data
    df.iloc[:,column] = df.iloc[:,column].fillna('missing')
    # Remove irrelevant columns
    # Feature engineering

    # QC
    print(f'missing: {df.iloc[:, column].isnull().sum()}')


    # EDA
    # col 6
    column = 6
    print(df.columns[column]) # 'Mileage'

    # understand values:
    print(df.iloc[:,column].dtype)
    #dtype: int64
    print(np.sort(df.iloc[:,column].unique()))
    df.iloc[:, column].plot(kind='hist', bins=20)
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # numerical



    # changes:
    # Edit values
    # Update data types
    # Remove missing data
    # Remove irrelevant columns
    # Feature engineering

    # QC
    print(f'missing: {df.iloc[:, column].isnull().sum()}')


    # EDA
    # col 7
    column = 7
    print(df.columns[column]) # 'RegistrationMonth'

    # understand values:
    print(df.iloc[:,column].dtype) # int64
    print(np.sort(df.iloc[:,column].unique()))
    df.iloc[:,column].value_counts().plot(kind='bar')
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.loc[df.iloc[:, column] == 0,df.columns[column]].count()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # numerical


    # changes:
    # Edit values
    df.iloc[:, column] = df.iloc[:, column].replace(0, np.nan)
    # Update data types
    # Remove missing data
    # Remove irrelevant columns
    df = df.drop(df.columns[column],axis = 1)
    # Feature engineering

    # QC
    print(f'missing: {df.iloc[:, column].isnull().sum()}')

    # there are 13 options for registration month, I assume 0 = missing and will drop it in the end. 
    # I ended up dropping the column because I don't see how this could correlate to Price. 


    # EDA
    # new col 7
    column = 7
    print(df.columns[column]) # 'FuelType'

    # understand values:
    print(df.iloc[:,column].dtype) # object
    print((df.iloc[:,column].unique()))
    df.iloc[:,column].value_counts().plot(kind='bar')
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # categorical
    categorical.append(df.columns[column])




    # changes:
    # Edit values
    df.iloc[:, column] = df.iloc[:, column].replace(np.nan, 'missing')
    # Update data types
    # Remove missing data
    # Remove irrelevant columns
    # Feature engineering

    # QC
    print(f'missing: {df.iloc[:, column].isnull().sum()}')


    # EDA
    # col 8
    column = 8
    print(df.columns[column]) # 'Brand'

    # understand values:
    print(df.iloc[:,column].dtype) # object
    print((df.iloc[:,column].unique()))
    df.iloc[:,column].value_counts().plot(kind='bar')
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # categorical
    categorical.append(df.columns[column])


    # changes:
    # Edit values
    # Update data types
    # Remove missing data
    # Remove irrelevant columns
    # Feature engineering

    # QC
    print(f'missing: {df.iloc[:, column].isnull().sum()}')


    # EDA
    # col 9
    column = 9
    print(df.columns[column]) # 'NotRepaired'

    # understand values:
    print(df.iloc[:,column].dtype) # object
    print((df.iloc[:,column].unique()))
    df.iloc[:,column].value_counts().plot(kind='bar')
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # categorical
    categorical.append(df.columns[column])


    # changes:
    # Edit values
    df.iloc[:, column] = df.iloc[:, column].fillna('missing')
    # Update data types
    # Remove missing data
    # Remove irrelevant columns
    # Feature engineering

    # QC
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    df.iloc[:,column].value_counts().plot(kind='bar')
    plt.show()


    # EDA
    # col 10
    column = 10
    print(df.columns[column]) # 'DateCreated'

    # understand values:
    print(df.iloc[:,column].dtype) # object
    print((df.iloc[:,column].unique()))
    df.iloc[:,column].value_counts().plot(kind='bar')
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # numerical


    # changes:
    # Edit values
    # Update data types
    # Remove missing data
    # Remove irrelevant columns
    df.drop('DateCreated', axis=1, inplace=True)
    # Feature engineering

    # QC
    print("Deleted" if 'DateCreated' not in df.columns else "Not deleted")



    # EDA
    # new col 10
    column = 10
    print(df.columns[column]) # 'NumberOfPictures'

    # understand values:
    print(df.iloc[:,column].dtype) # int64
    print((df.iloc[:,column].unique()))
    df.iloc[:,column].value_counts().plot(kind='bar')
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')


    # changes:
    # Edit values
    # Update data types
    # Remove missing data
    # Remove irrelevant columns
    df.drop('NumberOfPictures', axis=1, inplace=True)
    # Feature engineering

    # QC
    print("Deleted" if 'NumberOfPictures' not in df.columns else "Not deleted")


    # EDA
    # new col 10
    column = 10
    print(df.columns[column]) # 'PostalCode'

    # understand values:
    print(df.iloc[:,column].dtype) # object
    print((df.iloc[:,column].unique()))
    df.iloc[:,column].value_counts().plot(kind='bar')
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # categorical


    # changes:
    # Edit values
    # Update data types
    # df.iloc[:, column] = df.iloc[:, column].astype(str)
    # Remove missing data
    # Remove irrelevant columns - too many unique values
    df.drop('PostalCode', axis=1, inplace=True)

    # Feature engineering

    # QC
    print("Deleted" if 'PostalCode' not in df.columns else "Not deleted")


    # EDA
    # new col 10
    column = 10
    print(df.columns[column]) # 'LastSeen'

    # understand values:
    print(df.iloc[:,column].dtype) # object
    print((df.iloc[:,column].unique()))
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # numerical


    # changes:
    # Edit values
    # Update data types
    df.iloc[:, column] = pd.to_datetime(df.iloc[:, column], errors='coerce')
    # Remove missing data
    # Feature engineering
    

    df['LastSeen_month'] = pd.to_datetime(df['LastSeen']).dt.month
    df['LastSeen_day'] = pd.to_datetime(df['LastSeen']).dt.day
    # Remove irrelevant columns
    df.drop('LastSeen', axis=1, inplace=True)

    # drop
    print("Deleted" if 'LastSeen' not in df.columns else "Not deleted")

    # QC
    print(df.iloc[:,column].dtype)
    print(df[f'LastSeen_month'].head())
    print(df[f'LastSeen_day'].head())

    # EDA
    # new col 10
    column = 10
    print(df.columns[column]) # 'LastSeen_month'

    # understand values:
    print(df.iloc[:,column].dtype) # int64
    print((df.iloc[:,column].unique()))
    df.iloc[:,column].value_counts().plot(kind='bar')
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # numerical


    # changes:
    # Edit values
    # Update data types
    # Remove missing data
    # Remove irrelevant columns
    df = df.drop(df.columns[column],axis = 1)
    # Feature engineering

    # QC
    # decided to remove. 

    # EDA
    # new col 10
    column = 10
    print(df.columns[column]) # 'LastSeen_day'

    # understand values:
    print(df.iloc[:,column].dtype) # int64
    print((df.iloc[:,column].unique()))
    df.iloc[:,column].value_counts().plot(kind='bar')
    plt.show()
    print(df.iloc[:,column].describe())
    print(f'missing: {df.iloc[:, column].isnull().sum()}')
    # how I will treat the col for encoding (numerical, ordinal, categorical):
    # numerical

    # changes:
    # Edit values
    # Update data types
    # Remove missing data
    # Remove irrelevant columns
    df = df.drop(df.columns[column],axis = 1)
    # Feature engineering

    # QC
    # decided not to keep

    # row clean up
    # drop duplicates in the end
    df.drop_duplicates(inplace = True)

    #dropna
    df.dropna(inplace = True)

    # move target to be first column
    target = 'Price'
    df = df[ [target] + [col for col in df.columns if col != target]]

    ## prepare for vectorization
    # Split data
    random_state = 12345
    train_ratio = .6
    valid_ratio = .2
    test_ratio = .2
    # data.split(split_ratio=(0.6, 0.2, 0.2), target_name=target_name, random_state=random_state).vectorize()

    # First split: separate out the test set.
    df_temp, df_test = train_test_split(df, test_size=test_ratio, random_state=random_state)

    # Recalculate validation ratio relative to the remaining data (df_temp).
    valid_ratio = valid_ratio / (1 - test_ratio)

    df_train, df_valid = train_test_split(df_temp, test_size=valid_ratio, random_state=random_state)

    # encode data - regression
    # 'Regressions':
    # -- One-hot encodes categorical_cols. 
    # -- Ordinal  encoding for ordinal cols. 
    # 'Machine Learning': 
    # -- Ordinal encodes for both ordinal cols and categorical cols.

    # Regression model data
    df_train_regressions = df_train.copy()
    df_valid_regressions = df_valid.copy()
    df_test_regressions = df_test.copy()


    # initialize OHE
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # did not drop='first' because valid/train has other categories. 

    # generate OHE array
    train_encoded = ohe.fit_transform(df_train[categorical])
    print('shape: ',train_encoded.shape) # (3214, 244)
    # Get OHE array feature names to make df
    ohe_cols = ohe.get_feature_names_out(categorical)
  
    print('shape: ',ohe_cols.shape) # (244,)
    # convert OHE array to df using feature names
    train_encoded_df = pd.DataFrame(train_encoded, columns=ohe_cols, index=df_train.index) # 

    # drop old df_train feature cols and add new OHE array
    df_train_regressions = pd.concat([df_train.drop(columns=categorical), train_encoded_df], axis=1)


    # apply array to valid and test and update respective dfs
    # valid
    valid_encoded = ohe.transform(df_valid[categorical])
    valid_encoded_df = pd.DataFrame(valid_encoded, columns=ohe_cols, index=df_valid.index)
    df_valid_regressions = pd.concat([df_valid.drop(columns=categorical), valid_encoded_df], axis=1)

    # test
    test_encoded = ohe.transform(df_test[categorical])
    test_encoded_df  = pd.DataFrame(test_encoded,  columns=ohe_cols, index=df_test.index)
    df_test_regressions  = pd.concat([df_test.drop(columns=categorical),  test_encoded_df],  axis=1)

    # QC
    print(f'{df_train_regressions.columns},\n {df_valid_regressions.columns},\n {df_test_regressions.columns}')

    #Encode data - ML
    # ML model data
    df_train_ML = df_train.copy()
    df_valid_ML = df_valid.copy()
    df_test_ML = df_test.copy()
    dfs = [df_train_ML, df_valid_ML, df_test_ML]

    # label encode
    text_values_dict = {'ordinal': {}, 'categorical': {}}

    for each_df in dfs:
        for each_col in categorical:
            
            # get unique values
            unique_values = sorted(each_df[each_col].dropna().unique())
            
            # make a dictionary of category: value
            mapping_dict = {val: idx for idx, val in enumerate(unique_values)}
            
            # Replace col val with idx
            each_df[each_col] = each_df[each_col].map(mapping_dict)
            
            # Save dict to text_values_dict tracker
            text_values_dict['categorical'][each_col] = mapping_dict

    # QC
    print(df_train_ML.shape, df_valid_ML.shape, df_test_ML.shape)

    # scale features

    # duplicate df
    # regression
    df_train_regressions_scaled = df_train_regressions.copy()
    df_valid_regressions_scaled = df_valid_regressions.copy()
    df_test_regressions_scaled = df_test_regressions.copy()

    # ML
    df_train_ML_scaled = df_train_ML.copy()
    df_valid_ML_scaled = df_valid_ML.copy()
    df_test_ML_scaled = df_test_ML.copy()

    # initialize scaler
    scaler = StandardScaler()

    # define feature and target
    # regression
    df_train_regressions_scaled.columns = df_train_regressions_scaled.columns.map(str)
    df_train_regressions_scaled.columns = df_train_regressions_scaled.columns.map(str)

    target_name_reg = df_train_regressions_scaled.columns[0]
    features_name_reg = df_train_regressions_scaled.columns[1:]

    # scale train features - regression
    print(f'dtypes:' ,df_train_regressions_scaled[features_name_reg].dtypes)
    features_train_regressions_scaled = scaler.fit_transform(df_train_regressions_scaled[features_name_reg])

    print(f'features_name_reg: {features_name_reg}')
    # apply same scaler to valid and test features
    feature_valid_regressions_scaled = scaler.transform(df_valid_regressions_scaled[features_name_reg])
    feature_test_regressions_scaled = scaler.transform(df_test_regressions_scaled[features_name_reg])

    # ML

    # define feature and target
    # ML
    target_name_ML = df_train_ML_scaled.columns[0]
    features_name_ML = df_train_ML_scaled.columns[1:]

    # scale train features - ML
    feature_train_ML_scaled = scaler.fit_transform(df_train_ML_scaled[features_name_ML])

    # apply scaler to valid and test features
    feature_valid_ML_scaled = scaler.transform(df_valid_ML_scaled[features_name_ML])
    feature_test_ML_scaled = scaler.transform(df_test_ML_scaled[features_name_ML])

    # vectorize y (x already vectorized via scaling)
    # regression
    target_train_reg_vectorized = df_train_regressions_scaled['Price'].to_numpy()
    target_valid_reg_vectorized = df_valid_regressions_scaled['Price'].to_numpy()
    target_test_reg_vectorized = df_test_regressions_scaled['Price'].to_numpy()

    # ML
    target_train_ML_vectorized = df_train_ML_scaled['Price'].to_numpy()
    target_valid_ML_vectorized = df_valid_ML_scaled['Price'].to_numpy()
    target_test_ML_vectorized = df_test_ML_scaled['Price'].to_numpy()

    return df, ordinal, categorical, \
           df_train_regressions, df_valid_regressions, df_test_regressions, \
           df_train_ML, df_valid_ML, df_test_ML, df_train_regressions_scaled, df_valid_regressions_scaled, df_test_regressions_scaled, \
        df_train_ML_scaled, df_valid_ML_scaled, df_test_ML_scaled, \
        features_train_regressions_scaled, feature_valid_regressions_scaled, feature_test_regressions_scaled, \
        feature_train_ML_scaled, feature_valid_ML_scaled, feature_test_ML_scaled, \
        target_train_reg_vectorized, target_valid_reg_vectorized, target_test_reg_vectorized, \
        target_train_ML_vectorized, target_valid_ML_vectorized, target_test_ML_vectorized 

# model_training() has been moved to src/model_training.py
