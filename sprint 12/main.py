'''
Business case: Client wants to offer their customers an automated estiamte of what their car could sell for.  
Project: Predict 'Price'. 
'''

from src.data_preprocessing import *

path = 'data/car_data.csv'

# load data
df = load_data(path)
df = df.sample(10000, random_state = 12345)
print(df.shape)
print(df.columns)

# preprocess data
df, ordinal, categorical, df_train_regressions, df_valid_regressions, df_test_regressions, df_train_ML, df_valid_ML, df_test_ML, df_train_regressions_scaled, df_valid_regressions_scaled, df_test_regressions_scaled, \
        df_train_ML_scaled, df_valid_ML_scaled, df_test_ML_scaled, features_train_regressions_scaled, feature_valid_regressions_scaled, feature_test_regressions_scaled, \
        feature_train_ML_scaled, feature_valid_ML_scaled, feature_test_ML_scaled, target_train_reg_vectorized, target_valid_reg_vectorized, target_test_reg_vectorized, \
        target_train_ML_vectorized, target_valid_ML_vectorized, target_test_ML_vectorized = preprocess_data(df)


# Model training
model_training(features_train_regressions_scaled, target_train_reg_vectorized, df_valid_regressions, feature_train_ML_scaled, target_train_ML_vectorized, feature_valid_ML_scaled, target_valid_ML_vectorized, feature_test_ML_scaled, target_test_ML_vectorized)

'''
observation
Cat did just about the same. 

analysis/conclusion
See Stats above.
Cat model scored the best. Training time was highest for xgb, while prediction time is extremely small for all.
I chose cat for hyperparamter optimization because it had the smallest RMSE.

Conclusion: The task was to predict Price using available data. Features were reviewed one at a time and for each action in the following order:

Edit values
either correct, or identify as missing Update data types
eg. change date from string to datetime Remove missing data
either relabel as 'missing' or set to NA (for dropping later) Remove irrelevant columns
drop the column if having 1:1 unique values/row, or no variation. Feature engineering
adding columns as a function of other columns. ie. creating a month or day column from a date column.
the data was then processed as a whole with this following actions in this order: dropped duplicate rows Drop rows with missing values Moved target to first column encoding feature scaling/vectorization target vectorization

Once both features and target were in a vectorized format, we fit the following models: LinearRegression RandomForestRegressor lgb.LGBMRegressor cb.CatBoostRegressor xgb.XGBRegressor

Training time, prediction time, and RMSE were measured for each model: linear_rmse: 2724.0585558060775 lgb_rmse: 2063.187614013004 rfr_rmse: 2176.8222270351207 cat_rmse: 2026.1914059369433 xgb_rmse:2236.4188514668167

linear_train/pred: (0.25896334648132324, 0.025099754333496094) lgb_train/pred: (2.220857858657837, 0.04092001914978027) rfr_train/pred: (1.5917150974273682, 0.04019355773925781) cat_train/pred: (1.093503475189209, 0.0024204254150390625) xgb_train/pred:(14.426959037780762, 0.003343343734741211)

CAT had he lowest RMSE. It was further lowered to when hyperparameter: 'max_depth' was optimized at 4: max_depth: 4, rmse: 2008.1364553366802

It was also the second fastest model to train, so I choose CAT as the best model.
'''
