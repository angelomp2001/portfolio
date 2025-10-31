# libraries
from libraries import *

# Modules (files)
# from Interfaces.DataLoader import DataLoader
class DataLoader:
    @staticmethod
    def from_csv(paths: dict) -> dict:
        return {name: pd.read_csv(path) for name, path in paths.items()}

# from infrastructure.DataCleaner import DataCleaner
class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def replace_from_col(self, col: str, value, from_col: str):
        self.df.loc[self.df[col] == value, col] = self.df[from_col]

    def standardize_enddate(self, col: str):
        self.df[col] = self.df[col].where(self.df[col] == 'No', 'Yes')

    def fix_types(self, types: dict):
        for col, dtype in types.items():
            if 'datetime' in dtype:
                self.df[col] = pd.to_datetime(self.df[col])
            else:
                self.df[col] = self.df[col].astype(dtype)

# from infrastructure.FeatureEngineer import FeatureEngineer
class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def extract_date_parts(self, col: str):
        self.df['Year'] = self.df[col].dt.year
        self.df['Month'] = self.df[col].dt.month
        self.df['Day'] = self.df[col].dt.day

# from infrastructure.DataMerger import DataMerger
class DataMerger:
    @staticmethod
    def merge_all(dfs: dict, on='customerID', how='left') -> pd.DataFrame:
        merged_df = dfs['contract']
        for name, df in dfs.items():
            if name != 'contract':
                merged_df = merged_df.merge(df, on=on, how=how)
        return merged_df

# from infrastructure.ModelTrainer import ModelTrainer
class ModelTrainer:
    def __init__(self, features, k_folds: int, random_state: int):
        self.features = features
        self.categorical_features = features.select_dtypes(include=['object']).columns.tolist()
        self.numerical_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.k_folds = k_folds
        self.random_state = random_state
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ]
        )
        
        self.kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        self.models = [
            LogisticRegression(random_state=self.random_state),
            RandomForestClassifier(random_state=self.random_state),
            KNeighborsClassifier(),
            DecisionTreeClassifier(random_state=self.random_state),
            GradientBoostingClassifier(random_state=self.random_state),
            lgb.LGBMClassifier(random_state=self.random_state),
            xgb.XGBClassifier(random_state=self.random_state),
            CatBoostClassifier(verbose=0, random_state=self.random_state)
        ]
        


    def evaluate(self, target):
        self.target = target
        results = {}
        
        
        for model in self.models:
            pipe = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])


            model_name = model.__class__.__name__

            cv_results = cross_validate(
                estimator = pipe,
                X = self.features,
                y = self.target,
                cv=self.kf,
                scoring='roc_auc',
                return_train_score=True
            )
            train_auc = np.mean(cv_results['train_score'])
            test_auc = np.mean(cv_results['test_score'])
            results[model_name] = {'Train AUC': train_auc, 'Test AUC': test_auc}
        return results

# from infrastructure.NeuralNetTrainer import NeuralNetTrainer 
class NeuralNetTrainer:
    def __init__(self, features, k_folds: int, drop_rate: float):
        self.features = features
        self.k_folds = k_folds
        self.drop_rate = drop_rate
        self.kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=12345)

    def evaluate(self, target, preprocessor):
        train_scores = []
        test_scores = []

        # cross val score
        for train_idx, test_idx in self.kf.split(features):
            feature_train, feature_test = features.iloc[train_idx], features.iloc[test_idx]
            target_train, target_test = target.iloc[train_idx], target.iloc[test_idx]

            # Preprocess
            X_train = preprocessor.fit_transform(feature_train)
            X_test = preprocessor.transform(feature_test)

            # Build new model for each fold
            nn_model = Sequential([
                Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(self.drop_rate),
                Dense(256, activation='relu'),
                Dropout(self.drop_rate),
                Dense(128, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            nn_model.compile(optimizer=Adam(learning_rate=0.001),
                            loss='binary_crossentropy',
                            metrics=['AUC'])

            # Fit model
            nn_model.fit(X_train, target_train, epochs=30, batch_size=32, verbose=0)

            # Calc Predictions
            y_pred_train = nn_model.predict(X_train).ravel()
            y_pred_test = nn_model.predict(X_test).ravel()

            # Calc AUC
            train_auc = roc_auc_score(target_train, y_pred_train)
            test_auc = roc_auc_score(target_test, y_pred_test)
            train_scores.append(train_auc)
            test_scores.append(test_auc)

        return {'NeuralNetwork': {
            'Train AUC': np.mean(train_scores),
            'Test AUC': np.mean(test_scores)
        }}

# from interfaces.Output import Output
class Output:
    @staticmethod
    def to_console(data):
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{key:<30} Train AUC: {value['Train AUC']:.3f} | Test AUC: {value['Test AUC']:.3f}")
        elif isinstance(data, pd.DataFrame):
            print(data.head())
        else:
            print(f"Unsupported data type: {type(data)}")
        return data

# Main
if __name__ == "__main__":
    # from modules import DataLoader, DataCleaner, FeatureEngineer, DataMerger, Vectorizer, ModelTrainer, NeuralNetTrainer, Output

    # Load
    data_paths = {
        'contract': 'data/contract.csv',
        'internet': 'data/internet.csv',
        'personal': 'data/personal.csv',
        'phone': 'data/phone.csv'
    }
    dfs_dict = DataLoader.from_csv(data_paths)

    # Merge
    df = DataMerger.merge_all(dfs_dict)

    # Clean
    cleaner = DataCleaner(df)
    cleaner.replace_from_col(col = 'TotalCharges', value = " ", from_col= 'MonthlyCharges')
    cleaner.fix_types({'BeginDate': 'datetime', 'TotalCharges': 'float'})
    cleaner.standardize_enddate('EndDate')

    # Engineer
    engineer = FeatureEngineer(df)
    engineer.extract_date_parts('BeginDate')



    # Prepare data
    target_col = 'EndDate'
    target = df[target_col].replace({'No': 0, 'Yes': 1})
    features = df.drop([target_col, 'customerID'], axis=1)

    # Train classical models
    trainer = ModelTrainer(features=features, k_folds=5, random_state=12345)

    # results = trainer.evaluate(target) ✅

    # Train neural net
    nn_trainer = NeuralNetTrainer(features, k_folds=5, drop_rate=0.1)
    nn_results = nn_trainer.evaluate(target, trainer.preprocessor)

    # Output
    # Output.to_console(results) ✅
    Output.to_console(nn_results)
