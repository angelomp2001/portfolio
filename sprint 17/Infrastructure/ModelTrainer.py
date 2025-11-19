from libraries import *

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
            LogisticRegression(random_state=self.random_state, solver='liblinear'),
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
