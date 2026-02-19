# Main
if __name__ == "__main__":
    from Interfaces.DataLoader import DataLoader
    from Infrastructure.DataCleaner import DataCleaner
    from Infrastructure.FeatureEngineer import FeatureEngineer
    from Infrastructure.DataMerger import DataMerger
    from Infrastructure.ModelTrainer import ModelTrainer
    from Infrastructure.NeuralNetTrainer import NeuralNetTrainer
    from Interfaces.Output import Output

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

    # Train neural net
    nn_trainer = NeuralNetTrainer(features, k_folds=5, drop_rate=0.1)
    
    # evaluate classic models
    results = trainer.evaluate(target)

    # evaluate nn
    nn_results = nn_trainer.evaluate(target, trainer.preprocessor)

    # Output both results
    Output.to_console(results)
    Output.to_console(nn_results)

'''
LogisticRegression             Train AUC: 0.841 | Test AUC: 0.838
RandomForestClassifier         Train AUC: 1.000 | Test AUC: 0.809
KNeighborsClassifier           Train AUC: 0.897 | Test AUC: 0.772
DecisionTreeClassifier         Train AUC: 1.000 | Test AUC: 0.662
GradientBoostingClassifier     Train AUC: 0.881 | Test AUC: 0.845
LGBMClassifier                 Train AUC: 0.950 | Test AUC: 0.834
XGBClassifier                  Train AUC: 0.983 | Test AUC: 0.820
CatBoostClassifier             Train AUC: 0.939 | Test AUC: 0.838
NeuralNetwork                  Train AUC: 0.980 | Test AUC: 0.764
'''
