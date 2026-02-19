import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
from src.results import results
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from src.train_test_split import train_test_split



# initialize scores
scores_dict = {'Test':[], 'normalize':[], 'lemmatize':[], 'stopword':[], 'tokenizer':[], 'model':[], 'rows':[], 'F1':[], 'ROC AUC':[], 'APS':[], 'Accuracy':[]}


def test_0(main_params, train_target, test_target):
    scores_dict['Test'].append(0)

    # Baseline prediction: predict the mean as probability, and threshold at 0.5 for class labels
    baseline_prob = pd.Series(int(train_target.mean()), index =test_target.index) 
    baseline_pred = (baseline_prob >= 0.5).astype(int)

    # copy main_params to scores_dict
    for key in main_params:
        scores_dict[key].append(main_params[key])

    # calculate and append metrics to scores_dict
    scores_dict['F1'].append(round(f1_score(test_target, baseline_pred),2))
    scores_dict['ROC AUC'].append(round(roc_auc_score(test_target, baseline_prob),2))
    scores_dict['APS'].append(round(average_precision_score(test_target, baseline_prob),2))
    scores_dict['Accuracy'].append(round(accuracy_score(test_target, baseline_pred),2))

    # convert dict to df and save to csv
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_1(main_params, other_params):
    scores_dict['Test'].append(1)
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}

    # rename model from class type to string type
    scores_dict['model'][-1] = main_params['model'].__name__

    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_2(main_params, other_params):
    scores_dict['Test'].append(2)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_3(main_params, other_params):
    scores_dict['Test'].append(3)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_4(main_params, other_params):
    scores_dict['Test'].append(4)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_5(main_params, other_params):
    scores_dict['Test'].append(5)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_6(main_params, other_params):
    scores_dict['Test'].append(6)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_7(main_params, other_params):
    scores_dict['Test'].append(7)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_8(main_params, other_params):
    scores_dict['Test'].append(8)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

def test_9(main_params, other_params):
    
    scores_dict['Test'].append(9)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

def test_10(main_params, other_params):
    
    scores_dict['Test'].append(10)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_11(main_params, other_params):
    
    scores_dict['Test'].append(11)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_12(main_params, other_params):
    
    scores_dict['Test'].append(12)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_13(main_params, other_params):
    scores_dict['Test'].append(13)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_14(main_params, other_params):
    scores_dict['Test'].append(14)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_15(main_params, other_params):
    scores_dict['Test'].append(15)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df

def test_16(main_params, other_params):
    scores_dict['Test'].append(16)
    
    stats = results(**main_params, **other_params)
    stats_dict = stats.to_dict(orient='records')[0]

    {scores_dict[key].append(main_params[key]) for key in main_params.keys()}
    {scores_dict[key].append(stats_dict[key]) for key in stats_dict.keys()}
    scores_dict['model'][-1] = main_params['model'].__name__
    print(scores_dict)
    scores_df = pd.DataFrame(scores_dict)
    scores_df.iloc[[-1]].to_csv('output.csv', mode='a', header=False, index=False)

    return scores_df