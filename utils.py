import warnings
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_curve, auc, precision_recall_curve
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import json
from loguru import logger
import itertools 
import seaborn as sns
import re
import yaml
from preprocessor import Preprocessor
from joblib import dump, load

# 数据加载
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def convert_floats(o):
    if isinstance(o, np.float32):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()  # Convert ndarray to list
    elif isinstance(o, dict):
        return {k: convert_floats(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [convert_floats(v) for v in o]
    return o

def LoadFeaturesfromnniimportance(filepath):
    if filepath is None:
        return None
    with open(filepath, 'r') as f:
        importancedata = json.load(f)
    alldata = [item for item in importancedata if item['label'] == 'all'][0]
    return alldata['top_n']

def sorted_features_list(importance_sorting_filepath):
    if importance_sorting_filepath is None:
        return None
    importance_df = pd.read_csv(importance_sorting_filepath)
    # group by feature and average weight
    # rename Feature to feature and Importance to weight
    importance_df = importance_df.rename(columns={'Feature': 'feature'})
    importance_df = importance_df.rename(columns={'Importance': 'weight'})
    # get feature and weight
    importance_df = importance_df[['feature', 'weight']]   
    # group by feature and average weight
    importance_df = importance_df.groupby('feature').mean().reset_index()
    # sort by weight from large to small
    importance_df = importance_df.sort_values(by='weight', ascending=False)
    return importance_df['feature'].tolist()
    


def plot_roc_curve(fpr, tpr, auc, savepath):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'roc curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(savepath)

def plot_prc_curve(precision, recall, auc, savepath):
    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange', lw=lw, label=f'prc curve (area = %0.2f)' % auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve example')
    plt.legend(loc="lower right")
    plt.savefig(savepath)

def plot_feature_importance(model, savepath):
    dflist = [] 
    for target in ['weight', 'total_gain', 'total_cover', 'gain', 'cover']:
        importance = model.get_score(importance_type= target)
        importance_df = pd.DataFrame(index=importance.keys(), data={target: list(importance.values())})

        dflist.append(importance_df)
    importance_df = pd.concat(dflist, axis=1)
    importance_df['feature'] = importance_df.index  
    importance_df.to_csv(os.path.join(savepath, 'feature_importance.csv'), index=False)
    if importance_df.empty:
        print("No feature importance data available to plot.")
        return
    importance_df = importance_df.sort_values('weight', ascending=False)
    #scale to 0-1
    for col in ['weight', 'total_gain', 'total_cover', 'gain', 'cover']:
        importance_df[col] = importance_df[col] / importance_df[col].max()
    # plot first 25 features
    importance_df = importance_df.head(25)
    importance_df = importance_df.loc[:, ['feature', 'weight', 'total_gain', 'total_cover']]
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 10)) # figsize=(width, height) in inches
    ax.set_position([0.1, 0.3, 0.8, 0.6])  # [left, bottom, width, height] in figure coordinates

    importance_df.plot(kind='bar', ax = ax)
    plt.xlabel('Feature')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.savefig(os.path.join(savepath, 'feature_importance.png'))


# 自定义评估函数1
def prerec_auc_metric(y_test, y_pred):
    auc_list = []
    for binary_threshold in [42, 100, 365]:
        y_test_binary = np.where(y_test.copy() > binary_threshold, 1, 0)
        prec, rec, thresholds = precision_recall_curve(y_test_binary, y_pred) 
        prerec_auc = auc(rec, prec)
        f1 = 2 * (prec * rec) / (prec + rec)
        auc_list.append(
            {"binary_threshold": binary_threshold,
            "precision": prec,
            "recall": rec,
            "thresholds": thresholds,
            "prerec_auc": prerec_auc,
            "f1": f1})
    return auc_list

# 自定义评估函数2
def roc_auc_metric(y_test, y_pred):
    auc_list = []
    for binary_threshold in [42, 100, 365]:
        y_test_binary = np.where(y_test.copy() > binary_threshold, 1, 0)
        fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred)
        specifity = 1 - fpr
        sensitivity = tpr
        roc_auc = auc(fpr, tpr)
        auc_list.append(
            {"binary_threshold": binary_threshold,
            "fpr": fpr,
            "tpr": tpr,
            "specifity": specifity,
            "sensitivity": sensitivity,
            "thresholds": thresholds,
            "roc_auc": roc_auc})
    return auc_list

def reverse_y_scaling(y, scale_factor, log_transform):
    preprocesor = Preprocessor()
    return preprocesor.reverse_scalingY(y, scale_factor, log_transform)

# 模型评估
def evaluate_model(model, model_type, X_test, y_test, sw_test,
                  scale_factor, log_transform):
    if model_type == "xgboost":
        dtest = xgb.DMatrix(X_test, label=y_test, weight=sw_test)
        y_pred = model.predict(dtest)
    elif model_type == "svm":
        y_pred = model.predict(X_test)
    elif model_type == "random_forest":
        y_pred = model.predict(X_test)
    elif model_type == "lightgbm":
        import lightgbm as lgb
        y_pred = model.predict(X_test)
    elif model_type == "gbm":
        y_pred = model.predict(X_test)
    elif model_type == "adaboost":
        y_pred = model.predict(X_test)
                               
    else:
        raise ValueError(f"Model type {model_type} is not supported.")
    
    # reverse log transform
    y_test_reversed = reverse_y_scaling(y_test, scale_factor, log_transform)
    y_pred_reversed = reverse_y_scaling(y_pred, scale_factor, log_transform)
    
    # loss
    loss = mean_squared_error(y_test_reversed, y_pred_reversed)

    # roc auc
    roc_auc_json = roc_auc_metric(y_test_reversed, y_pred_reversed)
   
    # prerec auc
    prerec_auc_json = prerec_auc_metric(y_test_reversed, y_pred_reversed)
    
    return loss, roc_auc_json, prerec_auc_json, y_test_reversed, y_pred_reversed


# 保存模型checkpoint
def save_checkpoint(model_type, model, checkpoint_path):
    if model_type == "xgboost":
        # Create directory if not exists
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # Save
        model.save_model(checkpoint_path)
    elif model_type == "svm":
        dump(model, checkpoint_path)
    elif model_type == "random_forest":
        dump(model, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

def custom_eval_roc_auc_factory(custom_metric_key, scale_factor, log_transform):
    ## reverse the y_test and y_pred so that the evaluation metrics are in the original scale
    if custom_metric_key == 'prerec_auc':
        # 用于传入 train_model 的自定义评估函数1
        def custom_eval_prerec_auc(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
            '''
            In the above code snippet, squared_log is the objective function we want. 
            It accepts a numpy array predt as model prediction, and the training DMatrix for obtaining required information, including labels and weights (not used here). 
            This objective is then used as a callback function for XGBoost during training by passing it as an argument to xgb.train:
            '''
            y_train = dtrain.get_label()
            y_pred = predt
            y_train_reversed = reverse_y_scaling(y_train, scale_factor, log_transform)
            y_pred_reversed = reverse_y_scaling(y_pred, scale_factor, log_transform)

            auc_json = prerec_auc_metric(y_train_reversed, y_pred_reversed)
            avg_prerec_auc = np.mean([auc_obj['prerec_auc'] for auc_obj in auc_json])
            return 'prerec_auc', avg_prerec_auc
        return custom_eval_prerec_auc, True
    elif custom_metric_key == 'roc_auc':
        # 用于传入 train_model 的自定义评估函数2
        def custom_eval_roc_auc(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
            '''
            In the above code snippet, squared_log is the objective function we want. 
            It accepts a numpy array predt as model prediction, and the training DMatrix for obtaining required information, including labels and weights (not used here). 
            This objective is then used as a callback function for XGBoost during training by passing it as an argument to xgb.train:
            '''
            y_train = dtrain.get_label()
            y_pred = predt
            y_train_reversed = reverse_y_scaling(y_train, scale_factor, log_transform)
            y_pred_reversed = reverse_y_scaling(y_pred, scale_factor, log_transform)

            auc_json = roc_auc_metric(y_train_reversed, y_pred_reversed)
            avg_roc_auc = np.mean([auc_obj['roc_auc'] for auc_obj in auc_json])
            return 'roc_auc', avg_roc_auc
        return custom_eval_roc_auc, True
    
    elif  custom_metric_key == None or custom_metric_key == 'default':
        def custom_eval_default(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
            '''
            In the above code snippet, squared_log is the objective function we want. 
            It accepts a numpy array predt as model prediction, and the training DMatrix for obtaining required information, including labels and weights (not used here). 
            This objective is then used as a callback function for XGBoost during training by passing it as an argument to xgb.train:
            '''
            y_train = dtrain.get_label()
            y_pred = predt
            y_train_reversed = reverse_y_scaling(y_train, scale_factor, log_transform)
            y_pred_reversed = reverse_y_scaling(y_pred, scale_factor, log_transform)
            loss = mean_squared_error(y_train_reversed, y_pred_reversed)
            return 'loss', loss
        return custom_eval_default, False

def parse_gr_results(grdir):
    logger.info(f"Loading data from {grdir}")
    allfolder = [dir for dir in os.listdir(grdir) if os.path.isdir(os.path.join(grdir, dir))]
    results_list = []  # Change this to a list
    logger.info(f"found {len(allfolder)} folders")
    for folder in allfolder:
        labels_trained = [dir for dir in os.listdir(os.path.join(grdir, folder)) if os.path.isdir(os.path.join(grdir, folder, dir))]
        logger.info(f"found {len(labels_trained)} labels in {folder}")
        for label in labels_trained:
            df_path = os.path.join(grdir, folder, label,'paramandresult.json')
            with open(df_path, 'r') as f:
                results = json.load(f)
            # Append each result as a dictionary to the list
            results_list.append({
                'group': results.get('group', ''),
                'avg_roc_auc': results.get('avg_roc_auc', 0)
            })
    
    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(results_list)
    return df

def parse_nni_results(nni_results, metric:str, minimize:bool, number_of_trials:int):
    '''
    input: jsonfile path
    output: dataframe with avg_roc_auc and group

    '''
    if metric == 'default':
        metric = 'loss'
    if metric == 'roc_auc':
        metric = 'avg_roc_auc'
    if metric == 'prerec_auc':
        metric = 'avg_prerec_auc'
        
    with open(nni_results, 'r') as f:
        results = [json.loads(line) for line in f.readlines()]
    df = pd.DataFrame({
    'avg_roc_auc': [r['avg_roc_auc'] for r in results],
    'metric': [r[metric] for r in results],
    'group': ['all' for r in results],
    })
    df = df.sort_values(by='metric', ascending=minimize)
    df = df.head(number_of_trials)
    df.drop(columns=['metric'], inplace=True)
    return df

def plot_roc_summary(df, outdir):

    # plot dot plot avg_roc_auc in different group x axis is group y axis is avg_roc_auc
    # different objective with different color
    df['avg_roc_auc'] = df['avg_roc_auc'].astype(float)
    uniquegroups = df['group'].unique() 
    orderedlist = sorted(uniquegroups, key = lambda x: int(re.split(r'[-+]', x)[0] if re.match(r'^\d', x) else 9999))
    df['group'] = pd.Categorical(df['group'], categories = orderedlist, ordered = True)

    plt.figure(figsize=(6, 5))
    # violinplot with dots  
    sns.violinplot(data = df, x = 'group', y = 'avg_roc_auc')
    sns.stripplot(data = df, x = 'group', y = 'avg_roc_auc', color = 'orange', size = 6, jitter = 0.25)
    plt.xlabel('group')
    plt.ylabel('avg_roc_auc')
    plt.ylim(0.5, 1)
    plt.title('avg_roc_auc in different group')
    plt.savefig(os.path.join(outdir, 'avg_roc_auc.png'))
    plt.close()


def augment_samples(X, y, sample_weight):
    """
    Augment samples based on sample weights.
    
    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target variable.
    sample_weight (pd.Series): Sample weights.
    
    Returns:
    pd.DataFrame, pd.Series: Augmented feature matrix and target variable.
    """
    logger.info(f"Augmenting samples with sample weights")
    X_augmented = []
    y_augmented = []
    
    for i in range(len(sample_weight)):
        weight = int(np.round(sample_weight.iloc[i]))
        for _ in range(weight):
            X_augmented.append(X.iloc[i])
            y_augmented.append(y.iloc[i])
    
    X_augmented = pd.DataFrame(X_augmented, columns=X.columns)
    y_augmented = pd.Series(y_augmented)
    
    return X_augmented, y_augmented

def load_feature_list_from_boruta_file(boruta_file:str):
    with open(boruta_file, 'r') as f:
        varlist = f.readlines()
    varlist = [vars.split(',') for vars in varlist] 
    new_varlist = []
    for vars in varlist:
        newvars = [v.split('_')[0] for v in vars]
        new_varlist.append(newvars)
    
    # intersect all the list in new_varlist
    intersected_vars = set.intersection(*map(set, new_varlist))
    logger.info(f"Found {len(intersected_vars)} common variables in all the lists")
    return list(intersected_vars)

if __name__ == '__main__':
    l = load_feature_list_from_boruta_file('boruta_explog/09647097-60b1-4c47-bc04-47eb678f73ea/confirmed_vars.txt')
    print('boruta selection with no derived features')
    print(l)
    l = load_feature_list_from_boruta_file('boruta_explog/ffac6478-301d-4127-9794-360f42ac2386/confirmed_vars.txt')
    print('boruta selection with no derived features nni7')
    print(l)
    # l1 = load_feature_list_from_boruta_file('boruta_explog/bfa01a02-738f-4570-9031-9884a3202a07/confirmed_vars.txt')
    # print('boruta selection with derived features')
    # print(l1)
    # l2 = load_feature_list_from_boruta_file('boruta_explog/70e68ddc-06c0-4778-ad89-231ceba214ad/confirmed_vars.txt')
    # print('boruta selection with derived features try2')
    # print(l2)
    # l3 = load_feature_list_from_boruta_file('boruta_explog/fcd9f840-fb57-42be-bbaf-c033710f050c/confirmed_vars.txt')
    # print('boruta selection with derived features try with maxiteration 100')
    # print(l3)
