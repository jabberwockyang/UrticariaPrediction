# todo 全部数据用作validation
# 全部数据用作画图结果
import xgboost as xgb
from sklearn.model_selection import train_test_split
import argparse
import yaml
import os
import json
from loguru import logger
   
from best_params import opendb, get_best_params, get_params_by_sequence_id
from utils import load_data, custom_eval_roc_auc_factory, evaluate_model, convert_floats, load_feature_list_from_boruta_file
from preprocessor import Preprocessor, FeatureDrivator, FeatureFilter
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import numpy as np

# 主函数
def trainbyhyperparam(datapath, 
                      log_dir,
                      experiment_id, sequence_id, params,
                      preprocessor: Preprocessor, subsetlabel: str = 'all',
                      n_splits=5, 
                      n_repeats=1,
                      topn = None):
    
    functionparmas = locals()
    functionparmas.pop('preprocessor')

    # checked processed data
    if os.path.exists(f'{log_dir}/{experiment_id}/ppshape.jsonl'):
        with open(f'{log_dir}/{experiment_id}/ppshape.jsonl', 'r') as f:
            for line in f:
                if json.loads(line)['sequence_id'] == sequence_id:
                    logger.warning(f"sequence_id {sequence_id} already processed")
                    return None, None, None 


    hyperparameters = {
        "experiment_id": experiment_id,
        "sequence_id": sequence_id,
        "params": params,
        'functionparmas': functionparmas,  
    }
    with open(f'{log_dir}/{experiment_id}/hyperparameters.jsonl', 'a') as f:
        json.dump(convert_floats(hyperparameters), f, ensure_ascii=False)
        f.write('\n')
    
    # 从超参数中提取预处理参数
    scale_factor = params.pop('scale_factor') # 用于线性缩放目标变量
    log_transform = params.pop('log_transform') # 是否对目标变量进行对数变换
    row_na_threshold = params.pop('row_na_threshold') # 用于删除缺失值过多的行
    col_na_threshold = params.pop('col_na_threshold') # 用于删除缺失值过多的列

    # 加载数据
    data = load_data(datapath)
    # 预处理数据
    X, y, sample_weight, avg_missing_perc_row, avg_missing_perc_col = preprocessor.preprocess(data, 
                                                                            scale_factor,
                                                                            log_transform,
                                                                            row_na_threshold,
                                                                            col_na_threshold,
                                                                            pick_key= subsetlabel,
                                                                            topn=topn)
    
    if X.shape[0] < 800:
        logger.warning(f"less than 800 data for {subsetlabel}")
        return None, None, None
       

    ppshape = {
        "experiment_id": experiment_id,
        "sequence_id" : sequence_id,
        'functionparmas': functionparmas,
        'pp_params':{'row_na_threshold': row_na_threshold, 'col_na_threshold': col_na_threshold},
        'ppresults':{
            'shape': [X.shape[0], X.shape[1]],
            'avg_missing_perc_row': avg_missing_perc_row,
            'avg_missing_perc_col': avg_missing_perc_col
        }
    }
    # 预处理结果保存
    with open(f'{log_dir}/{experiment_id}/ppshape.jsonl', 'a') as f:
        json.dump(convert_floats(ppshape), f, ensure_ascii=False)
        f.write('\n')

    
    # external test set split
    X_deriva, X_test_ext, y_deriva, y_test_ext, sw_deriva, sw_test_ext = train_test_split(X, y, sample_weight, test_size=0.3, random_state=42)   
    
    
    # 初始化 KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    fold = 0
    model_type = params.pop('model')
    # k fold cross validation internal test set
    for train_index, val_index in kf.split(X_deriva):
        X_train_int, X_val = X_deriva.iloc[train_index], X_deriva.iloc[val_index]
        y_train_int, y_val = y_deriva.iloc[train_index], y_deriva.iloc[val_index]
        sw_train_int, sw_val = sw_deriva.iloc[train_index], sw_deriva.iloc[val_index]

        def train_model(params):
            if model_type == "xgboost":
                # XGBoost 特有的 GPU 参数
                params["device"] = "cuda"
                params["tree_method"] = "hist"

                custom_metric_key = params.pop('custom_metric')
                num_boost_round = params.pop('num_boost_round')
                early_stopping_rounds = params.pop('early_stopping_rounds')

                custom_metric, maximize = custom_eval_roc_auc_factory(custom_metric_key, scale_factor, log_transform) # 'prerec_auc' 'roc_auc' None

                xgb_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
                # 构建 DMatrix
                dtrain = xgb.DMatrix(X_train_int, label=y_train_int, weight=sw_train_int)
                dval = xgb.DMatrix(X_val, label=y_val, weight=sw_val)
                # 训练 XGBoost 模型，传入早停参数
                model = xgb.train(xgb_params, 
                                dtrain, 
                                custom_metric=custom_metric,
                                evals=[(dtrain, 'train'), 
                                    (dval, 'validation')],
                                maximize=maximize,
                                num_boost_round=num_boost_round,
                                early_stopping_rounds=early_stopping_rounds)

            elif model_type == "svm":
                svm_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
                model = svm.SVR(**svm_params)
                model.fit(X_train_int, y_train_int, sample_weight=sw_train_int)

            elif model_type == "random_forest":
                rf_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
                model = RandomForestRegressor(**rf_params)
                model.fit(X_train_int, y_train_int, sample_weight=sw_train_int)

            # elif model_type == "lightgbm":
            #     from lightgbm import LGBMRegressor
            #     lgb_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
            #     model = LGBMRegressor(**lgb_params)
            #     model.fit(X_train_int, y_train_int, sample_weight=sw_train_int)

            elif model_type == "gbm":
                from sklearn.ensemble import GradientBoostingRegressor
                gbm_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
                model = GradientBoostingRegressor(**gbm_params)
                model.fit(X_train_int, y_train_int, sample_weight=sw_train_int)

            elif model_type == "adaboost":
                from sklearn.ensemble import AdaBoostRegressor
                ada_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
                model = AdaBoostRegressor(**ada_params)
                model.fit(X_train_int, y_train_int, sample_weight=sw_train_int)

            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            return model

        model = train_model(params.copy())  # 传入超参数

        loss, roc_auc_json, _, ry, rypredict= evaluate_model(model, model_type, X_val, y_val, sw_val,
                                                        scale_factor, log_transform)
        fold_results.append({
            
            'fold': fold,
            'loss': loss,
            'roc_auc_json': roc_auc_json,
            'avg_roc_auc': np.mean([roc_obj["roc_auc"] for roc_obj in roc_auc_json]), # 一个fold里不同二分类阈值的平均 roc_auc
            'roc_auc_42': [roc_obj["roc_auc"] for roc_obj in roc_auc_json if roc_obj["binary_threshold"] == 42][0], # 二分类阈值为 42 时的 roc_auc
            'roc_auc_100': [roc_obj["roc_auc"] for roc_obj in roc_auc_json if roc_obj["binary_threshold"] == 100][0], # 二分类阈值为 100 时的 roc_auc
            'roc_auc_365': [roc_obj["roc_auc"] for roc_obj in roc_auc_json if roc_obj["binary_threshold"] == 365][0], # 二分类阈值为 365 时的 roc_auc
            'ry': ry.tolist(),
            'rypredict': rypredict.tolist()
        })
        fold += 1
        if fold >= n_repeats:
            break
    
    # 保存实验 id 超参数 和 结果 # 逐行写入
    result = {
        "experiment_id": experiment_id,
        'sequence_id': sequence_id,
        'functionparmas': functionparmas,
        'fold_results': fold_results
    }

    with open(f'{log_dir}/{experiment_id}/results.jsonl', 'a') as f:
        json.dump(convert_floats(result), f, ensure_ascii=False)
        f.write('\n')
    
    avg_loss = np.mean([result['loss'] for result in fold_results])
    avg_42_roc_auc = np.mean([result['roc_auc_42'] for result in fold_results])
    avg_100_roc_auc = np.mean([result['roc_auc_100'] for result in fold_results])


    return avg_loss, avg_42_roc_auc, avg_100_roc_auc


def parse_args():
    parser = argparse.ArgumentParser(description='Run XGBoost Model Training with Grouping Parameters')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--expfolder', type=str, default=None, help='Experiment folder of nni results to run')
    parser.add_argument('--expid', type=str, default=None, help='Experiment ID of nni results to run')
    
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        nni_config = config['nni']
        train_config = config['train']
    return config, nni_config, train_config

if __name__ == "__main__":
    args = parse_args()
    config, nni_config, train_config = load_config(args.config)

    # if expid not provided in arg then use the one in config
    former_exp_stp = args.expfolder
    best_exp_id = args.expid
    best_db_path = f'{former_exp_stp}/{best_exp_id}/db/nni.sqlite'
    df = opendb(best_db_path)

    if nni_config['sequence_id'] is not None:
        ls_of_params = get_params_by_sequence_id(df, nni_config['sequence_id'])
    else:
        metric_to_optimize = nni_config['metric_to_optimize']
        number_of_trials = nni_config['number_of_trials']
        ls_of_params = get_best_params(df, metric_to_optimize, number_of_trials)
    

    # 训练数据相关参数
    current_exp_stp = train_config['exp_stp']
    if not os.path.exists(current_exp_stp):
        os.makedirs(current_exp_stp)
    filepath = train_config['filepath']
    target_column = train_config['target_column']

    kfold_k = train_config['kfold_k']
    kfold_reps = train_config['kfold_reps']
    assert kfold_k >= kfold_reps, "kfold_k should be greater than or equal to kfold_reps"
    assert isinstance(kfold_k, int) and isinstance(kfold_reps, int), "kfold_k and kfold_reps should be integers"

    # 分组参数
    label_toTrain = train_config['label_toTrain']
    groupingparams = {'grouping_parameter_id': train_config['grouping_parameter_id'],
                      'bins': train_config['groupingparams']['bins'],
                      'labels': train_config['groupingparams']['labels']}

    

    pp = Preprocessor(target_column, groupingparams)
    # 实验日志目录
    experiment_id = f'{best_exp_id}_{"&".join([m[0] for m in metric_to_optimize])}_top{number_of_trials}_gr{train_config["grouping_parameter_id"]}'

    if not os.path.exists(f'{current_exp_stp}/{experiment_id}'):
        os.makedirs(f'{current_exp_stp}/{experiment_id}')
    for best_param_id, best_params, sequence_ids in ls_of_params:
        for label in label_toTrain:
            sequence_id = str(best_exp_id)+ '_' + str(sequence_ids[0]) + '_' + str(train_config['grouping_parameter_id']) + '_' + label

            avg_loss, avg_42_roc_auc, avg_100_roc_auc = trainbyhyperparam(filepath, 
                                                                          current_exp_stp,
                                                                          experiment_id, sequence_id, best_params,
                                                                          pp, label, n_splits=kfold_k, n_repeats=kfold_reps)

            logger.info(f"Experiment ID: {experiment_id}, Sequence ID: {sequence_id}, Label: {label}, Loss: {avg_loss}, ROC AUC 42: {avg_42_roc_auc}, ROC AUC 100: {avg_100_roc_auc}")
        



