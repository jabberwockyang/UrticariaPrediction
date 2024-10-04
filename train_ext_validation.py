# todo 全部数据用作validation
# 全部数据用作画图结果
import xgboost as xgb
from sklearn.model_selection import train_test_split
import argparse
import yaml
import os
import json
from loguru import logger
   
from best_params import opendb, get_params_by_sequence_id
from utils import load_data, custom_eval_roc_auc_factory, evaluate_model, convert_floats
from preprocessor import Preprocessor, FeatureDrivator, FeatureFilter
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import numpy as np

# 主函数
def trainbyhyperparam(datapath, 
                      log_dir,
                      experiment_id, sequence_id, params,
                      preprocessor: Preprocessor,
                      subsetlabel: str = 'all',
                      topn = None):
    functionparmas = locals()
    functionparmas.pop('preprocessor')
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
    
    model_type = params.pop('model')

    def train_model(model_type, param):
        if model_type == "xgboost":
            # XGBoost 特有的 GPU 参数
            param["device"] = "cuda"
            param["tree_method"] = "hist"

            custom_metric_key = param.pop('custom_metric')
            num_boost_round = param.pop('num_boost_round')
            early_stopping_rounds = param.pop('early_stopping_rounds')

            custom_metric, maximize = custom_eval_roc_auc_factory(custom_metric_key, scale_factor, log_transform) # 'prerec_auc' 'roc_auc' None

            xgb_params = {k: v for k, v in param.items() if v is not None}  # 去除 None
            # 构建 DMatrix
            dtrain = xgb.DMatrix(X_deriva, label=y_deriva, weight=sw_deriva)
            dval = xgb.DMatrix(X_test_ext, label=y_test_ext, weight=sw_test_ext)
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
            svm_params = {k: v for k, v in param.items() if v is not None}  # 去除 None
            model = svm.SVR(**svm_params)
            model.fit(X_deriva, y_deriva, sample_weight=sw_deriva)

        elif model_type == "random_forest":
            rf_params = {k: v for k, v in param.items() if v is not None}  # 去除 None
            model = RandomForestRegressor(**rf_params)
            model.fit(X_deriva, y_deriva, sample_weight=sw_deriva)

        # elif model_type == "lightgbm":
        #     from lightgbm import LGBMRegressor
        #     lgb_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
        #     model = LGBMRegressor(**lgb_params)
        #     model.fit(X_deriva, y_deriva, sample_weight=sw_deriva)

        elif model_type == "gbm":
            from sklearn.ensemble import GradientBoostingRegressor
            gbm_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
            model = GradientBoostingRegressor(**gbm_params)
            model.fit(X_deriva, y_deriva, sample_weight=sw_deriva)

        elif model_type == "adaboost":
            from sklearn.ensemble import AdaBoostRegressor
            ada_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
            model = AdaBoostRegressor(**ada_params)
            model.fit(X_deriva, y_deriva, sample_weight=sw_deriva)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model

    model = train_model(model_type, params.copy())  # 传入超参数

    fold_results = []
    for trainset in ['deriva', 'test_ext']:
        if trainset == 'deriva':
            loss, roc_auc_json,  _, ry, rypredict = evaluate_model(model, model_type,  X_deriva, y_deriva, sw_deriva, 
                                                            scale_factor, log_transform)
        else:
            loss, roc_auc_json,  _, ry, rypredict = evaluate_model(model, model_type,  X_test_ext, y_test_ext, sw_test_ext, 
                                                                scale_factor, log_transform)

        fold_results.append({
                'trainset': trainset,
                'loss': loss,
                'roc_auc_json': roc_auc_json,
                'avg_roc_auc': np.mean([roc_obj["roc_auc"] for roc_obj in roc_auc_json]), # 一个fold里不同二分类阈值的平均 roc_auc

                'roc_auc_42': [roc_obj["roc_auc"] for roc_obj in roc_auc_json if roc_obj["binary_threshold"] == 42][0], # 二分类阈值为 42 时的 roc_auc
                'roc_auc_100': [roc_obj["roc_auc"] for roc_obj in roc_auc_json if roc_obj["binary_threshold"] == 100][0], # 二分类阈值为 100 时的 roc_auc
                'roc_auc_365': [roc_obj["roc_auc"] for roc_obj in roc_auc_json if roc_obj["binary_threshold"] == 365][0], # 二分类阈值为 365 时的 roc_auc

                'ry': ry.tolist(),  
                'rypredict': rypredict.tolist()
            })

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
    parser.add_argument('--expid', type=str, default=None, help='Experiment ID of nni results to run external validation')
    parser.add_argument('--sequenceid', type=int, default=0, help='Sequence ID of nni results to run external validation')
    parser.add_argument('--featurelistfolder', type=str, default=None, help='Folder containing feature lists')
    
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

    # 读取nni实验中最佳超参数
    former_exp_stp = nni_config['exp_stp']
    # if expid not provided in arg then use the one in config
    
    best_exp_id = args.expid if args.expid else nni_config['best_exp_id']
    best_sequence_id = args.sequenceid if args.sequenceid else nni_config['sequence_id']

    df = opendb(f'{former_exp_stp}/{best_exp_id}/db/nni.sqlite')
    paramid, parmas, sequenceid = get_params_by_sequence_id(df, [best_sequence_id])[0]
    

    # 训练数据相关参数
    current_exp_stp = train_config['exp_stp']
    if not os.path.exists(current_exp_stp):
        os.makedirs(current_exp_stp)
    filepath = train_config['filepath']
    target_column = train_config['target_column']

    # 分组参数
    label_toTrain = train_config['label_toTrain']
    groupingparams = {'grouping_parameter_id': train_config['grouping_parameter_id'],
                      'bins': train_config['groupingparams']['bins'],
                      'labels': train_config['groupingparams']['labels']}

    
    # 实例化预处理器
    
    featurelistfolder = args.featurelistfolder if args.featurelistfolder else train_config['featurelistfolder']
    # find all '{featurelistfolder}/top*_confirmed_vars.txt' by os by matching _confirmed_vars.txt
    featurelistpaths = [os.path.join(featurelistfolder, f) for f in os.listdir(featurelistfolder) if f.endswith('_confirmed_vars.txt')]

    for featurelistpath in featurelistpaths:
        featurelist = open(featurelistpath, 'r').read().splitlines()
        featurelist_name = os.path.basename(featurelistpath).split('_')[0]
        ff = FeatureFilter(target_column= target_column, 
                        method = 'selection',
                        features_list=featurelist)
        pp = Preprocessor(target_column, groupingparams, FeaturFilter=ff)

        # 实验日志目录
        experiment_id = f'{best_exp_id}_{best_sequence_id}_extval_gr{train_config["grouping_parameter_id"]}_{featurelist_name}'
        sequence_id = f"{best_exp_id}_{best_sequence_id}_gr{train_config['grouping_parameter_id']}"
        if not os.path.exists(f'{current_exp_stp}/{experiment_id}'):
            os.makedirs(f'{current_exp_stp}/{experiment_id}')
    
        avg_loss, avg_42_roc_auc, avg_100_roc_auc = trainbyhyperparam(filepath, 
                                                                            current_exp_stp,
                                                                            experiment_id, sequence_id, parmas.copy(),
                                                                            pp, 'all')


        logger.info(f"Experiment ID: {experiment_id}, Sequence ID: {best_sequence_id }, feature_list: {featurelistpath}, Loss: {avg_loss}, ROC AUC 42: {avg_42_roc_auc}, ROC AUC 100: {avg_100_roc_auc}")




