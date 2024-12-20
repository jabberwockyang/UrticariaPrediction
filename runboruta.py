
import pandas as pd
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
import seaborn as sns
from utils import load_data, load_config, augment_samples, load_feature_list_from_boruta_file
from best_params import opendb, get_params_by_sequence_id
from preprocessor import Preprocessor, FeatureDrivator
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import queue
import argparse
import uuid
import json
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor

class PoisonPill:
    pass

def select_model(model_type, params):
    """根据模型类型初始化对应的模型"""

    if model_type == "xgboost":
        custom_metric_key = params.pop('custom_metric')
        n_estimators = params.pop('num_boost_round')
        early_stopping_rounds = params.pop('early_stopping_rounds')
        
        params['reg_alpha'] = params.pop('alpha')
        params["device"] = "cuda"
        params["tree_method"] = "hist"

        model = xgb.XGBRegressor(n_estimators=n_estimators, **params)
    elif model_type == "random_forest":
        rf_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
        model = RandomForestRegressor(**rf_params)
    elif model_type == "adaboost":
        ada_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
        model = AdaBoostRegressor(**ada_params)

    elif model_type == "gbm":
        gbm_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
        model = GradientBoostingRegressor(**gbm_params)

    elif model_type == "lightgbm":
        lgb_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
        model = LGBMRegressor(**lgb_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def main(filepath,  params, preprocessor, experiment_name, log_dir, max_iteration = 50):
    scale_factor = params.pop('scale_factor') # 用于线性缩放目标变量
    log_transform = params.pop('log_transform') # 是否对目标变量进行对数变换
    row_na_threshold = params.pop('row_na_threshold') # 丢弃缺失值比例超过阈值的样本
    col_na_threshold = params.pop('col_na_threshold') # 丢弃缺失值比例超过阈值的特征

    data = load_data(filepath)
    X, y, sample_weight,_,_ = preprocessor.preprocess(data,
                                                  scale_factor,
                                                  log_transform,
                                                    row_na_threshold,
                                                    col_na_threshold,
                                                  pick_key= 'all')
    # 权重
    logger.info(f"Before augmentation: {X.shape}, {y.shape}")
    X, y = augment_samples(X, y, sample_weight)
    logger.info(f"After augmentation: {X.shape}, {y.shape}")

    # 初始化一个 树模型
    modeltype = params.pop('model')
    model = select_model(modeltype, params)
    # 初始化一个 DataFrame 来存储特征排名
    ranking_df = pd.DataFrame(columns=X.columns)
    output_queue = queue.Queue()

    def run_boruta(X, y, i):
        logger.info(f"experiment {experiment_name}: Iteration {i+1}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
        # 初始化Boruta特征选择器
        boruta_selector = BorutaPy(model, verbose=2, 
                                   random_state=i, max_iter=max_iteration)
    
        boruta_selector.fit(X_train, y_train)
        confirmed_vars = X.columns[boruta_selector.support_]
        feature_ranks = boruta_selector.ranking_
        logger.info(f"experiment {experiment_name}: Iteration {i+1} finished")
        output_queue.put((i+1, confirmed_vars, feature_ranks))
        logger.info(f"experiment {experiment_name}: Iteration {i+1} put into queue")

    def get_and_write(output_queue):
        while True:
            item = output_queue.get()
            if isinstance(item, PoisonPill):
                logger.info(f"PoisonPill received, exiting...")
                return
            elif isinstance(item, tuple):
                i, confirmed_vars, feature_ranks = item
                ranking_df.loc[i] = feature_ranks
                ranking_df.to_csv(os.path.join(log_dir,'ranking_df.csv'))
                
                with open(os.path.join(log_dir, 'confirmed_vars.txt'), 'a') as f:
                    f.write(f"{i},{','.join(confirmed_vars)}\n")
            else:
                raise ValueError(f"Invalid item type: {type(item)}")

    # 使用多线程执行 Boruta
    with ThreadPoolExecutor(max_workers=6) as executor:
        consumer = executor.submit(get_and_write, output_queue)
        producers = [executor.submit(run_boruta, X, y, i) for i in range(15)]
        for future in producers:
            future.result()  # 等待所有线程完成
        logger.info(f"All threads finished, sending PoisonPill...")
        output_queue.put(PoisonPill())
        logger.info(f"PoisonPill sent")

        re = consumer.result()
        logger.info(f"Consumer result: {re}")
        


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str)
    parser.add_argument('--best_db_path', type=str)
    parser.add_argument('--best_sequence_id', type=int)
    parser.add_argument('--max_iteration', type=int, default=50)

    parser.add_argument('--target_column', type=str, default='VisitDuration')
    parser.add_argument('--log_dir', type=str, default='boruta_explog')
    parser.add_argument('--groupingparams', type=str, default='groupingsetting.yml')
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    filepath = args.filepath
    target_column = args.target_column
    groupingparams = load_config(args.groupingparams)['groupingparams']
    log_dir = args.log_dir
    # generate unique experiment name
    experiment_name = str(uuid.uuid4()).split('-')[0]
    logger.info(f"Experiment name: {experiment_name}")
    log_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    # save copy of args
    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, ensure_ascii=False,indent=4)

    
    df  = opendb(args.best_db_path)
    paramid, best_param, sequenceid = get_params_by_sequence_id(df, [args.best_sequence_id])[0]
    
    # save copy of best_param
    with open(os.path.join(log_dir, 'best_param.json'), 'w') as f:
        json.dump(best_param, f, ensure_ascii=False,indent=4)
    
    # 实例化预处理器
    preprocessor = Preprocessor(target_column,
                                 groupingparams)

    max_iteration = args.max_iteration
    main(filepath, best_param, preprocessor, experiment_name, log_dir, max_iteration)

