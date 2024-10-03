
import xgboost as xgb
from sklearn.model_selection import train_test_split
import yaml
from loguru import logger
   
from best_params import opendb, get_params_by_sequence_id
from utils import load_data, custom_eval_roc_auc_factory,check_y, reverse_y_scaling
from preprocessor import Preprocessor, FeatureFilter

from sklearn.ensemble import RandomForestRegressor
from sklearn import svm


# 主函数
def trainbyhyperparam(datapath,  params,
                      preprocessor: Preprocessor, 
                      subsetlabel: str = 'all',
                      topn = None):

    # 从超参数中提取预处理参数
    scale_factor = params.pop('scale_factor') # 用于线性缩放目标变量
    log_transform = params.pop('log_transform') # 是否对目标变量进行对数变换
    row_na_threshold = params.pop('row_na_threshold') # 用于删除缺失值过多的行
    col_na_threshold = params.pop('col_na_threshold') # 用于删除缺失值过多的列

    # 加载数据
    data = load_data(datapath)
    # 预处理数据
    X, y, sample_weight, _, _ = preprocessor.preprocess(data, 
                                                        scale_factor,
                                                        log_transform,
                                                        row_na_threshold,
                                                        col_na_threshold,
                                                        pick_key= subsetlabel,
                                                        topn=topn)
    
    if X.shape[0] < 800:
        logger.warning(f"less than 800 data for {subsetlabel}")
        return None, None, None
       
    
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
            dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
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
            model.fit(X, y, sample_weight=sample_weight)

        elif model_type == "random_forest":
            rf_params = {k: v for k, v in param.items() if v is not None}  # 去除 None
            model = RandomForestRegressor(**rf_params)
            model.fit(X, y, sample_weight=sample_weight)
        # elif model_type == "lightgbm":
        #     from lightgbm import LGBMRegressor
        #     lgb_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
        #     model = LGBMRegressor(**lgb_params)
        #     model.fit(X, y, sample_weight=sample_weight)

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

    return model

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        nni_config = config['nni']
        train_config = config['train']
    return config, nni_config, train_config

def get_data_for_Shap(model, filepath, parmas, 
                      row_na_threshold,
                      preprocessor: Preprocessor, k, randomrate, pick_key):
    # filter x, y for shap explainer
    scale_factor = parmas['scale_factor'] # 用于线性缩放目标变量
    log_transform = parmas['log_transform'] # 是否对目标变量进行对数变换
    col_na_threshold = parmas['col_na_threshold'] # 用于删除缺失值过多的列

    # 加载数据
    data = load_data(filepath)
    # 预处理数据
    X, y, sampleweight, _, _ = preprocessor.preprocess(data, 
                                scale_factor,
                                log_transform,
                                row_na_threshold,
                                col_na_threshold,
                                pick_key= pick_key,
                                common_blood_test= True,
                                )
    
    # predict 

    dtest = xgb.DMatrix(X, label=y)
    y_pred = model.predict(dtest)
    okindex = check_y(y, y_pred, k, randomrate)
    X = X[okindex]
    return X


def get_model_data_for_shap(config, expid, sequenceid):

    config, nni_config, train_config = load_config(config)

    # 读取nni实验中最佳超参数
    former_exp_stp = nni_config['exp_stp']
    best_exp_id = expid
    best_db_path = f'{former_exp_stp}/{best_exp_id}/db/nni.sqlite'
    df = opendb(best_db_path)
    
    paramid, parmas, sequenceid = get_params_by_sequence_id(df, [sequenceid])[0]

    # 训练数据相关参数
    filepath = train_config['filepath']
    target_column = train_config['target_column']

    groupingparams = {'grouping_parameter_id': train_config['grouping_parameter_id'],
                      'bins': train_config['groupingparams']['bins'],
                      'labels': train_config['groupingparams']['labels']}

    
    # 实例化预处理器
    featurelistpath = train_config['feature_list'] 
    featurelist = open(featurelistpath, 'r').read().splitlines() if featurelistpath else None
    ff = FeatureFilter(target_column= target_column, 
                       method = 'selection',
                       features_list=featurelist) if featurelist else None
    pp = Preprocessor(target_column, groupingparams, FeaturFilter=ff)

    model = trainbyhyperparam(filepath, parmas.copy(), pp, 'all')
    
    # fmodel, params, X, pp, fp
    return model, parmas.copy(), pp, filepath
        

class ModelReversingY():
    def __init__(self, model, params: dict):
        self.model = model
        self.params = params
        self.scale_factor = params['scale_factor']
        self.log_transform = params['log_transform']
    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        y = self.model.predict(dtest)
        ry = reverse_y_scaling(y, self.scale_factor, self.log_transform)
        return ry

if __name__ == '__main__':
    # beA3o82D_1112_1_all
    fmodel, preprocessor, X, y= get_model_data_for_shap('trainshap_timeseries.yaml', 'beA3o82D', 1112)
    model = ModelReversingY(fmodel, preprocessor)
