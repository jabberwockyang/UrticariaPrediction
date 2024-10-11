from train_shap import get_model_data_for_shap
import joblib
from preprocessor import Preprocessor, FeatureFilter
from utils import load_data, check_y
import xgboost as xgb

def get_data_for_des(model, filepath, parmas, 
                      row_na_threshold,
                      preprocessor: Preprocessor, k, randomrate, pick_key):
    # filter x, y for shap explainer
    _ = parmas['scale_factor'] # 用于线性缩放目标变量
    _ = parmas['log_transform'] # 是否对目标变量进行对数变换
    col_na_threshold = parmas['col_na_threshold'] # 用于删除缺失值过多的列

    # 加载数据
    data = load_data(filepath)
    # 预处理数据
    X, y, sampleweight, _, _ = preprocessor.preprocess(data, 
                                1,
                                None,
                                row_na_threshold,
                                col_na_threshold,
                                pick_key= pick_key,
                                common_blood_test= True,
                                disable_scalingX = True
                                )
    
    # predict 
    dtest = xgb.DMatrix(X, label=y)
    y_pred = model.predict(dtest)
    okindex = check_y(y, y_pred, k, randomrate)
    X = X[okindex]
    y = y[okindex]
    return X, y

def get_data_for_des_main(status):

    # beA3o82D_1112_1_all

    fmodel, params, pp, fp= get_model_data_for_shap('trainshap_timeseries.yaml', 'beA3o82D', 1112)
    joblib.dump(fmodel, 'fmodel.pkl')

    key = 'all'

    X, y = get_data_for_des(fmodel, fp, params.copy(), 
                            0.5,
                            pp, k = 2.5, randomrate= 0.1,
                            pick_key= key)
    return X, y