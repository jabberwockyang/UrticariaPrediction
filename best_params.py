import sqlite3
import json
import pandas as pd
from loguru import logger
# 定义数据库文件的路径

def opendb(db_path):
    # 连接到数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()


    cursor.execute("""
        SELECT sequenceId, trialJobId, data FROM TrialJobEvent
    """)

    # 解析结果
    results_t = []
    for row in cursor.fetchall(): 
        sequenceId = row[0]
        trialJobId = row[1]
        params = json.loads(row[2]) if row[2] else None # 将 params 字符串解析为字典
        if params:
            tobe_append = params.copy()
            tobe_append['trialJobId'] = trialJobId
            tobe_append['sequenceId'] = sequenceId
            results_t.append(tobe_append)

    cursor.execute("""
        SELECT trialJobId, data FROM MetricData
    """)
    results_m = []
    for row in cursor.fetchall(): 
        metrics = json.loads(row[1])  # 将 metrics 字符串解析为字典
        metrics = json.loads(metrics)  # 将 metrics 字符串解析为字典

        if isinstance(metrics, dict):
            tobe_append = metrics.copy()
            tobe_append['trialJobId'] = row[0]
            results_m.append(tobe_append)
        elif isinstance(metrics, float):
            tobe_append = {
                'trialJobId': row[0],
                'default': metrics
            }

    cursor.close()
    dft = pd.DataFrame(results_t)
    dfm = pd.DataFrame(results_m)
    # 确保 dft 和 dfm 的 'parameter_id' 和 'parameterId' 列的数据类型一致
    dft['trialJobId'] = dft['trialJobId'].astype(str)  # 或者 dfm['parameterId'] = dfm['parameterId'].astype(int)
    dfm['trialJobId'] = dfm['trialJobId'].astype(str)
    

    # 进行 merge 操作
    df = pd.merge(dft, dfm, on='trialJobId', how='inner')

    return df

def get_best_params(df, metric_to_optimize:list=[('roc_auc',"maximize")], number_of_trials = 5):
    # 找到最优的参数
    best_parameter_id_list = []
    for metric in metric_to_optimize:
        if metric[1] == 'maximize':
            # df 的 uniqueid 是sequenceid 先 groupby parameter_id 算出 mean metric 找到metric表现最好前 n个 parameter_id  返回list(（params(dict), sequenceid(int)）)
            best_params = df.groupby('parameter_id')[metric[0]].mean().nlargest(number_of_trials).index
        elif metric[1] == 'minimize':
            best_params = df.groupby('parameter_id')[metric[0]].mean().nsmallest(number_of_trials).index

        for param_id in best_params:
            param_rows = df[df['parameter_id'] == param_id]
            params = param_rows['parameters'].iloc[0]
            sequence_ids = param_rows['sequenceId'].tolist()
            best_parameter_id_list.append((param_id, params, sequence_ids))

    logger.info(f"top {number_of_trials} best params: {best_parameter_id_list[:5]}")
    return best_parameter_id_list

def get_params_by_sequence_id(df, sequence_ids):
    best_parameter_id_list = []
    # 找到最优的参数
    for sequence_id in sequence_ids:
        param_rows = df[df['sequenceId'] == sequence_id]
        params = param_rows['parameters'].iloc[0]
        param_id = param_rows['parameter_id'].iloc[0]
        best_parameter_id_list.append((param_id, params, [sequence_id]))
    return best_parameter_id_list


if __name__ == '__main__':
    db_path = '/root/ClinicalXgboost/nni5_explog/8Y9XvkQq/db/nni.sqlite'
    df = opendb(db_path)
    list_of_tuple = get_best_params(df, 
                                    metric_to_optimize=[('roc_auc',"maximize")], 
                                    number_of_trials = 1)
    for i in list_of_tuple:
        print(i[0])
        print(i[1])
        print(i[2]) 