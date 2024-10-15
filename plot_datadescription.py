import pandas as pd
import numpy as np
from scipy import stats
from utils import load_data, check_y
from preprocessor import Preprocessor, FeatureFilter
from train_shap import get_model_data_for_shap
from best_params import opendb, get_params_by_sequence_id
import yaml
from loguru import logger
import json
import xgboost as xgb
import os


def get_data_for_des(model, filepath, parmas, 
                      row_na_threshold,
                      preprocessor: Preprocessor, k, randomrate, pick_key):
    # filter x, y for shap explainer
    scale_factor = parmas['scale_factor'] # 用于线性缩放目标变量
    log_transform = parmas['log_transform'] # 是否对目标变量进行对数变换
    col_na_threshold = parmas['col_na_threshold'] # 用于删除缺失值过多的列

    # 加载数据
    data = load_data(filepath)
    # 预处理数据
    X, y, _, _, _ = preprocessor.preprocess(data, 
                                1,
                                None,
                                row_na_threshold,
                                col_na_threshold,
                                pick_key= pick_key,
                                common_blood_test= True,
                                disable_scalingX= True
                                )
    Xs, ys, _, _, _ = preprocessor.preprocess(data, 
                                scale_factor,
                                log_transform,
                                row_na_threshold,
                                col_na_threshold,
                                pick_key= pick_key,
                                common_blood_test= True
                                )
    # predict 

    dtest = xgb.DMatrix(Xs, label=ys)
    y_pred = model.predict(dtest)
    okindex = check_y(ys, y_pred, k, randomrate)
    X = X[okindex]
    y = y[okindex]
    return X, y

def load_sorted_varlist():
    clinicalvarlistfile = 'ClinicalItemClass_ID.json'
    with open(clinicalvarlistfile, 'r') as file:
        clinicalvarjson = json.load(file)
    laboratoryvarlistfile = 'ExaminationItemClass_ID.json'
    with open(laboratoryvarlistfile, 'r') as file:
        laboratoryvarjson = json.load(file)
    clinicalvarlist = []
    for k,v in clinicalvarjson.items():
        clinicalvarlist.extend([l[1] for l in v])
    laboratoryvarlist = []
    for k,v in laboratoryvarjson.items():
        laboratoryvarlist.extend([l[1] for l in v])
    sortedvarlist = clinicalvarlist + laboratoryvarlist
    return sortedvarlist


def getranking(var, sortedvarlist):
    if var == 'Outcome':
        return -1
    for i, v in enumerate(sortedvarlist):
        if var.split('_')[0] == v:
            return i
        
    raise ValueError(f"Variable {var} not found in sortedvarlist")
    


def prepare_params(config):
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    # 读取nni实验中最佳超参数
    expidfolder = config['bestfolder']
    bestexpid = expidfolder.split('/')[-1]
    sequenceid = config['bestsequenceid']
    return bestexpid, sequenceid





# ① 数据拆分函数
def split_data(df, outcome_col=None, train_ratio=0.7, outcome_threshold=100,  random_state=25):
    
    # 按照好/坏结果拆分
    good_outcome = df[df[outcome_col] < outcome_threshold]
    poor_outcome = df[df[outcome_col] >= outcome_threshold]
     
    logger.info(f"Good outcome: {good_outcome.shape}")
    logger.info(f"Poor outcome: {poor_outcome.shape}")

    # 按照训练集/测试集拆分
    train_df = df.sample(frac=train_ratio, random_state=random_state)
    test_df = df.drop(train_df.index)
    logger.info(f"Train: {train_df.shape}")
    logger.info(f"Test: {test_df.shape}")

    return df, good_outcome, poor_outcome, train_df, test_df

# ② 统计总结函数
def generate_summary_statistics(df_all, df_split1, df_split2, binary_threshold=2):
    summary = {}
    columns = df_all.columns

    def format_summary(mean, std):
        return {"mean": mean, "std": std}

    for col in columns:
        if col == "index":
            continue
        summary[col] = {}

        if df_all[col].nunique() <= binary_threshold:
            logger.info(f"Binary variable: {col}")
            # 二分类变量，使用卡方检验
            dfforcross1 = df_split1.copy()
            dfforcross1['split'] = 1
            dfforcross2 = df_split2.copy()
            dfforcross2['split'] = 2
            dfforcross = pd.concat([dfforcross1, dfforcross2], axis=0)
            contingency_table = pd.crosstab(dfforcross[col], dfforcross['split'])
            _, p_val, _, _ = stats.chi2_contingency(contingency_table)
            summary[col]["split1"] = df_split1[col].value_counts(normalize=True).to_dict()
            summary[col]["split2"] = df_split2[col].value_counts(normalize=True).to_dict()
        else:
            # 连续变量，使用t检验
            t_stat, p_val = stats.ttest_ind(df_split1[col].dropna(), df_split2[col].dropna())
            summary[col]["split1"] = format_summary(df_split1[col].mean(), df_split1[col].std())
            summary[col]["split2"] = format_summary(df_split2[col].mean(), df_split2[col].std())
        summary[col]["p_value"] = p_val
    
    summary['N'] = {
        'all': len(df_all),
        'split1': len(df_split1),
        'split2': len(df_split2),
        'p_value': np.nan
    }

    return summary

# ③ JSON转换为LaTeX函数
import re

def join_with_max_length(splitedvar, maxlength=23):
    result = []
    current_string = ""
    
    for word in splitedvar:
        if word == 'Avg': # remove this word
            continue
        # 如果加上下一个词后长度超过maxlength，则不再加入
        if len(current_string + word + " ") <= maxlength:
            if current_string:
                current_string += " " + word
            else:
                current_string = word
        else:
            # 如果当前字符串达到了最大长度，则保存到result，并开始新的字符串
            result.append(current_string)
            current_string = word

    # 处理最后一个没有添加到result的字符串
    if current_string:
        result.append(current_string)
    
    return result

def json_to_csv(summary, dataset_name1="Dataset 1", dataset_name2="Dataset 2"):
    csv_table = "Characteristic,{},{},".format(dataset_name1, dataset_name2)
    csv_table += "P-value\n"
    csv_table += "Number of patients,{},{},".format(summary['N']['split1'], summary['N']['split2'])
    varlist = list(summary.keys())
    varlist.remove('N')
    varlist = sorted(varlist, key=lambda x: getranking(x, load_sorted_varlist()))

    for var in varlist:
        stats = summary[var]
        splitedvarlist = re.findall(r'[A-Z][a-z]+|\d+|[a-z]+|[A-Z]+', var)
        splitedvar = join_with_max_length(splitedvarlist)
        var = " ".join(splitedvar)
        csv_table += f"{var},"
        if "mean" in stats["split1"]:
            # 连续变量
            csv_table += f'{stats["split1"]["mean"]:.2f} ± {stats["split1"]["std"]:.2f},'
            csv_table += f'{stats["split2"]["mean"]:.2f} ± {stats["split2"]["std"]:.2f},'
        else:
            # 二分类变量
            split1_values = ', '.join([f"{k}: {v * 100:.1f}%" for k, v in stats["split1"].items()])
            split2_values = ', '.join([f"{k}: {v * 100:.1f}%" for k, v in stats["split2"].items()])
            csv_table += f"{split1_values}, {split2_values},"
        csv_table += f"{stats['p_value']:.3f}\n"
    return csv_table

def json_to_latex(summary, dataset_name1="Dataset 1", dataset_name2="Dataset 2"):
    latex_table = r"\begin{table}[htbp]\centering\begin{tabular}{lccc}\hline"
    latex_table += f"\nCharacteristic & {dataset_name1} & {dataset_name2} & P-value \\\\\n"
    latex_table += r"\hline"
    # 添加数据量信息的行
    latex_table += f"\nNumber of patients & {summary['N']['split1']} & {summary['N']['split2']} & \\\\\n"

    varlist = list(summary.keys())
    varlist.remove('N')
    # sort varlist by sortedvarlist
    varlist = sorted(varlist, key=lambda x: getranking(x, load_sorted_varlist()))
    for var in varlist:
        stats = summary[var]
        splitedvarlist = re.findall(r'[A-Z][a-z]+|\d+|[a-z]+|[A-Z]+', var)
        
        splitedvar = join_with_max_length(splitedvarlist)
        var = r"\makecell[l]{" + r" \\ ".join(splitedvar) + "}"

        latex_table += f"\n{var} & "

        if "mean" in stats["split1"]:
            # 连续变量
            latex_table += f'${stats["split1"]["mean"]:.2f} \\pm {stats["split1"]["std"]:.2f}$ & '
            latex_table += f'${stats["split2"]["mean"]:.2f} \\pm {stats["split2"]["std"]:.2f}$ & '
        else:
            # 二分类变量
            split1_values = ', '.join([f"{k}: {v * 100:.1f}\\%" for k, v in stats["split1"].items()])
            split2_values = ', '.join([f"{k}: {v * 100:.1f}\\%" for k, v in stats["split2"].items()])
            latex_table += f"{split1_values} & {split2_values} & "
        # add arrow indicating the direction of the difference
        if "mean" in stats["split1"] and stats['p_value'] < 0.01: 
            if stats['split1']['mean'] > stats['split2']['mean']:
                arrow = r"$\downarrow$"
            else:
                arrow = r"$\uparrow$"
        else:
            arrow = ""
        latex_table += f"{stats['p_value']:.3f} {arrow} \\\\\n"

    label =  f"{dataset_name1}_{dataset_name2}_{set_}".replace(" ", "_").lower()
    latex_table += r"\hline\end{tabular}"
    latex_table += f"\\caption{{{caption[label]}}} \\label{{tab:{label}}}\n"
    latex_table += r"\end{table}"
    return latex_table


def calculatecolinearity(df, set_):
    from scipy.stats import spearmanr

    if set_ == 'origi':
        df = df.drop(columns=['Outcome'])

        # Spearman
        spearmancorr = []

        # 逐对列计算 Spearman 相关系数和 p 值
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 == col2:
                    continue
                corr, p_value = spearmanr(df[col1], df[col2])
                spearmancorr.append({'var1': col1, 'var2': col2, 'correlation': corr, 'p_value': p_value, 'significant': p_value < 0.05 and abs(corr) > 0.5})

        # 将结果转换为 DataFrame 并保存
        spearmancorr = pd.DataFrame(spearmancorr)
        spearmancorr.to_csv("descriptoontable/spearman_correlation_origi.csv")

        
    if set_ == 'time':
        spearmanr_corr = []
        # group var by str before '_Avg'
        df = df.drop(columns=['Outcome'])
        unique_var_group = set([var.split('_Avg')[0] for var in df.columns if '_Avg' in var])
        for var_group in unique_var_group:
            var_group_vars = [var for var in df.columns if var_group in var]
            df_group = df[var_group_vars]
            for col1 in df_group.columns:
                for col2 in df_group.columns:
                    if col1 == col2:
                        continue
                    corr, p_value = spearmanr(df_group[col1], df_group[col2])
                    spearmanr_corr.append({'var1': col1, 'var2': col2, 'correlation': corr, 'p_value': p_value, 'significant': p_value < 0.05 and abs(corr) > 0.5})
        
        spearmanr_corr = pd.DataFrame(spearmanr_corr)
        spearmanr_corr.to_csv("descriptoontable/spearman_correlation_time.csv")




        


if __name__ == '__main__':
    caption = {
        'good_outcome_poor_outcome_origi': 'Comparison of the characteristics between patients with good and poor outcomes in the time independent dataset \\\ continuous variables are presented as mean ± standard deviation, categorical variables are presented as number (percentage) \\\ good outcome is defined as visit duration < 100 days, poor outcome is defined as visit duration $\geq$ 100 days',
        'good_outcome_poor_outcome_time': 'Comparison of the characteristics between patients with good and poor outcomes in the time dependent dataset \\\ continuous variables are presented as mean ± standard deviation, categorical variables are presented as number (percentage) \\\ good outcome is defined as visit duration < 100 days, poor outcome is defined as visit duration $\geq$ 100 days',
        'train_test_origi': 'Comparison of the characteristics between the training and testing datasets in the time independent dataset \\\ continuous variables are presented as mean ± standard deviation, categorical variables are presented as number (percentage)',
        'train_test_time': 'Comparison of the characteristics between the training and testing datasets in the time dependent dataset \\\ continuous variables are presented as mean ± standard deviation, categorical variables are presented as number (percentage)'

    }
    # nni9/beA3o82D 1112
    for config_plot, config_train, set_ in [('plot_distribution_normal.yaml','trainshap_normal.yaml','origi'),
                          ('plot_distribution_time.yaml','trainshap_timeseries.yaml', 'time')]:
        logger.info(f"Processing {config_plot} with {set_}")
        bestexpid, sequenceid = prepare_params(config_plot)
        fmodel, params, pp, fp= get_model_data_for_shap(config_train, bestexpid, sequenceid)

        rowna = 0.3
        k = 100
        X, y = get_data_for_des(fmodel, fp, params.copy(), 
                                rowna,
                                pp, k = k, randomrate= 0.2,
                                pick_key= 'all')
        logger.info(f"X shape: {X.shape}")
        
        df = pd.DataFrame(X)
        df['Outcome'] = y

        df_all, good_outcome, poor_outcome ,train_df, test_df= split_data(df, outcome_col='Outcome')
        # 生成统计总结
        logger.info("outcome summary")
        status = 'outcome'
        if not os.path.exists('descriptoontable'):
            os.makedirs('descriptoontable')

        summary = generate_summary_statistics(df_all, good_outcome, poor_outcome)
        latex_code = json_to_latex(summary, dataset_name1="Good Outcome", dataset_name2="Poor Outcome")
        with open (f"descriptoontable/latex_data_description_table_outcome_{set_}.tex", "w") as f:
            f.write(latex_code)

        csvtables = json_to_csv(summary, dataset_name1="Good Outcome", dataset_name2="Poor Outcome")
        with open (f"descriptoontable/csv_data_description_table_outcome_{set_}.csv", "w") as f:
            f.write(csvtables)

        logger.info("train test summary")
        status = 'train_test'

        summary = generate_summary_statistics(df_all, train_df, test_df)
        latex_code = json_to_latex(summary, dataset_name1="Train", dataset_name2="Test")
        with open (f"descriptoontable/latex_data_description_table_train_test_{set_}.tex", "w") as f:
            f.write(latex_code)

        csvtables = json_to_csv(summary, dataset_name1="Train", dataset_name2="Test")
        with open (f"descriptoontable/csv_data_description_table_train_test_{set_}.csv", "w") as f:
            f.write(csvtables)

        rowna = 0.5
        k = 100
        X, y = get_data_for_des(fmodel, fp, params.copy(), 
                                rowna,
                                pp, k = k, randomrate= 0.2,
                                pick_key= 'all')
        logger.info(f"X shape: {X.shape}")
        
        df = pd.DataFrame(X)
        df['Outcome'] = y

        calculatecolinearity(df, set_)