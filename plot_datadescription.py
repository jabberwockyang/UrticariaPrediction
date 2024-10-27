import pandas as pd
import numpy as np
from scipy import stats

import yaml
from loguru import logger
import json
import xgboost as xgb
import os
import seaborn as sns
from matplotlib import pyplot as plt
from joint_distribution import plot_gaussianmixture

from utils import custom_sort_key, load_data, check_y, reverse_y_scaling
from preprocessor import Preprocessor, get_asso_feat
from train_shap import get_model_data_for_shap


def plot_kde_distribution(X, y , exposure):
    # 如果用户选择保存图像，则检查是否已经存在
    col2 = 'VisitDuration'

    new_df = pd.DataFrame({
        exposure: X[exposure],
        col2: y
    })
    # 转换所有列为numerical 如果无法转换 coerce to NaN and drop
    new_df = new_df.apply(pd.to_numeric, errors='coerce')
    new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    new_df.dropna(inplace=True)
    # 判断是否任何一列全为空值
    if new_df[exposure].isnull().all() or new_df[col2].isnull().all() or len(new_df) == 0:
        logger.debug(f"No data for {exposure} vs {col2}, saving an empty plot.")
        # 如果两列数据全为空值，则不绘制图形 
        plt.figure()
        plt.title(f'KDE plot:  {exposure} vs {col2}')

        return plt.gcf()
    else:
        new_df = new_df.assign(VisitDuration_g = pd.cut(new_df[col2], 
                                              bins=[0, 42, 100, 365, 100000], 
                                              labels=['<6w', '6w-3m', '3m-1y', '>1y'],
                                              ordered = True, right =False))
        fig, ax = plt.subplots()
        for key, grp in new_df.groupby('VisitDuration_g'):
            if len(grp[exposure]) > 1 and pd.api.types.is_numeric_dtype(grp[exposure]) and not np.var(grp[exposure])  < 1e-6:
                
                # ax = grp[exposure].plot(kind='kde', ax=ax, label=key)
                # grp_df = pd.DataFrame(grp[exposure])
                sns.kdeplot(data=grp, x=exposure, ax=ax, label=key, common_norm= False)
                
            else:
                logger.debug(f"Group {key} is empty or has no variability. Skipping.")
        ax.legend()
        ax.set_xlabel(f"min-max scaled {exposure}")

        plt.title(f'KDE plot:  {exposure} vs {col2}')
        return fig
    


def plot_kde_in_group(X, y, kdedir, pp, name, show=False):
    if not os.path.exists(kdedir):
        os.makedirs(kdedir)
    for featgroup in pp.feature_filter.features_list:
        featlist = get_asso_feat(featgroup, X.columns)
        # sort featlist by order .split('_')[-1] preclinicals, acute, chronic 
        sorted_list = sorted(featlist, key=custom_sort_key)
        logger.debug(f"Plotting KDE for {featgroup} features: {sorted_list}")
        figs = [plot_kde_distribution(X, y, feat) for feat in sorted_list]
            
        # make a big plot with subplots for each feature
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i, feat in enumerate(sorted_list[:3]):  # Only take the first three features
            figs[i].canvas.draw()  # 渲染figure
            # 将每个子图的内容画到axs中的对应位置
            axs[i].imshow(figs[i].canvas.buffer_rgba())  # 将单个图像画到子图中
            axs[i].axis('off')  # 隐藏坐标轴
            axs[i].set_title(feat)  # 设置子图标题
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(kdedir, f'{featgroup}_{name}.png'))
        plt.clf()

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
    X, y, _, _, _ = preprocessor.preprocess(data,  # the unscaled version of data
                                1,
                                None,
                                row_na_threshold,
                                col_na_threshold,
                                pick_key= pick_key,
                                common_blood_test= True,
                                disable_scalingX= True
                                )
    Xs, ys, _, _, _ = preprocessor.preprocess(data, # the scaled version of data
                                scale_factor,
                                log_transform,
                                row_na_threshold,
                                col_na_threshold,
                                pick_key= pick_key,
                                common_blood_test= True
                                )
    # predict with scaled data get y_pred_by_xs
    dtest = xgb.DMatrix(Xs, label=ys)
    y_pred_by_xs = model.predict(dtest)

    reversed_ys = reverse_y_scaling(ys, scale_factor, log_transform) # reverse y scaling
    reversed_ys_pred = reverse_y_scaling(y_pred_by_xs, scale_factor, log_transform) # reverse y scaling
    okindex = check_y(reversed_ys, reversed_ys_pred, k, randomrate) # checky receive original data and return index of ok data
    X = X[okindex] # subset original X with okindex
    y = y[okindex] # subset original y with okindex
    # return X not scaled, y not scaled
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

    def format_summary(mean, std,median, q1, q3):
        return {"mean": mean, "std": std, "median": median, "q1": q1, "q3": q3}

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
            summary[col]["split1"] = format_summary(df_split1[col].mean(), df_split1[col].std(), df_split1[col].median(), df_split1[col].quantile(0.25), df_split1[col].quantile(0.75))
            summary[col]["split2"] = format_summary(df_split2[col].mean(), df_split2[col].std(), df_split2[col].median(), df_split2[col].quantile(0.25), df_split2[col].quantile(0.75))
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
            # 连续变量 Continuous values were presented as median [interquartile range].
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


def calculatecolinearity(df, corrdir, set_):
    if not os.path.exists(corrdir):
        os.makedirs(corrdir)
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
                spearmancorr.append({'var1': col1, 'var2': col2, 'correlation': corr, 'p_value': p_value, 'significant': p_value < 0.01 and abs(corr) > 0.75})
                
                if p_value < 0.01 and abs(corr) > 0.5:
                    gaus_fig = plot_gaussianmixture(df, col1, col2)
                    gaus_fig.savefig(f"{corrdir}/gaussian_{col1}_{col2}.png")
                    plt.clf()

        # 将结果转换为 DataFrame 并保存
        spearmancorr = pd.DataFrame(spearmancorr)
        spearmancorr.to_csv(f"{corrdir}/spearman_correlation_origi.csv")

        
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
                    spearmanr_corr.append({'var1': col1, 'var2': col2, 'correlation': corr, 'p_value': p_value, 'significant': p_value < 0.01 and abs(corr) > 0.75})

                    if p_value < 0.01 and abs(corr) > 0.5:
                        gaus_fig = plot_gaussianmixture(df_group, col1, col2)
                        gaus_fig.savefig(f"{corrdir}/gaussian_{col1}_{col2}.png")
                        plt.clf()

        spearmanr_corr = pd.DataFrame(spearmanr_corr)
        spearmanr_corr.to_csv(F"{corrdir}/spearman_correlation_time.csv")



def plot_outcome(df, tabledir, set_):
    # calculate percentage of df['Outcome'] under 42 100 365
    perlessthan42 = df[df['Outcome'] < 42].shape[0] / df.shape[0]
    per42_100 = df[(df['Outcome'] >= 42) & (df['Outcome'] < 100)].shape[0] / df.shape[0]
    per100_365 = df[(df['Outcome'] >= 100) & (df['Outcome'] < 365)].shape[0] / df.shape[0]
    per365_5y = df[(df['Outcome'] >= 365) & (df['Outcome'] < 1825)].shape[0] / df.shape[0]
    permorethan5y = df[df['Outcome'] >= 1825].shape[0] / df.shape[0]

    # plot histogram of outcome
    fig, ax = plt.subplots()
    ax.hist(df['Outcome'], bins=50, color='blue', alpha=0.7)
    ax.set_title('Outcome Distribution')
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Frequency')
    ax.axvline(x=42, color='red', linestyle='--', label='42 days')
    ax.axvline(x=100, color='green', linestyle='--', label='100 days')
    ax.axvline(x=365, color='purple', linestyle='--', label='365 days')
    ax.axvline(x=1825, color='orange', linestyle='--', label='5 years')
    ax.legend(
        title=f"Percentage of Outcome \n < 6w: {perlessthan42:.2f} \n 6w-3m: {per42_100:.2f} \n 3m-1y: {per100_365:.2f} \n 1-5 years: {per365_5y:.2f} \n > 5 years: {permorethan5y:.2f}",
        loc='upper right'
    )
    plt.savefig(f"{tabledir}/outcome_distribution_{set_}.png")
        
def plot_kde_summary(kdedir, name, modelexplanation):
    featurelist = modelexplanation['main']

    picdirs = [os.path.join(kdedir, f'{featgroup}_{name}.png') for featgroup in featurelist]
    fig, axs = plt.subplots(5 , 2, figsize=(30, 24))
    for i, picdir in enumerate(picdirs):
        img = plt.imread(picdir)
        row = i // 2
        col = i % 2
        axs[row, col].imshow(img)
        axs[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(kdedir, f'kde_summary_{name}.png'))
    plt.clf()


if __name__ == '__main__':
    caption = {
        'good_outcome_poor_outcome_origi': 'Comparison of the characteristics between patients with good and poor outcomes in the time independent dataset \\\ continuous variables are presented as mean ± standard deviation, categorical variables are presented as number (percentage) \\\ good outcome is defined as visit duration < 100 days, poor outcome is defined as visit duration $\geq$ 100 days',
        'good_outcome_poor_outcome_time': 'Comparison of the characteristics between patients with good and poor outcomes in the time dependent dataset \\\ continuous variables are presented as mean ± standard deviation, categorical variables are presented as number (percentage) \\\ good outcome is defined as visit duration < 100 days, poor outcome is defined as visit duration $\geq$ 100 days',
        'train_test_origi': 'Comparison of the characteristics between the training and testing datasets in the time independent dataset \\\ continuous variables are presented as mean ± standard deviation, categorical variables are presented as number (percentage)',
        'train_test_time': 'Comparison of the characteristics between the training and testing datasets in the time dependent dataset \\\ continuous variables are presented as mean ± standard deviation, categorical variables are presented as number (percentage)'
    }

    # nni9/beA3o82D 1112
    tabledir = f"data_description_tables"
    if not os.path.exists(tabledir):
        os.makedirs(tabledir)

    for bestexpid, sequenceid, config_train, set_ in [('dTBCXYGr', 429, 'trainshap_normal.yaml','origi'),
                                                      ('43KOTlpS', 186, 'trainshap_timeseries.yaml', 'time')]:
        logger.info(f"Processing {bestexpid} {sequenceid} with {set_}")
        
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

        def generate_description():
            df_all, good_outcome, poor_outcome ,train_df, test_df= split_data(df, outcome_col='Outcome')
            # 生成统计总结
            logger.info("outcome summary")

            summary = generate_summary_statistics(df_all, good_outcome, poor_outcome)
            latex_code = json_to_latex(summary, dataset_name1="Good Outcome", dataset_name2="Poor Outcome")
            with open (f"{tabledir}/latex_data_description_table_outcome_{set_}.tex", "w") as f:
                f.write(latex_code)

            csvtables = json_to_csv(summary, dataset_name1="Good Outcome", dataset_name2="Poor Outcome")
            with open (f"{tabledir}/csv_data_description_table_outcome_{set_}.csv", "w") as f:
                f.write(csvtables)

            logger.info("train test summary")

            summary = generate_summary_statistics(df_all, train_df, test_df)
            latex_code = json_to_latex(summary, dataset_name1="Train", dataset_name2="Test")
            with open (f"{tabledir}/latex_data_description_table_train_test_{set_}.tex", "w") as f:
                f.write(latex_code)

            csvtables = json_to_csv(summary, dataset_name1="Train", dataset_name2="Test")
            with open (f"{tabledir}/csv_data_description_table_train_test_{set_}.csv", "w") as f:
                f.write(csvtables)

            plot_outcome(df, tabledir, set_)


        generate_description()


        rowna = 0.5
        k = 100
        corrdir = f'correlation_plots_all_rowna{rowna}_k{k}_{set_}'
        corrdir = os.path.join(tabledir, corrdir)
        X, y = get_data_for_des(fmodel, fp, params.copy(), 
                                rowna,
                                pp, k = k, randomrate= 0.2,
                                pick_key= 'all')
        logger.info(f"X shape: {X.shape}")

        df = pd.DataFrame(X)
        df['Outcome'] = y

        calculatecolinearity(df, corrdir, set_)
        

        if set_ == 'time':
            rowna = 0.5
            k = 100
            kdedir = f'kde_plots_all_rowna{rowna}_k{k}'
            kdedir = os.path.join(tabledir, kdedir)
            X, y = get_data_for_des(fmodel, fp, params.copy(), 
                                    rowna, 
                                    pp, k = k, randomrate= 0.2,
                                    pick_key= 'all')
            logger.debug(f"Data shape: {X.shape}, {y.shape}")
            plot_kde_in_group(X, y, kdedir, pp, 'all')
            with open('model_explanation.json', 'r') as f:
                modelexplanation = json.load(f)

            plot_kde_summary(kdedir, 'all', modelexplanation)

            # plot_kde_summary('descriptoontable/kde_plots_all_rowna0.5_k100_top25', 'all', modelexplanation)