
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
from loguru import logger
from preprocessor import get_asso_feat
import seaborn as sns
from train_shap import get_model_data_for_shap, ModelReversingY, get_data_for_Shap
import shap
import joblib
import yaml
from utils import reverse_y_scaling



def plot_kde_distribution(X, y , exposure):
    # 如果用户选择保存图像，则检查是否已经存在
    col2 = 'VisitDuration'

    new_df = pd.DataFrame({
        exposure: X[exposure]*100,
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
    

def custom_sort_key(string):
    order = {'preclinical': 0, 'acute': 1, 'chronic': 2}
    for key in order:
        if key in string:
            return order[key]
    return float('inf')  # 如果字符串不包含任何key，放在最后

def plot_kde_in_group(X, y, name):
    if not os.path.exists(KDEDIR):
        os.makedirs(KDEDIR)
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
        plt.savefig(os.path.join(KDEDIR, f'{featgroup}_{name}.png'))
        plt.clf()

def get_shap_values(X, model):
    
    if X.shape[0] > 300:
        X100 = shap.utils.sample(X, 300) 
    else:
        X100 = X

    explainer = shap.Explainer(model.predict, X100)
    shap_values = explainer(X100)
    return shap_values, X100

def plot_beeswarm_in_group(shap_values, X, ageggroup):
    if not os.path.exists(SHAPDIR):
        os.makedirs(SHAPDIR)
    for index, featgroup in enumerate(pp.feature_filter.features_list):
        print(featgroup)
        featlist = get_asso_feat(featgroup, X.columns)
        sorted_list = sorted(featlist, key=custom_sort_key)
        shap.plots.beeswarm(shap_values[:,sorted_list],show=False)
        fig = plt.gcf()  # plt.gcf() 用于获取当前的图像对象
        fig.suptitle(f"{featgroup} in {ageggroup} age group")
        fig.savefig(f"{SHAPDIR}/{featgroup}_{ageggroup}.png",bbox_inches = 'tight')
        logger.debug(f"Saved shap plot for {featgroup} in {ageggroup} age group")
        plt.clf()

def plot_heatmap(shap_values, agegroup):
    if not os.path.exists(SHAPDIR):
        os.makedirs(SHAPDIR)
    ax = shap.plots.heatmap(shap_values, max_display=25, 
                            # instance_order=shap_values.sum(1),
                            show=False)
    ax.set_aspect(5) # 设置图像的纵横比
    ax.set_title(f"SHAP heatmap for {agegroup}")
    
    plt.savefig(f"{SHAPDIR}/heatmap_{agegroup}_orderx.png",bbox_inches = 'tight')

def plot_correlation(shap_values):
    for autoantibodyfeature in ['SMRNP_Avg_acute', 'SMRNP_Avg_chronic',
                                'Nucleosome_Avg_chronic', 'Nucleosome_Avg_acute',
                                'AntiJo1_Avg_acute','AntiJo1_Avg_chronic',
                                'Ro52_Avg_acute', 'Ro52_Avg_chronic', 
                                'AntiSSA_Avg_acute','AntiSSA_Avg_chronic',
                                'RibosomalPProtein_Avg_acute', 'RibosomalPProtein_Avg_chronic']:
        for n in ['NeutrophilsPercentage_Avg_acute', 'NeutrophilsPercentage_Avg_chronic']:
            shap.plots.scatter(shap_values[:, n], color=shap_values[:, autoantibodyfeature])
            plt.title(f"Correlation between {n} and {autoantibodyfeature}")
            plt.savefig(f"{SHAPDIR}/scatter_{n}_vs_{autoantibodyfeature}.png",bbox_inches = 'tight')
            

if __name__ == '__main__':
    # beA3o82D_1112_1_all
    config_path = 'trainshap_timeseries.yaml'
    fmodel, params, pp, fp= get_model_data_for_shap(config_path, 'beA3o82D', 1112)
    joblib.dump(fmodel, 'fmodel.pkl')

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    featurelist = config['train']['feature_list']
    topn = featurelist.split('/')[-1].split('_')[0]
    key = 'all'
    keyname = 'all'
    rowna = 0.5
    k = 100
    SHAPDIR = f'shap_plots_{keyname}_rowna{rowna}_k{k}_{topn}'
    KDEDIR = f'kde_plots_{keyname}_rowna{rowna}_k{k}_{topn}'

    X, y = get_data_for_Shap(fmodel, fp, params.copy(), 
                            rowna, 
                            pp, k = k, randomrate= 0.2,
                            pick_key= key)
    logger.debug(f"Data shape: {X.shape}, {y.shape}")
    y = reverse_y_scaling(y, params['scale_factor'], params['log_transform'])
    # plot_kde_in_group(X, y, keyname)


    shap_values, X = get_shap_values(X, ModelReversingY(fmodel, params))
    # plot_beeswarm_in_group(shap_values, X, keyname)

    # plot_heatmap(shap_values, keyname)

    plot_correlation(shap_values)
    
