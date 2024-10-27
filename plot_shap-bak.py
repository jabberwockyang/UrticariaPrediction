
from matplotlib import pyplot as plt
import os
import pandas as pd
from loguru import logger
import xgboost as xgb
import shap
import joblib
import yaml 
import json

from utils import custom_sort_key, load_data, check_y, reverse_y_scaling
from preprocessor import Preprocessor, get_asso_feat
from train_shap import get_model_data_for_shap, ModelReversingY



def get_shap_values(X, model, k=300):
    
    if X.shape[0] > k:
        X100 = shap.utils.sample(X, k, random_state=25)
    else:
        X100 = X

    explainer = shap.Explainer(model.predict, X100)
    shap_values = explainer(X100)
    return shap_values, X100

def get_data_for_Shap(model, filepath, parmas, 
                      row_na_threshold,
                      preprocessor: Preprocessor, k, randomrate, pick_key):
    # filter x, y for shap explainer
    scale_factor = parmas['scale_factor'] # 用于线性缩放目标变量
    log_transform = parmas['log_transform'] # 是否对目标变量进行对数变换
    col_na_threshold = parmas['col_na_threshold'] # 用于删除缺失值过多的列
    model_type = parmas.pop('model')    
    # 加载数据
    data = load_data(filepath)
    # 预处理数据
    X, y, _, _, _ = preprocessor.preprocess(data, 
                                scale_factor,
                                log_transform,
                                row_na_threshold,
                                col_na_threshold,
                                pick_key= pick_key,
                                common_blood_test= True
                                )
    
    # predict 
    if model_type == "xgboost":
        dtest = xgb.DMatrix(X, label=y)
        y_pred = model.predict(dtest)
    else:
        y_pred = model.predict(X)
    reversed_y = reverse_y_scaling(y, scale_factor, log_transform)
    reversed_y_pred = reverse_y_scaling(y_pred, scale_factor, log_transform)
    okindex = check_y(reversed_y, reversed_y_pred, k, randomrate) # checky receive original data and return index of ok data
    X = X[okindex]
    y = y[okindex]
    # return scaled X, y
    return X, y


def plot_beeswarm_in_group(shap_values, ageggroup, shapdir, show=False):
    if not os.path.exists(shapdir):
        os.makedirs(shapdir)
    for index, featgroup in enumerate(pp.feature_filter.features_list):
        print(featgroup)
        featlist = get_asso_feat(featgroup, X.columns)
        sorted_list = sorted(featlist, key=custom_sort_key)
        shap.plots.beeswarm(shap_values[:,sorted_list],show=False)
        fig = plt.gcf()  # plt.gcf() 用于获取当前的图像对象
        fig.suptitle(f"{featgroup} in {ageggroup} age group")
        if show:
            plt.show()
        else:
            fig.savefig(f"{shapdir}/{featgroup}_{ageggroup}.png",bbox_inches = 'tight')

        logger.debug(f"Saved shap plot for {featgroup} in {ageggroup} age group")
        plt.clf()

def plot_beeswarm_in_group_by_cohort(shap_values, shapdir, show=False):
    if not os.path.exists(f"{shapdir}/coho"):
        os.makedirs(f"{shapdir}/coho")
    cohorts = shap.Cohorts(
        age_0_2=shap_values[shap_values[:, "FirstVisitAge"].data < 2],
        age_2_6=shap_values[(shap_values[:, "FirstVisitAge"].data >= 2) & (shap_values[:, "FirstVisitAge"].data < 6)],
        age_6_12=shap_values[(shap_values[:, "FirstVisitAge"].data >= 6) & (shap_values[:, "FirstVisitAge"].data < 12)],
        age_12_plus=shap_values[shap_values[:, "FirstVisitAge"].data >= 12]
    )
    for co in cohorts.cohorts.keys():
        
        shap.plots.beeswarm(cohorts.cohorts[co],show=False)
        fig = plt.gcf()
        fig.suptitle(f"top 10 features in {co} group")
        if show:
            plt.show()
        else:
            fig.savefig(f"{shapdir}/coho/top10_coho_{co}.png",bbox_inches = 'tight')
        logger.debug(f"Saved shap plot for {co} age group")
        plt.clf()

        for index, featgroup in enumerate(pp.feature_filter.features_list):
            print(featgroup)
            featlist = get_asso_feat(featgroup, X.columns)
            sorted_list = sorted(featlist, key=custom_sort_key)

            shap.plots.beeswarm(cohorts.cohorts[co][:,sorted_list],show=False)
            fig = plt.gcf()
            fig.suptitle(f"{featgroup} in {co} group")
            if show:
                plt.show()
            else:
                fig.savefig(f"{shapdir}/coho/{featgroup}_coho_{co}.png",bbox_inches = 'tight')
            logger.debug(f"Saved shap plot for {featgroup} in {co} age group")
            plt.clf()
    

        
def plot_shap_summary(plotshap_config, model_explanation):
    pic_df = pd.DataFrame(columns=['expid', 'keyname','itemname', 'pic_dir'])
    for expid, _ in [('beA3o82D', 1112),( '43KOTlpS', 186),('lKesaFNR', 31)]:
        for plotparams in plotshap_config:
            key = plotparams['key']
            keyname = plotparams['keyname']
            rowna = plotparams['rowna']
            k = plotparams['k']
            shapdir = f'{SHAPFOLDER}/shap_plots_{keyname}_rowna{rowna}_k{k}_{TOPN}_{expid}'
            pics_dirs = [d for d in os.listdir(shapdir) if d.endswith('.png')]
            for pic_dir in pics_dirs:
                itemname = pic_dir.split('_')[0]
                pic_dir = os.path.join(shapdir, pic_dir)
                pic_df = pic_df._append({'expid': expid, 'keyname': keyname, 'itemname': itemname, 'pic_dir': pic_dir}, ignore_index=True)
    # sort by itemname then keyname
    # sort keyname by ['all', '0-2', '2-6', '6-12', '6+', '12+']
    pic_df['keyname'] = pd.Categorical(pic_df['keyname'], ['all', '0-2', '2-6', '6-12', '6+','12+'])
    pic_df = pic_df.sort_values(by=['itemname', 'keyname'])
    pic_df.to_csv('pic_df.csv', index=False)


    # remove '6+' from keyname
    pic_df = pic_df[pic_df['keyname'] != '6+']
    # remove 'beA3o82D' 'lKesaFNR' from expid
    pic_df = pic_df[~pic_df['expid'].isin(['lKesaFNR'])]

    # merge pics by itemname
    if not os.path.exists(SHAPSUMDIR):
        os.makedirs(SHAPSUMDIR)
    for itemname in pic_df['itemname'].unique():
        item_df = pic_df[pic_df['itemname'] == itemname]
        # nrows with number of keyname
        nrows = len(item_df['keyname'].unique())
        # ncols with number of expid
        ncols = len(item_df['expid'].unique())

        # Create a figure with 6 rows and 3 columns, adjusting the size the unit figure size is 10 in width and 3 in height
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize = (10*ncols, 3*nrows))
        
        # Loop through the pic_dir for each item and add images to the grid
        for i, pic_dir in enumerate(item_df['pic_dir']):
            img = plt.imread(pic_dir)
            if ncols == 1:
                axs[i].imshow(img)
                axs[i].axis('off')
            else:
                # Get the row and column index of the subplot
                row = i // ncols
                col = i % ncols
                
                # Display the image in the corresponding subplot
                axs[row, col].imshow(img)
                axs[row, col].axis('off')  # Turn off axis

        # Save the combined figure for each itemname
        plt.tight_layout()
        plt.savefig(f'{SHAPSUMDIR}/{itemname}_summary.png')
        plt.close(fig)

    # combine itemname in list for a certain expid
    for expid in pic_df['expid'].unique():
        for figtype in model_explanation.keys():
            # 2 columns and rows depending on the number of itemnames
            nrows = (len(model_explanation[figtype]) + 1) // 2
            fig, axs = plt.subplots(nrows, 2, figsize=(20, 3*nrows))
            for i, itemname in enumerate(model_explanation[figtype]):
                pic_dir = pic_df[(pic_df['expid'] == expid) & (pic_df['itemname'] == itemname) & (pic_df['keyname'] == 'all')]['pic_dir'].values[0]
                img = plt.imread(pic_dir)
                row = i // 2
                col = i % 2
                axs[row, col].imshow(img)
                axs[row, col].axis('off')
            plt.tight_layout()
            plt.savefig(f'{SHAPSUMDIR}/{expid}_{figtype}_summary.png')
            plt.close(fig)




if __name__ == '__main__':
    SHAPFOLDER = 'shap'
    config_path = 'trainshap_timeseries.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    featurelist = config['train']['feature_list']
    TOPN = featurelist.split('/')[-1].split('_')[0]

    plotshap_param = 'plot_shap.json'
    with open(plotshap_param, 'r') as file:
        plotshap_config = json.load(file)

    for expid, seqid in [
                        # ('beA3o82D', 1112),
                         ('43KOTlpS', 186),
                        #  ('lKesaFNR', 31)
                         ]:
        fmodel, params, pp, fp= get_model_data_for_shap(config_path, expid, seqid)
        joblib.dump(fmodel, 'fmodel.pkl')

        for plotparams in plotshap_config:

            key = plotparams['key']
            keyname = plotparams['keyname']
            rowna = plotparams['rowna']
            k = plotparams['k']

            if keyname != 'all':
                continue
            shapdir = f'{SHAPFOLDER}/shap_plots_{keyname}_rowna{rowna}_k{k}_{TOPN}_{expid}'
            
            X, y = get_data_for_Shap(fmodel, fp, params.copy(), 
                                    rowna, 
                                    pp, k = k, randomrate= 0.2,
                                    pick_key= key)
            # SCALED X, y only x is used for shap values
            logger.debug(f"{expid} {seqid} {keyname} {rowna} {k}")
            logger.debug(f"Data shape: {X.shape}, {y.shape}")            
            # shap requires a model receive scaled X and original y
            shap_values, X = get_shap_values(X, ModelReversingY(fmodel, params), 300) 
            print(f"shapdir  {shapdir}")
            print(f"shap_values base value: {shap_values.base_values[0]}")

            # plot_beeswarm_in_group(shap_values, ageggroup = keyname, shapdir = shapdir)
            if keyname == 'all':
                plot_beeswarm_in_group_by_cohort(shap_values, shapdir = shapdir)

    SHAPSUMDIR = f'{SHAPFOLDER}/shap_summary_{TOPN}'
    # with open('model_explanation.json', 'r') as file:
    #     model_explanation = json.load(file)

    # plot_shap_summary(plotshap_config, model_explanation)

            
