# plot multiple auc curve in one plot
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os
from utils import check_y

def get_result(log_dir, experiment_id):
    with open (f'{log_dir}/{experiment_id}/results.jsonl', 'r') as f:
        for line in f:
            result = json.loads(line)
            yield result

def get_ppshape(log_dir, experiment_id):
    with open (f'{log_dir}/{experiment_id}/ppshape.jsonl', 'r') as f:
        for line in f:
            ppshape = json.loads(line)
            yield ppshape

def checkppshape(log_dir, experiment_id, sequence_id,threshold = 1000):
    for ppshape in get_ppshape(log_dir, experiment_id):
        if ppshape['sequence_id'] == sequence_id:
            if ppshape['ppresults']['shape'][0] > threshold:
                return True
            else:
                return False
            
    raise ValueError(f"ppshape not found for {experiment_id} {sequence_id}")

def get_result_by_sequence_id(log_dir, experiment_id, sequence_id):
    for result in get_result(log_dir, experiment_id):
        if result['sequence_id'] == sequence_id:
            return result
        
def best_sequence_id_byroc(log_dir, experiment_id, ppshape_threshold = 1000):
    best_sequence_id = None
    best_avg_avg_roc_auc = 0
    
    for result in get_result(log_dir, experiment_id):
        sequence_id = result['sequence_id']
        if checkppshape(log_dir, experiment_id, sequence_id, threshold= ppshape_threshold) == False:

            continue
        fold_results = result['fold_results']

        avg_42_roc_auc = np.mean([result['roc_auc_42'] for result in fold_results])
        avg_100_roc_auc = np.mean([result['roc_auc_100'] for result in fold_results])


        avg_avg_roc_auc = np.mean([avg_42_roc_auc, avg_100_roc_auc])
        if avg_avg_roc_auc > best_avg_avg_roc_auc:
            best_avg_avg_roc_auc = avg_avg_roc_auc
            best_sequence_id = sequence_id


    return best_sequence_id

def best_sequence_id_byloss(log_dir, experiment_id, ppshape_threshold = 1000):
    best_sequence_id = None
    minloss = 1000000
    
    for result in get_result(log_dir, experiment_id):
        sequence_id = result['sequence_id']
        if checkppshape(log_dir, experiment_id, sequence_id, threshold= ppshape_threshold) == False:

            continue
        fold_results = result['fold_results']

        loss = np.mean([result['loss'] for result in fold_results])
        if  loss < minloss:
            minloss = loss
            best_sequence_id = sequence_id

    return best_sequence_id
def plot_roc(data, groupname, log_dir, binary_threshold, colors_set, label_set, ppthreshold = 1000):

    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(8, 8))

    for experiment_id in data.keys():
        tprs = []
        aucs = []
        thr = []
        result = data[experiment_id]['results']

        for fold_result in result['fold_results']:
            # to array and check y
            y_array = np.array(fold_result['ry'])
            y_pred_array = np.array(fold_result['rypredict'])
            # okindex = check_y(y_array, y_pred_array, k=K)
            # y_array = y_array[okindex]
            # y_pred_array = y_pred_array[okindex]
            y_bi = np.where(y_array > binary_threshold, 1, 0)
            fpr, tpr, thresholds = roc_curve(y_bi, y_pred_array)
            # 计算Youden's Index
            youden_index = tpr - fpr
            best_index = youden_index.argmax()
            thr.append(thresholds[best_index])
            # 计算AUC
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, color=colors_set[experiment_id], lw=2, alpha=0.3)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)


        plt.plot(mean_fpr, mean_tpr, color=colors_set[experiment_id],
                label=f'{label_set[experiment_id]}: {mean_auc:.3f} ± {std_auc:.3f} (best thre: {np.mean(thr):.3f})',
                lw=2)
        
        plt.fill_between(mean_fpr, np.maximum(mean_tpr - np.std(tprs, axis=0), 0),
                        np.minimum(mean_tpr + np.std(tprs, axis=0), 1), color=colors_set[experiment_id], alpha=0.2)
    # 图形格式设置
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - 5-fold Cross Validation')
    plt.legend(loc="lower right")
    plt.savefig(f'{log_dir}/{groupname}_roc_curve_{binary_threshold}_thre{ppthreshold}.png')
    plt.close()

def plot_loss(data, log_dir, colors_set, label_set, ppthreshold = 1000):
    # plot loss dot plot for each experiment
    lossdf = pd.DataFrame()

    for experiment_id in data.keys():
        result = data[experiment_id]['results']
        loss_list = []
        for fr in result['fold_results']:
            y_array = np.array(fr['ry'])
            y_pred_array = np.array(fr['rypredict'])
            # okindex = check_y(y_array, y_pred_array, k=K)
            # y_array = y_array[okindex]
            # y_pred_array = y_pred_array[okindex]
            loss = mean_squared_error(y_array, y_pred_array)
            loss_list.append(loss)
        lossdf[label_set[experiment_id]] = loss_list

    plt.figure(figsize=(8, 8))
    
    # violin plot
    plt.violinplot(lossdf.values, showmeans=True, showmedians=True)

    plt.title('Loss Value - 5-fold Cross Validation')
    plt.ylabel('Loss Value')
    # make space under x-axis for labels
    plt.subplots_adjust(bottom=0.25)
    plt.xticks(rotation=45, ha='right')
    plt.xticks(range(1, len(lossdf.columns) + 1), lossdf.columns)
    
    # Save plot
    plt.savefig(f'{log_dir}/loss_boxplot_thre{ppthreshold}.png')
    plt.close()


def plot_y_predy(data, log_dir, colors_set, label_set, ppthreshold = 1000):
    for experiment_id in data.keys():
        result = data[experiment_id]['results']
        # 5 plot in one figure
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))  # 1行5列，调整figure大小
        
        for idx, fr in enumerate(result['fold_results']):
            y_array = np.array(fr['ry'])
            y_pred_array = np.array(fr['rypredict'])
            # okindex = check_y(y_array, y_pred_array, k=K)
            # y_array = y_array[okindex]
            # y_pred_array = y_pred_array[okindex]

            ax = axs[idx]  # 选择当前子图
            ax.scatter(y_array, y_pred_array, color=colors_set[experiment_id], alpha=0.3)
            ax.plot([0, 500], [0, 500], 
                    color='red', linestyle='-', linewidth=2)
            ax.set_ylabel('Predicted', fontsize=12)
            ax.set_xlabel('Actual', fontsize=12)
            ax.set_title(f'{label_set[experiment_id]} fold {fr["fold"]}', fontsize=12)
        # 保存包含五张图的figure
        plt.tight_layout()
        plt.savefig(f'{log_dir}/y_pred_{experiment_id}.png')
        plt.close()


def write_kfoldint_results(data, log_dir, label_set, ppthreshold = 1000):
    results_list = []
    #five fold receiver-operating-characteristic (ROC) curve (AUC), sensitivity, specificity, 
    # positive predictive value (PPV), negative predictive value (NPV), accuracy, and F1 score, were used to evaluated the reliability of these models
    for experiment_id in data.keys():
        result = data[experiment_id]['results']
        for fr in result['fold_results']:
            y_array = np.array(fr['ry'])
            y_pred_array = np.array(fr['rypredict'])
            # okindex = check_y(y_array, y_pred_array, k=K)
            # y_array = y_array[okindex]
            # y_pred_array = y_pred_array[okindex]

            loss = mean_squared_error(y_array, y_pred_array)
            for binary_threshold in [42, 100, 365]:
                y_bi = np.where(y_array > binary_threshold, 1, 0)
                fpr, tpr, thresholds = roc_curve(y_bi, y_pred_array)
                roc_auc = auc(fpr, tpr)
                youden_index = tpr - fpr
                best_index = youden_index.argmax()
                # The sensitivity, specificity, PPV, NPV, accuracy, and F1 score were calculated at the optimal cutoff value that maximized the Youden index.
                sensitivity = tpr[best_index]
                specificity = 1 - fpr[best_index]
                PPV = sensitivity / (sensitivity + (1 - specificity))
                NPV = specificity / (specificity + (1 - sensitivity))
                accuracy = (sensitivity + specificity) / 2
                F1 = 2 * sensitivity * PPV / (sensitivity + PPV)
                results_list.append({
                    'experiment_id': experiment_id,
                    'experiment_label': label_set[experiment_id],
                    'fold': fr['fold'],
                    'loss': loss,
                    'binary_threshold': binary_threshold,
                    'roc_auc': roc_auc,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'PPV': PPV,
                    'NPV': NPV,
                    'accuracy': accuracy,
                    'F1': F1
                })
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f'{log_dir}/kfoldint_results_thre{ppthreshold}.csv', index=False)

            

def main(log_dir, plotdir, colors_set, label_set, group_set, ppthreshold = 1000):
    data = {}

    for experiment_id in colors_set.keys():
        # sequence_id = best_sequence_id_byloss(log_dir, experiment_id, ppshape_threshold = ppthreshold)
        sequence_id = best_sequence_id_byroc(log_dir, experiment_id, ppshape_threshold = ppthreshold)
        result = get_result_by_sequence_id(log_dir, experiment_id, sequence_id)
        data[experiment_id] = {
            'sequence_id': sequence_id,
            'results': result   
        }

    if not os.path.exists(plotdir):
        os.makedirs(plotdir)

    with open(f'{plotdir}/bestparams_thre{ppthreshold}.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    
    for binary_threshold in [42, 100, 365]:
        for group in group_set.keys():
            subset_data = {experiment_id: data[experiment_id] for experiment_id in group_set[group]}
            plot_roc(subset_data, group, plotdir, binary_threshold, colors_set, label_set, ppthreshold = ppthreshold)

    plot_loss(data, plotdir, colors_set, label_set, ppthreshold = ppthreshold)
    plot_y_predy(data, plotdir, colors_set, label_set, ppthreshold = ppthreshold)

    write_kfoldint_results(data, plotdir, label_set, ppthreshold = ppthreshold)

if __name__ == "__main__":
    log_dir = 'kfoldint_explog'
    plotdir = 'kfoldint_plot_HDQAuzN8_default_top200_gr1'

    K = 5
    color_set = {
        "dTBCXYGr_default_top100_gr1": "blue",
        # "beA3o82D_default_top100_gr1": "red",
        "HDQAuzN8_default_top200_gr1": "red",
        "XE0MhN5r_default_top100_gr1": "blue",
        "1aTxj7zc_default_top100_gr1": "red",
        "FAbyiLmG_default_top100_gr1": "blue",
        "25m9QoAi_default_top250_gr1": "red",
        "YR1DQb9A_default_top100_gr1": "blue",
        "lKesaFNR_default_top100_gr1": "red",
        "7mJ4VYe5_default_top100_gr1": "blue",
        "NKgRQfcV_default_top25_gr1": "red"
    }
    

    label_set = {
        "dTBCXYGr_default_top100_gr1": "Xgboost + time independent",
        # "beA3o82D_default_top100_gr1": "Xgboost + time dependent",
        "HDQAuzN8_default_top200_gr1": "Xgboost + time dependent",
        "XE0MhN5r_default_top100_gr1": "random forest + time independent",
        "1aTxj7zc_default_top100_gr1": "random forest + time dependent",
        "FAbyiLmG_default_top100_gr1": "Adaboost + time independent",
        "25m9QoAi_default_top250_gr1": "Adaboost + time dependent",
        "YR1DQb9A_default_top100_gr1": "GBM + time independent",
        "lKesaFNR_default_top100_gr1": "GBM + time dependent",
        "7mJ4VYe5_default_top100_gr1": "SVM + time independent",
        "NKgRQfcV_default_top25_gr1": "SVM + time dependent"
    }


    group_set = {
        # "Xgboost": ["dTBCXYGr_default_top100_gr1", "beA3o82D_default_top100_gr1"],
        "Xgboost": ["dTBCXYGr_default_top100_gr1", "HDQAuzN8_default_top200_gr1"],
        "random forest": ["XE0MhN5r_default_top100_gr1", "1aTxj7zc_default_top100_gr1"],
        "Adaboost": ["FAbyiLmG_default_top100_gr1", "25m9QoAi_default_top250_gr1"],
        "GBM": ["YR1DQb9A_default_top100_gr1", "lKesaFNR_default_top100_gr1"],
        "SVM": ["7mJ4VYe5_default_top100_gr1", "NKgRQfcV_default_top25_gr1"]
    }

    for threshold in [1000]:
        main(log_dir, plotdir, color_set, label_set, group_set, ppthreshold = threshold)
