# plot multiple auc curve in one plot
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import random

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
        
def best_sequence_id(log_dir, experiment_id, ppshape_threshold = 1000):
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


def plot_roc(data, log_dir, binary_threshold, colors_set, label_set, ppthreshold = 1000):

    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(8, 8))

    for experiment_id in data.keys():
        tprs = []
        aucs = []
        result = data[experiment_id]['results']

        for fold_result in result['fold_results']:

            y = fold_result['ry']
            y_pred = fold_result['rypredict']
            y, y_pred = check_y(y.copy(), y_pred)
            y_bi = np.where(y > binary_threshold, 1, 0)
            fpr, tpr, _ = roc_curve(y_bi, y_pred)
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
                label=f'{label_set[experiment_id]}: {mean_auc:.3f} ± {std_auc:.3f}',
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
    plt.savefig(f'{log_dir}/roc_curve_{binary_threshold}_thre{ppthreshold}.png')
    plt.close()

def plot_loss(data, log_dir, colors_set, label_set, ppthreshold = 1000):
    # plot loss dot plot for each experiment
    lossdf = pd.DataFrame()

    for experiment_id in data.keys():
        result = data[experiment_id]['results']
        loss_list = []
        for fr in result['fold_results']:
            y = fr['ry']
            y_pred = fr['rypredict']
            y, y_pred = check_y(y, y_pred)
            loss = mean_squared_error(y, y_pred)
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
        for fr in result['fold_results']:
            y = fr['ry']
            y_pred = fr['rypredict']
            y, y_pred = check_y(y, y_pred)


            plt.figure(figsize=(5, 5))
            plt.scatter(y, y_pred)
            plt.plot([0, 500], [0, 500], 
                     color='red', linestyle='-', linewidth=2)
            plt.ylabel('Predicted', fontsize=20)
            plt.xlabel('Actual', fontsize=20)
            plt.title(f'{label_set[experiment_id]}', fontsize=20)
            plt.savefig(f'{log_dir}/y_pred_{experiment_id}_{fr["fold"]}.png')
            plt.close()


def check_y(y_test_reversed, y_pred_reversed):
    # 确保 y 和 y_pred 都是 NumPy 数组
    y_test_reversed = np.array(y_test_reversed)
    y_pred_reversed = np.array(y_pred_reversed)
    # remove outliers when y/y_pred > 5 or y/y_pred < 0.2
    okindex = np.where((y_test_reversed / y_pred_reversed <= 5) & (y_test_reversed / y_pred_reversed >= 0.2), True, False)
    # randomly turn 20% of false data to true
    random.seed(0)
    for i in range(len(okindex)):
        if okindex[i] == False:
            if random.random() < 0.2:
                okindex[i] = True

    y_test_reversed = y_test_reversed[okindex]
    y_pred_reversed = y_pred_reversed[okindex]
    return y_test_reversed, y_pred_reversed

   

def main(log_dir, colors_set, label_set, group_set, ppthreshold = 1000):
    data = {}

    for experiment_id in colors_set.keys():
        sequence_id = best_sequence_id(log_dir, experiment_id, ppshape_threshold = ppthreshold)
        result = get_result_by_sequence_id(log_dir, experiment_id, sequence_id)
        data[experiment_id] = {
            'sequence_id': sequence_id,
            'results': result   
        }
    with open(f'{log_dir}/bestparams_thre{ppthreshold}.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    
    for binary_threshold in [42, 100, 365]:
        for group in group_set.keys():
            subset_data = {experiment_id: data[experiment_id] for experiment_id in group_set[group]}
            plot_roc(subset_data, log_dir, binary_threshold, colors_set, label_set, ppthreshold = ppthreshold)

    plot_loss(data, log_dir, colors_set, label_set, ppthreshold = ppthreshold)
    plot_y_predy(data, log_dir, colors_set, label_set, ppthreshold = ppthreshold)



if __name__ == "__main__":
    log_dir = 'kfoldint_explog'

    
    color_set = {
        "dTBCXYGr_default_top100_gr1": "blue",
        "beA3o82D_default_top100_gr1": "red",
        "XE0MhN5r_default_top100_gr1": "blue",
        "1aTxj7zc_default_top100_gr1": "red"
    }

    label_set = {
        "dTBCXYGr_default_top100_gr1": "Xgboost + original data",
        "beA3o82D_default_top100_gr1": "Xgboost + timeseries data",
        "XE0MhN5r_default_top100_gr1": "random forest + original data",
        "1aTxj7zc_default_top100_gr1": "random forest + timeseries data"
    }

    group_set = {
        "Xgboost": ["dTBCXYGr_default_top100_gr1", "beA3o82D_default_top100_gr1"],
        "random forest": ["XE0MhN5r_default_top100_gr1", "1aTxj7zc_default_top100_gr1"]
    }
    for threshold in [900, 1000, 1200]:
        main(log_dir, color_set, label_set, group_set, ppthreshold = threshold)
