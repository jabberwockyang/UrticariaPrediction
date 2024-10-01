# plot multiple auc curve in one plot
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd

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


def plot_roc(log_dir, binary_threshold, colors_set, label_set, ppthreshold = 1000):

    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(12, 12))

    for experiment_id in colors_set.keys():
        tprs = []
        aucs = []
        sequence_id = best_sequence_id(log_dir, experiment_id, ppshape_threshold = ppthreshold)
        result = get_result_by_sequence_id(log_dir, experiment_id, sequence_id)

        for fold_result in result['fold_results']:
            loss = fold_result['loss']
            associated_roc_auc_json = [roc_obj for roc_obj in fold_result['roc_auc_json'] if roc_obj['binary_threshold'] == binary_threshold][0]
            fpr = associated_roc_auc_json['fpr']
            tpr = associated_roc_auc_json['tpr']
            roc_auc = associated_roc_auc_json['roc_auc']
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

def plot_loss(log_dir, colors_set, label_set, ppthreshold = 1000):
    # plot loss dot plot for each experiment
    lossdf = pd.DataFrame()
    bestparams = {}
    for experiment_id in colors_set.keys():
        sequence_id = best_sequence_id(log_dir, experiment_id, ppshape_threshold = ppthreshold)
        result = get_result_by_sequence_id(log_dir, experiment_id, sequence_id)       
        loss_list = [fr['loss'] for fr in result['fold_results']]
        lossdf[label_set[experiment_id]] = loss_list
        bestparams[experiment_id] = {
            'sequence_id': sequence_id,
            'roc_auc_42': [fr['roc_auc_42'] for fr in result['fold_results']],
            'roc_auc_100': [fr['roc_auc_100'] for fr in result['fold_results']],
        }
    with open(f'{log_dir}/bestparams_thre{ppthreshold}.json', 'w') as f:
        json.dump(bestparams, f, ensure_ascii=False, indent=4)

    plt.figure(figsize=(8, 8))
    
    # Scatter plot for each experiment
    plt.boxplot(lossdf.values, labels=lossdf.columns, patch_artist=True)
    
    plt.title('Loss Value - 5-fold Cross Validation')
    plt.ylabel('Loss Value')
    plt.xticks(rotation=45, ha='right')
    
    # Display the legend (if needed)
    plt.legend(title='Experiments', loc='best')
    
    # Save plot
    plt.savefig(f'{log_dir}/loss_boxplot.png')
    plt.close()


# def plot_y_predy(log_dir, colors_set, label_set, ppthreshold = 1000):
#     for experiment_id in colors_set.keys():
#         sequence_id = best_sequence_id(log_dir, experiment_id, ppshape_threshold = ppthreshold)
#         result = get_result_by_sequence_id(log_dir, experiment_id, sequence_id)

#         for fr in result['fold_results']:
#             y = fr['ry']
#             y_pred = fr['rypredict']

#             plt.figure(figsize=(5, 5))
#             plt.scatter(y, y_pred)
#             plt.plot([0, 500], [0, 500], 
#                      color='red', linestyle='-', linewidth=2)
#             plt.ylabel('Predicted', fontsize=20)
#             plt.xlabel('Actual', fontsize=20)
#             plt.title(f'{label_set[experiment_id]}', fontsize=20)
#             plt.savefig(f'{log_dir}/y_pred_{experiment_id}_{fr["fold"]}.png')
#             plt.close()

if __name__ == "__main__":
    log_dir = 'kfoldint_explog'

    
    color_set = {
        "dTBCXYGr_default_top100_gr1": "blue",
        "beA3o82D_default_top100_gr1": "red",
        "XE0MhN5r_default_top100_gr1": "orange",
        "1aTxj7zc_default_top100_gr1": "green"
    }

    label_set = {
        "dTBCXYGr_default_top100_gr1": "Xgboost + original data",
        "beA3o82D_default_top100_gr1": "Xgboost + timeseries data",
        "XE0MhN5r_default_top100_gr1": "random forest + original data",
        "1aTxj7zc_default_top100_gr1": "random forest + timeseries data"
    }

    for threshold in [900, 1000, 1200]:
        for binary_threshold in [42, 100, 365]:
            plot_roc(log_dir, binary_threshold, color_set, label_set, ppthreshold = threshold)


    for threshold in [900, 1000, 1200]:
        plot_loss(log_dir, color_set, label_set, ppthreshold = threshold)
