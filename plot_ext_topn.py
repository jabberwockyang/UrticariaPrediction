
import pandas as pd
from utils import check_y
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def plot_roc(fold_result_list, log_dir, binary_threshold, colors_set, label_set):
    plt.figure(figsize=(8, 8))

    for sequence_id in fold_result_list.keys():
        fold_result = fold_result_list[sequence_id]
        # to array and check y
        y_array = np.array(fold_result['ry'])
        y_pred_array = np.array(fold_result['rypredict'])
        okindex = check_y(y_array, y_pred_array)

        y_array = y_array[okindex]
        y_pred_array = y_pred_array[okindex]
        y_bi = np.where(y_array > binary_threshold, 1, 0)
        fpr, tpr, thresholds = roc_curve(y_bi, y_pred_array)
        # 计算Youden's Index
        youden_index = tpr - fpr
        best_index = youden_index.argmax()
        best_threshold = thresholds[best_index]
        # 计算AUC
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors_set[sequence_id], lw=2,
                label=f'{label_set[sequence_id]}: {roc_auc:.3f} (best thre: {best_threshold:.3f})'
            )

       
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - topn')
    plt.legend(loc="lower right")
    plt.savefig(f'{log_dir}/roc_curve_{binary_threshold}.png')
    plt.close()



def plot_y_predy(fold_result_list, log_dir, colors_set, label_set):
    
    # 5 plot in one figure
    fig, axs = plt.subplots(1, len(fold_result_list.keys()), figsize=(5*len(fold_result_list.keys()), 5))
    
    for idx, sequence_id in enumerate(fold_result_list.keys()):
        fr = fold_result_list[sequence_id]
        y_array = np.array(fr['ry'])
        y_pred_array = np.array(fr['rypredict'])
        okindex = check_y(y_array, y_pred_array)
        y_array = y_array[okindex]
        y_pred_array = y_pred_array[okindex]

        ax = axs[idx]  # 选择当前子图
        ax.scatter(y_array, y_pred_array, color=colors_set[sequence_id], alpha=0.5)
        ax.plot([0, 500], [0, 500], 
                color='red', linestyle='-', linewidth=2)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_xlabel('Actual', fontsize=12)
        ax.set_title(f'{label_set[sequence_id]} {fr["trainset"]}', fontsize=12)
    # 保存包含五张图的figure
    plt.tight_layout()
    plt.savefig(f'{log_dir}/y_pred_{sequence_id}.png')
    plt.close()

def main(logdir, expid, evalset):

    resultfilepath = f'{logdir}/{expid}/results.jsonl'
    with open(resultfilepath, 'r') as f:
        lines = f.readlines()
    jsonlist = [json.loads(line) for line in lines]
    fold_result_list = {}
    for j in jsonlist:
        sequence_id = j['sequence_id']
        fold_result_list[sequence_id] = [fr for fr in j['fold_results'] if fr['trainset'] == evalset][0]
    if not os.path.exists(os.path.join(logdir, evalset)):
        os.makedirs(os.path.join(logdir, evalset))
    # geenrate colorset by length of fold_result_list
    colors_set = {}
    label_set = {}
    for idx, sequence_id in enumerate(fold_result_list.keys()):
        colors_set[sequence_id] = plt.cm.tab10(idx)
        label_set[sequence_id] = sequence_id.split('_')[-1]
    for binary_threshold in [42,100, 365]:
        plot_roc(fold_result_list, os.path.join(logdir, evalset), binary_threshold, colors_set, label_set)
    plot_y_predy(fold_result_list, os.path.join(logdir, evalset), colors_set, label_set)


if __name__ == '__main__':


    main('extval_explog', 'beA3o82D_1112_extval_gr1', 'deriva')
    main('extval_explog', 'beA3o82D_1112_extval_gr1', 'test_ext')

