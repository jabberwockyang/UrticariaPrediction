
import pandas as pd
from utils import check_y
from sklearn.metrics import roc_curve, auc, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def plot_roc(fold_result_list, log_dir, binary_threshold, colors_set, label_set, topn=None):
    plt.figure(figsize=(8, 8))

    for sequence_id in fold_result_list.keys():
        if topn:
            if label_set[sequence_id] != topn:
                continue

        fold_result = fold_result_list[sequence_id]
        # to array and check y
        y_array = np.array(fold_result['ry'])
        y_pred_array = np.array(fold_result['rypredict'])
        okindex = check_y(y_array, y_pred_array, k=K)

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
    if topn:
        plt.savefig(f'{log_dir}/roc_curve_{binary_threshold}_{topn}.png')
    else:
        plt.savefig(f'{log_dir}/roc_curve_{binary_threshold}.png')
    plt.close()



def plot_y_predy(fold_result_list, log_dir, colors_set, label_set):
    
    # 5 plot in one figure
    fig, axs = plt.subplots(1, len(fold_result_list.keys()), figsize=(5*len(fold_result_list.keys()), 5))
    
    for idx, sequence_id in enumerate(fold_result_list.keys()):
        fr = fold_result_list[sequence_id]
        y_array = np.array(fr['ry'])
        y_pred_array = np.array(fr['rypredict'])
        okindex = check_y(y_array, y_pred_array, k=K)
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

    fig, axs = plt.subplots(2, len(fold_result_list.keys()), figsize=(5*len(fold_result_list.keys()), 10))

    for idx, sequence_id in enumerate(fold_result_list.keys()):
        fr = fold_result_list[sequence_id]
        y_array = np.array(fr['ry'])
        y_pred_array = np.array(fr['rypredict'])
        okindex = check_y(y_array, y_pred_array, k=K)
        y_array = y_array[okindex]
        y_pred_array = y_pred_array[okindex]

        ax = axs[0][idx]  # 选择当前子图
        # plot histogram of y 
        ax.hist(y_array, bins=50, color='blue', alpha=0.7)
        ax.set_title(f'{label_set[sequence_id]} {fr["trainset"]}', fontsize=12)        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Frequency')
        ax.axvline(x=42, color='red', linestyle='--', label='42 days')
        ax.axvline(x=100, color='green', linestyle='--', label='100 days')
        ax.axvline(x=365, color='purple', linestyle='--', label='365 days')
        ax.legend(loc='upper right')
        ax = axs[1][idx]  # 选择当前子图
        # plot histogram of y_pred
        ax.hist(y_pred_array, bins=50, color='blue', alpha=0.7)
        ax.set_title(f'{label_set[sequence_id]} {fr["trainset"]}', fontsize=12)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Frequency')
        ax.axvline(x=42, color='red', linestyle='--', label='42 days')
        ax.axvline(x=100, color='green', linestyle='--', label='100 days')
        ax.axvline(x=365, color='purple', linestyle='--', label='365 days')
        ax.legend(loc='upper right')
    # 保存包含五张图的figure
    plt.tight_layout()
    plt.savefig(f'{log_dir}/y_pred_hist_{sequence_id}.png')
    plt.close()



def write_extval_result(fold_result_list, log_dir, experiment_id, label_set):
    result_list = []
    for sequence_id in fold_result_list.keys():
        fr = fold_result_list[sequence_id]
        y_array = np.array(fr['ry'])
        y_pred_array = np.array(fr['rypredict'])
        okindex = check_y(y_array, y_pred_array, k=K)
        y_array = y_array[okindex]
        y_pred_array = y_pred_array[okindex]


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
            result_list.append({
                'experiment_id': experiment_id,
                'trainset': fr['trainset'],
                'sequence_id': sequence_id,
                'label': label_set[sequence_id],
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
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(f'{log_dir}/extval_result.csv', index=False)



def main(logdir, expid, evalset):

    resultfilepath = f'{logdir}/{expid}/results.jsonl'
    with open(resultfilepath, 'r') as f:
        lines = f.readlines()
    jsonlist = [json.loads(line) for line in lines]
    fold_result_list = {}
    for j in jsonlist:
        sequence_id = j['sequence_id']
        fold_result_list[sequence_id] = [fr for fr in j['fold_results'] if fr['trainset'] == evalset][0]
    if not os.path.exists(os.path.join(logdir, expid,evalset)):
        os.makedirs(os.path.join(logdir, expid,evalset))
    # geenrate colorset by length of fold_result_list
    colors_set = {}
    label_set = {}
    for idx, sequence_id in enumerate(fold_result_list.keys()):
        colors_set[sequence_id] = plt.cm.tab10(idx)
        label_set[sequence_id] = sequence_id.split('_')[-1]

    # for binary_threshold in [42,100, 365]:
    #     plot_roc(fold_result_list, os.path.join(logdir, expid,evalset), binary_threshold, colors_set, label_set)
    #     plot_roc(fold_result_list, os.path.join(logdir, expid,evalset), binary_threshold, colors_set, label_set, topn='top25')
    plot_y_predy(fold_result_list, os.path.join(logdir, expid,evalset), colors_set, label_set)

    # write_extval_result(fold_result_list, os.path.join(logdir, expid,evalset), expid,label_set)



if __name__ == '__main__':
    K = 5

    main('extval_explog', 'beA3o82D_1112_extval_gr1', 'deriva')
    main('extval_explog', 'beA3o82D_1112_extval_gr1', 'test_ext')
    # # --expid CWQJ9nlD --sequenceid 1052
    main('extval_explog', 'CWQJ9nlD_1052_extval_gr1', 'deriva')
    main('extval_explog', 'CWQJ9nlD_1052_extval_gr1', 'test_ext')
    # #  --expid 1aTxj7zc --sequenceid 266 
    # main('extval_explog', '1aTxj7zc_266_extval_gr1', 'deriva')
    # main('extval_explog', '1aTxj7zc_266_extval_gr1', 'test_ext')
    # # --expid lKesaFNR --sequenceid 31 
    # main('extval_explog', 'lKesaFNR_31_extval_gr1', 'deriva')
    # main('extval_explog', 'lKesaFNR_31_extval_gr1', 'test_ext')

    # HDQAuzN8_4866_extval_gr1
    main('extval_explog', 'HDQAuzN8_4866_extval_gr1', 'deriva')
    main('extval_explog', 'HDQAuzN8_4866_extval_gr1', 'test_ext')

    # zLCPym1l_374_1_all
    main('extval_explog', 'zLCPym1l_374_extval_gr1', 'deriva')
    main('extval_explog', 'zLCPym1l_374_extval_gr1', 'test_ext')



