    
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from utils import load_feature_list_from_boruta_file

def plot_boruta(ranking_df,log_dir, name = 'boruta'):
    numeric_ranking_df = ranking_df.apply(pd.to_numeric, errors='coerce')

    median_values = numeric_ranking_df.median()
    sorted_columns = median_values.sort_values().index

    # 设置绘图风格
    plt.figure(figsize=(25, 8))
    sns.set_theme(style="whitegrid")

    # 绘制箱线图
    sns.boxplot(data=numeric_ranking_df[sorted_columns], palette="Greens")
    # invert the y axis
    plt.gca().invert_yaxis()
    plt.xticks(rotation=90)
    plt.title("Sorted Feature Ranking Distribution by Boruta", fontsize=16)
    plt.xlabel("Attributes", fontsize=14)
    plt.ylabel("Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{name}.png'))
    plt.close()

def plot_boruta_by_group(ranking_df, log_dir):
    numeric_ranking_df = ranking_df.apply(pd.to_numeric, errors='coerce')
    features = numeric_ranking_df.columns
    groups = set([f.split('_')[0] for f in features])
    # make a new df to store the median ranking for each group
    group_ranking_df = pd.DataFrame(columns=list(groups))   

    for group in groups:
        # Filter features for the current group
        group_features = [f for f in features if f.startswith(group)]
        group_ranking_df[group] = numeric_ranking_df[group_features].max(axis=1)
    
    plot_boruta(group_ranking_df, log_dir, name='boruta_by_group')
    return group_ranking_df

def generate_topN_features(ranking_df, confirmed_vars_file, log_dir):
    confirmed_vars = load_feature_list_from_boruta_file(confirmed_vars_file)
    nvars = len(confirmed_vars)

    ranking_df = plot_boruta_by_group(ranking_df, log_dir)
    numeric_ranking_df = ranking_df.apply(pd.to_numeric, errors='coerce')

    median_values = numeric_ranking_df.median()
    sorted_columns = median_values.sort_values().index
    # rank confirmed vars by ranking_df
    confirmed_vars = [f for f in sorted_columns if f in confirmed_vars]
    ilist = [i for i in range(5, nvars+1, 5)]
    if nvars+1 not in ilist: # make sure nvars is in the list
        ilist.append(nvars+1)
    for i in ilist:
        topN_vars = confirmed_vars[:i]
        with open(os.path.join(log_dir, f'top{i}_confirmed_vars.txt'), 'w') as f:
            f.write('\n'.join(topN_vars))
    
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='boruta_explog')
    parser.add_argument('--experiment_name', type=str)
    return parser.parse_args()
    
if __name__ == '__main__':
    args = argparser()
    exp_dir = os.path.join(args.log_dir, args.experiment_name)
    ranking_df = pd.read_csv(os.path.join(exp_dir, 'ranking_df.csv'))
    plot_boruta(ranking_df, exp_dir)
    plot_boruta_by_group(ranking_df, exp_dir)

    generate_topN_features(ranking_df, os.path.join(exp_dir, 'confirmed_vars.txt'), exp_dir)