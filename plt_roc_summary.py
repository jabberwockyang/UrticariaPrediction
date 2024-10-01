import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
def plot_roc_summary(json_file): 
    with open(json_file, 'r') as f:
        results = json.load(f)
    alllist = []
    paramresults = [r['paramresult'] for r in results]
    for paramresult in paramresults:
        alllist.extend(paramresult)
    df = pd.DataFrame(alllist)
    df.to_csv(json_file.replace('.json', '.csv'), index=False)
    # plot dot plot max_roc_auc in different group x axis is group y axis is max_roc_auc
    # different objective with different color
    df['max_roc_auc'] = df['max_roc_auc'].astype(float)
    uniquegroups = df['group'].unique() 
    orderedlist = sorted(uniquegroups, key = lambda x: int(re.split(r'[-+]', x)[0] if re.match(r'^\d', x) else 9999))
    df['group'] = pd.Categorical(df['group'], categories = orderedlist, ordered = True)

    plt.figure(figsize=(6, 5))
    # violinplot with dots  
    sns.violinplot(data = df, x = 'group', y = 'max_roc_auc')
    sns.stripplot(data = df, x = 'group', y = 'max_roc_auc', color = 'orange', size = 6, jitter = 0.25)
    plt.xlabel('group')
    plt.ylabel('max_roc_auc')
    plt.ylim(0.5, 1)
    plt.title('max_roc_auc in different group')
    plt.savefig(json_file.replace('.json', '_roc_auc.png'))

if __name__ == '__main__':
    f = ['/root/ClinicalXgboost/gr_explog/3eQbjfcG_default_top7/3eQbjfcG_results.json'
]
    for json_file in f:
        plot_roc_summary(json_file) 
