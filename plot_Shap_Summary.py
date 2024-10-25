import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt




def plot_shap_summary(plotshap_config, topn):
    pic_df = pd.DataFrame(columns=['expid', 'keyname','itemname', 'pic_dir'])
    for expid, _ in [('beA3o82D', 1112),( '43KOTlpS', 186),('lKesaFNR', 31)]:
        for plotparams in plotshap_config:
            key = plotparams['key']
            keyname = plotparams['keyname']
            rowna = plotparams['rowna']
            k = plotparams['k']
            shapdir = f'shap_plots_{keyname}_rowna{rowna}_k{k}_{topn}_{expid}'
            pics_dirs = [d for d in os.listdir(shapdir) if d.endswith('.png')]
            for pic_dir in pics_dirs:
                itemname = pic_dir.split('_')[0]
                pic_dir = os.path.join(shapdir, pic_dir)
                pic_df = pic_df._append({'expid': expid, 'keyname': keyname, 'itemname': itemname, 'pic_dir': pic_dir}, ignore_index=True)
    # sort by itemname then keyname
    # sort keyname by ['all', '0-2', '2-6', '6-12', '6+', '12+']
    pic_df['keyname'] = pd.Categorical(pic_df['keyname'], ['all', '0-2', '2-6', '6-12', '6+', '12+'])
    pic_df = pic_df.sort_values(by=['itemname', 'keyname'])

    pic_df.to_csv('pic_df.csv', index=False)

    # merge pics by itemname
    if not os.path.exists('shapsummary'):
        os.makedirs('shapsummary')
    for itemname in pic_df['itemname'].unique():
        item_df = pic_df[pic_df['itemname'] == itemname]
        
        # Create a figure with 6 rows and 3 columns, adjusting the size
        fig, axs = plt.subplots(6, 3, figsize=(15, 10))
        
        # Loop through the pic_dir for each item and add images to the grid
        for i, pic_dir in enumerate(item_df['pic_dir']):
            img = plt.imread(pic_dir)
            row = i // 3  # Determine the row index
            col = i % 3   # Determine the column index
            
            # Display the image in the corresponding subplot
            axs[row, col].imshow(img)
            axs[row, col].axis('off')  # Turn off axis

        # Save the combined figure for each itemname
        plt.tight_layout()
        plt.savefig(f'shapsummary/{itemname}_summary.png')
        plt.close(fig)
