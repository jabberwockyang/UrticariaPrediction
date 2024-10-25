
from matplotlib import pyplot as plt
import os
import numpy as np
import shap


def plot_correlation(shap_values, a, b, subfolder=None):
    # if unqiue value length of shap_values[:, n] is less than 10, set x-jitter to 1
    if len(np.unique(shap_values[:, a].data)) < 10:
        shap.plots.scatter(shap_values[:, a], color=shap_values[:, b], x_jitter=3)
    else:
        shap.plots.scatter(shap_values[:, a], color=shap_values[:, b])
    plt.title(f"Correlation between {a} and {b}")
    if subfolder:
        if not os.path.exists(f"{SHAPDIR}/{subfolder}"):
            os.makedirs(f"{SHAPDIR}/{subfolder}")
        plt.savefig(f"{SHAPDIR}/{subfolder}/scatter_{a}_vs_{b}.png",bbox_inches = 'tight')
    else:
        plt.savefig(f"{SHAPDIR}/scatter_{a}_vs_{b}.png",bbox_inches = 'tight')  
    plt.clf()

def plot_correlation_main(shap_values):
    for autoantibodyfeature in ['SMRNP_Avg_acute', 'SMRNP_Avg_chronic',
                                'Nucleosome_Avg_chronic', 'Nucleosome_Avg_acute',
                                'AntiJo1_Avg_acute','AntiJo1_Avg_chronic',
                                'Ro52_Avg_acute', 'Ro52_Avg_chronic', 
                                'AntiSSA_Avg_acute','AntiSSA_Avg_chronic',
                                'RibosomalPProtein_Avg_acute', 'RibosomalPProtein_Avg_chronic']:
        for n in ['NeutrophilsPercentage_Avg_acute', 'NeutrophilsPercentage_Avg_chronic']:
            plot_correlation(shap_values, n, autoantibodyfeature)
            plot_correlation(shap_values, autoantibodyfeature, n)
    for a in [
            #   'ImmunoglobulinE_Avg_acute', 'ImmunoglobulinE_Avg_chronic',
              'WhiteBloodCellCount_Avg_preclinical', 'WhiteBloodCellCount_Avg_acute', 'WhiteBloodCellCount_Avg_chronic',
              'BasophilsPercentage_Avg_preclinical', 'BasophilsPercentage_Avg_acute', 'BasophilsPercentage_Avg_chronic',
              'NeutrophilsPercentage_Avg_preclinical', 'NeutrophilsPercentage_Avg_acute', 'NeutrophilsPercentage_Avg_chronic',
              'EosinophilCountAbsolute_Avg_acute', 'EosinophilCountAbsolute_Avg_chronic', 'EosinophilCountAbsolute_Avg_preclinical',
               'AbsoluteEosinophilCount_Avg_preclinical', 'AbsoluteEosinophilCount_Avg_acute', 'AbsoluteEosinophilCount_Avg_chronic',
               'LymphocytesPercentage_Avg_preclinical', 'LymphocytesPercentage_Avg_acute', 'LymphocytesPercentage_Avg_chronic',]:
        for b in [
                    'EosinophilsPercentage_Avg_preclinical', 'EosinophilsPercentage_Avg_acute', 'EosinophilsPercentage_Avg_chronic']:
            plot_correlation(shap_values, a, b, 'correlation3')
            plot_correlation(shap_values, b, a, 'correlation3')
            
if __name__ == "__main__":
    SHAPDIR = "shapcorr"
    if not os.path.exists(SHAPDIR):
        os.makedirs(SHAPDIR)
