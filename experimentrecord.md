
```bash
conda activate nni
cd /root/UrticariaPrediction
```

# weights and topn strategy


## nni1 
data: feature engineering with min max avg of all records

feature derivation and selection: none

code: old code with some error regarding imputation

#### nni1 params generation and importance
```bash
# nni and determine top variables with importance 
nnictl create --config config_nni1.yml --port 8081 # ZpoUyrIC  nnictl resume ZpoUyrIC --port 8081
for expid in nni1_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done
for expid in nni1_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 25; done

```
#### grouping experiments and importance summary based on nni1
```bash
# Loop through YAML files and run train_grouping.py for each experiment
for yml in grouping_nni1_*.yaml; do
    for expid in nni1_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

# Run importance.py for each grouping experiment folder
for expid in nni1_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done
```
## nni2

data: feature engineering with min max avg of all records

feature derivation and selection: dv based on ZpoUyrIC_default_top7_fromnni1 feature_importance_summary

code: old code with some error regarding imputation

### nni2 params generation and importance
```bash 
# nni with derived variables nni2_explog derived variables was from top 15 variables ranked by importance in nni1
nnictl create --config config_dv.yml --port 8081 # 3eQbjfcG  nnictl resume 3eQbjfcG --port 8081
for expid in nni2_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done
for expid in nni2_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 25; done
```
### grouping experiments and importance summary based on nni2
```bash
# Loop through YAML files and run train_grouping.py for each experiment
for yml in grouping_nni2_*.yaml; do
    for expid in nni2_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

# Run importance.py for each grouping experiment folder
for expid in nni2_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done
```

## nni3
data: feature engineering with min max avg of all records

feature derivation and selection: dv based on ZpoUyrIC_default_top7_fromnni1 feature_importance_summary topn is provided in searchspace for nni3

code: old code with some error regarding imputation

### nni3 params generation and importance

```bash
# nni with derived variables and topn nni3_explog derived variables was from top 15 variables ranked by importance in nni2 and topn is in searchspace for nni3 
nnictl create --config config_dv_topn.yml --port 8081 # 
for expid in nni3_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done
for expid in nni3_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 25; done
```

### grouping experiments and importance summary based on nni3

```bash
# Loop through YAML files and run train_grouping.py for each experiment
for yml in grouping_nni3_*.yaml; do
    for expid in nni3_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

# Run importance.py for each grouping experiment folder
for expid in nni3_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done
```


## nni4
data: feature engineering with min max avg of all records

feature derivation and selection:  topn is provided in searchspace for nni4

code: old code with some error regarding imputation

### nni4 params generation and importance   
```bash
# nni with topn nni4_explog topn is in searchspace for nni4 no derived variables
nnictl create --config config_topn.yml --port 7860
for expid in nni4_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done
for expid in nni4_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 25; done
```

### grouping experiments and importance summary based on nni4
```bash
# Loop through YAML files and run train_grouping.py for each experiment
for yml in grouping_nni4_*.yaml; do
    for expid in nni4_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

# Run importance.py for each grouping experiment folder
for expid in nni4_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done
```


some notification regarding different experiment id in nni3 and nni4 

have different topn values in searchspace

SyguB7Fb  n60M47dW topn 参数 50-200 

0T2GXABC  lcMYyo0V topn 参数 0.1-1

```bash
# to run importance for all gr_explog
for grfolder in gr_explog/*; do
    python3 importance.py --grdir "$grfolder"
done
```
## conclusion
summary it shows no benefit of feature derivation and subsequent topn selection

[/root/UrticariaPrediction/VariablesImportance/summary.md](/root/UrticariaPrediction/VariablesImportance/summary.md)

use nni1 results for further analysis


# try different grouping strategies

```bash
for yml in grouping_nni1_*_gr2.yaml; do
    for expid in nni1_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

for expid in nni1_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*_gr2; do
        python3 importance.py --grdir "$grfolder"
    done
done

```
## conclusion
results show the previous grouping strategy is better

# try boruta

```bash
python3 runboruta.py
```

## conclusion

during boruta discover problem with imputation about to run nni1 again and grouping and importance not done

BORUTA shows good performance in feature selection   

# workflow based on boruta selected features derived variables and selection

## feature engingeering with minmxavg and acute chronic avg
done in sqlquery and saved in csv

dataforxgboost_ac.csv

## nni5
data: feature engineering with min max avg of all records and avg of acute and chronic records
feature derivation and selection: none
code: updated code

### nni5 params generation and importance
```bash
# nni5 GET best params for boruta
nnictl create --config config_nni5.yml --port 8081 # ZpoUyrIC  nnictl resume ZpoUyrIC --port 8081
# run importance for nni5
for expid in nni5_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done

```
experiment id BD3oGFia
newsqlquery  experiment id 8Y9XvkQq

### boruta selection with no derived features
```bash
best_expid=8Y9XvkQq
best_db_path=nni5_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_ac.csv --best_db_path $best_db_path --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml 
```
experiment id 09647097
newsqlquery experiment id d2d8b927-0e09-44f5-890b-3016f1ea5d68


### boruta selection with derived features from selection using id 09647097
```bash 
best_expid=BD3oGFia
best_db_path=nni5_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_ac.csv --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml --best_db_path $best_db_path --features_for_derivation boruta_explog/09647097-60b1-4c47-bc04-47eb678f73ea/confirmed_vars.txt
```

### retry boruta selection with derived features with larger number of iterations
```bash
best_expid=BD3oGFia
best_db_path=nni5_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_ac.csv --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml --best_db_path $best_db_path --features_for_derivation boruta_explog/09647097-60b1-4c47-bc04-47eb678f73ea/confirmed_vars.txt --max_iteration 100
```

## conclusion
boruta select less features with derived features

no derived features were selected by boruta 

```bash
2024-09-13 14:06:23.031 | INFO     | __main__:load_feature_list_from_boruta_file:490 - Found 42 common variables in all the lists
boruta selection with no derived features
['PlateletCount', 'MeanCorpuscularHemoglobinConcentration', 'AntiJo1', 'AbsoluteEosinophilCount', 'AntiScl70', 'CReactiveProtein', 'BasophilsPercentage', 'Histone', 'Plateletcrit', 'LymphocytesPercentage', 'TotalThyroxine', 'MeanPlateletVolume', 'EosinophilsPercentage', 'MeanCorpuscularHemoglobin', 'AntiDoubleStrandedDNA', 'DogDander', 'AntiSSA', 'WhiteBloodCellCount', 'ProliferatingCellNuclearAntigen', 'SMRNP', 'EggWhite', 'RedCellDistributionWidth', 'TotalTriiodothyronine', 'AbsoluteNeutrophilCount', 'ThyroidStimulatingHormone', 'Ragweed', 'MeanCorpuscularVolume', 'NeutrophilsPercentage', 'RedCellDistributionWidthCV', 'Hemoglobin', 'MonocytesPercentage', 'AbsoluteLymphocyteCount', 'AbsoluteMonocyteCount', 'AntiM2', 'AbsoluteBasophilCount', 'FreeThyroxine', 'AntiSM', 'EosinophilCountAbsolute', 'Cockroach', 'ImmunoglobulinE', 'Ro52', 'PlateletDistributionWidth']

2024-09-13 14:06:23.036 | INFO     | __main__:load_feature_list_from_boruta_file:490 - Found 26 common variables in all the lists
boruta selection with derived features
['PlateletCount', 'MeanCorpuscularHemoglobinConcentration', 'AbsoluteEosinophilCount', 'AntiScl70', 'BasophilsPercentage', 'LymphocytesPercentage', 'TotalThyroxine', 'MeanPlateletVolume', 'EosinophilsPercentage', 'MeanCorpuscularHemoglobin', 'DogDander', 'ProliferatingCellNuclearAntigen', 'TotalTriiodothyronine', 'AbsoluteNeutrophilCount', 'Ragweed', 'MeanCorpuscularVolume', 'NeutrophilsPercentage', 'Hemoglobin', 'MonocytesPercentage', 'AbsoluteLymphocyteCount', 'AbsoluteMonocyteCount', 'AntiM2', 'AbsoluteBasophilCount', 'EosinophilCountAbsolute', 'ImmunoglobulinE', 'PlateletDistributionWidth']

2024-09-13 14:06:23.039 | INFO     | __main__:load_feature_list_from_boruta_file:490 - Found 19 common variables in all the lists
boruta selection with derived features try2
['PlateletCount', 'MeanCorpuscularHemoglobinConcentration', 'AbsoluteEosinophilCount', 'AntiScl70', 'BasophilsPercentage', 'LymphocytesPercentage', 'MeanPlateletVolume', 'EosinophilsPercentage', 'MeanCorpuscularHemoglobin', 'DogDander', 'ProliferatingCellNuclearAntigen', 'AbsoluteNeutrophilCount', 'Ragweed', 'MeanCorpuscularVolume', 'MonocytesPercentage', 'AbsoluteLymphocyteCount', 'AntiM2', 'AbsoluteBasophilCount', 'PlateletDistributionWidth']

2024-09-13 15:21:49.931 | INFO     | __main__:load_feature_list_from_boruta_file:490 - Found 30 common variables in all the lists
boruta selection with derived features try with maxiteration 100
['AbsoluteLymphocyteCount', 'ImmunoglobulinE', 'MeanPlateletVolume', 'AbsoluteNeutrophilCount', 'NeutrophilsPercentage', 'Hemoglobin', 'ThyroidStimulatingHormone', 'WhiteBloodCellCount', 'CReactiveProtein', 'MeanCorpuscularHemoglobinConcentration', 'AbsoluteBasophilCount', 'DogDander', 'Ragweed', 'Cockroach', 'EosinophilCountAbsolute', 'EosinophilsPercentage', 'LymphocytesPercentage', 'AntiM2', 'MonocytesPercentage', 'PlateletCount', 'Plateletcrit', 'PlateletDistributionWidth', 'MeanCorpuscularVolume', 'AntiScl70', 'MeanCorpuscularHemoglobin', 'AbsoluteMonocyteCount', 'ProliferatingCellNuclearAntigen', 'TotalTriiodothyronine', 'AbsoluteEosinophilCount', 'BasophilsPercentage']
```

# workflow based on different feature engineering and boruta selection 

feature engineering

nni5: feature engineering with min max avg of acute and chronic records

nni6: feature engineering with min max avg of acute records only

nni7: feature engineering with min max avg of all records

nni8: feature engineering with min max avg of acute records and preclinical records

## nni5

data: feature engineering with min max avg of all records and avg of acute and chronic records

feature derivation and selection: none

code: updated code

### reuse the results of nni5 BD3oGFia
 
### reuse boruta results from nni5 09647097


### group training with nni5 results and with nni5 results + bselected features
```bash
for yml in grouping_nni5*.yaml; do
    for expid in nni5_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done
```


### Run importance.py for each grouping experiment folder
```bash
for expid in nni5_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done
```
|nni|nni+boruta|
|---|---|
|![nni](VariablesImportance/gr_explog/8Y9XvkQq_default_top25_gr1/max_roc_auc.png)|![nni+boruta](VariablesImportance/gr_explog/8Y9XvkQq_default_top25_gr1_selection_d2d8b927/max_roc_auc.png)|


## nni6

data: feature engineering with min max avg of acute records only

feature derivation and selection: none

code: updated code

### nni6 params generation and importance
```bash
nnictl create --config config_nni6.yml --port 8081 # ZpoUyrIC  nnictl resume ZpoUyrIC --port 8081
# run importance for nni6
for expid in nni6_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done
```
experiment id 4jmrxVsX

results of acute data prediction was bad roc was about 0.6

### run boruta for nni6
```bash
best_expid=4jmrxVsX
best_db_path=nni6_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_a.csv --best_db_path $best_db_path --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml
```
experiment id bc631828


### group training with nni6 results and with nni6 results + bselected features
```bash
for yml in grouping_nni6*.yaml; do 
    for expid in nni6_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done
```

### Run importance.py for each grouping experiment folder
```bash
for expid in nni6_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done
```

|nni|nni+boruta|
|---|---|
|![nni](VariablesImportance/gr_explog/4jmrxVsX_default_top25_gr1/max_roc_auc.png)|![nni+boruta](VariablesImportance/gr_explog/4jmrxVsX_default_top25_gr1_selection_bc631828/max_roc_auc.png)|
## nni7

data: feature engineering with min max avg of all records
feature derivation and selection: none
code: updated code

### nni7 params generation and importance
```bash
# nni7 use original min max avg data like nni1 but with new code
nnictl create --config config_nni7.yml --port 8081 # ZpoUyrIC  nnictl resume ZpoUyrIC --port 8081
# run importance for nni7
for expid in nni7_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done

```
experiment id Itr01zQY

### boruta selection with nni7 results
```bash
best_expid=Itr01zQY
best_db_path=nni7_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost.csv --best_db_path $best_db_path --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml
```

experiment id 57d5c9dd


### group training with nni7 results and with nni7 results + bselected features
```bash
for yml in grouping_nni7*.yaml; do
    for expid in nni7_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done
```

### Run importance.py for each grouping experiment folder
```bash
for expid in nni7_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done
```
|nni|nni+boruta|
|---|---|
|![nni](VariablesImportance/gr_explog/Itr01zQY_default_top25_gr1/max_roc_auc.png)|![nni+boruta](VariablesImportance/gr_explog/Itr01zQY_default_top25_gr1_selection_57d5c9dd/max_roc_auc.png)|

## nni8

data: feature engineering with min max avg of acute records and preclinical records

feature derivation and selection: none

code: updated code

### nni8 params generation and importance
```bash
# nni8 use acute data and preclinical data
nnictl create --config config_nni8.yml --port 8081 # ZpoUyrIC  nnictl resume ZpoUyrIC --port 8081
for expid in nni8_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done
```
experiment id AhUO5lFk

### boruta selection with nni8 results
```bash
# run boruta for nni8
best_expid=AhUO5lFk
best_db_path=nni8_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_ap.csv --best_db_path $best_db_path --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml
```

experiment id 6a7a1e15

### group training with nni8 results and with nni8 results + bselected features
```bash
for yml in grouping_nni8*.yaml; do
    for expid in nni8_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done
```

### Run importance.py for each grouping experiment folder
```bash
for expid in nni8_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done
```
|nni|nni+boruta|
|---|---|
|![nni](VariablesImportance/gr_explog/AhUO5lFk_default_top25_gr1/max_roc_auc.png)|![nni+boruta](VariablesImportance/gr_explog/AhUO5lFk_default_top25_gr1_selection_6a7a1e15/max_roc_auc.png)|





## add data save code in train grouping code rerun all grouping yaml

```bash

for yml in grouping_nni*.yaml; do 
    python3 train_grouping.py --config "$yml"
done

```

## change preprocessor setting so the dropna threshold of commonbloodtest is 50% 

### rerun all nni and check available data amount

#### nni5
```bash
nnictl create --config config_nni5.yml --port 8081 
nnictl resume CFBmkWQ1 --port 8081 
```
experiment id: CFBmkWQ1

boruta
```bash
best_expid=CFBmkWQ1
best_db_path=nni5_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_ac.csv --best_db_path $best_db_path --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml 
```
Experiment name: e5a86e18-dca6-49d1-a9c5-4955afa26f84



##### nni6
```bash
nnictl create --config config_nni6.yml --port 7860
```
The experiment id is ia27HVx6
```bash
best_expid=ia27HVx6
best_db_path=nni6_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_a.csv --best_db_path $best_db_path --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml
```
 Experiment name: aa54898a-5bf4-419f-8422-4c4e91de67d2


#### nni7
```bash
nnictl create --config config_nni7.yml --port 8000 
```
The experiment id is kiGBxqL6
```bash
best_expid=kiGBxqL6
best_db_path=nni7_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost.csv --best_db_path $best_db_path --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml
```
Experiment name: 28ca5bf9-6014-4676-92b0-fc14ccff2dc8

#### nni8 
```bash
nnictl create --config config_nni8.yml --port 6666 
```
The experiment id is agAUczMO
```bash
best_expid=agAUczMO
best_db_path=nni8_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_ap.csv --best_db_path $best_db_path --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml
```
Experiment name: 5ba609a3-33ca-4809-869a-004d915e1d74

### rerun all grouping importance
```bash
for yml in grouping_nni5*.yaml; do 
    python3 train_grouping.py --config "$yml"
done
```

```bash
for expid in nni5_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done
```

```bash
for yml in grouping_nni6*.yaml; do 
    python3 train_grouping.py --config "$yml"
done
```
```bash
for expid in nni6_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done
```

```bash
for yml in grouping_nni7*.yaml; do 
    python3 train_grouping.py --config "$yml"
done
```
```bash
for expid in nni7_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done
```

```bash
for yml in grouping_nni8*.yaml; do 
    python3 train_grouping.py --config "$yml"
done
```
```bash
for expid in nni8_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done
```


## 240923 reconstruct code with svm and rf available in nni searchspace

reconstruct code with svm and rf available in nni searchspace
find error in reversing yscaling in evaluation within xgboost training
nni5 results shows less negative point with 42 as binary threshold add 100 and 365 as binary threshold and take max of them to report the final result
### rerun all nni


#### nni7 based on avg min max of all records
```bash 
cp search_space_xgboost.json search_space.json
nnictl create --config config_nni7.yml --port 7860
nnictl resume Fyepnu5G --port 7860
```
The experiment id is Fyepnu5G
![42roc](nni7_explog/Fyepnu5G/result/428/42_roc.png)
![100roc](nni7_explog/Fyepnu5G/result/428/100_roc.png)
![365roc](nni7_explog/Fyepnu5G/result/428/365_roc.png)

```bash
cp search_space_rf.json search_space.json
nnictl create --config config_nni7.yml --port 8081
nnictl resume 5BnrHsbC --port 8081
```
The experiment id is 5BnrHsbC
![42roc](nni7_explog/5BnrHsbC/result/99/42_roc.png)
![100roc](nni7_explog/5BnrHsbC/result/99/100_roc.png)
![365roc](nni7_explog/5BnrHsbC/result/99/365_roc.png)

```bash
cp search_space_svm.json search_space.json
nnictl create --config config_nni7.yml --port 8081

```
The experiment id is 5j2hasGx

#### nni5 based on avg min max of acute and chronic records

```bash
cp search_space_xgboost.json search_space.json
nnictl create --config config_nni5.yml --port 8081
```
```bash
for expid in nni5_explog/*; do python3 importance.py --nnidir $expid --metric roc_auc --minimize False --number_of_trials 1; done
```
240923
```bash
cp search_space_xgboost.json search_space.json
cp output/dataforxgboost_ac_2024-09-14.csv output/dataforxgboost_ac.csv
nnictl create --config config_nni5.yml --port 8081
```
The experiment id is a6gA1Bip finished
best id 3512
![42roc](nni5_explog/a6gA1Bip/result/3512/42_roc.png)
![100roc](nni5_explog/a6gA1Bip/result/3512/100_roc.png)
![365roc](nni5_explog/a6gA1Bip/result/3512/365_roc.png)

240924
```bash
cp search_space_xgboost.json search_space.json
cp output/dataforxgboost_ac_2024-09-24.csv output/dataforxgboost_ac.csv
nnictl create --config config_nni5.yml --port 8081
```
The experiment id is e3VIHk1m finished
best id 507
![42roc](nni5_explog/e3VIHk1m/result/507/42_roc.png)
![100roc](nni5_explog/e3VIHk1m/result/507/100_roc.png)
![365roc](nni5_explog/e3VIHk1m/result/507/365_roc.png)



#### nni6 based on avg min max of acute records
##### xgboost
```bash
cp search_space_xgboost.json search_space.json
nnictl create --config config_nni6.yml --port 7860
```
```bash
for expid in nni6_explog/*; do python3 importance.py --nnidir $expid --metric roc_auc --minimize False --number_of_trials 1; done
```
The experiment id is 3bXuOPBQ finished 
best id 401
![42roc](nni6_explog/3bXuOPBQ/result/401/42_roc.png)
![100roc](nni6_explog/3bXuOPBQ/result/401/100_roc.png)
![365roc](nni6_explog/3bXuOPBQ/result/401/365_roc.png)


new a: The experiment id is QNH5u2ST
best id 32
![42roc](nni6_explog/QNH5u2ST/result/32/42_roc.png)
![100roc](nni6_explog/QNH5u2ST/result/32/100_roc.png)
![365roc](nni6_explog/QNH5u2ST/result/32/365_roc.png)


#### nni8 based on avg min max of acute and preclinical records
##### xgboost
```bash
cp search_space_xgboost.json search_space.json
nnictl create --config config_nni8.yml --port 8081
```
```bash
for expid in nni8_explog/*; do python3 importance.py --nnidir $expid --metric roc_auc --minimize False --number_of_trials 1; done
```

The experiment id is gpd0DfVh  repeat htrx8Aep(deleted) xiLbhs1V (quote randomstate)
best id 898
![42roc](nni8_explog/gpd0DfVh/result/898/42_roc.png)
![100roc](nni8_explog/gpd0DfVh/result/898/100_roc.png)
![365roc](nni8_explog/gpd0DfVh/result/898/365_roc.png)

new ap:The experiment id is Bo4S0FNf(deleted)  AOopRwcC(deleted) ZfLhMPY3 (quote randomstate)


240924 
find possible bug in tpe final report the 3 args
```plain text

Reports final result to NNI.

metric should either be a float, or a dict that metric['default'] is a float.

If metric is a dict, metric['default'] will be used by tuner, and other items can be visualized with web portal.

Typically metric is the final accuracy or loss.
```
nothing

go on

found bug in disecting the data for xgboost rerun nni5 nni6 and nni8

240925
why getting strange results after change to the new df
the problem is in random state 


## 240925 add some more features in codes
1. 5 fold cross validation
2. more examination data total visit > 2

### nni7
#### xgboost 
```bash
cd ~/UrticariaPrediction
cp search_space_xgboost.json search_space.json
cp mysql/output-20240925/dataforxgboost_2024-09-25.csv output/dataforxgboost.csv
nnictl create --config config_nni7.yml --port 7860
```
The experiment id is fcNqxBOy
best id 39 0.78


#### rf
```bash
cd ~/UrticariaPrediction
cp search_space_rf.json search_space.json
cp mysql/output-20240925/dataforxgboost_2024-09-25.csv output/dataforxgboost.csv
nnictl create --config config_nni7.yml --port 8081
```

#### svm
```bash
cd ~/UrticariaPrediction
cp search_space_svm.json search_space.json
cp mysql/output-20240925/dataforxgboost_2024-09-25.csv output/dataforxgboost.csv
nnictl create --config config_nni7.yml --port 8081
```

### nni5
#### xgboost
```bash
cd ~/UrticariaPrediction
cp search_space_xgboost.json search_space.json
cp mysql/output-20240925/dataforxgboost_ac_2024-09-25.csv output/dataforxgboost_ac.csv
nnictl create --config config_nni5.yml --port 8081
```
The experiment id is dnNrA0gl 0.66
maximize roc 



### nni6
#### xgboost
```bash
cd ~/UrticariaPrediction
cp search_space_xgboost.json search_space.json
cp mysql/output-20240925/dataforxgboost_a_2024-09-25.csv output/dataforxgboost_a.csv
nnictl create --config config_nni6.yml --port 8081
```

### nni8
```bash
cd ~/UrticariaPrediction
cp search_space_xgboost.json search_space.json
cp mysql/output-20240925/dataforxgboost_ap_2024-09-25.csv output/dataforxgboost_ap.csv
nnictl create --config config_nni8.yml --port 8081
```
8TiIWMPx


## 240927
### new sqlquery to eliminate multi-collinearity and compilance filteration 

### new nni strategies to involve preprocessing params such as dropna percentage of row and columns

### new code structure and evaluation protocol + Shap explanation

sql: mysql/SQLquery/sql/dataforxgboost_timeseries_2024-09-27.sql


每个超参数配置
- 过 preprocessor 
    - 存一个 preprocesseddata.csv
    - jsonl 存 data shape 存 missing rate
- 分 30% 外部验证 70% 开发集
    - 存一个 外部验证.csv 开发集.csv
- train [开发集内部 0.7 train 0.3 val] 明确最佳框架和超参数
    - jsonl 存超参数 
    - jsonl 存内部验证结果 [auc specificty sensitivity f1] * [42 100 365]

load each框架 top100 最佳超参数配置
    - jsonl 存开发集五倍交叉验证结果 [auc specificty sensitivity f1] * [42 100 365] * [5个模型]
    - jsonl 存 y ypredict
plot 
    - 开发集五倍交叉验证作图 【roc  loss ry/ryp】
    - kfoldint 结果获得each框架 最佳超参数配置

load best框架 最佳超参数配置 
    - get variable importance ranking
    - run boruta [GENERATE TOPN VARIABLE FILES] # TODO IN RUN BORUTA

load best框架 最佳超参数配置  topn变量
    - jsonl 存开发集五倍交叉验证结果 [auc specificty sensitivity f1] * [42 100 365] * [5个模型]
    - jsonl 存 y ypredict
    - 并再次超参数调优

load 最佳框架最佳参数 外部验证
    -  内部数据 训练 内部外部数据验证

shap验证
- 全局解释
    - 全年龄段数据训练全局解释
    - 不同年龄段数据训练观察全局解释变化
    - 对应相关性分析



# 240928

## nni
xgboost-timeseries  nni9_explog/beA3o82D
```bash
cd ~/UrticariaPrediction
cp search_space_xgboost.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
nnictl create --config config_nni9.yml --port 8081
```

xgboost-normal nni10_explog/dTBCXYGr
```bash
cd ~/UrticariaPrediction
cp search_space_xgboost.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240929/dataforxgboost_2024-09-29.csv output/dataforxgboost.csv
nnictl create --config config_nni10.yml --port 8081
```

rf-timeseries nni9_explog/1aTxj7zc
```bash
cd ~/UrticariaPrediction
cp search_space_rf.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
nnictl create --config config_nni9.yml --port 8081
```

rf-normal nni10_explog/XE0MhN5r
```bash
cd ~/UrticariaPrediction
cp search_space_rf.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240929/dataforxgboost_2024-09-29.csv output/dataforxgboost.csv
nnictl create --config config_nni10.yml --port 8081
```

svm-timeseries nni9_explog/NKgRQfcV
```bash
cd ~/UrticariaPrediction
cp search_space_svm.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
nnictl create --config config_nni9.yml --port 8081
```

svm-normal nni10_explog/7mJ4VYe5
```bash
cd ~/UrticariaPrediction
cp search_space_svm.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240929/dataforxgboost_2024-09-29.csv output/dataforxgboost.csv
nnictl create --config config_nni10.yml --port 8081
```

adaboost-timeseries nni9_explog/25m9QoAi
```bash
cd ~/UrticariaPrediction
cp search_space_adaboost.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
nnictl create --config config_nni9.yml --port 8081
```

adaboost-normal nni10_explog/FAbyiLmG
```bash
cd ~/UrticariaPrediction
cp search_space_adaboost.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240929/dataforxgboost_2024-09-29.csv output/dataforxgboost.csv
nnictl create --config config_nni10.yml --port 8081
```

gbm-timeseries nni9_explog/lKesaFNR
```bash
cd ~/UrticariaPrediction
cp search_space_gbm.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
nnictl create --config config_nni9.yml --port 8081
```

gbm-normal nni10_explog/YR1DQb9A
```bash
cd ~/UrticariaPrediction
cp search_space_gbm.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240929/dataforxgboost_2024-09-29.csv output/dataforxgboost.csv
nnictl create --config config_nni10.yml --port 8081
```

<!-- lightgbm-timeseries nni9_explog/
```bash
cd ~/UrticariaPrediction
cp search_space_lgb.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
nnictl create --config config_nni9.yml --port 8081
```

lightgbm-normal nni10_explog/
```bash
cd ~/UrticariaPrediction
cp search_space_lgb.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240929/dataforxgboost_2024-09-29.csv output/dataforxgboost.csv
nnictl create --config config_nni10.yml --port 8081
``` -->


## kfoldint
xgboost-timeseries nni9_explog/beA3o82D
```bash
cd ~/UrticariaPrediction
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
python3 train_kfoldint.py --config kfoldint_timeseries.yaml --expfolder nni9_explog --expid beA3o82D
```

xgboost-normal nni10_explog/dTBCXYGr
```bash
cd ~/UrticariaPrediction
cp /root/UrticariaPrediction/mysql/output-20240929/dataforxgboost_2024-09-29.csv output/dataforxgboost.csv
python3 train_kfoldint.py --config kfoldint_normal.yaml --expfolder nni10_explog --expid dTBCXYGr
```

rf-timeseries nni9_explog/1aTxj7zc
```bash
cd ~/UrticariaPrediction
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
python3 train_kfoldint.py --config kfoldint_timeseries.yaml --expfolder nni9_explog --expid 1aTxj7zc
```

rf-normal nni10_explog/XE0MhN5r
```bash
cd ~/UrticariaPrediction
cp /root/UrticariaPrediction/mysql/output-20240929/dataforxgboost_2024-09-29.csv output/dataforxgboost.csv
python3 train_kfoldint.py --config kfoldint_normal.yaml --expfolder nni10_explog --expid XE0MhN5r
```

svm-timeseries nni9_explog/NKgRQfcV
```bash
cd ~/UrticariaPrediction
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
python3 train_kfoldint.py --config kfoldint_timeseries.yaml --expfolder nni9_explog --expid NKgRQfcV
```

svm-normal nni10_explog/7mJ4VYe5
```bash
cd ~/UrticariaPrediction
cp /root/UrticariaPrediction/mysql/output-20240929/dataforxgboost_2024-09-29.csv output/dataforxgboost.csv
python3 train_kfoldint.py --config kfoldint_normal.yaml --expfolder nni10_explog --expid 7mJ4VYe5
```

adaboost-timeseries nni9_explog/25m9QoAi
```bash
cd ~/UrticariaPrediction
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
python3 train_kfoldint.py --config kfoldint_timeseries.yaml --expfolder nni9_explog --expid 25m9QoAi
```

ada-normal nni10_explog/FAbyiLmG
```bash
cd ~/UrticariaPrediction
cp /root/UrticariaPrediction/mysql/output-20240929/dataforxgboost_2024-09-29.csv output/dataforxgboost.csv
python3 train_kfoldint.py --config kfoldint_normal.yaml --expfolder nni10_explog --expid FAbyiLmG
```

gbm-timeseries nni9_explog/lKesaFNR
```bash
cd ~/UrticariaPrediction
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
python3 train_kfoldint.py --config kfoldint_timeseries.yaml --expfolder nni9_explog --expid lKesaFNR
```

gbm-normal nni10_explog/YR1DQb9A
```bash
cd ~/UrticariaPrediction
cp /root/UrticariaPrediction/mysql/output-20240929/dataforxgboost_2024-09-29.csv output/dataforxgboost.csv
python3 train_kfoldint.py --config kfoldint_normal.yaml --expfolder nni10_explog --expid YR1DQb9A
```

## boruta

xgboost-timeseries nni9_explog/beA3o82D
```bash
cd ~/UrticariaPrediction
python3 runboruta.py --filepath output/dataforxgboost_timeseries.csv --best_db_path nni9_explog/beA3o82D/db/nni.sqlite --best_sequence_id 1112 --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml
```

```bash
cd ~/UrticariaPrediction
python3 plot_boruta.py --log_dir boruta_explog --experiment_name e2f721e9
```

## topn 




## external validation
xgboost-timeseries nni9_explog/beA3o82D

```bash
cd ~/UrticariaPrediction
python3 train_ext_validation.py --config extval_timeseries.yaml --expid beA3o82D --sequenceid 1112 --featurelistfolder boruta_explog/e2f721e9
```