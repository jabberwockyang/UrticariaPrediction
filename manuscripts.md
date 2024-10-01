
# introduction

what is urticaria
classification of urticaria
meaning of predicting urticaria duration
what factors affect the disease duration of chronic urticaria
aiming to build a ml model for predicting the disease duration of urticaria

introduction of ml predicting disease duration 
# methods and materials
## patients and data acquisition


## statistical methods
## database building

mysql relationshp database was build
patients visit examination 
dbml

### feature engineering and feature selection
4 types of feature engineering methods are chosed for training model
1. total: min max avg of a patients' total records
2. a: min max avg of a patients' records within 42 days of disease onset
3. ac: min max avg of a patients' records before 42 days of disease onset and after 42 days of disease onset
4. ap: min max avg of a patients' records before disease onset and and within 42 days of disease onset
the procedures are done in sql selection language


feature selection is done by boruta selection

### model
rf, xgboost, svm were used for training


## hyperparameter optimization and evaluation of different models performance
dataset is divided in a 7:3 ratio the 7 parts for training and validation the 3 parts for testing model performance.
Internal 5 fold cross-validation was employed to discern the most suitable hyperparameters for each distinct model, individually applied to each model for enhanced precision
auc for different duration cuttoff (42, 100, 365) is measured for selecting best model for different cuttoff purposes



## analysis of importance of variables 

importance (weight, ) of xgboost are used for analyzing variables importance

since age seems to play major affect 
model training on seperating age group data are further adopted to see various variables importance across different age group, kde plot of value distribution across different outcome group is used for validation of the indication of variable importance results for clinical use.



# results
## clinical characteristics

provides a comparison of the baseline characteristic between the training set and external testing set data
https://link.springer.com/article/10.1186/s12967-024-04896-3/tables/1


## Multiple machine learning model performance on total records
xgboost best



## model performance based on Multiple feature engineering strategies 
feature engineering utilizing the Time Series feature of db
boruta selection is seperatedly done in each dataset
