# Experiment Record
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


load 最佳框架最佳参数 外部验证 topn变量
    -  内部数据 训练 内部外部数据验证

shap验证
- 全局解释
    - 全年龄段数据训练全局解释
    - 不同年龄段数据训练观察全局解释变化
    - 对应相关性分析



# 240928

## nni
xgboost-timeseries  nni9_explog/beA3o82D
xgboost-timeseries  nni9_explog/Y4Vfd6Zx new search space
```bash
cd ~/UrticariaPrediction
cp search_space_xgboost.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
nnictl create --config config_nni9.yml --port 8081
```
xgboost-timeseries  nni9_explog/HDQAuzN8 new search space2
```bash
cd ~/UrticariaPrediction
cp search_space_xgboost2.json search_space.json
cp /root/UrticariaPrediction/mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
nnictl create --config config_nni9.yml --port 8081
```
xgboost-timeseries  nni9_explog/zLCPym1l new search space2 with data less than  800 failed
```bash
cd ~/UrticariaPrediction
cp search_space_xgboost2.json search_space.json
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

xgboost-timeseries nni9_explog/CWQJ9nlD with new search space
```bash
cd ~/UrticariaPrediction
cp mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
python3 train_kfoldint.py --config kfoldint_timeseries.yaml --expfolder nni9_explog --expid CWQJ9nlD
```

xgboost-timeseries nni9_explog/HDQAuzN8 with new but restricted search space
```bash
cd ~/UrticariaPrediction
cp mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
python3 train_kfoldint.py --config kfoldint_timeseries.yaml --expfolder nni9_explog --expid HDQAuzN8
```
xgboost-timeseries nni9_explog/zLCPym1l with new but restricted search space and fail when data < 800
```bash
cd ~/UrticariaPrediction
cp mysql/output-20240927/dataforxgboost_timeseries_2024-09-27.csv output/dataforxgboost_timeseries.csv
python3 train_kfoldint.py --config kfoldint_timeseries.yaml --expfolder nni9_explog --expid zLCPym1l
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

plot kfoldint results
```bash
cd ~/UrticariaPrediction
python3 plot_kfoldint.py
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

## external validation and topn exploration
xgboost-timeseries nni9_explog/beA3o82D

```bash
cd ~/UrticariaPrediction
python3 train_ext_validation.py --config extval_timeseries.yaml --expid beA3o82D --sequenceid 1112 --featurelistfolder boruta_explog/e2f721e9
```
CWQJ9nlD_1052_1_all
```bash
cd ~/UrticariaPrediction
python3 train_ext_validation.py --config extval_timeseries.yaml --expid CWQJ9nlD --sequenceid 1052 --featurelistfolder boruta_explog/e2f721e9
```

HDQAuzN8_4866_1_all
```bash
cd ~/UrticariaPrediction
python3 train_ext_validation.py --config extval_timeseries.yaml --expid HDQAuzN8 --sequenceid 4866 --featurelistfolder boruta_explog/e2f721e9
```

zLCPym1l_374_1_all
```bash
cd ~/UrticariaPrediction
python3 train_ext_validation.py --config extval_timeseries.yaml --expid zLCPym1l --sequenceid 374 --featurelistfolder boruta_explog/e2f721e9
```


```bash
cd ~/UrticariaPrediction
python3 train_ext_validation.py --config extval_timeseries.yaml --expid 1aTxj7zc --sequenceid 266 --featurelistfolder boruta_explog/e2f721e9
```

```bash
cd ~/UrticariaPrediction
python3 train_ext_validation.py --config extval_timeseries.yaml --expid lKesaFNR --sequenceid 31 --featurelistfolder boruta_explog/e2f721e9
```


```bash
cd ~/UrticariaPrediction
python3 plot_ext_topn.py
```

## plot shap and kde

```bash
cd ~/UrticariaPrediction
python3 plot_shap.py
```


multicolinearity:
https://datascience.stackexchange.com/questions/12554/does-xgboost-handle-multicollinearity-by-itself
https://github.com/shap/shap/issues/1120
https://github.com/shap/shap/issues/288