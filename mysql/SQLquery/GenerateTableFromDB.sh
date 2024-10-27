cd /mnt/yangyijun/UrticariaPrediction/mysql/SQLquery

python3 dataxgboost.py
python3 dataxgboost_ac.py
python3 dataxgboost_a.py
python3 dataxgboost_ap.py
python3 dataxgboost_timeseries.py

cd /mnt/yangyijun/UrticariaPrediction/mysql
mkdir output-20240925
for sql in SQLquery/sql/dataforxgboost*_2024-09-25.sql 
do
    sqlite3 -header -csv db/urticaria.db < $sql > output-20240925/$(basename $sql .sql).csv
done

cd /mnt/yangyijun/UrticariaPrediction/mysql
mkdir output-20241027
for sql in SQLquery/sql/dataforxgboost*_2024-10-27.sql 
do
    sqlite3 -header -csv db/urticaria.db < $sql > output-20241027/$(basename $sql .sql).csv
done

cd /mnt/yangyijun/UrticariaPrediction/mysql
mkdir output-20240929
sqlite3 -header -csv db/urticaria.db < SQLquery/sql/dataforxgboost_2024-09-29.sql > output-20240929/dataforxgboost_2024-09-29.csv