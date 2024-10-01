
import json
from datetime import datetime

# insert AVG(CASE WHEN npe.ExaminationItem = 'Item1' THEN npe.Result ELSE NULL END) AS Item1_Avg,
sql_template = '''
SELECT 
    CASE WHEN p.Gender = '男' THEN 1 ELSE 0 END as Gender,
    p.FirstVisitAge,
    p.VisitDuration,
    p.CIndU-- INSERT_LOCATION
FROM 
    Patients p
LEFT JOIN 
    NewPatientExaminations npe ON p.PatientID = npe.PatientID
WHERE 
    p.TotalVisits > 2
GROUP BY 
    p.PatientID;

'''

def modify_sql(sql_template, itemlistlist, classifications):

    insert_location = "-- INSERT_LOCATION"
    # variance, standard deviation, median, mode, range, max, min
    statements = ''
    # for classification in itemlistlist.keys():
    for classification in classifications:
        for item_zh,item_english, item_ID in itemlistlist[classification]:
            statement = f""",
    MAX(CASE WHEN npe.ExaminationItemID = '{item_ID}' AND npe.ExaminationDate >= p.FirstVisitDate AND npe.ExaminationDate <= DATE(p.FirstVisitDate, '+42 days') THEN npe.Result ELSE NULL END) AS {item_english}_Max_acute,
    MIN(CASE WHEN npe.ExaminationItemID = '{item_ID}' AND npe.ExaminationDate >= p.FirstVisitDate AND npe.ExaminationDate <= DATE(p.FirstVisitDate, '+42 days') THEN npe.Result ELSE NULL END) AS {item_english}_Min_acute,
    AVG(CASE WHEN npe.ExaminationItemID = '{item_ID}' AND npe.ExaminationDate >= p.FirstVisitDate AND npe.ExaminationDate <= DATE(p.FirstVisitDate, '+42 days') THEN npe.Result ELSE NULL END) AS {item_english}_Avg_acute,
    MAX(CASE WHEN npe.ExaminationItemID = '{item_ID}' AND npe.ExaminationDate < p.FirstVisitDate THEN npe.Result ELSE NULL END) AS {item_english}_Max_preclinical,
    MIN(CASE WHEN npe.ExaminationItemID = '{item_ID}' AND npe.ExaminationDate < p.FirstVisitDate THEN npe.Result ELSE NULL END) AS {item_english}_Min_preclinical,
    AVG(CASE WHEN npe.ExaminationItemID = '{item_ID}' AND npe.ExaminationDate < p.FirstVisitDate THEN npe.Result ELSE NULL END) AS {item_english}_Avg_preclinical"""
            statements += statement

    new_sql = sql_template.replace(insert_location, statements)
    return new_sql

# Load the ExaminationItemClass from JSON
with open('ExaminationItemClass_ID.json', 'r') as json_file:
    ExaminationItemClass = json.load(json_file)
classifications = ExaminationItemClass.keys() # ['血细胞指标',"甲状腺功能","凝血功能","尿液检查","自身免疫标注物","免疫指标"]
# Iterate over each classification and each examination item within it
modified_sql = modify_sql(sql_template, ExaminationItemClass, classifications)

# Write the modified SQL to a file for the current examination item
current_date = datetime.now().strftime('%Y-%m-%d')

with open(f'sql/dataforxgboost_ap_{current_date}.sql', 'w') as f:
    f.write(modified_sql)