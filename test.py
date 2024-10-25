import sqlite3

# 数据库路径
DB_PATH = 'mysql/db/urticaria.db'

# 连接数据库
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 查询PatientVisits中VisitDate的最小值和最大值
query = "SELECT MIN(VisitDate), MAX(VisitDate) FROM PatientVisits"
cursor.execute(query)

# 获取结果
result = cursor.fetchone()

# 打印最小值和最大值
print(f"VisitDate 最小值: {result[0]}")
print(f"VisitDate 最大值: {result[1]}")

# 关闭连接
cursor.close()
conn.close()
