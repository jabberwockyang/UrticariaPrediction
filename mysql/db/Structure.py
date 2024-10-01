import sqlite3
DB_PATH = 'mysql/db/urticaria.db'
# 连接数据库
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 获取所有表名
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# 打印每个表的结构和外键关系
for table in tables:
    table_name = table[0]
    print(f"# {table_name}:")
    print(f"## Structure of {table_name}:")
    cursor.execute(f"PRAGMA table_info('{table_name}');")
    columns = cursor.fetchall()
    for column in columns:
        print(column)
    print(f"## Foreign keys of {table_name}:")
    cursor.execute(f"PRAGMA foreign_key_list('{table_name}');")
    foreign_keys = cursor.fetchall()
    for foreign_key in foreign_keys:
        print(foreign_key)
    print("\n")
    # 打印index
    print(f"## Indexes of {table_name}:")
    cursor.execute(f"PRAGMA index_list('{table_name}');")
    indexes = cursor.fetchall()
    for index in indexes:
        print(index)
    print("\n")
    

# 关闭数据库连接
cursor.close()
conn.close()
