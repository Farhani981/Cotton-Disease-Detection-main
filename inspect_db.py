import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

db_user = os.getenv('DB_USER', 'root')
db_password = os.getenv('DB_PASSWORD', '')
db_host = os.getenv('DB_HOST', 'localhost')
db_name = os.getenv('DB_NAME', 'cotton_disease_db')

try:
    connection = pymysql.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name
    )
    cursor = connection.cursor()
    
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    print("Existing tables:")
    for table in tables:
        print(table[0])
        cursor.execute(f"DESCRIBE {table[0]}")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col[0]} ({col[1]})")

    cursor.close()
    connection.close()

except Exception as e:
    print(f"Error inspecting DB: {e}")
