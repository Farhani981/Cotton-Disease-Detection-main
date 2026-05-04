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
    
    # Drop existing tables
    print("Dropping existing tables...")
    cursor.execute("DROP TABLE IF EXISTS report")
    cursor.execute("DROP TABLE IF EXISTS session_history")
    cursor.execute("DROP TABLE IF EXISTS login_history")
    cursor.execute("DROP TABLE IF EXISTS signup_history")
    cursor.execute("DROP TABLE IF EXISTS user")
    
    print("Tables dropped successfully!")
    
    cursor.close()
    connection.close()
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    print(traceback.format_exc())
