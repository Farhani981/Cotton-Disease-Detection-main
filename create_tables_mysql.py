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
    
    # Create User table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(150) UNIQUE NOT NULL,
            email VARCHAR(150) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            upload_attempts INT DEFAULT 0,
            is_admin INT DEFAULT 0,
            role VARCHAR(50) DEFAULT 'user'
        )
    """)
    
    # Create Report table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS report (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            filename VARCHAR(100) NOT NULL,
            prediction VARCHAR(100) NOT NULL,
            confidence FLOAT,
            timestamp DATETIME
        )
    """)
    
    # Create SignupHistory table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signup_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(150),
            timestamp DATETIME
        )
    """)
    
    # Create LoginHistory table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            username VARCHAR(150),
            success INT,
            timestamp DATETIME
        )
    """)
    
    # Create SessionHistory table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            username VARCHAR(150),
            action VARCHAR(50),
            timestamp DATETIME
        )
    """)
    
    connection.commit()
    print("Database tables created successfully!")
    
    cursor.close()
    connection.close()
    
except Exception as e:
    import traceback
    print(f"Error creating tables: {e}")
    print(traceback.format_exc())
