import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

db_user = os.getenv('DB_USER', 'root')
db_password = os.getenv('DB_PASSWORD', '')
db_host = os.getenv('DB_HOST', 'localhost')
db_name = os.getenv('DB_NAME', 'cotton_disease_db')

# Connection string
db_uri = f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}'

def fix_schema():
    engine = create_engine(db_uri)
    with engine.connect() as conn:
        print("Increasing password column length...")
        try:
            # MySQL syntax
            conn.execute(text("ALTER TABLE user MODIFY COLUMN password VARCHAR(255) NOT NULL;"))
            conn.commit()
            print("Successfully altered table 'user'. Password column is now VARCHAR(255).")
        except Exception as e:
            print(f"Error altering table: {e}")

if __name__ == "__main__":
    fix_schema()
