import os
from app import app, db, User

def promote_to_admin(username):
    with app.app_context():
        user = User.query.filter_by(username=username).first()
        if user:
            user.is_admin = 1
            user.role = 'admin'
            db.session.commit()
            print(f"\n[SUCCESS] User '{username}' has been promoted to Administrator.")
            print("Now log in with this account and you will see the Admin Dashboard link.")
        else:
            print(f"\n[ERROR] User '{username}' not found in the database.")

if __name__ == "__main__":
    print("--- Cotton Disease Detection - Admin Manager ---")
    target_user = input("Enter the username you want to make Admin: ").strip()
    promote_to_admin(target_user)
