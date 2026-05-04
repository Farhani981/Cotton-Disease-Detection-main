import os
from app import app, db, User
from werkzeug.security import generate_password_hash, check_password_hash

def debug_auth():
    with app.app_context():
        print("--- Debugging Authentication ---")
        
        # 1. Test Hash Generation
        password = "testpassword123"
        hashed = generate_password_hash(password)
        print(f"Generated Hash: {hashed}")
        print(f"Hash Length: {len(hashed)}")
        
        # 2. Test Verification
        is_valid = check_password_hash(hashed, password)
        print(f"Verification (Immediate): {is_valid}")
        
        # 3. Inspect Database Users
        users = User.query.all()
        print(f"\nFound {len(users)} users in DB:")
        for user in users:
            print(f"ID: {user.id}, Username: {user.username}, Hash Len: {len(user.password)}")
            # print(f"Hash: {user.password}") # Don't print full hash for privacy/security logs if possible, but for debugging I might need to if length looks okay.
            
            # Try to verify against 'testpassword123' just in case
            if check_password_hash(user.password, password):
                 print(f"  -> Matches 'testpassword123'")
            else:
                 print(f"  -> Does NOT match 'testpassword123'")

        # 4. Create a test user
        test_username = "debug_user_99"
        
        # Clean up if exists
        existing = User.query.filter_by(username=test_username).first()
        if existing:
            db.session.delete(existing)
            db.session.commit()
            print(f"\nDeleted existing {test_username}")
            
        new_user = User(username=test_username, email="debug99@test.com")
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        print(f"Created {test_username} with password '{password}'")
        
        # 5. Fetch and Verify
        retrieved_user = User.query.filter_by(username=test_username).first()
        if retrieved_user:
            print(f"Retrieved Hash Len: {len(retrieved_user.password)}")
            print(f"Stored Hash: {retrieved_user.password}")
            success = check_password_hash(retrieved_user.password, password)
            print(f"Login Verification: {'SUCCESS' if success else 'FAILED'}")
            
            if not success:
               print("Possible Truncation detected!" if len(retrieved_user.password) == 150 else "Unknown cause")

if __name__ == "__main__":
    debug_auth()
