import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.schemas.user import UserCreate
from app.crud.user import create_user, get_user_by_email
from app.utils.logging import get_logger

# Initialize logger
logger = get_logger("admin_init")

def init_admin():
    logger.info("Starting admin user initialization")
    db = SessionLocal()
    try:
        # Check if admin already exists
        admin = get_user_by_email(db, email="admin@example.com")
        if admin:
            logger.info("Admin user already exists")
            return
        
        # Create admin user
        logger.info("Creating new admin user")
        admin_data = UserCreate(
            email="admin@example.com",
            username="admin",
            password="AdminPassword123!",
            is_active=True,
            role="admin"
        )
        admin = create_user(db=db, obj_in=admin_data)
        logger.info(f"Admin user created successfully: {admin.username}")
    except Exception as e:
        logger.error(f"Error creating admin user: {str(e)}", exc_info=True)
    finally:
        db.close()
        logger.info("Admin initialization completed")

if __name__ == "__main__":
    init_admin()