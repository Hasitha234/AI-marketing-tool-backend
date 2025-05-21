from typing import Optional, Any, Union, Dict, List
from sqlalchemy.orm import Session

from app.core.security import get_password_hash, verify_password
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.utils.logging import get_logger

# Initialize logger
logger = get_logger("user_crud")

def get_user(db: Session, *, user_id: int) -> Optional[User]:
    """ Get a user by ID """
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_email(db: Session, *, email: str) -> Optional[User]:
    """ Get a user by email """
    logger.debug(f"Looking up user with email: {email}")
    user = db.query(User).filter(User.email == email).first()
    if user:
        logger.info(f"Found user with email: {email}")
    else:
        logger.info(f"No user found with email: {email}")
    return user

def get_user_by_username(db: Session, *, username: str) -> Optional[User]:
    """ Get a user by username """
    return db.query(User).filter(User.username == username).first()

def get_users(db: Session, *, skip: int = 0, limit: int = 10) -> List[User]:
    """Get list of users with pagination"""
    logger.debug(f"Fetching users with skip={skip}, limit={limit}")
    users = db.query(User).offset(skip).limit(limit).all()
    logger.info(f"Retrieved {len(users)} users")
    return users

def create_user(db: Session, *, obj_in: UserCreate) -> User:
    """Create new user"""
    logger.info(f"Creating new user with email: {obj_in.email}")
    db_obj = User(
        email=obj_in.email,
        username=obj_in.username,
        hashed_password=get_password_hash(obj_in.password),
        is_active=obj_in.is_active,
        role=obj_in.role
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    logger.info(f"Successfully created user: {db_obj.username}")
    return db_obj

def update_user(db: Session, *, db_obj: User, obj_in: UserUpdate) -> User:
    """Update user"""
    logger.info(f"Updating user: {db_obj.username}")
    update_data = obj_in.dict(exclude_unset=True)
    if "password" in update_data:
        hashed_password = get_password_hash(update_data["password"])
        del update_data["password"]
        update_data["hashed_password"] = hashed_password
    
    for field in update_data:
        setattr(db_obj, field, update_data[field])
    
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    logger.info(f"Successfully updated user: {db_obj.username}")
    return db_obj

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """ Authenticate a user with email and password """
    user = get_user_by_email(db, email=email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def delete_user(db: Session, *, user_id: int) -> Optional[User]:
    """Delete user"""
    logger.info(f"Attempting to delete user with ID: {user_id}")
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        db.delete(user)
        db.commit()
        logger.info(f"Successfully deleted user: {user.username}")
    else:
        logger.warning(f"Attempted to delete non-existent user with ID: {user_id}")
    return user
    
        
    

   