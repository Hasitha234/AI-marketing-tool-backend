
from sqlalchemy.orm import Session
from app.db import base  # This imports all models
from app.db.session import engine
from app.models import *  # Import all models to register them

def init_db(db: Session) -> None:
    """Initialize database with all models"""
    # Import all models to make sure they are registered with SQLAlchemy
    from app.models.user import User
    from app.models.content import Content
    from app.models.chatbot import ChatbotSession, ChatbotMessage, FAQ, ChatbotAnalytics
    
    # Create all tables
    base.Base.metadata.create_all(bind=engine)

# app/db/base.py
# Import all the models, so that Base has them before being
# imported by Alembic
from app.db.base_class import Base  # noqa
from app.models.user import User  # noqa
from app.models.content import Content  # noqa  
from app.models.chatbot import ChatbotSession, ChatbotMessage, FAQ, ChatbotAnalytics  # noqa