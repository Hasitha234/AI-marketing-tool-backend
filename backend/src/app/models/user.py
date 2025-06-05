from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base_class import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    role = Column(String, default="user")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships - use string references to avoid circular imports
    social_campaigns = relationship("SocialCampaign", back_populates="created_by", lazy="dynamic")
    social_accounts = relationship("SocialAccount", back_populates="created_by", lazy="dynamic")
    chatbot_sessions = relationship("ChatbotSession", back_populates="user", lazy="dynamic")
    chatbot_messages = relationship("ChatbotMessage", back_populates="user", lazy="dynamic")
    
    # Fix: Use string reference and remove foreign_keys parameter
    contents = relationship("Content", back_populates="created_by", lazy="dynamic")