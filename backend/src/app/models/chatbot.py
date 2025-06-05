from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base

class ChatbotSession(Base):
    """Model for chatbot conversation sessions"""
    __tablename__ = "chatbot_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    started_at = Column(DateTime, nullable=False, server_default=func.now())
    ended_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships - use string references
    user = relationship("User", back_populates="chatbot_sessions")
    messages = relationship("ChatbotMessage", back_populates="session", cascade="all, delete-orphan")


class ChatbotMessage(Base):
    """Model for individual chatbot messages"""
    __tablename__ = "chatbot_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chatbot_sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    message = Column(Text, nullable=False)
    is_user = Column(Boolean, nullable=False)
    intent = Column(String(255), nullable=True)
    confidence = Column(Float, nullable=True)
    timestamp = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships - use string references
    session = relationship("ChatbotSession", back_populates="messages")
    user = relationship("User", back_populates="chatbot_messages")


class FAQ(Base):
    """Model for frequently asked questions"""
    __tablename__ = "faqs"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    keywords = Column(Text, nullable=True)  # JSON string of keywords
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class ChatbotAnalytics(Base):
    """Model for chatbot analytics and metrics"""
    __tablename__ = "chatbot_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False)
    total_sessions = Column(Integer, default=0)
    total_messages = Column(Integer, default=0)
    unique_users = Column(Integer, default=0)
    avg_session_duration = Column(Float, default=0.0)  # in minutes
    avg_messages_per_session = Column(Float, default=0.0)
    top_intents = Column(Text, nullable=True)  # JSON string
    conversion_rate = Column(Float, default=0.0)  # leads generated / total sessions
    created_at = Column(DateTime, nullable=False, server_default=func.now())