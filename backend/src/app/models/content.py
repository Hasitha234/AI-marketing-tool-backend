from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base_class import Base

class Content(Base):
    """Model for storing generated content."""
    __tablename__ = "contents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)  # This will store the prompt or a title
    body = Column(Text)  # The generated content
    type = Column(String, index=True)  # Corresponds to channel in dataset
    status = Column(String, default="draft")
    
    # Generation metadata
    ai_generated = Column(Boolean, default=True)
    generation_params = Column(JSON, nullable=True)  # Will store prompt, tone, etc.
    model_version = Column(String, nullable=True)
    content_metadata = Column(JSON, nullable=True)  
    
    # Creator info
    created_by_id = Column(Integer, ForeignKey("users.id"))  # Updated to match User table name
    
    # Optional fields
    target_audience = Column(JSON, nullable=True)
    engagement_metrics = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    created_by = relationship("User", back_populates="contents")