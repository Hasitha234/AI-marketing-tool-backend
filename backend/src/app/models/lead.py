from sqlalchemy import Column, TIMESTAMP, ForeignKey, Integer, String, Float, JSON, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base_class import Base

class Lead(Base):
    """ Lead Model for storing lead information"""
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    phone = Column(String, nullable=True, unique=True)
    company = Column(String, nullable=False)

    source = Column(String)
    campaign = Column(String, nullable=False)

    industry = Column(String, nullable=False)
    job_title = Column(String, nullable=False)
    city = Column(String, nullable=True)
    country = Column(String, nullable=True)

    last_activity = Column(TIMESTAMP(timezone=True), nullable=True)
    website_visits = Column(Integer, default=0)
    time_spent_on_website = Column(Integer, default=0.0)
    page_views = Column(Integer, default=0)

    status = Column(String, default="new")
    is_converted = Column(Boolean, default=False)
    converted_at = Column(TIMESTAMP(timezone=True), nullable=True)

    tags = Column(JSONB, nullable=True)
    custom_fields = Column(JSONB, nullable=True)

    notes = Column(Text, nullable=True)

    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=func.now())
    
    #relationships
    scores = relationship("LeadScore", back_populates="lead", cascade="all, delete-orphan")
    activities = relationship("LeadActivity", back_populates="lead", cascade="all, delete-orphan")


class LeadScore(Base):
    """ Lead Score Model for storing lead score information"""
    __tablename__ = "lead_scores"

    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"))

    # Overall score
    score = Column(Float, nullable=False)
    
    #Score Components
    demographic_score = Column(Float, default=0.0)
    behavioral_score = Column(Float, default=0.0)
    firmographic_score = Column(Float, default=0.0)

    #Model version
    model_version = Column(String, nullable=True)

    #Detailed scoring factors
    factors = Column(JSONB)

    #confidence level(0-1)
    confidence = Column(Float, default=0.0)

    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=func.now())
    
    #relationships
    lead = relationship("Lead", back_populates="scores")


class LeadActivity(Base):
    """ Lead Activity Model for storing lead activity information"""
    __tablename__ = "lead_activities"

    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"))

    activity_type = Column(String, index=True)
    description = Column(Text, nullable=True)

    activity_metadata = Column(JSONB, nullable=True)

    occurred_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=func.now())

    lead = relationship("Lead", back_populates="activities")



    
