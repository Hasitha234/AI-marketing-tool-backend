from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.base_class import Base


class PricingRecommendation(Base):
    """Store dynamic pricing recommendations"""
    __tablename__ = "pricing_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Route information
    origin = Column(String, nullable=False)
    destination = Column(String, nullable=False)
    route = Column(String, index=True)
    
    # Pricing details
    current_price = Column(Float, nullable=False)
    optimal_price = Column(Float, nullable=False)
    recommended_price = Column(Float)
    competitor_price = Column(Float)
    
    # Revenue metrics
    current_revenue = Column(Float)
    optimal_revenue = Column(Float)
    uplift_percent = Column(Float)
    
    # Demand estimation
    estimated_demand = Column(Float)
    
    # Advanced features (optional)
    distance_miles = Column(Float)
    passengers = Column(Integer)
    market_share_largest = Column(Float)
    market_share_lowest_fare = Column(Float)
    
    # Metadata
    is_applied = Column(Boolean, default=False)
    notes = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="pricing_recommendations")