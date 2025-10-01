from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.pricing import PricingRecommendation
from app.schemas.pricing import PricingRecommendationCreate, PricingRecommendationUpdate


def create_pricing_recommendation(
    db: Session,
    *,
    obj_in: PricingRecommendationCreate
) -> PricingRecommendation:
    """Create a new pricing recommendation"""
    db_obj = PricingRecommendation(
        user_id=obj_in.user_id,
        origin=obj_in.origin,
        destination=obj_in.destination,
        route=f"{obj_in.origin} → {obj_in.destination}",
        current_price=obj_in.current_price,
        optimal_price=obj_in.optimal_price,
        recommended_price=obj_in.recommended_price,
        competitor_price=obj_in.competitor_price,
        current_revenue=obj_in.current_revenue,
        optimal_revenue=obj_in.optimal_revenue,
        uplift_percent=obj_in.uplift_percent,
        estimated_demand=obj_in.estimated_demand,
        distance_miles=obj_in.distance_miles,
        passengers=obj_in.passengers,
        market_share_largest=obj_in.market_share_largest,
        market_share_lowest_fare=obj_in.market_share_lowest_fare,
        notes=obj_in.notes,
        is_applied=obj_in.is_applied
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def get_pricing_recommendation(db: Session, recommendation_id: int) -> Optional[PricingRecommendation]:
    """Get a pricing recommendation by ID"""
    return db.query(PricingRecommendation).filter(PricingRecommendation.id == recommendation_id).first()


def get_pricing_recommendations(
    db: Session,
    *,
    skip: int = 0,
    limit: int = 100,
    user_id: Optional[int] = None
) -> List[PricingRecommendation]:
    """Get all pricing recommendations with optional user filter"""
    query = db.query(PricingRecommendation)
    if user_id:
        query = query.filter(PricingRecommendation.user_id == user_id)
    return query.order_by(desc(PricingRecommendation.created_at)).offset(skip).limit(limit).all()


def get_pricing_recommendations_by_route(
    db: Session,
    *,
    origin: str,
    destination: str,
    skip: int = 0,
    limit: int = 10
) -> List[PricingRecommendation]:
    """Get pricing recommendations for a specific route"""
    return db.query(PricingRecommendation).filter(
        PricingRecommendation.origin == origin,
        PricingRecommendation.destination == destination
    ).order_by(desc(PricingRecommendation.created_at)).offset(skip).limit(limit).all()


def update_pricing_recommendation(
    db: Session,
    *,
    db_obj: PricingRecommendation,
    obj_in: PricingRecommendationUpdate
) -> PricingRecommendation:
    """Update a pricing recommendation"""
    update_data = obj_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def delete_pricing_recommendation(db: Session, *, recommendation_id: int) -> Optional[PricingRecommendation]:
    """Delete a pricing recommendation"""
    obj = db.query(PricingRecommendation).filter(PricingRecommendation.id == recommendation_id).first()
    if obj:
        db.delete(obj)
        db.commit()
    return obj


def get_pricing_stats(db: Session, user_id: Optional[int] = None) -> dict:
    """Get pricing recommendation statistics"""
    query = db.query(PricingRecommendation)
    if user_id:
        query = query.filter(PricingRecommendation.user_id == user_id)
    
    total = query.count()
    applied = query.filter(PricingRecommendation.is_applied == True).count()
    
    # Calculate average uplift
    avg_uplift = db.query(PricingRecommendation.uplift_percent).filter(
        PricingRecommendation.uplift_percent.isnot(None)
    ).all()
    
    avg_uplift_value = sum([r[0] for r in avg_uplift]) / len(avg_uplift) if avg_uplift else 0
    
    return {
        "total_recommendations": total,
        "applied_recommendations": applied,
        "average_uplift_percent": round(avg_uplift_value, 2),
        "pending_recommendations": total - applied
    }