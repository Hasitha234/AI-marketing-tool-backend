from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd

from app.api import dependencies
from app.schemas.pricing import (
    SimplePricingRequest,
    AdvancedPricingRequest,
    PricingResponse,
    PricingRecommendation,
    PricingRecommendationCreate,
    PricingRecommendationUpdate
)
from app.models.user import User
from app.services.pricing import pricing_service
from app.crud import pricing as pricing_crud

router = APIRouter()


@router.post("/simple", response_model=PricingResponse)
async def get_simple_pricing(
    *,
    db: Session = Depends(dependencies.get_db),
    request: SimplePricingRequest,
    current_user: User = Depends(dependencies.get_current_user)
):
    """
    Get pricing recommendation using simplified elasticity model.
    
    Perfect for quick pricing decisions without detailed route data.
    No ML model required.
    """
    result = pricing_service.get_simple_pricing(
        origin=request.origin,
        destination=request.destination,
        current_price=request.price
    )
    
    # Save to database
    pricing_rec = PricingRecommendationCreate(
        user_id=current_user.id,
        origin=request.origin,
        destination=request.destination,
        current_price=request.price,
        optimal_price=result["optimal_price"],
        competitor_price=result["competitor_price"],
        estimated_demand=result["estimated_demand"],
        current_revenue=result["current_revenue"],
        optimal_revenue=result["optimal_revenue"],
        uplift_percent=result["uplift_percent"]
    )
    
    pricing_crud.create_pricing_recommendation(db=db, obj_in=pricing_rec)
    
    return PricingResponse(**result)


@router.post("/advanced", response_model=PricingResponse)
async def get_advanced_pricing(
    *,
    db: Session = Depends(dependencies.get_db),
    request: AdvancedPricingRequest,
    current_user: User = Depends(dependencies.get_current_user)
):
    """
    Get advanced pricing recommendation using ML models.
    
    Requires detailed route data including historical metrics and market characteristics.
    ML model must be available.
    """
    if not pricing_service.is_model_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML pricing model not available. Use /simple endpoint instead."
        )
    
    if not request.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No route data provided"
        )
    
    # Convert to DataFrame
    route_data = [item.model_dump() for item in request.data]
    route_df = pd.DataFrame(route_data)
    
    result = pricing_service.get_advanced_pricing(route_df)
    
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to process route data"
        )
    
    # Save to database
    first_route = request.data[0]
    pricing_rec = PricingRecommendationCreate(
        user_id=current_user.id,
        origin=first_route.city1,
        destination=first_route.city2,
        current_price=first_route.fare,
        optimal_price=result["optimal_price"],
        recommended_price=result.get("recommended_price"),
        competitor_price=result["competitor_price"],
        estimated_demand=result["estimated_demand"],
        current_revenue=result["current_revenue"],
        optimal_revenue=result["optimal_revenue"],
        uplift_percent=result["uplift_percent"],
        distance_miles=first_route.nsmiles,
        passengers=first_route.passengers,
        market_share_largest=first_route.large_ms,
        market_share_lowest_fare=first_route.lf_ms
    )
    
    pricing_crud.create_pricing_recommendation(db=db, obj_in=pricing_rec)
    
    return PricingResponse(**result)


@router.get("/recommendations", response_model=List[PricingRecommendation])
async def get_pricing_recommendations(
    *,
    db: Session = Depends(dependencies.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(dependencies.get_current_user)
):
    """
    Get all pricing recommendations for the current user.
    """
    recommendations = pricing_crud.get_pricing_recommendations(
        db=db,
        skip=skip,
        limit=limit,
        user_id=current_user.id
    )
    return recommendations


@router.get("/recommendations/{recommendation_id}", response_model=PricingRecommendation)
async def get_pricing_recommendation(
    *,
    db: Session = Depends(dependencies.get_db),
    recommendation_id: int,
    current_user: User = Depends(dependencies.get_current_user)
):
    """
    Get a specific pricing recommendation by ID.
    """
    recommendation = pricing_crud.get_pricing_recommendation(db=db, recommendation_id=recommendation_id)
    
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pricing recommendation not found"
        )
    
    # Check ownership
    if recommendation.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this recommendation"
        )
    
    return recommendation


@router.patch("/recommendations/{recommendation_id}", response_model=PricingRecommendation)
async def update_pricing_recommendation(
    *,
    db: Session = Depends(dependencies.get_db),
    recommendation_id: int,
    recommendation_update: PricingRecommendationUpdate,
    current_user: User = Depends(dependencies.get_current_user)
):
    """
    Update a pricing recommendation (e.g., mark as applied).
    """
    recommendation = pricing_crud.get_pricing_recommendation(db=db, recommendation_id=recommendation_id)
    
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pricing recommendation not found"
        )
    
    # Check ownership
    if recommendation.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this recommendation"
        )
    
    updated_recommendation = pricing_crud.update_pricing_recommendation(
        db=db,
        db_obj=recommendation,
        obj_in=recommendation_update
    )
    
    return updated_recommendation


@router.delete("/recommendations/{recommendation_id}")
async def delete_pricing_recommendation(
    *,
    db: Session = Depends(dependencies.get_db),
    recommendation_id: int,
    current_user: User = Depends(dependencies.get_current_user)
):
    """
    Delete a pricing recommendation.
    """
    recommendation = pricing_crud.get_pricing_recommendation(db=db, recommendation_id=recommendation_id)
    
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pricing recommendation not found"
        )
    
    # Check ownership
    if recommendation.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this recommendation"
        )
    
    pricing_crud.delete_pricing_recommendation(db=db, recommendation_id=recommendation_id)
    
    return {"message": "Pricing recommendation deleted successfully"}


@router.get("/stats")
async def get_pricing_stats(
    *,
    db: Session = Depends(dependencies.get_db),
    current_user: User = Depends(dependencies.get_current_user)
):
    """
    Get pricing recommendation statistics for the current user.
    """
    stats = pricing_crud.get_pricing_stats(db=db, user_id=current_user.id)
    return stats


@router.get("/model/info")
async def get_model_info(
    current_user: User = Depends(dependencies.get_current_user)
):
    """
    Get information about the loaded ML pricing model.
    """
    return pricing_service.get_model_info()
