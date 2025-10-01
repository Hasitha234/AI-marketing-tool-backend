from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime


# --- Request Schemas ---

class SimplePricingRequest(BaseModel):
    """Simplified pricing request for basic analysis"""
    origin: str = Field(..., description="Origin city", example="Los Angeles, CA")
    destination: str = Field(..., description="Destination city", example="New York City, NY")
    price: float = Field(..., ge=0, description="Current price", example=320.0)


class RouteData(BaseModel):
    """Detailed route data for advanced pricing analysis"""
    city1: str = Field(..., description="Origin city", example="Los Angeles, CA (Metropolitan Area)")
    city2: str = Field(..., description="Destination city", example="New York City, NY (Metropolitan Area)")
    fare: float = Field(..., ge=0, description="Current fare price", example=320.0)
    nsmiles: Optional[float] = Field(2451, ge=0, description="Distance in miles", example=2451)
    passengers: Optional[int] = Field(150, ge=0, description="Number of passengers", example=150)
    large_ms: Optional[float] = Field(0.8, ge=0, le=1, description="Market share of largest carrier", example=0.8)
    lf_ms: Optional[float] = Field(0.9, ge=0, le=1, description="Market share of lowest fare carrier", example=0.9)
    month_sin: Optional[float] = Field(0.5, ge=-1, le=1, description="Month sine encoding", example=0.5)
    month_cos: Optional[float] = Field(0.87, ge=-1, le=1, description="Month cosine encoding", example=0.87)
    quarter_sin: Optional[float] = Field(0.3, ge=-1, le=1, description="Quarter sine encoding", example=0.3)
    quarter_cos: Optional[float] = Field(0.95, ge=-1, le=1, description="Quarter cosine encoding", example=0.95)
    fare_lag1y: Optional[float] = Field(300, ge=0, description="Fare from previous year", example=300)
    passengers_roll4: Optional[float] = Field(145, ge=0, description="4-quarter rolling average passengers", example=145)


class AdvancedPricingRequest(BaseModel):
    """Advanced pricing request with detailed route data"""
    data: List[RouteData] = Field(..., description="List of route data for analysis")


# --- Response Schemas ---

class PricingResponse(BaseModel):
    """Pricing recommendation response"""
    route: str = Field(..., description="Route description", example="Los Angeles, CA → New York City, NY")
    avg_price: float = Field(..., description="Average current price", example=320.0)
    competitor_price: float = Field(..., description="Estimated competitor price", example=335.0)
    estimated_demand: float = Field(..., description="Estimated demand at current price", example=185.0)
    current_revenue: float = Field(..., description="Current revenue estimate", example=59200.0)
    optimal_price: float = Field(..., description="Recommended optimal price", example=327.5)
    optimal_revenue: float = Field(..., description="Projected revenue at optimal price", example=60487.5)
    uplift_percent: float = Field(..., description="Revenue uplift percentage", example=2.17)
    recommended_price: Optional[float] = Field(None, description="Business-rule adjusted price", example=330.0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "route": "Los Angeles, CA → New York City, NY",
                "avg_price": 320.0,
                "competitor_price": 335.0,
                "estimated_demand": 185.0,
                "current_revenue": 59200.0,
                "optimal_price": 327.5,
                "optimal_revenue": 60487.5,
                "uplift_percent": 2.17,
                "recommended_price": 330.0
            }
        }
    )


# --- Database Schemas ---

class PricingRecommendationBase(BaseModel):
    """Base schema for pricing recommendations"""
    origin: str
    destination: str
    current_price: float
    optimal_price: float
    recommended_price: Optional[float] = None
    competitor_price: Optional[float] = None
    current_revenue: Optional[float] = None
    optimal_revenue: Optional[float] = None
    uplift_percent: Optional[float] = None
    estimated_demand: Optional[float] = None
    distance_miles: Optional[float] = None
    passengers: Optional[int] = None
    market_share_largest: Optional[float] = None
    market_share_lowest_fare: Optional[float] = None
    notes: Optional[str] = None
    is_applied: bool = False


class PricingRecommendationCreate(PricingRecommendationBase):
    """Schema for creating pricing recommendation"""
    user_id: Optional[int] = None


class PricingRecommendationUpdate(BaseModel):
    """Schema for updating pricing recommendation"""
    is_applied: Optional[bool] = None
    notes: Optional[str] = None


class PricingRecommendationInDB(PricingRecommendationBase):
    """Schema for pricing recommendation from database"""
    id: int
    user_id: Optional[int] = None
    route: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PricingRecommendation(PricingRecommendationInDB):
    """Complete pricing recommendation schema"""
    pass
