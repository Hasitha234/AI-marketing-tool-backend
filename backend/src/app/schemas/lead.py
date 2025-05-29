from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, EmailStr, Field, validator

# Base Lead Schema
class LeadBase(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    source: Optional[str] = None
    campaign: Optional[str] = None
    industry: Optional[str] = None
    job_title: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    website_visits: Optional[int] = 0
    time_spent_on_website: Optional[float] = 0.0
    page_views: Optional[int] = 0
    status: Optional[str] = "new"
    tags: Optional[List[str]] = []
    custom_fields: Optional[Dict[str, Any]] = {}
    notes: Optional[str] = None

# Create Lead Schema
class LeadCreate(LeadBase):
    name: str
    email: EmailStr
    source: str

# Update Lead Schema
class LeadUpdate(LeadBase):
    pass

# Lead Activity Schema
class LeadActivityBase(BaseModel):
    activity_type: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    occurred_at: Optional[datetime] = None

class LeadActivityCreate(LeadActivityBase):
    lead_id: int

class LeadActivity(LeadActivityBase):
    id: int
    lead_id: int
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = Field({}, alias="activity_metadata")
    
    class Config:
        from_attributes = True

# Lead Score Schema
class LeadScoreBase(BaseModel):
    score: float = Field(..., ge=0, le=100)
    demographic_score: Optional[float] = Field(0.0, ge=0, le=100)
    behavioral_score: Optional[float] = Field(0.0, ge=0, le=100)
    firmographic_score: Optional[float] = Field(0.0, ge=0, le=100)
    factors: Dict[str, Any] = {}
    confidence: float = Field(0.0, ge=0, le=1.0)
    model_version: Optional[str] = None

    model_config = {
        "protected_namespaces": ()
    }

class LeadScoreCreate(LeadScoreBase):
    lead_id: int

class LeadScoreUpdate(BaseModel):
    score: Optional[float] = Field(None, ge=0, le=100)
    demographic_score: Optional[float] = Field(None, ge=0, le=100)
    behavioral_score: Optional[float] = Field(None, ge=0, le=100)
    firmographic_score: Optional[float] = Field(None, ge=0, le=100)
    factors: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = Field(None, ge=0, le=1.0)
    model_version: Optional[str] = None

    model_config = {
        "protected_namespaces": ()
    }

class LeadScore(LeadScoreBase):
    id: int
    lead_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Complete Lead Schema with Scores and Activities
class Lead(LeadBase):
    id: int
    is_converted: bool
    converted_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    scores: List[LeadScore] = []
    activities: List[LeadActivity] = []
    
    class Config:
        from_attributes = True

# Lead List Schema (for pagination)
class LeadList(BaseModel):
    items: List[Lead]
    total: int
    page: int
    size: int
    pages: int



    
    
    