from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field

# Content schemas simplified for UI
class ContentBase(BaseModel):
    title: str
    body: str
    type: Optional[str] = "general"  # Corresponds to channel in the UI
    status: Optional[str] = "draft"
    
class ContentCreate(ContentBase):
    pass

class ContentUpdate(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    status: Optional[str] = None
    
class Content(ContentBase):
    id: int
    ai_generated: bool
    model_version: Optional[str] = None
    created_by_id: int
    generation_params: Optional[Dict[str, Any]] = None
    target_audience: Optional[Dict[str, Any]] = None
    engagement_metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    model_config = {
        "protected_namespaces": (),
        "from_attributes": True
    }

# List model for pagination
class ContentList(BaseModel):
    items: List[Content]
    total: int
    page: int
    size: int
    pages: int

# UI-specific schemas
class ContentGenerateRequest(BaseModel):
    prompt: str
    tone: str = "Professional"
    industry: Optional[str] = None
    channel: Optional[str] = None

class ContentGenerateResponse(BaseModel):
    content: str
    word_count: int
    character_count: int

class ContentSaveRequest(BaseModel):
    content: str
    prompt: str
    tone: str = "Professional"
    industry: Optional[str] = None
    channel: Optional[str] = None