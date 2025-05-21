from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl

# Enum schemas
class SocialPlatformEnum(str, Enum):
    twitter = "twitter"
    facebook = "facebook"
    instagram = "instagram"
    linkedin = "linkedin"
    pinterest = "pinterest"
    tiktok = "tiktok"

class PostStatusEnum(str, Enum):
    draft = "draft"
    scheduled = "scheduled"
    posted = "posted"
    failed = "failed"

# Social Account schemas
class SocialAccountBase(BaseModel):
    name: str
    platform: SocialPlatformEnum
    username: str
    account_id: Optional[str] = None
    is_active: Optional[bool] = True

class SocialAccountCreate(SocialAccountBase):
    access_token: str
    access_token_secret: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None

class SocialAccountUpdate(BaseModel):
    name: Optional[str] = None
    username: Optional[str] = None
    access_token: Optional[str] = None
    access_token_secret: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    is_active: Optional[bool] = None

class SocialAccount(SocialAccountBase):
    id: int
    created_by_id: int
    last_checked: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Social Post schemas
class SocialPostBase(BaseModel):
    account_id: int
    content: str
    scheduled_time: datetime
    media_urls: Optional[List[str]] = None
    link: Optional[str] = None
    platform_data: Optional[Dict[str, Any]] = None

class SocialPostCreate(SocialPostBase):
    campaign_id: Optional[int] = None

class SocialPostUpdate(BaseModel):
    content: Optional[str] = None
    scheduled_time: Optional[datetime] = None
    media_urls: Optional[List[str]] = None
    link: Optional[str] = None
    status: Optional[PostStatusEnum] = None
    platform_data: Optional[Dict[str, Any]] = None

class SocialPost(SocialPostBase):
    id: int
    campaign_id: Optional[int] = None
    status: PostStatusEnum
    posted_time: Optional[datetime] = None
    platform_post_id: Optional[str] = None
    likes: Optional[int] = 0
    shares: Optional[int] = 0
    comments: Optional[int] = 0
    clicks: Optional[int] = 0
    reach: Optional[int] = 0
    engagement_rate: Optional[float] = 0.0
    error_message: Optional[str] = None
    created_by_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Social Campaign schemas
class SocialCampaignBase(BaseModel):
    name: str
    description: Optional[str] = None
    start_date: datetime
    end_date: datetime
    target_audience: Optional[str] = None
    budget: Optional[float] = None
    status: Optional[str] = "active"

class SocialCampaignCreate(SocialCampaignBase):
    account_ids: List[int] = []

class SocialCampaignUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    target_audience: Optional[str] = None
    budget: Optional[float] = None
    status: Optional[str] = None
    account_ids: Optional[List[int]] = None

class SocialCampaign(SocialCampaignBase):
    id: int
    created_by_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class SocialCampaignDetail(SocialCampaign):
    accounts: List[SocialAccount] = []
    posts: List[SocialPost] = []
    
    class Config:
        from_attributes = True

# List models for pagination
class SocialAccountList(BaseModel):
    items: List[SocialAccount]
    total: int
    page: int
    size: int
    pages: int

class SocialPostList(BaseModel):
    items: List[SocialPost]
    total: int
    page: int
    size: int
    pages: int

class SocialCampaignList(BaseModel):
    items: List[SocialCampaign]
    total: int
    page: int
    size: int
    pages: int