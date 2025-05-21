from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, Boolean, JSON, Enum, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base_class import Base

class SocialPlatform(enum.Enum):
    """Enum for social media platforms."""
    twitter = "twitter"
    facebook = "facebook"
    instagram = "instagram"
    linkedin = "linkedin"
    pinterest = "pinterest"
    tiktok = "tiktok"

class PostStatus(enum.Enum):
    """Enum for post status."""
    draft = "draft"
    scheduled = "scheduled"
    posted = "posted"
    failed = "failed"

class SocialAccount(Base):
    """Model for storing social media accounts."""
    __tablename__ = "social_accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)  # Display name for the account
    platform = Column(Enum(SocialPlatform))
    
    # Authentication details
    account_id = Column(String)  # Platform's unique account identifier
    username = Column(String)  # Username on the platform
    access_token = Column(String)
    access_token_secret = Column(String, nullable=True)  # For platforms that need it like Twitter
    refresh_token = Column(String, nullable=True)
    token_expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Connection status
    is_active = Column(Boolean, default=True)
    last_checked = Column(DateTime(timezone=True), nullable=True)
    
    # Creator info
    created_by_id = Column(Integer, ForeignKey("users.id"))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    created_by = relationship("User", back_populates="social_accounts")
    posts = relationship("SocialPost", back_populates="account", cascade="all, delete-orphan")
    campaigns = relationship("SocialCampaign", secondary="campaign_accounts", back_populates="accounts")

class SocialPost(Base):
    """Model for storing social media posts."""
    __tablename__ = "social_posts"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("social_accounts.id"))
    campaign_id = Column(Integer, ForeignKey("social_campaigns.id"), nullable=True)
    
    # Post content
    content = Column(Text)
    media_urls = Column(JSONB, nullable=True)  # URLs to images or videos
    link = Column(String, nullable=True)  # URL to be shared
    
    # Scheduling info
    scheduled_time = Column(DateTime(timezone=True))
    status = Column(Enum(PostStatus), default=PostStatus.draft)
    posted_time = Column(DateTime(timezone=True), nullable=True)
    
    # Platform-specific data
    platform_post_id = Column(String, nullable=True)  # ID of the post on the platform once posted
    platform_data = Column(JSONB, nullable=True)  # Platform-specific configuration
    
    # Performance metrics
    likes = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    reach = Column(Integer, default=0)
    engagement_rate = Column(Float, default=0.0)
    
    # Status tracking
    error_message = Column(String, nullable=True)
    last_status_check = Column(DateTime(timezone=True), nullable=True)
    
    # Creator info
    created_by_id = Column(Integer, ForeignKey("users.id"))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    account = relationship("SocialAccount", back_populates="posts")
    campaign = relationship("SocialCampaign", back_populates="posts")
    created_by = relationship("User")

class SocialCampaign(Base):
    """Model for storing social media campaigns."""
    __tablename__ = "social_campaigns"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    
    # Campaign details
    start_date = Column(DateTime(timezone=True))
    end_date = Column(DateTime(timezone=True))
    target_audience = Column(String, nullable=True)
    budget = Column(Float, nullable=True)
    
    # Status
    status = Column(String, default="active")  # active, paused, completed
    
    # Creator info
    created_by_id = Column(Integer, ForeignKey("users.id"))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    created_by = relationship("User")
    posts = relationship("SocialPost", back_populates="campaign")
    accounts = relationship("SocialAccount", secondary="campaign_accounts", back_populates="campaigns")

# Association table for many-to-many relationship between campaigns and accounts
class CampaignAccount(Base):
    """Association model for campaigns and social accounts."""
    __tablename__ = "campaign_accounts"
    
    campaign_id = Column(Integer, ForeignKey("social_campaigns.id"), primary_key=True)
    account_id = Column(Integer, ForeignKey("social_accounts.id"), primary_key=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())