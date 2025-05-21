from typing import List, Optional, Dict, Any, Union, Tuple
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timedelta

from app.models.social_media import SocialAccount, SocialPost, SocialCampaign, CampaignAccount, PostStatus
from app.schemas.social import SocialAccountCreate, SocialAccountUpdate, SocialPostCreate, SocialPostUpdate, SocialCampaignCreate, SocialCampaignUpdate

# Social Account CRUD operations
def get_social_account(db: Session, account_id: int) -> Optional[SocialAccount]:
    """Get a single social account by ID."""
    return db.query(SocialAccount).filter(SocialAccount.id == account_id).first()

def get_social_accounts(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    platform: Optional[str] = None,
    created_by_id: Optional[int] = None,
    is_active: Optional[bool] = None,
) -> List[SocialAccount]:
    """Get multiple social accounts with filtering."""
    query = db.query(SocialAccount)
    
    if platform:
        query = query.filter(SocialAccount.platform == platform)
    
    if created_by_id:
        query = query.filter(SocialAccount.created_by_id == created_by_id)
    
    if is_active is not None:
        query = query.filter(SocialAccount.is_active == is_active)
    
    return query.offset(skip).limit(limit).all()

def get_social_account_count(
    db: Session,
    platform: Optional[str] = None,
    created_by_id: Optional[int] = None,
    is_active: Optional[bool] = None,
) -> int:
    """Get count of social accounts with filtering."""
    query = db.query(SocialAccount)
    
    if platform:
        query = query.filter(SocialAccount.platform == platform)
    
    if created_by_id:
        query = query.filter(SocialAccount.created_by_id == created_by_id)
    
    if is_active is not None:
        query = query.filter(SocialAccount.is_active == is_active)
    
    return query.count()

def create_social_account(db: Session, account_in: SocialAccountCreate, user_id: int) -> SocialAccount:
    """Create a new social account."""
    account_data = account_in.dict(exclude_unset=True)
    account_data["created_by_id"] = user_id
    
    db_account = SocialAccount(**account_data)
    
    db.add(db_account)
    db.commit()
    db.refresh(db_account)
    
    return db_account

def update_social_account(
    db: Session, 
    account_id: int, 
    account_in: Union[SocialAccountUpdate, Dict[str, Any]]
) -> Optional[SocialAccount]:
    """Update an existing social account."""
    db_account = get_social_account(db, account_id)
    if not db_account:
        return None
    
    update_data = account_in if isinstance(account_in, dict) else account_in.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(db_account, field, value)
    
    db.add(db_account)
    db.commit()
    db.refresh(db_account)
    
    return db_account

def delete_social_account(db: Session, account_id: int) -> bool:
    """Delete a social account."""
    db_account = get_social_account(db, account_id)
    if not db_account:
        return False
    
    db.delete(db_account)
    db.commit()
    
    return True

# Social Post CRUD operations
def get_social_post(db: Session, post_id: int) -> Optional[SocialPost]:
    """Get a single social post by ID."""
    return db.query(SocialPost).filter(SocialPost.id == post_id).first()

def get_social_posts(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    account_id: Optional[int] = None,
    campaign_id: Optional[int] = None,
    status: Optional[str] = None,
    created_by_id: Optional[int] = None,
    scheduled_after: Optional[datetime] = None,
    scheduled_before: Optional[datetime] = None,
) -> List[SocialPost]:
    """Get multiple social posts with filtering."""
    query = db.query(SocialPost)
    
    if account_id:
        query = query.filter(SocialPost.account_id == account_id)
    
    if campaign_id:
        query = query.filter(SocialPost.campaign_id == campaign_id)
    
    if status:
        query = query.filter(SocialPost.status == status)
    
    if created_by_id:
        query = query.filter(SocialPost.created_by_id == created_by_id)
    
    if scheduled_after:
        query = query.filter(SocialPost.scheduled_time >= scheduled_after)
    
    if scheduled_before:
        query = query.filter(SocialPost.scheduled_time <= scheduled_before)
    
    return query.order_by(SocialPost.scheduled_time).offset(skip).limit(limit).all()

def get_social_post_count(
    db: Session,
    account_id: Optional[int] = None,
    campaign_id: Optional[int] = None,
    status: Optional[str] = None,
    created_by_id: Optional[int] = None,
    scheduled_after: Optional[datetime] = None,
    scheduled_before: Optional[datetime] = None,
) -> int:
    """Get count of social posts with filtering."""
    query = db.query(SocialPost)
    
    if account_id:
        query = query.filter(SocialPost.account_id == account_id)
    
    if campaign_id:
        query = query.filter(SocialPost.campaign_id == campaign_id)
    
    if status:
        query = query.filter(SocialPost.status == status)
    
    if created_by_id:
        query = query.filter(SocialPost.created_by_id == created_by_id)
    
    if scheduled_after:
        query = query.filter(SocialPost.scheduled_time >= scheduled_after)
    
    if scheduled_before:
        query = query.filter(SocialPost.scheduled_time <= scheduled_before)
    
    return query.count()

def create_social_post(db: Session, post_in: SocialPostCreate, user_id: int) -> SocialPost:
    """Create a new social post."""
    post_data = post_in.dict(exclude_unset=True)
    post_data["created_by_id"] = user_id
    post_data["status"] = PostStatus.scheduled
    
    # Handle JSON fields
    if "media_urls" in post_data and post_data["media_urls"] is not None:
        post_data["media_urls"] = jsonable_encoder(post_data["media_urls"])
    
    if "platform_data" in post_data and post_data["platform_data"] is not None:
        post_data["platform_data"] = jsonable_encoder(post_data["platform_data"])
        db_post = SocialPost(**post_data)
    
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    
    return db_post

def update_social_post(
    db: Session, 
    post_id: int, 
    post_in: Union[SocialPostUpdate, Dict[str, Any]]
) -> Optional[SocialPost]:
    """Update an existing social post."""
    db_post = get_social_post(db, post_id)
    if not db_post:
        return None
    
    # Only allow updates for posts that haven't been posted yet
    if db_post.status == PostStatus.posted:
        return None
    
    update_data = post_in if isinstance(post_in, dict) else post_in.dict(exclude_unset=True)
    
    # Handle JSON fields
    if "media_urls" in update_data and update_data["media_urls"] is not None:
        update_data["media_urls"] = jsonable_encoder(update_data["media_urls"])
    
    if "platform_data" in update_data and update_data["platform_data"] is not None:
        update_data["platform_data"] = jsonable_encoder(update_data["platform_data"])
    
    for field, value in update_data.items():
        setattr(db_post, field, value)
    
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    
    return db_post

def delete_social_post(db: Session, post_id: int) -> bool:
    """Delete a social post."""
    db_post = get_social_post(db, post_id)
    if not db_post:
        return False
    
    # Only allow deletion for posts that haven't been posted yet
    if db_post.status == PostStatus.posted:
        return False
    
    db.delete(db_post)
    db.commit()
    
    return True

# Social Campaign CRUD operations
def get_social_campaign(db: Session, campaign_id: int) -> Optional[SocialCampaign]:
    """Get a single social campaign by ID."""
    return db.query(SocialCampaign).filter(SocialCampaign.id == campaign_id).first()

def get_social_campaigns(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    status: Optional[str] = None,
    created_by_id: Optional[int] = None,
    active_on_date: Optional[datetime] = None,
) -> List[SocialCampaign]:
    """Get multiple social campaigns with filtering."""
    query = db.query(SocialCampaign)
    
    if status:
        query = query.filter(SocialCampaign.status == status)
    
    if created_by_id:
        query = query.filter(SocialCampaign.created_by_id == created_by_id)
    
    if active_on_date:
        query = query.filter(
            SocialCampaign.start_date <= active_on_date,
            SocialCampaign.end_date >= active_on_date
        )
    
    return query.order_by(desc(SocialCampaign.start_date)).offset(skip).limit(limit).all()

def get_social_campaign_count(
    db: Session,
    status: Optional[str] = None,
    created_by_id: Optional[int] = None,
    active_on_date: Optional[datetime] = None,
) -> int:
    """Get count of social campaigns with filtering."""
    query = db.query(SocialCampaign)
    
    if status:
        query = query.filter(SocialCampaign.status == status)
    
    if created_by_id:
        query = query.filter(SocialCampaign.created_by_id == created_by_id)
    
    if active_on_date:
        query = query.filter(
            SocialCampaign.start_date <= active_on_date,
            SocialCampaign.end_date >= active_on_date
        )
    
    return query.count()

def create_social_campaign(db: Session, campaign_in: SocialCampaignCreate, user_id: int) -> SocialCampaign:
    """Create a new social campaign."""
    # Extract account IDs
    account_ids = campaign_in.account_ids
    campaign_data = campaign_in.dict(exclude={"account_ids"})
    campaign_data["created_by_id"] = user_id
    
    # Create campaign
    db_campaign = SocialCampaign(**campaign_data)
    db.add(db_campaign)
    db.commit()
    db.refresh(db_campaign)
    
    # Add accounts to campaign
    if account_ids:
        for account_id in account_ids:
            account = db.query(SocialAccount).filter(SocialAccount.id == account_id).first()
            if account:
                campaign_account = CampaignAccount(
                    campaign_id=db_campaign.id,
                    account_id=account_id
                )
                db.add(campaign_account)
        
        db.commit()
        db.refresh(db_campaign)
    
    return db_campaign

def update_social_campaign(
    db: Session, 
    campaign_id: int, 
    campaign_in: Union[SocialCampaignUpdate, Dict[str, Any]]
) -> Optional[SocialCampaign]:
    """Update an existing social campaign."""
    db_campaign = get_social_campaign(db, campaign_id)
    if not db_campaign:
        return None
    
    update_data = campaign_in if isinstance(campaign_in, dict) else campaign_in.dict(exclude_unset=True)
    
    # Handle account IDs separately
    account_ids = None
    if "account_ids" in update_data:
        account_ids = update_data.pop("account_ids")
    
    # Update campaign fields
    for field, value in update_data.items():
        setattr(db_campaign, field, value)
    
    db.add(db_campaign)
    db.commit()
    db.refresh(db_campaign)
    
    # Update campaign accounts if provided
    if account_ids is not None:
        # Remove existing associations
        db.query(CampaignAccount).filter(CampaignAccount.campaign_id == campaign_id).delete()
        
        # Add new associations
        for account_id in account_ids:
            account = db.query(SocialAccount).filter(SocialAccount.id == account_id).first()
            if account:
                campaign_account = CampaignAccount(
                    campaign_id=campaign_id,
                    account_id=account_id
                )
                db.add(campaign_account)
        
        db.commit()
        db.refresh(db_campaign)
    
    return db_campaign

def delete_social_campaign(db: Session, campaign_id: int) -> bool:
    """Delete a social campaign."""
    db_campaign = get_social_campaign(db, campaign_id)
    if not db_campaign:
        return False
    
    # Remove campaign account associations
    db.query(CampaignAccount).filter(CampaignAccount.campaign_id == campaign_id).delete()
    
    # Remove campaign
    db.delete(db_campaign)
    db.commit()
    
    return True

def get_campaign_accounts(db: Session, campaign_id: int) -> List[SocialAccount]:
    """Get all social accounts associated with a campaign."""
    campaign = db.query(SocialCampaign).filter(SocialCampaign.id == campaign_id).first()
    if not campaign:
        return []
    
    return campaign.accounts

def get_upcoming_scheduled_posts(db: Session, days: int = 7) -> List[SocialPost]:
    """Get upcoming scheduled posts for the next X days."""
    now = datetime.utcnow()
    end_date = now + timedelta(days=days)
    
    return db.query(SocialPost).filter(
        SocialPost.status == PostStatus.scheduled,
        SocialPost.scheduled_time >= now,
        SocialPost.scheduled_time <= end_date
    ).order_by(SocialPost.scheduled_time).all()

def get_campaign_performance(db: Session, campaign_id: int) -> Dict[str, Any]:
    """Get performance metrics for a campaign."""
    # Get all posts for the campaign
    posts = db.query(SocialPost).filter(
        SocialPost.campaign_id == campaign_id,
        SocialPost.status == PostStatus.posted
    ).all()
    
    # Calculate metrics
    total_likes = sum(post.likes or 0 for post in posts)
    total_shares = sum(post.shares or 0 for post in posts)
    total_comments = sum(post.comments or 0 for post in posts)
    total_clicks = sum(post.clicks or 0 for post in posts)
    total_reach = sum(post.reach or 0 for post in posts)
    
    # Calculate engagement rate
    engagement_rate = 0
    if total_reach > 0:
        engagement_rate = (total_likes + total_shares + total_comments) / total_reach * 100
    
    return {
        "post_count": len(posts),
        "total_likes": total_likes,
        "total_shares": total_shares,
        "total_comments": total_comments,
        "total_clicks": total_clicks,
        "total_reach": total_reach,
        "engagement_rate": engagement_rate
    }