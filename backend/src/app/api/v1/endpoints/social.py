from typing import Any, List, Optional, Dict
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.core.security import get_current_user
from app.core.permissions import allow_all_authenticated
from app.models.user import User
from app.services.social_media import SocialMediaService
from app.crud import social as social_media_crud
from app.schemas.social import (
    SocialAccount, SocialAccountCreate, SocialAccountUpdate, SocialAccountList,
    SocialPost, SocialPostCreate, SocialPostUpdate, SocialPostList,
    SocialCampaign, SocialCampaignCreate, SocialCampaignUpdate, SocialCampaignList, SocialCampaignDetail,
    SocialPlatformEnum
)

router = APIRouter()

# Get social media service
def get_social_media_service():
    return SocialMediaService()

# Social Account endpoints
@router.get("/accounts", response_model=SocialAccountList)
def read_social_accounts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    platform: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> Any:
    """
    Retrieve social media accounts.
    """
    # Regular users can only see their own accounts, admins can see all
    if current_user.role in ["admin", "manager"]:
        accounts = social_media_crud.get_social_accounts(
            db=db,
            skip=skip,
            limit=limit,
            platform=platform,
            is_active=is_active
        )
        total = social_media_crud.get_social_account_count(
            db=db,
            platform=platform,
            is_active=is_active
        )
    else:
        accounts = social_media_crud.get_social_accounts(
            db=db,
            skip=skip,
            limit=limit,
            platform=platform,
            created_by_id=current_user.id,
            is_active=is_active
        )
        total = social_media_crud.get_social_account_count(
            db=db,
            platform=platform,
            created_by_id=current_user.id,
            is_active=is_active
        )
    
    # Calculate pagination information
    pages = (total + limit - 1) // limit  # Ceiling division
    
    return {
        "items": accounts,
        "total": total,
        "page": skip // limit + 1,
        "size": limit,
        "pages": pages
    }

@router.post("/accounts", response_model=SocialAccount)
def create_social_account(
    *,
    db: Session = Depends(get_db),
    account_in: SocialAccountCreate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Create new social media account.
    """
    account = social_media_crud.create_social_account(
        db=db, 
        account_in=account_in, 
        user_id=current_user.id
    )
    return account

@router.get("/accounts/{account_id}", response_model=SocialAccount)
def read_social_account(
    *,
    db: Session = Depends(get_db),
    account_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get social media account by ID.
    """
    account = social_media_crud.get_social_account(db=db, account_id=account_id)
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social account not found"
        )
    
    # Check if user has permission to view this account
    if account.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    return account

@router.put("/accounts/{account_id}", response_model=SocialAccount)
def update_social_account(
    *,
    db: Session = Depends(get_db),
    account_id: int = Path(..., ge=1),
    account_in: SocialAccountUpdate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Update social media account.
    """
    account = social_media_crud.get_social_account(db=db, account_id=account_id)
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social account not found"
        )
    
    # Check if user has permission to update this account
    if account.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    account = social_media_crud.update_social_account(
        db=db,
        account_id=account_id,
        account_in=account_in
    )
    
    return account

@router.delete("/accounts/{account_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_social_account(
    *,
    db: Session = Depends(get_db),
    account_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> None:
    """
    Delete social media account.
    """
    account = social_media_crud.get_social_account(db=db, account_id=account_id)
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social account not found"
        )
    
    # Check if user has permission to delete this account
    if account.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    social_media_crud.delete_social_account(db=db, account_id=account_id)
    return None

@router.get("/accounts/{account_id}/optimal-times", response_model=List[datetime])
def get_optimal_posting_times(
    *,
    db: Session = Depends(get_db),
    account_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
    social_media_service: SocialMediaService = Depends(get_social_media_service),
) -> Any:
    """
    Get optimal posting times for a social media account.
    """
    account = social_media_crud.get_social_account(db=db, account_id=account_id)
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social account not found"
        )
    
    # Check if user has permission to view this account
    if account.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    optimal_times = social_media_service.suggest_optimal_posting_time(account)
    return optimal_times

# Social Post endpoints
@router.get("/posts", response_model=SocialPostList)
def read_social_posts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    account_id: Optional[int] = None,
    campaign_id: Optional[int] = None,
    status: Optional[str] = None,
    days_ahead: Optional[int] = Query(None, ge=1, le=365),
) -> Any:
    """
    Retrieve social media posts.
    """
    # Set date filters if days_ahead is provided
    scheduled_after = None
    scheduled_before = None
    
    if days_ahead:
        scheduled_after = datetime.utcnow()
        scheduled_before = scheduled_after + timedelta(days=days_ahead)
    
    # Regular users can only see their own posts, admins can see all
    if current_user.role in ["admin", "manager"]:
        posts = social_media_crud.get_social_posts(
            db=db,
            skip=skip,
            limit=limit,
            account_id=account_id,
            campaign_id=campaign_id,
            status=status,
            scheduled_after=scheduled_after,
            scheduled_before=scheduled_before
        )
        total = social_media_crud.get_social_post_count(
            db=db,
            account_id=account_id,
            campaign_id=campaign_id,
            status=status,
            scheduled_after=scheduled_after,
            scheduled_before=scheduled_before
        )
    else:
        posts = social_media_crud.get_social_posts(
            db=db,
            skip=skip,
            limit=limit,
            account_id=account_id,
            campaign_id=campaign_id,
            status=status,
            created_by_id=current_user.id,
            scheduled_after=scheduled_after,
            scheduled_before=scheduled_before
        )
        total = social_media_crud.get_social_post_count(
            db=db,
            account_id=account_id,
            campaign_id=campaign_id,
            status=status,
            created_by_id=current_user.id,
            scheduled_after=scheduled_after,
            scheduled_before=scheduled_before
        )
    
    # Calculate pagination information
    pages = (total + limit - 1) // limit  # Ceiling division
    
    return {
        "items": posts,
        "total": total,
        "page": skip // limit + 1,
        "size": limit,
        "pages": pages
    }

@router.post("/posts", response_model=SocialPost)
def create_social_post(
    *,
    db: Session = Depends(get_db),
    post_in: SocialPostCreate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Create new social media post.
    """
    # Check if account exists and user has access
    account = social_media_crud.get_social_account(db=db, account_id=post_in.account_id)
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social account not found"
        )
    
    if account.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to post to this account"
        )
    
    # Check campaign if provided
    if post_in.campaign_id:
        campaign = social_media_crud.get_social_campaign(db=db, campaign_id=post_in.campaign_id)
        if not campaign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign not found"
            )
        
        if campaign.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to post to this campaign"
            )
    
    post = social_media_crud.create_social_post(
        db=db, 
        post_in=post_in, 
        user_id=current_user.id
    )
    return post

@router.get("/posts/{post_id}", response_model=SocialPost)
def read_social_post(
    *,
    db: Session = Depends(get_db),
    post_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get social media post by ID.
    """
    post = social_media_crud.get_social_post(db=db, post_id=post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social post not found"
        )
    
    # Check if user has permission to view this post
    if post.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    return post

@router.put("/posts/{post_id}", response_model=SocialPost)
def update_social_post(
    *,
    db: Session = Depends(get_db),
    post_id: int = Path(..., ge=1),
    post_in: SocialPostUpdate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Update social media post.
    """
    post = social_media_crud.get_social_post(db=db, post_id=post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social post not found"
        )
    
    # Check if user has permission to update this post
    if post.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Check if post can be updated (not already posted)
    if post.status == "posted":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot update a post that has already been published"
        )
    
    post = social_media_crud.update_social_post(
        db=db,
        post_id=post_id,
        post_in=post_in
    )
    
    if not post:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update post"
        )
    
    return post

@router.delete("/posts/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_social_post(
    *,
    db: Session = Depends(get_db),
    post_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> None:
    """
    Delete social media post.
    """
    post = social_media_crud.get_social_post(db=db, post_id=post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social post not found"
        )
    
    # Check if user has permission to delete this post
    if post.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Check if post can be deleted (not already posted)
    if post.status == "posted":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a post that has already been published"
        )
    
    success = social_media_crud.delete_social_post(db=db, post_id=post_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete post"
        )
    
    return None

@router.post("/posts/{post_id}/publish", response_model=SocialPost)
def publish_post_now(
    *,
    db: Session = Depends(get_db),
    post_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
    social_media_service: SocialMediaService = Depends(get_social_media_service),
) -> Any:
    """
    Publish a scheduled post immediately.
    """
    post = social_media_crud.get_social_post(db=db, post_id=post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social post not found"
        )
    
    # Check if user has permission to publish this post
    if post.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Check if post can be published (not already posted)
    if post.status == "posted":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Post has already been published"
        )
    
    # Publish post
    success = social_media_service.post_to_social_media(db=db, post_id=post_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to publish post"
        )
    
    # Refresh post data
    post = social_media_crud.get_social_post(db=db, post_id=post_id)
    return post

@router.post("/posts/{post_id}/fetch-metrics", response_model=SocialPost)
def fetch_post_metrics(
    *,
    db: Session = Depends(get_db),
    post_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
    social_media_service: SocialMediaService = Depends(get_social_media_service),
) -> Any:
    """
    Fetch the latest metrics for a published post.
    """
    post = social_media_crud.get_social_post(db=db, post_id=post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social post not found"
        )
    
    # Check if user has permission to access this post
    if post.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Check if post is published
    if post.status != "posted":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot fetch metrics for a post that hasn't been published"
        )
    
    # Fetch metrics
    success = social_media_service.fetch_post_metrics(db=db, post_id=post_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch post metrics"
        )
    
    # Refresh post data
    post = social_media_crud.get_social_post(db=db, post_id=post_id)
    return post

# Social Campaign endpoints
@router.get("/campaigns", response_model=SocialCampaignList)
def read_social_campaigns(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    status: Optional[str] = None,
    active_now: Optional[bool] = None,
) -> Any:
    """
    Retrieve social media campaigns.
    """
    # Set date filter if active_now is provided
    active_on_date = None
    if active_now:
        active_on_date = datetime.utcnow()
    
    # Regular users can only see their own campaigns, admins can see all
    if current_user.role in ["admin", "manager"]:
        campaigns = social_media_crud.get_social_campaigns(
            db=db,
            skip=skip,
            limit=limit,
            status=status,
            active_on_date=active_on_date
        )
        total = social_media_crud.get_social_campaign_count(
            db=db,
            status=status,
            active_on_date=active_on_date
        )
    else:
        campaigns = social_media_crud.get_social_campaigns(
            db=db,
            skip=skip,
            limit=limit,
            status=status,
            created_by_id=current_user.id,
            active_on_date=active_on_date
        )
        total = social_media_crud.get_social_campaign_count(
            db=db,
            status=status,
            created_by_id=current_user.id,
            active_on_date=active_on_date
        )
    
    # Calculate pagination information
    pages = (total + limit - 1) // limit  # Ceiling division
    
    return {
        "items": campaigns,
        "total": total,
        "page": skip // limit + 1,
        "size": limit,
        "pages": pages
    }

@router.post("/campaigns", response_model=SocialCampaign)
def create_social_campaign(
    *,
    db: Session = Depends(get_db),
    campaign_in: SocialCampaignCreate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Create new social media campaign.
    """
    # Check account permissions if accounts are specified
    if campaign_in.account_ids:
        for account_id in campaign_in.account_ids:
            account = social_media_crud.get_social_account(db=db, account_id=account_id)
            if not account:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Social account {account_id} not found"
                )
            
            if account.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not enough permissions to use account {account_id} in campaign"
                )
    
    # Create campaign with user_id parameter
    campaign = social_media_crud.create_social_campaign(
        db=db,
        campaign_in=campaign_in,
        user_id=current_user.id  # Using user_id instead of created_by_id
    )
    return campaign

@router.get("/campaigns/{campaign_id}", response_model=SocialCampaignDetail)
def read_social_campaign(
    *,
    db: Session = Depends(get_db),
    campaign_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get social media campaign by ID with associated accounts and posts.
    """
    campaign = social_media_crud.get_social_campaign(db=db, campaign_id=campaign_id)
    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social campaign not found"
        )
    
    # Check if user has permission to view this campaign
    if campaign.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Get associated accounts
    accounts = social_media_crud.get_campaign_accounts(db=db, campaign_id=campaign_id)
    
    # Get associated posts
    posts = social_media_crud.get_social_posts(db=db, campaign_id=campaign_id)
    
    # Build detailed response
    result = {
        **campaign.__dict__,
        "accounts": accounts,
        "posts": posts
    }
    
    return result

@router.put("/campaigns/{campaign_id}", response_model=SocialCampaign)
def update_social_campaign(
    *,
    db: Session = Depends(get_db),
    campaign_id: int = Path(..., ge=1),
    campaign_in: SocialCampaignUpdate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Update social media campaign.
    """
    campaign = social_media_crud.get_social_campaign(db=db, campaign_id=campaign_id)
    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social campaign not found"
        )
    
    # Check if user has permission to update this campaign
    if campaign.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    # Check account permissions if accounts are specified
    if campaign_in.account_ids:
        for account_id in campaign_in.account_ids:
            account = social_media_crud.get_social_account(db=db, account_id=account_id)
            if not account:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Social account {account_id} not found"
                )
            
            if account.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not enough permissions to use account {account_id} in campaign"
                )
    
    campaign = social_media_crud.update_social_campaign(
        db=db,
        campaign_id=campaign_id,
        campaign_in=campaign_in
    )
    
    return campaign

@router.delete("/campaigns/{campaign_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_social_campaign(
    *,
    db: Session = Depends(get_db),
    campaign_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> None:
    """
    Delete social media campaign.
    """
    campaign = social_media_crud.get_social_campaign(db=db, campaign_id=campaign_id)
    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social campaign not found"
        )
    
    # Check if user has permission to delete this campaign
    if campaign.created_by_id != current_user.id and current_user.role not in ["admin", "manager"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    social_media_crud.delete_social_campaign(db=db, campaign_id=campaign_id)
    return None

@router.get("/campaigns/{campaign_id}/performance", response_model=Dict[str, Any])
def get_campaign_performance(
    *,
    db: Session = Depends(get_db),
    campaign_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get performance metrics for a campaign.
    """
    campaign = social_media_crud.get_social_campaign(db=db, campaign_id=campaign_id)
    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social campaign not found"
        )
    
    # Check if user has permission to view this campaign
    if campaign.created_by_id != current_user.id and current_user.role not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    performance = social_media_crud.get_campaign_performance(db=db, campaign_id=campaign_id)
    return performance

# Background tasks and utility endpoints
@router.post("/process-scheduled-posts", status_code=status.HTTP_202_ACCEPTED)
def process_scheduled_posts(
    *,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(allow_all_authenticated),
    social_media_service: SocialMediaService = Depends(get_social_media_service),
) -> Any:
    """
    Process all scheduled posts that are due. Only for admins and managers.
    """
    # Add task to background
    background_tasks.add_task(social_media_service.process_scheduled_posts, db)
    
    return {"message": "Processing scheduled posts in the background"}

@router.get("/upcoming-posts", response_model=List[SocialPost])
def get_upcoming_posts(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    days: int = Query(7, ge=1, le=30),
) -> Any:
    """
    Get upcoming scheduled posts for the next X days.
    """
    posts = social_media_crud.get_upcoming_scheduled_posts(db=db, days=days)
    
    # Filter posts for non-admin users
    if current_user.role not in ["admin", "manager"]:
        posts = [post for post in posts if post.created_by_id == current_user.id]
    
    return posts

@router.get("/platforms", response_model=List[str])
def get_available_platforms() -> Any:
    """
    Get list of supported social media platforms.
    """
    return [platform.value for platform in SocialPlatformEnum]