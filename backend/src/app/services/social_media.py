import os
import json
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import tweepy
import facebook
import time
import random

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.social_media import SocialAccount, SocialPost, PostStatus, SocialCampaign
from app.schemas.social import SocialPlatformEnum

class SocialMediaService:
    """Service for interacting with social media platforms."""
    
    def __init__(self):
        # Platform API keys from settings
        self.twitter_api_key = settings.TWITTER_API_KEY
        self.twitter_api_secret = settings.TWITTER_API_SECRET
        self.facebook_app_id = settings.FACEBOOK_APP_ID
        self.facebook_app_secret = settings.FACEBOOK_APP_SECRET
        self.linkedin_client_id = settings.LINKEDIN_CLIENT_ID
        self.linkedin_client_secret = settings.LINKEDIN_CLIENT_SECRET
    
    def get_client(self, account: SocialAccount) -> Any:
        """Get the appropriate client for the social media platform."""
        if account.platform == SocialPlatformEnum.twitter:
            return self._get_twitter_client(account)
        elif account.platform == SocialPlatformEnum.facebook:
            return self._get_facebook_client(account)
        elif account.platform == SocialPlatformEnum.linkedin:
            return self._get_linkedin_client(account)
        else:
            # Default for platforms without specific implementations
            return None
    
    def _get_twitter_client(self, account: SocialAccount) -> tweepy.API:
        """Get Twitter client."""
        auth = tweepy.OAuth1UserHandler(
            self.twitter_api_key,
            self.twitter_api_secret,
            account.access_token,
            account.access_token_secret
        )
        return tweepy.API(auth)
    
    def _get_facebook_client(self, account: SocialAccount) -> facebook.GraphAPI:
        """Get Facebook client."""
        return facebook.GraphAPI(account.access_token)
    
    def _get_linkedin_client(self, account: SocialAccount) -> Any:
        """Get LinkedIn client."""
        # In a real implementation, you would use a LinkedIn client library
        # For simplicity, we'll just return the access token
        return account.access_token
    
    def post_to_social_media(self, db: Session, post_id: int) -> bool:
        """Post to social media platform."""
        # Get the post
        post = db.query(SocialPost).filter(SocialPost.id == post_id).first()
        if not post:
            return False
        
        # Get the account
        account = db.query(SocialAccount).filter(SocialAccount.id == post.account_id).first()
        if not account:
            return False
        
        # Verify scheduling time
        now = datetime.utcnow()
        if post.scheduled_time > now:
            # Not time to post yet
            return False
        
        try:
            # Get the appropriate client
            client = self.get_client(account)
            if not client:
                post.status = PostStatus.failed
                post.error_message = f"Unsupported platform: {account.platform}"
                db.add(post)
                db.commit()
                return False
            
            # Post to the platform
            if account.platform == SocialPlatformEnum.twitter:
                return self._post_to_twitter(db, post, client)
            elif account.platform == SocialPlatformEnum.facebook:
                return self._post_to_facebook(db, post, client)
            elif account.platform == SocialPlatformEnum.linkedin:
                return self._post_to_linkedin(db, post, client)
            else:
                post.status = PostStatus.failed
                post.error_message = f"Posting not implemented for {account.platform}"
                db.add(post)
                db.commit()
                return False
        
        except Exception as e:
            # Handle posting error
            post.status = PostStatus.failed
            post.error_message = str(e)
            db.add(post)
            db.commit()
            return False
    
    def _post_to_twitter(self, db: Session, post: SocialPost, client: tweepy.API) -> bool:
        """Post to Twitter."""
        try:
            # For demonstration, simulate a successful post
            # In real implementation, would use tweepy to post
            # tweet = client.update_status(post.content)
            
            # Simulate API response with random ID
            tweet_id = f"{random.randint(1000000000000000000, 9999999999999999999)}"
            
            # Update post with success status
            post.status = PostStatus.posted
            post.posted_time = datetime.utcnow()
            post.platform_post_id = tweet_id
            
            db.add(post)
            db.commit()
            return True
        
        except Exception as e:
            post.status = PostStatus.failed
            post.error_message = f"Twitter error: {str(e)}"
            db.add(post)
            db.commit()
            return False
    
    def _post_to_facebook(self, db: Session, post: SocialPost, client: facebook.GraphAPI) -> bool:
        """Post to Facebook."""
        try:
            # For demonstration, simulate a successful post
            # In real implementation, would use Facebook SDK
            # if post.media_urls and len(post.media_urls) > 0:
            #     response = client.put_photo(image=open(post.media_urls[0], 'rb'),
            #                            message=post.content)
            # else:
            #     response = client.put_object(
            #         parent_object='me',
            #         connection_name='feed',
            #         message=post.content,
            #         link=post.link if post.link else None,
            #     )
            
            # Simulate API response with random ID
            fb_post_id = f"{random.randint(1000000000000000000, 9999999999999999999)}"
            
            # Update post with success status
            post.status = PostStatus.posted
            post.posted_time = datetime.utcnow()
            post.platform_post_id = fb_post_id
            
            db.add(post)
            db.commit()
            return True
        
        except Exception as e:
            post.status = PostStatus.failed
            post.error_message = f"Facebook error: {str(e)}"
            db.add(post)
            db.commit()
            return False
    
    def _post_to_linkedin(self, db: Session, post: SocialPost, access_token: str) -> bool:
        """Post to LinkedIn."""
        try:
            # For demonstration, simulate a successful post
            # In real implementation, would use LinkedIn API
            
            # Simulate API response with random ID
            linkedin_post_id = f"urn:li:share:{random.randint(1000000000000, 9999999999999)}"
            
            # Update post with success status
            post.status = PostStatus.posted
            post.posted_time = datetime.utcnow()
            post.platform_post_id = linkedin_post_id
            
            db.add(post)
            db.commit()
            return True
        
        except Exception as e:
            post.status = PostStatus.failed
            post.error_message = f"LinkedIn error: {str(e)}"
            db.add(post)
            db.commit()
            return False
    
    def fetch_post_metrics(self, db: Session, post_id: int) -> bool:
        """Fetch metrics for a post."""
        # Get the post
        post = db.query(SocialPost).filter(SocialPost.id == post_id).first()
        if not post or post.status != PostStatus.posted:
            return False
        
        # Get the account
        account = db.query(SocialAccount).filter(SocialAccount.id == post.account_id).first()
        if not account:
            return False
        
        try:
            # Get the appropriate client
            client = self.get_client(account)
            if not client:
                return False
            
            # For demonstration, simulate fetching metrics
            # In real implementation, would use platform APIs
            
            # Generate some random metrics
            post.likes = random.randint(5, 100)
            post.shares = random.randint(1, 20)
            post.comments = random.randint(2, 30)
            post.clicks = random.randint(10, 200)
            post.reach = random.randint(100, 1000)
            
            # Calculate engagement rate
            if post.reach > 0:
                post.engagement_rate = (post.likes + post.shares + post.comments) / post.reach * 100
            
            post.last_status_check = datetime.utcnow()
            
            db.add(post)
            db.commit()
            return True
        
        except Exception as e:
            print(f"Error fetching metrics for post {post_id}: {str(e)}")
            return False
    
    def process_scheduled_posts(self, db: Session) -> int:
        """Process all scheduled posts that are due."""
        now = datetime.utcnow()
        
        # Find all scheduled posts that are due
        due_posts = db.query(SocialPost).filter(
            SocialPost.status == PostStatus.scheduled,
            SocialPost.scheduled_time <= now
        ).all()
        
        posts_processed = 0
        for post in due_posts:
            success = self.post_to_social_media(db, post.id)
            if success:
                posts_processed += 1
        
        return posts_processed
    
    def suggest_optimal_posting_time(self, account: SocialAccount) -> List[datetime]:
        """Suggest optimal posting times based on platform and audience."""
        # In a real implementation, this would use platform analytics
        # For demonstration, we'll use some general best practices
        
        suggested_times = []
        now = datetime.utcnow()
        
        # Start from tomorrow
        base_date = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        if account.platform == SocialPlatformEnum.twitter:
            # Twitter: 9 AM, 12 PM, 3 PM weekdays
            for day in range(7):
                if day < 5:  # Weekdays
                    suggested_times.append(base_date + timedelta(days=day, hours=9))
                    suggested_times.append(base_date + timedelta(days=day, hours=12))
                    suggested_times.append(base_date + timedelta(days=day, hours=15))
                else:  # Weekends
                    suggested_times.append(base_date + timedelta(days=day, hours=11))
                    suggested_times.append(base_date + timedelta(days=day, hours=15))
        
        elif account.platform == SocialPlatformEnum.facebook:
            # Facebook: 1 PM, 3 PM, 9 PM
            for day in range(7):
                suggested_times.append(base_date + timedelta(days=day, hours=13))
                suggested_times.append(base_date + timedelta(days=day, hours=15))
                suggested_times.append(base_date + timedelta(days=day, hours=21))
        
        elif account.platform == SocialPlatformEnum.instagram:
            # Instagram: 11 AM, 1 PM, 5 PM
            for day in range(7):
                suggested_times.append(base_date + timedelta(days=day, hours=11))
                suggested_times.append(base_date + timedelta(days=day, hours=13))
                suggested_times.append(base_date + timedelta(days=day, hours=17))
        
        elif account.platform == SocialPlatformEnum.linkedin:
            # LinkedIn: 9 AM, 12 PM, 5 PM (weekdays only)
            for day in range(5):  # Weekdays only
                suggested_times.append(base_date + timedelta(days=day, hours=9))
                suggested_times.append(base_date + timedelta(days=day, hours=12))
                suggested_times.append(base_date + timedelta(days=day, hours=17))
        
        else:
            # Default times for other platforms
            for day in range(7):
                suggested_times.append(base_date + timedelta(days=day, hours=12))
                suggested_times.append(base_date + timedelta(days=day, hours=18))
        
        return suggested_times