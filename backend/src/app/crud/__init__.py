from .social import (
    get_social_account,
    get_social_accounts,
    get_social_account_count,
    create_social_account,
    update_social_account,
    delete_social_account,
    get_social_post,
    get_social_posts,
    get_social_post_count,
    create_social_post,
    update_social_post,
    delete_social_post,
    get_social_campaign,
    get_social_campaigns,
    get_social_campaign_count,
    create_social_campaign,
    update_social_campaign,
    delete_social_campaign,
    get_campaign_accounts,
    get_upcoming_scheduled_posts,
    get_campaign_performance
)

# Create a module-like object that contains all the functions
class social_media:
    get_social_account = get_social_account
    get_social_accounts = get_social_accounts
    get_social_account_count = get_social_account_count
    create_social_account = create_social_account
    update_social_account = update_social_account
    delete_social_account = delete_social_account
    get_social_post = get_social_post
    get_social_posts = get_social_posts
    get_social_post_count = get_social_post_count
    create_social_post = create_social_post
    update_social_post = update_social_post
    delete_social_post = delete_social_post
    get_social_campaign = get_social_campaign
    get_social_campaigns = get_social_campaigns
    get_social_campaign_count = get_social_campaign_count
    create_social_campaign = create_social_campaign
    update_social_campaign = update_social_campaign
    delete_social_campaign = delete_social_campaign
    get_campaign_accounts = get_campaign_accounts
    get_upcoming_scheduled_posts = get_upcoming_scheduled_posts
    get_campaign_performance = get_campaign_performance

from app.crud.base import CRUDBase
from app.models.lead import Lead
from app.models.chatbot import ChatbotSession, ChatbotMessage, FAQ
from app.schemas.lead import LeadCreate, LeadUpdate
from app.schemas.chatbot import (
    ChatbotSessionCreate, ChatbotSessionUpdate,
    ChatbotMessageCreate, ChatbotMessageUpdate,
    FAQCreate, FAQUpdate
)

from typing import Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

class CRUDLead(CRUDBase[Lead, LeadCreate, LeadUpdate]):
    pass

class CRUDChatbotSession(CRUDBase[ChatbotSession, ChatbotSessionCreate, ChatbotSessionUpdate]):
    def get_session_by_id(self, db: Session, session_id: str) -> Optional[ChatbotSession]:
        """Get chatbot session by session_id"""
        return db.query(ChatbotSession).filter(ChatbotSession.session_id == session_id).first()
    
    def create_session(self, db: Session, session_id: str, user_id: Optional[int] = None, 
                      started_at: Optional[datetime] = None) -> ChatbotSession:
        """Create new chatbot session"""
        db_session = ChatbotSession(
            session_id=session_id,
            user_id=user_id,
            started_at=started_at or datetime.utcnow(),
            is_active=True
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        return db_session
    
    def get_session_messages(self, db: Session, session_id: int, limit: int = 50) -> List[ChatbotMessage]:
        """Get all messages for a session"""
        return (
            db.query(ChatbotMessage)
            .filter(ChatbotMessage.session_id == session_id)
            .order_by(ChatbotMessage.timestamp)
            .limit(limit)
            .all()
        )
    
    def create_message(self, db: Session, session_id: int, message: str, is_user: bool,
                      user_id: Optional[int] = None, intent: Optional[str] = None,
                      confidence: Optional[float] = None, timestamp: Optional[datetime] = None) -> ChatbotMessage:
        """Create new chatbot message"""
        db_message = ChatbotMessage(
            session_id=session_id,
            user_id=user_id,
            message=message,
            is_user=is_user,
            intent=intent,
            confidence=confidence,
            timestamp=timestamp or datetime.utcnow()
        )
        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        return db_message

class CRUDChatbotMessage(CRUDBase[ChatbotMessage, ChatbotMessageCreate, ChatbotMessageUpdate]):
    pass

class CRUDFAQ(CRUDBase[FAQ, FAQCreate, FAQUpdate]):
    pass

lead_crud = CRUDLead(Lead)
chatbot_crud = CRUDChatbotSession(ChatbotSession)
