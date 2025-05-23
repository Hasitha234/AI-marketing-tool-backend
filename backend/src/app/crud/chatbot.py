from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime, timedelta
from app.crud.base import CRUDBase
from app.models.chatbot import ChatbotSession, ChatbotMessage, FAQ, ChatbotAnalytics
from app.schemas.chatbot import (
    ChatbotSessionCreate, ChatbotSessionUpdate,
    ChatbotMessageCreate, ChatbotMessageUpdate,
    FAQCreate, FAQUpdate
)


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
    
    def end_session(self, db: Session, session_id: str) -> Optional[ChatbotSession]:
        """End chatbot session"""
        session = self.get_session_by_id(db, session_id)
        if session:
            session.ended_at = datetime.utcnow()
            session.is_active = False
            db.commit()
            db.refresh(session)
        return session
    
    def get_active_sessions(self, db: Session, skip: int = 0, limit: int = 100) -> List[ChatbotSession]:
        """Get active chatbot sessions"""
        return (
            db.query(ChatbotSession)
            .filter(ChatbotSession.is_active == True)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_user_sessions(self, db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[ChatbotSession]:
        """Get all sessions for a specific user"""
        return (
            db.query(ChatbotSession)
            .filter(ChatbotSession.user_id == user_id)
            .order_by(desc(ChatbotSession.started_at))
            .offset(skip)
            .limit(limit)
            .all()
        )


class CRUDChatbotMessage(CRUDBase[ChatbotMessage, ChatbotMessageCreate, ChatbotMessageUpdate]):
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
    
    def get_session_messages(self, db: Session, session_id: int, limit: int = 50) -> List[ChatbotMessage]:
        """Get all messages for a session"""
        return (
            db.query(ChatbotMessage)
            .filter(ChatbotMessage.session_id == session_id)
            .order_by(ChatbotMessage.timestamp)
            .limit(limit)
            .all()
        )
    
    def get_user_messages(self, db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[ChatbotMessage]:
        """Get all messages for a user"""
        return (
            db.query(ChatbotMessage)
            .filter(ChatbotMessage.user_id == user_id)
            .order_by(desc(ChatbotMessage.timestamp))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_messages_by_intent(self, db: Session, intent: str, skip: int = 0, limit: int = 100) -> List[ChatbotMessage]:
        """Get messages by intent"""
        return (
            db.query(ChatbotMessage)
            .filter(ChatbotMessage.intent == intent)
            .order_by(desc(ChatbotMessage.timestamp))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_low_confidence_messages(self, db: Session, threshold: float = 0.5, 
                                   skip: int = 0, limit: int = 100) -> List[ChatbotMessage]:
        """Get messages with low confidence scores"""
        return (
            db.query(ChatbotMessage)
            .filter(ChatbotMessage.confidence < threshold)
            .filter(ChatbotMessage.confidence.isnot(None))
            .order_by(desc(ChatbotMessage.timestamp))
            .offset(skip)
            .limit(limit)
            .all()
        )


class CRUDFAQ(CRUDBase[FAQ, FAQCreate, FAQUpdate]):
    def get_by_category(self, db: Session, category: str) -> List[FAQ]:
        """Get FAQs by category"""
        return (
            db.query(FAQ)
            .filter(FAQ.category == category)
            .filter(FAQ.is_active == True)
            .all()
        )
    
    def search_faqs(self, db: Session, query: str) -> List[FAQ]:
        """Search FAQs by question content"""
        return (
            db.query(FAQ)
            .filter(FAQ.question.contains(query))
            .filter(FAQ.is_active == True)
            .all()
        )
    
    def get_active_faqs(self, db: Session) -> List[FAQ]:
        """Get all active FAQs"""
        return (
            db.query(FAQ)
            .filter(FAQ.is_active == True)
            .order_by(FAQ.category, FAQ.question)
            .all()
        )


class CRUDChatbotAnalytics(CRUDBase[ChatbotAnalytics, None, None]):
    def get_daily_analytics(self, db: Session, date: datetime) -> Optional[ChatbotAnalytics]:
        """Get analytics for a specific date"""
        return (
            db.query(ChatbotAnalytics)
            .filter(func.date(ChatbotAnalytics.date) == date.date())
            .first()
        )
    
    def get_analytics_range(self, db: Session, start_date: datetime, 
                           end_date: datetime) -> List[ChatbotAnalytics]:
        """Get analytics for a date range"""
        return (
            db.query(ChatbotAnalytics)
            .filter(ChatbotAnalytics.date >= start_date)
            .filter(ChatbotAnalytics.date <= end_date)
            .order_by(ChatbotAnalytics.date)
            .all()
        )
    
    def create_daily_analytics(self, db: Session, date: datetime, 
                              analytics_data: Dict[str, Any]) -> ChatbotAnalytics:
        """Create daily analytics record"""
        db_analytics = ChatbotAnalytics(
            date=date,
            total_sessions=analytics_data.get("total_sessions", 0),
            total_messages=analytics_data.get("total_messages", 0),
            unique_users=analytics_data.get("unique_users", 0),
            avg_session_duration=analytics_data.get("avg_session_duration", 0.0),
            avg_messages_per_session=analytics_data.get("avg_messages_per_session", 0.0),
            top_intents=analytics_data.get("top_intents"),
            conversion_rate=analytics_data.get("conversion_rate", 0.0)
        )
        db.add(db_analytics)
        db.commit()
        db.refresh(db_analytics)
        return db_analytics
    
    def calculate_daily_analytics(self, db: Session, date: datetime) -> Dict[str, Any]:
        """Calculate analytics for a specific date"""
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        
        # Get sessions for the day
        sessions = (
            db.query(ChatbotSession)
            .filter(ChatbotSession.started_at >= start_date)
            .filter(ChatbotSession.started_at < end_date)
            .all()
        )
        
        # Get messages for the day
        messages = (
            db.query(ChatbotMessage)
            .filter(ChatbotMessage.timestamp >= start_date)
            .filter(ChatbotMessage.timestamp < end_date)
            .all()
        )
        
        total_sessions = len(sessions)
        total_messages = len(messages)
        unique_users = len(set([s.user_id for s in sessions if s.user_id]))
        
        # Calculate average session duration
        session_durations = []
        for session in sessions:
            if session.ended_at:
                duration = (session.ended_at - session.started_at).total_seconds() / 60
                session_durations.append(duration)
        
        avg_session_duration = sum(session_durations) / len(session_durations) if session_durations else 0
        
        # Calculate average messages per session
        avg_messages_per_session = total_messages / total_sessions if total_sessions > 0 else 0
        
        # Get top intents
        intents = [m.intent for m in messages if m.intent and not m.is_user]
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        top_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "unique_users": unique_users,
            "avg_session_duration": round(avg_session_duration, 2),
            "avg_messages_per_session": round(avg_messages_per_session, 2),
            "top_intents": str(top_intents),
            "conversion_rate": 0.0  # This would need to be calculated based on leads generated
        }


# Create instances
chatbot_session_crud = CRUDChatbotSession(ChatbotSession)
chatbot_message_crud = CRUDChatbotMessage(ChatbotMessage)
faq_crud = CRUDFAQ(FAQ)
chatbot_analytics_crud = CRUDChatbotAnalytics(ChatbotAnalytics)

# For backward compatibility
chatbot_crud = chatbot_session_crud