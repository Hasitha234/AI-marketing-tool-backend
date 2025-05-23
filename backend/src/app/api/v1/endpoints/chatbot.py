# app/api/v1/endpoints/chatbot.py
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid
import logging

from app.api.dependencies import get_current_user, get_db
from app.models.user import User
from app.services.chatbot_service import ChatbotService
from app.schemas.chatbot import (
    ChatbotMessageRequest, ChatbotMessageResponse, ChatHistoryResponse,
    SessionAnalyticsResponse, DashboardAnalytics, FAQ, FAQCreate, FAQUpdate,
    WebhookRequest, WebhookResponse
)
from app.crud.chatbot import faq_crud, chatbot_analytics_crud

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatbotMessageResponse)
async def send_message(
    request: ChatbotMessageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Send a message to the chatbot and get a response
    """
    try:
        # Initialize chatbot service
        chatbot_service = ChatbotService(db)
        
        # Use authenticated user ID if available, otherwise use session-based interaction
        user_id = current_user.id if current_user else request.user_id
        
        # Process the message
        response = chatbot_service.process_message(
            session_id=request.session_id,
            message=request.message,
            user_id=user_id
        )
        
        # Schedule background tasks for analytics
        background_tasks.add_task(update_session_analytics, db, request.session_id)
        
        return ChatbotMessageResponse(**response)
        
    except Exception as e:
        logger.error(f"Error processing chatbot message: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing message")


@router.get("/sessions/{session_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get chat history for a specific session
    """
    try:
        chatbot_service = ChatbotService(db)
        messages = chatbot_service.get_chat_history(session_id, limit)
        
        return ChatHistoryResponse(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages)
        )
        
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving chat history")


@router.get("/sessions/{session_id}/analytics", response_model=SessionAnalyticsResponse)
async def get_session_analytics(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # Require authentication for analytics
):
    """
    Get analytics for a specific chat session
    """
    try:
        chatbot_service = ChatbotService(db)
        analytics = chatbot_service.get_session_analytics(session_id)
        
        if not analytics:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionAnalyticsResponse(**analytics)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving analytics")


@router.get("/analytics/dashboard", response_model=DashboardAnalytics)
async def get_dashboard_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dashboard analytics for chatbot performance
    """
    try:
        today = datetime.utcnow().date()
        
        # Get today's analytics
        today_analytics = chatbot_analytics_crud.get_daily_analytics(db, datetime.combine(today, datetime.min.time()))
        
        # Get trend data (last 7 days)
        end_date = datetime.combine(today, datetime.min.time())
        start_date = end_date - timedelta(days=7)
        trend_data = chatbot_analytics_crud.get_analytics_range(db, start_date, end_date)
        
        # Format trend data
        sessions_trend = [
            {
                "date": record.date.strftime("%Y-%m-%d"),
                "sessions": record.total_sessions,
                "messages": record.total_messages
            }
            for record in trend_data
        ]
        
        # Get popular FAQs (mock data - would need to track FAQ usage)
        popular_faqs = [
            {"question": "What are your pricing plans?", "usage_count": 25},
            {"question": "How do I schedule a demo?", "usage_count": 18},
            {"question": "What features do you offer?", "usage_count": 15}
        ]
        
        return DashboardAnalytics(
            total_sessions_today=today_analytics.total_sessions if today_analytics else 0,
            total_messages_today=today_analytics.total_messages if today_analytics else 0,
            active_sessions=5,  # Would need to count from active sessions
            avg_confidence_today=0.85,  # Would calculate from today's messages
            top_intents_today=[
                {"intent": "Default Welcome Intent", "count": 10},
                {"intent": "FAQ - Pricing", "count": 8},
                {"intent": "Schedule Appointment", "count": 6}
            ],
            conversion_rate_today=today_analytics.conversion_rate if today_analytics else 0.0,
            sessions_trend=sessions_trend,
            popular_faqs=popular_faqs
        )
        
    except Exception as e:
        logger.error(f"Error retrieving dashboard analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving dashboard analytics")


@router.get("/sessions/new")
async def create_new_session():
    """
    Create a new chat session and return session ID
    """
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


# FAQ Management Endpoints
@router.get("/faqs", response_model=List[FAQ])
async def get_faqs(
    category: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get FAQs, optionally filtered by category
    """
    try:
        if category:
            faqs = faq_crud.get_by_category(db, category)
        else:
            faqs = faq_crud.get_active_faqs(db)
        
        return faqs[skip:skip + limit]
        
    except Exception as e:
        logger.error(f"Error retrieving FAQs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving FAQs")


@router.post("/faqs", response_model=FAQ)
async def create_faq(
    faq: FAQCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new FAQ
    """
    try:
        return faq_crud.create(db, obj_in=faq)
        
    except Exception as e:
        logger.error(f"Error creating FAQ: {str(e)}")
        raise HTTPException(status_code=500, detail="Error creating FAQ")


@router.put("/faqs/{faq_id}", response_model=FAQ)
async def update_faq(
    faq_id: int,
    faq_update: FAQUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update an existing FAQ
    """
    try:
        faq = faq_crud.get(db, id=faq_id)
        if not faq:
            raise HTTPException(status_code=404, detail="FAQ not found")
        
        return faq_crud.update(db, db_obj=faq, obj_in=faq_update)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating FAQ: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating FAQ")


@router.delete("/faqs/{faq_id}")
async def delete_faq(
    faq_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete an FAQ
    """
    try:
        faq = faq_crud.get(db, id=faq_id)
        if not faq:
            raise HTTPException(status_code=404, detail="FAQ not found")
        
        faq_crud.remove(db, id=faq_id)
        return {"message": "FAQ deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting FAQ: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting FAQ")


@router.get("/faqs/search")
async def search_faqs(
    q: str,
    db: Session = Depends(get_db)
):
    """
    Search FAQs by question content
    """
    try:
        if not q or len(q.strip()) < 2:
            raise HTTPException(status_code=400, detail="Search query must be at least 2 characters")
        
        faqs = faq_crud.search_faqs(db, q.strip())
        return faqs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching FAQs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error searching FAQs")


# Webhook endpoint for Dialogflow integration
@router.post("/webhook", response_model=WebhookResponse)
async def dialogflow_webhook(
    request: WebhookRequest,
    db: Session = Depends(get_db)
):
    """
    Webhook endpoint for Dialogflow fulfillment
    """
    try:
        chatbot_service = ChatbotService(db)
        
        # Process the webhook request
        response = chatbot_service.process_message(
            session_id=request.session_id,
            message=request.query_text,
            user_id=request.user_id
        )
        
        return WebhookResponse(
            fulfillment_text=response["response"],
            source="ai-marketing-chatbot",
            payload={
                "quick_replies": response.get("quick_replies", []),
                "actions": response.get("actions", [])
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return WebhookResponse(
            fulfillment_text="I'm experiencing some technical difficulties. Please try again later.",
            source="ai-marketing-chatbot"
        )


# Background task functions
async def update_session_analytics(db: Session, session_id: str):
    """Background task to update session analytics"""
    try:
        # This would update various analytics metrics
        # Implementation depends on specific requirements
        pass
    except Exception as e:
        logger.error(f"Error updating session analytics: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check endpoint for the chatbot service
    """
    return {
        "status": "healthy",
        "service": "chatbot",
        "timestamp": datetime.utcnow().isoformat()
    }