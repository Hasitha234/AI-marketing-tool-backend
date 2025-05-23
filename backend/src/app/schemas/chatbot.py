# app/schemas/chatbot.py
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# ChatbotSession Schemas
class ChatbotSessionBase(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[int] = Field(None, description="User ID if authenticated")


class ChatbotSessionCreate(ChatbotSessionBase):
    started_at: Optional[datetime] = None


class ChatbotSessionUpdate(BaseModel):
    ended_at: Optional[datetime] = None
    is_active: Optional[bool] = None


class ChatbotSession(ChatbotSessionBase):
    id: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    is_active: bool = True
    
    class Config:
        from_attributes = True


class ChatbotSessionWithMessages(ChatbotSession):
    messages: List["ChatbotMessage"] = []


# ChatbotMessage Schemas
class ChatbotMessageBase(BaseModel):
    message: str = Field(..., description="Message content")
    is_user: bool = Field(..., description="True if message is from user, False if from bot")


class ChatbotMessageCreate(ChatbotMessageBase):
    session_id: int
    user_id: Optional[int] = None
    intent: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    timestamp: Optional[datetime] = None


class ChatbotMessageUpdate(BaseModel):
    intent: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class ChatbotMessage(ChatbotMessageBase):
    id: int
    session_id: int
    user_id: Optional[int] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: datetime
    
    class Config:
        from_attributes = True


# FAQ Schemas
class FAQBase(BaseModel):
    question: str = Field(..., description="FAQ question")
    answer: str = Field(..., description="FAQ answer")
    category: str = Field(..., description="FAQ category")
    keywords: Optional[str] = Field(None, description="Keywords for search (JSON string)")


class FAQCreate(FAQBase):
    is_active: bool = True


class FAQUpdate(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    category: Optional[str] = None
    keywords: Optional[str] = None
    is_active: Optional[bool] = None


class FAQ(FAQBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Chatbot API Request/Response Schemas
class ChatbotMessageRequest(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    user_id: Optional[int] = Field(None, description="User ID if authenticated")


class ChatbotMessageResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    confidence: float
    quick_replies: List[str] = []
    actions: List[str] = []
    timestamp: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    total_messages: int


class SessionAnalyticsResponse(BaseModel):
    session_id: str
    total_messages: int
    user_messages: int
    bot_messages: int
    unique_intents: List[str]
    average_confidence: float
    session_duration: float  # in minutes
    started_at: str


# Analytics Schemas
class ChatbotAnalyticsBase(BaseModel):
    date: datetime
    total_sessions: int = 0
    total_messages: int = 0
    unique_users: int = 0
    avg_session_duration: float = 0.0
    avg_messages_per_session: float = 0.0
    top_intents: Optional[str] = None
    conversion_rate: float = 0.0


class ChatbotAnalytics(ChatbotAnalyticsBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class DashboardAnalytics(BaseModel):
    total_sessions_today: int
    total_messages_today: int
    active_sessions: int
    avg_confidence_today: float
    top_intents_today: List[Dict[str, Any]]
    conversion_rate_today: float
    sessions_trend: List[Dict[str, Any]]  # Last 7 days
    popular_faqs: List[Dict[str, Any]]


# Dialogflow Integration Schemas
class DialogflowResponse(BaseModel):
    intent: str
    confidence: float
    response_text: str
    parameters: Dict[str, Any]
    all_required_params_present: bool


class IntentTrainingData(BaseModel):
    intent_name: str
    training_phrases: List[str]
    responses: List[str]
    parameters: Optional[List[Dict[str, Any]]] = None


# Business Logic Schemas
class LeadQualificationData(BaseModel):
    company_name: Optional[str] = None
    company_size: Optional[str] = None
    industry: Optional[str] = None
    budget: Optional[str] = None
    timeline: Optional[str] = None
    pain_points: Optional[List[str]] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None


class AppointmentRequest(BaseModel):
    consultation_type: str = Field(..., description="Type of consultation requested")
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    timezone: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    additional_notes: Optional[str] = None


class ProductRecommendation(BaseModel):
    recommended_plan: str
    plan_features: List[str]
    pricing: Optional[str] = None
    match_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


# Webhook Schemas (for external integrations)
class WebhookRequest(BaseModel):
    session_id: str
    intent: str
    parameters: Dict[str, Any]
    query_text: str
    user_id: Optional[int] = None


class WebhookResponse(BaseModel):
    fulfillment_text: str
    fulfillment_messages: Optional[List[Dict[str, Any]]] = None
    source: str = "ai-marketing-chatbot"
    payload: Optional[Dict[str, Any]] = None


# Update forward references
ChatbotSessionWithMessages.model_rebuild()