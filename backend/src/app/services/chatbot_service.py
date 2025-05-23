# app/services/chatbot_service.py
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from google.cloud import dialogflow
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.chatbot import ChatbotSession, ChatbotMessage, FAQ
from app.models.lead import Lead
from app.models.user import User
from app.crud import chatbot_crud, lead_crud
import logging

logger = logging.getLogger(__name__)

class DialogflowService:
    def __init__(self):
        self.project_id = settings.GOOGLE_CLOUD_PROJECT_ID
        self.language_code = "en-US"
        
    def create_session_client(self):
        """Create Dialogflow session client"""
        return dialogflow.SessionsClient()
    
    def detect_intent(self, session_id: str, text_input: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Detect intent from user input using Dialogflow"""
        try:
            session_client = self.create_session_client()
            session = session_client.session_path(self.project_id, session_id)
            
            text_input = dialogflow.TextInput(text=text_input, language_code=self.language_code)
            query_input = dialogflow.QueryInput(text=text_input)
            
            response = session_client.detect_intent(
                request={"session": session, "query_input": query_input}
            )
            
            return {
                "intent": response.query_result.intent.display_name,
                "confidence": response.query_result.intent_detection_confidence,
                "response_text": response.query_result.fulfillment_text,
                "parameters": dict(response.query_result.parameters),
                "all_required_params_present": response.query_result.all_required_params_present
            }
            
        except Exception as e:
            logger.error(f"Error in Dialogflow intent detection: {str(e)}")
            return {
                "intent": "Default Fallback Intent",
                "confidence": 0.0,
                "response_text": "I'm sorry, I didn't understand that. Could you please rephrase?",
                "parameters": {},
                "all_required_params_present": False
            }

class ChatbotService:
    def __init__(self, db: Session):
        self.db = db
        self.dialogflow_service = DialogflowService()
        
    def process_message(self, session_id: str, message: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Process incoming chatbot message"""
        try:
            # Get or create chatbot session
            chatbot_session = self._get_or_create_session(session_id, user_id)
            
            # Save user message
            user_message = self._save_message(
                session_id=chatbot_session.id,
                message=message,
                is_user=True,
                user_id=user_id
            )
            
            # Get Dialogflow response
            intent_result = self.dialogflow_service.detect_intent(session_id, message, user_id)
            
            # Process intent and generate response
            response = self._process_intent(intent_result, chatbot_session, user_id)
            
            # Save bot response
            bot_message = self._save_message(
                session_id=chatbot_session.id,
                message=response["message"],
                is_user=False,
                intent=intent_result["intent"],
                confidence=intent_result["confidence"]
            )
            
            return {
                "session_id": session_id,
                "response": response["message"],
                "intent": intent_result["intent"],
                "confidence": intent_result["confidence"],
                "quick_replies": response.get("quick_replies", []),
                "actions": response.get("actions", []),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing chatbot message: {str(e)}")
            return {
                "session_id": session_id,
                "response": "I'm experiencing some technical difficulties. Please try again later.",
                "intent": "error",
                "confidence": 0.0,
                "quick_replies": [],
                "actions": [],
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _get_or_create_session(self, session_id: str, user_id: Optional[int]) -> ChatbotSession:
        """Get existing session or create new one"""
        session = chatbot_crud.get_session_by_id(self.db, session_id)
        if not session:
            session = chatbot_crud.create_session(
                self.db,
                session_id=session_id,
                user_id=user_id,
                started_at=datetime.utcnow()
            )
        return session
    
    def _save_message(self, session_id: int, message: str, is_user: bool, 
                     user_id: Optional[int] = None, intent: Optional[str] = None, 
                     confidence: Optional[float] = None) -> ChatbotMessage:
        """Save message to database"""
        return chatbot_crud.create_message(
            self.db,
            session_id=session_id,
            message=message,
            is_user=is_user,
            user_id=user_id,
            intent=intent,
            confidence=confidence,
            timestamp=datetime.utcnow()
        )
    
    def _process_intent(self, intent_result: Dict[str, Any], session: ChatbotSession, 
                      user_id: Optional[int]) -> Dict[str, Any]:
        """Process detected intent and generate appropriate response"""
        intent = intent_result["intent"]
        parameters = intent_result["parameters"]
        
        # Intent handlers
        intent_handlers = {
            "Default Welcome Intent": self._handle_welcome,
            "FAQ - Pricing": self._handle_pricing_faq,
            "FAQ - Features": self._handle_features_faq,
            "FAQ - Support": self._handle_support_faq,
            "Schedule Appointment": self._handle_appointment_scheduling,
            "Product Recommendation": self._handle_product_recommendation,
            "Lead Qualification": self._handle_lead_qualification,
            "Contact Sales": self._handle_contact_sales,
            "Default Fallback Intent": self._handle_fallback
        }
        
        handler = intent_handlers.get(intent, self._handle_fallback)
        return handler(intent_result, session, user_id)
    
    def _handle_welcome(self, intent_result: Dict, session: ChatbotSession, user_id: Optional[int]) -> Dict:
        """Handle welcome intent"""
        return {
            "message": "Hello! 👋 Welcome to our AI Marketing Platform. I'm here to help you with:\n\n• Product information and pricing\n• Feature demonstrations\n• Scheduling consultations\n• Answering your questions\n\nHow can I assist you today?",
            "quick_replies": [
                "Tell me about pricing",
                "What features do you offer?",
                "Schedule a demo",
                "Contact sales team"
            ]
        }
    
    def _handle_pricing_faq(self, intent_result: Dict, session: ChatbotSession, user_id: Optional[int]) -> Dict:
        """Handle pricing FAQ"""
        return {
            "message": "Our AI Marketing Platform offers flexible pricing plans:\n\n💼 **Starter Plan - $49/month**\n• Lead scoring for up to 1,000 leads\n• Basic content generation\n• Email support\n\n🚀 **Professional Plan - $149/month**\n• Lead scoring for up to 10,000 leads\n• Advanced content generation\n• Social media automation\n• Priority support\n\n🏢 **Enterprise Plan - Custom pricing**\n• Unlimited leads\n• Custom integrations\n• Dedicated account manager\n• 24/7 phone support\n\nWould you like to schedule a demo or speak with our sales team?",
            "quick_replies": [
                "Schedule a demo",
                "Contact sales",
                "More about features"
            ]
        }
    
    def _handle_features_faq(self, intent_result: Dict, session: ChatbotSession, user_id: Optional[int]) -> Dict:
        """Handle features FAQ"""
        return {
            "message": "Our AI Marketing Platform includes powerful features:\n\n🎯 **Lead Scoring & Management**\n• AI-powered lead qualification\n• Behavioral tracking\n• Automated lead routing\n\n✍️ **Content Generation**\n• AI-powered blog posts\n• Social media content\n• Email campaigns\n\n🤖 **Chatbot Integration**\n• 24/7 customer support\n• Lead qualification\n• Appointment scheduling\n\n📱 **Social Media Automation**\n• Multi-platform posting\n• Content scheduling\n• Performance analytics\n\nWhich feature interests you most?",
            "quick_replies": [
                "Lead scoring details",
                "Content generation demo",
                "Social media automation",
                "Schedule consultation"
            ]
        }
    
    def _handle_support_faq(self, intent_result: Dict, session: ChatbotSession, user_id: Optional[int]) -> Dict:
        """Handle support FAQ"""
        return {
            "message": "We offer comprehensive support for our platform:\n\n📧 **Email Support** - Available on all plans\n• Response within 24 hours\n• Technical assistance\n• How-to guides\n\n💬 **Live Chat** - Professional & Enterprise plans\n• Business hours support\n• Real-time assistance\n\n📞 **Phone Support** - Enterprise plan only\n• 24/7 availability\n• Dedicated account manager\n• Priority escalation\n\n📚 **Self-Service Resources**\n• Knowledge base\n• Video tutorials\n• API documentation\n\nHow can I help you get started?",
            "quick_replies": [
                "Contact support team",
                "View documentation",
                "Schedule training session"
            ]
        }
    
    def _handle_appointment_scheduling(self, intent_result: Dict, session: ChatbotSession, user_id: Optional[int]) -> Dict:
        """Handle appointment scheduling"""
        parameters = intent_result.get("parameters", {})
        
        if not intent_result.get("all_required_params_present", False):
            return {
                "message": "I'd be happy to help you schedule a consultation! 📅\n\nTo book your appointment, I'll need:\n• Your preferred date and time\n• Type of consultation (demo, technical discussion, pricing)\n• Your contact information\n\nWhat type of consultation are you interested in?",
                "quick_replies": [
                    "Product demo",
                    "Technical consultation",
                    "Pricing discussion",
                    "General consultation"
                ]
            }
        
        # Create lead entry for appointment
        if user_id:
            self._create_appointment_lead(parameters, user_id, session)
        
        return {
            "message": "Perfect! I've noted your appointment request. 🎉\n\nOur sales team will contact you within 2 hours to confirm the details and send you a calendar invitation.\n\nIn the meantime, would you like me to send you some resources about our platform?",
            "quick_replies": [
                "Yes, send resources",
                "No, that's all for now",
                "I have more questions"
            ],
            "actions": ["create_lead"]
        }
    
    def _handle_product_recommendation(self, intent_result: Dict, session: ChatbotSession, user_id: Optional[int]) -> Dict:
        """Handle product recommendation"""
        parameters = intent_result.get("parameters", {})
        company_size = parameters.get("company-size", "").lower()
        budget = parameters.get("budget", "").lower()
        
        if "small" in company_size or "startup" in company_size:
            plan = "Starter Plan"
            features = "lead scoring, basic content generation, and email support"
        elif "enterprise" in company_size or "large" in company_size:
            plan = "Enterprise Plan"
            features = "unlimited leads, custom integrations, and dedicated support"
        else:
            plan = "Professional Plan"
            features = "advanced lead scoring, content automation, and priority support"
        
        return {
            "message": f"Based on your requirements, I'd recommend our **{plan}**! 🎯\n\nThis plan includes {features}, which aligns perfectly with your needs.\n\nWould you like to:\n• Schedule a personalized demo\n• Speak with our sales team\n• Start a free trial",
            "quick_replies": [
                "Schedule demo",
                "Start free trial",
                "Contact sales team",
                "Compare all plans"
            ]
        }
    
    def _handle_lead_qualification(self, intent_result: Dict, session: ChatbotSession, user_id: Optional[int]) -> Dict:
        """Handle lead qualification"""
        parameters = intent_result.get("parameters", {})
        
        # Create qualified lead
        if user_id and parameters:
            self._create_qualified_lead(parameters, user_id, session)
        
        return {
            "message": "Thank you for providing that information! 📝\n\nBased on your responses, you seem like a great fit for our platform. I've forwarded your details to our sales team.\n\nA specialist will reach out to you within the next hour to discuss how we can help grow your business.\n\nIs there anything else I can help you with while you wait?",
            "quick_replies": [
                "Tell me about integrations",
                "Show me case studies",
                "Pricing information",
                "That's all for now"
            ],
            "actions": ["create_qualified_lead"]
        }
    
    def _handle_contact_sales(self, intent_result: Dict, session: ChatbotSession, user_id: Optional[int]) -> Dict:
        """Handle contact sales request"""
        return {
            "message": "I'll connect you with our sales team right away! 📞\n\n**Sales Team Contact:**\n• Email: sales@aimarketingplatform.com\n• Phone: +1 (555) 123-4567\n• Live Chat: Available 9 AM - 6 PM EST\n\nOr I can have someone call you back. Would you prefer:\n• Immediate callback (if available)\n• Scheduled call at your convenience\n• Email follow-up with detailed information",
            "quick_replies": [
                "Request callback now",
                "Schedule a call",
                "Email me information",
                "Continue chatting"
            ],
            "actions": ["route_to_sales"]
        }
    
    def _handle_fallback(self, intent_result: Dict, session: ChatbotSession, user_id: Optional[int]) -> Dict:
        """Handle fallback when intent is not recognized"""
        return {
            "message": "I'm not sure I understood that completely. Let me help you with some common topics:\n\n• Product information and pricing\n• Feature demonstrations\n• Scheduling consultations\n• Technical support\n• Contacting our sales team\n\nWhat would you like to know more about?",
            "quick_replies": [
                "Product pricing",
                "Feature overview",
                "Schedule demo",
                "Contact support",
                "Speak to sales"
            ]
        }
    
    def _create_appointment_lead(self, parameters: Dict, user_id: int, session: ChatbotSession):
        """Create lead entry for appointment scheduling"""
        try:
            lead_data = {
                "user_id": user_id,
                "source": "chatbot",
                "status": "appointment_requested",
                "appointment_type": parameters.get("consultation-type", "general"),
                "preferred_date": parameters.get("date"),
                "preferred_time": parameters.get("time"),
                "notes": f"Appointment requested via chatbot in session {session.session_id}"
            }
            lead_crud.create_lead(self.db, **lead_data)
            
        except Exception as e:
            logger.error(f"Error creating appointment lead: {str(e)}")
    
    def _create_qualified_lead(self, parameters: Dict, user_id: int, session: ChatbotSession):
        """Create qualified lead entry"""
        try:
            lead_data = {
                "user_id": user_id,
                "source": "chatbot",
                "status": "qualified",
                "company_size": parameters.get("company-size"),
                "budget": parameters.get("budget"),
                "industry": parameters.get("industry"),
                "notes": f"Qualified lead from chatbot session {session.session_id}"
            }
            lead_crud.create_lead(self.db, **lead_data)
            
        except Exception as e:
            logger.error(f"Error creating qualified lead: {str(e)}")
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get chat history for a session"""
        try:
            session = chatbot_crud.get_session_by_id(self.db, session_id)
            if not session:
                return []
            
            messages = chatbot_crud.get_session_messages(self.db, session.id, limit)
            
            return [
                {
                    "message": msg.message,
                    "is_user": msg.is_user,
                    "timestamp": msg.timestamp.isoformat(),
                    "intent": msg.intent,
                    "confidence": msg.confidence
                }
                for msg in messages
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a chat session"""
        try:
            session = chatbot_crud.get_session_by_id(self.db, session_id)
            if not session:
                return {}
            
            messages = chatbot_crud.get_session_messages(self.db, session.id)
            
            total_messages = len(messages)
            user_messages = len([msg for msg in messages if msg.is_user])
            bot_messages = len([msg for msg in messages if not msg.is_user])
            
            intents = [msg.intent for msg in messages if msg.intent and not msg.is_user]
            unique_intents = list(set(intents))
            
            avg_confidence = sum([msg.confidence for msg in messages if msg.confidence]) / len([msg for msg in messages if msg.confidence]) if messages else 0
            
            return {
                "session_id": session_id,
                "total_messages": total_messages,
                "user_messages": user_messages,
                "bot_messages": bot_messages,
                "unique_intents": unique_intents,
                "average_confidence": round(avg_confidence, 2),
                "session_duration": (datetime.utcnow() - session.started_at).total_seconds() / 60,  # in minutes
                "started_at": session.started_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting session analytics: {str(e)}")
            return {}