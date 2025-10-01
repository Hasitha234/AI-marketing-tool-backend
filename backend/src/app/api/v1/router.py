from fastapi import APIRouter

from app.api.v1.endpoints import auth, users, leads, analytics, social, content, chatbot, health, pricing

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(leads.router, prefix="/leads", tags=["leads"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(social.router, prefix="/social", tags=["social_media"])
api_router.include_router(content.router, prefix="/content", tags=["content"])
api_router.include_router(chatbot.router, prefix="/chatbot", tags=["chatbot"])
api_router.include_router(pricing.router, prefix="/pricing", tags=["pricing"])
api_router.include_router(health.router, tags=["health"])
