from typing import Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import get_current_user, oauth2_scheme
from app.db.session import SessionLocal
from app.models.user import User
from app.schemas.token import TokenPayload
from app.services.lead_scoring import LeadScoringService
from app.services.social_media import SocialMediaService

def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_lead_scoring_service(db: Session = Depends(get_db)) -> LeadScoringService:
    """Get lead scoring service."""
    return LeadScoringService(db)

def get_social_media_service() -> SocialMediaService:
    """Get social media service."""
    return SocialMediaService()