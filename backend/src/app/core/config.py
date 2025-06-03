import os
import logging
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import PostgresDsn, Field
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.info("Loading environment variables from .env file")

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Marketing Tool"
    API_V1_STR: str = "/api/v1"
    
    # SECURITY
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALGORITHM: str = "HS256"  # JWT algorithm
    
    # POSTGRESQL
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "")
    
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None
    
    @property
    def get_database_uri(self) -> str:
        """Construct database URI from components."""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}?sslmode=require"
    

    # SOCIAL MEDIA API KEYS
    TWITTER_API_KEY: str = os.getenv("TWITTER_API_KEY", "")
    TWITTER_API_SECRET: str = os.getenv("TWITTER_API_SECRET", "")
    FACEBOOK_APP_ID: str = os.getenv("FACEBOOK_APP_ID", "")
    FACEBOOK_APP_SECRET: str = os.getenv("FACEBOOK_APP_SECRET", "")
    LINKEDIN_CLIENT_ID: str = os.getenv("LINKEDIN_CLIENT_ID", "")
    LINKEDIN_CLIENT_SECRET: str = os.getenv("LINKEDIN_CLIENT_SECRET", "")

    # GEMINI API CONFIGURATION
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable is not set")
        raise ValueError("GEMINI_API_KEY environment variable must be set")
    else:
        logger.info("GEMINI_API_KEY environment variable is set")
    
    # GEMINI SPECIFIC SETTINGS
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    GEMINI_TOP_P: float = float(os.getenv("GEMINI_TOP_P", "0.8"))
    GEMINI_TOP_K: int = int(os.getenv("GEMINI_TOP_K", "40"))
    GEMINI_MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))
    
    # RATE LIMITING FOR GEMINI (Free tier: 15 requests per minute)
    GEMINI_RATE_LIMIT_PER_MINUTE: int = int(os.getenv("GEMINI_RATE_LIMIT_PER_MINUTE", "15"))
    GEMINI_RATE_LIMIT_PER_DAY: int = int(os.getenv("GEMINI_RATE_LIMIT_PER_DAY", "1500"))
    GEMINI_TOKENS_PER_DAY: int = int(os.getenv("GEMINI_TOKENS_PER_DAY", "1000000"))

    # Google Cloud / Dialogflow Configuration
    GOOGLE_CLOUD_PROJECT_ID: str = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "ai-marketing-chatbot-460708")
    if not GOOGLE_CLOUD_PROJECT_ID:
        logger.error("GOOGLE_CLOUD_PROJECT_ID environment variable is not set")
        raise ValueError("GOOGLE_CLOUD_PROJECT_ID environment variable must be set")
    else:
        logger.info("GOOGLE_CLOUD_PROJECT_ID environment variable is set")
    
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)
    DIALOGFLOW_LANGUAGE_CODE: str = os.getenv("DIALOGFLOW_LANGUAGE_CODE", "en-US")
    
    # Chatbot Configuration
    CHATBOT_SESSION_TIMEOUT: int = os.getenv("CHATBOT_SESSION_TIMEOUT", 1800)  # 30 minutes
    CHATBOT_MAX_MESSAGE_LENGTH: int = os.getenv("CHATBOT_MAX_MESSAGE_LENGTH", 1000)
    CHATBOT_CONFIDENCE_THRESHOLD: float = os.getenv("CHATBOT_CONFIDENCE_THRESHOLD", 0.5)
    
    # Integration URLs
    CALENDAR_BOOKING_URL: Optional[str] = Field(None, env="CALENDAR_BOOKING_URL")
    CRM_WEBHOOK_URL: Optional[str] = Field(None, env="CRM_WEBHOOK_URL")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000", 
        "http://localhost:8000",
        # Add more frontend origins as needed
        "*"  # This allows any origin - for development only, remove in production!
    ]
    
    class Config:
        case_sensitive = True
        env_file = ".env"

try:
    settings = Settings()
    settings.SQLALCHEMY_DATABASE_URI = settings.get_database_uri
    logger.info("Settings loaded successfully")
except Exception as e:
    logger.error(f"Failed to load settings: {str(e)}")
    raise
