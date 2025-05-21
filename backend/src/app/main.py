from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.v1.router import api_router
from app.core.config import settings
from app.utils.logging import get_logger

# Initialize logger
logger = get_logger("main")

app = FastAPI(
    title="AI Marketing Tool API",
    description="""
    Advanced AI-driven Marketing Tool API with intelligent lead scoring and management.
    
    ## Features
    
    * **Hybrid Lead Scoring**: Combines rule-based and machine learning approaches
    * **Lead Management**: Track and analyze leads throughout the marketing funnel
    * **Data Visualization**: Comprehensive dashboard with lead scoring analytics
    * **CSV Import/Export**: Bulk process leads and export scoring results
    * **API Integrations**: Connect with external systems via RESTful API
    
    The API provides endpoints for lead management, lead scoring, analytics, and more.
    """,
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    logger.info(f"Configuring CORS with origins: {settings.BACKEND_CORS_ORIGINS}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)
logger.info(f"API router included with prefix: {settings.API_V1_STR}")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Welcome to AI Marketing Tool API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "online"
    }

if __name__ == "__main__":
    logger.info("Starting application server")
    # For development with self-signed certificate
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    #     ssl_keyfile="./certs/key.pem",
    #     ssl_certfile="./certs/cert.pem",
    )