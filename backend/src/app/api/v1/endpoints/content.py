from typing import Any, List, Optional, Dict
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime

from app.api.dependencies import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.content_generation import (
    get_gemini_content_generation_service, 
    GeminiGenerationConfig
)
from app.crud import content as content_crud
from app.schemas.content import (
    Content, ContentGenerateResponse
)
from app.core.config import settings

# Pydantic models for request/response
class GeminiContentGenerateRequest(BaseModel):
    keywords: str = Field(..., min_length=1, max_length=500, description="Keywords for content generation")
    content_type: str = Field(default="blog", description="Type of content to generate")
    tone: str = Field(default="professional", description="Tone of the content")
    industry: Optional[str] = Field(None, description="Target industry")
    additional_context: Optional[str] = Field(None, max_length=1000, description="Additional context for generation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "keywords": "artificial intelligence in healthcare",
                "content_type": "blog",
                "tone": "professional",
                "industry": "healthcare",
                "additional_context": "Focus on recent AI developments in medical diagnosis"
            }
        }

class GeminiBatchGenerateRequest(BaseModel):
    prompts: List[GeminiContentGenerateRequest] = Field(..., max_items=10)
    
class GeminiContentSaveRequest(BaseModel):
    content: str = Field(..., min_length=10)
    keywords: str = Field(..., min_length=1, max_length=500)
    content_type: str = Field(default="blog")
    tone: str = Field(default="professional")
    industry: Optional[str] = None
    scheduled_date: Optional[str] = None  # For UI schedule feature
    additional_metadata: Optional[Dict] = None

class GeminiConfigurationRequest(BaseModel):
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=100)
    max_output_tokens: Optional[int] = Field(None, ge=100, le=2048)

class ContentAnalysisResponse(BaseModel):
    word_count: int
    character_count: int
    readability_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    keywords_density: Dict[str, float] = {}
    suggestions: List[str] = []

router = APIRouter()

def get_gemini_service():
    """Get Gemini content generation service with error handling."""
    try:
        return get_gemini_content_generation_service()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize Gemini service: {str(e)}"
        )

@router.post("/generate", response_model=ContentGenerateResponse)
async def generate_content_with_gemini(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    gemini_service = Depends(get_gemini_service),
    request: GeminiContentGenerateRequest
) -> Any:
    """
    Generate content using Google's Gemini API.
    Supports the UI's content generation with keywords, tone, and content type.
    """
    try:
        # Generate content using Gemini
        generated_text = await gemini_service.generate_content_async(
            keywords=request.keywords,
            content_type=request.content_type,
            tone=request.tone,
            industry=request.industry,
            additional_context=request.additional_context
        )
        
        # Calculate metrics
        word_count = len(generated_text.split())
        character_count = len(generated_text)
        
        # Auto-save to database
        saved_content = gemini_service.save_generated_content(
            db=db,
            content=generated_text,
            prompt=request.keywords,
            tone=request.tone,
            user_id=current_user.id,
            channel=request.content_type,
            industry=request.industry,
            additional_metadata={
                "additional_context": request.additional_context,
                "auto_generated": True
            }
        )
        
        return ContentGenerateResponse(
            content=generated_text,
            word_count=word_count,
            character_count=character_count
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating content with Gemini: {str(e)}"
        )

@router.post("/generate-stream")
async def generate_content_stream(
    *,
    current_user: User = Depends(get_current_user),
    gemini_service = Depends(get_gemini_service),
    request: GeminiContentGenerateRequest
):
    """
    Generate content with streaming response for better UX.
    """
    async def generate_stream():
        try:
            # This is a simplified streaming approach
            # In a real implementation, you'd use Gemini's streaming capabilities
            content = await gemini_service.generate_content_async(
                keywords=request.keywords,
                content_type=request.content_type,
                tone=request.tone,
                industry=request.industry,
                additional_context=request.additional_context
            )
            
            # Simulate streaming by breaking content into chunks
            words = content.split()
            chunk_size = 5
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                yield f"data: {chunk} \n\n"
                await asyncio.sleep(0.1)  # Small delay for realistic streaming
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")

@router.post("/save", response_model=Content)
async def save_generated_content(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    gemini_service = Depends(get_gemini_service),
    request: GeminiContentSaveRequest
) -> Any:
    """
    Save generated content to the database.
    This endpoint is used by the UI's 'Save content' button.
    """
    try:
        additional_metadata = request.additional_metadata or {}
        
        # Add scheduled date to metadata if provided
        if request.scheduled_date:
            additional_metadata["scheduled_date"] = request.scheduled_date
        
        saved_content = gemini_service.save_generated_content(
            db=db,
            content=request.content,
            prompt=request.keywords,
            tone=request.tone,
            user_id=current_user.id,
            channel=request.content_type,
            industry=request.industry,
            additional_metadata=additional_metadata
        )
        
        return saved_content
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving content: {str(e)}"
        )

@router.post("/batch-generate", response_model=List[Dict[str, Any]])
async def batch_generate_content(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    gemini_service = Depends(get_gemini_service),
    request: GeminiBatchGenerateRequest,
    background_tasks: BackgroundTasks
) -> Any:
    """
    Generate content for multiple prompts in batch.
    Useful for bulk content generation campaigns.
    """
    try:
        # Convert requests to format expected by service
        prompt_configs = []
        for prompt_request in request.prompts:
            prompt_configs.append({
                "keywords": prompt_request.keywords,
                "content_type": prompt_request.content_type,
                "tone": prompt_request.tone,
                "industry": prompt_request.industry,
                "additional_context": prompt_request.additional_context
            })
        
        # Generate content in batch asynchronously
        results = await gemini_service.batch_generate_content_async(prompt_configs)
        
        # Save successful generations to database in background
        def save_batch_results():
            for result in results:
                if result["status"] == "success":
                    try:
                        gemini_service.save_generated_content(
                            db=db,
                            content=result["generated_content"],
                            prompt=result["keywords"],
                            tone=result["tone"],
                            user_id=current_user.id,
                            channel=result["content_type"],
                            industry=result.get("industry"),
                            additional_metadata={
                                "batch_generation": True,
                                "batch_timestamp": result["timestamp"]
                            }
                        )
                    except Exception as save_error:
                        result["save_error"] = str(save_error)
        
        background_tasks.add_task(save_batch_results)
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch generation: {str(e)}"
        )

@router.post("/analyze", response_model=ContentAnalysisResponse)
async def analyze_content(
    *,
    content: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Analyze generated content for metrics and suggestions.
    """
    try:
        # Basic analysis
        words = content.split()
        word_count = len(words)
        character_count = len(content)
        
        # Simple keyword density calculation
        word_freq = {}
        for word in words:
            word_lower = word.lower().strip('.,!?";')
            if len(word_lower) > 3:  # Only count meaningful words
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # Calculate density (frequency / total words)
        keywords_density = {
            word: (freq / word_count) * 100 
            for word, freq in word_freq.items() 
            if freq > 1  # Only include words that appear more than once
        }
        
        # Sort by density and take top 10
        keywords_density = dict(
            sorted(keywords_density.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        # Generate basic suggestions
        suggestions = []
        if word_count < 50:
            suggestions.append("Consider expanding the content for better engagement")
        if word_count > 1000:
            suggestions.append("Content might be too long; consider breaking into sections")
        if not any(char in content for char in '.!?'):
            suggestions.append("Add more punctuation for better readability")
        
        return ContentAnalysisResponse(
            word_count=word_count,
            character_count=character_count,
            keywords_density=keywords_density,
            suggestions=suggestions
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing content: {str(e)}"
        )

@router.get("/tones", response_model=List[str])
async def get_available_tones(
    gemini_service = Depends(get_gemini_service),
) -> Any:
    """Get available tone options for the UI."""
    return gemini_service.TONES

@router.get("/content-types", response_model=List[str])
async def get_available_content_types(
    gemini_service = Depends(get_gemini_service),
) -> Any:
    """Get available content type options for the UI."""
    return gemini_service.CHANNELS

@router.get("/industries", response_model=List[str])
async def get_available_industries(
    gemini_service = Depends(get_gemini_service),
) -> Any:
    """Get available industry options for the UI."""
    return gemini_service.INDUSTRIES

@router.get("/model-info", response_model=Dict[str, Any])
async def get_model_info(
    gemini_service = Depends(get_gemini_service),
) -> Any:
    """Get information about the Gemini model."""
    return gemini_service.get_model_info()

@router.post("/configure", response_model=Dict[str, str])
async def configure_generation_parameters(
    *,
    current_user: User = Depends(get_current_user),
    config_request: GeminiConfigurationRequest
) -> Any:
    """
    Configure generation parameters for the current session.
    Note: This creates a new service instance with updated config.
    """
    try:
        # Create new configuration
        new_config = GeminiGenerationConfig()
        
        if config_request.temperature is not None:
            new_config.temperature = config_request.temperature
        if config_request.top_p is not None:
            new_config.top_p = config_request.top_p
        if config_request.top_k is not None:
            new_config.top_k = config_request.top_k
        if config_request.max_output_tokens is not None:
            new_config.max_output_tokens = config_request.max_output_tokens
        
        # For simplicity, we'll return success message
        # In a production environment, you might want to store user preferences
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "applied_config": {
                "temperature": new_config.temperature,
                "top_p": new_config.top_p,
                "top_k": new_config.top_k,
                "max_output_tokens": new_config.max_output_tokens
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error updating configuration: {str(e)}"
        )

@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    gemini_service = Depends(get_gemini_service),
) -> Any:
    """Check the health status of the Gemini service."""
    try:
        is_healthy = await gemini_service.health_check_async()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "gemini",
            "timestamp": datetime.now().isoformat(),
            "model_info": gemini_service.get_model_info()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "gemini",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# Template-specific endpoints for backward compatibility
@router.post("/generate-template", response_model=ContentGenerateResponse)
async def generate_content_template_format(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    gemini_service = Depends(get_gemini_service),
    keywords: str = Body(..., embed=True),
    content_type: str = Body("blog", embed=True),
    tone: str = Body("professional", embed=True)
) -> Any:
    """
    Generate content using template format for backward compatibility.
    """
    request = GeminiContentGenerateRequest(
        keywords=keywords,
        content_type=content_type,
        tone=tone
    )
    
    return await generate_content_with_gemini(
        db=db,
        current_user=current_user,
        gemini_service=gemini_service,
        request=request
    )