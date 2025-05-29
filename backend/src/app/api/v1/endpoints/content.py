from typing import Any, List, Optional
import os
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.core.security import get_current_user
from app.core.permissions import allow_all_authenticated
from app.models.user import User
from app.services.content_generation import get_content_generation_service, GenerationConfig
from app.crud import content as content_crud
from app.schemas.content import (
    Content, ContentList, ContentGenerateRequest, ContentGenerateResponse, ContentSaveRequest
)
from app.core.config import settings

router = APIRouter()

def get_content_service():
    """Get content generation service with fine-tuned model."""
    try:
        return get_content_generation_service(
            model_path=settings.TRAINED_MODEL_PATH,
            config=GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_length=1024,
                repetition_penalty=1.1,
                max_new_tokens=512
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize content generation service: {str(e)}"
        )

@router.post("/generate", response_model=ContentGenerateResponse)
def generate_content(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    content_service = Depends(get_content_service),
    request: ContentGenerateRequest
) -> Any:
    """
    Generate content based on prompt and specifications using the fine-tuned Zephyr model.
    This endpoint is optimized for the UI content editor.
    """
    try:
        # Generate content using the fine-tuned model
        generated_text = content_service.generate_content(
            prompt=request.prompt,
            tone=request.tone,
            industry=request.industry,
            channel=request.channel
        )
        
        # Calculate metrics
        word_count = len(generated_text.split())
        character_count = len(generated_text)
        
        # Save to database automatically
        content = content_service.save_generated_content(
            db=db,
            content=generated_text,
            prompt=request.prompt,
            tone=request.tone,
            user_id=current_user.id,
            channel=request.channel,
            industry=request.industry
        )
        
        # Return generated content with metrics
        return ContentGenerateResponse(
            content=generated_text,
            word_count=word_count,
            character_count=character_count
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating content: {str(e)}"
        )

@router.post("/save", response_model=Content)
def save_generated_content(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    content_service = Depends(get_content_service),
    request: ContentSaveRequest
) -> Any:
    """
    Save generated content to the database.
    This endpoint is optimized for the UI's 'Save content' button.
    """
    try:
        # Save the content using the service
        saved_content = content_service.save_generated_content(
            db=db,
            content=request.content,
            prompt=request.prompt,
            tone=request.tone,
            user_id=current_user.id,
            channel=request.channel,
            industry=request.industry
        )
        
        return saved_content
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving content: {str(e)}"
        )

@router.get("/tones", response_model=List[str])
def get_available_tones(
    content_service = Depends(get_content_service),
) -> Any:
    """Get available tone options for the UI."""
    return content_service.TONES

@router.get("/channels", response_model=List[str])
def get_available_channels(
    content_service = Depends(get_content_service),
) -> Any:
    """Get available channel options for the UI."""
    return content_service.CHANNELS

@router.get("/industries", response_model=List[str])
def get_available_industries(
    content_service = Depends(get_content_service),
) -> Any:
    """Get available industry options for the UI."""
    return content_service.INDUSTRIES

@router.get("/model-info", response_model=dict)
def get_model_info(
    content_service = Depends(get_content_service),
) -> Any:
    """Get information about the loaded model."""
    return content_service.get_model_info()

@router.post("/batch-generate", response_model=List[dict])
def batch_generate_content(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    content_service = Depends(get_content_service),
    prompts: List[ContentGenerateRequest] = Body(...)
) -> Any:
    """
    Generate content for multiple prompts in batch.
    Useful for bulk content generation.
    """
    try:
        # Convert requests to format expected by service
        prompt_configs = []
        for request in prompts:
            prompt_configs.append({
                "prompt": request.prompt,
                "tone": request.tone,
                "industry": request.industry,
                "channel": request.channel
            })
        
        # Generate content in batch
        results = content_service.batch_generate_content(prompt_configs)
        
        # Save successful generations to database
        for i, result in enumerate(results):
            if result["status"] == "success":
                try:
                    content_service.save_generated_content(
                        db=db,
                        content=result["generated_content"],
                        prompt=result["prompt"],
                        tone=result["tone"],
                        user_id=current_user.id,
                        channel=result.get("channel"),
                        industry=result.get("industry")
                    )
                except Exception as save_error:
                    result["save_error"] = str(save_error)
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch generation: {str(e)}"
        )

# Original content CRUD endpoints
@router.get("/", response_model=ContentList)
def read_contents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    type: Optional[str] = None,
    status: Optional[str] = None,
    ai_generated: Optional[bool] = None,
) -> Any:
    """
    Retrieve contents with optional filtering.
    """
    # Get contents with filtering
    contents = content_crud.get_contents(
        db=db,
        skip=skip,
        limit=limit,
        type=type,
        status=status,
        created_by_id=current_user.id,
        ai_generated=ai_generated
    )
    
    # Get total count
    total = content_crud.get_content_count(
        db=db,
        type=type,
        status=status,
        created_by_id=current_user.id,
        ai_generated=ai_generated
    )
    
    # Calculate pagination information
    pages = (total + limit - 1) // limit  # Ceiling division
    
    return ContentList(
        items=contents,
        total=total,
        page=skip // limit + 1,
        size=limit,
        pages=pages
    )

@router.get("/{content_id}", response_model=Content)
def read_content(
    *,
    db: Session = Depends(get_db),
    content_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Get content by ID."""
    content = content_crud.get_content(db=db, content_id=content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    return content

@router.put("/{content_id}", response_model=Content)
def update_content(
    *,
    db: Session = Depends(get_db),
    content_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
    content_update: dict = Body(...)
) -> Any:
    """Update content by ID."""
    content = content_crud.get_content(db=db, content_id=content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    # Check if user owns the content
    if content.created_by_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this content"
        )
    
    updated_content = content_crud.update_content(
        db=db, 
        content_id=content_id, 
        content_in=content_update
    )
    
    return updated_content

@router.delete("/{content_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_content(
    *,
    db: Session = Depends(get_db),
    content_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete content by ID."""
    content = content_crud.get_content(db=db, content_id=content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    # Check if user owns the content
    if content.created_by_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this content"
        )
    
    content_crud.delete_content(db=db, content_id=content_id)