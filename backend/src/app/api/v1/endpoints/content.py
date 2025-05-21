from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.core.security import get_current_user
from app.core.permissions import allow_all_authenticated
from app.models.user import User
from app.services.content_generation import ContentGenerationService
from app.crud import content as content_crud
from app.schemas.content import (
    Content,ContentList,
)

router = APIRouter()

# Get content generation service
def get_content_service():
    return ContentGenerationService()

# UI-focused content endpoints
@router.post("/generate", response_model=dict)
def generate_content(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    content_service: ContentGenerationService = Depends(get_content_service),
    prompt: str = Body(...),
    tone: str = Body("Professional"),
    industry: Optional[str] = Body(None),
    channel: Optional[str] = Body(None),
) -> Any:
    """
    Generate content based on prompt and tone.
    This endpoint is optimized for the UI content editor.
    """
    try:
        # Generate content
        generated_text = content_service.generate_content(
            prompt=prompt,
            tone=tone,
            industry=industry,
            channel=channel
        )
        
        # Return generated content
        return {
            "content": generated_text,
            "word_count": len(generated_text.split()),
            "character_count": len(generated_text)
        }
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
    content_service: ContentGenerationService = Depends(get_content_service),
    content: str = Body(...),
    prompt: str = Body(...),
    tone: str = Body("Professional"),
    industry: Optional[str] = Body(None),
    channel: Optional[str] = Body(None),
) -> Any:
    """
    Save generated content to the database.
    This endpoint is optimized for the UI's 'Save content' button.
    """
    try:
        # Save the content
        saved_content = content_service.save_generated_content(
            db=db,
            content=content,
            prompt=prompt,
            tone=tone,
            user_id=current_user.id,
            channel=channel,
            industry=industry
        )
        
        return saved_content
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving content: {str(e)}"
        )

@router.get("/tones", response_model=List[str])
def get_available_tones(
    content_service: ContentGenerationService = Depends(get_content_service),
) -> Any:
    """
    Get available tone options for the UI.
    """
    return content_service.TONES

# Original content CRUD endpoints (simplified)
@router.get("/", response_model=ContentList)
def read_contents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    type: Optional[str] = None,
    status: Optional[str] = None,
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
        created_by_id=current_user.id,  # Only show user's content by default
    )
    
    # Get total count
    total = content_crud.get_content_count(
        db=db,
        type=type,
        status=status,
        created_by_id=current_user.id,
    )
    
    # Calculate pagination information
    pages = (total + limit - 1) // limit  # Ceiling division
    
    return {
        "items": contents,
        "total": total,
        "page": skip // limit + 1,
        "size": limit,
        "pages": pages
    }

@router.get("/{content_id}", response_model=Content)
def read_content(
    *,
    db: Session = Depends(get_db),
    content_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get content by ID.
    """
    content = content_crud.get_content(db=db, content_id=content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    return content

@router.delete("/{content_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_content(
    *,
    db: Session = Depends(get_db),
    content_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> None:
    """
    Delete content.
    Any authenticated user can delete content.
    """
    content = content_crud.get_content(db=db, content_id=content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    content_crud.delete_content(db=db, content_id=content_id)