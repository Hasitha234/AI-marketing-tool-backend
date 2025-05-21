from typing import List, Optional, Dict, Any, Union
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.content import Content
from app.schemas.content import ContentCreate, ContentUpdate

# Content CRUD operations
def get_content(db: Session, content_id: int) -> Optional[Content]:
    """Get a single content by ID."""
    return db.query(Content).filter(Content.id == content_id).first()

def get_contents(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    type: Optional[str] = None,
    status: Optional[str] = None,
    created_by_id: Optional[int] = None,
    ai_generated: Optional[bool] = None,
) -> List[Content]:
    """Get multiple contents with filtering."""
    query = db.query(Content)
    
    # Apply filters
    if type:
        query = query.filter(Content.type == type)
    
    if status:
        query = query.filter(Content.status == status)
    
    if created_by_id:
        query = query.filter(Content.created_by_id == created_by_id)
    
    if ai_generated is not None:
        query = query.filter(Content.ai_generated == ai_generated)
    
    # Order by creation date (newest first)
    query = query.order_by(desc(Content.created_at))
    
    return query.offset(skip).limit(limit).all()

def get_content_count(
    db: Session,
    type: Optional[str] = None,
    status: Optional[str] = None,
    created_by_id: Optional[int] = None,
    ai_generated: Optional[bool] = None,
) -> int:
    """Get count of contents with filtering."""
    query = db.query(Content)
    
    # Apply filters
    if type:
        query = query.filter(Content.type == type)
    
    if status:
        query = query.filter(Content.status == status)
    
    if created_by_id:
        query = query.filter(Content.created_by_id == created_by_id)
    
    if ai_generated is not None:
        query = query.filter(Content.ai_generated == ai_generated)
    
    return query.count()

def create_content(db: Session, content_in: ContentCreate, user_id: int) -> Content:
    """Create a new content."""
    # Convert Pydantic model to dict
    content_data = content_in.dict(exclude_unset=True)
    
    # Add user ID
    content_data["created_by_id"] = user_id
    
    # Create Content object
    db_content = Content(**content_data)
    
    # Add to database
    db.add(db_content)
    db.commit()
    db.refresh(db_content)
    
    return db_content

def update_content(
    db: Session, 
    content_id: int, 
    content_in: Union[ContentUpdate, Dict[str, Any]]
) -> Optional[Content]:
    """Update an existing content."""
    # Get the existing content
    db_content = get_content(db, content_id)
    if not db_content:
        return None
    
    # Convert to dict if Pydantic model
    update_data = content_in if isinstance(content_in, dict) else content_in.dict(exclude_unset=True)
    
    # Update the content attributes
    for field, value in update_data.items():
        setattr(db_content, field, value)
    
    # Save to database
    db.add(db_content)
    db.commit()
    db.refresh(db_content)
    
    return db_content

def delete_content(db: Session, content_id: int) -> bool:
    """Delete a content."""
    db_content = get_content(db, content_id)
    if not db_content:
        return False
    
    db.delete(db_content)
    db.commit()
    
    return True

def get_user_contents(
    db: Session, 
    user_id: int,
    skip: int = 0, 
    limit: int = 10,
) -> List[Content]:
    """Get contents created by a specific user (most recent first)."""
    return (
        db.query(Content)
        .filter(Content.created_by_id == user_id)
        .order_by(desc(Content.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )