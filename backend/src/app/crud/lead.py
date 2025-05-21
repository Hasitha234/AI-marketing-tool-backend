from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.lead import Lead, LeadScore, LeadActivity
from app.schemas.lead import LeadCreate, LeadUpdate, LeadActivityCreate

def get_lead(db: Session, lead_id: int) -> Optional[Lead]:
    """Get a single lead by ID."""
    return db.query(Lead).filter(Lead.id == lead_id).first()

def get_lead_by_email(db: Session, email: str) -> Optional[Lead]:
    """Get a single lead by email."""
    return db.query(Lead).filter(Lead.email == email).first()

def get_leads(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    status: Optional[str] = None,
    is_converted: Optional[bool] = None,
    min_score: Optional[float] = None,
    source: Optional[str] = None
) -> List[Lead]:
    """
    Get multiple leads with optional filtering.
    """
    query = db.query(Lead)
    
    # Apply filters
    if status:
        query = query.filter(Lead.status == status)
    
    if is_converted is not None:
        query = query.filter(Lead.is_converted == is_converted)
    
    if source:
        query = query.filter(Lead.source == source)
    
    # Filter by score if specified
    if min_score is not None:
        # This is a more complex query joining with LeadScore
        query = query.join(LeadScore).filter(LeadScore.score >= min_score)
    
    # Return paginated results
    return query.offset(skip).limit(limit).all()

def get_lead_count(
    db: Session,
    status: Optional[str] = None,
    is_converted: Optional[bool] = None,
    min_score: Optional[float] = None,
    source: Optional[str] = None
) -> int:
    """
    Get count of leads with optional filtering.
    """
    query = db.query(Lead)
    
    # Apply filters
    if status:
        query = query.filter(Lead.status == status)
    
    if is_converted is not None:
        query = query.filter(Lead.is_converted == is_converted)
    
    if source:
        query = query.filter(Lead.source == source)
    
    # Filter by score if specified
    if min_score is not None:
        query = query.join(LeadScore).filter(LeadScore.score >= min_score)
    
    return query.count()

def create_lead(db: Session, lead_in: LeadCreate) -> Lead:
    """Create a new lead."""
    # Convert Pydantic model to dict, excluding unset fields
    lead_data = lead_in.dict(exclude_unset=True)
    
    # Convert tags and custom_fields to JSON if present
    if "tags" in lead_data and lead_data["tags"]:
        lead_data["tags"] = jsonable_encoder(lead_data["tags"])
    
    if "custom_fields" in lead_data and lead_data["custom_fields"]:
        lead_data["custom_fields"] = jsonable_encoder(lead_data["custom_fields"])
    
    # Create Lead object
    db_lead = Lead(**lead_data)
    
    # Add to database
    db.add(db_lead)
    db.commit()
    db.refresh(db_lead)
    
    return db_lead

def update_lead(
    db: Session, 
    lead_id: int, 
    lead_in: Union[LeadUpdate, Dict[str, Any]]
) -> Optional[Lead]:
    """Update an existing lead."""
    # Get the existing lead
    db_lead = get_lead(db, lead_id)
    if not db_lead:
        return None
    
    # Convert to dict if Pydantic model
    update_data = lead_in if isinstance(lead_in, dict) else lead_in.dict(exclude_unset=True)
    
    # Update lead status if is_converted is being set to True
    if "is_converted" in update_data and update_data["is_converted"] and not db_lead.is_converted:
        update_data["converted_at"] = datetime.utcnow()
        update_data["status"] = "converted"
    
    # Handle JSON fields
    if "tags" in update_data and update_data["tags"] is not None:
        update_data["tags"] = jsonable_encoder(update_data["tags"])
    
    if "custom_fields" in update_data and update_data["custom_fields"] is not None:
        # Merge existing custom fields with new ones
        if db_lead.custom_fields:
            current_fields = db_lead.custom_fields.copy()
            current_fields.update(update_data["custom_fields"])
            update_data["custom_fields"] = jsonable_encoder(current_fields)
        else:
            update_data["custom_fields"] = jsonable_encoder(update_data["custom_fields"])
    
    # Update the lead attributes
    for field, value in update_data.items():
        setattr(db_lead, field, value)
    
    # Save to database
    db.add(db_lead)
    db.commit()
    db.refresh(db_lead)
    
    return db_lead

def delete_lead(db: Session, lead_id: int) -> bool:
    """Delete a lead."""
    db_lead = get_lead(db, lead_id)
    if not db_lead:
        return False
    
    db.delete(db_lead)
    db.commit()
    
    return True

def get_lead_scores(db: Session, lead_id: int, limit: int = 10) -> List[LeadScore]:
    """Get lead scores for a specific lead."""
    return (
        db.query(LeadScore)
        .filter(LeadScore.lead_id == lead_id)
        .order_by(desc(LeadScore.created_at))
        .limit(limit)
        .all()
    )

def get_latest_lead_score(db: Session, lead_id: int) -> Optional[LeadScore]:
    """Get the latest lead score for a specific lead."""
    return (
        db.query(LeadScore)
        .filter(LeadScore.lead_id == lead_id)
        .order_by(desc(LeadScore.created_at))
        .first()
    )

def create_lead_activity(db: Session, activity_in: LeadActivityCreate) -> LeadActivity:
    """Create a new lead activity."""
    # Convert Pydantic model to dict
    activity_data = activity_in.dict(exclude_unset=True)
    
    # Set occurred_at if not provided
    if "occurred_at" not in activity_data or not activity_data["occurred_at"]:
        activity_data["occurred_at"] = datetime.utcnow()
    
    # Map metadata to activity_metadata field
    if "metadata" in activity_data:
        metadata = activity_data.pop("metadata")
        activity_data["activity_metadata"] = jsonable_encoder(metadata)
    
    # Create LeadActivity object
    db_activity = LeadActivity(**activity_data)
    
    # Add to database
    db.add(db_activity)
    db.commit()
    db.refresh(db_activity)
    
    # Update lead's last_activity timestamp
    lead = get_lead(db, db_activity.lead_id)
    if lead:
        lead.last_activity = db_activity.occurred_at
        db.add(lead)
        db.commit()
    
    return db_activity

def get_lead_activities(
    db: Session, 
    lead_id: int, 
    skip: int = 0, 
    limit: int = 50,
    activity_type: Optional[str] = None
) -> List[LeadActivity]:
    """Get activities for a specific lead."""
    query = db.query(LeadActivity).filter(LeadActivity.lead_id == lead_id)
    
    if activity_type:
        query = query.filter(LeadActivity.activity_type == activity_type)
    
    return query.order_by(desc(LeadActivity.occurred_at)).offset(skip).limit(limit).all()