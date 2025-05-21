import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.models.lead import Lead, LeadActivity
from app.schemas.lead import LeadCreate, LeadActivityCreate
from app.crud.lead import create_lead, create_lead_activity
from app.services.lead_scoring import LeadScoringService

# Test data
TEST_LEAD = LeadCreate(
    name="Test Lead",
    email="test@example.com",
    phone="+1234567890",
    company="Test Company",
    source="website",
    industry="technology",
    job_title="CTO",
    country="United States",
    city="New York",
    website_visits=10,
    time_spent_on_website=30.5,
    page_views=25,
    custom_fields={"company_size": "enterprise"}
)

@pytest.fixture
def db_lead(db: Session):
    """Create a test lead in the database."""
    return create_lead(db, TEST_LEAD)

@pytest.fixture
def lead_with_activities(db: Session, db_lead: Lead):
    """Create a test lead with activities."""
    # Create activities for the lead
    activities = [
        LeadActivityCreate(
            lead_id=db_lead.id,
            activity_type="email_open",
            description="Email open activity",
            metadata={"email_subject": "Test Email"},
            occurred_at=datetime.utcnow() - timedelta(days=2)
        ),
        LeadActivityCreate(
            lead_id=db_lead.id,
            activity_type="email_click",
            description="Email click activity",
            metadata={"email_subject": "Test Email", "link_url": "https://example.com"},
            occurred_at=datetime.utcnow() - timedelta(days=1)
        ),
        LeadActivityCreate(
            lead_id=db_lead.id,
            activity_type="form_submission",
            description="Form submission activity",
            metadata={"form_name": "Contact Form"},
            occurred_at=datetime.utcnow() - timedelta(hours=12)
        )
    ]
    
    for activity_data in activities:
        create_lead_activity(db, activity_data)
    
    return db_lead

def test_score_lead(db: Session, lead_with_activities: Lead):
    """Test scoring a lead."""
    # Create scoring service
    service = LeadScoringService(db)
    
    # Score the lead
    score = service.score_lead(lead_with_activities.id)
    
    # Test assertions
    assert score is not None
    assert score.lead_id == lead_with_activities.id
    assert 0 <= score.score <= 100
    assert 0 <= score.demographic_score <= 100
    assert 0 <= score.behavioral_score <= 100
    assert 0 <= score.firmographic_score <= 100
    assert 0 <= score.confidence <= 1.0
    assert score.model_version == service.MODEL_VERSION
    
    # Check factors
    assert "demographic" in score.factors
    assert "behavioral" in score.factors
    assert "firmographic" in score.factors

def test_get_industry_relevance(db: Session):
    """Test industry relevance scoring."""
    service = LeadScoringService(db)
    
    # Test exact matches
    assert service._get_industry_relevance("technology") == 1.0
    assert service._get_industry_relevance("finance") == 0.9
    
    # Test partial matches
    assert service._get_industry_relevance("financial services") == 0.9
    assert service._get_industry_relevance("healthcare technology") == 1.0
    
    # Test unknown industry
    assert service._get_industry_relevance("unknown industry") == 0.5

def test_get_job_title_relevance(db: Session):
    """Test job title relevance scoring."""
    service = LeadScoringService(db)
    
    # Test exact matches
    assert service._get_job_title_relevance("CEO") == 1.0
    assert service._get_job_title_relevance("CTO") == 0.95
    
    # Test partial matches
    assert service._get_job_title_relevance("Marketing Manager") == 0.85
    assert service._get_job_title_relevance("Digital Marketing Specialist") == 0.8
    
    # Test unknown job title
    assert service._get_job_title_relevance("unknown role") == 0.5

def test_calculate_confidence(db: Session, lead_with_activities: Lead):
    """Test confidence calculation."""
    service = LeadScoringService(db)
    
    # Get activities
    activities = db.query(LeadActivity).filter(LeadActivity.lead_id == lead_with_activities.id).all()
    
    # Calculate confidence
    confidence = service._calculate_confidence(lead_with_activities, activities)
    
    # Test assertions
    assert 0 <= confidence <= 1.0
    
    # Lead with minimal data should have lower confidence
    minimal_lead = Lead(
        name="Minimal Lead",
        email="minimal@example.com",
        source="website"
    )
    minimal_confidence = service._calculate_confidence(minimal_lead, [])
    
    assert minimal_confidence < confidence