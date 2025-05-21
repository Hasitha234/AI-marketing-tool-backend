from fastapi.testclient import TestClient
import pytest
from sqlalchemy.orm import Session

from app.core.config import settings
from app.main import app
from app.models.lead import Lead
from app.schemas.lead import LeadCreate
from app.crud.lead import create_lead
from app.services.lead_scoring import LeadScoringService

client = TestClient(app)

# Test data
TEST_LEAD = LeadCreate(
    name="API Test Lead",
    email="apitest@example.com",
    phone="+1234567890",
    company="API Test Company",
    source="website",
    industry="technology",
    job_title="CTO"
)

@pytest.fixture
def db_lead(db: Session):
    """Create a test lead in the database."""
    return create_lead(db, TEST_LEAD)

def get_token_headers(client: TestClient, email: str, password: str):
    """Get authentication token headers."""
    login_data = {
        "username": email,
        "password": password,
    }
    r = client.post(
        f"{settings.API_V1_STR}/auth/login",
        data=login_data,
    )
    response = r.json()
    auth_token = response["access_token"]
    return {"Authorization": f"Bearer {auth_token}"}

def test_create_lead(admin_token_headers):
    """Test creating a new lead."""
    data = {
        "name": "New Test Lead",
        "email": "newlead@example.com",
        "phone": "+1987654321",
        "company": "New Test Company",
        "source": "social_media",
        "industry": "finance",
        "job_title": "CFO"
    }
    
    response = client.post(
        f"{settings.API_V1_STR}/leads/",
        headers=admin_token_headers,
        json=data,
    )
    
    assert response.status_code == 200
    content = response.json()
    assert content["email"] == data["email"]
    assert content["name"] == data["name"]
    assert content["source"] == data["source"]
    assert "id" in content
    assert "scores" in content

def test_read_leads(admin_token_headers):
    """Test reading leads with pagination."""
    response = client.get(
        f"{settings.API_V1_STR}/leads/",
        headers=admin_token_headers,
    )
    
    assert response.status_code == 200
    content = response.json()
    assert "items" in content
    assert "total" in content
    assert "page" in content
    assert "size" in content
    assert "pages" in content
    assert isinstance(content["items"], list)

def test_read_lead(db_lead: Lead, admin_token_headers):
    """Test reading a specific lead."""
    response = client.get(
        f"{settings.API_V1_STR}/leads/{db_lead.id}",
        headers=admin_token_headers,
    )
    
    assert response.status_code == 200
    content = response.json()
    assert content["id"] == db_lead.id
    assert content["name"] == db_lead.name
    assert content["email"] == db_lead.email

def test_update_lead(db_lead: Lead, admin_token_headers):
    """Test updating a lead."""
    data = {
        "industry": "healthcare",
        "job_title": "Director of Technology",
        "custom_fields": {
            "company_size": "enterprise",
            "budget": "high"
        }
    }
    
    response = client.put(
        f"{settings.API_V1_STR}/leads/{db_lead.id}",
        headers=admin_token_headers,
        json=data,
    )
    
    assert response.status_code == 200
    content = response.json()
    assert content["id"] == db_lead.id
    assert content["industry"] == data["industry"]
    assert content["job_title"] == data["job_title"]
    assert "custom_fields" in content
    assert content["custom_fields"]["company_size"] == data["custom_fields"]["company_size"]
    assert content["custom_fields"]["budget"] == data["custom_fields"]["budget"]

def test_get_lead_score(db_lead: Lead, admin_token_headers, db: Session):
    """Test getting a lead's score."""
    # First ensure the lead has a score
    scoring_service = LeadScoringService(db)
    scoring_service.score_lead(db_lead.id)
    
    response = client.get(
        f"{settings.API_V1_STR}/leads/{db_lead.id}/score",
        headers=admin_token_headers,
    )
    
    assert response.status_code == 200
    content = response.json()
    assert content["lead_id"] == db_lead.id
    assert 0 <= content["score"] <= 100
    assert 0 <= content["demographic_score"] <= 100
    assert 0 <= content["behavioral_score"] <= 100
    assert 0 <= content["firmographic_score"] <= 100
    assert 0 <= content["confidence"] <= 1.0
    assert "factors" in content
    assert "demographic" in content["factors"]
    assert "behavioral" in content["factors"]
    assert "firmographic" in content["factors"]

def test_calculate_lead_score(db_lead: Lead, admin_token_headers):
    """Test calculating a new score for a lead."""
    response = client.post(
        f"{settings.API_V1_STR}/leads/{db_lead.id}/score",
        headers=admin_token_headers,
    )
    
    assert response.status_code == 200
    content = response.json()
    assert content["lead_id"] == db_lead.id
    assert 0 <= content["score"] <= 100

def test_add_lead_activity(db_lead: Lead, admin_token_headers):
    """Test adding an activity to a lead."""
    data = {
        "lead_id": db_lead.id,
        "activity_type": "email_open",
        "description": "Test email open",
        "metadata": {
            "email_subject": "Test Subject",
            "campaign_id": "test-campaign"
        }
    }
    
    response = client.post(
        f"{settings.API_V1_STR}/leads/{db_lead.id}/activity",
        headers=admin_token_headers,
        json=data,
    )
    
    assert response.status_code == 200
    content = response.json()
    assert content["lead_id"] == db_lead.id
    assert content["activity_type"] == data["activity_type"]
    assert content["description"] == data["description"]
    assert content["metadata"] == data["metadata"]
    assert "id" in content
    assert "occurred_at" in content
    assert "created_at" in content

def test_get_lead_activities(db_lead: Lead, admin_token_headers, db: Session):
    """Test getting activities for a lead."""
    # First add an activity
    data = {
        "lead_id": db_lead.id,
        "activity_type": "website_visit",
        "description": "Test website visit",
        "metadata": {
            "page_url": "https://example.com/test",
            "time_spent": 120
        }
    }
    
    client.post(
        f"{settings.API_V1_STR}/leads/{db_lead.id}/activity",
        headers=admin_token_headers,
        json=data,
    )
    
    response = client.get(
        f"{settings.API_V1_STR}/leads/{db_lead.id}/activities",
        headers=admin_token_headers,
    )
    
    assert response.status_code == 200
    content = response.json()
    assert isinstance(content, list)
    assert len(content) > 0
    assert content[0]["lead_id"] == db_lead.id