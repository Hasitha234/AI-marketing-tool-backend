import sys
import os
import random
from datetime import datetime, timedelta
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.lead import Lead, LeadActivity
from app.schemas.lead import LeadCreate, LeadActivityCreate
from app.crud.lead import create_lead, create_lead_activity
from app.services.lead_scoring import LeadScoringService

# Sample data for generating leads
FIRST_NAMES = ["John", "Jane", "Michael", "Sara", "David", "Emily", "James", "Emma", "Robert", "Olivia"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor"]
COMPANIES = ["Acme Inc", "TechCorp", "Globex", "Initech", "Umbrella Corp", "Stark Industries", "Wayne Enterprises", "Cyberdyne Systems", "Massive Dynamic", "Oscorp"]
DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "example.com", "company.com"]
SOURCES = ["website", "referral", "email_campaign", "social_media", "webinar", "trade_show", "cold_call"]
INDUSTRIES = ["technology", "finance", "healthcare", "retail", "manufacturing", "education", "media", "real_estate", "construction", "consulting"]
JOB_TITLES = ["CEO", "CTO", "CMO", "Director", "Manager", "VP of Marketing", "VP of Sales", "Analyst", "Consultant", "Specialist"]
COUNTRIES = ["United States", "United Kingdom", "Canada", "Australia", "Germany", "France", "Japan", "India", "Brazil", "Singapore"]
CITIES = {
    "United States": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
    "United Kingdom": ["London", "Manchester", "Birmingham", "Glasgow", "Liverpool"],
    "Canada": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa"],
    "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"],
    "Germany": ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt"],
    "France": ["Paris", "Marseille", "Lyon", "Toulouse", "Nice"],
    "Japan": ["Tokyo", "Osaka", "Yokohama", "Nagoya", "Sapporo"],
    "India": ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai"],
    "Brazil": ["São Paulo", "Rio de Janeiro", "Brasília", "Salvador", "Fortaleza"],
    "Singapore": ["Singapore"],
}
COMPANY_SIZES = ["small", "mid-market", "enterprise"]
ACTIVITY_TYPES = ["email_open", "email_click", "form_submission", "website_visit", "webinar_attendance", "download"]

def random_date(start_date, end_date):
    """Generate a random date between start_date and end_date."""
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + timedelta(days=random_number_of_days)

def generate_fake_lead():
    """Generate a fake lead with random data."""
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    email = f"{first_name.lower()}.{last_name.lower()}@{random.choice(DOMAINS)}"
    
    country = random.choice(COUNTRIES)
    city = random.choice(CITIES.get(country, ["Unknown"]))
    
    company = random.choice(COMPANIES)
    industry = random.choice(INDUSTRIES)
    job_title = random.choice(JOB_TITLES)
    
    # Generate custom fields
    custom_fields = {
        "company_size": random.choice(COMPANY_SIZES),
        "budget": random.choice(["low", "medium", "high"]),
        "timeline": random.choice(["immediate", "1-3 months", "3-6 months", "6+ months"]),
    }
    
    # Generate random engagement metrics
    website_visits = random.randint(0, 20)
    time_spent = round(random.uniform(0, 120), 2) if website_visits > 0 else 0
    page_views = random.randint(0, 50) if website_visits > 0 else 0
    
    # Create lead data
    lead_data = LeadCreate(
        name=f"{first_name} {last_name}",
        email=email,
        phone=f"+1{random.randint(1000000000, 9999999999)}",
        company=company,
        source=random.choice(SOURCES),
        campaign=f"campaign_{random.randint(1, 10)}",
        industry=industry,
        job_title=job_title,
        city=city,
        country=country,
        website_visits=website_visits,
        time_spent_on_website=time_spent,
        page_views=page_views,
        status=random.choice(["new", "contacted", "qualified"]),
        custom_fields=custom_fields,
    )
    
    return lead_data

def generate_fake_activities(lead_id, count=5):
    """Generate fake activities for a lead."""
    activities = []
    
    # Start date range (30 days ago to now)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    
    for _ in range(count):
        activity_type = random.choice(ACTIVITY_TYPES)
        
        # Generate activity-specific metadata
        metadata = {}
        if activity_type == "email_open":
            metadata = {
                "email_subject": f"Email subject {random.randint(1, 100)}",
                "campaign_id": f"campaign_{random.randint(1, 10)}",
            }
        elif activity_type == "email_click":
            metadata = {
                "email_subject": f"Email subject {random.randint(1, 100)}",
                "campaign_id": f"campaign_{random.randint(1, 10)}",
                "link_url": f"https://example.com/link/{random.randint(1, 100)}",
            }
        elif activity_type == "form_submission":
            metadata = {
                "form_id": f"form_{random.randint(1, 10)}",
                "form_name": f"Form {random.randint(1, 10)}",
                "fields_completed": random.randint(3, 10),
            }
        elif activity_type == "website_visit":
            metadata = {
                "page_url": f"https://example.com/page/{random.randint(1, 100)}",
                "time_spent": random.randint(10, 300),
                "referrer": random.choice(["google", "direct", "email", "social"]),
            }
        
        # Create activity data
        activity_data = LeadActivityCreate(
            lead_id=lead_id,
            activity_type=activity_type,
            description=f"{activity_type.replace('_', ' ').title()} activity",
            metadata=metadata,
            occurred_at=random_date(start_date, end_date),
        )
        
        activities.append(activity_data)
    
    # Sort activities by occurred_at
    activities.sort(key=lambda x: x.occurred_at)
    
    return activities

def seed_leads(db: Session, count=50):
    """Seed the database with fake leads and their activities."""
    leads_created = []
    
    for i in range(count):
        # Generate and create lead
        lead_data = generate_fake_lead()
        try:
            lead = create_lead(db, lead_data)
            print(f"Created lead {i+1}/{count}: {lead.name} ({lead.email})")
            
            # Generate and create activities for the lead
            activity_count = random.randint(0, 10)
            if activity_count > 0:
                activities = generate_fake_activities(lead.id, activity_count)
                for activity_data in activities:
                    activity = create_lead_activity(db, activity_data)
                    print(f"  - Created activity: {activity.activity_type}")
            
            leads_created.append(lead)
        except Exception as e:
            print(f"Error creating lead: {e}")
            continue
    
    # Score all leads
    scoring_service = LeadScoringService(db)
    for lead in leads_created:
        score = scoring_service.score_lead(lead.id)
        print(f"Scored lead {lead.name}: {score.score:.2f} (confidence: {score.confidence:.2f})")
    
    print(f"Successfully seeded {len(leads_created)} leads with activities and scores")

if __name__ == "__main__":
    db = SessionLocal()
    try:
        seed_count = int(sys.argv[1]) if len(sys.argv) > 1 else 50
        seed_leads(db, count=seed_count)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()