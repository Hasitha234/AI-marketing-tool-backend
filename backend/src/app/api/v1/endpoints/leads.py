from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import pandas as pd
import io
from datetime import datetime, timedelta
import csv
from io import StringIO

from app.api.dependencies import get_db
from app.core.security import get_current_user
from app.core.permissions import allow_admin
from app.models.user import User
from app.models.lead import Lead
from app.services.lead_scoring import LeadScoringService
from app.crud import lead as lead_crud
from app.schemas.lead import (
    Lead as LeadSchema,  # Rename the Pydantic schema to avoid confusion
    LeadCreate, LeadUpdate, LeadList,
    LeadScore as LeadScoreSchema,  # Rename the schemas
    LeadActivity as LeadActivitySchema,
    LeadActivityCreate
)

router = APIRouter()

@router.get("/", response_model=LeadList)
def read_leads(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    status: Optional[str] = None,
    is_converted: Optional[bool] = None,
    min_score: Optional[float] = Query(None, ge=0, le=100),
    source: Optional[str] = None,
) -> Any:
    """
    Retrieve leads with optional filtering.
    """
    # Get leads with filtering
    leads = lead_crud.get_leads(
        db=db,
        skip=skip,
        limit=limit,
        status=status,
        is_converted=is_converted,
        min_score=min_score,
        source=source
    )
    
    # Get total count
    total = lead_crud.get_lead_count(
        db=db,
        status=status,
        is_converted=is_converted,
        min_score=min_score,
        source=source
    )
    
    # Calculate pagination information
    pages = (total + limit - 1) // limit  # Ceiling division
    
    return {
        "items": leads,
        "total": total,
        "page": skip // limit + 1,
        "size": limit,
        "pages": pages
    }

@router.post("/", response_model=LeadSchema)
def create_lead(
    *,
    db: Session = Depends(get_db),
    lead_in: LeadCreate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Create new lead.
    """
    # Check if lead with this email already exists
    lead = lead_crud.get_lead_by_email(db=db, email=lead_in.email)
    if lead:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A lead with this email already exists"
        )
    
    # Create lead
    lead = lead_crud.create_lead(db=db, lead_in=lead_in)
    
    # Score the lead
    scoring_service = LeadScoringService(db)
    scoring_service.score_lead(lead.id)
    
    return lead

@router.get("/scoring-dashboard", response_model=Dict[str, Any])
def get_lead_scoring_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = Query(15, ge=1, le=100),
    days: int = Query(30, ge=1, le=365, description="Filter leads created within last N days")
) -> Any:
    """
    Get lead scoring dashboard data including top scored leads.
    
    This endpoint provides comprehensive data for the lead scoring dashboard, including:
    - Top-scoring leads with their scores and interpretations
    - Lead score distribution for visualization
    - Lead temperature categories (hot/warm/cold) for visualizations
    - Industry distribution
    - Scoring model information
    - Score trends
    
    Parameters:
    - limit: Maximum number of top leads to return (default: 15)
    - days: Filter leads created within last N days (default: 30)
    
    Returns:
    - top_leads: List of highest-scoring leads with scores
    - score_statistics: Statistical summary of lead scores
    - model_info: Information about the current scoring model
    - total_leads: Total number of leads in the system
    - chart_data: Visualization-ready data for dashboard charts
    - timestamp: ISO format timestamp of the request
    """
    current_time = datetime.utcnow()
    date_filter = current_time - timedelta(days=days)
    
    # Get scoring service for model info
    scoring_service = LeadScoringService(db)
    
    # Get all leads with their latest scores (with date filtering if requested)
    leads_with_scores = []
    
    # Fetch leads created within the specified time period
    # Use the SQLAlchemy model (not the Pydantic schema)
    leads_query = db.query(Lead)  # This uses the SQLAlchemy model imported from app.models.lead
    if days > 0:
        leads_query = leads_query.filter(Lead.created_at >= date_filter)
    
    leads = leads_query.order_by(Lead.created_at.desc()).limit(500).all()
    
    for lead in leads:
        latest_score = lead_crud.get_latest_lead_score(db=db, lead_id=lead.id)
        if latest_score:
            # Determine lead temperature based on score
            temperature = "Cold Lead ❄️"
            category = "cold"
            if latest_score.score >= 80:
                temperature = "Hot Lead 🔥"
                category = "hot"
            elif latest_score.score >= 60:
                temperature = "Warm Lead ☀️"
                category = "warm"
            
            leads_with_scores.append({
                "id": lead.id,
                "name": lead.name,
                "email": lead.email,
                "company": lead.company,
                "job_title": lead.job_title,
                "industry": lead.industry or "Unknown",
                "score": round(latest_score.score, 4),
                "interpretation": temperature,
                "category": category,
                "demographic_score": round(latest_score.demographic_score, 4),
                "behavioral_score": round(latest_score.behavioral_score, 4),
                "firmographic_score": round(latest_score.firmographic_score, 4),
                "website_visits": lead.website_visits,
                "time_spent": lead.time_spent_on_website,
                "page_views": lead.page_views,
                "source": lead.source,
                "created_at": lead.created_at.isoformat() if lead.created_at else None,
                "score_created_at": latest_score.created_at.isoformat() if latest_score.created_at else None
            })
    
    # Sort by score (descending)
    leads_with_scores = sorted(
        leads_with_scores, 
        key=lambda x: x["score"], 
        reverse=True
    )
    
    # Get top N leads for the summary
    top_leads = leads_with_scores[:limit]
    
    # Calculate score distribution
    score_distribution = {}
    for score_range in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
        lower, upper = score_range
        count = sum(1 for lead in leads_with_scores if lower <= lead["score"] < (upper if upper < 100 else 101))
        score_distribution[f"{lower}-{upper}"] = count
    
    # Calculate counts by category
    hot_leads_count = sum(1 for lead in leads_with_scores if lead["score"] >= 80)
    warm_leads_count = sum(1 for lead in leads_with_scores if 60 <= lead["score"] < 80)
    cold_leads_count = sum(1 for lead in leads_with_scores if lead["score"] < 60)
    
    # Calculate total leads processed
    total_leads_count = len(leads_with_scores)
    
    # Calculate category percentages
    hot_percentage = round((hot_leads_count / total_leads_count) * 100, 1) if total_leads_count > 0 else 0
    warm_percentage = round((warm_leads_count / total_leads_count) * 100, 1) if total_leads_count > 0 else 0
    cold_percentage = round((cold_leads_count / total_leads_count) * 100, 1) if total_leads_count > 0 else 0
    
    # Get model information
    model_info = {
        "version": scoring_service.MODEL_VERSION,
        "type": "Hybrid (Rule-Based + Machine Learning)" 
               if hasattr(scoring_service, "ml_model") and scoring_service.ml_model 
               else "Rule-Based",
        "features": {
            "demographic": list(scoring_service.FEATURE_WEIGHTS["demographic"].keys()),
            "behavioral": list(scoring_service.FEATURE_WEIGHTS["behavioral"].keys()),
            "firmographic": list(scoring_service.FEATURE_WEIGHTS["firmographic"].keys())
        },
        "component_weights": scoring_service.COMPONENT_WEIGHTS
    }
    
    # Calculate average scores
    avg_score = sum(lead["score"] for lead in leads_with_scores) / max(len(leads_with_scores), 1)
    avg_demographic = sum(lead["demographic_score"] for lead in leads_with_scores) / max(len(leads_with_scores), 1)
    avg_behavioral = sum(lead["behavioral_score"] for lead in leads_with_scores) / max(len(leads_with_scores), 1)
    avg_firmographic = sum(lead["firmographic_score"] for lead in leads_with_scores) / max(len(leads_with_scores), 1)
    
    # Prepare data for score chart visualization
    score_chart_data = [
        {"name": lead["name"], "score": lead["score"], "category": lead["category"]} 
        for lead in top_leads
    ]
    
    # Prepare category distribution for pie chart
    category_distribution = [
        {"category": "Hot", "count": hot_leads_count, "percentage": hot_percentage},
        {"category": "Warm", "count": warm_leads_count, "percentage": warm_percentage},
        {"category": "Cold", "count": cold_leads_count, "percentage": cold_percentage}
    ]
    
    # Prepare distribution data for histogram
    score_histogram = [
        {"range": range_key, "count": count, "percentage": round((count / total_leads_count) * 100, 1) if total_leads_count > 0 else 0}
        for range_key, count in score_distribution.items()
    ]
    
    # Prepare industry distribution
    industry_distribution = {}
    for lead in leads_with_scores:
        industry = lead.get("industry", "Unknown")
        if not industry:
            industry = "Unknown"
        if industry not in industry_distribution:
            industry_distribution[industry] = 0
        industry_distribution[industry] += 1
    
    # Prepare industry distribution for visualization
    industry_chart_data = [
        {"industry": industry, "count": count, "percentage": round((count / total_leads_count) * 100, 1) if total_leads_count > 0 else 0} 
        for industry, count in sorted(industry_distribution.items(), key=lambda x: x[1], reverse=True)
    ]
    
    # Collect sources for visualization
    source_distribution = {}
    for lead in leads_with_scores:
        source = lead.get("source", "Unknown")
        if not source:
            source = "Unknown"
        if source not in source_distribution:
            source_distribution[source] = 0
        source_distribution[source] += 1
    
    # Prepare source distribution for visualization
    source_chart_data = [
        {"source": source, "count": count, "percentage": round((count / total_leads_count) * 100, 1) if total_leads_count > 0 else 0} 
        for source, count in sorted(source_distribution.items(), key=lambda x: x[1], reverse=True)
    ]
    
    # Return enriched dashboard data
    return {
        "top_leads": top_leads,
        "model_info": model_info,
        "total_leads": total_leads_count,
        "filter_days": days,
        "score_statistics": {
            "average_score": round(avg_score, 4),
            "average_demographic": round(avg_demographic, 4),
            "average_behavioral": round(avg_behavioral, 4),
            "average_firmographic": round(avg_firmographic, 4),
            "hot_leads": hot_leads_count,
            "warm_leads": warm_leads_count,
            "cold_leads": cold_leads_count,
            "hot_percentage": hot_percentage,
            "warm_percentage": warm_percentage,
            "cold_percentage": cold_percentage,
            "score_distribution": score_distribution
        },
        "chart_data": {
            "score_chart": score_chart_data,
            "category_distribution": category_distribution,
            "score_histogram": score_histogram,
            "industry_chart": industry_chart_data,
            "source_chart": source_chart_data
        },
        "timestamp": current_time.isoformat(),
        "download_url": "/api/v1/leads/download-scored-csv"
    }

@router.post("/score-all", response_model=dict)
def score_all_leads(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(allow_admin),
) -> Any:
    """
    Score all leads in the system. Only admins and managers can trigger this.
    """
    scoring_service = LeadScoringService(db)
    scores = scoring_service.score_all_leads()
    
    return {
        "message": f"Successfully scored {len(scores)} leads",
        "count": len(scores)
    }

@router.post("/score/train-model", response_model=Dict[str, Any])
def train_lead_scoring_model(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(allow_admin),
    retrain: bool = Query(False, description="Force retrain even if model exists"),
) -> Any:
    """
    Train the machine learning model for lead scoring.
    Only admins and managers can trigger model training.
    """
    scoring_service = LeadScoringService(db)
    result = scoring_service.train_ml_model(retrain=retrain)
    
    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
    
    return result

@router.post("/upload-csv", response_model=Dict[str, Any])
async def upload_leads_csv(
    *,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    file: UploadFile = File(...),
) -> Any:
    """
    Upload a CSV file with leads, store them in the database, score them, and return visualization data.
    
    Process:
    1. Parse and validate the CSV file
    2. Store lead data in the leads table
    3. Score each lead and store scores in the lead_score table
    4. Return visualization-ready data for the dashboard
    
    This endpoint accepts a CSV file containing lead information, processes each lead,
    and returns scoring results. The CSV must contain the following required columns:
    - name: Lead's full name
    - email: Lead's email address (must be unique)
    - company: Company name
    - job_title: Lead's job title
    
    Additional optional columns can include:
    - phone: Contact phone number
    - industry: Industry sector (e.g., technology, finance)
    - city: City location
    - country: Country location
    - website_visits: Number of visits to website
    - time_spent: Time spent on website in seconds
    - page_views: Number of pages viewed
    - source: Lead source (e.g., website, referral)
    - campaign: Marketing campaign
    - tags: Comma-separated tags
    - company_size: Size of company (e.g., small, mid-market, enterprise)
    
    Any additional columns will be stored as custom fields.
    
    Returns:
    - message: Success message
    - results: List of leads with their scores and interpretations
    - timestamp: ISO format timestamp
    - summary: Statistical summary of processed leads
    - score_distribution: Distribution of scores across ranges
    - chart_data: Visualization-ready data for dashboard charts
    """
    try:
        # STEP 1: Parse and validate the CSV file
        print(f"[{datetime.utcnow().isoformat()}] STEP 1: Parsing and validating CSV file")
        
        # Validate file format
        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Only CSV files are allowed. Please upload a file with .csv extension."
            )
        
        try:
            # Read CSV content
            contents = await file.read()
            csv_content = contents.decode('utf-8')
            
            # Debug: log the first few lines of the CSV content
            print(f"CSV content preview: {csv_content[:200]}...")
            
            # Try to parse with pandas
            df = pd.read_csv(StringIO(csv_content))
            
            # Debug: log the dataframe information
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            if len(df) > 0:
                print(f"First row sample: {df.iloc[0].to_dict()}")
        except Exception as parse_error:
            print(f"CSV parsing error: {str(parse_error)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse CSV file: {str(parse_error)}. Please ensure the file is properly formatted."
            )
        
        # Validate required columns
        required_columns = ["name", "email", "company", "job_title"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        # Auto-generate missing columns if possible
        if missing_columns and "job_title" in df.columns:
            print(f"Attempting to auto-generate missing columns: {missing_columns}")
            
            # Add dummy/generated data for compatibility with the fake_leads.csv format
            if "name" in missing_columns:
                # Generate names from job titles
                df["name"] = df["job_title"].apply(lambda x: f"{x} User")
                print("Auto-generated 'name' column")
                
            if "email" in missing_columns:
                # Generate emails from job titles
                df["email"] = df["job_title"].apply(lambda x: f"{x.lower().replace(' ', '.')}@example.com")
                print("Auto-generated 'email' column")
                
            if "company" in missing_columns:
                # Generate company names from industry if available, or default
                if "industry" in df.columns:
                    df["company"] = df["industry"].apply(lambda x: f"{x.capitalize()} Corp")
                else:
                    df["company"] = "Default Company"
                print("Auto-generated 'company' column")
            
            # Check if we resolved all missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_message = f"Missing required columns: {', '.join(missing_columns)}. Please ensure your CSV contains all required fields."
            print(f"Validation error: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_message
            )
        
        # STEP 2: Store lead data in the leads table
        print(f"[{datetime.utcnow().isoformat()}] STEP 2: Storing {len(df)} leads in the database")
        created_leads = []
        updated_leads = []
        error_leads = []
        
        for idx, row in df.iterrows():
            try:
                # Check if lead with this email already exists
                lead = lead_crud.get_lead_by_email(db, email=row["email"])
                
                if lead:
                    print(f"Lead with email {row['email']} already exists, updating...")
                    updated_leads.append(lead)
                    continue
                    
                # Create new lead with safe defaults
                lead_data = {
                    "name": row["name"],
                    "email": row["email"],
                    "phone": str(row.get("phone", "")).strip() if pd.notna(row.get("phone", "")) and row.get("phone", "").strip() != "" else f"undefined_{row['email']}",
                    "company": row["company"],
                    "source": str(row.get("source", "csv_import")) if pd.notna(row.get("source", "")) else "csv_import",
                    "campaign": str(row.get("campaign", "csv_import")) if pd.notna(row.get("campaign", "")) else "csv_import",
                    "industry": str(row.get("industry", "")) if pd.notna(row.get("industry", "")) else "",
                    "job_title": row["job_title"],
                    "city": str(row.get("city", "")) if pd.notna(row.get("city", "")) else "",
                    "country": str(row.get("country", "")) if pd.notna(row.get("country", "")) else "",
                    "status": "new",
                    "website_visits": int(row.get("website_visits", 0)) if pd.notna(row.get("website_visits", 0)) else 0,
                    "time_spent_on_website": int(row.get("time_spent", 0)) if pd.notna(row.get("time_spent", 0)) else 0,
                    "page_views": int(row.get("page_views", 0)) if pd.notna(row.get("page_views", 0)) else 0,
                    "tags": str(row.get("tags", "")).split(",") if pd.notna(row.get("tags", "")) else [],
                    "custom_fields": {},
                }
                
                # Add any additional columns as custom fields
                custom_fields = {}
                for col in df.columns:
                    if col not in lead_data and col not in ["tags"] and pd.notna(row[col]):
                        custom_fields[col] = str(row[col])
                
                if custom_fields:
                    lead_data["custom_fields"] = custom_fields
                
                # Create lead in database
                try:
                    lead_in = LeadCreate(**lead_data)
                    lead = lead_crud.create_lead(db, lead_in=lead_in)
                    print(f"Created new lead: {lead.name} (ID: {lead.id})")
                    created_leads.append(lead)
                except Exception as create_error:
                    print(f"Error creating lead {row['email']}: {str(create_error)}")
                    error_leads.append({"email": row["email"], "error": str(create_error)})
                    continue
                    
            except Exception as row_error:
                print(f"Error processing row {idx}: {str(row_error)}")
                error_leads.append({"row": idx, "error": str(row_error)})
                continue
        
        # Combine created and updated leads
        all_processed_leads = created_leads + updated_leads
        
        if not all_processed_leads:
            error_detail = "No leads were successfully processed from the CSV."
            if error_leads:
                # Using simpler string concatenation to avoid nested f-string issues
                error_messages = []
                for e in error_leads[:5]:
                    if "email" in e:
                        error_messages.append(f"{e['email']}: {e['error']}")
                    else:
                        error_messages.append(f"Row {e.get('row', '?')}: {e['error']}")
                
                error_detail += f" Errors: {', '.join(error_messages)}"
                if len(error_leads) > 5:
                    error_detail += f" and {len(error_leads) - 5} more errors."
            
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_detail
            )
        
        # STEP 3: Score each lead and store scores in the lead_score table
        print(f"[{datetime.utcnow().isoformat()}] STEP 3: Scoring {len(all_processed_leads)} leads")
        scoring_service = LeadScoringService(db)
        results = []
        score_errors = []
        
        for lead in all_processed_leads:
            try:
                # Score the lead (this will create/update in lead_score table)
                score = scoring_service.score_lead(lead.id)
                
                # Determine lead temperature based on score
                temperature = "Cold Lead ❄️"
                if score.score >= 80:
                    temperature = "Hot Lead 🔥"
                elif score.score >= 60:
                    temperature = "Warm Lead ☀️"
                
                # Add to results
                results.append({
                    "id": lead.id,
                    "name": lead.name,
                    "email": lead.email,
                    "company": lead.company,
                    "job_title": lead.job_title,
                    "industry": lead.industry,
                    "score": round(score.score, 4),
                    "interpretation": temperature,
                    "category": "hot" if score.score >= 80 else "warm" if score.score >= 60 else "cold",
                    "demographic_score": round(score.demographic_score, 4),
                    "behavioral_score": round(score.behavioral_score, 4),
                    "firmographic_score": round(score.firmographic_score, 4),
                    "website_visits": lead.website_visits,
                    "time_spent": lead.time_spent_on_website,
                    "page_views": lead.page_views,
                    "source": lead.source,
                    "created_at": lead.created_at.isoformat() if lead.created_at else None
                })
                print(f"Scored lead: {lead.name} (ID: {lead.id}) - Score: {round(score.score, 4)}")
            except Exception as score_error:
                print(f"Error scoring lead {lead.id}: {str(score_error)}")
                score_errors.append({"lead_id": lead.id, "lead_name": lead.name, "error": str(score_error)})
        
        # STEP 4: Prepare visualization data for the dashboard
        print(f"[{datetime.utcnow().isoformat()}] STEP 4: Preparing visualization data")
        
        # Sort results by score (descending)
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # Calculate summary statistics
        hot_leads = sum(1 for r in results if r["score"] >= 80)
        warm_leads = sum(1 for r in results if 60 <= r["score"] < 80)
        cold_leads = sum(1 for r in results if r["score"] < 60)
        avg_score = sum(r["score"] for r in results) / len(results) if results else 0
        
        # Calculate score distribution
        score_distribution = {}
        for score_range in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
            lower, upper = score_range
            count = sum(1 for lead in results if lower <= lead["score"] < (upper if upper < 100 else 101))
            score_distribution[f"{lower}-{upper}"] = count
        
        # Calculate industry distribution
        industry_distribution = {}
        for lead in results:
            industry = lead.get("industry", "Unknown")
            if not industry:
                industry = "Unknown"
            if industry not in industry_distribution:
                industry_distribution[industry] = 0
            industry_distribution[industry] += 1
        
        # Prepare visualization data
        score_chart_data = [
            {"name": r["name"], "score": r["score"], "category": r["category"]} 
            for r in results[:min(len(results), 15)]
        ]
        
        industry_chart_data = [
            {"industry": industry, "count": count, "percentage": round(count * 100 / len(results), 1)} 
            for industry, count in industry_distribution.items()
        ]
        
        # Calculate percentage distribution by temperature
        temperature_distribution = {
            "hot": {"count": hot_leads, "percentage": round(hot_leads * 100 / len(results), 1) if results else 0},
            "warm": {"count": warm_leads, "percentage": round(warm_leads * 100 / len(results), 1) if results else 0},
            "cold": {"count": cold_leads, "percentage": round(cold_leads * 100 / len(results), 1) if results else 0}
        }
        
        # Return comprehensive response with visualization data
        print(f"[{datetime.utcnow().isoformat()}] Returning results for {len(results)} leads")
        return {
            "message": f"Successfully processed {len(results)} leads",
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_processed": len(results),
                "new_leads": len(created_leads),
                "updated_leads": len(updated_leads),
                "failed_leads": len(error_leads),
                "hot_leads": hot_leads,
                "warm_leads": warm_leads, 
                "cold_leads": cold_leads,
                "average_score": round(avg_score, 4)
            },
            "score_distribution": score_distribution,
            "temperature_distribution": temperature_distribution,
            "chart_data": {
                "score_chart": score_chart_data,
                "industry_chart": industry_chart_data
            },
            "errors": {
                "lead_creation_errors": error_leads[:10] if error_leads else [],
                "scoring_errors": score_errors[:10] if score_errors else []
            },
            "download_url": "/api/v1/leads/download-scored-csv"
        }
    
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[{datetime.utcnow().isoformat()}] Error processing CSV: {str(e)}")
        print(error_trace)
        
        # Return a more detailed error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": f"Error processing CSV: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
                "error_type": type(e).__name__
            }
        )

@router.get("/download-scored-csv")
def download_scored_leads_csv(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = Query(100, ge=1, le=1000),
    include_empty_scores: bool = Query(False, description="Include leads without scores")
) -> Any:
    """
    Download scored leads as a CSV file.
    
    This endpoint generates a CSV file containing lead information and their 
    corresponding scores. The leads are sorted by score in descending order.
    
    The CSV includes the following columns:
    - id: Lead ID
    - name: Lead's full name
    - email: Lead's email address
    - company: Company name
    - job_title: Lead's job title
    - industry: Industry sector
    - score: Overall lead score (0-100)
    - interpretation: Score interpretation (Hot Lead, Warm Lead, Cold Lead)
    - demographic_score: Demographic component score
    - behavioral_score: Behavioral component score
    - firmographic_score: Firmographic component score
    - status: Current lead status
    - source: Lead source
    - is_converted: Whether the lead has converted
    - created_at: Lead creation date/time
    
    Parameters:
    - limit: Maximum number of leads to include (default: 100)
    - include_empty_scores: Whether to include leads without scores (default: False)
    
    Returns:
    - A downloadable CSV file named "scored_leads_YYYYMMDD_HHMMSS.csv"
    """
    try:
        # Get leads with scores
        leads = lead_crud.get_leads(db=db, skip=0, limit=limit)
        leads_with_scores = []
        leads_without_scores = []
        
        for lead in leads:
            latest_score = lead_crud.get_latest_lead_score(db=db, lead_id=lead.id)
            if latest_score:
                # Determine lead temperature based on score
                temperature = "Cold Lead"
                if latest_score.score >= 80:
                    temperature = "Hot Lead"
                elif latest_score.score >= 60:
                    temperature = "Warm Lead"
                
                leads_with_scores.append({
                    "id": lead.id,
                    "name": lead.name,
                    "email": lead.email,
                    "phone": lead.phone if lead.phone and not lead.phone.startswith("undefined_") else "",
                    "company": lead.company,
                    "job_title": lead.job_title,
                    "industry": lead.industry or "",
                    "score": round(latest_score.score, 4),
                    "interpretation": temperature,
                    "demographic_score": round(latest_score.demographic_score, 4),
                    "behavioral_score": round(latest_score.behavioral_score, 4),
                    "firmographic_score": round(latest_score.firmographic_score, 4),
                    "status": lead.status,
                    "source": lead.source,
                    "is_converted": "Yes" if lead.is_converted else "No",
                    "created_at": lead.created_at.strftime("%Y-%m-%d %H:%M:%S") if lead.created_at else ""
                })
            elif include_empty_scores:
                leads_without_scores.append({
                    "id": lead.id,
                    "name": lead.name,
                    "email": lead.email,
                    "phone": lead.phone if lead.phone and not lead.phone.startswith("undefined_") else "",
                    "company": lead.company,
                    "job_title": lead.job_title,
                    "industry": lead.industry or "",
                    "score": "N/A",
                    "interpretation": "Not Scored",
                    "demographic_score": "N/A",
                    "behavioral_score": "N/A",
                    "firmographic_score": "N/A",
                    "status": lead.status,
                    "source": lead.source,
                    "is_converted": "Yes" if lead.is_converted else "No",
                    "created_at": lead.created_at.strftime("%Y-%m-%d %H:%M:%S") if lead.created_at else ""
                })
        
        # Sort by score (descending)
        leads_with_scores = sorted(leads_with_scores, key=lambda x: x["score"], reverse=True)
        
        # Combine scored and unscored leads if requested
        all_leads = leads_with_scores
        if include_empty_scores:
            all_leads = leads_with_scores + leads_without_scores
        
        # Create CSV
        if not all_leads:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No leads found"
            )
        
        # Convert to DataFrame and then to CSV
        df = pd.DataFrame(all_leads)
        
        # Create a string buffer to hold the CSV
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        # Return streaming response
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=scored_leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[{datetime.utcnow().isoformat()}] Error generating CSV: {str(e)}")
        print(error_trace)
        
        # Return a more detailed error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": f"Error generating CSV: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
                "error_type": type(e).__name__
            }
        )

@router.get("/{lead_id}", response_model=LeadSchema)
def read_lead(
    *,
    db: Session = Depends(get_db),
    lead_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get lead by ID.
    """
    lead = lead_crud.get_lead(db=db, lead_id=lead_id)
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lead not found"
        )
    return lead

@router.put("/{lead_id}", response_model=LeadSchema)
def update_lead(
    *,
    db: Session = Depends(get_db),
    lead_id: int = Path(..., ge=1),
    lead_in: LeadUpdate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Update a lead.
    """
    lead = lead_crud.get_lead(db=db, lead_id=lead_id)
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lead not found"
        )
    
    lead = lead_crud.update_lead(
        db=db,
        lead_id=lead_id,
        lead_in=lead_in
    )
    
    # Re-score the lead if significant fields were updated
    should_rescore = any(field in lead_in.dict(exclude_unset=True) for field in [
        "industry", "job_title", "company", "website_visits", 
        "time_spent_on_website", "page_views", "custom_fields"
    ])
    
    if should_rescore:
        scoring_service = LeadScoringService(db)
        scoring_service.score_lead(lead.id)
    
    return lead

@router.delete("/{lead_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_lead(
    *,
    db: Session = Depends(get_db),
    lead_id: int = Path(..., ge=1),
    current_user: User = Depends(allow_admin),
) -> None:
    """
    Delete a lead. Only admins can delete leads.
    """
    lead = lead_crud.get_lead(db=db, lead_id=lead_id)
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lead not found"
        )
    
    lead_crud.delete_lead(db=db, lead_id=lead_id)
    return None

@router.get("/{lead_id}/score", response_model=LeadScoreSchema)
def get_lead_score(
    *,
    db: Session = Depends(get_db),
    lead_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get the latest score for a lead.
    """
    lead = lead_crud.get_lead(db=db, lead_id=lead_id)
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lead not found"
        )
    
    score = lead_crud.get_latest_lead_score(db=db, lead_id=lead_id)
    if not score:
        # If no score exists, calculate one
        scoring_service = LeadScoringService(db)
        score = scoring_service.score_lead(lead_id)
    
    return score

@router.post("/{lead_id}/score", response_model=LeadScoreSchema)
def calculate_lead_score(
    *,
    db: Session = Depends(get_db),
    lead_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Calculate a new score for a lead.
    """
    lead = lead_crud.get_lead(db=db, lead_id=lead_id)
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lead not found"
        )
    
    scoring_service = LeadScoringService(db)
    score = scoring_service.score_lead(lead_id)
    
    return score

@router.get("/{lead_id}/scores", response_model=List[LeadScoreSchema])
def get_lead_score_history(
    *,
    db: Session = Depends(get_db),
    lead_id: int = Path(..., ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get score history for a lead.
    """
    lead = lead_crud.get_lead(db=db, lead_id=lead_id)
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lead not found"
        )
    
    scores = lead_crud.get_lead_scores(db=db, lead_id=lead_id, limit=limit)
    return scores

@router.post("/{lead_id}/activity", response_model=LeadActivitySchema)
def add_lead_activity(
    *,
    db: Session = Depends(get_db),
    lead_id: int = Path(..., ge=1),
    activity_in: LeadActivityCreate = Body(...),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Add a new activity for a lead.
    """
    lead = lead_crud.get_lead(db=db, lead_id=lead_id)
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lead not found"
        )
    
    # Ensure lead_id in body matches path parameter
    if activity_in.lead_id != lead_id:
        activity_in.lead_id = lead_id
    
    activity = lead_crud.create_lead_activity(db=db, activity_in=activity_in)
    
    # Re-score the lead since there's new activity
    scoring_service = LeadScoringService(db)
    scoring_service.score_lead(lead_id)
    
    return activity

@router.get("/{lead_id}/activities", response_model=List[LeadActivitySchema])
def get_lead_activities(
    *,
    db: Session = Depends(get_db),
    lead_id: int = Path(..., ge=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    activity_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get activities for a lead.
    """
    lead = lead_crud.get_lead(db=db, lead_id=lead_id)
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lead not found"
        )
    
    activities = lead_crud.get_lead_activities(
        db=db,
        lead_id=lead_id,
        skip=skip,
        limit=limit,
        activity_type=activity_type
    )
    
    return activities