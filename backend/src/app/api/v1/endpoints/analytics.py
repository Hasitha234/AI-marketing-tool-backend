from typing import Any, Dict, List
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy import func, desc, and_, asc
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import os
from pathlib import Path

from app.api.dependencies import get_db
from app.core.security import get_current_user
from app.core.permissions import allow_admin
from app.models.user import User
from app.models.lead import Lead, LeadScore, LeadActivity
from app.services.lead_scoring import LeadScoringService

router = APIRouter()

@router.get("/lead-summary")
def get_lead_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(allow_admin),
    days: int = Query(30, ge=1, le=365),
) -> Dict[str, Any]:
    """
    Get summary statistics for leads.
    """
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get total leads
    total_leads = db.query(func.count(Lead.id)).scalar()
    
    # Get new leads in period
    new_leads = db.query(func.count(Lead.id)).filter(
        Lead.created_at >= start_date
    ).scalar()
    
    # Get converted leads in period
    converted_leads = db.query(func.count(Lead.id)).filter(
        and_(
            Lead.is_converted == True,
            Lead.converted_at >= start_date
        )
    ).scalar()
    
    # Get conversion rate
    conversion_rate = 0
    if new_leads > 0:
        conversion_rate = round((converted_leads / new_leads) * 100, 2)
    
    # Get leads by source
    leads_by_source = db.query(
        Lead.source,
        func.count(Lead.id).label("count")
    ).group_by(Lead.source).all()
    
    # Get leads by status
    leads_by_status = db.query(
        Lead.status,
        func.count(Lead.id).label("count")
    ).group_by(Lead.status).all()
    
    # Get average lead score
    avg_score = db.query(func.avg(LeadScore.score)).join(Lead).scalar() or 0
    
    # Get top scoring leads
    top_leads = db.query(Lead, LeadScore.score).join(
        LeadScore, Lead.id == LeadScore.lead_id
    ).order_by(
        desc(LeadScore.score)
    ).limit(10).all()
    
    # Format top leads
    top_leads_data = [
        {
            "id": lead.id,
            "name": lead.name,
            "email": lead.email,
            "company": lead.company,
            "score": round(score, 2)
        }
        for lead, score in top_leads
    ]
    
    # Get lead score distribution
    score_ranges = [
        (0, 20), (20, 40), (40, 60), (60, 80), (80, 100)
    ]
    
    score_distribution = []
    for lower, upper in score_ranges:
        count = db.query(func.count(LeadScore.id)).filter(
            and_(
                LeadScore.score >= lower,
                LeadScore.score < upper
            )
        ).scalar()
        
        score_distribution.append({
            "range": f"{lower}-{upper}",
            "count": count
        })
    
    # Return summary data
    return {
        "total_leads": total_leads,
        "new_leads": new_leads,
        "converted_leads": converted_leads,
        "conversion_rate": conversion_rate,
        "average_score": round(avg_score, 2),
        "leads_by_source": [
            {"source": source, "count": count}
            for source, count in leads_by_source
        ],
        "leads_by_status": [
            {"status": status, "count": count}
            for status, count in leads_by_status
        ],
        "top_scoring_leads": top_leads_data,
        "score_distribution": score_distribution,
        "period_days": days
    }

@router.get("/lead-activity-summary")
def get_lead_activity_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(allow_admin),
    days: int = Query(30, ge=1, le=365),
) -> Dict[str, Any]:
    """
    Get summary statistics for lead activities.
    """
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get total activities
    total_activities = db.query(func.count(LeadActivity.id)).filter(
        LeadActivity.occurred_at >= start_date
    ).scalar()
    
    # Get activities by type
    activities_by_type = db.query(
        LeadActivity.activity_type,
        func.count(LeadActivity.id).label("count")
    ).filter(
        LeadActivity.occurred_at >= start_date
    ).group_by(LeadActivity.activity_type).all()
    
    # Get activities by day
    activities_by_day = db.query(
        func.date_trunc('day', LeadActivity.occurred_at).label('day'),
        func.count(LeadActivity.id).label('count')
    ).filter(
        LeadActivity.occurred_at >= start_date
    ).group_by('day').order_by('day').all()
    
    # Return summary data
    return {
        "total_activities": total_activities,
        "activities_by_type": [
            {"type": type_, "count": count}
            for type_, count in activities_by_type
        ],
        "activities_by_day": [
            {"date": day.strftime('%Y-%m-%d'), "count": count}
            for day, count in activities_by_day
        ],
        "period_days": days
    }

@router.get("/lead-conversion-funnel")
def get_lead_conversion_funnel(
    db: Session = Depends(get_db),
    current_user: User = Depends(allow_admin),
    days: int = Query(90, ge=1, le=365),
) -> Dict[str, Any]:
    """
    Get lead conversion funnel statistics.
    """
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Define the lead funnel stages
    stages = ["new", "contacted", "qualified", "converted"]
    
    # Get leads by stage
    funnel_data = []
    for stage in stages:
        if stage == "converted":
            count = db.query(func.count(Lead.id)).filter(
                and_(
                    Lead.is_converted == True,
                    Lead.created_at >= start_date
                )
            ).scalar()
        else:
            count = db.query(func.count(Lead.id)).filter(
                and_(
                    Lead.status == stage,
                    Lead.created_at >= start_date
                )
            ).scalar()
        
        funnel_data.append({
            "stage": stage,
            "count": count
        })
    
    # Calculate conversion rates between stages
    conversion_rates = []
    for i in range(len(funnel_data) - 1):
        current_stage = funnel_data[i]
        next_stage = funnel_data[i + 1]
        
        if current_stage["count"] > 0:
            rate = round((next_stage["count"] / current_stage["count"]) * 100, 2)
        else:
            rate = 0
        
        conversion_rates.append({
            "from_stage": current_stage["stage"],
            "to_stage": next_stage["stage"],
            "rate": rate
        })
    
    # Return funnel data
    return {
        "funnel_stages": funnel_data,
        "conversion_rates": conversion_rates,
        "period_days": days
    }

@router.get("/lead-scoring-model")
def get_lead_scoring_model_info(
    db: Session = Depends(get_db),
    current_user: User = Depends(allow_admin),
) -> Dict[str, Any]:
    """
    Get information about the lead scoring model.
    """
    scoring_service = LeadScoringService(db)
    
    # Check if ML model exists
    model_exists = False
    model_path = scoring_service.MODEL_PATH / scoring_service.RF_MODEL_FILE
    if os.path.exists(model_path):
        model_exists = True
    
    # Get most recent scores
    recent_scores = db.query(LeadScore).order_by(
        desc(LeadScore.created_at)
    ).limit(100).all()
    
    # Calculate average scores
    avg_score = db.query(func.avg(LeadScore.score)).scalar() or 0
    avg_demographic = db.query(func.avg(LeadScore.demographic_score)).scalar() or 0
    avg_behavioral = db.query(func.avg(LeadScore.behavioral_score)).scalar() or 0
    avg_firmographic = db.query(func.avg(LeadScore.firmographic_score)).scalar() or 0
    
    # Get conversion counts by score range
    conversion_by_score = []
    score_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    
    for lower, upper in score_ranges:
        # Count leads in this score range
        leads_in_range = db.query(Lead).join(
            LeadScore, Lead.id == LeadScore.lead_id
        ).filter(
            and_(
                LeadScore.score >= lower,
                LeadScore.score < upper
            )
        ).all()
        
        total_in_range = len(leads_in_range)
        converted_in_range = sum(1 for lead in leads_in_range if lead.is_converted)
        
        conversion_rate = 0
        if total_in_range > 0:
            conversion_rate = round((converted_in_range / total_in_range) * 100, 2)
        
        conversion_by_score.append({
            "range": f"{lower}-{upper}",
            "total_leads": total_in_range,
            "converted_leads": converted_in_range,
            "conversion_rate": conversion_rate
        })
    
    # Get AI/ML model feature importance if available
    feature_importance = {}
    if model_exists and hasattr(scoring_service, "ml_model") and scoring_service.ml_model:
        try:
            # Get sample lead with activities to extract feature names
            lead = db.query(Lead).first()
            if lead:
                activities = db.query(LeadActivity).filter(
                    LeadActivity.lead_id == lead.id
                ).all()
                features = scoring_service._extract_ml_features(lead, activities)
                feature_names = [
                    "job_title", "industry", "location", "visits", 
                    "time_spent", "page_views", "email_opens", "email_clicks", 
                    "form_submissions", "recency", "engagement_ratio", 
                    "activity_freq", "has_company", "company_size"
                ]
                feature_importance = dict(zip(
                    feature_names,
                    scoring_service.ml_model.feature_importances_
                ))
                # Sort by importance
                feature_importance = {
                    k: float(v) for k, v in sorted(
                        feature_importance.items(), 
                        key=lambda item: item[1], 
                        reverse=True
                    )
                }
                # Calculate the contribution of each feature to the mean score
                for feature in feature_importance:
                    # Map raw feature importance to its score contribution
                    feature_importance[feature] = {
                        "importance": feature_importance[feature],
                        "score": round(feature_importance[feature] * avg_score, 4)
                    }
        except Exception as e:
            feature_importance = {"error": str(e)}
    
    # Return model information
    return {
        "model_version": scoring_service.MODEL_VERSION,
        "model_type": "Hybrid (Rule-Based + Machine Learning)" if model_exists else "Rule-Based",
        "machine_learning_model": {
            "exists": model_exists,
            "type": "RandomForestClassifier" if model_exists else None,
            "feature_importance": feature_importance if feature_importance else None
        },
        "score_statistics": {
            "average_score": round(avg_score, 4),
            "average_demographic_score": round(avg_demographic, 4),
            "average_behavioral_score": round(avg_behavioral, 4),
            "average_firmographic_score": round(avg_firmographic, 4),
        },
        "conversion_by_score_range": conversion_by_score,
        "component_weights": {
            "demographic": scoring_service.COMPONENT_WEIGHTS["demographic"],
            "behavioral": scoring_service.COMPONENT_WEIGHTS["behavioral"],
            "firmographic": scoring_service.COMPONENT_WEIGHTS["firmographic"],
        },
        "ml_contribution": 0.6 if model_exists else 0.0,
        "rule_based_contribution": 0.4 if model_exists else 1.0
    }