import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path

from app.models.lead import Lead, LeadScore, LeadActivity
from app.schemas.lead import LeadScoreCreate

class LeadScoringService:
    """Service for scoring leads based on their attributes and activities."""
    
    # Model version
    MODEL_VERSION = "2.0.0"
    
    # Feature weights for the rule-based scoring model
    FEATURE_WEIGHTS = {
        # Demographic features
        "demographic": {
            "job_title": 5,
            "industry": 7,
            "country": 3,
            "city": 2,
        },
        # Behavioral features
        "behavioral": {
            "website_visits": 8,
            "time_spent_on_website": 10,
            "page_views": 7,
            "email_opens": 5,
            "email_clicks": 8,
            "form_submissions": 15,
            "recency": 12,
        },
        # Firmographic features
        "firmographic": {
            "company": 5,
            "company_size": 8,
            "industry_relevance": 10,
        },
    }
    
    # Component weights for rule-based final score
    COMPONENT_WEIGHTS = {
        "demographic": 0.25,
        "behavioral": 0.60,
        "firmographic": 0.15,
    }
    
    # Industry relevance score (tailored for your business)
    INDUSTRY_RELEVANCE = {
        "technology": 1.0,
        "finance": 0.9,
        "healthcare": 0.85,
        "education": 0.8,
        "retail": 0.75,
        "manufacturing": 0.7,
        "other": 0.5,
    }
    
    # Job title relevance score (tailored for your business)
    JOB_TITLE_RELEVANCE = {
        "ceo": 1.0,
        "cto": 0.95,
        "cmo": 0.9,
        "marketing manager": 0.85,
        "marketing director": 0.85,
        "digital marketing": 0.8,
        "manager": 0.7,
        "analyst": 0.6,
        "other": 0.5,
    }
    
    # ML model file path
    MODEL_PATH = Path(__file__).parent.parent.parent / "ml_models"
    RF_MODEL_FILE = "lead_scoring_rf_model.joblib"
    SCALER_FILE = "lead_scoring_scaler.joblib"
    
    def __init__(self, db: Session):
        self.db = db
        self.ml_model = None
        self.scaler = None
        self._load_ml_models()
    
    def _load_ml_models(self):
        """Load machine learning models if available."""
        try:
            if not os.path.exists(self.MODEL_PATH):
                os.makedirs(self.MODEL_PATH)
                
            rf_path = self.MODEL_PATH / self.RF_MODEL_FILE
            scaler_path = self.MODEL_PATH / self.SCALER_FILE
            
            if os.path.exists(rf_path) and os.path.exists(scaler_path):
                self.ml_model = joblib.load(rf_path)
                self.scaler = joblib.load(scaler_path)
                print("ML models loaded successfully")
            else:
                print("ML models not found, using rule-based scoring only")
        except Exception as e:
            print(f"Error loading ML models: {e}")
            # Fall back to rule-based scoring if model loading fails
    
    def score_lead(self, lead_id: int) -> LeadScore:
        """
        Score a lead based on their attributes and activities.
        Returns a LeadScore object with the lead's score.
        """
        # Get lead from database
        lead = self.db.query(Lead).filter(Lead.id == lead_id).first()
        if not lead:
            raise ValueError(f"Lead with ID {lead_id} not found")
        
        # Get lead activities
        activities = self.db.query(LeadActivity).filter(LeadActivity.lead_id == lead_id).all()
        
        # Calculate rule-based scores
        demographic_score = self._calculate_demographic_score(lead)
        behavioral_score = self._calculate_behavioral_score(lead, activities)
        firmographic_score = self._calculate_firmographic_score(lead)
        
        # Calculate rule-based score
        rule_based_score = (
            demographic_score * self.COMPONENT_WEIGHTS["demographic"] +
            behavioral_score * self.COMPONENT_WEIGHTS["behavioral"] +
            firmographic_score * self.COMPONENT_WEIGHTS["firmographic"]
        ) * 100  # Scale to 0-100
        
        # Determine confidence level based on completeness of data
        confidence = self._calculate_confidence(lead, activities)
        
        # Create detailed factors dictionary
        factors = self._create_factors_dict(
            lead, 
            demographic_score, 
            behavioral_score, 
            firmographic_score
        )
        
        # Calculate ML-based prediction if model is available
        ml_prediction = None
        if self.ml_model and self.scaler:
            ml_prediction = self._predict_conversion_probability(lead, activities)
            # Combine rule-based and ML-based scores with 40/60 weighting
            total_score = (0.4 * rule_based_score) + (0.6 * ml_prediction * 100)
            # Add ML prediction to factors
            factors["ml_prediction"] = {
                "conversion_probability": round(ml_prediction, 4),
                "weight": 0.6
            }
            factors["rule_based_score"] = {
                "score": round(rule_based_score, 4),
                "weight": 0.4
            }
        else:
            # Use only rule-based score if ML model not available
            total_score = rule_based_score
        
        # Create or update lead score
        existing_score = (
            self.db.query(LeadScore)
            .filter(LeadScore.lead_id == lead_id)
            .order_by(LeadScore.created_at.desc())
            .first()
        )
        
        if existing_score:
            # Update existing score
            existing_score.score = round(total_score, 4)
            existing_score.demographic_score = round(demographic_score * 100, 4)
            existing_score.behavioral_score = round(behavioral_score * 100, 4)
            existing_score.firmographic_score = round(firmographic_score * 100, 4)
            existing_score.factors = factors
            existing_score.confidence = round(confidence, 4)
            existing_score.model_version = self.MODEL_VERSION
            existing_score.updated_at = datetime.utcnow()
            
            self.db.add(existing_score)
            self.db.commit()
            self.db.refresh(existing_score)
            return existing_score
        else:
            # Create new score
            lead_score = LeadScore(
                lead_id=lead_id,
                score=round(total_score, 4),
                demographic_score=round(demographic_score * 100, 4),
                behavioral_score=round(behavioral_score * 100, 4),
                firmographic_score=round(firmographic_score * 100, 4),
                factors=factors,
                confidence=round(confidence, 4),
                model_version=self.MODEL_VERSION
            )
            
            self.db.add(lead_score)
            self.db.commit()
            self.db.refresh(lead_score)
            return lead_score
    
    def _predict_conversion_probability(self, lead: Lead, activities: List[LeadActivity]) -> float:
        """Use machine learning model to predict lead conversion probability."""
        try:
            # Extract features from lead and activities
            features = self._extract_ml_features(lead, activities)
            
            # Scale features
            scaled_features = self.scaler.transform([features])
            
            # Predict conversion probability
            probability = self.ml_model.predict_proba(scaled_features)[0, 1]
            
            return probability
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            # Fall back to confidence score if prediction fails
            return self._calculate_confidence(lead, activities)
    
    def _extract_ml_features(self, lead: Lead, activities: List[LeadActivity]) -> List[float]:
        """Extract features for machine learning model."""
        features = []
        
        # Demographic features
        job_title_relevance = self._get_job_title_relevance(lead.job_title) if lead.job_title else 0.0
        industry_relevance = self._get_industry_relevance(lead.industry) if lead.industry else 0.0
        has_location = 1.0 if lead.country else 0.0
        
        # Behavioral features
        website_visits = lead.website_visits or 0
        time_spent = lead.time_spent_on_website or 0
        page_views = lead.page_views or 0
        
        email_opens = self._count_activities(activities, "email_open")
        email_clicks = self._count_activities(activities, "email_click")
        form_submissions = self._count_activities(activities, "form_submission")
        
        # Recency (days since last activity, capped at 30 days)
        days_since_activity = 30.0
        if lead.last_activity:
            now = datetime.now(timezone.utc)
            days_since_activity = min((now - lead.last_activity).days, 30.0)
        
        # Engagement ratio (clicks/opens)
        engagement_ratio = email_clicks / max(email_opens, 1)
        
        # Activity frequency (activities per day since creation)
        now = datetime.now(timezone.utc)
        days_since_created = max((now - lead.created_at).days, 1)
        activity_frequency = len(activities) / days_since_created
        
        # Firmographic features
        has_company = 1.0 if lead.company else 0.0
        
        # Company size score
        company_size_score = 0.0
        if lead.custom_fields and "company_size" in lead.custom_fields:
            company_size = lead.custom_fields["company_size"]
            if company_size == "enterprise":
                company_size_score = 1.0
            elif company_size == "mid-market":
                company_size_score = 0.8
            elif company_size == "small":
                company_size_score = 0.6
            else:
                company_size_score = 0.4
        
        # Aggregate into feature vector
        features = [
            job_title_relevance,
            industry_relevance,
            has_location,
            website_visits,
            time_spent,
            page_views,
            email_opens,
            email_clicks,
            form_submissions,
            30.0 - days_since_activity,  # Invert so higher is better
            engagement_ratio,
            activity_frequency,
            has_company,
            company_size_score
        ]
        
        return features
    
    def train_ml_model(self, retrain: bool = False) -> Dict[str, Any]:
        """
        Train a machine learning model for lead scoring.
        Uses historical data of converted and non-converted leads.
        """
        # Check if model exists and we're not forcing retrain
        rf_path = self.MODEL_PATH / self.RF_MODEL_FILE
        scaler_path = self.MODEL_PATH / self.SCALER_FILE
        
        if not retrain and os.path.exists(rf_path) and os.path.exists(scaler_path):
            return {
                "status": "skipped",
                "message": "Model already exists. Use retrain=True to force retraining."
            }
        
        # Get all leads with conversion status
        leads = self.db.query(Lead).all()
        if len(leads) < 50:
            return {
                "status": "error", 
                "message": "Not enough leads to train model. Need at least 50 leads."
            }
        
        # Prepare training data
        X = []  # Features
        y = []  # Target (converted or not)
        
        for lead in leads:
            activities = self.db.query(LeadActivity).filter(LeadActivity.lead_id == lead.id).all()
            features = self._extract_ml_features(lead, activities)
            X.append(features)
            y.append(1 if lead.is_converted else 0)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Create and train scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train random forest model
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        self.ml_model.fit(X_scaled, y)
        
        # Save models
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)
        
        joblib.dump(self.ml_model, rf_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Calculate training metrics
        train_accuracy = self.ml_model.score(X_scaled, y)
        feature_importances = dict(zip(
            ["job_title", "industry", "location", "visits", "time_spent", 
             "page_views", "email_opens", "email_clicks", "form_submissions", 
             "recency", "engagement_ratio", "activity_freq", "has_company", "company_size"],
            self.ml_model.feature_importances_
        ))
        
        return {
            "status": "success",
            "model_version": self.MODEL_VERSION,
            "training_samples": len(y),
            "converted_ratio": sum(y) / len(y),
            "accuracy": train_accuracy,
            "feature_importances": feature_importances
        }
    
    def score_all_leads(self) -> List[LeadScore]:
        """Score all leads in the database."""
        leads = self.db.query(Lead).all()
        scores = []
        
        for lead in leads:
            score = self.score_lead(lead.id)
            scores.append(score)
        
        return scores
    
    def _calculate_demographic_score(self, lead: Lead) -> float:
        """Calculate demographic score based on lead attributes."""
        score = 0.0
        total_weight = 0.0
        
        # Job title score
        if lead.job_title:
            weight = self.FEATURE_WEIGHTS["demographic"]["job_title"]
            job_relevance = self._get_job_title_relevance(lead.job_title)
            score += weight * job_relevance
            total_weight += weight
        
        # Industry score
        if lead.industry:
            weight = self.FEATURE_WEIGHTS["demographic"]["industry"]
            industry_relevance = self._get_industry_relevance(lead.industry)
            score += weight * industry_relevance
            total_weight += weight
        
        # Location score
        if lead.country:
            weight = self.FEATURE_WEIGHTS["demographic"]["country"]
            # Default location relevance is 0.7
            score += weight * 0.7
            total_weight += weight
        
        # Normalize score to 0-1 range
        if total_weight > 0:
            return score / total_weight
        return 0.0
    
    def _calculate_behavioral_score(self, lead: Lead, activities: List[LeadActivity]) -> float:
        """Calculate behavioral score based on lead activities."""
        score = 0.0
        total_weight = 0.0
        
        # Website engagement score
        if lead.website_visits > 0:
            weight = self.FEATURE_WEIGHTS["behavioral"]["website_visits"]
            visit_score = min(lead.website_visits / 10, 1.0)  # Normalize visits
            score += weight * visit_score
            total_weight += weight
        
        if lead.time_spent_on_website > 0:
            weight = self.FEATURE_WEIGHTS["behavioral"]["time_spent_on_website"]
            time_score = min(lead.time_spent_on_website / 30, 1.0)  # Normalize time
            score += weight * time_score
            total_weight += weight
        
        if lead.page_views > 0:
            weight = self.FEATURE_WEIGHTS["behavioral"]["page_views"]
            view_score = min(lead.page_views / 20, 1.0)  # Normalize page views
            score += weight * view_score
            total_weight += weight
        
        # Email engagement score
        email_opens = self._count_activities(activities, "email_open")
        if email_opens > 0:
            weight = self.FEATURE_WEIGHTS["behavioral"]["email_opens"]
            open_score = min(email_opens / 5, 1.0)  # Normalize opens
            score += weight * open_score
            total_weight += weight
        
        email_clicks = self._count_activities(activities, "email_click")
        if email_clicks > 0:
            weight = self.FEATURE_WEIGHTS["behavioral"]["email_clicks"]
            click_score = min(email_clicks / 3, 1.0)  # Normalize clicks
            score += weight * click_score
            total_weight += weight
        
        # Form submissions
        form_submissions = self._count_activities(activities, "form_submission")
        if form_submissions > 0:
            weight = self.FEATURE_WEIGHTS["behavioral"]["form_submissions"]
            submission_score = min(form_submissions / 2, 1.0)  # Normalize submissions
            score += weight * submission_score
            total_weight += weight
        
        # Recency score
        if lead.last_activity:
            # Make utcnow timezone-aware
            now = datetime.now(timezone.utc)
            days_since_last_activity = (now - lead.last_activity).days
            recency_score = max(1.0 - (days_since_last_activity / 30), 0.0)  # Normalize recency
            
            weight = self.FEATURE_WEIGHTS["behavioral"]["recency"]
            score += weight * recency_score
            total_weight += weight
        
        # Normalize score to 0-1 range
        if total_weight > 0:
            return score / total_weight
        return 0.0
    
    def _calculate_firmographic_score(self, lead: Lead) -> float:
        """Calculate firmographic score based on company attributes."""
        score = 0.0
        total_weight = 0.0
        
        # Company name present
        if lead.company:
            weight = self.FEATURE_WEIGHTS["firmographic"]["company"]
            score += weight * 1.0  # Full score for having company info
            total_weight += weight
        
        # Company size from custom fields
        company_size = None
        if lead.custom_fields and "company_size" in lead.custom_fields:
            company_size = lead.custom_fields["company_size"]
        
        if company_size:
            weight = self.FEATURE_WEIGHTS["firmographic"]["company_size"]
            # Score large companies higher (assuming enterprise focus)
            if company_size == "enterprise":
                size_score = 1.0
            elif company_size == "mid-market":
                size_score = 0.8
            elif company_size == "small":
                size_score = 0.6
            else:
                size_score = 0.4
                
            score += weight * size_score
            total_weight += weight
        
        # Industry relevance
        if lead.industry:
            weight = self.FEATURE_WEIGHTS["firmographic"]["industry_relevance"]
            industry_score = self._get_industry_relevance(lead.industry)
            score += weight * industry_score
            total_weight += weight
        
        # Normalize score to 0-1 range
        if total_weight > 0:
            return score / total_weight
        return 0.0
    
    def _calculate_confidence(self, lead: Lead, activities: List[LeadActivity]) -> float:
        """Calculate confidence level based on data completeness."""
        # Define data points we want to check
        demographic_fields = ["job_title", "industry", "country", "city"]
        behavioral_data = [
            lead.website_visits > 0,
            lead.time_spent_on_website > 0,
            lead.page_views > 0,
            lead.last_activity is not None,
            len(activities) > 0
        ]
        firmographic_fields = ["company"]
        
        # Check custom fields
        custom_fields_present = 1 if lead.custom_fields else 0
        
        # Calculate completeness percentage
        demographic_completeness = sum(1 for field in demographic_fields if getattr(lead, field)) / len(demographic_fields)
        behavioral_completeness = sum(1 for item in behavioral_data if item) / len(behavioral_data)
        firmographic_completeness = sum(1 for field in firmographic_fields if getattr(lead, field)) / len(firmographic_fields)
        
        # Weighted confidence
        confidence = (
            demographic_completeness * 0.25 +
            behavioral_completeness * 0.5 +
            firmographic_completeness * 0.15 +
            custom_fields_present * 0.1
        )
        
        return confidence
    
    def _create_factors_dict(
        self, 
        lead: Lead, 
        demographic_score: float, 
        behavioral_score: float, 
        firmographic_score: float
    ) -> Dict[str, Any]:
        """Create a detailed dictionary of scoring factors."""
        factors = {
            "demographic": {
                "score": round(demographic_score * 100, 4),
                "weight": self.COMPONENT_WEIGHTS["demographic"],
                "details": {}
            },
            "behavioral": {
                "score": round(behavioral_score * 100, 4),
                "weight": self.COMPONENT_WEIGHTS["behavioral"],
                "details": {}
            },
            "firmographic": {
                "score": round(firmographic_score * 100, 4),
                "weight": self.COMPONENT_WEIGHTS["firmographic"],
                "details": {}
            }
        }
        
        # Demographic factors
        if lead.job_title:
            factors["demographic"]["details"]["job_title"] = {
                "value": lead.job_title,
                "relevance": self._get_job_title_relevance(lead.job_title)
            }
            
        if lead.industry:
            factors["demographic"]["details"]["industry"] = {
                "value": lead.industry,
                "relevance": self._get_industry_relevance(lead.industry)
            }
            
        if lead.country:
            factors["demographic"]["details"]["location"] = {
                "country": lead.country,
                "city": lead.city
            }
        
        # Behavioral factors
        factors["behavioral"]["details"]["website_engagement"] = {
            "visits": lead.website_visits,
            "time_spent": lead.time_spent_on_website,
            "page_views": lead.page_views
        }
        
        if lead.last_activity:
            now = datetime.now(timezone.utc)
            days_since = (now - lead.last_activity).days
            factors["behavioral"]["details"]["recency"] = {
                "last_activity": lead.last_activity.isoformat(),
                "days_since": days_since
            }
        
        # Firmographic factors
        if lead.company:
            factors["firmographic"]["details"]["company"] = lead.company
            
        if lead.custom_fields and "company_size" in lead.custom_fields:
            factors["firmographic"]["details"]["company_size"] = lead.custom_fields["company_size"]
        
        return factors
    
    def _count_activities(self, activities: List[LeadActivity], activity_type: str) -> int:
        """Count activities of a specific type."""
        return sum(1 for activity in activities if activity.activity_type == activity_type)
    
    def _get_industry_relevance(self, industry: str) -> float:
        """Get industry relevance score."""
        industry_lower = industry.lower()
        for key, score in self.INDUSTRY_RELEVANCE.items():
            if key in industry_lower:
                return score
        return self.INDUSTRY_RELEVANCE.get("other", 0.5)
    
    def _get_job_title_relevance(self, job_title: str) -> float:
        """Get job title relevance score."""
        title_lower = job_title.lower()
        for key, score in self.JOB_TITLE_RELEVANCE.items():
            if key in title_lower:
                return score
        return self.JOB_TITLE_RELEVANCE.get("other", 0.5)