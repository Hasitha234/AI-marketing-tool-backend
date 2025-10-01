import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .pricing_utils import pricing_policy

logger = logging.getLogger(__name__)


class PricingService:
    """Dynamic Pricing Service with ML-based optimization"""
    
    def __init__(self):
        self.model_path = Path(__file__).parent
        self.demand_model = None
        self.feature_cols = None
        self._load_models()
    
    def _load_models(self):
        """Load ML models on initialization"""
        try:
            demand_model_file = self.model_path / "demand_model.pkl"
            feature_cols_file = self.model_path / "feature_cols.pkl"
            
            if demand_model_file.exists() and feature_cols_file.exists():
                self.demand_model = joblib.load(demand_model_file)
                self.feature_cols = joblib.load(feature_cols_file)
                logger.info("Pricing ML models loaded successfully")
            else:
                logger.warning("Pricing model files not found. Advanced pricing unavailable.")
        except Exception as e:
            logger.error(f"Error loading pricing models: {e}")
            self.demand_model = None
            self.feature_cols = None
    
    def is_model_available(self) -> bool:
        """Check if ML model is loaded"""
        return self.demand_model is not None and self.feature_cols is not None
    
    def get_simple_pricing(
        self,
        origin: str,
        destination: str,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Get pricing recommendation using simplified elasticity model.
        Used when ML model is not available or for quick estimates.
        """
        # Basic demand elasticity model
        base_demand = 200
        elasticity = -0.05
        demand = max(50, base_demand + elasticity * (current_price - 300))
        
        # Competitor pricing simulation (in production, this would come from real data)
        competitor_price = np.random.randint(int(current_price * 0.9), int(current_price * 1.15))
        
        # Revenue calculations
        current_revenue = current_price * demand
        optimal_price = (competitor_price + current_price) / 2
        optimal_demand = max(50, base_demand + elasticity * (optimal_price - 300))
        optimal_revenue = optimal_price * optimal_demand
        
        # Calculate uplift
        uplift_percent = ((optimal_revenue - current_revenue) / current_revenue) * 100 if current_revenue > 0 else 0
        
        return {
            "route": f"{origin} → {destination}",
            "avg_price": current_price,
            "competitor_price": float(competitor_price),
            "estimated_demand": demand,
            "current_revenue": current_revenue,
            "optimal_price": round(optimal_price, 2),
            "optimal_revenue": round(optimal_revenue, 2),
            "uplift_percent": round(uplift_percent, 2)
        }
    
    def get_advanced_pricing(
        self,
        route_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Get pricing recommendation using ML model.
        Requires detailed route data with all features.
        """
        if not self.is_model_available():
            raise ValueError("ML models not available. Use simple pricing instead.")
        
        if route_data.empty:
            return None
        
        # Use advanced pricing policy with ML model
        result = pricing_policy(route_data, self.demand_model, self.feature_cols)
        
        if result is None:
            return None
        
        # Add competitor price (simulated for now)
        competitor_price = np.random.randint(
            int(result["avg_price"] * 0.9),
            int(result["avg_price"] * 1.15)
        )
        
        # Get estimated demand
        try:
            estimated_demand = float(
                self.demand_model.predict(route_data[self.feature_cols + ['fare']]).mean()
            )
        except Exception as e:
            logger.warning(f"Error predicting demand: {e}")
            estimated_demand = 0.0
        
        return {
            "route": result["route"],
            "avg_price": float(result["avg_price"]),
            "competitor_price": float(competitor_price),
            "estimated_demand": estimated_demand,
            "current_revenue": float(result["current_revenue"]),
            "optimal_price": float(result["best_price"]),
            "optimal_revenue": float(result["optimal_revenue"]),
            "uplift_percent": float(result["uplift_%"]),
            "recommended_price": float(result.get("recommended_price", result["best_price"]))
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        if not self.is_model_available():
            return {
                "model_loaded": False,
                "message": "No models loaded. Advanced pricing features unavailable."
            }
        
        try:
            model_params = self.demand_model.get_params() if hasattr(self.demand_model, 'get_params') else {}
        except Exception:
            model_params = {}
        
        return {
            "model_loaded": True,
            "model_type": type(self.demand_model).__name__,
            "feature_columns": self.feature_cols,
            "num_features": len(self.feature_cols),
            "model_params": model_params
        }


# Global instance
pricing_service = PricingService()
