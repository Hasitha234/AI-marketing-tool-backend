# 🚀 Dynamic Pricing Integration Documentation

**Integration Date:** October 1, 2025  
**Version:** 1.0.0  
**Engineer:** Hasitha Dilshan Bandara  
**Branch:** `integrate-dynamic-pricing`

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [API Endpoints](#api-endpoints)
4. [Database Schema](#database-schema)
5. [Usage Examples](#usage-examples)
6. [Frontend Integration Guide](#frontend-integration-guide)
7. [Production Considerations](#production-considerations)
8. [Testing](#testing)
9. [Next Steps](#next-steps)

---

## 🎯 Overview

Successfully integrated **Dynamic Pricing Module** with ML-based optimization into the AI Marketing Tool backend.

### What Was Integrated

- ✅ **XGBoost ML Model** for demand forecasting and price optimization
- ✅ **8 RESTful API Endpoints** with JWT authentication
- ✅ **PostgreSQL Database** for pricing recommendations storage
- ✅ **Revenue Optimization Engine** with business rules
- ✅ **Simple & Advanced Pricing** algorithms

### Key Features

| Feature | Description |
|---------|-------------|
| Simple Pricing | Quick estimates using elasticity model |
| Advanced Pricing | ML-based optimization with XGBoost |
| Price Tracking | Store and retrieve pricing history |
| Revenue Analytics | Calculate uplift and ROI |
| User Management | User-specific recommendations |
| Business Rules | Price caps, rounding, validation |

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React/Vue)                  │
│              http://localhost:3000                       │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP/REST API
                        │
┌───────────────────────▼─────────────────────────────────┐
│              FastAPI Backend (Port 8000)                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │         API Endpoints (/api/v1/pricing)         │   │
│  └──────────────┬──────────────────────────────────┘   │
│                 │                                        │
│  ┌──────────────▼──────────────┐  ┌──────────────────┐ │
│  │   Pricing Service            │  │   CRUD Layer     │ │
│  │  - Simple Pricing            │  │  - Create        │ │
│  │  - Advanced Pricing          │  │  - Read          │ │
│  │  - XGBoost Model             │  │  - Update        │ │
│  │  - Revenue Optimization      │  │  - Delete        │ │
│  └──────────────┬──────────────┘  └────────┬─────────┘ │
│                 │                            │           │
│  ┌──────────────▼────────────────────────────▼────────┐ │
│  │              SQLAlchemy ORM                        │ │
│  └──────────────┬─────────────────────────────────────┘ │
└─────────────────┼───────────────────────────────────────┘
                  │
┌─────────────────▼─────────────────────────────────────┐
│         PostgreSQL Database (Neon.tech)               │
│  ┌────────────────────────────────────────────────┐  │
│  │    pricing_recommendations Table               │  │
│  │  - id, user_id, origin, destination            │  │
│  │  - current_price, optimal_price               │  │
│  │  - revenue metrics, demand estimates          │  │
│  │  - created_at, updated_at                     │  │
│  └────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

### ML Model Pipeline

```
User Input → Feature Engineering → XGBoost Model → Revenue Optimization → Business Rules → Final Price
```

---

## 🔌 API Endpoints

### Base URL
```
http://localhost:8000/api/v1/pricing
```

### Authentication
All endpoints require JWT Bearer token:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Endpoint Reference

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/simple` | Get simple pricing recommendation | ✅ Yes |
| POST | `/advanced` | Get ML-based pricing recommendation | ✅ Yes |
| GET | `/recommendations` | List all user recommendations | ✅ Yes |
| GET | `/recommendations/{id}` | Get specific recommendation | ✅ Yes |
| PATCH | `/recommendations/{id}` | Update recommendation | ✅ Yes |
| DELETE | `/recommendations/{id}` | Delete recommendation | ✅ Yes |
| GET | `/stats` | Get pricing statistics | ✅ Yes |
| GET | `/model/info` | Get ML model information | ✅ Yes |

---

## 📊 Database Schema

### `pricing_recommendations` Table

```sql
CREATE TABLE pricing_recommendations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    
    -- Route Information
    origin VARCHAR NOT NULL,
    destination VARCHAR NOT NULL,
    route VARCHAR,
    
    -- Pricing Details
    current_price FLOAT NOT NULL,
    optimal_price FLOAT NOT NULL,
    recommended_price FLOAT,
    competitor_price FLOAT,
    
    -- Revenue Metrics
    current_revenue FLOAT,
    optimal_revenue FLOAT,
    uplift_percent FLOAT,
    
    -- Demand Estimation
    estimated_demand FLOAT,
    
    -- Advanced Features
    distance_miles FLOAT,
    passengers INTEGER,
    market_share_largest FLOAT,
    market_share_lowest_fare FLOAT,
    
    -- Metadata
    is_applied BOOLEAN DEFAULT FALSE,
    notes VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP
);

CREATE INDEX idx_pricing_created_at ON pricing_recommendations(created_at);
CREATE INDEX idx_pricing_route ON pricing_recommendations(route);
```

---

## 💻 Usage Examples

### Example 1: Simple Pricing

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/pricing/simple" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "Los Angeles, CA",
    "destination": "New York City, NY",
    "price": 320.0
  }'
```

**Response:**
```json
{
  "route": "Los Angeles, CA → New York City, NY",
  "avg_price": 320.0,
  "competitor_price": 335.0,
  "estimated_demand": 185.0,
  "current_revenue": 59200.0,
  "optimal_price": 327.5,
  "optimal_revenue": 60487.5,
  "uplift_percent": 2.17,
  "recommended_price": null
}
```

### Example 2: Advanced Pricing (ML-Based)

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/pricing/advanced" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{
      "city1": "Los Angeles, CA (Metropolitan Area)",
      "city2": "New York City, NY (Metropolitan Area)",
      "fare": 320.0,
      "nsmiles": 2451,
      "passengers": 150,
      "large_ms": 0.8,
      "lf_ms": 0.9,
      "month_sin": 0.5,
      "month_cos": 0.87,
      "quarter_sin": 0.3,
      "quarter_cos": 0.95,
      "fare_lag1y": 300,
      "passengers_roll4": 145
    }]
  }'
```

**Response:**
```json
{
  "route": "Los Angeles, CA → New York City, NY",
  "avg_price": 320.0,
  "competitor_price": 340.0,
  "estimated_demand": 192.5,
  "current_revenue": 61600.0,
  "optimal_price": 330.0,
  "optimal_revenue": 63525.0,
  "uplift_percent": 3.12,
  "recommended_price": 330.0
}
```

### Example 3: Get Pricing Statistics

**Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/pricing/stats" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "total_recommendations": 45,
  "applied_recommendations": 12,
  "average_uplift_percent": 4.23,
  "pending_recommendations": 33
}
```

### Example 4: Python Client

```python
import requests

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
TOKEN = "your_jwt_token_here"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Simple Pricing
def get_simple_pricing(origin, destination, price):
    response = requests.post(
        f"{BASE_URL}/pricing/simple",
        headers=headers,
        json={
            "origin": origin,
            "destination": destination,
            "price": price
        }
    )
    return response.json()

# Get Recommendations
def get_recommendations():
    response = requests.get(
        f"{BASE_URL}/pricing/recommendations",
        headers=headers
    )
    return response.json()

# Example usage
result = get_simple_pricing("LA", "NYC", 350.0)
print(f"Optimal Price: ${result['optimal_price']}")
print(f"Revenue Uplift: {result['uplift_percent']}%")
```

---

## 🎨 Frontend Integration Guide

### React Example

```jsx
// src/services/pricingService.js
import axios from 'axios';

const API_URL = 'http://localhost:8000/api/v1/pricing';

export const pricingService = {
  getSimplePricing: async (origin, destination, price) => {
    const response = await axios.post(
      `${API_URL}/simple`,
      { origin, destination, price },
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      }
    );
    return response.data;
  },

  getRecommendations: async () => {
    const response = await axios.get(`${API_URL}/recommendations`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    return response.data;
  },

  getStats: async () => {
    const response = await axios.get(`${API_URL}/stats`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    return response.data;
  }
};
```

```jsx
// src/components/PricingCalculator.jsx
import React, { useState } from 'react';
import { pricingService } from '../services/pricingService';

function PricingCalculator() {
  const [origin, setOrigin] = useState('');
  const [destination, setDestination] = useState('');
  const [price, setPrice] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const data = await pricingService.getSimplePricing(
        origin,
        destination,
        parseFloat(price)
      );
      setResult(data);
    } catch (error) {
      console.error('Pricing error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="pricing-calculator">
      <h2>Dynamic Pricing Calculator</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Origin"
          value={origin}
          onChange={(e) => setOrigin(e.target.value)}
        />
        <input
          type="text"
          placeholder="Destination"
          value={destination}
          onChange={(e) => setDestination(e.target.value)}
        />
        <input
          type="number"
          placeholder="Current Price"
          value={price}
          onChange={(e) => setPrice(e.target.value)}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Calculating...' : 'Get Pricing'}
        </button>
      </form>

      {result && (
        <div className="pricing-results">
          <h3>Results</h3>
          <div className="result-item">
            <strong>Route:</strong> {result.route}
          </div>
          <div className="result-item">
            <strong>Current Price:</strong> ${result.avg_price}
          </div>
          <div className="result-item">
            <strong>Optimal Price:</strong> ${result.optimal_price}
          </div>
          <div className="result-item">
            <strong>Revenue Uplift:</strong> {result.uplift_percent}%
          </div>
          <div className="result-item">
            <strong>Current Revenue:</strong> ${result.current_revenue}
          </div>
          <div className="result-item">
            <strong>Optimal Revenue:</strong> ${result.optimal_revenue}
          </div>
        </div>
      )}
    </div>
  );
}

export default PricingCalculator;
```

---

## ⚠️ Production Considerations

### Security

- [ ] Add rate limiting to prevent abuse
- [ ] Implement API key validation for external clients
- [ ] Add input validation for all endpoints
- [ ] Sanitize user inputs to prevent SQL injection
- [ ] Use HTTPS in production
- [ ] Implement request size limits

### Performance

- [ ] Add Redis caching for frequently requested routes
- [ ] Implement database query optimization
- [ ] Add connection pooling
- [ ] Monitor API response times
- [ ] Set up load balancing for high traffic

### Data Quality

- [ ] Validate route existence in database
- [ ] Add geographic distance validation
- [ ] Implement minimum/maximum price bounds
- [ ] Add competitor price API integration
- [ ] Create route master data table

### Monitoring

- [ ] Set up logging with ELK stack
- [ ] Add Prometheus metrics
- [ ] Create Grafana dashboards
- [ ] Implement error tracking (Sentry)
- [ ] Set up uptime monitoring

### ML Model

- [ ] Schedule model retraining pipeline
- [ ] Add model versioning
- [ ] Implement A/B testing framework
- [ ] Monitor model drift
- [ ] Add model performance metrics

---

## 🧪 Testing

### Manual Testing Checklist

- [x] Simple pricing endpoint works
- [x] Advanced pricing endpoint works
- [x] Recommendations are stored in database
- [x] User authentication required
- [x] Model info endpoint returns data
- [x] Statistics endpoint works
- [ ] Invalid input handling
- [ ] Error responses are correct
- [ ] CORS headers present

### Automated Tests (To Be Created)

```python
# tests/test_pricing.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_simple_pricing():
    """Test simple pricing endpoint"""
    token = get_test_token()  # Helper function
    
    response = client.post(
        "/api/v1/pricing/simple",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "origin": "Los Angeles, CA",
            "destination": "New York, NY",
            "price": 320.0
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "optimal_price" in data
    assert data["uplift_percent"] is not None

def test_pricing_requires_auth():
    """Test that pricing endpoints require authentication"""
    response = client.post(
        "/api/v1/pricing/simple",
        json={
            "origin": "LA",
            "destination": "NYC",
            "price": 320.0
        }
    )
    
    assert response.status_code == 401
```

---

## 🚀 Next Steps

### Phase 1: Enhancement (Week 1-2)
1. ✅ ~~Integrate Dynamic Pricing Module~~ (COMPLETE)
2. Add comprehensive input validation
3. Create automated test suite
4. Add route database for validation
5. Implement caching layer

### Phase 2: Frontend (Week 3-4)
1. Build React pricing calculator component
2. Create pricing dashboard with charts
3. Add recommendation history view
4. Implement bulk pricing analysis
5. Create pricing analytics dashboard

### Phase 3: Data Integration (Week 5-6)
1. Integrate real competitor pricing APIs
2. Add historical price tracking
3. Implement seasonal adjustment factors
4. Create market intelligence module
5. Add demand forecasting dashboard

### Phase 4: Production (Week 7-8)
1. Deploy to staging environment
2. Perform load testing
3. Set up monitoring and alerts
4. Create user documentation
5. Deploy to production

### Phase 5: Advanced Features (Month 2-3)
1. Add multi-currency support
2. Implement dynamic discounting
3. Create pricing strategy templates
4. Add A/B testing framework
5. Build customer segmentation pricing

---

## 📚 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pydantic Models](https://docs.pydantic.dev/)

### Related Files
- `backend/src/app/models/pricing.py` - Database model
- `backend/src/app/schemas/pricing.py` - Pydantic schemas
- `backend/src/app/crud/pricing.py` - CRUD operations
- `backend/src/app/services/pricing/` - Pricing service
- `backend/src/app/api/v1/endpoints/pricing.py` - API endpoints

### Contact
- **Engineer:** Hasitha Dilshan Bandara
- **Email:** hasithab66@gmail.com
- **Branch:** `integrate-dynamic-pricing`
- **Commit:** 395fd80

---

**Documentation Version:** 1.0.0  
**Last Updated:** October 1, 2025  
**Status:** ✅ Production Ready (with noted improvements)

