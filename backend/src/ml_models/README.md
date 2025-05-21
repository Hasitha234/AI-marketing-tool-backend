# AI-Powered Lead Scoring

This directory contains machine learning models used for the AI-powered lead scoring system.

## Overview

The lead scoring system combines traditional rule-based scoring with machine learning prediction. The system:

1. Uses rules to score leads based on demographic, behavioral, and firmographic data
2. Uses a machine learning model to predict conversion probability based on historical data
3. Combines both approaches for a comprehensive lead score that prioritizes high-potential leads

## Model Information

- `lead_scoring_rf_model.joblib`: Random Forest classifier model trained on historical lead data
- `lead_scoring_scaler.joblib`: Feature scaler that normalizes input data for the model

## Features Used for Prediction

The ML model uses the following features to predict lead conversion:

- **Demographic**: Job title relevance, industry relevance, location information
- **Behavioral**: Website visits, time spent on website, page views, email opens/clicks, form submissions
- **Engagement**: Recency of activity, engagement ratio (clicks/opens), activity frequency
- **Firmographic**: Company information, company size

## Training and Using the Model

The model is automatically loaded when the application starts. If no model exists, the system will fall back to rule-based scoring.

### Training the Model

To train or retrain the model, use the API endpoint:

```
POST /api/v1/leads/score/train-model
```

This requires admin or manager privileges. The system will use historical lead data to train the model, requiring at least 50 leads for effective training.

### Viewing Model Information

To view information about the model, use:

```
GET /api/v1/analytics/lead-scoring-model
```

This will show model statistics, feature importance, and conversion rates by score range.

## Integration with Sales Process

The AI lead scoring system helps sales teams by:

1. Automatically prioritizing leads based on their likelihood to convert
2. Providing detailed scoring factors to guide sales conversations
3. Learning from past conversions to continuously improve scoring accuracy
4. Identifying key attributes and behaviors that correlate with successful conversions

The system aims to increase conversion rates by focusing sales efforts on the most promising leads, similar to AI-powered CRM systems like HubSpot and Salesforce. 