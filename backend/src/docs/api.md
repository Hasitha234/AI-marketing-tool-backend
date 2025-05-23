# AI Marketing Tool API Documentation

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication

### Register New User
```http
POST /auth/register
```

### Login
```http
POST /auth/login
```

**Response:**
- JWT token for authentication
- User details

### Authentication Headers
```http
Authorization: Bearer {jwt_token}
```

## User Management

### Get All Users
```http
GET /users
```
**Parameters:**
- `skip` (optional): Number of records to skip (default: 0)
- `limit` (optional): Number of records to return (default: 100)

**Access:** Admins only

### Get User by ID
```http
GET /users/{user_id}
```
**Access:** Users can only see themselves, admins can see anyone

## Lead Management

### Get Leads
```http
GET /leads
```
**Parameters:**
- `skip` (optional): Number of records to skip (default: 0)
- `limit` (optional): Number of records per page (default: 100)
- `status` (optional): Filter by lead status
- `is_converted` (optional): Filter by conversion status
- `min_score` (optional): Filter by minimum lead score (0-100)
- `source` (optional): Filter by lead source

**Response:**
```json
{
    "items": [...],
    "total": integer,
    "page": integer,
    "size": integer,
    "pages": integer
}
```

## Analytics

### Get Lead Summary
```http
GET /analytics/lead-summary
```
**Parameters:**
- `days` (optional): Number of days to analyze (1-365, default: 30)

**Response:**
```json
{
    "total_leads": integer,
    "new_leads": integer,
    "converted_leads": integer,
    "conversion_rate": float,
    "average_score": float,
    "leads_by_source": [...],
    "leads_by_status": [...],
    "top_scoring_leads": [...],
    "score_distribution": [...],
    "period_days": integer
}
```

### Get Lead Scoring Model Info
```http
GET /analytics/lead-scoring-model
```
**Response:**
```json
{
    "model_version": string,
    "model_type": string,
    "machine_learning_model": {
        "exists": boolean,
        "type": string,
        "feature_importance": object
    },
    "score_statistics": {
        "average_score": float,
        "average_demographic_score": float,
        "average_behavioral_score": float,
        "average_firmographic_score": float
    },
    "conversion_by_score_range": [...],
    "component_weights": object
}
```

## Social Media Management

### Get Social Accounts
```http
GET /social/accounts
```
**Parameters:**
- `skip` (optional): Number of records to skip
- `limit` (optional): Number of records per page (1-100)
- `platform` (optional): Filter by platform
- `is_active` (optional): Filter by active status

### Get Social Posts
```http
GET /social/posts
```
**Parameters:**
- `skip` (optional): Records to skip
- `limit` (optional): Records per page (1-100)
- `account_id` (optional): Filter by account
- `campaign_id` (optional): Filter by campaign
- `status` (optional): Filter by status
- `days_ahead` (optional): Filter by scheduled date (1-365 days)

### Get Optimal Posting Times
```http
GET /social/accounts/{account_id}/optimal-times
```

### Get Campaign Performance
```http
GET /social/campaigns/{campaign_id}/performance
```

### Get Available Platforms
```http
GET /social/platforms
```

## Content Management

### Get Contents
```http
GET /content
```
**Parameters:**
- `skip` (optional): Records to skip
- `limit` (optional): Records per page (1-100)
- `type` (optional): Filter by content type
- `status` (optional): Filter by status

### Get Available Tones
```http
GET /content/tones
```

## Chatbot

### Get FAQs
```http
GET /chatbot/faqs
```
**Parameters:**
- `category` (optional): Filter by category
- `skip` (optional): Records to skip
- `limit` (optional): Records per page (default: 100)

### Health Check
```http
GET /chatbot/health
```

## Response Codes

```
200: Success
201: Created
202: Accepted
400: Bad Request
401: Unauthorized
403: Forbidden
404: Not Found
500: Internal Server Error
```

## Rate Limiting

The API implements rate limiting to prevent abuse. Limits are as follows:
- Authentication endpoints: 5 requests per minute
- Other endpoints: 100 requests per minute per authenticated user

## Pagination

Most list endpoints support pagination with the following parameters:
- `skip`: Number of records to skip
- `limit`: Number of records per page

Response includes:
```json
{
    "items": [...],
    "total": integer,
    "page": integer,
    "size": integer,
    "pages": integer
}
```

## Error Responses

Error responses follow this format:
```json
{
    "detail": "Error message describing what went wrong"
}
```

## CORS

The API supports Cross-Origin Resource Sharing (CORS) for specified origins. All methods and headers are allowed for authenticated requests. 