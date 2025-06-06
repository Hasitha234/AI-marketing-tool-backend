# AI Marketing Tool Backend

A powerful backend system for an AI-powered marketing automation platform that helps businesses streamline their marketing operations, generate content, manage leads, and analyze performance.

## Features

### 1. Authentication & User Management
- Secure user authentication system
- User registration and management
- Role-based access control
- Session management

### 2. Content Generation & Management
- AI-powered content generation 
- Content templates and customization
- Content performance tracking
- Multi-format content support

### 3. Lead Management
- Lead capture and tracking
- Lead scoring and qualification
- Lead activity monitoring
- Integration with CRM systems

### 4. Social Media Management
- Multi-platform social media integration
- Post scheduling and automation
- Social media analytics
- Engagement tracking
- Content calendar management

### 5. Analytics & Reporting
- Real-time performance metrics
- Custom report generation
- Campaign performance tracking
- ROI analysis
- Data visualization

### 6. AI Chatbot Integration
- Intelligent chatbot functionality
- Natural language processing
- Automated customer support
- Lead qualification through chat
- Conversation analytics

## Technical Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL
- **ORM**: SQLAlchemy
- **Authentication**: JWT
- **API Documentation**: GeminiApi
- **Containerization**: Docker
- **Deployment**: Docker Compose
- **Database Migrations**: Alembic

## Project Structure

```
backend/
├── src/
│   ├── app/
│   │   ├── api/            # API endpoints and routes
│   │   │   ├── core/           # Core application settings
│   │   │   ├── crud/           # Database operations
│   │   │   ├── db/             # Database configuration
│   │   │   ├── middleware/     # Custom middleware
│   │   │   ├── models/         # Database models
│   │   │   ├── schemas/        # Pydantic schemas
│   │   │   ├── services/       # Business logic
│   │   │   └── utils/          # Utility functions
│   │   ├── ml_models/          # Machine learning models
│   │   ├── scripts/            # Utility scripts
│   │   └── tests/              # Test suite
│   ├── alembic/                # Database migrations
│   └── docker-compose.yml      # Docker configuration
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables
4. Run database migrations:
   ```bash
   alembic upgrade head
   ```
5. Start the application:
   ```bash
   docker-compose up
   ```

## API Documentation

Once the application is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`






