# Marketing Tool Development Guide

## Project Structure
```
backend/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       └── endpoints/
│   │   ├── core/
│   │   ├── crud/
│   │   ├── db/
│   │   ├── middleware/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── utils/
│   ├── ml_models/
│   ├── tests/
│   └── docs/
├── alembic/
└── docker/
```

## Prerequisites
- Python 3.8+
- Docker and Docker Compose
- PostgreSQL 13+
- Redis (for caching and rate limiting)

## Setup Development Environment

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-marketing-tool-backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the backend directory with the following variables:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/marketing_tool
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

5. Initialize the database:
```bash
alembic upgrade head
```

## Running the Application

### Development Mode
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Using Docker
```bash
docker-compose up --build
```
## Database Migrations

### Creating a New Migration
```bash
alembic revision --autogenerate -m "description"
```

### Applying Migrations
```bash
alembic upgrade head
```

### Rolling Back Migrations
```bash
alembic downgrade -1  # Roll back one migration
```

## Code Style and Linting

### Running Linters
```bash
flake8
black .
isort .
```

### Pre-commit Hooks
The project uses pre-commit hooks for code quality. Install them with:
```bash
pre-commit install
```

## API Documentation

### Swagger UI
Access the interactive API documentation at:
```
http://localhost:8000/docs
```

### ReDoc
Alternative API documentation at:
```
http://localhost:8000/redoc
```

## Key Features Implementation

### Authentication
- JWT-based authentication
- Role-based access control
- Rate limiting middleware

### Lead Management
- Lead scoring system
- Lead tracking and analytics
- Conversion tracking

### Social Media Integration
- Multi-platform support
- Post scheduling
- Analytics and reporting

### Content Generation
- AI-powered content creation
- Template management
- Content optimization

### Analytics
- Real-time analytics
- Custom reporting
- Performance metrics

## Deployment

### Docker Deployment
1. Build the Docker image:
```bash
docker build -t marketing-tool-backend .
```

2. Run the container:
```bash
docker run -p 8000:8000 marketing-tool-backend
```

## Monitoring and Logging

### Logging
- Application logs are stored in `src/logs/`
- Log rotation is configured
- Different log levels for development and production

### Monitoring
- Health check endpoints
- Performance metrics
- Error tracking

## Troubleshooting

### Common Issues
1. Database connection issues
   - Check DATABASE_URL in .env
   - Verify PostgreSQL is running

3. Authentication issues
   - Verify SECRET_KEY is set
   - Check token expiration time

### Getting Help
- Check the logs in `src/logs/`
- Review the API documentation
