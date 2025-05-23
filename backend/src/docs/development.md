# AI Marketing Tool Development Guide

## 🛠️ Development Setup

### Prerequisites

1. **Python Environment**
   - Python 3.8 or higher
   - pip (Python package manager)
   - virtualenv or venv for isolated environments

2. **Database**
   - PostgreSQL 12.0 or higher
   - pgAdmin (optional, for database management)

3. **Development Tools**
   - Git
   - Docker & Docker Compose
   - Your preferred IDE (VSCode recommended)
   - Postman or similar API testing tool

4. **API Keys**
   - OpenAI API key
   - HuggingFace token
   - Social media platform API keys (Twitter, Facebook, LinkedIn)
   - Google Cloud Project credentials (for Dialogflow)

### Initial Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd AI-marketing-tool
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create `.env` file in the root directory:
   ```env
   # Database Configuration
   POSTGRES_SERVER=localhost
   POSTGRES_USER=your_user
   POSTGRES_PASSWORD=your_password
   POSTGRES_DB=ai_marketing
   POSTGRES_PORT=5432

   # Security
   SECRET_KEY=your_secure_secret_key

   # API Keys
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_TOKEN=your_huggingface_token

   # Social Media API Keys
   TWITTER_API_KEY=your_twitter_api_key
   TWITTER_API_SECRET=your_twitter_api_secret
   FACEBOOK_APP_ID=your_facebook_app_id
   FACEBOOK_APP_SECRET=your_facebook_app_secret
   LINKEDIN_CLIENT_ID=your_linkedin_client_id
   LINKEDIN_CLIENT_SECRET=your_linkedin_client_secret

   # Google Cloud / Dialogflow
   GOOGLE_CLOUD_PROJECT_ID=your_project_id
   ```

5. **Database Setup**
   ```bash
   # Start PostgreSQL service
   docker-compose up -d db

   # Run migrations
   alembic upgrade head

   # Seed initial data (if needed)
   python src/scripts/seed_data.py
   ```

## 🏗️ Project Structure

```
AI-marketing-tool/
├── backend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── api/
│   │   │   │   └── v1/
│   │   │   │       └── endpoints/
│   │   │   ├── core/
│   │   │   ├── crud/
│   │   │   ├── db/
│   │   │   ├── models/
│   │   │   ├── schemas/
│   │   │   └── services/
│   │   ├── scripts/
│   │   └── tests/
│   ├── alembic/
│   └── requirements.txt
├── docker/
├── docs/
└── docker-compose.yml
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest src/tests/test_lead_scoring.py

# Run with coverage report
pytest --cov=src
```

### Test Structure
- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests
- `tests/api/`: API endpoint tests
- `tests/services/`: Service layer tests

## 🔄 Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Code Style**
   - Follow PEP 8 guidelines
   - Use type hints
   - Document functions and classes
   - Keep functions focused and small

3. **Pre-commit Checks**
   ```bash
   # Run linting
   flake8 src

   # Run type checking
   mypy src

   # Run tests
   pytest
   ```

4. **Commit Guidelines**
   ```
   feat: Add new feature
   fix: Bug fix
   docs: Documentation changes
   style: Code style changes
   refactor: Code refactoring
   test: Add or modify tests
   chore: Routine tasks, maintenance
   ```

## 🚀 Local Development

1. **Start Development Server**
   ```bash
   # Start all services
   docker-compose up -d

   # Or start individual services
   python -m src.app.main
   ```

2. **Access Points**
   - API: http://localhost:8000
   - Swagger Docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

3. **Hot Reload**
   The development server includes hot reload by default

## 🔍 Debugging

1. **Logging**
   - Logs are stored in `logs/`
   - Development logs: `logs/development.log`
   - Error logs: `logs/error.log`

2. **Debug Configuration (VSCode)**
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python: FastAPI",
         "type": "python",
         "request": "launch",
         "module": "uvicorn",
         "args": ["src.app.main:app", "--reload"],
         "jinja": true
       }
     ]
   }
   ```

## 📊 Monitoring

1. **Performance Monitoring**
   - API response times
   - Database query performance
   - Memory usage
   - Error rates

2. **Tools**
   - Prometheus for metrics
   - Grafana for visualization
   - Sentry for error tracking

## 🔐 Security Practices

1. **Code Security**
   - No secrets in code
   - Input validation
   - SQL injection prevention
   - XSS protection

2. **API Security**
   - JWT authentication
   - Rate limiting
   - CORS configuration
   - Input sanitization

## 📦 Deployment

1. **Build Process**
   ```bash
   # Build Docker images
   docker-compose build

   # Run production setup
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **Environment Specific Configs**
   - Development: `.env.development`
   - Staging: `.env.staging`
   - Production: `.env.production`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Pull Request Process
1. Update documentation
2. Add/update tests
3. Ensure CI passes
4. Get code review
5. Merge after approval

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [Docker Documentation](https://docs.docker.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/) 