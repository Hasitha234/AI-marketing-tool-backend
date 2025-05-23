# AI Marketing Tool

A powerful AI-driven marketing automation platform that combines intelligent lead scoring, content generation, chatbot functionality, and social media integration.

## 🌟 Key Features

### 1. Intelligent Lead Scoring System
- Hybrid scoring system combining rule-based and machine learning approaches
- Real-time lead qualification and scoring (0-100)
- Multi-factor analysis including:
  - Demographic data
  - Behavioral tracking
  - Firmographic information
- Advanced analytics dashboard
- Lead activity monitoring
- Automated lead routing

### 2. AI-Powered Content Generation
- Blog post generation
- Social media content creation
- Email campaign content
- Content performance analytics

### 3. Smart Chatbot Integration
- 24/7 customer support automation
- Intent recognition and smart responses
- Lead qualification through conversation
- Appointment scheduling
- Integration with popular platforms
- Customizable conversation flows

### 4. Social Media Automation
- Multi-platform posting capabilities
- Content scheduling
- Performance analytics
- Engagement tracking
- Automated response handling

### 5. Analytics & Reporting
- Comprehensive dashboard
- Lead scoring insights
- Conversion tracking
- ROI measurement
- Custom report generation
- Real-time performance monitoring

## 🚀 Getting Started

### Prerequisites
- Docker and Docker Compose
- Git
- Python 3.10.0
- PostgreSQL


### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-marketing-tool
```

2. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
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
HUGGINGFACE_TOKEN=your_huggingface_token



# Google Cloud / Dialogflow
GOOGLE_CLOUD_PROJECT_ID=your_project_id
```

3. Build and start the services:
```bash
docker-compose up -d
```

4. Run database migrations:
```bash
docker-compose exec api alembic upgrade head
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once the server is running, you can access:
- Swagger UI documentation: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## 💻 Development

For local development:

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:-
```bash
pip install -r requirements.txt
```

3. Navigate to the backend directory:
```bash
cd backend
```

4. Run the application:
```bash
python -m src.app.main
```

## 🧪 Testing

Run the test suite:
```bash
pytest
```

For chatbot testing:
```bash
python src/scripts/test_chatbot.py
```

## 🔒 Security Features

- JWT-based authentication
- Role-based access control
- Password hashing with bcrypt
- HTTPS/TLS encryption
- Rate limiting
- Input validation
- SQL injection protection
- CORS protection

## 🎯 Performance Metrics

- API response times: < 100ms
- Database query performance: < 50ms
- Content generation time: < 5s
- Chatbot response time: < 2s
- System capacity: 100+ concurrent users

## 📈 Deployment

For production deployment, use the provided deployment script:
```bash
./deploy.sh
```

This will:
1. Check prerequisites
2. Build and deploy containers
3. Run database migrations
4. Verify deployment status

