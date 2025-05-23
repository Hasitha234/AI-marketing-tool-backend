"""
Deployment script for the chatbot service-
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_requirements():
    """Check if all requirements are met for deployment"""
    print("🔍 Checking deployment requirements...")
    
    checks = []
    
    # Check environment variables
    required_env_vars = [
        'GOOGLE_CLOUD_PROJECT_ID',
        'GOOGLE_APPLICATION_CREDENTIALS',
        'DATABASE_URL'
    ]
    
    for var in required_env_vars:
        if os.getenv(var):
            checks.append(f"✅ {var} is set")
        else:
            checks.append(f"❌ {var} is missing")
    
    # Check if service account key exists
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if creds_path and Path(creds_path).exists():
        checks.append("✅ Service account key file exists")
    else:
        checks.append("❌ Service account key file not found")
    
    # Check database connection
    try:
        from app.db.session import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        checks.append("✅ Database connection successful")
    except Exception as e:
        checks.append(f"❌ Database connection failed: {str(e)}")
    
    # Print results
    for check in checks:
        print(f"   {check}")
    
    # Return True if all checks pass
    return all("✅" in check for check in checks)


def run_migrations():
    """Run database migrations"""
    print("\n📄 Running database migrations...")
    
    try:
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Migrations completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Migration failed: {e.stderr}")
        return False


def seed_initial_data():
    """Seed initial chatbot data"""
    print("\n🌱 Seeding initial data...")
    
    try:
        from scripts.seed_chatbot_data import main as seed_main
        seed_main()
        return True
    except Exception as e:
        print(f"❌ Seeding failed: {str(e)}")
        return False


def test_chatbot():
    """Run basic chatbot tests"""
    print("\n🧪 Running chatbot tests...")
    
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/test_services/test_chatbot.py", "-v"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e.stderr}")
        return False


def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting server...")
    
    try:
        print("Server starting on http://localhost:8000")
        print("Chatbot API available at: http://localhost:8000/api/v1/chatbot/")
        print("Interactive docs at: http://localhost:8000/docs")
        
        subprocess.run([
            "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {str(e)}")


def main():
    """Main deployment function"""
    print("🚀 AI Marketing Chatbot Deployment")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Deployment requirements not met. Please fix the issues above.")
        sys.exit(1)
    
    # Run migrations
    if not run_migrations():
        print("\n❌ Failed to run migrations")
        sys.exit(1)
    
    # Seed data
    if not seed_initial_data():
        print("\n⚠️ Warning: Failed to seed initial data, but continuing...")
    
    # Optional: Run tests
    test_choice = input("\n🧪 Run tests before starting? (y/N): ").lower()
    if test_choice == 'y':
        if not test_chatbot():
            print("\n⚠️ Warning: Tests failed, but continuing...")
    
    # Start server
    print("\n✅ All checks passed! Starting the chatbot service...")
    start_server()


if __name__ == "__main__":
    main()